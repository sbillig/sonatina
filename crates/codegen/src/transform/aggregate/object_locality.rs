use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use sonatina_ir::{
    Function, InstId, Module, Type, ValueId,
    inst::{cast, control_flow, data, downcast, evm},
    module::{FuncRef, ModuleCtx},
};
use std::ops::ControlFlow;

use super::{
    ObjectEffectSummaryMap, compute_object_effect_summaries, object_abi::OutputBufferContract,
    object_tracking::objref_element_ty, shape,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootInit {
    UndefFresh,
    LoadLiveIn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LocalObjectArgInfo {
    Borrowed,
    Output(OutputBufferContract),
}

impl LocalObjectArgInfo {
    pub(crate) fn output_contract(
        self,
        function: &Function,
        index: usize,
    ) -> Option<OutputBufferContract> {
        match self {
            Self::Output(contract) if contract.matches(function, index) => Some(contract),
            _ => None,
        }
    }

    pub(crate) fn is_valid_for(self, function: &Function, index: usize) -> bool {
        match self {
            Self::Borrowed => function.arg_values.get(index).is_some_and(|&argument| {
                function.dfg.value_ty(argument).is_obj_ref(function.ctx())
            }),
            Self::Output(contract) => contract.matches(function, index),
        }
    }

    pub(crate) fn init(self, function: &Function, index: usize) -> Option<RootInit> {
        self.is_valid_for(function, index).then_some(match self {
            Self::Borrowed => RootInit::LoadLiveIn,
            Self::Output(_) => RootInit::UndefFresh,
        })
    }
}

pub(crate) type LocalObjectArgMap = FxHashMap<FuncRef, FxHashMap<usize, LocalObjectArgInfo>>;
pub(crate) type LocalObjectArgs = FxHashMap<FuncRef, FxHashSet<usize>>;

pub(crate) enum SpecialObjectUse<'a> {
    MaterializeStack(Option<ValueId>),
    MaterializeHeap,
    Call {
        inst: sonatina_ir::InstId,
        value: ValueId,
        call: &'a control_flow::Call,
    },
    Return {
        value: ValueId,
        ret: &'a control_flow::Return,
    },
    /// A value that reaches the object enters an SSA aggregate or the memory
    /// of a local allocation. Continuing follows it to the parts that keep it:
    /// through the aggregate's extractions, and through loads from the
    /// allocation, whose other uses then count as uses of the object.
    Carry,
    Unknown,
}

/// How a walked value reaches the object: through these hops, outermost
/// first. Without hops the value is a reference into the object. Otherwise the
/// first hop lies in the value when it is an SSA aggregate, or else in the
/// pointee of the value, a view of a holder: a local allocation whose memory
/// keeps what the hop leads to. Each later hop lies in the pointee of the
/// reference kept at the leaves of the hop before it.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
struct Carrier(SmallVec<[Hop; 2]>);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Hop {
    /// The leaves `start..end` that can keep the next hop, or `None` if any
    /// can. A view covering none of them still exposes them when it is
    /// materialized, since that exposes the whole allocation.
    leaves: Option<(usize, usize)>,
    /// The holder whose memory the hop lies in, or `None` in an SSA
    /// aggregate, which only the first hop can be. Walks of lowered objects
    /// see pointers rather than references, so types cannot tell these apart.
    holder: Option<ValueId>,
}

impl Carrier {
    /// Whether the value this carrier describes is an SSA aggregate.
    fn is_aggregate(&self) -> bool {
        self.0.first().is_some_and(|hop| hop.holder.is_none())
    }

    /// The carrier of the part, found by `slice`, of the value or pointee this
    /// one describes; `None` is an unknown part.
    fn within(&self, slice: impl FnOnce() -> Option<shape::AggregateSlice>) -> Self {
        let mut carrier = self.clone();
        if let Some(hop) = carrier.0.first_mut()
            && let Some((start, end)) = hop.leaves
        {
            hop.leaves = slice().map(|slice| {
                let (first, last) = (slice.first_leaf, slice.first_leaf + slice.leaf_count);
                (
                    start.clamp(first, last) - first,
                    end.clamp(first, last) - first,
                )
            });
        }
        carrier
    }

    /// The carrier of a value or pointee whose part from leaf `first` on this
    /// one describes; `None` is an unknown position.
    fn shifted(&self, first: Option<usize>) -> Self {
        let mut carrier = self.clone();
        if let Some(hop) = carrier.0.first_mut() {
            hop.leaves = hop
                .leaves
                .zip(first)
                .map(|((start, end), first)| (start + first, end + first));
        }
        carrier
    }

    /// The carrier of an aggregate or memory keeping the value this one
    /// describes from leaf `first` on.
    fn placed(&self, first: Option<usize>) -> Self {
        let mut carrier = self.clone();
        if !self.is_aggregate() {
            // A reference is a single leaf.
            carrier.0.insert(
                0,
                Hop {
                    leaves: Some((0, 1)),
                    holder: None,
                },
            );
        }
        carrier.shifted(first)
    }

    /// The carrier of a value of type `ty` read from the part this one
    /// describes: the reference its first hop leads to, or an aggregate
    /// keeping that reference.
    fn read(&self, function: &Function, ty: Type) -> Option<Self> {
        let mut carrier = self.clone();
        if carrier
            .0
            .first()?
            .leaves
            .is_some_and(|(start, end)| start >= end)
        {
            return None;
        }
        if ty.is_obj_ref(function.ctx()) {
            carrier.0.remove(0);
        } else if shape::is_reference_aggregate(function.ctx(), ty) {
            carrier.0[0].holder = None;
        } else {
            return None;
        }
        Some(carrier)
    }

    /// The carrier of a reference that a callee derives, perhaps through a
    /// projection, from a value this one describes.
    fn borrowed(&self) -> Self {
        let mut carrier = self.clone();
        if let Some(hop) = carrier.0.first_mut() {
            hop.leaves = None;
        }
        carrier
    }
}

// This walks the full module and must run before any `func_store.modify(...)` loop.
pub(crate) fn collect_local_object_arg_info(module: &Module) -> LocalObjectArgMap {
    let object_effects = compute_object_effect_summaries(module);
    collect_local_object_arg_info_with_effects(module, &object_effects)
}

pub(crate) fn collect_local_object_arg_info_with_effects(
    module: &Module,
    object_effects: &ObjectEffectSummaryMap,
) -> LocalObjectArgMap {
    module
        .funcs()
        .into_iter()
        .filter_map(|func| {
            let args = local_object_arg_info(module, func, object_effects);
            (!args.is_empty()).then_some((func, args))
        })
        .collect()
}

pub(crate) fn local_object_arg_info(
    module: &Module,
    func: FuncRef,
    object_effects: &ObjectEffectSummaryMap,
) -> FxHashMap<usize, LocalObjectArgInfo> {
    let mut args = FxHashMap::default();
    if let Some(sig) = module.ctx.get_sig(func)
        && let Some(summary) = object_effects.get(&func)
    {
        for (idx, &root_ty) in sig.args().iter().enumerate() {
            if root_ty.is_obj_ref(&module.ctx)
                && summary
                    .arg_effects
                    .get(idx)
                    .is_some_and(|effect| effect.local_only)
                && !summary
                    .captures
                    .iter()
                    .any(|capture| capture.src_arg == idx)
            {
                args.insert(idx, LocalObjectArgInfo::Borrowed);
            }
        }
    }
    args
}

pub(crate) fn collect_local_object_args(module: &Module) -> LocalObjectArgs {
    info_to_local_object_args(module, &collect_local_object_arg_info(module))
}

pub(crate) fn info_to_local_object_args(
    module: &Module,
    local_object_args: &LocalObjectArgMap,
) -> LocalObjectArgs {
    module
        .funcs()
        .into_iter()
        .filter_map(|func| {
            let args = local_object_args.get(&func)?;
            let valid = module.func_store.view(func, |function| {
                args.iter()
                    .filter_map(|(&index, info)| {
                        info.is_valid_for(function, index).then_some(index)
                    })
                    .collect()
            });
            Some((func, valid))
        })
        .collect()
}

pub(crate) fn merge_local_object_arg_info(
    module: &Module,
    local_object_args: &mut LocalObjectArgMap,
    extra: &LocalObjectArgMap,
) {
    for func in module.funcs() {
        if let Some(args) = extra.get(&func) {
            module.func_store.view(func, |function| {
                local_object_args
                    .entry(func)
                    .or_default()
                    .extend(args.iter().filter_map(|(&index, &info)| {
                        info.is_valid_for(function, index).then_some((index, info))
                    }));
            });
        }
    }
}

pub(crate) fn call_passes_object_to_local_args(
    ctx: &ModuleCtx,
    call: &control_flow::Call,
    value: ValueId,
    value_ty: Type,
    local_object_args: &LocalObjectArgs,
) -> bool {
    let Some(sig) = ctx.get_sig(*call.callee()) else {
        return false;
    };
    let Some(local_args) = local_object_args.get(call.callee()) else {
        return false;
    };

    let mut saw_value = false;
    for (idx, &arg) in call.args().iter().enumerate() {
        if arg != value {
            continue;
        }
        saw_value = true;
        if sig.args().get(idx) != Some(&value_ty) || !local_args.contains(&idx) {
            return false;
        }
    }

    saw_value
}

fn call_passes_object_to_local_arg_info(
    ctx: &ModuleCtx,
    call: &control_flow::Call,
    value: ValueId,
    value_ty: Type,
    local_object_args: &LocalObjectArgMap,
) -> bool {
    let Some(sig) = ctx.get_sig(*call.callee()) else {
        return false;
    };
    let Some(local_args) = local_object_args.get(call.callee()) else {
        return false;
    };

    let mut saw_value = false;
    for (idx, &arg) in call.args().iter().enumerate() {
        if arg != value {
            continue;
        }
        saw_value = true;
        if sig.args().get(idx) != Some(&value_ty) || !local_args.contains_key(&idx) {
            return false;
        }
    }

    saw_value
}

pub(crate) fn object_root_stays_local(
    function: &Function,
    root: ValueId,
    local_object_args: &LocalObjectArgMap,
    allow_return_root: bool,
) -> bool {
    object_root_stays_local_with(
        function,
        root,
        local_object_args,
        |value| value == root,
        allow_return_root,
    )
}

pub(crate) fn object_root_stays_local_with_effects(
    function: &Function,
    root: ValueId,
    object_effects: &ObjectEffectSummaryMap,
    mut is_allowed_root_value: impl FnMut(ValueId) -> bool,
    allow_return_root: bool,
    allow_carry: bool,
) -> bool {
    walk_object_root_uses_with_effects(function, root, object_effects, |special| match special {
        SpecialObjectUse::MaterializeStack(Some(ptr)) if raw_pointer_stays_local(function, ptr) => {
            ControlFlow::Continue(())
        }
        SpecialObjectUse::Carry if allow_carry => ControlFlow::Continue(()),
        SpecialObjectUse::Return { value, ret }
            if allow_return_root
                && is_allowed_root_value(value)
                && ret.returns_single()
                && ret.arg() == Some(&value) =>
        {
            ControlFlow::Continue(())
        }
        SpecialObjectUse::Call { inst, value, call }
            if is_allowed_root_value(value)
                && call_root_preserves_locality(
                    function,
                    inst,
                    call,
                    value,
                    object_effects,
                    &mut is_allowed_root_value,
                ) =>
        {
            ControlFlow::Continue(())
        }
        _ => ControlFlow::Break(()),
    })
    .is_none()
}

pub(crate) fn object_root_stays_local_with(
    function: &Function,
    root: ValueId,
    local_object_args: &LocalObjectArgMap,
    mut is_allowed_root_value: impl FnMut(ValueId) -> bool,
    allow_return_root: bool,
) -> bool {
    walk_object_root_uses(function, root, |special| match special {
        SpecialObjectUse::MaterializeStack(Some(ptr)) if raw_pointer_stays_local(function, ptr) => {
            ControlFlow::Continue(())
        }
        SpecialObjectUse::Return { value, ret }
            if allow_return_root
                && is_allowed_root_value(value)
                && ret.returns_single()
                && ret.arg() == Some(&value) =>
        {
            ControlFlow::Continue(())
        }
        SpecialObjectUse::Call { value, call, .. }
            if is_allowed_root_value(value)
                && call_passes_object_to_local_arg_info(
                    function.ctx(),
                    call,
                    value,
                    function.dfg.value_ty(value),
                    local_object_args,
                ) =>
        {
            ControlFlow::Continue(())
        }
        _ => ControlFlow::Break(()),
    })
    .is_none()
}

pub(crate) fn raw_pointer_stays_local(function: &Function, root: ValueId) -> bool {
    let mut worklist = vec![root];
    let mut seen = FxHashSet::default();

    while let Some(value) = worklist.pop() {
        if !seen.insert(value) {
            continue;
        }

        for &user in function.dfg.users(value) {
            if !function.layout.is_inst_inserted(user) {
                continue;
            }

            if let Some(gep) = downcast::<&data::Gep>(function.inst_set(), function.dfg.inst(user))
                && gep.values().first() == Some(&value)
            {
                if let Some(result) = function.dfg.inst_result(user) {
                    worklist.push(result);
                }
                continue;
            }

            if let Some(bitcast) =
                downcast::<&cast::Bitcast>(function.inst_set(), function.dfg.inst(user))
                && *bitcast.from() == value
            {
                if let Some(result) = function.dfg.inst_result(user) {
                    worklist.push(result);
                }
                continue;
            }

            if let Some(phi) =
                downcast::<&control_flow::Phi>(function.inst_set(), function.dfg.inst(user))
                && phi.args().iter().any(|(arg, _)| *arg == value)
            {
                if let Some(result) = function.dfg.inst_result(user) {
                    worklist.push(result);
                }
                continue;
            }

            if let Some(load) =
                downcast::<&data::Mload>(function.inst_set(), function.dfg.inst(user))
                && *load.addr() == value
            {
                continue;
            }

            if let Some(store) =
                downcast::<&data::Mstore>(function.inst_set(), function.dfg.inst(user))
            {
                if *store.addr() == value {
                    continue;
                }
                if *store.value() == value {
                    return false;
                }
            }

            if let Some(copy) =
                downcast::<&evm::EvmCodeCopy>(function.inst_set(), function.dfg.inst(user))
            {
                if *copy.dst_addr() == value && *copy.code_offset() != value && *copy.len() != value
                {
                    continue;
                }
                return false;
            }

            if let Some(copy) =
                downcast::<&evm::EvmMcopy>(function.inst_set(), function.dfg.inst(user))
            {
                if (*copy.dest() == value || *copy.addr() == value) && *copy.len() != value {
                    continue;
                }
                return false;
            }

            if downcast::<&control_flow::Call>(function.inst_set(), function.dfg.inst(user))
                .is_some()
                || downcast::<&control_flow::Return>(function.inst_set(), function.dfg.inst(user))
                    .is_some_and(|ret| ret.args().iter().any(|arg| *arg == value))
                || downcast::<&cast::PtrToInt>(function.inst_set(), function.dfg.inst(user))
                    .is_some_and(|ptr_to_int| *ptr_to_int.from() == value)
            {
                return false;
            }

            return false;
        }
    }

    true
}

pub(crate) fn walk_object_root_uses<T>(
    function: &Function,
    root: ValueId,
    mut on_special: impl FnMut(SpecialObjectUse<'_>) -> ControlFlow<T>,
) -> Option<T> {
    walk_object_root_uses_impl(function, root, None, &mut on_special)
}

pub(crate) fn walk_object_root_uses_with_effects<T>(
    function: &Function,
    root: ValueId,
    object_effects: &ObjectEffectSummaryMap,
    mut on_special: impl FnMut(SpecialObjectUse<'_>) -> ControlFlow<T>,
) -> Option<T> {
    walk_object_root_uses_impl(function, root, Some(object_effects), &mut on_special)
}

fn walk_object_root_uses_impl<T>(
    function: &Function,
    root: ValueId,
    object_effects: Option<&ObjectEffectSummaryMap>,
    on_special: &mut impl FnMut(SpecialObjectUse<'_>) -> ControlFlow<T>,
) -> Option<T> {
    let mut worklist = vec![(root, Carrier::default())];
    let mut seen = FxHashSet::default();

    while let Some((value, carrier)) = worklist.pop() {
        if !seen.insert((value, carrier.clone())) {
            continue;
        }

        for &user in function.dfg.users(value) {
            if !function.layout.is_inst_inserted(user) {
                continue;
            }
            if let ControlFlow::Break(result) = walk_use(
                function,
                user,
                value,
                &carrier,
                object_effects,
                &mut worklist,
                on_special,
            ) {
                return Some(result);
            }
        }
    }

    None
}

/// Walks one use of `value`, which reaches the object as `carrier` describes.
fn walk_use<T>(
    function: &Function,
    user: InstId,
    value: ValueId,
    carrier: &Carrier,
    object_effects: Option<&ObjectEffectSummaryMap>,
    worklist: &mut Vec<(ValueId, Carrier)>,
    on_special: &mut impl FnMut(SpecialObjectUse<'_>) -> ControlFlow<T>,
) -> ControlFlow<T> {
    let ctx = function.ctx();
    let dfg = &function.dfg;
    let inst_set = function.inst_set();
    let inst = dfg.inst(user);
    // Only single-result instructions are asked for their result.
    let result = || dfg.inst_result(user);

    if let Some(store) = downcast::<&data::ObjStore>(inst_set, inst)
        && *store.value() == value
    {
        let carrier = carrier.placed(Some(0));
        return carry_into(function, *store.object(), carrier, worklist, on_special);
    }
    if let Some(write) = downcast::<&data::EnumWriteVariant>(inst_set, inst)
        && write.values().contains(&value)
    {
        let enum_ty = objref_element_ty(ctx, dfg.value_ty(*write.object()));
        for field in value_fields(write.values(), value) {
            let slot = enum_ty
                .and_then(|ty| shape::enum_variant_field_slice(ctx, ty, *write.variant(), field));
            let carrier = carrier.placed(slot.map(|slot| slot.first_leaf));
            carry_into(function, *write.object(), carrier, worklist, on_special)?;
        }
        return ControlFlow::Continue(());
    }
    if let Some(make) = downcast::<&data::EnumMake>(inst_set, inst)
        && make.values().contains(&value)
    {
        on_special(SpecialObjectUse::Carry)?;
        for field in value_fields(make.values(), value) {
            let slot = shape::enum_variant_field_slice(ctx, *make.ty(), *make.variant(), field);
            let carrier = carrier.placed(slot.map(|slot| slot.first_leaf));
            worklist.extend(result().map(|result| (result, carrier)));
        }
        return ControlFlow::Continue(());
    }
    if let Some(insert) = downcast::<&data::InsertValue>(inst_set, inst)
        && (*insert.dest() == value || *insert.value() == value)
    {
        if *insert.value() == value {
            on_special(SpecialObjectUse::Carry)?;
            let slot = shape::aggregate_slice_for_object_path(
                ctx,
                dfg.value_ty(*insert.dest()),
                &[*insert.idx()],
                dfg,
            );
            let carrier = carrier.placed(slot.map(|slot| slot.first_leaf));
            worklist.extend(result().map(|result| (result, carrier)));
        }
        if *insert.dest() == value {
            worklist.extend(result().map(|result| (result, carrier.clone())));
        }
        return ControlFlow::Continue(());
    }

    if downcast::<&control_flow::Phi>(inst_set, inst)
        .is_some_and(|phi| phi.args().iter().any(|(arg, _)| *arg == value))
        || downcast::<&data::EnumAssertVariantRef>(inst_set, inst)
            .is_some_and(|enum_assert_ref| *enum_assert_ref.object() == value)
    {
        worklist.extend(result().map(|result| (result, carrier.clone())));
        return ControlFlow::Continue(());
    }
    if view_base(function, user) == Some(value) {
        // Every view of the object reaches it; a view of a holder keeps only
        // the leaves it covers.
        let carrier = carrier.within(|| view_slice(function, user));
        worklist.extend(result().map(|result| (result, carrier)));
        return ControlFlow::Continue(());
    }

    let read = if let Some(load) = downcast::<&data::ObjLoad>(inst_set, inst)
        && *load.object() == value
    {
        Some(carrier.clone())
    } else if let Some(extract) = downcast::<&data::ExtractValue>(inst_set, inst)
        && *extract.dest() == value
    {
        Some(carrier.within(|| {
            shape::aggregate_slice_for_object_path(ctx, dfg.value_ty(value), &[*extract.idx()], dfg)
        }))
    } else if let Some(extract) = downcast::<&data::EnumExtract>(inst_set, inst)
        && *extract.value() == value
    {
        Some(carrier.within(|| {
            let field = shape::const_u32(dfg, *extract.field())?;
            shape::enum_variant_field_slice(ctx, dfg.value_ty(value), *extract.variant(), field)
        }))
    } else {
        None
    };
    if let Some(part) = read {
        // Loading through a reference into the object reads no reference into
        // it; the walk of the object as its own holder follows any it keeps.
        worklist.extend(
            result().and_then(|result| Some((result, part.read(function, dfg.value_ty(result))?))),
        );
        return ControlFlow::Continue(());
    }

    if downcast::<&data::ObjStore>(inst_set, inst).is_some_and(|store| *store.object() == value)
        || downcast::<&data::ObjInitConst>(inst_set, inst)
            .is_some_and(|init| *init.object() == value)
        || downcast::<&data::EnumGetTag>(inst_set, inst)
            .is_some_and(|enum_get_tag| *enum_get_tag.object() == value)
        || downcast::<&data::EnumSetTag>(inst_set, inst)
            .is_some_and(|enum_set_tag| *enum_set_tag.object() == value)
        || downcast::<&data::EnumWriteVariant>(inst_set, inst)
            .is_some_and(|enum_write_variant| *enum_write_variant.object() == value)
        || downcast::<&data::EnumTag>(inst_set, inst).is_some_and(|tag| *tag.value() == value)
        || downcast::<&data::EnumIsVariant>(inst_set, inst)
            .is_some_and(|is_variant| *is_variant.value() == value)
        || downcast::<&data::EnumAssertVariant>(inst_set, inst)
            .is_some_and(|assert| *assert.value() == value)
    {
        return ControlFlow::Continue(());
    }

    if let Some(mat_stack) = downcast::<&data::ObjMaterializeStack>(inst_set, inst)
        && *mat_stack.object() == value
    {
        // A raw pointer into a holder can read the reference it keeps.
        return on_special(if carrier.0.is_empty() {
            SpecialObjectUse::MaterializeStack(result())
        } else {
            SpecialObjectUse::Unknown
        });
    }
    if let Some(mat_heap) = downcast::<&data::ObjMaterializeHeap>(inst_set, inst)
        && *mat_heap.object() == value
    {
        return on_special(SpecialObjectUse::MaterializeHeap);
    }
    // Callee summaries describe object arguments, not references inside SSA
    // aggregates.
    if let Some(call) = downcast::<&control_flow::Call>(inst_set, inst)
        && call.args().contains(&value)
        && !carrier.is_aggregate()
    {
        let flow = on_special(SpecialObjectUse::Call {
            inst: user,
            value,
            call,
        });
        if let Some(result) = call_borrowed_root_result(function, user, call, value, object_effects)
        {
            worklist.push((result, carrier.borrowed()));
        }
        return flow;
    }
    if let Some(ret) = downcast::<&control_flow::Return>(inst_set, inst)
        && ret.args().iter().any(|arg| *arg == value)
        && carrier.0.is_empty()
    {
        return on_special(SpecialObjectUse::Return { value, ret });
    }

    on_special(SpecialObjectUse::Unknown)
}

/// The fields of an enum variant whose value is `value`.
fn value_fields(values: &[ValueId], value: ValueId) -> impl Iterator<Item = u32> + '_ {
    values
        .iter()
        .enumerate()
        .filter(move |(_, field_value)| **field_value == value)
        .filter_map(|(field, _)| u32::try_from(field).ok())
}

/// Follows a value stored into the memory `object` points to, which `carrier`
/// then describes, back to every local allocation `object` may point into:
/// each becomes a holder. Memory anywhere else is unknown, as is a holder
/// already on the carrier's chain, which would keep a reference into itself.
fn carry_into<T>(
    function: &Function,
    object: ValueId,
    carrier: Carrier,
    worklist: &mut Vec<(ValueId, Carrier)>,
    on_special: &mut impl FnMut(SpecialObjectUse<'_>) -> ControlFlow<T>,
) -> ControlFlow<T> {
    on_special(SpecialObjectUse::Carry)?;
    let mut pending = vec![(object, carrier)];
    let mut seen = FxHashSet::default();
    while let Some((value, mut carrier)) = pending.pop() {
        if !seen.insert((value, carrier.clone())) {
            continue;
        }
        let Some(inst) = function.dfg.value_inst(value) else {
            return on_special(SpecialObjectUse::Unknown);
        };
        let inst_set = function.inst_set();
        let inst_data = function.dfg.inst(inst);
        if downcast::<&data::ObjAlloc>(inst_set, inst_data).is_some() {
            if carrier.0.iter().any(|hop| hop.holder == Some(value)) {
                return on_special(SpecialObjectUse::Unknown);
            }
            carrier.0[0].holder = Some(value);
            worklist.push((value, carrier));
        } else if let Some(phi) = downcast::<&control_flow::Phi>(inst_set, inst_data) {
            pending.extend(phi.args().iter().map(|(arg, _)| (*arg, carrier.clone())));
        } else if let Some(enum_assert_ref) =
            downcast::<&data::EnumAssertVariantRef>(inst_set, inst_data)
        {
            pending.push((*enum_assert_ref.object(), carrier));
        } else if let Some(base) = view_base(function, inst) {
            let first = view_slice(function, inst).map(|slice| slice.first_leaf);
            pending.push((base, carrier.shifted(first)));
        } else {
            return on_special(SpecialObjectUse::Unknown);
        }
    }
    ControlFlow::Continue(())
}

/// The reference a projection views.
fn view_base(function: &Function, inst: InstId) -> Option<ValueId> {
    let inst_set = function.inst_set();
    let inst = function.dfg.inst(inst);
    downcast::<&data::ObjProj>(inst_set, inst)
        .and_then(|proj| proj.values().first().copied())
        .or_else(|| downcast::<&data::ObjIndex>(inst_set, inst).map(|index| *index.object()))
        .or_else(|| {
            downcast::<&data::EnumProj>(inst_set, inst).map(|enum_proj| *enum_proj.object())
        })
}

/// The part of its base's pointee that a projection views, if known.
fn view_slice(function: &Function, inst: InstId) -> Option<shape::AggregateSlice> {
    let ctx = function.ctx();
    let dfg = &function.dfg;
    let inst_set = function.inst_set();
    let inst = dfg.inst(inst);
    let pointee = |base: ValueId| objref_element_ty(ctx, dfg.value_ty(base));
    if let Some(proj) = downcast::<&data::ObjProj>(inst_set, inst) {
        let (&base, indices) = proj.values().split_first()?;
        return shape::aggregate_slice_for_object_path(ctx, pointee(base)?, indices, dfg);
    }
    if let Some(index) = downcast::<&data::ObjIndex>(inst_set, inst) {
        let ty = pointee(*index.object())?;
        return shape::aggregate_slice_for_object_path(ctx, ty, &[*index.index()], dfg);
    }
    let enum_proj = downcast::<&data::EnumProj>(inst_set, inst)?;
    let field = shape::const_u32(dfg, *enum_proj.field())?;
    shape::enum_variant_field_slice(
        ctx,
        pointee(*enum_proj.object())?,
        *enum_proj.variant(),
        field,
    )
}

fn call_root_preserves_locality(
    function: &Function,
    inst: sonatina_ir::InstId,
    call: &control_flow::Call,
    value: ValueId,
    object_effects: &ObjectEffectSummaryMap,
    is_allowed_root_value: &mut impl FnMut(ValueId) -> bool,
) -> bool {
    let Some(sig) = function.ctx().get_sig(*call.callee()) else {
        return false;
    };
    let Some(summary) = object_effects.get(call.callee()) else {
        return false;
    };

    // The walker follows projections, so the argument can have a different type
    // from the allocation whose lifetime is being checked.
    let value_ty = function.dfg.value_ty(value);
    let mut saw_value = false;
    for (idx, &arg) in call.args().iter().enumerate() {
        if arg != value {
            continue;
        }
        saw_value = true;
        let Some(effect) = summary.arg_effects.get(idx) else {
            return false;
        };
        if sig.args().get(idx) != Some(&value_ty)
            || effect.needs_unknown_object_barrier()
            || summary
                .captures
                .iter()
                .any(|capture| capture.src_arg == idx)
        {
            return false;
        }
        if effect.local_only {
            continue;
        }

        let &[result] = function.dfg.inst_results(inst) else {
            return false;
        };
        if !summary.ret_effect.borrows_arg(idx) || !is_allowed_root_value(result) {
            return false;
        }
    }

    saw_value
}

fn call_borrowed_root_result(
    function: &Function,
    inst: sonatina_ir::InstId,
    call: &control_flow::Call,
    value: ValueId,
    object_effects: Option<&ObjectEffectSummaryMap>,
) -> Option<ValueId> {
    let &[result] = function.dfg.inst_results(inst) else {
        return None;
    };
    let summary = object_effects.and_then(|effects| effects.get(call.callee()))?;
    for (idx, &arg) in call.args().iter().enumerate() {
        if arg != value {
            continue;
        }
        if summary.ret_effect.borrows_arg(idx) {
            return Some(result);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{lookup_func, parse_test_module};

    #[test]
    fn projected_call_arguments_preserve_locality_without_retention() {
        for (ty, projection) in [
            ("i64", ""),
            ("[i64; 2]", "v2.objref<i64> = obj.index v1 1.i64;"),
            ("@pair", "v2.objref<i64> = obj.proj v1 1.i64;"),
            (
                "@nested",
                "v3.objref<[@pair; 2]> = obj.proj v1 0.i64;\n\
                 v4.objref<@pair> = obj.index v3 1.i64;\n\
                 v2.objref<i64> = obj.proj v4 1.i64;",
            ),
        ] {
            let argument = if projection.is_empty() { "v1" } else { "v2" };
            for retains in [false, true] {
                let operation = if retains {
                    "obj.store v1 v0;"
                } else {
                    "obj.store v0 42.i64;"
                };
                let module = parse_test_module(&format!(
                    r#"
target = "aarch64-unknown-native"
type @pair = {{ i64, i64 }};
type @nested = {{ [@pair; 2] }};
func private %helper(v0.objref<i64>, v1.objref<objref<i64>>) {{
block0:
    {operation}
    return;
}}
func public %probe(v0.objref<objref<i64>>) {{
block0:
    v1.objref<{ty}> = obj.alloc {ty};
    {projection}
    call %helper {argument} v0;
    return;
}}
"#
                ));
                let effects = compute_object_effect_summaries(&module);
                let local_args = collect_local_object_arg_info_with_effects(&module, &effects);
                let probe = lookup_func(&module, "probe");
                module.func_store.view(probe, |function| {
                    let root = ValueId::from_u32(1);
                    assert_eq!(
                        object_root_stays_local_with_effects(
                            function,
                            root,
                            &effects,
                            |_| true,
                            false,
                            false,
                        ),
                        !retains,
                        "effect-summary locality: {ty}, retains={retains}",
                    );
                    assert_eq!(
                        object_root_stays_local_with(function, root, &local_args, |_| true, false),
                        !retains,
                        "local-argument locality: {ty}, retains={retains}",
                    );
                });
            }
        }
    }

    #[test]
    fn collect_local_object_arg_info_allows_enum_object_ops() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @option_i256 = enum {
    #None,
    #Some(i256),
};

func private %f(v0.objref<@option_i256>) {
block0:
    enum.write_variant v0 #Some (7.i256);
    v1.enumtag(@option_i256) = enum.get_tag v0;
    v2.objref<@option_i256> = enum.assert_variant_ref v0 #Some;
    v3.objref<i256> = enum.proj v2 #Some 0.i8;
    v4.i256 = obj.load v3;
    return;
}
"#,
        );

        let func = lookup_func(&module, "f");
        let local = collect_local_object_arg_info(&module);
        assert_eq!(
            local.get(&func).and_then(|args| args.get(&0)),
            Some(&LocalObjectArgInfo::Borrowed)
        );
    }

    #[test]
    fn object_root_stays_local_rejects_enum_write_variant_payload_escape() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256 };
type @option_pair = enum {
    #None,
    #Some(objref<@pair>),
};

func private %f(v0.objref<@pair>, v1.objref<@option_pair>) {
block0:
    enum.write_variant v1 #Some (v0);
    return;
}
"#,
        );

        let func = lookup_func(&module, "f");
        let stays_local = module.func_store.view(func, |function| {
            let root = function.arg_values[0];
            object_root_stays_local(function, root, &LocalObjectArgMap::default(), false)
        });
        assert!(
            !stays_local,
            "enum.write_variant payload use should make the root non-local"
        );
    }

    #[test]
    fn object_root_stays_local_through_enum_proj_phi_web() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @inner = enum {
    #Off,
    #On(i256),
};

type @outer = enum {
    #None,
    #Some(objref<@inner>),
};

func private %f(v0.objref<@outer>, v1.i1) {
block0:
    v2.objref<@outer> = enum.assert_variant_ref v0 #Some;
    v3.objref<@inner> = enum.proj v2 #Some 0.i8;
    br v1 block1 block2;

block1:
    jump block3;

block2:
    jump block3;

block3:
    v4.objref<@inner> = phi (v3 block1) (v3 block2);
    enum.set_tag v4 #Off;
    return;
}
"#,
        );

        let func = lookup_func(&module, "f");
        let local = collect_local_object_arg_info(&module);
        assert!(
            local.get(&func).is_some_and(|args| args.contains_key(&0)),
            "enum.proj + phi + nested enum op should keep the outer enum root local"
        );
    }

    #[test]
    fn collect_local_object_arg_info_follows_same_root_call_results() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %id(v0.objref<@pair>) -> objref<@pair> {
block0:
    return v0;
}

func private %f(v0.objref<@pair>, v1.i256) {
block0:
    v2.objref<@pair> = call %id v0;
    v3.objref<i256> = obj.proj v2 0.i8;
    obj.store v3 v1;
    return;
}
"#,
        );

        let func = lookup_func(&module, "f");
        let local = collect_local_object_arg_info(&module);
        assert!(
            local.get(&func).is_some_and(|args| args.contains_key(&0)),
            "same-root helper return should keep the arg local"
        );
    }

    #[test]
    fn collect_local_object_arg_info_rejects_returned_arg() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %f(v0.objref<@pair>) -> objref<@pair> {
block0:
    return v0;
}
"#,
        );

        let func = lookup_func(&module, "f");
        let local = collect_local_object_arg_info(&module);
        assert!(
            !local.get(&func).is_some_and(|args| args.contains_key(&0)),
            "returned arg should not stay local"
        );
    }

    #[test]
    fn collect_local_object_arg_info_rejects_phi_escape_for_all_roots() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %f(v0.objref<@pair>, v1.objref<@pair>, v2.i1) {
block0:
    br v2 block1 block2;

block1:
    jump block3;

block2:
    jump block3;

block3:
    v3.objref<@pair> = phi (v0 block1) (v1 block2);
    obj.materialize.heap v3;
    return;
}
"#,
        );

        let func = lookup_func(&module, "f");
        let local = collect_local_object_arg_info(&module);
        assert!(
            !local
                .get(&func)
                .is_some_and(|args| args.contains_key(&0) || args.contains_key(&1)),
            "phi-merged escape should block local-object classification for every root"
        );
    }

    #[test]
    fn collect_local_object_arg_info_rejects_escaping_callee() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %escape(v0.objref<@pair>) {
block0:
    obj.materialize.heap v0;
    return;
}

func private %f(v0.objref<@pair>) {
block0:
    call %escape v0;
    return;
}
"#,
        );

        let func = lookup_func(&module, "f");
        let local = collect_local_object_arg_info(&module);
        assert!(
            !local.get(&func).is_some_and(|args| args.contains_key(&0)),
            "escaping callee should block local-object classification"
        );
    }

    #[test]
    fn collect_local_object_arg_info_rejects_stack_materialized_arg() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %f(v0.objref<@pair>) {
block0:
    v1.*@pair = obj.materialize.stack v0;
    v2.*i256 = gep v1 0.i64 0.i8;
    mstore v2 7.i256 i256;
    return;
}
"#,
        );

        let func = lookup_func(&module, "f");
        let local = collect_local_object_arg_info(&module);
        assert!(
            !local.get(&func).is_some_and(|args| args.contains_key(&0)),
            "stack-materialized args should not be treated as local-only"
        );
    }
}
