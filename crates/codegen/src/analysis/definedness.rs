//! Conservative SSA definedness, with optional point-specific memory evidence.
//!
//! Memory reads, call results and parameters are never undef: uninitialized
//! memory holds arbitrary fixed bits, and an undef argument or return value
//! reaches the other side as one arbitrary value. Undef arises only from
//! `undef` itself and from operations with undefined results, and spreads
//! through the instructions that use it.

use std::cell::{Cell, RefCell};

use cranelift_entity::EntityRef;
use rustc_hash::{FxHashMap, FxHashSet};
use sonatina_ir::{
    Function, I256, Immediate, InstId, Type, Value, ValueId,
    func_cursor::{CursorLocation, FuncCursor, InstInserter},
    inst::{BinaryInstKind, InstClassKind, data},
    types::EnumVariantRef,
};

pub(crate) fn value_may_be_undef(
    func: &Function,
    value: ValueId,
    cache: &mut FxHashMap<ValueId, bool>,
    mut override_value: impl FnMut(ValueId) -> Option<bool>,
) -> bool {
    let mut visiting = FxHashSet::default();
    let mut stack = vec![(value, false)];
    while let Some((value, post_order)) = stack.pop() {
        if cache.contains_key(&value) {
            continue;
        }
        if let Some(may_be_undef) = override_value(value) {
            cache.insert(value, may_be_undef);
            continue;
        }
        if post_order {
            visiting.remove(&value);
            let may_be_undef = match func.dfg.value(value) {
                Value::Undef { .. } => true,
                Value::Immediate { .. } | Value::Arg { .. } | Value::Global { .. } => false,
                Value::Inst { inst, .. } => inst_result_may_be_undef(func, *inst, |used| {
                    cache.get(&used).copied().unwrap_or(true)
                }),
            };
            cache.insert(value, may_be_undef);
            continue;
        }
        if !visiting.insert(value) {
            cache.insert(value, true);
            continue;
        }
        stack.push((value, true));
        if let Value::Inst { inst, .. } = func.dfg.value(value)
            && !is_freeze(func, *inst)
        {
            for used in func.dfg.inst(*inst).collect_values().into_iter().rev() {
                if cache.contains_key(&used) {
                    continue;
                }
                if visiting.contains(&used) {
                    cache.insert(used, true);
                } else {
                    stack.push((used, false));
                }
            }
        }
    }
    cache.get(&value).copied().unwrap_or(true)
}

fn is_freeze(func: &Function, inst: InstId) -> bool {
    func.dfg.downcast::<&data::Freeze>(inst).is_some()
}

/// The values of `func` that may be undef, including the `undef` operands
/// themselves: those an `undef` or an operation with an undefined result
/// reaches. Unlike [`value_may_be_undef`], a cycle
/// of phis that no such source reaches stays defined.
pub(crate) fn maybe_undef_values(func: &Function) -> FxHashSet<ValueId> {
    let mut maybe_undef = FxHashSet::default();
    loop {
        let mut changed = false;
        for block in func.layout.iter_block() {
            for inst in func.layout.iter_inst(block) {
                for value in func.dfg.inst(inst).collect_values() {
                    if matches!(func.dfg.value(value), Value::Undef { .. }) {
                        changed |= maybe_undef.insert(value);
                    }
                }
                if inst_result_may_be_undef(func, inst, |value| maybe_undef.contains(&value)) {
                    for &result in func.dfg.inst_results(inst) {
                        changed |= maybe_undef.insert(result);
                    }
                }
            }
        }
        if !changed {
            return maybe_undef;
        }
    }
}

/// The scalar values a `freeze` takes: integers, pointers and enum tags.
/// Undef taint is exact for these; an aggregate built from `undef` taints
/// every part of it, even the parts later inserts define.
pub(crate) fn is_scalar(func: &Function, ty: Type) -> bool {
    ty.is_integral() || ty.is_enum_tag() || ty.is_pointer(func.ctx())
}

/// Whether a scalar `value` may be undef: the operand walk settles most
/// values, and a cycle, which it takes as undef, falls to [`maybe_undef_values`].
pub(crate) fn scalar_may_be_undef(func: &Function, value: ValueId) -> bool {
    is_scalar(func, func.dfg.value_ty(value))
        && value_may_be_undef(func, value, &mut FxHashMap::default(), |_| None)
        && maybe_undef_values(func).contains(&value)
}

/// Freezes `value` just before `inst`, for a transform that puts it where a
/// read, call result or parameter was.
pub(crate) fn freeze_before(func: &mut Function, inst: InstId, value: ValueId) -> ValueId {
    let loc = func.layout.prev_inst_of(inst).map_or(
        CursorLocation::BlockTop(func.layout.inst_block(inst)),
        CursorLocation::At,
    );
    let mut cursor = InstInserter::at_location(loc);
    let freeze = cursor.insert_inst_data_from(
        func,
        inst,
        data::Freeze::new_unchecked(func.inst_set(), value),
    );
    let frozen = func.dfg.make_value(Value::Inst {
        inst: freeze,
        result_idx: 0,
        ty: func.dfg.value_ty(value),
    });
    func.dfg.attach_result(freeze, frozen);
    frozen
}

/// Undef taint while a pass changes a function: [`maybe_undef_values`],
/// recomputed when a value created since cannot be cleared by walking its
/// operands, which takes a cycle of new phis as undef.
#[derive(Default)]
pub(crate) struct UndefTaint {
    maybe_undef: RefCell<FxHashSet<ValueId>>,
    known: Cell<usize>,
}

impl UndefTaint {
    pub(crate) fn new(func: &Function) -> Self {
        Self {
            maybe_undef: RefCell::new(maybe_undef_values(func)),
            known: Cell::new(func.dfg.num_values()),
        }
    }

    pub(crate) fn may_be_undef(&self, func: &Function, value: ValueId) -> bool {
        if value.index() >= self.known.get() {
            let known = self.known.get();
            let walked = value_may_be_undef(func, value, &mut FxHashMap::default(), |value| {
                (value.index() < known).then(|| self.maybe_undef.borrow().contains(&value))
            });
            if !walked {
                return false;
            }
            *self.maybe_undef.borrow_mut() = maybe_undef_values(func);
            self.known.set(func.dfg.num_values());
        }
        self.maybe_undef.borrow().contains(&value)
    }

    /// Whether a scalar `value` may be undef, so it cannot replace a read,
    /// call result or parameter without a `freeze`.
    pub(crate) fn scalar_may_be_undef(&self, func: &Function, value: ValueId) -> bool {
        is_scalar(func, func.dfg.value_ty(value)) && self.may_be_undef(func, value)
    }
}

/// Whether field `idx` of `aggregate` may be undef, following `insert_value`
/// and phis to the values that set it. A cycle of phis adds no undef.
fn field_may_be_undef(
    func: &Function,
    aggregate: ValueId,
    idx: I256,
    operand_may_be_undef: &impl Fn(ValueId) -> bool,
    visiting: &mut FxHashSet<ValueId>,
) -> bool {
    if !visiting.insert(aggregate) {
        return false;
    }
    let Some(inst) = func.dfg.value_inst(aggregate) else {
        return operand_may_be_undef(aggregate);
    };
    if let Some(insert) = func.dfg.downcast::<&data::InsertValue>(inst) {
        return match func.dfg.value_imm(*insert.idx()) {
            Some(field) if field.as_i256() == idx => operand_may_be_undef(*insert.value()),
            Some(_) => {
                field_may_be_undef(func, *insert.dest(), idx, operand_may_be_undef, visiting)
            }
            None => operand_may_be_undef(aggregate),
        };
    }
    if let Some(phi) = func.dfg.cast_phi(inst) {
        return phi
            .args()
            .iter()
            .any(|&(arg, _)| field_may_be_undef(func, arg, idx, operand_may_be_undef, visiting));
    }
    operand_may_be_undef(aggregate)
}

/// Whether `field` of `variant` in the enum `value` may be undef, following
/// phis to the `enum.make`s that built it.
fn variant_field_may_be_undef(
    func: &Function,
    value: ValueId,
    variant: EnumVariantRef,
    field: I256,
    operand_may_be_undef: &impl Fn(ValueId) -> bool,
    visiting: &mut FxHashSet<ValueId>,
) -> bool {
    if !visiting.insert(value) {
        return false;
    }
    let Some(inst) = func.dfg.value_inst(value) else {
        return operand_may_be_undef(value);
    };
    if let Some(make) = func.dfg.downcast::<&data::EnumMake>(inst) {
        // An extract runs only on its own variant, which a variant check
        // ensures; another variant's `enum.make` never reaches it.
        return *make.variant() == variant
            && usize::try_from(field.to_u256().low_u64())
                .ok()
                .and_then(|field| make.values().get(field))
                .is_none_or(|&field| operand_may_be_undef(field));
    }
    if let Some(phi) = func.dfg.cast_phi(inst) {
        return phi.args().iter().any(|&(arg, _)| {
            variant_field_may_be_undef(func, arg, variant, field, operand_may_be_undef, visiting)
        });
    }
    operand_may_be_undef(value)
}

fn inst_result_may_be_undef(
    func: &Function,
    inst: InstId,
    operand_may_be_undef: impl Fn(ValueId) -> bool,
) -> bool {
    if is_freeze(func, inst) {
        return false;
    }
    // A field is as defined as the value last inserted there, even into an
    // aggregate that started as undef.
    if let Some(extract) = func.dfg.downcast::<&data::ExtractValue>(inst)
        && let Some(idx) = func.dfg.value_imm(*extract.idx())
    {
        return field_may_be_undef(
            func,
            *extract.dest(),
            idx.as_i256(),
            &operand_may_be_undef,
            &mut FxHashSet::default(),
        );
    }
    if let Some(extract) = func.dfg.downcast::<&data::EnumExtract>(inst)
        && let Some(field) = func.dfg.value_imm(*extract.field())
    {
        return variant_field_may_be_undef(
            func,
            *extract.value(),
            *extract.variant(),
            field.as_i256(),
            &operand_may_be_undef,
            &mut FxHashSet::default(),
        );
    }
    let inst_data = func.dfg.inst(inst);
    let values = inst_data.collect_values();
    if values.iter().copied().any(&operand_may_be_undef) {
        return true;
    }

    if let InstClassKind::Binary(kind) = inst_data.kind()
        && matches!(
            kind,
            BinaryInstKind::Udiv
                | BinaryInstKind::Sdiv
                | BinaryInstKind::Umod
                | BinaryInstKind::Smod
        )
    {
        let [_, rhs] = values.as_slice() else {
            return true;
        };
        return func.dfg.value_imm(*rhs).is_none_or(Immediate::is_zero);
    }

    false
}
