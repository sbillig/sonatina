use sonatina_ir::{
    BlockId, Function, InstId, InstSetExt, Type, Value, ValueId,
    cfg::ControlFlowGraph,
    inst::{cast::Bitcast, data::Alloca, evm::inst_set::EvmInstKind},
    isa::{Isa, evm::Evm},
    module::ModuleCtx,
};

use crate::{domtree::DomTree, loop_analysis::LoopTree};

use super::{
    malloc_plan::compute_malloc_escape_kinds_for_function,
    private_malloc::PrivateMallocUseAnalysis, ptr_escape::PtrEscapeSummaries,
    ptr_provenance::compute_provenance,
};

/// The largest malloc moved into the frame, so frames stay small.
const MAX_LOCAL_MALLOC_BYTES: u32 = 256;

/// Moves each small constant-size malloc that cannot escape the function,
/// whose address only addresses its own bytes (so no address bits are
/// observable), and that runs at most once per call (it lies in no loop) into
/// a frame slot. Return and revert payloads keep their fixed terminal
/// buffers. Memory
/// planning places the slot in the static arena, so the buffer needs neither
/// the free pointer nor heap memory above every frame. A loop could keep an
/// earlier iteration's buffer alive through a local container, which one
/// frame slot cannot represent, so mallocs in loops stay on the heap.
pub(crate) fn promote_local_mallocs(
    function: &mut Function,
    module: &ModuleCtx,
    isa: &Evm,
    ptr_escape: &PtrEscapeSummaries,
) -> bool {
    let candidates: Vec<(InstId, BlockId, u32, ValueId)> = function
        .layout
        .iter_block()
        .flat_map(|block| {
            function
                .layout
                .iter_inst(block)
                .map(move |inst| (block, inst))
        })
        .filter_map(|(block, inst)| {
            let EvmInstKind::EvmMalloc(malloc) =
                isa.inst_set().resolve_inst(function.dfg.inst(inst))
            else {
                return None;
            };
            let size = function.dfg.value_imm(*malloc.size())?.as_i256().to_u256();
            let size = u32::try_from(size).ok()?;
            (size != 0 && size <= MAX_LOCAL_MALLOC_BYTES).then_some((
                inst,
                block,
                size,
                *malloc.size(),
            ))
        })
        .collect();
    if candidates.is_empty() {
        return false;
    }

    let mut cfg = ControlFlowGraph::new();
    cfg.compute(function);
    let mut dom = DomTree::new();
    dom.compute(&cfg);
    let mut loops = LoopTree::new();
    loops.compute(&cfg, &dom);
    let prov = compute_provenance(function, module, isa, ptr_escape);
    let escape_kinds =
        compute_malloc_escape_kinds_for_function(function, module, isa, ptr_escape, &prov);

    let mut changed = false;
    for (inst, block, size, size_value) in candidates {
        if loops.loop_of_block(block).is_some()
            || escape_kinds.get(&inst).is_some_and(|kinds| kinds.escapes())
        {
            continue;
        }
        let uses = PrivateMallocUseAnalysis::new(function, module, isa, inst);
        // Terminal payloads already get fixed buffers from memory planning.
        let feeds_exit = function.layout.iter_all_insts().any(|user| {
            uses.inst_uses_derived_value(user)
                && matches!(
                    isa.inst_set().resolve_inst(function.dfg.inst(user)),
                    EvmInstKind::EvmReturn(_) | EvmInstKind::EvmRevert(_)
                )
        });
        if feeds_exit
            || uses
                .analyze_bounded_private_address_uses(size_value, false)
                .is_none()
        {
            continue;
        }
        let result = function
            .dfg
            .inst_result(inst)
            .expect("evm_malloc has a result");
        let ptr_ty = function.dfg.value_ty(result);
        let words_ty = function
            .ctx()
            .with_ty_store_mut(|s| s.make_array(Type::I256, size.div_ceil(32) as usize));
        let slot_ty = words_ty.to_ptr(function.ctx());
        let slot = insert_before(
            function,
            inst,
            Alloca::new(isa.inst_set(), words_ty),
            slot_ty,
        );
        let ptr = insert_before(
            function,
            inst,
            Bitcast::new(isa.inst_set(), slot, ptr_ty),
            ptr_ty,
        );
        function.dfg.change_to_alias(result, ptr);
        function.layout.remove_inst(inst);
        function.erase_inst(inst);
        changed = true;
    }
    changed
}

fn insert_before<I: sonatina_ir::Inst>(
    function: &mut Function,
    before: InstId,
    data: I,
    ty: Type,
) -> ValueId {
    let inst = function.dfg.make_inst(data);
    function.propagate_inst_attribution(inst, before);
    function.layout.insert_inst_before(inst, before);
    let value = function.dfg.make_value(Value::Inst {
        inst,
        result_idx: 0,
        ty,
    });
    function.dfg.append_result(inst, value);
    value
}
