//! Promote scalar local memory slots into SSA values.
//!
//! This is intentionally narrower than the aggregate scalarizer: it only
//! promotes alloca-backed integral scalar slots whose complete use-chain is
//! exact offset-zero full-width memory traffic.

use rustc_hash::{FxHashMap, FxHashSet};
use sonatina_ir::{
    Function, InstDowncast, InstId, Type, ValueId, Variable,
    builder::SsaBuilder,
    inst::data::{Alloca, Mload, Mstore},
};

use crate::analysis::memory_access::{ExactLocalAddr, MemoryAccessAnalysis};

#[derive(Debug, Default)]
pub struct ScalarAllocaPromote;

struct PromotableAlloca {
    alloca: InstId,
    ty: Type,
    var: Option<Variable>,
    addr_values: FxHashSet<ValueId>,
    path_insts: FxHashSet<InstId>,
}

#[derive(Debug)]
enum Rewrite {
    Load { candidate: usize, result: ValueId },
    Store { candidate: usize, value: ValueId },
}

impl ScalarAllocaPromote {
    pub fn new() -> Self {
        Self
    }

    pub fn run(&mut self, func: &mut Function) -> bool {
        func.rebuild_users();

        let mut analysis = MemoryAccessAnalysis::new();
        let mut candidates = collect_promotable_allocas(func, &mut analysis);
        if candidates.is_empty() {
            return false;
        }

        let mut ssa = SsaBuilder::new();
        ssa.append_all_block_preds(func);
        for candidate in &mut candidates {
            let var = ssa.declare_var(candidate.ty);
            let init = func.dfg.make_undef_value(candidate.ty);
            ssa.def_var(var, init, func.layout.inst_block(candidate.alloca));
            candidate.var = Some(var);
        }

        let addr_to_candidate = build_addr_candidate_map(&candidates);
        let mut changed = rewrite_memory_ops(func, &mut ssa, &candidates, &addr_to_candidate);
        ssa.seal_all(func);
        func.rebuild_users();
        changed |= remove_dead_address_artifacts(func, &candidates);
        if changed {
            func.rebuild_users();
        }
        changed
    }
}

fn collect_promotable_allocas(
    func: &Function,
    analysis: &mut MemoryAccessAnalysis,
) -> Vec<PromotableAlloca> {
    func.layout
        .iter_block()
        .flat_map(|block| func.layout.iter_inst(block))
        .filter_map(|inst| collect_promotable_alloca(func, analysis, inst))
        .collect()
}

fn collect_promotable_alloca(
    func: &Function,
    analysis: &mut MemoryAccessAnalysis,
    alloca: InstId,
) -> Option<PromotableAlloca> {
    let alloca_data = <&Alloca as InstDowncast>::downcast(func.inst_set(), func.dfg.inst(alloca))?;
    let ty = *alloca_data.ty();
    if !ty.is_integral() {
        return None;
    }

    let root = func.dfg.inst_result(alloca)?;
    let mut candidate = PromotableAlloca {
        alloca,
        ty,
        var: None,
        addr_values: FxHashSet::default(),
        path_insts: FxHashSet::default(),
    };
    candidate.addr_values.insert(root);

    let mut saw_memory_op = false;
    let mut worklist = vec![root];
    while let Some(addr) = worklist.pop() {
        let users: Vec<_> = func.dfg.users(addr).copied().collect();
        for user in users {
            if !func.layout.is_inst_inserted(user) {
                continue;
            }
            if promotable_load_user(func, analysis, user, addr, alloca, ty).is_some()
                || promotable_store_user(func, analysis, user, addr, alloca, ty).is_some()
            {
                saw_memory_op = true;
                continue;
            }

            let result = transparent_addr_user(func, analysis, user, alloca)?;
            if candidate.addr_values.insert(result) {
                worklist.push(result);
            }
            candidate.path_insts.insert(user);
        }
    }

    saw_memory_op.then_some(candidate)
}

fn promotable_load_user(
    func: &Function,
    analysis: &mut MemoryAccessAnalysis,
    inst: InstId,
    addr: ValueId,
    root_alloca: InstId,
    ty: Type,
) -> Option<ValueId> {
    let mload = <&Mload as InstDowncast>::downcast(func.inst_set(), func.dfg.inst(inst))?;
    if *mload.addr() != addr
        || *mload.ty() != ty
        || !is_exact_zero_addr(analysis, func, addr, root_alloca)
    {
        return None;
    }
    let result = func.dfg.inst_result(inst)?;
    (func.dfg.value_ty(result) == ty).then_some(result)
}

fn promotable_store_user(
    func: &Function,
    analysis: &mut MemoryAccessAnalysis,
    inst: InstId,
    addr: ValueId,
    root_alloca: InstId,
    ty: Type,
) -> Option<ValueId> {
    let mstore = <&Mstore as InstDowncast>::downcast(func.inst_set(), func.dfg.inst(inst))?;
    if *mstore.addr() != addr
        || *mstore.ty() != ty
        || !is_exact_zero_addr(analysis, func, addr, root_alloca)
    {
        return None;
    }
    let value = *mstore.value();
    (func.dfg.value_ty(value) == ty).then_some(value)
}

fn transparent_addr_user(
    func: &Function,
    analysis: &mut MemoryAccessAnalysis,
    inst: InstId,
    root_alloca: InstId,
) -> Option<ValueId> {
    if func.dfg.effects(inst).summary().has_effect() {
        return None;
    }
    let [result] = func.dfg.inst_results(inst) else {
        return None;
    };
    is_exact_zero_addr(analysis, func, *result, root_alloca).then_some(*result)
}

fn is_exact_zero_addr(
    analysis: &mut MemoryAccessAnalysis,
    func: &Function,
    value: ValueId,
    root_alloca: InstId,
) -> bool {
    analysis.exact_local_addr(func, value)
        == Some(ExactLocalAddr {
            root_alloca,
            offset_bytes: 0,
        })
}

fn build_addr_candidate_map(candidates: &[PromotableAlloca]) -> FxHashMap<ValueId, usize> {
    let mut map = FxHashMap::default();
    for (idx, candidate) in candidates.iter().enumerate() {
        for &addr in &candidate.addr_values {
            map.insert(addr, idx);
        }
    }
    map
}

fn rewrite_memory_ops(
    func: &mut Function,
    ssa: &mut SsaBuilder,
    candidates: &[PromotableAlloca],
    addr_to_candidate: &FxHashMap<ValueId, usize>,
) -> bool {
    let mut changed = false;
    let blocks: Vec<_> = func.layout.iter_block().collect();
    for block in blocks {
        let insts: Vec<_> = func.layout.iter_inst(block).collect();
        for inst in insts {
            if !func.layout.is_inst_inserted(inst) {
                continue;
            }
            let Some(rewrite) = rewrite_for_inst(func, inst, candidates, addr_to_candidate) else {
                continue;
            };
            match rewrite {
                Rewrite::Load { candidate, result } => {
                    let replacement =
                        ssa.use_var(func, candidate_var(&candidates[candidate]), block);
                    func.dfg.change_to_alias(result, replacement);
                    remove_inst(func, inst);
                }
                Rewrite::Store { candidate, value } => {
                    ssa.def_var(candidate_var(&candidates[candidate]), value, block);
                    remove_inst(func, inst);
                }
            }
            changed = true;
        }
    }
    changed
}

fn candidate_var(candidate: &PromotableAlloca) -> Variable {
    candidate
        .var
        .expect("promotable alloca should have an SSA variable before rewriting")
}

fn rewrite_for_inst(
    func: &Function,
    inst: InstId,
    candidates: &[PromotableAlloca],
    addr_to_candidate: &FxHashMap<ValueId, usize>,
) -> Option<Rewrite> {
    if let Some(mload) = <&Mload as InstDowncast>::downcast(func.inst_set(), func.dfg.inst(inst))
        && let Some(&candidate) = addr_to_candidate.get(mload.addr())
        && *mload.ty() == candidates[candidate].ty
    {
        let result = func.dfg.inst_result(inst)?;
        if func.dfg.value_ty(result) == candidates[candidate].ty {
            return Some(Rewrite::Load { candidate, result });
        }
    }

    if let Some(mstore) = <&Mstore as InstDowncast>::downcast(func.inst_set(), func.dfg.inst(inst))
        && let Some(&candidate) = addr_to_candidate.get(mstore.addr())
        && *mstore.ty() == candidates[candidate].ty
        && func.dfg.value_ty(*mstore.value()) == candidates[candidate].ty
    {
        return Some(Rewrite::Store {
            candidate,
            value: *mstore.value(),
        });
    }

    None
}

fn remove_dead_address_artifacts(func: &mut Function, candidates: &[PromotableAlloca]) -> bool {
    let mut changed = false;
    loop {
        let mut removed = false;
        for inst in removable_address_insts(candidates) {
            if func.layout.is_inst_inserted(inst) && inst_results_are_dead(func, inst) {
                remove_inst(func, inst);
                removed = true;
                changed = true;
            }
        }
        if !removed {
            return changed;
        }
        func.rebuild_users();
    }
}

fn removable_address_insts(candidates: &[PromotableAlloca]) -> Vec<InstId> {
    let mut insts = Vec::new();
    let mut seen = FxHashSet::default();
    for candidate in candidates {
        for &inst in &candidate.path_insts {
            if seen.insert(inst) {
                insts.push(inst);
            }
        }
        if seen.insert(candidate.alloca) {
            insts.push(candidate.alloca);
        }
    }
    insts
}

fn inst_results_are_dead(func: &Function, inst: InstId) -> bool {
    func.dfg.inst_results(inst).iter().all(|&value| {
        func.dfg
            .users(value)
            .all(|&user| !func.layout.is_inst_inserted(user))
    })
}

fn remove_inst(func: &mut Function, inst: InstId) {
    func.layout.remove_inst(inst);
    func.erase_inst(inst);
}
