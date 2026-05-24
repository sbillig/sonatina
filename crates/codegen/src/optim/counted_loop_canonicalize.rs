use rustc_hash::FxHashMap;
use sonatina_ir::{
    BlockId, ControlFlowGraph, Function, Immediate, Inst, InstDowncast, InstId, Type, Value,
    ValueId,
    func_cursor::{CursorLocation, FuncCursor, InstInserter},
    inst::{
        arith::Sub,
        cast::Zext,
        cmp::{Lt, Ne},
        control_flow::{Br, Jump},
    },
};
use sonatina_triple::Architecture;

use crate::{
    analysis::induction::{BasicInductionVar, StepKind, detect_basic_ivs_for_loop},
    domtree::DomTree,
    loop_analysis::{Loop, LoopTree},
};

#[derive(Debug, Default)]
pub struct CountedLoopCanonicalize;

#[derive(Debug)]
struct RewritePlan {
    preheader: BlockId,
    header: BlockId,
    latch: BlockId,
    body: BlockId,
    exit: BlockId,
    preheader_term: InstId,
    header_term: InstId,
    latch_term: InstId,
    cmp_inst: InstId,
    step_inst: InstId,
    iv_phi_inst: InstId,
    iv: ValueId,
    bound: ValueId,
    ty: Type,
    loop_blocks: Vec<BlockId>,
    exit_phis: Vec<ExitPhiPlan>,
}

#[derive(Debug)]
struct ExitPhiPlan {
    header_value: ValueId,
    init: ValueId,
    backedge: ValueId,
    ty: Type,
}

struct ExitReplacements {
    values: FxHashMap<ValueId, ValueId>,
    phi_insts: Vec<InstId>,
}

#[derive(Debug, Clone, Copy)]
struct HeaderShape {
    term: InstId,
    body: BlockId,
    exit: BlockId,
}

impl CountedLoopCanonicalize {
    pub fn new() -> Self {
        Self
    }

    pub fn run(
        &mut self,
        func: &mut Function,
        cfg: &mut ControlFlowGraph,
        domtree: &mut DomTree,
        lpt: &mut LoopTree,
    ) -> bool {
        let mut changed = false;

        loop {
            cfg.compute(func);
            domtree.compute(cfg);
            lpt.compute(cfg, domtree);

            let mut changed_this_round = false;
            let loops: Vec<_> = lpt.loops().collect();
            for loop_id in loops {
                if !func.layout.is_block_inserted(lpt.loop_header(loop_id)) {
                    continue;
                }

                let Some(plan) = plan_loop(func, cfg, lpt, loop_id) else {
                    continue;
                };

                apply_plan(func, &plan);
                changed = true;
                changed_this_round = true;
                break;
            }

            if !changed_this_round {
                break;
            }
        }

        changed
    }
}

fn plan_loop(
    func: &Function,
    cfg: &ControlFlowGraph,
    lpt: &LoopTree,
    loop_id: Loop,
) -> Option<RewritePlan> {
    let header = lpt.loop_header(loop_id);
    let header_term = func.layout.last_inst_of(header)?;
    let header_br = <&Br as InstDowncast>::downcast(func.inst_set(), func.dfg.inst(header_term))?;
    let body = *header_br.nz_dest();
    let exit = *header_br.z_dest();
    if body == header
        || exit == header
        || !lpt.is_in_loop(body, loop_id)
        || lpt.is_in_loop(exit, loop_id)
        || !has_only_loop_exit_edge(cfg, lpt, loop_id, header, exit)
        || !has_only_pred(cfg, body, header)
        || !has_only_pred(cfg, exit, header)
        || block_starts_with_phi(func, exit)
    {
        return None;
    }

    let header_shape = HeaderShape {
        term: header_term,
        body,
        exit,
    };
    for biv in detect_basic_ivs_for_loop(func, loop_id, cfg, lpt) {
        if let Some(plan) = plan_with_biv(func, cfg, lpt, loop_id, &biv, header_shape) {
            return Some(plan);
        }
    }

    None
}

fn plan_with_biv(
    func: &Function,
    cfg: &ControlFlowGraph,
    lpt: &LoopTree,
    loop_id: Loop,
    biv: &BasicInductionVar,
    header: HeaderShape,
) -> Option<RewritePlan> {
    let ty = func.dfg.value_ty(biv.phi);
    if ty == Type::I1
        || func.dfg.value_imm(biv.init) != Some(Immediate::zero(ty))
        || !matches!(biv.step, StepKind::AddConst(step) if step == Immediate::one(ty))
    {
        return None;
    }

    let (cmp_inst, bound) = match_loop_guard(func, header.term, biv.phi)?;
    if func.dfg.value_ty(bound) != ty || !is_loop_invariant(func, lpt, loop_id, bound) {
        return None;
    }

    let preheader_term = match_jump_to(func, biv.preheader, biv.header)?;
    let latch_term = match_jump_to(func, biv.latch, biv.header)?;
    let iv_phi_inst = func.dfg.value_inst(biv.phi)?;
    if !iv_uses_are_only_guard_and_step(func, biv.phi, cmp_inst, biv.step_inst)
        || !value_users_are_only(func, func.dfg.inst_result(cmp_inst)?, &[header.term])
        || !value_users_are_only(func, biv.step_value, &[iv_phi_inst])
    {
        return None;
    }

    let exit_phis = collect_exit_phi_plans(func, lpt, loop_id, biv)?;

    Some(RewritePlan {
        preheader: biv.preheader,
        header: biv.header,
        latch: biv.latch,
        body: header.body,
        exit: header.exit,
        preheader_term,
        header_term: header.term,
        latch_term,
        cmp_inst,
        step_inst: biv.step_inst,
        iv_phi_inst,
        iv: biv.phi,
        bound,
        ty,
        loop_blocks: lpt.iter_blocks_post_order(cfg, loop_id).collect(),
        exit_phis,
    })
}

fn match_loop_guard(
    func: &Function,
    header_term: InstId,
    iv: ValueId,
) -> Option<(InstId, ValueId)> {
    let br = <&Br as InstDowncast>::downcast(func.inst_set(), func.dfg.inst(header_term))?;
    let cmp_inst = func.dfg.value_inst(*br.cond())?;
    let lt = <&Lt as InstDowncast>::downcast(func.inst_set(), func.dfg.inst(cmp_inst))?;
    (*lt.lhs() == iv).then_some((cmp_inst, *lt.rhs()))
}

fn is_loop_invariant(func: &Function, lpt: &LoopTree, loop_id: Loop, value: ValueId) -> bool {
    match func.dfg.value(value) {
        Value::Immediate { .. } | Value::Arg { .. } | Value::Global { .. } => true,
        Value::Undef { .. } => false,
        Value::Inst { inst, .. } => {
            let inst = *inst;
            func.layout.is_inst_inserted(inst)
                && !lpt.is_in_loop(func.layout.inst_block(inst), loop_id)
        }
    }
}

fn match_jump_to(func: &Function, block: BlockId, dest: BlockId) -> Option<InstId> {
    let term = func.layout.last_inst_of(block)?;
    let jump = <&Jump as InstDowncast>::downcast(func.inst_set(), func.dfg.inst(term))?;
    (*jump.dest() == dest).then_some(term)
}

fn has_only_pred(cfg: &ControlFlowGraph, block: BlockId, pred: BlockId) -> bool {
    let mut preds = cfg.preds_of(block).copied();
    preds.next() == Some(pred) && preds.next().is_none()
}

fn has_only_loop_exit_edge(
    cfg: &ControlFlowGraph,
    lpt: &LoopTree,
    loop_id: Loop,
    exit_pred: BlockId,
    exit: BlockId,
) -> bool {
    let mut saw_exit = false;
    for block in lpt.iter_blocks_post_order(cfg, loop_id) {
        for &succ in cfg.succs_of(block) {
            if !lpt.is_in_loop(succ, loop_id) {
                if block != exit_pred || succ != exit {
                    return false;
                }
                saw_exit = true;
            }
        }
    }
    saw_exit
}

fn block_starts_with_phi(func: &Function, block: BlockId) -> bool {
    func.layout
        .first_inst_of(block)
        .is_some_and(|inst| func.dfg.is_phi(inst))
}

fn iv_uses_are_only_guard_and_step(
    func: &Function,
    iv: ValueId,
    cmp_inst: InstId,
    step_inst: InstId,
) -> bool {
    value_users_are_only(func, iv, &[cmp_inst, step_inst])
}

fn value_users_are_only(func: &Function, value: ValueId, allowed: &[InstId]) -> bool {
    func.dfg
        .users(value)
        .copied()
        .filter(|user| func.layout.is_inst_inserted(*user))
        .all(|user| allowed.contains(&user))
}

fn collect_exit_phi_plans(
    func: &Function,
    lpt: &LoopTree,
    loop_id: Loop,
    biv: &BasicInductionVar,
) -> Option<Vec<ExitPhiPlan>> {
    let mut plans = Vec::new();
    let mut next = func.layout.first_inst_of(biv.header);
    while let Some(phi_inst) = next {
        let Some(phi) = func.dfg.cast_phi(phi_inst) else {
            break;
        };
        let header_value = func.dfg.inst_result(phi_inst)?;
        if header_value == biv.phi {
            next = func.layout.next_inst_of(phi_inst);
            continue;
        }

        if has_outside_uses(func, lpt, loop_id, header_value) {
            plans.push(ExitPhiPlan {
                header_value,
                init: phi_arg(phi, biv.preheader)?,
                backedge: phi_arg(phi, biv.latch)?,
                ty: func.dfg.value_ty(header_value),
            });
        }

        next = func.layout.next_inst_of(phi_inst);
    }

    Some(plans)
}

fn has_outside_uses(func: &Function, lpt: &LoopTree, loop_id: Loop, value: ValueId) -> bool {
    func.dfg.users(value).copied().any(|user| {
        func.layout.is_inst_inserted(user) && !lpt.is_in_loop(func.layout.inst_block(user), loop_id)
    })
}

fn phi_arg(phi: &sonatina_ir::inst::control_flow::Phi, pred: BlockId) -> Option<ValueId> {
    phi.args()
        .iter()
        .find_map(|&(value, block)| (block == pred).then_some(value))
}

fn apply_plan(func: &mut Function, plan: &RewritePlan) {
    let exit_replacements = insert_exit_phis(func, plan);
    rewrite_exit_uses(func, plan, &exit_replacements);

    let zero = func.dfg.make_imm_value(Immediate::zero(plan.ty));
    let countdown_ty = select_countdown_ty(func, plan.ty);
    let countdown_init = make_countdown_init(func, plan, countdown_ty);
    let uses_existing_iv = countdown_ty == plan.ty;
    let remaining = if uses_existing_iv {
        plan.iv
    } else {
        let remaining = insert_phi_at_header_start(
            func,
            plan.header,
            plan.preheader,
            countdown_init,
            countdown_ty,
        );
        detach_iv_phi_backedge(func, plan);
        remaining
    };
    let countdown_one = func.dfg.make_imm_value(Immediate::one(countdown_ty));
    let countdown_zero = func.dfg.make_imm_value(Immediate::zero(countdown_ty));
    let remaining_next = insert_inst_before(
        func,
        plan.latch_term,
        Sub::new(func.inst_set(), remaining, countdown_one),
        countdown_ty,
    );
    let preheader_cond = insert_inst_before(
        func,
        plan.preheader_term,
        Ne::new(func.inst_set(), plan.bound, zero),
        Type::I1,
    );
    let latch_cond = if countdown_ty == plan.ty {
        insert_inst_before(
            func,
            plan.latch_term,
            Ne::new(func.inst_set(), remaining_next, countdown_zero),
            Type::I1,
        )
    } else {
        insert_inst_before(
            func,
            plan.latch_term,
            Lt::new(func.inst_set(), countdown_zero, remaining_next),
            Type::I1,
        )
    };

    if uses_existing_iv {
        rewrite_iv_phi(func, plan, countdown_init, remaining_next);
    } else {
        add_countdown_backedge(func, remaining, plan.latch, remaining_next);
    }
    func.dfg.replace_inst(
        plan.preheader_term,
        Box::new(Br::new(
            func.inst_set(),
            preheader_cond,
            plan.header,
            plan.exit,
        )),
    );
    func.dfg.replace_inst(
        plan.header_term,
        Box::new(Jump::new(func.inst_set(), plan.body)),
    );
    func.dfg.replace_inst(
        plan.latch_term,
        Box::new(Br::new(func.inst_set(), latch_cond, plan.header, plan.exit)),
    );

    remove_dead_single_result_inst(func, plan.step_inst);
    remove_dead_single_result_inst(func, plan.cmp_inst);
    remove_dead_single_result_inst(func, plan.iv_phi_inst);
    func.rebuild_users();
}

fn select_countdown_ty(func: &Function, ty: Type) -> Type {
    if matches!(func.ctx().triple.architecture, Architecture::Riscv64im)
        && matches!(ty, Type::I8 | Type::I16 | Type::I32)
    {
        Type::I64
    } else {
        ty
    }
}

fn make_countdown_init(func: &mut Function, plan: &RewritePlan, countdown_ty: Type) -> ValueId {
    if countdown_ty == plan.ty {
        return plan.bound;
    }

    insert_inst_before(
        func,
        plan.preheader_term,
        Zext::new(func.inst_set(), plan.bound, countdown_ty),
        countdown_ty,
    )
}

fn insert_phi_at_header_start(
    func: &mut Function,
    header: BlockId,
    preheader: BlockId,
    init: ValueId,
    ty: Type,
) -> ValueId {
    let phi = func.dfg.make_phi(vec![(init, preheader)]);
    let mut cursor = InstInserter::at_location(CursorLocation::BlockTop(header));
    let inst = cursor.insert_inst_data(func, phi);
    let value = cursor.make_result(func, inst, ty);
    cursor.attach_result(func, inst, value);
    value
}

fn insert_exit_phis(func: &mut Function, plan: &RewritePlan) -> ExitReplacements {
    let mut values = FxHashMap::default();
    let mut phi_insts = Vec::new();
    for exit_phi in &plan.exit_phis {
        let phi = func.dfg.make_phi(vec![
            (exit_phi.init, plan.preheader),
            (exit_phi.backedge, plan.latch),
        ]);
        let mut cursor = InstInserter::at_location(CursorLocation::BlockTop(plan.exit));
        let inst = cursor.insert_inst_data(func, phi);
        let value = cursor.make_result(func, inst, exit_phi.ty);
        cursor.attach_result(func, inst, value);
        values.insert(exit_phi.header_value, value);
        phi_insts.push(inst);
    }
    ExitReplacements { values, phi_insts }
}

fn rewrite_exit_uses(func: &mut Function, plan: &RewritePlan, replacements: &ExitReplacements) {
    if replacements.values.is_empty() {
        return;
    }

    let blocks: Vec<_> = func.layout.iter_block().collect();
    for block in blocks {
        if plan.loop_blocks.contains(&block) {
            continue;
        }

        let insts: Vec<_> = func.layout.iter_inst(block).collect();
        for inst in insts {
            if replacements.phi_insts.contains(&inst) {
                continue;
            }

            let mut used = false;
            func.dfg.inst(inst).for_each_value(&mut |value| {
                used |= replacements.values.contains_key(&value);
            });
            if !used {
                continue;
            }

            func.dfg.untrack_inst(inst);
            func.dfg.inst_mut(inst).for_each_value_mut(&mut |value| {
                if let Some(&replacement) = replacements.values.get(value) {
                    *value = replacement;
                }
            });
            func.dfg.attach_user(inst);
        }
    }
}

fn insert_inst_before<I>(func: &mut Function, before: InstId, inst: I, ty: Type) -> ValueId
where
    I: Inst + 'static,
{
    let inst = func.dfg.make_inst(inst);
    let value = func.dfg.make_value(Value::Inst {
        inst,
        result_idx: 0,
        ty,
    });
    func.dfg.attach_result(inst, value);
    func.layout.insert_inst_before(inst, before);
    value
}

fn rewrite_iv_phi(
    func: &mut Function,
    plan: &RewritePlan,
    countdown_init: ValueId,
    remaining_next: ValueId,
) {
    func.dfg.untrack_inst(plan.iv_phi_inst);
    let phi = func.dfg.cast_phi_mut(plan.iv_phi_inst).unwrap();
    for (value, pred) in phi.args_mut() {
        if *pred == plan.preheader {
            *value = countdown_init;
        } else if *pred == plan.latch {
            *value = remaining_next;
        }
    }
    func.dfg.attach_user(plan.iv_phi_inst);
}

fn detach_iv_phi_backedge(func: &mut Function, plan: &RewritePlan) {
    let init = {
        let phi = func.dfg.cast_phi(plan.iv_phi_inst).unwrap();
        phi_arg(phi, plan.preheader).unwrap()
    };
    func.dfg.untrack_inst(plan.iv_phi_inst);
    let phi = func.dfg.cast_phi_mut(plan.iv_phi_inst).unwrap();
    for (value, pred) in phi.args_mut() {
        if *pred == plan.latch {
            *value = init;
        }
    }
    func.dfg.attach_user(plan.iv_phi_inst);
}

fn add_countdown_backedge(
    func: &mut Function,
    remaining: ValueId,
    latch: BlockId,
    remaining_next: ValueId,
) {
    let phi_inst = func.dfg.value_inst(remaining).unwrap();
    func.dfg.untrack_inst(phi_inst);
    let phi = func.dfg.cast_phi_mut(phi_inst).unwrap();
    phi.append_phi_arg(remaining_next, latch);
    func.dfg.attach_user(phi_inst);
}

fn remove_dead_single_result_inst(func: &mut Function, inst: InstId) {
    if !func.layout.is_inst_inserted(inst) || !func.dfg.can_drop_if_unused(inst) {
        return;
    }

    let Some(result) = func.dfg.inst_result(inst) else {
        return;
    };
    if func.dfg.users_num(result) != 0 {
        return;
    }

    func.layout.remove_inst(inst);
    func.erase_inst(inst);
}
