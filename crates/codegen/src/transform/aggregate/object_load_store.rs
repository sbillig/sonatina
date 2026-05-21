use std::time::{Duration, Instant};

use rustc_hash::FxHashMap;
use sonatina_ir::{
    Function, InstId, ValueId,
    func_cursor::{CursorLocation, FuncCursor, InstInserter},
    inst::{data, downcast},
    module::FuncRef,
};

use super::{
    LocalObjectArgInfo, LocalObjectArgMap, ObjectEffectSummaryMap,
    cleanup::DeadPureInstCleanup,
    object_memory::{ObjectMemoryAnalysis, ObjectMemoryCarrier, ObjectReadSource},
    object_tracking::{AggregateObjectFacts, slice_is_covered_by},
    provenance::ProvenanceSnapshot,
    reconstruct::AggregateValueReconstructor,
    shape,
};
use crate::optim::pipeline::{duration_ms, emit_opt_stats, opt_stats_enabled};

#[derive(Default)]
struct ObjectLoadStoreStats {
    enabled: bool,
    func_ref: Option<FuncRef>,
    iterations: usize,
    dataflow_iterations: usize,
    changed_iterations: usize,
    rewrite_changed: usize,
    object_cleanup_changed: usize,
    pure_cleanup_changed: usize,
    rebuild_users_ms: Duration,
    facts_ms: Duration,
    object_memory_ms: Duration,
    rewrite_ms: Duration,
    object_cleanup_ms: Duration,
    pure_cleanup_ms: Duration,
}

impl ObjectLoadStoreStats {
    fn new(func_ref: Option<FuncRef>) -> Self {
        Self {
            enabled: opt_stats_enabled(),
            func_ref,
            ..Self::default()
        }
    }

    fn start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    fn finish(self, func: &Function, changed: bool) {
        if !self.enabled {
            return;
        }

        let blocks = func.layout.iter_block().count();
        let insts = func
            .layout
            .iter_block()
            .map(|block| func.layout.iter_inst(block).count())
            .sum::<usize>();
        let values = func.dfg.value_ids().count();
        emit_opt_stats(format!(
            "sonatina_object_load_store_stats\tfunc_ref={}\tchanged={changed}\tblocks={blocks}\tinsts={insts}\tvalues={values}\titerations={}\tdataflow_iterations={}\tchanged_iterations={}\trewrite_changed={}\tobject_cleanup_changed={}\tpure_cleanup_changed={}\trebuild_users_ms={:.3}\tfacts_ms={:.3}\tobject_memory_ms={:.3}\trewrite_ms={:.3}\tobject_cleanup_ms={:.3}\tpure_cleanup_ms={:.3}",
            self.func_ref
                .map(|func_ref| format!("{func_ref:?}"))
                .unwrap_or_else(|| "-".to_string()),
            self.iterations,
            self.dataflow_iterations,
            self.changed_iterations,
            self.rewrite_changed,
            self.object_cleanup_changed,
            self.pure_cleanup_changed,
            duration_ms(self.rebuild_users_ms),
            duration_ms(self.facts_ms),
            duration_ms(self.object_memory_ms),
            duration_ms(self.rewrite_ms),
            duration_ms(self.object_cleanup_ms),
            duration_ms(self.pure_cleanup_ms),
        ));
    }
}

#[derive(Default)]
pub struct ObjectLoadStore {
    changed: bool,
    layout_cache: shape::AggregateLayoutCache,
    dead_pure_cleanup: DeadPureInstCleanup,
}

impl ObjectLoadStore {
    pub fn run(&mut self, func: &mut Function) -> bool {
        self.run_with_module_facts(None, func, None, None)
    }

    // `local_object_args` must be computed before entering `func_store.modify(...)`.
    pub(crate) fn run_for_func(
        &mut self,
        func_ref: FuncRef,
        func: &mut Function,
        local_object_args: &LocalObjectArgMap,
        object_effects: &ObjectEffectSummaryMap,
    ) -> bool {
        self.run_with_module_facts(
            Some(func_ref),
            func,
            local_object_args.get(&func_ref),
            Some(object_effects),
        )
    }

    fn run_with_module_facts(
        &mut self,
        func_ref: Option<FuncRef>,
        func: &mut Function,
        local_object_args: Option<&FxHashMap<usize, LocalObjectArgInfo>>,
        object_effects: Option<&ObjectEffectSummaryMap>,
    ) -> bool {
        self.changed = false;
        self.layout_cache.clear();
        let mut stats = ObjectLoadStoreStats::new(func_ref);

        loop {
            stats.iterations += 1;
            let rebuild_users_start = stats.start();
            func.rebuild_users();
            if let Some(start) = rebuild_users_start {
                stats.rebuild_users_ms += start.elapsed();
            }

            let mut iter_changed = false;
            if has_object_dataflow_work(func) {
                stats.dataflow_iterations += 1;
                let facts_start = stats.start();
                let mut snapshot = ProvenanceSnapshot::new(func, object_effects);
                let facts = AggregateObjectFacts::for_local_objects_with_effects(
                    func,
                    local_object_args,
                    object_effects,
                    &mut self.layout_cache,
                    &mut snapshot,
                );
                if let Some(start) = facts_start {
                    stats.facts_ms += start.elapsed();
                }

                let object_memory_start = stats.start();
                let mut object_memory = ObjectMemoryAnalysis::default();
                object_memory.compute_object_load_store_facts(
                    func,
                    local_object_args,
                    object_effects,
                    &facts,
                );
                if let Some(start) = object_memory_start {
                    stats.object_memory_ms += start.elapsed();
                }

                let rewrite_start = stats.start();
                let rewrite_changed = self.rewrite_with_object_memory(func, &object_memory);
                if let Some(start) = rewrite_start {
                    stats.rewrite_ms += start.elapsed();
                }
                stats.rewrite_changed += usize::from(rewrite_changed);
                iter_changed |= rewrite_changed;
            }

            if iter_changed {
                let rebuild_users_start = stats.start();
                func.rebuild_users();
                if let Some(start) = rebuild_users_start {
                    stats.rebuild_users_ms += start.elapsed();
                }
            }

            let object_cleanup_start = stats.start();
            let object_cleanup_changed = self.cleanup_dead_object_artifacts(func);
            if let Some(start) = object_cleanup_start {
                stats.object_cleanup_ms += start.elapsed();
            }
            stats.object_cleanup_changed += usize::from(object_cleanup_changed);
            iter_changed |= object_cleanup_changed;

            if iter_changed {
                let rebuild_users_start = stats.start();
                func.rebuild_users();
                if let Some(start) = rebuild_users_start {
                    stats.rebuild_users_ms += start.elapsed();
                }
            }

            let pure_cleanup_start = stats.start();
            let pure_cleanup_changed = self.dead_pure_cleanup.run_with_current_users(func);
            if let Some(start) = pure_cleanup_start {
                stats.pure_cleanup_ms += start.elapsed();
            }
            stats.pure_cleanup_changed += usize::from(pure_cleanup_changed);
            iter_changed |= pure_cleanup_changed;

            self.changed |= iter_changed;
            if !iter_changed {
                stats.finish(func, self.changed);
                return self.changed;
            }
            stats.changed_iterations += 1;
        }
    }

    fn rewrite_with_object_memory(
        &mut self,
        func: &mut Function,
        object_memory: &ObjectMemoryAnalysis,
    ) -> bool {
        let mut changed = false;
        for block in func.layout.iter_block().collect::<Vec<_>>() {
            for inst in func.layout.iter_inst(block).collect::<Vec<_>>() {
                if !func.layout.is_inst_inserted(inst) {
                    continue;
                }
                if self.try_forward_read(func, inst, object_memory)
                    || self.try_remove_write(func, inst, object_memory)
                {
                    changed = true;
                }
            }
        }
        changed
    }

    fn try_forward_read(
        &mut self,
        func: &mut Function,
        inst: InstId,
        object_memory: &ObjectMemoryAnalysis,
    ) -> bool {
        let is_read = downcast::<&data::ObjLoad>(func.inst_set(), func.dfg.inst(inst)).is_some()
            || downcast::<&data::EnumGetTag>(func.inst_set(), func.dfg.inst(inst)).is_some();
        if !is_read {
            return false;
        }
        let Some(read_source) = object_memory.read_source(inst) else {
            return false;
        };
        if read_source.may_be_undef() {
            return false;
        }
        let Some(replacement) = self.replacement_for_read_source(func, inst, read_source) else {
            return false;
        };
        let Some(result) = func.dfg.inst_result(inst) else {
            return false;
        };
        func.dfg.change_to_alias(result, replacement);
        InstInserter::at_location(CursorLocation::At(inst)).remove_inst(func);
        true
    }

    fn replacement_for_read_source(
        &mut self,
        func: &mut Function,
        inst: InstId,
        read_source: ObjectReadSource,
    ) -> Option<ValueId> {
        let ObjectMemoryCarrier::Value {
            value,
            carrier_slice,
        } = read_source.carrier()
        else {
            return None;
        };
        if !func.dfg.has_value(value) {
            return None;
        }
        let read_slice = read_source.read_slice();
        if !slice_is_covered_by(carrier_slice, read_slice) {
            return None;
        }
        if carrier_slice == read_slice && func.dfg.value_ty(value) == read_slice.ty {
            return Some(value);
        }

        let source_slice = shape::aggregate_slice_for_leaf_range(
            func.ctx(),
            carrier_slice.ty,
            read_slice.first_leaf - carrier_slice.first_leaf,
            read_slice.leaf_count,
        )?;
        AggregateValueReconstructor::new(&mut self.layout_cache).rebuild_slice(
            func,
            inst,
            value,
            carrier_slice.ty,
            source_slice,
            read_slice.ty,
        )
    }

    fn try_remove_write(
        &mut self,
        func: &mut Function,
        inst: InstId,
        object_memory: &ObjectMemoryAnalysis,
    ) -> bool {
        if !is_object_write(func, inst) {
            return false;
        }
        let Some(write_state) = object_memory.write_state(inst) else {
            return false;
        };
        if write_state.is_redundant() {
            InstInserter::at_location(CursorLocation::At(inst)).remove_inst(func);
            return true;
        }
        false
    }

    fn cleanup_dead_object_artifacts(&mut self, func: &mut Function) -> bool {
        let mut changed = false;

        loop {
            let mut iter_changed = false;
            for block in func.layout.iter_block().collect::<Vec<_>>() {
                for inst in func.layout.iter_inst(block).collect::<Vec<_>>() {
                    if !func.layout.is_inst_inserted(inst) {
                        continue;
                    }
                    let removable =
                        downcast::<&data::ObjProj>(func.inst_set(), func.dfg.inst(inst)).is_some()
                            || downcast::<&data::ObjIndex>(func.inst_set(), func.dfg.inst(inst))
                                .is_some()
                            || downcast::<&data::EnumProj>(func.inst_set(), func.dfg.inst(inst))
                                .is_some()
                            || downcast::<&data::ObjAlloc>(func.inst_set(), func.dfg.inst(inst))
                                .is_some();
                    if !removable {
                        continue;
                    }
                    let Some(result) = func.dfg.inst_result(inst) else {
                        continue;
                    };
                    if func
                        .dfg
                        .users(result)
                        .copied()
                        .any(|user| func.layout.is_inst_inserted(user))
                    {
                        continue;
                    }
                    InstInserter::at_location(CursorLocation::At(inst)).remove_inst(func);
                    iter_changed = true;
                }
            }
            changed |= iter_changed;
            if !iter_changed {
                return changed;
            }
            func.rebuild_users();
        }
    }
}

fn has_object_dataflow_work(func: &Function) -> bool {
    func.layout.iter_block().any(|block| {
        func.layout.iter_inst(block).any(|inst| {
            let inst_data = func.dfg.inst(inst);
            downcast::<&data::ObjLoad>(func.inst_set(), inst_data).is_some()
                || downcast::<&data::ObjStore>(func.inst_set(), inst_data).is_some()
                || downcast::<&data::EnumGetTag>(func.inst_set(), inst_data).is_some()
                || downcast::<&data::EnumAssertVariantRef>(func.inst_set(), inst_data).is_some()
                || downcast::<&data::EnumSetTag>(func.inst_set(), inst_data).is_some()
                || downcast::<&data::EnumWriteVariant>(func.inst_set(), inst_data).is_some()
        })
    })
}

fn is_object_write(func: &Function, inst: InstId) -> bool {
    downcast::<&data::ObjStore>(func.inst_set(), func.dfg.inst(inst)).is_some()
        || downcast::<&data::EnumSetTag>(func.inst_set(), func.dfg.inst(inst)).is_some()
        || downcast::<&data::EnumWriteVariant>(func.inst_set(), func.dfg.inst(inst)).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonatina_ir::{ir_writer::FuncWriter, module::FuncRef};
    use sonatina_parser::parse_module;

    fn parse_test_module(src: &str) -> sonatina_ir::Module {
        parse_module(src).expect("parse should succeed").module
    }

    fn lookup_func(module: &sonatina_ir::Module, name: &str) -> FuncRef {
        module
            .funcs()
            .into_iter()
            .find(|&func_ref| module.ctx.func_sig(func_ref, |sig| sig.name() == name))
            .expect("function should exist")
    }

    fn run_with_effects(module: &sonatina_ir::Module, func_ref: FuncRef) {
        let object_effects = crate::transform::aggregate::compute_object_effect_summaries(module);
        let local_object_args = crate::transform::aggregate::collect_local_object_arg_info(module);
        module.func_store.modify(func_ref, |func| {
            ObjectLoadStore::default().run_for_func(
                func_ref,
                func,
                &local_object_args,
                &object_effects,
            );
        });
    }

    #[test]
    fn forwards_local_object_arg_field_store_then_load() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %f(v0.objref<@pair>, v1.i256) -> i256 {
    block0:
        v2.objref<i256> = obj.proj v0 0.i8;
        obj.store v2 v1;
        v3.i256 = obj.load v2;
        return v3;
}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        run_with_effects(&module, func_ref);

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                !dumped.contains("obj.load"),
                "local object arg load should be forwarded:\n{dumped}"
            );
            assert!(
                dumped.contains("obj.store v2 v1;"),
                "local object arg mutation must remain visible to the caller:\n{dumped}"
            );
            assert!(
                dumped.contains("return v1;"),
                "forwarded local object arg result should return the stored scalar:\n{dumped}"
            );
        });
    }

    #[test]
    fn forwards_local_object_arg_enum_field_store_then_load_without_lowering() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @option_i256 = enum {
    #None,
    #Some(i256),
};

type @wrapper = { @option_i256, i256 };

func private %f(v0.objref<@wrapper>, v1.i256) -> @option_i256 {
    block0:
        v2.@option_i256 = enum.make @option_i256 #Some (v1);
        v3.objref<@option_i256> = obj.proj v0 0.i8;
        obj.store v3 v2;
        v4.@option_i256 = obj.load v3;
        return v4;
}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        run_with_effects(&module, func_ref);

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                !dumped.contains("obj.load"),
                "enum field load should be forwarded without pre-lowering:\n{dumped}"
            );
            assert!(
                dumped.contains("obj.store v3 v2;"),
                "enum field store must remain visible to the caller:\n{dumped}"
            );
            assert!(
                dumped.contains("return v2;"),
                "forwarded enum field result should return the stored enum value:\n{dumped}"
            );
        });
    }

    #[test]
    fn summary_read_only_call_preserves_forwarding() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %peek(v0.objref<@pair>) -> i256 {
    block0:
        v1.objref<i256> = obj.proj v0 0.i8;
        v2.i256 = obj.load v1;
        return v2;
}

func private %f(v0.objref<@pair>, v1.i256) -> i256 {
    block0:
        v2.objref<i256> = obj.proj v0 0.i8;
        obj.store v2 v1;
        v3.i256 = call %peek v0;
        v4.i256 = obj.load v2;
        return v4;
}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        run_with_effects(&module, func_ref);

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                dumped.contains("call %peek v0;"),
                "call should remain:\n{dumped}"
            );
            assert!(
                !dumped.contains("obj.load v2"),
                "read-only call should not kill forwarding:\n{dumped}"
            );
            assert!(
                dumped.contains("return v1;"),
                "forwarded value should survive the call:\n{dumped}"
            );
        });
    }

    #[test]
    fn summary_write_one_field_only_kills_that_field() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %write_second(v0.objref<@pair>, v1.i256) {
    block0:
        v2.objref<i256> = obj.proj v0 1.i8;
        obj.store v2 v1;
        return;
}

func private %f(v0.objref<@pair>, v1.i256, v2.i256) -> i256 {
    block0:
        v3.objref<i256> = obj.proj v0 0.i8;
        obj.store v3 v1;
        v4.objref<i256> = obj.proj v0 1.i8;
        obj.store v4 v2;
        call %write_second v0 9.i256;
        v5.i256 = obj.load v3;
        return v5;
}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        run_with_effects(&module, func_ref);

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                !dumped.contains("obj.load v3"),
                "callee write to field 1 should not kill field 0 availability:\n{dumped}"
            );
            assert!(
                dumped.contains("return v1;"),
                "field 0 load should still forward:\n{dumped}"
            );
        });
    }

    #[test]
    fn summary_propagates_transitively_through_nested_calls() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %leaf(v0.objref<@pair>) -> i256 {
    block0:
        v1.objref<i256> = obj.proj v0 0.i8;
        v2.i256 = obj.load v1;
        return v2;
}

func private %mid(v0.objref<@pair>) -> i256 {
    block0:
        v1.i256 = call %leaf v0;
        return v1;
}

func private %f(v0.objref<@pair>, v1.i256) -> i256 {
    block0:
        v2.objref<i256> = obj.proj v0 0.i8;
        obj.store v2 v1;
        v3.i256 = call %mid v0;
        v4.i256 = obj.load v2;
        return v4;
}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        run_with_effects(&module, func_ref);

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                !dumped.contains("obj.load v2"),
                "transitive read-only summary should preserve forwarding:\n{dumped}"
            );
            assert!(
                dumped.contains("return v1;"),
                "transitive summary should keep stored value available:\n{dumped}"
            );
        });
    }

    #[test]
    fn fresh_return_summary_tracks_returned_root() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %make_pair() -> objref<@pair> {
    block0:
        v0.objref<@pair> = obj.alloc @pair;
        return v0;
}

func private %f(v0.i256) -> i256 {
    block0:
        v1.objref<@pair> = call %make_pair;
        v2.objref<i256> = obj.proj v1 0.i8;
        obj.store v2 v0;
        v3.i256 = obj.load v2;
        return v3;
}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        run_with_effects(&module, func_ref);

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                !dumped.contains("obj.load v2"),
                "fresh-return helper result should become a tracked root:\n{dumped}"
            );
            assert!(
                dumped.contains("return v0;"),
                "store/load on fresh call result should forward:\n{dumped}"
            );
        });
    }

    #[test]
    fn incomplete_phi_read_summary_keeps_precall_store_live() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

declare external %mystery() -> objref<@pair>;

func private %read_maybe(v0.i1, v1.objref<@pair>) -> i256 {
block0:
    br v0 block1 block2;

block1:
    jump block3;

block2:
    v2.objref<@pair> = call %mystery;
    jump block3;

block3:
    v3.objref<@pair> = phi (v1 block1) (v2 block2);
    v4.objref<i256> = obj.proj v3 0.i8;
    v5.i256 = obj.load v4;
    return v5;
}

func private %main(v0.i1, v1.objref<@pair>, v2.i256) -> i256 {
block0:
    v3.objref<i256> = obj.proj v1 0.i8;
    obj.store v3 v2;
    v4.i256 = call %read_maybe v0 v1;
    return v4;
}
"#,
        );
        let func_ref = lookup_func(&module, "main");
        run_with_effects(&module, func_ref);

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                dumped.contains("obj.store v3 v2;"),
                "pre-call store must stay live when callee may read the arg through incomplete provenance:\n{dumped}"
            );
        });
    }

    #[test]
    fn inexact_fresh_call_read_summary_keeps_precall_store_live() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @cell = { i256 };
type @take = { objref<i256> };

func private %take(v0.objref<@cell>) -> objref<@take> {
block0:
    v1.objref<@take> = obj.alloc @take;
    v2.objref<objref<i256>> = obj.proj v1 0.i8;
    v3.objref<i256> = obj.proj v0 0.i8;
    obj.store v2 v3;
    return v1;
}

func private %read_two_calls(v0.i1, v1.objref<@cell>) -> i256 {
block0:
    br v0 block1 block2;

block1:
    v2.objref<@take> = call %take v1;
    jump block3;

block2:
    v3.objref<@take> = call %take v1;
    jump block3;

block3:
    v4.objref<@take> = phi (v2 block1) (v3 block2);
    v5.objref<objref<i256>> = obj.proj v4 0.i8;
    v6.objref<i256> = obj.load v5;
    v7.i256 = obj.load v6;
    return v7;
}

func private %main(v0.i1, v1.objref<@cell>, v2.i256) -> i256 {
block0:
    v3.objref<i256> = obj.proj v1 0.i8;
    obj.store v3 v2;
    v4.i256 = call %read_two_calls v0 v1;
    return v4;
}
"#,
        );
        let func_ref = lookup_func(&module, "main");
        run_with_effects(&module, func_ref);

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                dumped.contains("obj.store v3 v2;"),
                "pre-call store must stay live when callee may read through inexact fresh helper roots:\n{dumped}"
            );
        });
    }

    #[test]
    fn returned_capture_chain_keeps_source_store_live() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @Take = { i256, objref<[i256; 8]> };

func private %reverse(v0.objref<[i256; 8]>) -> objref<[i256; 8]> {
block0:
    return v0;
}

func private %take(v0.i256, v1.objref<[i256; 8]>) -> objref<@Take> {
block0:
    v2.objref<@Take> = obj.alloc @Take;
    v3.objref<i256> = obj.proj v2 0.i8;
    obj.store v3 v0;
    v4.objref<objref<[i256; 8]>> = obj.proj v2 1.i8;
    obj.store v4 v1;
    return v2;
}

func private %take_get(v0.objref<@Take>, v1.i256) -> i256 {
block0:
    v2.objref<objref<[i256; 8]>> = obj.proj v0 1.i8;
    v3.objref<[i256; 8]> = obj.load v2;
    v4.objref<i256> = obj.index v3 v1;
    v5.i256 = obj.load v4;
    return v5;
}

func private %sum_last4(v0.objref<[i256; 8]>) -> i256 {
block0:
    v1.objref<[i256; 8]> = call %reverse v0;
    v2.objref<@Take> = call %take 4.i256 v1;
    v3.i256 = call %take_get v2 0.i256;
    return v3;
}

func private %main() -> i256 {
block0:
    v0.objref<[i256; 8]> = obj.alloc [i256; 8];
    v1.objref<i256> = obj.index v0 0.i256;
    obj.store v1 4.i256;
    v2.i256 = call %sum_last4 v0;
    return v2;
}
"#,
        );
        let func_ref = lookup_func(&module, "main");
        run_with_effects(&module, func_ref);

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                dumped.contains("obj.store v1 4.i256;"),
                "source store should stay live through returned capture chain:\n{dumped}"
            );
        });
    }

    #[test]
    fn ambiguous_return_capture_keeps_source_store_live() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @Cell = { i256 };
type @Inner = { objref<i256>, objref<i256> };
type @Outer = { @Inner, @Inner };

func private %pick(v0.i1, v1.objref<@Cell>, v2.objref<@Cell>) -> objref<@Inner> {
block0:
    v3.objref<@Outer> = obj.alloc @Outer;
    br v0 block1 block2;

block1:
    v4.objref<@Inner> = obj.proj v3 0.i8;
    v5.objref<objref<i256>> = obj.proj v4 1.i8;
    v6.objref<i256> = obj.proj v1 0.i8;
    obj.store v5 v6;
    v7.objref<objref<i256>> = obj.proj v4 0.i8;
    v8.objref<i256> = obj.proj v2 0.i8;
    obj.store v7 v8;
    jump block3;

block2:
    v9.objref<@Inner> = obj.proj v3 1.i8;
    v10.objref<objref<i256>> = obj.proj v9 0.i8;
    v11.objref<i256> = obj.proj v1 0.i8;
    obj.store v10 v11;
    v12.objref<objref<i256>> = obj.proj v9 1.i8;
    v13.objref<i256> = obj.proj v2 0.i8;
    obj.store v12 v13;
    jump block3;

block3:
    v14.objref<@Inner> = phi (v4 block1) (v9 block2);
    return v14;
}

func private %read_first(v0.objref<@Inner>) -> i256 {
block0:
    v1.objref<objref<i256>> = obj.proj v0 0.i8;
    v2.objref<i256> = obj.load v1;
    v3.i256 = obj.load v2;
    return v3;
}

func private %main(v0.i1) -> i256 {
block0:
    v1.objref<@Cell> = obj.alloc @Cell;
    v2.objref<i256> = obj.proj v1 0.i8;
    obj.store v2 4.i256;
    v3.objref<@Cell> = obj.alloc @Cell;
    v4.objref<i256> = obj.proj v3 0.i8;
    obj.store v4 9.i256;
    v5.objref<@Inner> = call %pick v0 v1 v3;
    v6.i256 = call %read_first v5;
    return v6;
}
"#,
        );
        let func_ref = lookup_func(&module, "main");
        run_with_effects(&module, func_ref);

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                dumped.contains("obj.store v2 4.i256;"),
                "ambiguous returned capture should keep the source store live:\n{dumped}"
            );
        });
    }

    #[test]
    fn overwritten_captured_pointer_store_stays_when_dead_write_elim_is_disabled() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @Cell = { i256 };
type @Holder = { objref<@Cell> };

func private %f(v0.i256) -> i256 {
block0:
    v1.objref<@Cell> = obj.alloc @Cell;
    v2.objref<i256> = obj.proj v1 0.i8;
    obj.store v2 11.i256;
    v3.objref<@Cell> = obj.alloc @Cell;
    v4.objref<i256> = obj.proj v3 0.i8;
    obj.store v4 v0;
    v5.objref<@Holder> = obj.alloc @Holder;
    v6.objref<objref<@Cell>> = obj.proj v5 0.i8;
    obj.store v6 v1;
    obj.store v6 v3;
    v7.objref<@Cell> = obj.load v6;
    v8.objref<i256> = obj.proj v7 0.i8;
    v9.i256 = obj.load v8;
    return v9;
}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        run_with_effects(&module, func_ref);

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                dumped.contains("obj.store v2 11.i256;"),
                "ObjectLoadStore should preserve stores only proven dead by write-use tracking:\n{dumped}"
            );
            assert!(
                dumped.contains("return v0;"),
                "precise overwritten capture provenance should let the final load collapse to the live source value:\n{dumped}"
            );
        });
    }

    #[test]
    fn forwards_store_into_successor_block() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %f(v0.i256) -> i256 {
    block0:
        v1.objref<@pair> = obj.alloc @pair;
        v2.objref<i256> = obj.proj v1 0.i8;
        obj.store v2 v0;
        jump block1;

    block1:
        v3.objref<i256> = obj.proj v1 0.i8;
        v4.i256 = obj.load v3;
        return v4;
}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        module.func_store.modify(func_ref, |func| {
            assert!(ObjectLoadStore::default().run(func))
        });

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                !dumped.contains("obj.load"),
                "store in predecessor should forward into successor:\n{dumped}"
            );
            assert!(
                dumped.contains("return v0;"),
                "successor should return the predecessor's stored value:\n{dumped}"
            );
        });
    }

    #[test]
    fn forwards_identical_pred_stores_into_join_block() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %f(v0.i1, v1.i256) -> i256 {
    block0:
        v2.objref<@pair> = obj.alloc @pair;
        br v0 block1 block2;

    block1:
        v3.objref<i256> = obj.proj v2 0.i8;
        obj.store v3 v1;
        jump block3;

    block2:
        v4.objref<i256> = obj.proj v2 0.i8;
        obj.store v4 v1;
        jump block3;

    block3:
        v5.objref<i256> = obj.proj v2 0.i8;
        v6.i256 = obj.load v5;
        return v6;
}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        module.func_store.modify(func_ref, |func| {
            assert!(ObjectLoadStore::default().run(func))
        });

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                !dumped.contains("obj.load"),
                "matching predecessor stores should meet at the join:\n{dumped}"
            );
            assert!(
                dumped.contains("return v1;"),
                "join block should forward the common stored value:\n{dumped}"
            );
        });
    }

    #[test]
    fn does_not_forward_differing_pred_stores_into_join_block() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %f(v0.i1, v1.i256, v2.i256) -> i256 {
    block0:
        v3.objref<@pair> = obj.alloc @pair;
        br v0 block1 block2;

    block1:
        v4.objref<i256> = obj.proj v3 0.i8;
        obj.store v4 v1;
        jump block3;

    block2:
        v5.objref<i256> = obj.proj v3 0.i8;
        obj.store v5 v2;
        jump block3;

    block3:
        v6.objref<i256> = obj.proj v3 0.i8;
        v7.i256 = obj.load v6;
        return v7;
}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        module.func_store.modify(func_ref, |func| {
            assert!(!ObjectLoadStore::default().run(func))
        });

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                dumped.contains("obj.load v6"),
                "join should not forward when predecessor stores disagree:\n{dumped}"
            );
        });
    }

    #[test]
    fn preserves_dead_predecessor_store_before_successor_overwrite() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %use(v0.objref<@pair>) {
    block0:
        return;
}

func private %f(v0.i256, v1.i256) -> i256 {
    block0:
        v2.objref<@pair> = obj.alloc @pair;
        v3.objref<i256> = obj.proj v2 0.i8;
        obj.store v3 v0;
        jump block1;

    block1:
        v4.objref<i256> = obj.proj v2 0.i8;
        obj.store v4 v1;
        call %use v2;
        return v1;

}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        module.func_store.modify(func_ref, |func| {
            assert!(!ObjectLoadStore::default().run(func))
        });

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert_eq!(
                dumped.matches("obj.store").count(),
                2,
                "ObjectLoadStore should not remove stores only proven dead by write-use tracking:\n{dumped}"
            );
            assert!(
                dumped.contains("return v1;"),
                "successor overwrite should remain as the visible store:\n{dumped}"
            );
        });
    }

    #[test]
    fn forwards_header_store_into_loop_body() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i256, i256 };

func private %f(v0.i256, v1.i1) -> i256 {
    block0:
        v2.objref<@pair> = obj.alloc @pair;
        jump block1;

    block1:
        v3.objref<i256> = obj.proj v2 0.i8;
        obj.store v3 v0;
        br v1 block2 block3;

    block2:
        v4.objref<i256> = obj.proj v2 0.i8;
        v5.i256 = obj.load v4;
        jump block1;

    block3:
        v6.objref<i256> = obj.proj v2 0.i8;
        v7.i256 = obj.load v6;
        return v7;
}
"#,
        );
        let func_ref = lookup_func(&module, "f");
        module.func_store.modify(func_ref, |func| {
            assert!(ObjectLoadStore::default().run(func))
        });

        module.func_store.view(func_ref, |func| {
            let dumped = FuncWriter::new(func_ref, func).dump_string();
            assert!(
                !dumped.contains("obj.load"),
                "header store should forward into both loop body and exit:\n{dumped}"
            );
            assert!(
                dumped.contains("return v0;"),
                "exit should return the header-stored value:\n{dumped}"
            );
        });
    }
}
