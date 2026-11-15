//! Counts of `undef` operands, printed when `SONATINA_UNDEF_STATS` is set.

use std::{collections::BTreeMap, env, fmt::Write};

use sonatina_ir::{ControlFlowGraph, Function, Module, Value, module::FuncRef};

/// The number of `undef` operands in `func`, by the kind of the operand and the mnemonic of the
/// instruction using it. The kind is `dead` in a block the entry does not reach, and otherwise
/// `scalar`, `pointer` or `aggregate` after the operand's type.
fn undef_uses(func: &Function) -> BTreeMap<(&'static str, &'static str), usize> {
    let mut cfg = ControlFlowGraph::default();
    cfg.compute(func);
    let reachable = cfg.reachable_blocks();
    let ctx = func.ctx();

    let mut uses = BTreeMap::new();
    for block in func.layout.iter_block() {
        for inst in func.layout.iter_inst(block) {
            let inst = func.dfg.inst(inst);
            inst.for_each_value(&mut |value| {
                let Value::Undef { ty } = *func.dfg.value(value) else {
                    return;
                };
                let kind = if !reachable[block] {
                    "dead"
                } else if ty.is_integral() || ty.is_enum_tag() {
                    "scalar"
                } else if ty.is_pointer(ctx) || ty.is_obj_ref(ctx) || ty.is_const_ref(ctx) {
                    "pointer"
                } else {
                    "aggregate"
                };
                *uses.entry((kind, inst.as_text())).or_default() += 1;
            });
        }
    }
    uses
}

/// Prints the `undef` operands of `funcs` at pipeline point `stage` as one `UNDEF_STATS` line, if
/// `SONATINA_UNDEF_STATS` is set.
pub(crate) fn report_undef_uses(stage: &str, module: &Module, funcs: &[FuncRef]) {
    if env::var_os("SONATINA_UNDEF_STATS").is_none() {
        return;
    }

    let mut total = BTreeMap::<_, usize>::new();
    let mut funcs_with_undef = 0;
    for &func in funcs {
        let uses = module.func_store.view(func, undef_uses);
        funcs_with_undef += usize::from(!uses.is_empty());
        for (key, count) in uses {
            *total.entry(key).or_default() += count;
        }
    }

    let mut line = format!(
        "UNDEF_STATS stage={stage} funcs={} funcs_with_undef={funcs_with_undef}",
        funcs.len()
    );
    for ((kind, mnemonic), count) in total {
        write!(line, " {kind}.{mnemonic}={count}").unwrap();
    }
    eprintln!("{line}");
}

#[cfg(test)]
mod tests {
    use super::undef_uses;
    use crate::test_support::parse_test_module;

    #[test]
    fn counts_undef_operands_by_kind_and_user() {
        let module = parse_test_module(
            r#"
target = "evm-ethereum-osaka"

type @pair = { i32, i32 };

func private %f(v0.i1, v1.i32) -> @pair {
block0:
    v2.@pair = insert_value undef.@pair 0.i8 v1;
    br v0 block1 block2;

block1:
    jump block2;

block2:
    v3.i32 = phi (undef.i32 block0) (v1 block1);
    v4.i32 = add v3 undef.i32;
    v5.@pair = insert_value v2 1.i8 v4;
    return v5;

block3:
    v6.i32 = add undef.i32 undef.i32;
    return undef.@pair;
}
"#,
        );
        let uses = module.func_store.view(module.funcs()[0], undef_uses);
        let uses: Vec<_> = uses.into_iter().collect();
        assert_eq!(
            uses,
            [
                (("aggregate", "insert_value"), 1),
                (("dead", "add"), 2),
                (("dead", "return"), 1),
                (("scalar", "add"), 1),
                (("scalar", "phi"), 1),
            ]
        );
    }
}
