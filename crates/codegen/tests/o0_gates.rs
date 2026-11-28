//! At O0 (`LateCleanupProfile::Off`) the EVM backend leaves out the steps
//! that change memory placement and calls beyond lowering, so that its
//! output can be checked against the IR: each test compiles a module at O0,
//! where the step does not happen, and at O1, where it does.

use sonatina_codegen::{
    EvmCompile, OptLevel,
    object::lowering::{FixedMallocRule, MallocLowering, SectionLowering},
};
use sonatina_parser::parse_module;

/// The lowering record of each section of `source` compiled at `opt`.
fn records(source: &str, opt: OptLevel) -> Vec<SectionLowering> {
    EvmCompile::new(parse_module(source).expect("the module parses").module)
        .with_opt_level(opt)
        .with_lowering_record(true)
        .compile()
        .expect("the module compiles")
        .iter()
        .flat_map(|artifact| artifact.sections.values())
        .map(|section| section.lowering.clone().expect("a lowering record"))
        .collect()
}

/// The text of the function `name` of the section IR of `record`.
fn func_text<'a>(record: &'a SectionLowering, name: &str) -> &'a str {
    let ir = &record.source_ir;
    let start = ir
        .find(&format!("%{name}("))
        .unwrap_or_else(|| panic!("{name} is in the section IR"));
    let end = ir[start..].find("\n}").map_or(ir.len(), |end| start + end);
    &ir[start..end]
}

/// `%caller` stores `%make`'s output into `v1`. Copy elision makes `%make`
/// write it there; at O0 `%caller` keeps the output in an `alloca` of its
/// own and copies it.
#[test]
fn copy_elision_is_off_at_o0() {
    let source = r#"
target = "evm-ethereum-osaka"

func inline(never) private %make(v0.objref<[i256; 20]>) -> [i256; 20] {
block0:
    v1.[i256; 20] = obj.load v0;
    return v1;
}

func inline(never) private %caller(v0.objref<[[i256; 20]; 2]>) {
block0:
    v1.objref<[i256; 20]> = obj.index v0 0.i8;
    v2.objref<[i256; 20]> = obj.index v0 1.i8;
    v3.[i256; 20] = call %make v2;
    obj.store v1 v3;
    return;
}

func public %entry() {
block0:
    v0.objref<[[i256; 20]; 2]> = obj.alloc [[i256; 20]; 2];
    call %caller v0;
    v1.objref<[i256; 20]> = obj.index v0 0.i8;
    v2.objref<i256> = obj.index v1 0.i8;
    v3.i256 = obj.load v2;
    mstore 0.i256 v3 i256;
    evm_return 0.i256 32.i256;
}

object @Elision {
  section runtime {
    entry %entry;
  }
}
"#;
    for (opt, copies) in [(OptLevel::O0, true), (OptLevel::O1, false)] {
        let records = records(source, opt);
        let caller = func_text(&records[0], "caller");
        assert_eq!(caller.contains("= alloca "), copies, "{opt:?}: {caller}");
    }
}

/// `%get` takes a constant table. At O1 the backend clones it for the
/// table `%runtime` passes, at O0 not.
#[test]
fn constref_specialization_is_off_at_o0() {
    let source = r#"
target = "evm-ethereum-osaka"

global private const [i256; 6] $table = [5, 17, 2, 99, 1234567, 8];

func inline(never) private %get(v0.constref<[i256; 6]>, v1.i256) -> i256 {
    block0:
        v2.constref<i256> = const.index v0 v1;
        v3.i256 = const.load v2;
        return v3;
}

func public %runtime() {
    block0:
        v0.i256 = evm_calldata_load 0.i256;
        v1.constref<[i256; 6]> = const.ref $table;
        v2.i256 = call %get v1 v0;
        mstore 0.i256 v2 i256;
        evm_return 0.i256 32.i256;
}

object @Table {
  section runtime {
    entry %runtime;
  }
}
"#;
    for (opt, cloned) in [(OptLevel::O0, false), (OptLevel::O1, true)] {
        let records = records(source, opt);
        let funcs: Vec<_> = records[0]
            .source_ir
            .lines()
            .filter(|line| line.starts_with("func "))
            .collect();
        let others = funcs
            .iter()
            .any(|line| !line.contains("%get(") && !line.contains("%runtime("));
        assert_eq!(others, cloned, "{opt:?}: {funcs:?}");
    }
}

/// `%entry`'s first buffer is dead before `%entry` halts. At O1 the backend
/// turns it into an `alloca`, at O0 not.
#[test]
fn local_malloc_promotion_is_off_at_o0() {
    let source = r#"
target = "evm-ethereum-osaka"

func inline(never) private %note() {
    block0:
        evm_sstore 0.i256 1.i256;
        return;
}

func public %entry() {
    block0:
        v1.*i256 = evm_malloc 32.i256;
        mstore v1 7.i256 i256;
        call %note;
        v2.i256 = mload v1 i256;
        v3.*i256 = evm_malloc 64.i256;
        v4.i256 = ptr_to_int v3 i256;
        mstore v3 v4 i256;
        v5.i256 = mload v3 i256;
        v6.i256 = add v2 v5;
        mstore 0.i256 v6 i256;
        evm_return 0.i256 32.i256;
}

object @Promote {
  section runtime {
    entry %entry;
  }
}
"#;
    for (opt, mallocs) in [(OptLevel::O0, 2), (OptLevel::O1, 1)] {
        let records = records(source, opt);
        let entry = func_text(&records[0], "entry");
        assert_eq!(
            entry.matches("= evm_malloc ").count(),
            mallocs,
            "{opt:?}: {entry}"
        );
    }
    // At O0 the buffer, allocated before anything moves the free pointer in
    // a section with a malloc that is not transient, and live across a call,
    // which rules out private static words, is placed by the exact-entry
    // rule.
    let records = records(source, OptLevel::O0);
    let rules: Vec<_> = records[0]
        .funcs
        .iter()
        .flat_map(|func| &func.mallocs)
        .collect();
    assert!(
        rules.iter().any(|(_, malloc)| matches!(
            malloc,
            MallocLowering::Fixed {
                rule: FixedMallocRule::ExactEntry,
                ..
            }
        )),
        "{rules:?}"
    );
}
