//! At O0 (`LateCleanupProfile::Off`) the EVM backend leaves out the steps
//! that change memory placement and calls beyond lowering, so that its
//! output can be checked against the IR: each test compiles a module at O0,
//! where the step does not happen, and at O1, where it does.

use sonatina_codegen::{EvmCompile, OptLevel, object::lowering::SectionLowering};
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
