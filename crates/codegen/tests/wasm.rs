#[path = "wasm/integers.rs"]
mod integers;
#[path = "wasm/memory.rs"]
mod memory;

use sonatina_codegen::{
    Compile, OptLevel,
    isa::wasm::{WasmBackend, WasmError},
};
use sonatina_ir::isa::{Isa, wasm32::Wasm32};
use sonatina_verifier::{VerificationLevel, VerifierConfig, verify_module};
use wasmi::{Engine, Linker, Memory, Store, WasmParams, WasmResults};

pub(crate) const LEVELS: [OptLevel; 3] = [OptLevel::O0, OptLevel::O1, OptLevel::O2];

pub(crate) fn parse(source: &str) -> sonatina_ir::Module {
    let module = sonatina_parser::parse_module(&format!(
        "target = \"{}\"\n{source}",
        Wasm32::default().triple()
    ))
    .unwrap_or_else(|errors| panic!("wasm IR should parse: {errors:?}"))
    .module;
    let report = verify_module(&module, &VerifierConfig::for_level(VerificationLevel::Full));
    assert!(!report.has_errors(), "{report}");
    module
}

pub(crate) fn try_compile(source: &str, level: OptLevel) -> Result<Vec<u8>, Vec<WasmError>> {
    Compile::new(parse(source), WasmBackend::new())
        .with_opt_level(level)
        .compile()
        .map(|artifact| artifact.into_bytes())
}

pub(crate) fn compile(source: &str, level: OptLevel) -> Vec<u8> {
    try_compile(source, level).unwrap_or_else(|errors| panic!("{level:?}: {errors:?}"))
}

/// An instantiated module and its exported memory.
pub(crate) struct Instance {
    store: Store<()>,
    instance: wasmi::Instance,
    memory: Memory,
}

impl Instance {
    pub(crate) fn new(bytes: &[u8]) -> Self {
        Self::with_imports(bytes, |_| {})
    }

    pub(crate) fn with_imports(bytes: &[u8], imports: impl FnOnce(&mut Linker<()>)) -> Self {
        let engine = Engine::default();
        let module = wasmi::Module::new(&engine, bytes).expect("emitted wasm should validate");
        let mut store = Store::new(&engine, ());
        let mut linker = Linker::new(&engine);
        imports(&mut linker);
        let instance = linker
            .instantiate_and_start(&mut store, &module)
            .expect("module should instantiate");
        let memory = instance
            .get_memory(&store, "memory")
            .expect("module exports its memory");
        Self {
            store,
            instance,
            memory,
        }
    }

    pub(crate) fn call<P: WasmParams, R: WasmResults>(
        &mut self,
        name: &str,
        params: P,
    ) -> Result<R, wasmi::Error> {
        self.instance
            .get_typed_func::<P, R>(&self.store, name)
            .unwrap_or_else(|error| panic!("export {name}: {error}"))
            .call(&mut self.store, params)
    }

    /// Fresh memory past the current end, which the module uses only when its
    /// heap grows.
    pub(crate) fn scratch(&mut self, len: usize) -> u32 {
        let pages = len.div_ceil(1 << 16) as u64;
        let previous = self
            .memory
            .grow(&mut self.store, pages)
            .expect("memory should grow");
        (previous << 16) as u32
    }

    pub(crate) fn write(&mut self, address: u32, bytes: &[u8]) {
        self.memory
            .write(&mut self.store, address as usize, bytes)
            .unwrap();
    }

    pub(crate) fn read(&self, address: u32, len: usize) -> Vec<u8> {
        let mut bytes = vec![0; len];
        self.memory
            .read(&self.store, address as usize, &mut bytes)
            .unwrap();
        bytes
    }
}

#[test]
fn adds_and_loops() {
    let source = r#"
func public %add(v0.i64, v1.i64) -> i64 {
block0:
    v2.i64 = add v0 v1;
    return v2;
}
func public %sum_to(v0.i32) -> i32 {
block0:
    jump block1;
block1:
    v1.i32 = phi (0.i32 block0) (v4 block2);
    v2.i32 = phi (0.i32 block0) (v5 block2);
    v3.i1 = lt v1 v0;
    br v3 block2 block3;
block2:
    v4.i32 = add v1 1.i32;
    v5.i32 = add v2 v4;
    jump block1;
block3:
    return v2;
}
"#;
    for level in LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        assert_eq!(instance.call::<_, i64>("add", (-10i64, 25i64)).unwrap(), 15);
        assert_eq!(instance.call::<_, i32>("sum_to", 100).unwrap(), 5050);
        assert_eq!(instance.call::<_, i32>("sum_to", 0).unwrap(), 0);
    }
}

#[test]
fn private_functions_stay_internal() {
    let source = r#"
func private %double(v0.i32) -> i32 {
block0:
    v1.i32 = add v0 v0;
    return v1;
}
func public %quadruple(v0.i32) -> i32 {
block0:
    v1.i32 = call %double v0;
    v2.i32 = call %double v1;
    return v2;
}
"#;
    for level in LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        assert_eq!(instance.call::<_, i32>("quadruple", 5).unwrap(), 20);
        assert!(
            instance
                .instance
                .get_func(&instance.store, "double")
                .is_none()
        );
    }
}

#[test]
fn branch_tables_compare_every_width() {
    let source = r#"
func public %classify(v0.i64) -> i32 {
block0:
    v1.i256 = zext v0 i256;
    br_table v1 block4 (1.i256 block1) (7.i256 block2) (-1.i256 block3);
block1:
    return 10.i32;
block2:
    return 70.i32;
block3:
    return 99.i32;
block4:
    return 0.i32;
}
func public %narrow(v0.i8) -> i32 {
block0:
    br_table v0 block2 (-1.i8 block1);
block1:
    return 1.i32;
block2:
    return 0.i32;
}
"#;
    for level in LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        for (input, expected) in [(1i64, 10), (7, 70), (3, 0), (-1, 0)] {
            assert_eq!(
                instance.call::<_, i32>("classify", input).unwrap(),
                expected
            );
        }
        // Narrow parameters are masked to their width.
        assert_eq!(instance.call::<_, i32>("narrow", -1).unwrap(), 1);
        assert_eq!(instance.call::<_, i32>("narrow", 0x1ff).unwrap(), 1);
        assert_eq!(instance.call::<_, i32>("narrow", 0x7f).unwrap(), 0);
    }
}

#[test]
fn wide_values_pass_as_limbs() {
    let source = r#"
func private %swap_halves(v0.i256) -> i256 {
block0:
    v1.i256 = shl 128.i256 v0;
    v2.i256 = shr 128.i256 v0;
    v3.i256 = or v1 v2;
    return v3;
}
func public %roundtrip(v0.i256, v1.i128) -> (i256, i128) {
block0:
    v2.i256 = call %swap_halves v0;
    v3.i128 = add v1 1.i128;
    return (v2, v3);
}
"#;
    for level in LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        let (a, b, c, d, e, f): (i64, i64, i64, i64, i64, i64) = instance
            .call("roundtrip", (1i64, 2i64, 3i64, 4i64, -1i64, 5i64))
            .unwrap();
        assert_eq!((a, b, c, d), (3, 4, 1, 2));
        assert_eq!((e, f), (0, 6));
    }
}

#[test]
fn imports_come_from_env() {
    let source = r#"
declare external %host_add(i32, i32) -> i32;
declare external %host_byte() -> i8;
func public %call_host(v0.i32) -> i32 {
block0:
    v1.i32 = call %host_add v0 5.i32;
    v2.i8 = call %host_byte;
    v3.i32 = zext v2 i32;
    v4.i32 = add v1 v3;
    return v4;
}
"#;
    for level in LEVELS {
        let mut instance = Instance::with_imports(&compile(source, level), |linker| {
            linker
                .func_wrap("env", "host_add", |a: i32, b: i32| a * 10 + b)
                .unwrap();
            // Bits above the host's i8 result are not part of it.
            linker.func_wrap("env", "host_byte", || 0x102i32).unwrap();
        });
        assert_eq!(instance.call::<_, i32>("call_host", 3).unwrap(), 37);
    }
}

#[test]
fn stack_overflow_traps() {
    let source = r#"
func public %recurse(v0.i64) -> i64 {
block0:
    v1.*[i64; 64] = alloca [i64; 64];
    mstore v1 v0 i64;
    v2.i1 = is_zero v0;
    br v2 block1 block2;
block1:
    return 0.i64;
block2:
    v3.i64 = sub v0 1.i64;
    v4.i64 = call %recurse v3;
    v5.i64 = mload v1 i64;
    v6.i64 = add v4 v5;
    return v6;
}
"#;
    for level in LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        assert_eq!(instance.call::<_, i64>("recurse", 100i64).unwrap(), 5050);
        // Returns release their frames.
        assert_eq!(instance.call::<_, i64>("recurse", 100i64).unwrap(), 5050);
        // A megabyte of 512-byte frames runs out long before a million calls.
        assert!(instance.call::<_, i64>("recurse", 1_000_000i64).is_err());
    }
}

#[test]
fn non_wasm_targets_are_rejected() {
    let module = sonatina_parser::parse_module(
        "target = \"evm-ethereum-osaka\"\nfunc public %f() {\nblock0:\n    return;\n}\n",
    )
    .unwrap()
    .module;
    let errors = Compile::new(module, WasmBackend::new())
        .compile()
        .unwrap_err();
    assert!(matches!(
        errors.as_slice(),
        [WasmError::UnsupportedTarget(_)]
    ));
}
