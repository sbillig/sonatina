#![doc = include_str!("../../../docs/wasm.md")]

mod abi;
mod builder;
mod runtime;
mod translate;
mod wide;

use sonatina_ir::Module;
use sonatina_triple::TargetTriple;
use waffle::OptOptions;

use crate::{
    backend::{Backend, BackendOptions},
    compile::OptLevel,
    isa::native_layout::legalize_native_module,
};

/// The default size of the linear-memory stack, in bytes.
pub const DEFAULT_STACK_SIZE: u32 = 1 << 20;

#[derive(Debug)]
pub enum WasmError {
    UnsupportedTarget(String),
    Translation(String),
    Emission(String),
}

impl std::fmt::Display for WasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedTarget(message) => write!(f, "unsupported target: {message}"),
            Self::Translation(message) => write!(f, "translation error: {message}"),
            Self::Emission(message) => write!(f, "emission error: {message}"),
        }
    }
}

impl std::error::Error for WasmError {}

#[derive(Debug)]
pub struct WasmArtifact {
    bytes: Vec<u8>,
}

impl WasmArtifact {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Compiles `wasm32-unknown-unknown` modules to WebAssembly binaries.
#[derive(Debug, Clone, Copy)]
pub struct WasmBackend {
    stack_size: u32,
}

impl WasmBackend {
    pub fn new() -> Self {
        Self {
            stack_size: DEFAULT_STACK_SIZE,
        }
    }

    /// Sets the size of the stack holding stack objects and aggregate values,
    /// rounded up to a multiple of 16 bytes.
    pub fn with_stack_size(mut self, bytes: u32) -> Self {
        self.stack_size = bytes.next_multiple_of(16);
        self
    }
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for WasmBackend {
    type Artifact = WasmArtifact;
    type Error = WasmError;

    fn compile_module(
        &self,
        module: &Module,
        options: &BackendOptions,
    ) -> Result<Self::Artifact, Vec<Self::Error>> {
        if module.ctx.triple != TargetTriple::WASM32 {
            return Err(vec![WasmError::UnsupportedTarget(format!(
                "the wasm backend compiles {}, not {}",
                TargetTriple::WASM32,
                module.ctx.triple
            ))]);
        }
        let module = module.clone_for_funcs(&module.funcs());
        legalize_native_module(&module).map_err(|error| vec![WasmError::Translation(error)])?;
        let mut wasm = translate::translate_module(&module, self.stack_size)
            .map_err(|error| vec![WasmError::Translation(error)])?;
        if options.opt_level != OptLevel::O0 {
            wasm.per_func_body(|body| body.optimize(&OptOptions::default()));
        }
        let bytes = wasm
            .to_wasm_bytes()
            .map_err(|error| vec![WasmError::Emission(error.to_string())])?;
        Ok(WasmArtifact { bytes })
    }
}
