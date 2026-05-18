use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use cranelift_module::FuncId;
use sonatina_ir::Module;
use sonatina_triple::{Architecture, OperatingSystem, Vendor};

use super::{CraneliftBackend, CraneliftError};

const SP1_STACK_TOP: u64 = 0x7800_0000;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct Sp1ElfArtifact {
    pub bytes: Vec<u8>,
    pub func_map: HashMap<String, FuncId>,
}

#[derive(Clone, Copy)]
enum Sp1Target {
    Riscv32im,
}

impl Sp1Target {
    fn from_module(module: &Module) -> Result<Self, CraneliftError> {
        let triple = module.ctx.triple;
        match (triple.architecture, triple.vendor, triple.operating_system) {
            (Architecture::Riscv32im, Vendor::Succinct, OperatingSystem::ZkvmElf) => {
                Ok(Self::Riscv32im)
            }
            (Architecture::Riscv64im, Vendor::Succinct, OperatingSystem::ZkvmElf) => {
                Err(CraneliftError::UnsupportedTarget(
                    "SP1 RV64 requires the LP64 soft-float ABI, but Cranelift's current RV64 \
                     backend implements LP64D hard-float"
                        .into(),
                ))
            }
            _ => Err(CraneliftError::UnsupportedTarget(format!(
                "SP1 ELF emission requires riscv32im-succinct-zkvm-elf or \
                 riscv64im-succinct-zkvm-elf, got {triple}"
            ))),
        }
    }

    fn rust_target(self) -> &'static str {
        match self {
            Self::Riscv32im => "riscv32im-succinct-zkvm-elf",
        }
    }

    fn linker_machine(self) -> &'static str {
        match self {
            Self::Riscv32im => "elf32lriscv",
        }
    }

    fn stack_type(self) -> &'static str {
        match self {
            Self::Riscv32im => "u32",
        }
    }

    fn stack_load(self) -> &'static str {
        match self {
            Self::Riscv32im => "lw",
        }
    }
}

impl CraneliftBackend {
    pub fn compile_module_to_sp1_elf(
        &self,
        module: &Module,
    ) -> Result<Sp1ElfArtifact, Vec<CraneliftError>> {
        let sp1_target = Sp1Target::from_module(module).map_err(|e| vec![e])?;
        let native_object = self.compile_module_to_object(module)?;
        let bytes = link_sp1_elf(sp1_target, &native_object.bytes).map_err(|e| vec![e])?;
        Ok(Sp1ElfArtifact {
            bytes,
            func_map: native_object.func_map,
        })
    }
}

fn link_sp1_elf(target: Sp1Target, module_object: &[u8]) -> Result<Vec<u8>, CraneliftError> {
    let temp_dir = create_temp_dir()?;
    let module_object_path = temp_dir.join("module.o");
    let runtime_source_path = temp_dir.join("runtime.rs");
    let runtime_object_path = temp_dir.join("runtime.o");
    let elf_path = temp_dir.join("program.elf");

    fs::write(&module_object_path, module_object)
        .map_err(|e| CraneliftError::Linking(format!("failed to write module object: {e}")))?;
    fs::write(&runtime_source_path, runtime_source(target))
        .map_err(|e| CraneliftError::Linking(format!("failed to write SP1 runtime source: {e}")))?;

    let mut rustc = Command::new("rustc");
    rustc
        .arg("+succinct")
        .arg("--target")
        .arg(target.rust_target())
        .arg("--edition")
        .arg("2024")
        .arg("-C")
        .arg("opt-level=z")
        .arg("-C")
        .arg("panic=abort")
        .arg("--emit=obj")
        .arg("-o")
        .arg(&runtime_object_path)
        .arg(&runtime_source_path);
    run_command(rustc, "failed to compile SP1 runtime object")?;

    let rust_lld = rust_lld_path()?;
    let image_base = format!("--image-base={SP1_STACK_TOP}");
    let mut linker = Command::new(rust_lld);
    linker
        .arg("-flavor")
        .arg("gnu")
        .arg("-m")
        .arg(target.linker_machine())
        .arg(image_base)
        .arg("--entry=_start")
        .arg("-o")
        .arg(&elf_path)
        .arg(&runtime_object_path)
        .arg(&module_object_path);
    run_command(linker, "failed to link SP1 ELF")?;

    fs::read(&elf_path).map_err(|e| CraneliftError::Linking(format!("failed to read SP1 ELF: {e}")))
}

fn create_temp_dir() -> Result<PathBuf, CraneliftError> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| CraneliftError::Linking(format!("system clock is before UNIX_EPOCH: {e}")))?;
    let mut path = env::temp_dir();
    path.push(format!(
        "sonatina-sp1-{}-{}-{}",
        std::process::id(),
        now.as_nanos(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path)
        .map_err(|e| CraneliftError::Linking(format!("failed to create temp dir: {e}")))?;
    Ok(path)
}

fn rust_lld_path() -> Result<PathBuf, CraneliftError> {
    if let Some(path) = env_path("SONATINA_SP1_RUST_LLD") {
        return Ok(path);
    }

    let sysroot = rustc_stdout(&["--print", "sysroot"])?;
    let host = rustc_host()?;
    let candidate = Path::new(&sysroot)
        .join("lib")
        .join("rustlib")
        .join(host)
        .join("bin")
        .join("rust-lld");
    if candidate.exists() {
        return Ok(candidate);
    }

    let rustlib_dir = Path::new(&sysroot).join("lib").join("rustlib");
    let entries = fs::read_dir(&rustlib_dir).map_err(|e| {
        CraneliftError::Toolchain(format!(
            "failed to inspect succinct rustlib directory {}: {e}",
            rustlib_dir.display()
        ))
    })?;
    for entry in entries {
        let path = entry
            .map_err(|e| CraneliftError::Toolchain(format!("failed to read rustlib entry: {e}")))?
            .path()
            .join("bin")
            .join("rust-lld");
        if path.exists() {
            return Ok(path);
        }
    }

    Err(CraneliftError::Toolchain(format!(
        "could not find rust-lld in succinct sysroot {sysroot}; set SONATINA_SP1_RUST_LLD"
    )))
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn rustc_host() -> Result<String, CraneliftError> {
    let output = rustc_stdout(&["-vV"])?;
    output
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .map(str::to_owned)
        .ok_or_else(|| CraneliftError::Toolchain("rustc +succinct -vV did not print host".into()))
}

fn rustc_stdout(args: &[&str]) -> Result<String, CraneliftError> {
    let mut command = Command::new("rustc");
    command.arg("+succinct").args(args);
    let output = run_command(command, "failed to run rustc +succinct")?;
    String::from_utf8(output.stdout)
        .map(|stdout| stdout.trim().to_owned())
        .map_err(|e| CraneliftError::Toolchain(format!("rustc output was not UTF-8: {e}")))
}

fn run_command(mut command: Command, context: &str) -> Result<Output, CraneliftError> {
    let output = command
        .output()
        .map_err(|e| CraneliftError::Toolchain(format!("{context}: {e}")))?;
    if output.status.success() {
        return Ok(output);
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let detail = if stderr.trim().is_empty() {
        stdout.trim()
    } else {
        stderr.trim()
    };
    Err(CraneliftError::Toolchain(format!("{context}: {detail}")))
}

fn runtime_source(target: Sp1Target) -> String {
    format!(
        r##"#![no_std]
#![no_main]

use core::arch::{{asm, global_asm}};
use core::panic::PanicInfo;

const COMMIT: u32 = 0x10;
const COMMIT_DEFERRED_PROOFS: u32 = 0x1a;
const HALT: u32 = 0x00;
const WRITE: u32 = 0x02;
const STACK_TOP: {stack_type} = 0x7800_0000;
const SHA256_EMPTY: [u32; 8] = [
    u32::from_le_bytes([0xe3, 0xb0, 0xc4, 0x42]),
    u32::from_le_bytes([0x98, 0xfc, 0x1c, 0x14]),
    u32::from_le_bytes([0x9a, 0xfb, 0xf4, 0xc8]),
    u32::from_le_bytes([0x99, 0x6f, 0xb9, 0x24]),
    u32::from_le_bytes([0x27, 0xae, 0x41, 0xe4]),
    u32::from_le_bytes([0x64, 0x9b, 0x93, 0x4c]),
    u32::from_le_bytes([0xa4, 0x95, 0x99, 0x1b]),
    u32::from_le_bytes([0x78, 0x52, 0xb8, 0x55]),
];

#[used]
static _STACK_TOP: {stack_type} = STACK_TOP;

global_asm!(
    r#"
    .section .text._start;
    .globl _start;
_start:
    .option push;
    .option norelax;
    la gp, __global_pointer$;
    .option pop;
    la sp, {{stack_top}}
    {stack_load} sp, 0(sp)
    call __start
"#,
    stack_top = sym _STACK_TOP,
);

#[unsafe(no_mangle)]
unsafe extern "C" fn __start() -> ! {{
    unsafe extern "C" {{
        fn main() -> i32;
    }}
    let exit_code = unsafe {{ main() }};
    syscall_halt((exit_code & 0xff) as u8);
}}

#[unsafe(no_mangle)]
pub extern "C" fn syscall_halt(exit_code: u8) -> ! {{
    let mut i = 0usize;
    while i < SHA256_EMPTY.len() {{
        let word = SHA256_EMPTY[i] as usize;
        unsafe {{
            asm!("ecall", in("t0") COMMIT, in("a0") i, in("a1") word);
        }}
        i += 1;
    }}

    let mut i = 0usize;
    while i < 8 {{
        unsafe {{
            asm!("ecall", in("t0") COMMIT_DEFERRED_PROOFS, in("a0") i, in("a1") 0usize);
        }}
        i += 1;
    }}

    unsafe {{
        asm!("ecall", in("t0") HALT, in("a0") exit_code as usize);
    }}
    loop {{
        core::hint::spin_loop();
    }}
}}

#[unsafe(no_mangle)]
pub extern "C" fn syscall_write(fd: u32, ptr: *const u8, len: usize) {{
    unsafe {{
        asm!(
            "ecall",
            in("t0") WRITE,
            in("a0") fd as usize,
            in("a1") ptr as usize,
            in("a2") len,
        );
    }}
}}

#[unsafe(no_mangle)]
pub extern "C" fn sys_write(fd: u32, ptr: *const u8, len: usize) {{
    syscall_write(fd, ptr, len);
}}

#[unsafe(no_mangle)]
pub extern "C" fn sys_panic(ptr: *const u8, len: usize) -> ! {{
    sys_write(2, ptr, len);
    syscall_halt(1);
}}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {{
    let mut offset = 0usize;
    while offset < len {{
        let byte = unsafe {{ src.add(offset).read() }};
        unsafe {{ dst.add(offset).write(byte) }};
        offset += 1;
    }}
    dst
}}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memset(dst: *mut u8, value: i32, len: usize) -> *mut u8 {{
    let mut offset = 0usize;
    while offset < len {{
        unsafe {{ dst.add(offset).write(value as u8) }};
        offset += 1;
    }}
    dst
}}

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {{
    syscall_halt(1)
}}
"##,
        stack_type = target.stack_type(),
        stack_load = target.stack_load(),
    )
}
