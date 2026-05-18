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
    Riscv64im,
}

impl Sp1Target {
    fn from_module(module: &Module) -> Result<Self, CraneliftError> {
        let triple = module.ctx.triple;
        match (triple.architecture, triple.vendor, triple.operating_system) {
            (Architecture::Riscv32im, Vendor::Succinct, OperatingSystem::ZkvmElf) => {
                Ok(Self::Riscv32im)
            }
            (Architecture::Riscv64im, Vendor::Succinct, OperatingSystem::ZkvmElf) => {
                Ok(Self::Riscv64im)
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
            Self::Riscv64im => "riscv64im-succinct-zkvm-elf",
        }
    }

    fn linker_machine(self) -> &'static str {
        match self {
            Self::Riscv32im => "elf32lriscv",
            Self::Riscv64im => "elf64lriscv",
        }
    }

    fn stack_type(self) -> &'static str {
        match self {
            Self::Riscv32im => "u32",
            Self::Riscv64im => "u64",
        }
    }

    fn stack_load(self) -> &'static str {
        match self {
            Self::Riscv32im => "lw",
            Self::Riscv64im => "ld",
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
    r##"#![no_std]
#![no_main]

use core::arch::{asm, global_asm};
use core::panic::PanicInfo;

const COMMIT: u32 = 0x10;
const COMMIT_DEFERRED_PROOFS: u32 = 0x1a;
const FD_PUBLIC_VALUES: u32 = 13;
const HALT: u32 = 0x00;
const HINT_LEN: u32 = 0xf0;
const HINT_READ: u32 = 0xf1;
const WRITE: u32 = 0x02;
const STACK_TOP: __STACK_TYPE__ = 0x7800_0000;

#[used]
static _STACK_TOP: __STACK_TYPE__ = STACK_TOP;

static mut PUBLIC_VALUES_HASHER: Sha256 = Sha256::new();

global_asm!(
    r#"
    .section .text._start;
    .globl _start;
_start:
    .option push;
    .option norelax;
    la gp, __global_pointer$;
    .option pop;
    la sp, {stack_top}
    __STACK_LOAD__ sp, 0(sp)
    call __start
"#,
    stack_top = sym _STACK_TOP,
);

#[unsafe(no_mangle)]
unsafe extern "C" fn __start() -> ! {
    unsafe extern "C" {
        fn main() -> i32;
    }
    let exit_code = unsafe { main() };
    syscall_halt((exit_code & 0xff) as u8);
}

#[unsafe(no_mangle)]
pub extern "C" fn syscall_halt(exit_code: u8) -> ! {
    let digest = unsafe { (*core::ptr::addr_of_mut!(PUBLIC_VALUES_HASHER)).finalize() };
    let mut i = 0usize;
    while i < 8 {
        let word = unsafe { *digest.as_ptr().add(i) }.swap_bytes() as usize;
        unsafe {
            asm!("ecall", in("t0") COMMIT, in("a0") i, in("a1") word);
        }
        i += 1;
    }

    let mut i = 0usize;
    while i < 8 {
        unsafe {
            asm!("ecall", in("t0") COMMIT_DEFERRED_PROOFS, in("a0") i, in("a1") 0usize);
        }
        i += 1;
    }

    unsafe {
        asm!("ecall", in("t0") HALT, in("a0") exit_code as usize);
    }
    loop {
        core::hint::spin_loop();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn syscall_write(fd: u32, ptr: *const u8, len: usize) {
    unsafe {
        asm!(
            "ecall",
            in("t0") WRITE,
            in("a0") fd as usize,
            in("a1") ptr as usize,
            in("a2") len,
        );
    }
    if fd == FD_PUBLIC_VALUES {
        unsafe {
            (*core::ptr::addr_of_mut!(PUBLIC_VALUES_HASHER)).update(ptr, len);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn syscall_hint_len() -> usize {
    let len: usize;
    unsafe {
        asm!("ecall", in("t0") HINT_LEN, lateout("t0") len);
    }
    len
}

#[unsafe(no_mangle)]
pub extern "C" fn syscall_hint_read(ptr: *mut u8, len: usize) {
    unsafe {
        asm!("ecall", in("t0") HINT_READ, in("a0") ptr as usize, in("a1") len);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_write(fd: u32, ptr: *const u8, len: usize) {
    syscall_write(fd, ptr, len);
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_panic(ptr: *const u8, len: usize) -> ! {
    sys_write(2, ptr, len);
    syscall_halt(1);
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_sp1_read_u32() -> u32 {
    let buffer = read_hint_buffer(4);
    unsafe {
        u32::from_le_bytes([
            *buffer.bytes.as_ptr().add(0),
            *buffer.bytes.as_ptr().add(1),
            *buffer.bytes.as_ptr().add(2),
            *buffer.bytes.as_ptr().add(3),
        ])
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_sp1_read_i32() -> i32 {
    let buffer = read_hint_buffer(4);
    unsafe {
        i32::from_le_bytes([
            *buffer.bytes.as_ptr().add(0),
            *buffer.bytes.as_ptr().add(1),
            *buffer.bytes.as_ptr().add(2),
            *buffer.bytes.as_ptr().add(3),
        ])
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_sp1_read_u64() -> u64 {
    let buffer = read_hint_buffer(8);
    unsafe {
        u64::from_le_bytes([
            *buffer.bytes.as_ptr().add(0),
            *buffer.bytes.as_ptr().add(1),
            *buffer.bytes.as_ptr().add(2),
            *buffer.bytes.as_ptr().add(3),
            *buffer.bytes.as_ptr().add(4),
            *buffer.bytes.as_ptr().add(5),
            *buffer.bytes.as_ptr().add(6),
            *buffer.bytes.as_ptr().add(7),
        ])
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_sp1_commit_u32(value: u32) {
    let bytes = value.to_le_bytes();
    syscall_write(FD_PUBLIC_VALUES, bytes.as_ptr(), bytes.len());
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_sp1_commit_i32(value: i32) {
    let bytes = value.to_le_bytes();
    syscall_write(FD_PUBLIC_VALUES, bytes.as_ptr(), bytes.len());
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_sp1_commit_u64(value: u64) {
    let bytes = value.to_le_bytes();
    syscall_write(FD_PUBLIC_VALUES, bytes.as_ptr(), bytes.len());
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_sp1_write_stdout_u8(value: u8) {
    syscall_write(1, &value as *const u8, 1);
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_sp1_write_stderr_u8(value: u8) {
    syscall_write(2, &value as *const u8, 1);
}

#[unsafe(no_mangle)]
pub extern "C" fn sys_sp1_halt_invalid_hint() -> ! {
    syscall_halt(3);
}

#[repr(align(8))]
struct AlignedHintBuffer {
    bytes: [u8; 8],
}

fn read_hint_buffer(len: usize) -> AlignedHintBuffer {
    if syscall_hint_len() != len {
        syscall_halt(3);
    }
    let mut buffer = AlignedHintBuffer { bytes: [0; 8] };
    syscall_hint_read(buffer.bytes.as_mut_ptr(), len);
    buffer
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {
    let mut offset = 0usize;
    while offset < len {
        let byte = unsafe { src.add(offset).read() };
        unsafe { dst.add(offset).write(byte) };
        offset += 1;
    }
    dst
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn memset(dst: *mut u8, value: i32, len: usize) -> *mut u8 {
    let mut offset = 0usize;
    while offset < len {
        unsafe { dst.add(offset).write(value as u8) };
        offset += 1;
    }
    dst
}

struct Sha256 {
    state: [u32; 8],
    len_bytes: u64,
    buffer: [u8; 64],
    buffer_len: usize,
}

impl Sha256 {
    const fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            len_bytes: 0,
            buffer: [0; 64],
            buffer_len: 0,
        }
    }

    fn update(&mut self, ptr: *const u8, len: usize) {
        let mut offset = 0usize;
        while offset < len {
            let remaining = 64 - self.buffer_len;
            let take = if len - offset < remaining { len - offset } else { remaining };
            let mut i = 0usize;
            while i < take {
                unsafe {
                    self.buffer
                        .as_mut_ptr()
                        .add(self.buffer_len + i)
                        .write(ptr.add(offset + i).read());
                }
                i += 1;
            }
            self.buffer_len += take;
            offset += take;
            if self.buffer_len == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.len_bytes += 64;
                self.buffer_len = 0;
            }
        }
    }

    fn finalize(&mut self) -> [u32; 8] {
        let total_bits = (self.len_bytes + self.buffer_len as u64) * 8;
        unsafe {
            self.buffer.as_mut_ptr().add(self.buffer_len).write(0x80);
        }
        self.buffer_len += 1;

        if self.buffer_len > 56 {
            while self.buffer_len < 64 {
                unsafe {
                    self.buffer.as_mut_ptr().add(self.buffer_len).write(0);
                }
                self.buffer_len += 1;
            }
            let block = self.buffer;
            self.compress(&block);
            self.buffer = [0; 64];
            self.buffer_len = 0;
        }

        while self.buffer_len < 56 {
            unsafe {
                self.buffer.as_mut_ptr().add(self.buffer_len).write(0);
            }
            self.buffer_len += 1;
        }

        let len_bytes = total_bits.to_be_bytes();
        let mut i = 0usize;
        while i < 8 {
            unsafe {
                self.buffer
                    .as_mut_ptr()
                    .add(56 + i)
                    .write(*len_bytes.as_ptr().add(i));
            }
            i += 1;
        }
        let block = self.buffer;
        self.compress(&block);
        self.state
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];

        let mut w = [0u32; 64];
        let mut i = 0usize;
        while i < 16 {
            let j = i * 4;
            unsafe {
                w.as_mut_ptr().add(i).write(u32::from_be_bytes([
                    *block.as_ptr().add(j),
                    *block.as_ptr().add(j + 1),
                    *block.as_ptr().add(j + 2),
                    *block.as_ptr().add(j + 3),
                ]));
            }
            i += 1;
        }
        while i < 64 {
            unsafe {
                let wm15 = *w.as_ptr().add(i - 15);
                let wm2 = *w.as_ptr().add(i - 2);
                let s0 = wm15.rotate_right(7) ^ wm15.rotate_right(18) ^ (wm15 >> 3);
                let s1 = wm2.rotate_right(17) ^ wm2.rotate_right(19) ^ (wm2 >> 10);
                let value = (*w.as_ptr().add(i - 16))
                    .wrapping_add(s0)
                    .wrapping_add(*w.as_ptr().add(i - 7))
                    .wrapping_add(s1);
                w.as_mut_ptr().add(i).write(value);
            }
            i += 1;
        }

        let mut a = unsafe { *self.state.as_ptr().add(0) };
        let mut b = unsafe { *self.state.as_ptr().add(1) };
        let mut c = unsafe { *self.state.as_ptr().add(2) };
        let mut d = unsafe { *self.state.as_ptr().add(3) };
        let mut e = unsafe { *self.state.as_ptr().add(4) };
        let mut f = unsafe { *self.state.as_ptr().add(5) };
        let mut g = unsafe { *self.state.as_ptr().add(6) };
        let mut h = unsafe { *self.state.as_ptr().add(7) };

        i = 0;
        while i < 64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(unsafe { *K.as_ptr().add(i) })
                .wrapping_add(unsafe { *w.as_ptr().add(i) });
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
            i += 1;
        }

        unsafe {
            let state = self.state.as_mut_ptr();
            state.add(0).write((*state.add(0)).wrapping_add(a));
            state.add(1).write((*state.add(1)).wrapping_add(b));
            state.add(2).write((*state.add(2)).wrapping_add(c));
            state.add(3).write((*state.add(3)).wrapping_add(d));
            state.add(4).write((*state.add(4)).wrapping_add(e));
            state.add(5).write((*state.add(5)).wrapping_add(f));
            state.add(6).write((*state.add(6)).wrapping_add(g));
            state.add(7).write((*state.add(7)).wrapping_add(h));
        }
    }
}

#[panic_handler]
fn panic(_: &PanicInfo<'_>) -> ! {
    syscall_halt(1)
}
"##
    .replace("__STACK_TYPE__", target.stack_type())
    .replace("__STACK_LOAD__", target.stack_load())
}
