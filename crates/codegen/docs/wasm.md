# WebAssembly backend

The optional `sonatina-codegen/wasm` feature compiles `wasm32-unknown-unknown`
modules to WebAssembly binaries through [WAFFLE](https://github.com/bytecodealliance/waffle),
which recovers structured control flow and assigns locals. The output uses
the multi-value, sign-extension and bulk-memory features of WebAssembly 2.0.

Build modules for the [`Wasm32`](sonatina_ir::isa::wasm32::Wasm32) ISA, which
accepts the same target-neutral instruction set as the host-native ISA, and
compile them with `Compile` and [`WasmBackend`]:

```no_run
use sonatina_codegen::{Compile, compile::OptLevel, isa::wasm::WasmBackend};
use sonatina_ir::Module;

fn compile_wasm(module: Module) -> Vec<u8> {
    Compile::new(module, WasmBackend::new())
        .with_opt_level(OptLevel::O2)
        .compile()
        .expect("wasm compilation failed")
        .into_bytes()
}
```

Above O0, WAFFLE's value numbering and constant propagation also run on each
function.

## Values and function ABI

Memory is little-endian with the native scalar and struct layout, except that
pointers and references take four bytes.

| Sonatina type | Wasm values |
| --- | --- |
| `i1`, `i8`, `i16`, `i32` | One `i32`, zero-extended; `i1` is 0 or 1 |
| `i64` | One `i64` |
| `i128`, `i256` | Two or four `i64` limbs, least significant first |
| `*T`, `objref<T>`, `constref<T>` | One `i32` address |
| Arrays and structs | One `i32` address of the value's storage |

Parameters and results use these values in order; multiple results are wasm
multi-value results. An aggregate argument points to storage the caller keeps
unchanged for the call. A single aggregate or `constref` result is copied
into a caller-allocated buffer whose address is a hidden first parameter.
Signatures with an explicit `unit`, or with several results one of which
returns indirectly, are rejected. Exported functions mask narrow integer
parameters to their width.

Public functions are exported under their names and the memory as `memory`.
Private functions are internal. External functions are imported from the
`env` module, except `malloc`, `free` and `memmove`, which the backend
defines (see below). Global variables live in linear memory; external
globals are unsupported.

## Memory

Linear memory holds, in order:

1. 1 KiB that is never allocated, so null and small sentinel addresses are
   distinct from every object;
2. allocator state and global variables, with their initializers as data
   segments;
3. the stack, 1 MiB by default (`WasmBackend::with_stack_size`), which grows
   down from its top;
4. the heap, which grows with `memory.grow`.

The mutable global `__stack_pointer` holds the stack top. A function that
needs stack storage (`alloca`, stack objects, aggregate values and indirect
call results) takes one 16-byte-aligned frame at entry, trapping if the stack
would overflow, and releases it when it returns. A trap leaves the stack
pointer where it was, so instantiate a module again after a trap.

Object references follow the native object legalization: fresh object returns
move to caller storage, and objects exported with `obj.materialize.heap` are
heap allocated.

## Runtime

When a module allocates or declares `malloc`, `free` or `memmove` as external
functions, the backend defines them with their wasm32 C signatures:
`malloc(i32) -> i32`, `free(i32)` and `memmove(i32, i32, i32) -> i32`.
`malloc` takes a power-of-two block of at least 16 bytes and returns a
16-byte-aligned address, or null when memory cannot grow. Freed blocks are
reused for allocations of the same size class. `mem.alloc_dynamic` traps
instead of returning null, and when its size does not fit in 32 bits.

`i128` and `i256` multiplication and division call helper functions defined
once per width.

## Unsupported operations

Enum operations must disappear during legalization. `get_function_ptr`,
`sym_addr` and `sym_size` are unsupported. Division by zero is undefined at
the IR level: scalar divisions trap and wide divisions return an unspecified
value. Signed division of the minimum by -1 wraps.
