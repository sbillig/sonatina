//! Linear-memory layout and the functions the backend defines itself.

use rustc_hash::FxHashMap;
use smallvec::smallvec;
use sonatina_ir::{GlobalVariableRef, module::ModuleCtx};
use waffle::{
    Func, FuncDecl, FunctionBody, Global, GlobalData, Memory, MemorySegment,
    Module as WaffleModule, Operator, SignatureData, Terminator, Type as WType, entity::EntityRef,
};

use super::{
    builder::{Builder, Repr, memarg},
    wide,
};
use crate::isa::native_layout::{
    serialize_initializer, value_storage_alignment, value_storage_size,
};

/// Addresses below this are never allocated, so null and small sentinel
/// addresses stay distinct from every object.
const RESERVED_BYTES: u32 = 1024;
/// Free-list heads for allocations of `2^k` payload bytes.
const SIZE_CLASSES: u32 = 32;
const MIN_CLASS: i32 = 4;
/// Each heap block starts with its size class, padded to keep payloads
/// 16-byte aligned.
const BLOCK_HEADER: i32 = 16;
const MAX_ALLOCATION: i32 = 1 << 30;
const PAGE_BITS: i64 = 16;

/// Static data, then the stack, which grows down, then the heap.
pub(super) struct MemoryLayout {
    free_lists: u32,
    pub(super) globals: FxHashMap<GlobalVariableRef, u32>,
    pub(super) segments: Vec<MemorySegment>,
    pub(super) stack_base: u32,
    pub(super) stack_top: u32,
}

impl MemoryLayout {
    pub(super) fn new(ctx: &ModuleCtx, stack_size: u32) -> Result<Self, String> {
        let free_lists = RESERVED_BYTES;
        let mut end = free_lists + SIZE_CLASSES * 4;
        let mut globals = FxHashMap::default();
        let mut segments = Vec::new();
        ctx.with_gv_store(|store| -> Result<(), String> {
            for global in store.all_gv_refs() {
                let data = store.gv_data(global);
                let name = &data.symbol;
                if data.linkage.is_external() {
                    return Err(format!("wasm backend cannot import global {name}"));
                }
                let align = value_storage_alignment(data.ty, ctx)
                    .map_err(|error| format!("global {name}: {error}"))?
                    as u32;
                let size = value_storage_size(data.ty, ctx)
                    .map_err(|error| format!("global {name}: {error}"))?;
                let address = end.next_multiple_of(align);
                end = address
                    .checked_add(size.max(1))
                    .ok_or("static data exceeds the wasm32 address space")?;
                globals.insert(global, address);
                if let Some(init) = &data.initializer {
                    let mut bytes = vec![0; size as usize];
                    serialize_initializer(init, data.ty, &mut bytes, ctx)
                        .map_err(|error| format!("global {name}: {error}"))?;
                    // Memory starts zeroed.
                    if bytes.iter().any(|&byte| byte != 0) {
                        segments.push(MemorySegment {
                            offset: address as usize,
                            data: bytes,
                        });
                    }
                }
            }
            Ok(())
        })?;
        let stack_base = end.next_multiple_of(16);
        let stack_top = stack_base
            .checked_add(stack_size)
            .filter(|top| *top <= u32::MAX - 15)
            .ok_or("stack exceeds the wasm32 address space")?;
        Ok(Self {
            free_lists,
            globals,
            segments,
            stack_base,
            stack_top,
        })
    }

    pub(super) fn heap_base(&self) -> u32 {
        self.stack_top
    }

    pub(super) fn initial_pages(&self) -> usize {
        (u64::from(self.heap_base()) >> PAGE_BITS) as usize + 1
    }
}

/// The C memory functions the backend provides in place of a host libc.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RuntimeSymbol {
    Malloc,
    Free,
    Memmove,
}

impl RuntimeSymbol {
    pub(super) fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "malloc" => Self::Malloc,
            "free" => Self::Free,
            "memmove" => Self::Memmove,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Malloc => "malloc",
            Self::Free => "free",
            Self::Memmove => "memmove",
        }
    }

    fn signature(self) -> SignatureData {
        let (params, returns) = match self {
            Self::Malloc => (vec![WType::I32], vec![WType::I32]),
            Self::Free => (vec![WType::I32], vec![]),
            Self::Memmove => (vec![WType::I32; 3], vec![WType::I32]),
        };
        SignatureData { params, returns }
    }

    /// External declarations of these symbols must use the wasm32 C ABI.
    pub(super) fn check_signature(self, data: &SignatureData) -> Result<(), String> {
        let expected = self.signature();
        if data.params == expected.params && data.returns == expected.returns {
            Ok(())
        } else {
            Err(format!(
                "external `{}` must have the wasm32 C signature {:?} -> {:?}",
                self.name(),
                expected.params,
                expected.returns
            ))
        }
    }
}

/// A size-class allocator over the heap and `memmove`.
///
/// Each allocation takes a power-of-two block of at least 16 payload bytes.
/// Freed blocks go on their class's free list and are reused before the heap
/// grows. Allocations are 16-byte aligned and uninitialized; failure returns
/// null.
pub(super) struct Runtime {
    malloc: Func,
    free: Func,
    memmove: Func,
    heap_end: Global,
}

impl Runtime {
    pub(super) fn declare(module: &mut WaffleModule<'static>, layout: &MemoryLayout) -> Self {
        let heap_end = module.globals.push(GlobalData {
            ty: WType::I32,
            value: Some(u64::from(layout.heap_base())),
            mutable: true,
        });
        let mut declare = |symbol: RuntimeSymbol| {
            let signature = module.signatures.push(symbol.signature());
            module.funcs.push(FuncDecl::Body(
                signature,
                format!("__sonatina_{}", symbol.name()),
                FunctionBody::default(),
            ))
        };
        Self {
            malloc: declare(RuntimeSymbol::Malloc),
            free: declare(RuntimeSymbol::Free),
            memmove: declare(RuntimeSymbol::Memmove),
            heap_end,
        }
    }

    pub(super) fn func(&self, symbol: RuntimeSymbol) -> Func {
        match symbol {
            RuntimeSymbol::Malloc => self.malloc,
            RuntimeSymbol::Free => self.free,
            RuntimeSymbol::Memmove => self.memmove,
        }
    }

    pub(super) fn define(&self, module: &mut WaffleModule<'static>, layout: &MemoryLayout) {
        let bodies = [
            (self.malloc, self.malloc_body(module, layout)),
            (self.free, free_body(module, self.free, layout)),
            (self.memmove, memmove_body(module, self.memmove)),
        ];
        for (func, body) in bodies {
            let FuncDecl::Body(_, _, slot) = &mut module.funcs[func] else {
                unreachable!("runtime functions are declared with bodies");
            };
            *slot = body;
        }
    }

    fn malloc_body(&self, module: &WaffleModule, layout: &MemoryLayout) -> FunctionBody {
        let mut b = Builder::new(module, module.funcs[self.malloc].sig());
        let size = b.params()[0];
        let fail = b.add_block();
        let sized = b.add_block();
        let max = b.i32(MAX_ALLOCATION);
        let too_big = b.op(Operator::I32GtU, &[size, max], WType::I32);
        b.branch(too_big, fail, sized);

        b.switch_to(fail);
        let null = b.i32(0);
        b.terminate(Terminator::Return { values: vec![null] });

        // The class of a size is the bit length of `max(size, 16) - 1`.
        b.switch_to(sized);
        let min = b.i32(1 << MIN_CLASS);
        let small = b.op(Operator::I32LtU, &[size, min], WType::I32);
        let size = b.select(small, min, size, WType::I32);
        let one = b.i32(1);
        let below = b.op(Operator::I32Sub, &[size, one], WType::I32);
        let leading = b.op(Operator::I32Clz, &[below], WType::I32);
        let thirty_two = b.i32(32);
        let class = b.op(Operator::I32Sub, &[thirty_two, leading], WType::I32);
        let head = free_list_head(&mut b, layout, class);
        let first = b.op(
            Operator::I32Load {
                memory: memarg(2, 0),
            },
            &[head],
            WType::I32,
        );
        let reuse = b.add_block();
        let fresh = b.add_block();
        b.branch(first, reuse, fresh);

        b.switch_to(reuse);
        let next = b.op(
            Operator::I32Load {
                memory: memarg(2, 0),
            },
            &[first],
            WType::I32,
        );
        b.effect(
            Operator::I32Store {
                memory: memarg(2, 0),
            },
            &[head, next],
        );
        b.terminate(Terminator::Return {
            values: vec![first],
        });

        b.switch_to(fresh);
        let block = b.op(
            Operator::GlobalGet {
                global_index: self.heap_end,
            },
            &[],
            WType::I32,
        );
        let payload = b.op(Operator::I32Shl, &[one, class], WType::I32);
        let start = b.op(Operator::I64ExtendI32U, &[block], WType::I64);
        let payload = b.op(Operator::I64ExtendI32U, &[payload], WType::I64);
        let header = b.i64(i64::from(BLOCK_HEADER));
        let end = b.op(Operator::I64Add, &[start, header], WType::I64);
        let end = b.op(Operator::I64Add, &[end, payload], WType::I64);
        let pages = b.op(
            Operator::MemorySize {
                mem: Memory::new(0),
            },
            &[],
            WType::I32,
        );
        let pages = b.op(Operator::I64ExtendI32U, &[pages], WType::I64);
        let page_bits = b.i64(PAGE_BITS);
        let capacity = b.op(Operator::I64Shl, &[pages, page_bits], WType::I64);
        let short = b.op(Operator::I64GtU, &[end, capacity], WType::I32);
        let grow = b.add_block();
        let ready = b.add_block();
        b.branch(short, grow, ready);

        b.switch_to(grow);
        let limit = b.i64(1 << 32);
        let beyond = b.op(Operator::I64GtU, &[end, limit], WType::I32);
        let try_grow = b.add_block();
        b.branch(beyond, fail, try_grow);
        b.switch_to(try_grow);
        let missing = b.op(Operator::I64Sub, &[end, capacity], WType::I64);
        let round = b.i64((1 << PAGE_BITS) - 1);
        let missing = b.op(Operator::I64Add, &[missing, round], WType::I64);
        let missing = b.op(Operator::I64ShrU, &[missing, page_bits], WType::I64);
        let missing = b.op(Operator::I32WrapI64, &[missing], WType::I32);
        let previous = b.op(
            Operator::MemoryGrow {
                mem: Memory::new(0),
            },
            &[missing],
            WType::I32,
        );
        let minus_one = b.i32(-1);
        let failed = b.op(Operator::I32Eq, &[previous, minus_one], WType::I32);
        b.branch(failed, fail, ready);

        b.switch_to(ready);
        let end = b.op(Operator::I32WrapI64, &[end], WType::I32);
        b.effect(
            Operator::GlobalSet {
                global_index: self.heap_end,
            },
            &[end],
        );
        b.effect(
            Operator::I32Store {
                memory: memarg(2, 0),
            },
            &[block, class],
        );
        let header = b.i32(BLOCK_HEADER);
        let address = b.op(Operator::I32Add, &[block, header], WType::I32);
        b.terminate(Terminator::Return {
            values: vec![address],
        });
        b.body
    }
}

fn free_list_head(b: &mut Builder, layout: &MemoryLayout, class: waffle::Value) -> waffle::Value {
    let two = b.i32(2);
    let offset = b.op(Operator::I32Shl, &[class, two], WType::I32);
    let base = b.i32(layout.free_lists as i32);
    b.op(Operator::I32Add, &[base, offset], WType::I32)
}

fn free_body(module: &WaffleModule, func: Func, layout: &MemoryLayout) -> FunctionBody {
    let mut b = Builder::new(module, module.funcs[func].sig());
    let address = b.params()[0];
    let release = b.add_block();
    let done = b.add_block();
    b.branch(address, release, done);
    b.body
        .set_terminator(done, Terminator::Return { values: vec![] });

    b.switch_to(release);
    let header = b.i32(BLOCK_HEADER);
    let block = b.op(Operator::I32Sub, &[address, header], WType::I32);
    let class = b.op(
        Operator::I32Load {
            memory: memarg(2, 0),
        },
        &[block],
        WType::I32,
    );
    let head = free_list_head(&mut b, layout, class);
    let first = b.op(
        Operator::I32Load {
            memory: memarg(2, 0),
        },
        &[head],
        WType::I32,
    );
    b.effect(
        Operator::I32Store {
            memory: memarg(2, 0),
        },
        &[address, first],
    );
    b.effect(
        Operator::I32Store {
            memory: memarg(2, 0),
        },
        &[head, address],
    );
    b.terminate(Terminator::Return { values: vec![] });
    b.body
}

fn memmove_body(module: &WaffleModule, func: Func) -> FunctionBody {
    let mut b = Builder::new(module, module.funcs[func].sig());
    let [dest, source, len] = b.params()[..] else {
        unreachable!("memmove takes three parameters");
    };
    b.effect(
        Operator::MemoryCopy {
            dst_mem: Memory::new(0),
            src_mem: Memory::new(0),
        },
        &[dest, source, len],
    );
    b.terminate(Terminator::Return { values: vec![dest] });
    b.body
}

/// Wide-integer operations defined as functions rather than inline.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Helper {
    /// The low half of a product.
    MulLow,
    /// The double-width product.
    MulFull,
    /// The unsigned quotient, then the remainder.
    DivMod,
}

impl Helper {
    pub(super) fn name(self, limbs: usize) -> String {
        let op = match self {
            Self::MulLow => "mul",
            Self::MulFull => "mul_full",
            Self::DivMod => "udivmod",
        };
        format!("__sonatina_{op}{}", limbs * 64)
    }

    fn results(self, limbs: usize) -> usize {
        match self {
            Self::MulLow => limbs,
            Self::MulFull | Self::DivMod => limbs * 2,
        }
    }
}

pub(super) fn helper_signature(helper: Helper, limbs: usize) -> SignatureData {
    SignatureData {
        params: vec![WType::I64; limbs * 2],
        returns: vec![WType::I64; helper.results(limbs)],
    }
}

pub(super) fn define_helper(
    module: &mut WaffleModule<'static>,
    func: Func,
    helper: Helper,
    limbs: usize,
) {
    let mut b = Builder::new(module, module.funcs[func].sig());
    let params = b.params();
    let (lhs, rhs) = params.split_at(limbs);
    let values = match helper {
        Helper::MulLow => wide::mul(&mut b, lhs, rhs, limbs).into_vec(),
        Helper::MulFull => wide::mul(&mut b, lhs, rhs, limbs * 2).into_vec(),
        Helper::DivMod => {
            let (quotient, remainder) = udivmod(&mut b, lhs, rhs);
            quotient.into_iter().chain(remainder).collect()
        }
    };
    b.terminate(Terminator::Return { values });
    let FuncDecl::Body(_, _, slot) = &mut module.funcs[func] else {
        unreachable!("helpers are declared with bodies");
    };
    *slot = b.body;
}

/// Restoring division, one numerator bit at a time from its highest set bit.
/// Leaves `b` in the block where both results are defined.
fn udivmod(
    b: &mut Builder,
    numerator: &[waffle::Value],
    denominator: &[waffle::Value],
) -> (Repr, Repr) {
    let limbs = numerator.len();
    let exit = b.add_block();
    let results: Vec<_> = (0..limbs * 2)
        .map(|_| b.body.add_blockparam(exit, WType::I64))
        .collect();
    let zero = b.i64(0);
    let zeros = vec![zero; limbs];

    // A smaller numerator is the remainder.
    let smaller = wide::lt(b, numerator, denominator, false);
    let check = b.add_block();
    b.terminate(Terminator::CondBr {
        cond: smaller,
        if_true: waffle::BlockTarget {
            block: exit,
            args: zeros.iter().chain(numerator).copied().collect(),
        },
        if_false: waffle::BlockTarget {
            block: check,
            args: vec![],
        },
    });

    // Divide single-limb values directly. This includes a zero numerator
    // whose denominator is zero, which would otherwise have no highest bit.
    b.switch_to(check);
    let high = numerator[1..]
        .iter()
        .chain(&denominator[1..])
        .copied()
        .collect::<Vec<_>>();
    let fits = wide::is_zero(b, &high);
    let direct = b.add_block();
    let start = b.add_block();
    b.branch(fits, direct, start);

    b.switch_to(direct);
    let quotient = b.op(
        Operator::I64DivU,
        &[numerator[0], denominator[0]],
        WType::I64,
    );
    let remainder = b.op(
        Operator::I64RemU,
        &[numerator[0], denominator[0]],
        WType::I64,
    );
    let mut args = vec![quotient];
    args.extend(&zeros[1..]);
    args.push(remainder);
    args.extend(&zeros[1..]);
    b.jump(exit, args);

    // Start at the numerator's highest set bit.
    b.switch_to(start);
    let mut leading = b.i64((limbs * 64) as i64);
    for (index, &limb) in numerator.iter().enumerate() {
        let zeros_above = b.i64(((limbs - 1 - index) * 64) as i64);
        let limb_leading = b.op(Operator::I64Clz, &[limb], WType::I64);
        let total = b.op(Operator::I64Add, &[zeros_above, limb_leading], WType::I64);
        let set = b.op(Operator::I64Ne, &[limb, zero], WType::I32);
        leading = b.select(set, total, leading, WType::I64);
    }
    let top = b.i64((limbs * 64 - 1) as i64);
    let bit = b.op(Operator::I64Sub, &[top, leading], WType::I64);
    let header = b.add_block();
    let bit_param = b.body.add_blockparam(header, WType::I64);
    let state: Vec<_> = (0..limbs * 2)
        .map(|_| b.body.add_blockparam(header, WType::I64))
        .collect();
    let mut args = vec![bit];
    args.extend(zeros.iter().chain(&zeros));
    b.jump(header, args);

    b.switch_to(header);
    let (quotient, remainder) = state.split_at(limbs);
    let six = b.i64(6);
    let limb_index = b.op(Operator::I64ShrU, &[bit_param, six], WType::I64);
    let mut limb = numerator[0];
    for (index, &candidate) in numerator.iter().enumerate().skip(1) {
        let index = b.i64(index as i64);
        let matches = b.op(Operator::I64Eq, &[limb_index, index], WType::I32);
        limb = b.select(matches, candidate, limb, WType::I64);
    }
    let sixty_three = b.i64(63);
    let shift = b.op(Operator::I64And, &[bit_param, sixty_three], WType::I64);
    let shifted = b.op(Operator::I64ShrU, &[limb, shift], WType::I64);
    let one = b.i64(1);
    let next = b.op(Operator::I64And, &[shifted, one], WType::I64);
    // The shifted remainder fits: it is at most the numerator prefix so far.
    let remainder = shift_in(b, remainder, next);
    let less = wide::lt(b, &remainder, denominator, false);
    let subtract = b.op(Operator::I32Eqz, &[less], WType::I32);
    let (difference, _) = wide::sub(b, &remainder, denominator);
    let remainder = wide::select(b, subtract, &difference, &remainder);
    let subtracted = b.op(Operator::I64ExtendI32U, &[subtract], WType::I64);
    let quotient = shift_in(b, quotient, subtracted);
    let done = b.op(Operator::I64Eqz, &[bit_param], WType::I32);
    let next_bit = b.op(Operator::I64Sub, &[bit_param, one], WType::I64);
    let mut loop_args = vec![next_bit];
    loop_args.extend(quotient.iter().chain(&remainder));
    b.terminate(Terminator::CondBr {
        cond: done,
        if_true: waffle::BlockTarget {
            block: exit,
            args: quotient.iter().chain(&remainder).copied().collect(),
        },
        if_false: waffle::BlockTarget {
            block: header,
            args: loop_args,
        },
    });

    b.switch_to(exit);
    let (quotient, remainder) = results.split_at(limbs);
    (
        quotient.iter().copied().collect(),
        remainder.iter().copied().collect(),
    )
}

/// Shifts left by one bit, shifting `bit` (0 or 1) in at the bottom.
fn shift_in(b: &mut Builder, value: &[waffle::Value], bit: waffle::Value) -> Repr {
    let one = b.i64(1);
    let sixty_three = b.i64(63);
    let mut carry = bit;
    let mut result = smallvec![];
    for &limb in value {
        let shifted = b.op(Operator::I64Shl, &[limb, one], WType::I64);
        result.push(b.op(Operator::I64Or, &[shifted, carry], WType::I64));
        carry = b.op(Operator::I64ShrU, &[limb, sixty_three], WType::I64);
    }
    result
}
