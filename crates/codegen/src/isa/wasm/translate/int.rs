use smallvec::smallvec;
use sonatina_ir::{InstId, Type, ValueId, types::CompoundType};
use waffle::{Func, FuncDecl, FunctionBody, Operator, Type as WType, Value as WValue};

use super::FunctionTranslator;
use crate::isa::{
    native_layout::value_storage_size,
    wasm::{
        abi::{IntRepr, int_repr, is_indirect, value_types},
        builder::Repr,
        runtime::{Helper, helper_signature},
        wide::{self, ShiftKind},
    },
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Arith {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CmpKind {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum OverflowOp {
    Uadd,
    Sadd,
    Usub,
    Ssub,
    Umul,
    Smul,
}

impl OverflowOp {
    fn signed(self) -> bool {
        matches!(self, Self::Sadd | Self::Ssub | Self::Smul)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SaturatingOp {
    Uadd,
    Sadd,
    Usub,
    Ssub,
    Umul,
    Smul,
}

impl SaturatingOp {
    fn overflow_op(self) -> OverflowOp {
        match self {
            Self::Uadd => OverflowOp::Uadd,
            Self::Sadd => OverflowOp::Sadd,
            Self::Usub => OverflowOp::Usub,
            Self::Ssub => OverflowOp::Ssub,
            Self::Umul => OverflowOp::Umul,
            Self::Smul => OverflowOp::Smul,
        }
    }
}

fn narrow_mask(bits: u32) -> i32 {
    if bits >= 32 {
        -1
    } else {
        ((1u32 << bits) - 1) as i32
    }
}

impl FunctionTranslator<'_> {
    /// Clears the bits of an `i32` carrier above a narrow value.
    pub(super) fn canonical(&mut self, value: WValue, bits: u32) -> WValue {
        if bits >= 32 {
            return value;
        }
        let mask = self.b.i32(narrow_mask(bits));
        self.b.op(Operator::I32And, &[value, mask], WType::I32)
    }

    /// A narrow value sign-extended to its `i32` carrier.
    pub(super) fn signed32(&mut self, value: WValue, bits: u32) -> WValue {
        match bits {
            1 => {
                let zero = self.b.i32(0);
                self.b.op(Operator::I32Sub, &[zero, value], WType::I32)
            }
            8 => self.b.op(Operator::I32Extend8S, &[value], WType::I32),
            16 => self.b.op(Operator::I32Extend16S, &[value], WType::I32),
            _ => value,
        }
    }

    fn extend64(&mut self, value: WValue, bits: u32, signed: bool) -> WValue {
        if signed {
            let value = self.signed32(value, bits);
            self.b.op(Operator::I64ExtendI32S, &[value], WType::I64)
        } else {
            self.b.op(Operator::I64ExtendI32U, &[value], WType::I64)
        }
    }

    fn call_helper(
        &mut self,
        helper: Helper,
        limbs: usize,
        args: &[WValue],
        results: usize,
    ) -> Repr {
        let wmod = &mut *self.wmod;
        let func: Func = *self
            .helpers
            .funcs
            .entry((helper, limbs))
            .or_insert_with(|| {
                let signature = wmod.signatures.push(helper_signature(helper, limbs));
                wmod.funcs.push(FuncDecl::Body(
                    signature,
                    helper.name(limbs),
                    FunctionBody::default(),
                ))
            });
        self.b.results(
            Operator::Call {
                function_index: func,
            },
            args,
            &vec![WType::I64; results],
        )
    }

    /// The integer behind a comparison operand. Like the native backend, an
    /// object reference to an integer compares as its referent.
    fn int_operand(&mut self, value: ValueId) -> Result<(IntRepr, Repr), String> {
        let ty = self.ty(value);
        if let Some(CompoundType::ObjRef(elem)) = ty.resolve_compound(self.ctx)
            && let Some(repr) = int_repr(self.ctx, elem)
        {
            let address = self.resolve1(value)?;
            return Ok((repr, self.load(elem, address, 0, true)?));
        }
        Ok((self.int_repr(value)?, self.resolve(value)?))
    }

    fn define_results(&mut self, inst: InstId, reprs: impl IntoIterator<Item = Repr>) {
        let results = self.function.dfg.inst_results(inst).to_vec();
        for (result, repr) in results.into_iter().zip(reprs) {
            self.values.insert(result, repr);
        }
    }

    pub(super) fn arith(
        &mut self,
        inst: InstId,
        op: Arith,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<(), String> {
        let repr = self.int_repr(lhs)?;
        let lhs = self.resolve(lhs)?;
        let rhs = self.resolve(rhs)?;
        let result = match repr {
            IntRepr::Narrow(bits) => {
                let op32 = match op {
                    Arith::Add => Operator::I32Add,
                    Arith::Sub => Operator::I32Sub,
                    Arith::Mul => Operator::I32Mul,
                    Arith::And => Operator::I32And,
                    Arith::Or => Operator::I32Or,
                    Arith::Xor => Operator::I32Xor,
                };
                let raw = self.b.op(op32, &[lhs[0], rhs[0]], WType::I32);
                smallvec![self.canonical(raw, bits)]
            }
            IntRepr::I64 => {
                let op64 = match op {
                    Arith::Add => Operator::I64Add,
                    Arith::Sub => Operator::I64Sub,
                    Arith::Mul => Operator::I64Mul,
                    Arith::And => Operator::I64And,
                    Arith::Or => Operator::I64Or,
                    Arith::Xor => Operator::I64Xor,
                };
                smallvec![self.b.op(op64, &[lhs[0], rhs[0]], WType::I64)]
            }
            IntRepr::Wide(limbs) => match op {
                Arith::Add => wide::add(&mut self.b, &lhs, &rhs).0,
                Arith::Sub => wide::sub(&mut self.b, &lhs, &rhs).0,
                Arith::Mul => {
                    let args: Vec<_> = lhs.iter().chain(&rhs).copied().collect();
                    self.call_helper(Helper::MulLow, limbs, &args, limbs)
                }
                Arith::And => wide::bitwise(&mut self.b, Operator::I64And, &lhs, &rhs),
                Arith::Or => wide::bitwise(&mut self.b, Operator::I64Or, &lhs, &rhs),
                Arith::Xor => wide::bitwise(&mut self.b, Operator::I64Xor, &lhs, &rhs),
            },
        };
        self.define(inst, result);
        Ok(())
    }

    pub(super) fn div_rem(
        &mut self,
        inst: InstId,
        signed: bool,
        rem: bool,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<(), String> {
        let repr = self.int_repr(lhs)?;
        let lhs = self.resolve(lhs)?;
        let rhs = self.resolve(rhs)?;
        let result = match repr {
            IntRepr::Narrow(bits) => {
                let (lhs, rhs) = if signed {
                    (self.signed32(lhs[0], bits), self.signed32(rhs[0], bits))
                } else {
                    (lhs[0], rhs[0])
                };
                let value = self.scalar_div_rem(WType::I32, signed, rem, lhs, rhs);
                smallvec![self.canonical(value, bits)]
            }
            IntRepr::I64 => smallvec![self.scalar_div_rem(WType::I64, signed, rem, lhs[0], rhs[0])],
            IntRepr::Wide(limbs) if signed => {
                let lhs_negative = wide::sign(&mut self.b, &lhs);
                let rhs_negative = wide::sign(&mut self.b, &rhs);
                let mut args = wide::abs(&mut self.b, &lhs);
                args.extend(wide::abs(&mut self.b, &rhs));
                let output = self.call_helper(Helper::DivMod, limbs, &args, limbs * 2);
                let (value, negative) = if rem {
                    // A remainder takes the dividend's sign.
                    (&output[limbs..], lhs_negative)
                } else {
                    let negative =
                        self.b
                            .op(Operator::I32Xor, &[lhs_negative, rhs_negative], WType::I32);
                    (&output[..limbs], negative)
                };
                let negated = wide::neg(&mut self.b, value);
                wide::select(&mut self.b, negative, &negated, value)
            }
            IntRepr::Wide(limbs) => {
                let args: Vec<_> = lhs.iter().chain(&rhs).copied().collect();
                let output = self.call_helper(Helper::DivMod, limbs, &args, limbs * 2);
                if rem {
                    output[limbs..].iter().copied().collect()
                } else {
                    output[..limbs].iter().copied().collect()
                }
            }
        };
        self.define(inst, result);
        Ok(())
    }

    fn scalar_div_rem(
        &mut self,
        ty: WType,
        signed: bool,
        rem: bool,
        lhs: WValue,
        rhs: WValue,
    ) -> WValue {
        let wide = ty == WType::I64;
        let op = match (signed, rem, wide) {
            (false, false, false) => Operator::I32DivU,
            (false, true, false) => Operator::I32RemU,
            (true, true, false) => Operator::I32RemS,
            (true, false, false) => Operator::I32DivS,
            (false, false, true) => Operator::I64DivU,
            (false, true, true) => Operator::I64RemU,
            (true, true, true) => Operator::I64RemS,
            (true, false, true) => Operator::I64DivS,
        };
        if !signed || rem {
            return self.b.op(op, &[lhs, rhs], ty);
        }
        // Dividing the minimum by -1 wraps to the minimum, where wasm traps.
        let (min, eq, and) = if wide {
            (i64::MIN, Operator::I64Eq, Operator::I32And)
        } else {
            (i64::from(i32::MIN), Operator::I32Eq, Operator::I32And)
        };
        let min = self.b.constant(ty, min);
        let minus_one = self.b.constant(ty, -1);
        let is_min = self.b.op(eq, &[lhs, min], WType::I32);
        let is_minus_one = self.b.op(eq, &[rhs, minus_one], WType::I32);
        let overflow = self.b.op(and, &[is_min, is_minus_one], WType::I32);
        let one = self.b.constant(ty, 1);
        let divisor = self.b.select(overflow, one, rhs, ty);
        self.b.op(op, &[lhs, divisor], ty)
    }

    /// The low 32 bits of a shift count, and whether it is at least `width`.
    fn shift_count(&mut self, count: ValueId, width: u32) -> Result<(WValue, WValue), String> {
        let repr = self.int_repr(count)?;
        let value = self.resolve(count)?;
        Ok(match repr {
            IntRepr::Narrow(_) => {
                let width = self.b.i32(width as i32);
                let oversized = self.b.op(Operator::I32GeU, &[value[0], width], WType::I32);
                (value[0], oversized)
            }
            IntRepr::I64 | IntRepr::Wide(_) => {
                let width = self.b.i64(i64::from(width));
                let mut oversized = self.b.op(Operator::I64GeU, &[value[0], width], WType::I32);
                if value.len() > 1 {
                    let high = wide::is_zero(&mut self.b, &value[1..]);
                    let high = self.b.op(Operator::I32Eqz, &[high], WType::I32);
                    oversized = self.b.op(Operator::I32Or, &[oversized, high], WType::I32);
                }
                let low = self.b.op(Operator::I32WrapI64, &[value[0]], WType::I32);
                (low, oversized)
            }
        })
    }

    /// Shifts by `count`; counts of at least the width shift every bit out.
    pub(super) fn shift(
        &mut self,
        inst: InstId,
        kind: ShiftKind,
        value: ValueId,
        count: ValueId,
    ) -> Result<(), String> {
        let repr = self.int_repr(value)?;
        let (count, oversized) = self.shift_count(count, repr.bits())?;
        let value = self.resolve(value)?;
        let result = match repr {
            IntRepr::Narrow(bits) => {
                let value = if kind == ShiftKind::Sar {
                    self.signed32(value[0], bits)
                } else {
                    value[0]
                };
                let op = match kind {
                    ShiftKind::Shl => Operator::I32Shl,
                    ShiftKind::Shr => Operator::I32ShrU,
                    ShiftKind::Sar => Operator::I32ShrS,
                };
                let shifted = self.b.op(op, &[value, count], WType::I32);
                let fill = if kind == ShiftKind::Sar {
                    let bits = self.b.i32(31);
                    self.b.op(Operator::I32ShrS, &[value, bits], WType::I32)
                } else {
                    self.b.i32(0)
                };
                let result = self.b.select(oversized, fill, shifted, WType::I32);
                smallvec![self.canonical(result, bits)]
            }
            IntRepr::I64 => {
                let count = self.b.op(Operator::I64ExtendI32U, &[count], WType::I64);
                let op = match kind {
                    ShiftKind::Shl => Operator::I64Shl,
                    ShiftKind::Shr => Operator::I64ShrU,
                    ShiftKind::Sar => Operator::I64ShrS,
                };
                let shifted = self.b.op(op, &[value[0], count], WType::I64);
                let fill = if kind == ShiftKind::Sar {
                    wide::sign_fill(&mut self.b, &value)
                } else {
                    self.b.i64(0)
                };
                smallvec![self.b.select(oversized, fill, shifted, WType::I64)]
            }
            IntRepr::Wide(limbs) => {
                let mask = self.b.i32((limbs * 64 - 1) as i32);
                let count = self.b.op(Operator::I32And, &[count, mask], WType::I32);
                let shifted = wide::shift(&mut self.b, kind, &value, count);
                let fill = if kind == ShiftKind::Sar {
                    let fill = wide::sign_fill(&mut self.b, &value);
                    smallvec![fill; limbs]
                } else {
                    wide::zero(&mut self.b, limbs)
                };
                wide::select(&mut self.b, oversized, &fill, &shifted)
            }
        };
        self.define(inst, result);
        Ok(())
    }

    pub(super) fn neg(&mut self, inst: InstId, arg: ValueId) -> Result<(), String> {
        let repr = self.int_repr(arg)?;
        let value = self.resolve(arg)?;
        let result = match repr {
            IntRepr::Narrow(bits) => {
                let zero = self.b.i32(0);
                let negated = self.b.op(Operator::I32Sub, &[zero, value[0]], WType::I32);
                smallvec![self.canonical(negated, bits)]
            }
            IntRepr::I64 => {
                let zero = self.b.i64(0);
                smallvec![self.b.op(Operator::I64Sub, &[zero, value[0]], WType::I64)]
            }
            IntRepr::Wide(_) => wide::neg(&mut self.b, &value),
        };
        self.define(inst, result);
        Ok(())
    }

    pub(super) fn not(&mut self, inst: InstId, arg: ValueId) -> Result<(), String> {
        let repr = self.int_repr(arg)?;
        let value = self.resolve(arg)?;
        let result = match repr {
            IntRepr::Narrow(bits) => {
                let mask = self.b.i32(narrow_mask(bits));
                smallvec![self.b.op(Operator::I32Xor, &[value[0], mask], WType::I32)]
            }
            IntRepr::I64 => {
                let ones = self.b.i64(-1);
                smallvec![self.b.op(Operator::I64Xor, &[value[0], ones], WType::I64)]
            }
            IntRepr::Wide(_) => wide::not(&mut self.b, &value),
        };
        self.define(inst, result);
        Ok(())
    }

    pub(super) fn bswap(&mut self, inst: InstId, arg: ValueId) -> Result<(), String> {
        let repr = self.int_repr(arg)?;
        let value = self.resolve(arg)?;
        let result = match repr {
            // A single byte has no order to reverse.
            IntRepr::Narrow(bits) if bits <= 8 => value,
            IntRepr::Narrow(16) => {
                let mask = self.b.i32(0xff);
                let eight = self.b.i32(8);
                let low = self.b.op(Operator::I32And, &[value[0], mask], WType::I32);
                let low = self.b.op(Operator::I32Shl, &[low, eight], WType::I32);
                let high = self.b.op(Operator::I32ShrU, &[value[0], eight], WType::I32);
                smallvec![self.b.op(Operator::I32Or, &[low, high], WType::I32)]
            }
            IntRepr::Narrow(_) => {
                let eight = self.b.i32(8);
                let even = self.b.i32(0x00ff_00ff);
                let odd = self.b.i32(0xff00_ff00_u32 as i32);
                let left = self.b.op(Operator::I32Rotl, &[value[0], eight], WType::I32);
                let left = self.b.op(Operator::I32And, &[left, even], WType::I32);
                let right = self.b.op(Operator::I32Rotr, &[value[0], eight], WType::I32);
                let right = self.b.op(Operator::I32And, &[right, odd], WType::I32);
                smallvec![self.b.op(Operator::I32Or, &[left, right], WType::I32)]
            }
            // Reversing every byte reverses the limbs and the bytes in each.
            IntRepr::I64 | IntRepr::Wide(_) => value
                .iter()
                .rev()
                .map(|&limb| wide::bswap64(&mut self.b, limb))
                .collect(),
        };
        self.define(inst, result);
        Ok(())
    }

    fn compare(
        &mut self,
        repr: IntRepr,
        kind: CmpKind,
        signed: bool,
        lhs: &[WValue],
        rhs: &[WValue],
    ) -> WValue {
        match repr {
            IntRepr::Narrow(bits) => {
                let (lhs, rhs) = if signed {
                    (self.signed32(lhs[0], bits), self.signed32(rhs[0], bits))
                } else {
                    (lhs[0], rhs[0])
                };
                let op = match (kind, signed) {
                    (CmpKind::Eq, _) => Operator::I32Eq,
                    (CmpKind::Ne, _) => Operator::I32Ne,
                    (CmpKind::Lt, false) => Operator::I32LtU,
                    (CmpKind::Lt, true) => Operator::I32LtS,
                    (CmpKind::Gt, false) => Operator::I32GtU,
                    (CmpKind::Gt, true) => Operator::I32GtS,
                    (CmpKind::Le, false) => Operator::I32LeU,
                    (CmpKind::Le, true) => Operator::I32LeS,
                    (CmpKind::Ge, false) => Operator::I32GeU,
                    (CmpKind::Ge, true) => Operator::I32GeS,
                };
                self.b.op(op, &[lhs, rhs], WType::I32)
            }
            IntRepr::I64 => {
                let op = match (kind, signed) {
                    (CmpKind::Eq, _) => Operator::I64Eq,
                    (CmpKind::Ne, _) => Operator::I64Ne,
                    (CmpKind::Lt, false) => Operator::I64LtU,
                    (CmpKind::Lt, true) => Operator::I64LtS,
                    (CmpKind::Gt, false) => Operator::I64GtU,
                    (CmpKind::Gt, true) => Operator::I64GtS,
                    (CmpKind::Le, false) => Operator::I64LeU,
                    (CmpKind::Le, true) => Operator::I64LeS,
                    (CmpKind::Ge, false) => Operator::I64GeU,
                    (CmpKind::Ge, true) => Operator::I64GeS,
                };
                self.b.op(op, &[lhs[0], rhs[0]], WType::I32)
            }
            IntRepr::Wide(_) => {
                let (result, invert) = match kind {
                    CmpKind::Eq => (wide::eq(&mut self.b, lhs, rhs), false),
                    CmpKind::Ne => (wide::eq(&mut self.b, lhs, rhs), true),
                    CmpKind::Lt => (wide::lt(&mut self.b, lhs, rhs, signed), false),
                    CmpKind::Gt => (wide::lt(&mut self.b, rhs, lhs, signed), false),
                    CmpKind::Le => (wide::lt(&mut self.b, rhs, lhs, signed), true),
                    CmpKind::Ge => (wide::lt(&mut self.b, lhs, rhs, signed), true),
                };
                if invert {
                    self.b.op(Operator::I32Eqz, &[result], WType::I32)
                } else {
                    result
                }
            }
        }
    }

    pub(super) fn cmp(
        &mut self,
        inst: InstId,
        kind: CmpKind,
        signed: bool,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<(), String> {
        let result = self.compare_values(kind, signed, lhs, rhs)?;
        self.define1(inst, result);
        Ok(())
    }

    fn compare_values(
        &mut self,
        kind: CmpKind,
        signed: bool,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<WValue, String> {
        let (repr, lhs) = self.int_operand(lhs)?;
        let (rhs_repr, rhs) = self.int_operand(rhs)?;
        if repr != rhs_repr {
            return Err(format!(
                "cannot compare mismatched {repr:?} and {rhs_repr:?} values"
            ));
        }
        Ok(self.compare(repr, kind, signed, &lhs, &rhs))
    }

    pub(super) fn equal(&mut self, lhs: ValueId, rhs: ValueId) -> Result<WValue, String> {
        self.compare_values(CmpKind::Eq, false, lhs, rhs)
    }

    pub(super) fn is_zero(&mut self, inst: InstId, value: ValueId) -> Result<(), String> {
        let repr = self.int_repr(value)?;
        let value = self.resolve(value)?;
        let result = match repr {
            IntRepr::Narrow(_) => self.b.op(Operator::I32Eqz, &[value[0]], WType::I32),
            IntRepr::I64 => self.b.op(Operator::I64Eqz, &[value[0]], WType::I32),
            IntRepr::Wide(_) => wide::is_zero(&mut self.b, &value),
        };
        self.define1(inst, result);
        Ok(())
    }

    fn sign(&mut self, repr: IntRepr, value: &[WValue]) -> WValue {
        match repr {
            IntRepr::Narrow(bits) => {
                let value = self.signed32(value[0], bits);
                let zero = self.b.i32(0);
                self.b.op(Operator::I32LtS, &[value, zero], WType::I32)
            }
            IntRepr::I64 | IntRepr::Wide(_) => wide::sign(&mut self.b, value),
        }
    }

    /// The wrapped result of an operation and whether it overflowed.
    fn overflow_values(
        &mut self,
        repr: IntRepr,
        op: OverflowOp,
        lhs: &[WValue],
        rhs: &[WValue],
    ) -> (Repr, WValue) {
        match repr {
            IntRepr::Narrow(bits) => {
                // Narrow operands give exact results in 64 bits.
                let signed = op.signed();
                let lhs = self.extend64(lhs[0], bits, signed);
                let rhs = self.extend64(rhs[0], bits, signed);
                let op64 = match op {
                    OverflowOp::Uadd | OverflowOp::Sadd => Operator::I64Add,
                    OverflowOp::Usub | OverflowOp::Ssub => Operator::I64Sub,
                    OverflowOp::Umul | OverflowOp::Smul => Operator::I64Mul,
                };
                let exact = self.b.op(op64, &[lhs, rhs], WType::I64);
                let overflow = if signed {
                    let shift = self.b.i64(i64::from(64 - bits));
                    let truncated = self.b.op(Operator::I64Shl, &[exact, shift], WType::I64);
                    let truncated = self
                        .b
                        .op(Operator::I64ShrS, &[truncated, shift], WType::I64);
                    self.b.op(Operator::I64Ne, &[truncated, exact], WType::I32)
                } else {
                    let shift = self.b.i64(i64::from(bits));
                    let high = self.b.op(Operator::I64ShrU, &[exact, shift], WType::I64);
                    let zero = self.b.i64(0);
                    self.b.op(Operator::I64Ne, &[high, zero], WType::I32)
                };
                let result = self.b.op(Operator::I32WrapI64, &[exact], WType::I32);
                (smallvec![self.canonical(result, bits)], overflow)
            }
            IntRepr::I64 => {
                let (lhs, rhs) = (lhs[0], rhs[0]);
                let zero = self.b.i64(0);
                match op {
                    OverflowOp::Uadd => {
                        let sum = self.b.op(Operator::I64Add, &[lhs, rhs], WType::I64);
                        let overflow = self.b.op(Operator::I64LtU, &[sum, lhs], WType::I32);
                        (smallvec![sum], overflow)
                    }
                    OverflowOp::Usub => {
                        let difference = self.b.op(Operator::I64Sub, &[lhs, rhs], WType::I64);
                        let overflow = self.b.op(Operator::I64LtU, &[lhs, rhs], WType::I32);
                        (smallvec![difference], overflow)
                    }
                    OverflowOp::Sadd | OverflowOp::Ssub => {
                        let add = op == OverflowOp::Sadd;
                        let op = if add {
                            Operator::I64Add
                        } else {
                            Operator::I64Sub
                        };
                        let result = self.b.op(op, &[lhs, rhs], WType::I64);
                        // Addition overflows when the result's sign differs from
                        // both operands'; subtraction when the operands' signs
                        // differ and the result's differs from the minuend's.
                        let changed = self.b.op(Operator::I64Xor, &[result, lhs], WType::I64);
                        let other = if add {
                            self.b.op(Operator::I64Xor, &[result, rhs], WType::I64)
                        } else {
                            self.b.op(Operator::I64Xor, &[lhs, rhs], WType::I64)
                        };
                        let both = self.b.op(Operator::I64And, &[changed, other], WType::I64);
                        let overflow = self.b.op(Operator::I64LtS, &[both, zero], WType::I32);
                        (smallvec![result], overflow)
                    }
                    OverflowOp::Umul => {
                        let (low, high) = wide::mul_wide(&mut self.b, lhs, rhs);
                        let overflow = self.b.op(Operator::I64Ne, &[high, zero], WType::I32);
                        (smallvec![low], overflow)
                    }
                    OverflowOp::Smul => {
                        let signs = self.b.op(Operator::I64Xor, &[lhs, rhs], WType::I64);
                        let negative = self.b.op(Operator::I64LtS, &[signs, zero], WType::I32);
                        let lhs_abs = wide::abs(&mut self.b, &[lhs]);
                        let rhs_abs = wide::abs(&mut self.b, &[rhs]);
                        let (low, high) = wide::mul_wide(&mut self.b, lhs_abs[0], rhs_abs[0]);
                        let min = self.b.i64(i64::MIN);
                        let max = self.b.i64(i64::MAX);
                        let limit = self.b.select(negative, min, max, WType::I64);
                        let over_limit = self.b.op(Operator::I64GtU, &[low, limit], WType::I32);
                        let high = self.b.op(Operator::I64Ne, &[high, zero], WType::I32);
                        let overflow = self.b.op(Operator::I32Or, &[high, over_limit], WType::I32);
                        let product = self.b.op(Operator::I64Mul, &[lhs, rhs], WType::I64);
                        (smallvec![product], overflow)
                    }
                }
            }
            IntRepr::Wide(limbs) => match op {
                OverflowOp::Uadd => wide::add(&mut self.b, lhs, rhs),
                OverflowOp::Usub => wide::sub(&mut self.b, lhs, rhs),
                OverflowOp::Sadd | OverflowOp::Ssub => {
                    let (result, _) = if op == OverflowOp::Sadd {
                        wide::add(&mut self.b, lhs, rhs)
                    } else {
                        wide::sub(&mut self.b, lhs, rhs)
                    };
                    let lhs_negative = wide::sign(&mut self.b, lhs);
                    let rhs_negative = wide::sign(&mut self.b, rhs);
                    let result_negative = wide::sign(&mut self.b, &result);
                    let operands = if op == OverflowOp::Sadd {
                        Operator::I32Eq
                    } else {
                        Operator::I32Ne
                    };
                    let operands = self
                        .b
                        .op(operands, &[lhs_negative, rhs_negative], WType::I32);
                    let changed = self.b.op(
                        Operator::I32Ne,
                        &[result_negative, lhs_negative],
                        WType::I32,
                    );
                    let overflow = self
                        .b
                        .op(Operator::I32And, &[operands, changed], WType::I32);
                    (result, overflow)
                }
                OverflowOp::Umul => {
                    let args: Vec<_> = lhs.iter().chain(rhs).copied().collect();
                    let product = self.call_helper(Helper::MulFull, limbs, &args, limbs * 2);
                    let high = wide::is_zero(&mut self.b, &product[limbs..]);
                    let overflow = self.b.op(Operator::I32Eqz, &[high], WType::I32);
                    (product[..limbs].iter().copied().collect(), overflow)
                }
                OverflowOp::Smul => {
                    let lhs_negative = wide::sign(&mut self.b, lhs);
                    let rhs_negative = wide::sign(&mut self.b, rhs);
                    let negative =
                        self.b
                            .op(Operator::I32Xor, &[lhs_negative, rhs_negative], WType::I32);
                    let mut args = wide::abs(&mut self.b, lhs);
                    args.extend(wide::abs(&mut self.b, rhs));
                    let product = self.call_helper(Helper::MulFull, limbs, &args, limbs * 2);
                    let high = wide::is_zero(&mut self.b, &product[limbs..]);
                    let high = self.b.op(Operator::I32Eqz, &[high], WType::I32);
                    let low = &product[..limbs];
                    let min = wide::signed_limit(&mut self.b, limbs, false);
                    let max = wide::signed_limit(&mut self.b, limbs, true);
                    let limit = wide::select(&mut self.b, negative, &min, &max);
                    let over_limit = wide::lt(&mut self.b, &limit, low, false);
                    let overflow = self.b.op(Operator::I32Or, &[high, over_limit], WType::I32);
                    // The product's low limbs are the magnitudes' with its sign.
                    let negated = wide::neg(&mut self.b, low);
                    (wide::select(&mut self.b, negative, &negated, low), overflow)
                }
            },
        }
    }

    pub(super) fn overflow(
        &mut self,
        inst: InstId,
        op: OverflowOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<(), String> {
        let repr = self.int_repr(lhs)?;
        let lhs = self.resolve(lhs)?;
        let rhs = self.resolve(rhs)?;
        let (result, overflow) = self.overflow_values(repr, op, &lhs, &rhs);
        self.define_results(inst, [result, smallvec![overflow]]);
        Ok(())
    }

    pub(super) fn snego(&mut self, inst: InstId, arg: ValueId) -> Result<(), String> {
        let repr = self.int_repr(arg)?;
        let value = self.resolve(arg)?;
        let (result, overflow) = match repr {
            IntRepr::Narrow(bits) => {
                let zero = self.b.i32(0);
                let negated = self.b.op(Operator::I32Sub, &[zero, value[0]], WType::I32);
                let min = self.b.i32((1u32 << (bits - 1)) as i32);
                let overflow = self.b.op(Operator::I32Eq, &[value[0], min], WType::I32);
                (smallvec![self.canonical(negated, bits)], overflow)
            }
            IntRepr::I64 => {
                let zero = self.b.i64(0);
                let negated = self.b.op(Operator::I64Sub, &[zero, value[0]], WType::I64);
                let min = self.b.i64(i64::MIN);
                let overflow = self.b.op(Operator::I64Eq, &[value[0], min], WType::I32);
                (smallvec![negated], overflow)
            }
            IntRepr::Wide(limbs) => {
                let min = wide::signed_limit(&mut self.b, limbs, false);
                let overflow = wide::eq(&mut self.b, &value, &min);
                (wide::neg(&mut self.b, &value), overflow)
            }
        };
        self.define_results(inst, [result, smallvec![overflow]]);
        Ok(())
    }

    fn limit(&mut self, repr: IntRepr, kind: Limit) -> Repr {
        match repr {
            IntRepr::Narrow(bits) => {
                let mask = narrow_mask(bits) as u32;
                let value = match kind {
                    Limit::Zero => 0,
                    Limit::UnsignedMax => mask,
                    Limit::SignedMin => 1 << (bits - 1),
                    Limit::SignedMax => mask >> 1,
                };
                smallvec![self.b.i32(value as i32)]
            }
            IntRepr::I64 => smallvec![self.b.i64(match kind {
                Limit::Zero => 0,
                Limit::UnsignedMax => -1,
                Limit::SignedMin => i64::MIN,
                Limit::SignedMax => i64::MAX,
            })],
            IntRepr::Wide(limbs) => match kind {
                Limit::Zero => wide::zero(&mut self.b, limbs),
                Limit::UnsignedMax => {
                    let ones = self.b.i64(-1);
                    smallvec![ones; limbs]
                }
                Limit::SignedMin => wide::signed_limit(&mut self.b, limbs, false),
                Limit::SignedMax => wide::signed_limit(&mut self.b, limbs, true),
            },
        }
    }

    pub(super) fn saturating(
        &mut self,
        inst: InstId,
        op: SaturatingOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<(), String> {
        let repr = self.int_repr(lhs)?;
        let lhs = self.resolve(lhs)?;
        let rhs = self.resolve(rhs)?;
        let (raw, overflow) = self.overflow_values(repr, op.overflow_op(), &lhs, &rhs);
        let saturated = match op {
            SaturatingOp::Uadd | SaturatingOp::Umul => self.limit(repr, Limit::UnsignedMax),
            SaturatingOp::Usub => self.limit(repr, Limit::Zero),
            SaturatingOp::Sadd | SaturatingOp::Ssub | SaturatingOp::Smul => {
                let negative = if op == SaturatingOp::Smul {
                    let lhs_negative = self.sign(repr, &lhs);
                    let rhs_negative = self.sign(repr, &rhs);
                    self.b
                        .op(Operator::I32Xor, &[lhs_negative, rhs_negative], WType::I32)
                } else {
                    self.sign(repr, &lhs)
                };
                let min = self.limit(repr, Limit::SignedMin);
                let max = self.limit(repr, Limit::SignedMax);
                self.select_repr(repr, negative, &min, &max)
            }
        };
        let result = self.select_repr(repr, overflow, &saturated, &raw);
        self.define(inst, result);
        Ok(())
    }

    fn select_repr(
        &mut self,
        repr: IntRepr,
        cond: WValue,
        if_true: &[WValue],
        if_false: &[WValue],
    ) -> Repr {
        match repr {
            IntRepr::Narrow(_) => {
                smallvec![self.b.select(cond, if_true[0], if_false[0], WType::I32)]
            }
            IntRepr::I64 | IntRepr::Wide(_) => wide::select(&mut self.b, cond, if_true, if_false),
        }
    }

    /// Converts between integer widths, truncating or extending.
    fn resize(&mut self, value: &[WValue], from: IntRepr, to: IntRepr, signed: bool) -> Repr {
        match (from, to) {
            (IntRepr::Narrow(from), IntRepr::Narrow(to)) => {
                let value = if signed && to > from {
                    self.signed32(value[0], from)
                } else {
                    value[0]
                };
                smallvec![self.canonical(value, to)]
            }
            (IntRepr::Narrow(from), IntRepr::I64) => {
                smallvec![self.extend64(value[0], from, signed)]
            }
            (IntRepr::Narrow(from), IntRepr::Wide(limbs)) => {
                let low = self.extend64(value[0], from, signed);
                wide::extend(&mut self.b, &[low], limbs, signed)
            }
            (IntRepr::I64 | IntRepr::Wide(_), IntRepr::Narrow(to)) => {
                let low = self.b.op(Operator::I32WrapI64, &[value[0]], WType::I32);
                smallvec![self.canonical(low, to)]
            }
            (IntRepr::I64 | IntRepr::Wide(_), IntRepr::I64) => smallvec![value[0]],
            (IntRepr::I64 | IntRepr::Wide(_), IntRepr::Wide(limbs)) => {
                wide::extend(&mut self.b, value, limbs, signed)
            }
        }
    }

    pub(super) fn extend(
        &mut self,
        inst: InstId,
        from: ValueId,
        ty: Type,
        signed: bool,
    ) -> Result<(), String> {
        let from_repr = self.int_repr(from)?;
        let to_repr = int_repr(self.ctx, ty).ok_or_else(|| format!("cannot extend to {ty:?}"))?;
        let value = self.resolve(from)?;
        let result = self.resize(&value, from_repr, to_repr, signed);
        self.define(inst, result);
        Ok(())
    }

    /// Truncates, or zero-extends a pointer converted to a wider integer.
    pub(super) fn trunc(&mut self, inst: InstId, from: ValueId, ty: Type) -> Result<(), String> {
        self.extend(inst, from, ty, false)
    }

    pub(super) fn bitcast(&mut self, inst: InstId, from: ValueId, ty: Type) -> Result<(), String> {
        let from_ty = self.ty(from);
        let size = value_storage_size(from_ty, self.ctx)?;
        if size != value_storage_size(ty, self.ctx)? {
            return Err(format!(
                "bitcast requires equal-sized types: {from_ty:?} to {ty:?}"
            ));
        }
        let value = self.resolve(from)?;
        let from_indirect = is_indirect(self.ctx, from_ty);
        let result = if is_indirect(self.ctx, ty) {
            // A cast creates a value, not an alias to its source storage.
            let slot = self.slot(ty)?;
            if from_indirect {
                self.copy(slot, value[0], size);
            } else {
                self.store(from_ty, &value, slot, 0)?;
            }
            smallvec![slot]
        } else if from_indirect {
            self.load(ty, value[0], 0, true)?
        } else if value_types(self.ctx, from_ty)? == value_types(self.ctx, ty)? {
            value
        } else {
            let slot = self.slot(from_ty)?;
            self.store(from_ty, &value, slot, 0)?;
            self.load(ty, slot, 0, true)?
        };
        self.define(inst, result);
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Limit {
    Zero,
    UnsignedMax,
    SignedMin,
    SignedMax,
}
