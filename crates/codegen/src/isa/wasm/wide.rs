//! Arithmetic on integers held as little-endian `i64` limbs.

use smallvec::smallvec;
use waffle::{Operator, Type as WType, Value as WValue};

use super::builder::{Builder, Repr};

pub(super) fn zero(b: &mut Builder, limbs: usize) -> Repr {
    let zero = b.i64(0);
    smallvec![zero; limbs]
}

/// Extends `value` to `limbs` limbs with zero or sign limbs.
pub(super) fn extend(b: &mut Builder, value: &[WValue], limbs: usize, signed: bool) -> Repr {
    let fill = if signed {
        let top = *value.last().expect("an integer has at least one limb");
        let bits = b.i64(63);
        b.op(Operator::I64ShrS, &[top, bits], WType::I64)
    } else {
        b.i64(0)
    };
    let mut result: Repr = value.iter().copied().take(limbs).collect();
    result.resize(limbs, fill);
    result
}

fn carry_out(b: &mut Builder, sum: WValue, operand: WValue) -> WValue {
    b.op(Operator::I64LtU, &[sum, operand], WType::I32)
}

fn widen_flag(b: &mut Builder, flag: WValue) -> WValue {
    b.op(Operator::I64ExtendI32U, &[flag], WType::I64)
}

/// Returns the wrapped sum and whether it carried out of the top limb.
pub(super) fn add(b: &mut Builder, lhs: &[WValue], rhs: &[WValue]) -> (Repr, WValue) {
    let mut result = Repr::new();
    let mut carry = None;
    for (&lhs, &rhs) in lhs.iter().zip(rhs) {
        let sum = b.op(Operator::I64Add, &[lhs, rhs], WType::I64);
        let mut flag = carry_out(b, sum, lhs);
        let sum = if let Some(carry) = carry {
            let carry = widen_flag(b, carry);
            let total = b.op(Operator::I64Add, &[sum, carry], WType::I64);
            let carried = carry_out(b, total, sum);
            flag = b.op(Operator::I32Or, &[flag, carried], WType::I32);
            total
        } else {
            sum
        };
        result.push(sum);
        carry = Some(flag);
    }
    (result, carry.expect("an integer has at least one limb"))
}

/// Returns the wrapped difference and whether it borrowed past the top limb.
pub(super) fn sub(b: &mut Builder, lhs: &[WValue], rhs: &[WValue]) -> (Repr, WValue) {
    let mut result = Repr::new();
    let mut borrow = None;
    for (&lhs, &rhs) in lhs.iter().zip(rhs) {
        let difference = b.op(Operator::I64Sub, &[lhs, rhs], WType::I64);
        let mut flag = b.op(Operator::I64LtU, &[lhs, rhs], WType::I32);
        let difference = if let Some(borrow) = borrow {
            let borrow = widen_flag(b, borrow);
            let total = b.op(Operator::I64Sub, &[difference, borrow], WType::I64);
            let borrowed = b.op(Operator::I64LtU, &[difference, borrow], WType::I32);
            flag = b.op(Operator::I32Or, &[flag, borrowed], WType::I32);
            total
        } else {
            difference
        };
        result.push(difference);
        borrow = Some(flag);
    }
    (result, borrow.expect("an integer has at least one limb"))
}

pub(super) fn neg(b: &mut Builder, value: &[WValue]) -> Repr {
    let zero = zero(b, value.len());
    sub(b, &zero, value).0
}

pub(super) fn bitwise(b: &mut Builder, op: Operator, lhs: &[WValue], rhs: &[WValue]) -> Repr {
    lhs.iter()
        .zip(rhs)
        .map(|(&lhs, &rhs)| b.op(op, &[lhs, rhs], WType::I64))
        .collect()
}

pub(super) fn not(b: &mut Builder, value: &[WValue]) -> Repr {
    let ones = b.i64(-1);
    value
        .iter()
        .map(|&limb| b.op(Operator::I64Xor, &[limb, ones], WType::I64))
        .collect()
}

pub(super) fn select(
    b: &mut Builder,
    cond: WValue,
    if_true: &[WValue],
    if_false: &[WValue],
) -> Repr {
    if_true
        .iter()
        .zip(if_false)
        .map(|(&if_true, &if_false)| b.select(cond, if_true, if_false, WType::I64))
        .collect()
}

/// The bitwise OR of all limbs.
pub(super) fn or_limbs(b: &mut Builder, value: &[WValue]) -> WValue {
    let (&first, rest) = value
        .split_first()
        .expect("an integer has at least one limb");
    rest.iter().fold(first, |acc, &limb| {
        b.op(Operator::I64Or, &[acc, limb], WType::I64)
    })
}

pub(super) fn is_zero(b: &mut Builder, value: &[WValue]) -> WValue {
    let any = or_limbs(b, value);
    b.op(Operator::I64Eqz, &[any], WType::I32)
}

pub(super) fn eq(b: &mut Builder, lhs: &[WValue], rhs: &[WValue]) -> WValue {
    let difference = bitwise(b, Operator::I64Xor, lhs, rhs);
    is_zero(b, &difference)
}

pub(super) fn lt(b: &mut Builder, lhs: &[WValue], rhs: &[WValue], signed: bool) -> WValue {
    let top = lhs.len() - 1;
    let mut result = None;
    for (index, (&lhs, &rhs)) in lhs.iter().zip(rhs).enumerate() {
        let op = if signed && index == top {
            Operator::I64LtS
        } else {
            Operator::I64LtU
        };
        let less = b.op(op, &[lhs, rhs], WType::I32);
        result = Some(match result {
            None => less,
            Some(lower) => {
                let equal = b.op(Operator::I64Eq, &[lhs, rhs], WType::I32);
                let lower = b.op(Operator::I32And, &[equal, lower], WType::I32);
                b.op(Operator::I32Or, &[less, lower], WType::I32)
            }
        });
    }
    result.expect("an integer has at least one limb")
}

/// Whether the two's-complement value is negative.
pub(super) fn sign(b: &mut Builder, value: &[WValue]) -> WValue {
    let top = *value.last().expect("an integer has at least one limb");
    let zero = b.i64(0);
    b.op(Operator::I64LtS, &[top, zero], WType::I32)
}

pub(super) fn abs(b: &mut Builder, value: &[WValue]) -> Repr {
    let negative = sign(b, value);
    let negated = neg(b, value);
    select(b, negative, &negated, value)
}

/// The signed minimum or maximum of `limbs` limbs.
pub(super) fn signed_limit(b: &mut Builder, limbs: usize, max: bool) -> Repr {
    let low = b.i64(if max { -1 } else { 0 });
    let high = b.i64(if max { i64::MAX } else { i64::MIN });
    let mut result: Repr = smallvec![low; limbs];
    result[limbs - 1] = high;
    result
}

/// The full 128-bit product of two 64-bit limbs, as `(low, high)`.
pub(super) fn mul_wide(b: &mut Builder, lhs: WValue, rhs: WValue) -> (WValue, WValue) {
    let mask = b.i64(0xffff_ffff);
    let thirty_two = b.i64(32);
    let mut split = |value| {
        let low = b.op(Operator::I64And, &[value, mask], WType::I64);
        let high = b.op(Operator::I64ShrU, &[value, thirty_two], WType::I64);
        (low, high)
    };
    let (lhs_low, lhs_high) = split(lhs);
    let (rhs_low, rhs_high) = split(rhs);
    let low_low = b.op(Operator::I64Mul, &[lhs_low, rhs_low], WType::I64);
    let low_high = b.op(Operator::I64Mul, &[lhs_low, rhs_high], WType::I64);
    let high_low = b.op(Operator::I64Mul, &[lhs_high, rhs_low], WType::I64);
    let high_high = b.op(Operator::I64Mul, &[lhs_high, rhs_high], WType::I64);
    // The middle column holds three 32-bit terms, so it cannot overflow.
    let carry = b.op(Operator::I64ShrU, &[low_low, thirty_two], WType::I64);
    let low_high_low = b.op(Operator::I64And, &[low_high, mask], WType::I64);
    let high_low_low = b.op(Operator::I64And, &[high_low, mask], WType::I64);
    let middle = b.op(Operator::I64Add, &[carry, low_high_low], WType::I64);
    let middle = b.op(Operator::I64Add, &[middle, high_low_low], WType::I64);
    let low_low_low = b.op(Operator::I64And, &[low_low, mask], WType::I64);
    let middle_low = b.op(Operator::I64Shl, &[middle, thirty_two], WType::I64);
    let low = b.op(Operator::I64Or, &[low_low_low, middle_low], WType::I64);
    let low_high_high = b.op(Operator::I64ShrU, &[low_high, thirty_two], WType::I64);
    let high_low_high = b.op(Operator::I64ShrU, &[high_low, thirty_two], WType::I64);
    let middle_high = b.op(Operator::I64ShrU, &[middle, thirty_two], WType::I64);
    let high = b.op(Operator::I64Add, &[high_high, low_high_high], WType::I64);
    let high = b.op(Operator::I64Add, &[high, high_low_high], WType::I64);
    let high = b.op(Operator::I64Add, &[high, middle_high], WType::I64);
    (low, high)
}

/// The low `limbs` limbs of the product.
pub(super) fn mul(b: &mut Builder, lhs: &[WValue], rhs: &[WValue], limbs: usize) -> Repr {
    let mut result: Vec<Option<WValue>> = vec![None; limbs];
    for (i, &lhs) in lhs.iter().enumerate().take(limbs) {
        let mut carry = None;
        for (j, &rhs) in rhs.iter().enumerate() {
            let k = i + j;
            if k + 1 == limbs {
                // Only the low half of the top column's product survives.
                let product = b.op(Operator::I64Mul, &[lhs, rhs], WType::I64);
                let mut sum = product;
                for addend in [result[k], carry].into_iter().flatten() {
                    sum = b.op(Operator::I64Add, &[sum, addend], WType::I64);
                }
                result[k] = Some(sum);
                carry = None;
                break;
            }
            // `result[k] + lhs * rhs + carry` fits in two limbs.
            let (mut sum, mut high) = mul_wide(b, lhs, rhs);
            for addend in [result[k], carry].into_iter().flatten() {
                let total = b.op(Operator::I64Add, &[sum, addend], WType::I64);
                let carried = carry_out(b, total, addend);
                let carried = widen_flag(b, carried);
                high = b.op(Operator::I64Add, &[high, carried], WType::I64);
                sum = total;
            }
            result[k] = Some(sum);
            carry = Some(high);
        }
        if let Some(carry) = carry
            && i + rhs.len() < limbs
        {
            result[i + rhs.len()] = Some(carry);
        }
    }
    let zero = b.i64(0);
    result
        .into_iter()
        .map(|limb| limb.unwrap_or(zero))
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ShiftKind {
    Shl,
    Shr,
    Sar,
}

/// Shifts `value` by `count` bits, an `i32` below the value's width.
pub(super) fn shift(b: &mut Builder, kind: ShiftKind, value: &[WValue], count: WValue) -> Repr {
    let limbs = value.len();
    let fill = if kind == ShiftKind::Sar {
        sign_fill(b, value)
    } else {
        b.i64(0)
    };
    // Move whole limbs, one power of two at a time.
    let mut current: Repr = value.iter().copied().collect();
    let mut step = 1;
    while step < limbs {
        let mask = b.i32((step * 64) as i32);
        let bit = b.op(Operator::I32And, &[count, mask], WType::I32);
        let moved: Repr = (0..limbs)
            .map(|index| {
                let source = if kind == ShiftKind::Shl {
                    index.checked_sub(step)
                } else {
                    Some(index + step).filter(|&source| source < limbs)
                };
                let source = source.map_or(fill, |source| current[source]);
                b.select(bit, source, current[index], WType::I64)
            })
            .collect();
        current = moved;
        step *= 2;
    }
    // Then funnel-shift by the remaining bits. Shifting the neighbor by one
    // and then by `63 - bits` gives its high bits, or nothing when `bits` is 0.
    let sixty_three = b.i32(63);
    let bits = b.op(Operator::I32And, &[count, sixty_three], WType::I32);
    let bits = b.op(Operator::I64ExtendI32U, &[bits], WType::I64);
    let one = b.i64(1);
    let sixty_three = b.i64(63);
    let complement = b.op(Operator::I64Sub, &[sixty_three, bits], WType::I64);
    (0..limbs)
        .map(|index| {
            let (main_op, spill_op, neighbor) = match kind {
                ShiftKind::Shl => (
                    Operator::I64Shl,
                    Operator::I64ShrU,
                    index.checked_sub(1).map(|source| current[source]),
                ),
                ShiftKind::Shr | ShiftKind::Sar => (
                    if kind == ShiftKind::Sar && index + 1 == limbs {
                        Operator::I64ShrS
                    } else {
                        Operator::I64ShrU
                    },
                    Operator::I64Shl,
                    current.get(index + 1).copied(),
                ),
            };
            let main = b.op(main_op, &[current[index], bits], WType::I64);
            let Some(neighbor) = neighbor else {
                return main;
            };
            let spill = b.op(spill_op, &[neighbor, one], WType::I64);
            let spill = b.op(spill_op, &[spill, complement], WType::I64);
            b.op(Operator::I64Or, &[main, spill], WType::I64)
        })
        .collect()
}

/// A limb of copies of the sign bit.
pub(super) fn sign_fill(b: &mut Builder, value: &[WValue]) -> WValue {
    let top = *value.last().expect("an integer has at least one limb");
    let bits = b.i64(63);
    b.op(Operator::I64ShrS, &[top, bits], WType::I64)
}

pub(super) fn bswap64(b: &mut Builder, value: WValue) -> WValue {
    let mut value = value;
    for (mask, bits) in [(0x00ff_00ff_00ff_00ff_u64, 8), (0x0000_ffff_0000_ffff, 16)] {
        let mask = b.i64(mask as i64);
        let bits = b.i64(bits);
        let high = b.op(Operator::I64ShrU, &[value, bits], WType::I64);
        let high = b.op(Operator::I64And, &[high, mask], WType::I64);
        let low = b.op(Operator::I64And, &[value, mask], WType::I64);
        let low = b.op(Operator::I64Shl, &[low, bits], WType::I64);
        value = b.op(Operator::I64Or, &[high, low], WType::I64);
    }
    let thirty_two = b.i64(32);
    b.op(Operator::I64Rotl, &[value, thirty_two], WType::I64)
}
