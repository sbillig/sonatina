//! Integer instructions at every width, checked against a reference model.

use sonatina_codegen::OptLevel;
use sonatina_ir::U256;

use super::{Instance, compile};

const BINARY: &[&str] = &[
    "add", "sub", "mul", "udiv", "sdiv", "umod", "smod", "shl", "shr", "sar", "and", "or", "xor",
];
const UNARY: &[&str] = &["not", "neg", "bswap", "is_zero"];
const COMPARE: &[&str] = &[
    "eq", "ne", "lt", "gt", "le", "ge", "slt", "sgt", "sle", "sge",
];
const OVERFLOW: &[&str] = &[
    "uaddo", "saddo", "usubo", "ssubo", "umulo", "smulo", "snego",
];
const SATURATING: &[&str] = &[
    "uaddsat", "saddsat", "usubsat", "ssubsat", "umulsat", "smulsat",
];

#[derive(Clone, Copy)]
struct Width(u32);

impl Width {
    fn name(self) -> String {
        format!("i{}", self.0)
    }

    fn bytes(self) -> usize {
        (self.0 as usize).div_ceil(8)
    }

    fn mask(self) -> U256 {
        if self.0 == 256 {
            U256::MAX
        } else {
            (U256::one() << self.0) - 1
        }
    }

    fn min(self) -> U256 {
        U256::one() << (self.0 - 1)
    }

    fn negative(self, value: U256) -> bool {
        value.bit(self.0 as usize - 1)
    }

    fn neg(self, value: U256) -> U256 {
        U256::zero().overflowing_sub(value).0 & self.mask()
    }

    fn magnitude(self, value: U256) -> U256 {
        if self.negative(value) {
            self.neg(value)
        } else {
            value
        }
    }

    /// Orders signed values like unsigned ones.
    fn biased(self, value: U256) -> U256 {
        value ^ self.min()
    }

    /// Interesting values: edges of both ranges and limb boundaries.
    fn edge_values(self) -> Vec<U256> {
        let mask = self.mask();
        let mut values = vec![
            U256::zero(),
            U256::one(),
            U256::from(2u8),
            U256::from(3u8),
            mask,
            mask - 1,
            self.min(),
            self.min() - 1,
            self.min() + 1,
            U256::from(0x7fu8),
            U256::from(0x80u8),
            U256::from(0xffu8),
            U256::from(0x1234_5678u32),
            U256::from(u64::MAX),
            U256::from(u64::MAX) + 1,
            (U256::from(u64::MAX) << 64) | U256::from(7u8),
        ];
        for value in &mut values {
            *value &= mask;
        }
        values.sort();
        values.dedup();
        values
    }
}

/// A deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn value(&mut self, width: Width) -> U256 {
        let mut bytes = [0u8; 32];
        for chunk in bytes.chunks_mut(8) {
            chunk.copy_from_slice(&self.next().to_le_bytes());
        }
        let value = U256::from_little_endian(&bytes) & width.mask();
        // Mix in values of every magnitude.
        match self.next() % 3 {
            0 => value,
            1 => value >> (self.next() % u64::from(width.0)),
            _ => width.neg(value >> (self.next() % u64::from(width.0))),
        }
    }
}

/// The result and overflow flag of `op`, or nothing when it is undefined.
fn reference(op: &str, width: Width, lhs: U256, rhs: U256) -> Option<(U256, bool)> {
    let mask = width.mask();
    let bits = U256::from(width.0);
    let (lhs_negative, rhs_negative) = (width.negative(lhs), width.negative(rhs));
    let wrap = |value: U256| value & mask;
    let (sum, sum_carry) = lhs.overflowing_add(rhs);
    let (difference, _) = lhs.overflowing_sub(rhs);
    let (product, product_carry) = lhs.overflowing_mul(rhs);
    let signed_add_overflow =
        lhs_negative == rhs_negative && width.negative(wrap(sum)) != lhs_negative;
    let signed_sub_overflow =
        lhs_negative != rhs_negative && width.negative(wrap(difference)) != lhs_negative;
    let signed_mul_overflow = {
        let negative = lhs_negative != rhs_negative;
        let (magnitude, carry) = width.magnitude(lhs).overflowing_mul(width.magnitude(rhs));
        let limit = if negative {
            width.min()
        } else {
            width.min() - 1
        };
        carry || magnitude > limit
    };
    let signed_saturation = |negative: bool| {
        if negative {
            width.min()
        } else {
            width.min() - 1
        }
    };
    let flag = |value: bool| U256::from(u8::from(value));
    Some(match op {
        "add" => (wrap(sum), false),
        "sub" => (wrap(difference), false),
        "mul" => (wrap(product), false),
        "udiv" | "umod" | "sdiv" | "smod" if rhs.is_zero() => return None,
        "udiv" => (lhs / rhs, false),
        "umod" => (lhs % rhs, false),
        "sdiv" => {
            let quotient = width.magnitude(lhs) / width.magnitude(rhs);
            let negative = lhs_negative != rhs_negative;
            (
                wrap(if negative {
                    width.neg(quotient)
                } else {
                    quotient
                }),
                false,
            )
        }
        "smod" => {
            let remainder = width.magnitude(lhs) % width.magnitude(rhs);
            (
                if lhs_negative {
                    width.neg(remainder)
                } else {
                    remainder
                },
                false,
            )
        }
        "shl" if rhs >= bits => (U256::zero(), false),
        "shl" => (wrap(lhs << rhs.as_u32()), false),
        "shr" if rhs >= bits => (U256::zero(), false),
        "shr" => (lhs >> rhs.as_u32(), false),
        "sar" if lhs_negative && rhs >= bits => (mask, false),
        "sar" if rhs >= bits => (U256::zero(), false),
        "sar" if lhs_negative => {
            let shift = rhs.as_u32();
            ((lhs >> shift) | (mask ^ (mask >> shift)), false)
        }
        "sar" => (lhs >> rhs.as_u32(), false),
        "and" => (lhs & rhs, false),
        "or" => (lhs | rhs, false),
        "xor" => (lhs ^ rhs, false),
        "not" => (wrap(!lhs), false),
        "neg" => (width.neg(lhs), false),
        "bswap" => {
            let bytes = lhs.to_little_endian();
            let mut swapped = [0u8; 32];
            for (index, byte) in bytes[..width.bytes()].iter().rev().enumerate() {
                swapped[index] = *byte;
            }
            (U256::from_little_endian(&swapped), false)
        }
        "is_zero" => (flag(lhs.is_zero()), false),
        "eq" => (flag(lhs == rhs), false),
        "ne" => (flag(lhs != rhs), false),
        "lt" => (flag(lhs < rhs), false),
        "gt" => (flag(lhs > rhs), false),
        "le" => (flag(lhs <= rhs), false),
        "ge" => (flag(lhs >= rhs), false),
        "slt" => (flag(width.biased(lhs) < width.biased(rhs)), false),
        "sgt" => (flag(width.biased(lhs) > width.biased(rhs)), false),
        "sle" => (flag(width.biased(lhs) <= width.biased(rhs)), false),
        "sge" => (flag(width.biased(lhs) >= width.biased(rhs)), false),
        "uaddo" => (wrap(sum), sum_carry || sum > mask),
        "usubo" => (wrap(difference), lhs < rhs),
        "umulo" => (wrap(product), product_carry || product > mask),
        "saddo" => (wrap(sum), signed_add_overflow),
        "ssubo" => (wrap(difference), signed_sub_overflow),
        "smulo" => (wrap(product), signed_mul_overflow),
        "snego" => (width.neg(lhs), lhs == width.min()),
        "uaddsat" if sum_carry || sum > mask => (mask, false),
        "uaddsat" => (wrap(sum), false),
        "usubsat" if lhs < rhs => (U256::zero(), false),
        "usubsat" => (wrap(difference), false),
        "umulsat" if product_carry || product > mask => (mask, false),
        "umulsat" => (wrap(product), false),
        "saddsat" if signed_add_overflow => (signed_saturation(lhs_negative), false),
        "saddsat" => (wrap(sum), false),
        "ssubsat" if signed_sub_overflow => (signed_saturation(lhs_negative), false),
        "ssubsat" => (wrap(difference), false),
        "smulsat" if signed_mul_overflow => {
            (signed_saturation(lhs_negative != rhs_negative), false)
        }
        "smulsat" => (wrap(product), false),
        _ => unreachable!("no reference for {op}"),
    })
}

fn check_op(op: &str, width: Width, cases: &[(U256, U256)]) {
    let ty = width.name();
    let overflow_op = OVERFLOW.contains(&op);
    let result_ty = if COMPARE.contains(&op) || op == "is_zero" {
        "i1".to_string()
    } else {
        ty.clone()
    };
    let expression = match op {
        "snego" => format!("(v5.{ty}, v6.i1) = snego v3;"),
        _ if overflow_op => format!("(v5.{ty}, v6.i1) = {op} v3 v4;"),
        _ if UNARY.contains(&op) => format!("v5.{result_ty} = {op} v3;"),
        "shl" | "shr" | "sar" => format!("v5.{ty} = {op} v4 v3;"),
        _ => format!("v5.{result_ty} = {op} v3 v4;"),
    };
    let status = if overflow_op {
        "v7.i8 = zext v6 i8;\n    return v7;"
    } else {
        "return 0.i8;"
    };
    let source = format!(
        r#"func public %apply(v0.*{ty}, v1.*{ty}, v2.*{result_ty}) -> i8 {{
block0:
    v3.{ty} = mload v0 {ty};
    v4.{ty} = mload v1 {ty};
    {expression}
    mstore v2 v5 {result_ty};
    {status}
}}
"#
    );
    let result_bytes = if result_ty == "i1" { 1 } else { width.bytes() };
    for level in [OptLevel::O0, OptLevel::O2] {
        let mut instance = Instance::new(&compile(&source, level));
        let base = instance.scratch(96);
        for &(lhs, rhs) in cases {
            let Some((expected, overflow)) = reference(op, width, lhs, rhs) else {
                continue;
            };
            instance.write(base, &lhs.to_little_endian()[..width.bytes()]);
            instance.write(base + 32, &rhs.to_little_endian()[..width.bytes()]);
            instance.write(base + 64, &[0xa5; 32]);
            let status: i32 = instance
                .call("apply", (base as i32, base as i32 + 32, base as i32 + 64))
                .unwrap_or_else(|error| panic!("{op} {ty} {lhs:#x} {rhs:#x} {level:?}: {error}"));
            let result = instance.read(base + 64, 32);
            assert_eq!(
                U256::from_little_endian(&result[..result_bytes]),
                expected,
                "{op} {ty} {lhs:#x} {rhs:#x} {level:?}"
            );
            assert!(
                result[result_bytes..].iter().all(|&byte| byte == 0xa5),
                "{op} {ty} wrote past its result"
            );
            assert_eq!(
                status,
                i32::from(overflow),
                "{op} {ty} {lhs:#x} {rhs:#x} {level:?} overflow"
            );
        }
    }
}

fn cases(op: &str, width: Width, rng: &mut Rng) -> Vec<(U256, U256)> {
    let edges = width.edge_values();
    let mut cases: Vec<_> = edges
        .iter()
        .flat_map(|&lhs| edges.iter().map(move |&rhs| (lhs, rhs)))
        .collect();
    if matches!(op, "shl" | "shr" | "sar") {
        let counts = [
            0u32, 1, 7, 8, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 300,
        ];
        for &lhs in &edges {
            for &count in &counts {
                cases.push((lhs, U256::from(count) & width.mask()));
            }
        }
    }
    for _ in 0..200 {
        cases.push((rng.value(width), rng.value(width)));
    }
    cases
}

#[test]
fn integer_operations_match_the_reference_at_every_width() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for bits in [1, 8, 16, 32, 64, 128, 256] {
        let width = Width(bits);
        for &op in BINARY
            .iter()
            .chain(UNARY)
            .chain(COMPARE)
            .chain(OVERFLOW)
            .chain(SATURATING)
        {
            // The IR has no one-bit byte swap or saturating arithmetic.
            if bits == 1 && (op == "bswap" || SATURATING.contains(&op)) {
                continue;
            }
            let cases = cases(op, width, &mut rng);
            check_op(op, width, &cases);
        }
    }
}

#[test]
fn casts_preserve_values() {
    let source = r#"
func public %sext8(v0.i8) -> i64 {
block0:
    v1.i64 = sext v0 i64;
    return v1;
}
func public %sext1(v0.i1) -> i32 {
block0:
    v1.i32 = sext v0 i32;
    return v1;
}
func public %trunc(v0.i64) -> i32 {
block0:
    v1.i8 = trunc v0 i8;
    v2.i32 = zext v1 i32;
    return v2;
}
func public %wide(v0.i64) -> (i64, i64) {
block0:
    v1.i256 = sext v0 i256;
    v2.i128 = trunc v1 i128;
    v3.i256 = zext v2 i256;
    v4.i64 = trunc v3 i64;
    v5.i256 = shr 192.i256 v3;
    v6.i64 = trunc v5 i64;
    return (v4, v6);
}
"#;
    for level in super::LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        assert_eq!(instance.call::<_, i64>("sext8", 0x80i32).unwrap(), -128);
        assert_eq!(instance.call::<_, i64>("sext8", 0x17f).unwrap(), 127);
        assert_eq!(instance.call::<_, i32>("sext1", 1).unwrap(), -1);
        assert_eq!(instance.call::<_, i32>("trunc", 0x1234i64).unwrap(), 0x34);
        assert_eq!(
            instance.call::<_, (i64, i64)>("wide", -5i64).unwrap(),
            (-5, 0)
        );
    }
}
