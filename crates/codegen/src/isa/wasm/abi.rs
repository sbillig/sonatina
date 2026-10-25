use smallvec::{SmallVec, smallvec};
use sonatina_ir::{Signature, Type, module::ModuleCtx, types::CompoundType};
use waffle::{SignatureData, Type as WType};

pub(super) type WasmTypes = SmallVec<[WType; 4]>;

/// How an integer-like Sonatina value is carried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IntRepr {
    /// An `i32` holding a value of this many bits, zero-extended.
    Narrow(u32),
    I64,
    /// This many `i64` limbs, least significant first.
    Wide(usize),
}

impl IntRepr {
    pub(super) fn bits(self) -> u32 {
        match self {
            Self::Narrow(bits) => bits,
            Self::I64 => 64,
            Self::Wide(limbs) => limbs as u32 * 64,
        }
    }
}

/// Integer types, and pointers as their 32-bit addresses.
pub(super) fn int_repr(ctx: &ModuleCtx, ty: Type) -> Option<IntRepr> {
    Some(match ty {
        Type::I1 => IntRepr::Narrow(1),
        Type::I8 => IntRepr::Narrow(8),
        Type::I16 => IntRepr::Narrow(16),
        Type::I32 => IntRepr::Narrow(32),
        Type::I64 => IntRepr::I64,
        Type::I128 => IntRepr::Wide(2),
        Type::I256 => IntRepr::Wide(4),
        _ if is_reference(ctx, ty) => IntRepr::Narrow(32),
        _ => return None,
    })
}

pub(super) fn is_reference(ctx: &ModuleCtx, ty: Type) -> bool {
    matches!(
        ty.resolve_compound(ctx),
        Some(CompoundType::Ptr(_) | CompoundType::ObjRef(_) | CompoundType::ConstRef(_))
    )
}

/// Aggregates are values held in linear memory and carried by address.
pub(super) fn is_indirect(ctx: &ModuleCtx, ty: Type) -> bool {
    matches!(
        ty.resolve_compound(ctx),
        Some(CompoundType::Array { .. } | CompoundType::Struct(_) | CompoundType::Enum(_))
    )
}

pub(super) fn value_types(ctx: &ModuleCtx, ty: Type) -> Result<WasmTypes, String> {
    Ok(match ty {
        Type::Unit => smallvec![],
        Type::I1 | Type::I8 | Type::I16 | Type::I32 => smallvec![WType::I32],
        Type::I64 => smallvec![WType::I64],
        Type::I128 => smallvec![WType::I64; 2],
        Type::I256 => smallvec![WType::I64; 4],
        _ if is_reference(ctx, ty) || is_indirect(ctx, ty) => smallvec![WType::I32],
        _ => return Err(format!("type {ty:?} has no wasm value representation")),
    })
}

/// The storage a single result is copied into for the caller: an aggregate,
/// or the referent of a returned constant reference.
pub(super) fn indirect_return_storage(ctx: &ModuleCtx, ty: Type) -> Option<Type> {
    if is_indirect(ctx, ty) {
        Some(ty)
    } else if let Some(CompoundType::ConstRef(inner)) = ty.resolve_compound(ctx) {
        Some(inner)
    } else {
        None
    }
}

pub(super) fn returns_indirect(ctx: &ModuleCtx, sig: &Signature) -> bool {
    matches!(sig.ret_tys(), [ret] if indirect_return_storage(ctx, *ret).is_some())
}

pub(super) fn validate_signature(ctx: &ModuleCtx, sig: &Signature) -> Result<(), String> {
    if sig
        .args()
        .iter()
        .chain(sig.ret_tys())
        .any(|ty| *ty == Type::Unit)
    {
        return Err(format!(
            "wasm backend does not support explicit unit values in function signatures: {}",
            sig.name()
        ));
    }
    if sig.ret_tys().len() > 1
        && sig
            .ret_tys()
            .iter()
            .any(|ty| indirect_return_storage(ctx, *ty).is_some())
    {
        return Err(format!(
            "wasm backend does not support multi-return signatures containing indirect return types: {}",
            sig.name()
        ));
    }
    Ok(())
}

/// Indirect results return through a caller-allocated buffer, passed as a
/// hidden first parameter. `i128` and `i256` values pass as several limbs.
pub(super) fn signature_data(ctx: &ModuleCtx, sig: &Signature) -> Result<SignatureData, String> {
    validate_signature(ctx, sig)?;
    let indirect = returns_indirect(ctx, sig);
    let mut params = Vec::new();
    if indirect {
        params.push(WType::I32);
    }
    for &arg in sig.args() {
        params.extend(value_types(ctx, arg)?);
    }
    let mut returns = Vec::new();
    if !indirect {
        for &ret in sig.ret_tys() {
            returns.extend(value_types(ctx, ret)?);
        }
    }
    Ok(SignatureData { params, returns })
}
