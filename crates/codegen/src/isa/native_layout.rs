//! Layout and legalization shared by the host-memory backends.

use sonatina_ir::{
    Function, Immediate, Module, Type, ValueId, global_variable::GvInitializer, module::ModuleCtx,
    types::CompoundType,
};
use sonatina_verifier::{VerifierConfig, verify_function_signature};

use crate::transform::aggregate::{EnumLowerToProduct, object_abi::legalize_native_object_returns};

/// Lowers enums to products and moves fresh object returns into caller
/// storage, after checking every signature.
pub(crate) fn legalize_native_module(module: &Module) -> Result<(), String> {
    EnumLowerToProduct.run(module);
    for func in module.funcs() {
        let report = verify_function_signature(&module.ctx, func, &VerifierConfig::default());
        if report.has_errors() {
            return Err(report.to_string());
        }
    }
    legalize_native_object_returns(module)
}

pub(crate) fn value_storage_size(ty: Type, ctx: &ModuleCtx) -> Result<u32, String> {
    let size = ctx
        .size_of(ty)
        .map_err(|error| format!("cannot lay out type {ty:?}: {error:?}"))?;
    u32::try_from(size).map_err(|_| format!("type {ty:?} is too large for the native layout"))
}

pub(crate) fn value_storage_alignment(ty: Type, ctx: &ModuleCtx) -> Result<u64, String> {
    let alignment = ctx
        .align_of(ty)
        .map_err(|error| format!("cannot align type {ty:?}: {error:?}"))?;
    if !alignment.is_power_of_two() {
        return Err(format!(
            "type {ty:?} has non-power-of-two alignment {alignment}"
        ));
    }
    Ok(alignment as u64)
}

pub(crate) fn aggregate_elem_offset(
    ctx: &ModuleCtx,
    aggregate_ty: Type,
    idx: usize,
) -> Result<(i32, Type), String> {
    if let Some((offset, ty)) = ctx.aggregate_elem_offset(aggregate_ty, idx) {
        return Ok((
            i32::try_from(offset)
                .map_err(|_| format!("aggregate offset {offset} overflows i32"))?,
            ty,
        ));
    }
    if let Some(CompoundType::ObjRef(inner) | CompoundType::ConstRef(inner)) =
        aggregate_ty.resolve_compound(ctx)
    {
        return aggregate_elem_offset(ctx, inner, idx);
    }
    Err(format!(
        "cannot compute aggregate element offset for {aggregate_ty:?}[{idx}]"
    ))
}

pub(crate) fn constant_value_index(
    function: &Function,
    value_id: ValueId,
    inst_name: &str,
) -> Result<usize, String> {
    function
        .dfg
        .value_imm(value_id)
        .and_then(|imm| imm.to_nonnegative_usize())
        .ok_or_else(|| format!("{inst_name} index must be a nonnegative constant"))
}

pub(crate) fn compute_element_size(obj_ty: Type, ctx: &ModuleCtx) -> Result<usize, String> {
    match obj_ty.resolve_compound(ctx) {
        Some(CompoundType::Array { elem, .. }) => ctx
            .size_of(elem)
            .map_err(|error| format!("cannot lay out array element {elem:?}: {error:?}")),
        Some(CompoundType::ObjRef(inner) | CompoundType::ConstRef(inner)) => {
            compute_element_size(inner, ctx)
        }
        _ => Err(format!("cannot index non-array value of type {obj_ty:?}")),
    }
}

pub(crate) fn serialize_initializer(
    init: &GvInitializer,
    ty: Type,
    bytes: &mut [u8],
    ctx: &ModuleCtx,
) -> Result<(), String> {
    match init {
        GvInitializer::Immediate(imm) => {
            if matches!(imm, Immediate::EnumTag { .. }) {
                return Err("enum-tag initializer survived legalization".into());
            }
            if imm.ty() != ty && !(ty.is_pointer(ctx) && imm.ty() == ctx.type_layout.pointer_repl())
            {
                return Err(format!(
                    "initializer type mismatch: expected {ty:?}, found {:?}",
                    imm.ty()
                ));
            }
            let words = imm.zext(Type::I256).as_i256().to_u256().to_little_endian();
            bytes.copy_from_slice(&words[..bytes.len()]);
        }
        GvInitializer::Array(elements) => {
            let Some(CompoundType::Array { elem, len }) = ty.resolve_compound(ctx) else {
                return Err(format!("array initializer used for {ty:?}"));
            };
            if elements.len() != len {
                return Err(format!(
                    "array initializer length mismatch: expected {len}, found {}",
                    elements.len()
                ));
            }
            let size = value_storage_size(elem, ctx)? as usize;
            for (index, init) in elements.iter().enumerate() {
                let start = index * size;
                serialize_initializer(init, elem, &mut bytes[start..start + size], ctx)?;
            }
        }
        GvInitializer::Struct(fields) => {
            let Some(CompoundType::Struct(data)) = ty.resolve_compound(ctx) else {
                return Err(format!("struct initializer used for {ty:?}"));
            };
            if fields.len() != data.fields.len() {
                return Err(format!(
                    "struct initializer field count mismatch: expected {}, found {}",
                    data.fields.len(),
                    fields.len()
                ));
            }
            for (index, (init, field)) in fields.iter().zip(data.fields).enumerate() {
                let (offset, _) = aggregate_elem_offset(ctx, ty, index)?;
                let start = offset as usize;
                let size = value_storage_size(field, ctx)? as usize;
                serialize_initializer(init, field, &mut bytes[start..start + size], ctx)?;
            }
        }
    }
    Ok(())
}
