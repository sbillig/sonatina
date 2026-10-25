use std::collections::HashMap;

use cranelift_codegen::ir::{
    self as clif, InstBuilder, MemFlagsData, StackSlotData, StackSlotKind,
};
use cranelift_frontend::FunctionBuilder;
use sonatina_ir::{Function, Type, ValueId, module::ModuleCtx, types::CompoundType};

use super::{
    aggregate_elem_offset, constant_value_index, copy_bytes, create_stack_slot_for_type,
    load_i256_limb, resize_int_value, resolve_value, signed_scalar, sonatina_type_to_clif_or_err,
    uses_indirect_value_representation, value_storage_alignment, value_storage_size,
};

pub(super) fn translate_bitcast(
    value: clif::Value,
    from_ty: Type,
    to_ty: Type,
    ctx: &ModuleCtx,
    pointer_type: clif::Type,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let size = value_storage_size(from_ty, ctx)?;
    if size != value_storage_size(to_ty, ctx)? {
        return Err(format!(
            "bitcast requires equal-sized types: {from_ty:?} to {to_ty:?}"
        ));
    }
    let from_indirect = uses_indirect_value_representation(ctx, from_ty);
    if uses_indirect_value_representation(ctx, to_ty) {
        // A cast creates a value, not an alias to its source storage. Allocate
        // for the destination layout, which may require stronger alignment.
        let dest = create_stack_slot_for_type(to_ty, ctx, builder)?;
        if from_indirect {
            copy_bytes(value, dest, size, builder);
        } else {
            builder.ins().store(MemFlagsData::new(), value, dest, 0);
        }
        Ok(dest)
    } else {
        let clif_ty = sonatina_type_to_clif_or_err(to_ty, pointer_type)?;
        if from_indirect {
            Ok(builder.ins().load(clif_ty, MemFlagsData::new(), value, 0))
        } else if builder.func.dfg.value_type(value) == clif_ty {
            Ok(value)
        } else {
            Err(format!(
                "unsupported scalar bitcast: {from_ty:?} to {to_ty:?}"
            ))
        }
    }
}

pub(super) fn storage_chunks(size: u32) -> impl Iterator<Item = (i32, clif::Type)> {
    let mut offset = 0;
    std::iter::from_fn(move || {
        let (bytes, ty) = match size - offset {
            0 => return None,
            1 => (1, clif::types::I8),
            2..=3 => (2, clif::types::I16),
            4..=7 => (4, clif::types::I32),
            _ => (8, clif::types::I64),
        };
        let chunk = (offset as i32, ty);
        offset += bytes;
        Some(chunk)
    })
}

pub(super) fn translate_aggregate_projection(
    ctx: &ModuleCtx,
    function: &Function,
    values: &[ValueId],
    inst_name: &str,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let Some((&base_value, indices)) = values.split_first() else {
        return Err(format!("{inst_name} requires a base value"));
    };
    let base = resolve_value(function, base_value, value_map, builder)?;
    let mut offset = 0i64;
    let mut current_ty = function.dfg.value_ty(base_value);

    for idx_value in indices {
        let idx = constant_value_index(function, *idx_value, inst_name)?;
        let (field_offset, elem_ty) = aggregate_elem_offset(ctx, current_ty, idx)?;
        offset += i64::from(field_offset);
        current_ty = elem_ty;
    }

    Ok(if offset == 0 {
        base
    } else {
        builder.ins().iadd_imm_s(base, offset)
    })
}

pub(super) fn translate_gep(
    ctx: &ModuleCtx,
    function: &Function,
    values: &[ValueId],
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let Some((&base_value, indices)) = values.split_first() else {
        return Err("gep requires a base pointer".into());
    };
    let mut addr = resolve_value(function, base_value, value_map, builder)?;
    let mut current_ty = function.dfg.value_ty(base_value);

    for &idx_value in indices {
        let Some(compound) = current_ty.resolve_compound(ctx) else {
            return Err(format!(
                "cannot index through non-compound type {current_ty:?}"
            ));
        };
        match compound {
            CompoundType::Ptr(elem) | CompoundType::Array { elem, .. } => {
                let index = resolve_index(function, idx_value, true, value_map, builder)?;
                let elem_size = builder
                    .ins()
                    .iconst(clif::types::I64, i64::from(value_storage_size(elem, ctx)?));
                let offset = builder.ins().imul(index, elem_size);
                addr = builder.ins().iadd(addr, offset);
                current_ty = elem;
            }
            CompoundType::Struct(_) => {
                let idx = constant_value_index(function, idx_value, "gep")?;
                let (offset, field_ty) = aggregate_elem_offset(ctx, current_ty, idx)?;
                addr = builder.ins().iadd_imm_s(addr, i64::from(offset));
                current_ty = field_ty;
            }
            CompoundType::Func { .. }
            | CompoundType::Enum(_)
            | CompoundType::ObjRef(_)
            | CompoundType::ConstRef(_) => {
                return Err(format!("cannot index through {compound:?}"));
            }
        }
    }

    Ok(addr)
}

pub(super) fn resolve_index(
    function: &Function,
    value: ValueId,
    signed: bool,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let ty = function.dfg.value_ty(value);
    let value = resolve_value(function, value, value_map, builder)?;
    let value = if ty == Type::I256 {
        load_i256_limb(value, 0, builder)
    } else {
        value
    };
    let value = if signed {
        signed_scalar(value, ty, builder)
    } else {
        value
    };
    Ok(resize_int_value(value, clif::types::I64, signed, builder))
}

pub(super) fn stack_slot_data(ty: Type, ctx: &ModuleCtx) -> Result<StackSlotData, String> {
    let alignment = value_storage_alignment(ty, ctx)?;
    let align_shift = u8::try_from(alignment.trailing_zeros())
        .map_err(|_| format!("type {ty:?} alignment {alignment} is too large for Cranelift"))?;
    Ok(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        value_storage_size(ty, ctx)?.max(1),
        align_shift,
    ))
}
