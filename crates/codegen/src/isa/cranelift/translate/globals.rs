use std::collections::HashMap;

use cranelift_codegen::ir::{self as clif, InstBuilder};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{DataDescription, DataId, Module as ClifModule};
use sonatina_ir::{GlobalVariableRef, module::ModuleCtx};

use super::{translate_linkage, value_storage_alignment, value_storage_size};
use crate::isa::native_layout::serialize_initializer;

pub(super) type GlobalDataMap = HashMap<GlobalVariableRef, DataId>;

pub(super) fn define_globals(
    ctx: &ModuleCtx,
    module: &mut impl ClifModule,
) -> Result<GlobalDataMap, String> {
    ctx.with_gv_store(|globals| {
        let mut data_ids = GlobalDataMap::new();
        for global in globals.all_gv_refs() {
            let data = globals.gv_data(global);
            let name = &data.symbol;
            let id = module
                .declare_data(name, translate_linkage(data.linkage), !data.is_const, false)
                .map_err(|error| format!("failed to declare global {name}: {error}"))?;
            data_ids.insert(global, id);
            if data.linkage.is_external() {
                if data.initializer.is_some() {
                    return Err(format!(
                        "external global {name} must not have an initializer"
                    ));
                }
                continue;
            }
            let mut description = DataDescription::new();
            description.set_align(
                value_storage_alignment(data.ty, ctx)
                    .map_err(|error| format!("global {name}: {error}"))?,
            );
            let size = value_storage_size(data.ty, ctx)
                .map_err(|error| format!("global {name}: {error}"))?
                as usize;
            if let Some(init) = &data.initializer {
                // Native padding has a deterministic value too. Initializers
                // write fields at their native offsets, after enum legalization.
                let mut bytes = vec![0; size];
                serialize_initializer(init, data.ty, &mut bytes, ctx)
                    .map_err(|error| format!("global {name}: {error}"))?;
                description.define(bytes.into_boxed_slice());
            } else {
                description.define_zeroinit(size);
            }
            module
                .define_data(id, &description)
                .map_err(|error| format!("failed to define global {name}: {error}"))?;
        }
        Ok(data_ids)
    })
}

pub(super) fn global_address(
    global: GlobalVariableRef,
    data_ids: &GlobalDataMap,
    module: &mut impl ClifModule,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let data = data_ids
        .get(&global)
        .ok_or_else(|| format!("undeclared native global {global:?}"))?;
    let reference = module.declare_data_in_func(*data, builder.func);
    Ok(builder
        .ins()
        .symbol_value(module.target_config().pointer_type(), reference))
}
