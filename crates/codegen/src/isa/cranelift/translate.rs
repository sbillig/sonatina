use std::{cmp::Ordering, collections::HashMap};

use cranelift_codegen::ir::{
    self as clif, InstBuilder, MemFlagsData, StackSlotData, StackSlotKind, condcodes::IntCC,
    instructions::BlockArg,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module as ClifModule};

use sonatina_ir::{
    BlockId, ControlFlowGraph, Function, Immediate, Linkage as SonatinaLinkage, Module, Signature,
    Type, Value, ValueId,
    ir_writer::{FuncWriteCtx, InstStatement, IrWrite},
    module::{FuncRef, ModuleCtx},
    types::CompoundType,
};
use sonatina_triple::{Architecture, OperatingSystem, Vendor};

const I256_LIMBS: usize = 4;
const I256_PRODUCT_LIMBS: usize = I256_LIMBS * 2;
const I256_BITS: i64 = 256;
const I256_LIMB_BITS: i64 = 64;

pub(super) fn translate_module(
    module: &Module,
    clif_module: &mut impl ClifModule,
) -> Result<HashMap<String, FuncId>, String> {
    let mut func_map: HashMap<String, FuncId> = HashMap::new();
    let mut func_id_map: HashMap<FuncRef, FuncId> = HashMap::new();

    let funcs = module.funcs();

    for &func_ref in &funcs {
        let (name, sig) = module.ctx.func_sig(func_ref, |sig| -> Result<_, String> {
            validate_cranelift_signature(&module.ctx, sig)?;
            let name = sig.name().to_string();
            let clif_sig = sonatina_sig_to_clif(&module.ctx, sig, clif_module);
            Ok((name, clif_sig))
        })?;

        let linkage = match module.ctx.func_linkage(func_ref) {
            SonatinaLinkage::Public => Linkage::Export,
            SonatinaLinkage::Private => Linkage::Local,
            SonatinaLinkage::External => Linkage::Import,
        };
        let func_id = clif_module
            .declare_function(&name, linkage, &sig)
            .map_err(|e| format!("failed to declare function {name}: {e}"))?;

        func_map.insert(name, func_id);
        func_id_map.insert(func_ref, func_id);
    }

    for &func_ref in &funcs {
        let name = module.ctx.func_sig(func_ref, |sig| sig.name().to_string());
        let translated = module.func_store.try_view(func_ref, |function| {
            if module.ctx.func_linkage(func_ref).is_external() {
                return Ok(());
            }
            if function.layout.entry_block().is_none() {
                return Ok(());
            }
            let func_id = func_id_map[&func_ref];
            translate_function(
                module,
                function,
                func_ref,
                func_id,
                &func_id_map,
                clif_module,
            )
        });
        if let Some(Err(e)) = translated {
            return Err(format!("failed to translate function {name}: {e}"));
        }
    }

    Ok(func_map)
}

fn uses_indirect_return_abi(ctx: &ModuleCtx, ty: Type) -> bool {
    ty == Type::I256
        || matches!(
            ty.resolve_compound(ctx),
            Some(CompoundType::Array { .. } | CompoundType::Struct(_) | CompoundType::Enum(_))
        )
}

fn returns_indirect(ctx: &ModuleCtx, sig: &Signature) -> bool {
    sig.ret_tys().len() == 1 && uses_indirect_return_abi(ctx, sig.ret_tys()[0])
}

fn validate_cranelift_signature(ctx: &ModuleCtx, sig: &Signature) -> Result<(), String> {
    if sig.ret_tys().len() > 1
        && sig
            .ret_tys()
            .iter()
            .any(|ty| uses_indirect_return_abi(ctx, *ty))
    {
        return Err(format!(
            "Cranelift backend does not support multi-return signatures containing indirect return types: {}",
            sig.name()
        ));
    }
    Ok(())
}

fn sonatina_sig_to_clif(
    ctx: &ModuleCtx,
    sig: &Signature,
    clif_module: &impl ClifModule,
) -> clif::Signature {
    let mut clif_sig = clif_module.make_signature();

    // Values represented as pointers to owned storage return through a
    // caller-allocated buffer so the result outlives the callee frame.
    if returns_indirect(ctx, sig) {
        clif_sig.params.push(clif::AbiParam::new(clif::types::I64));
    }

    for &arg_ty in sig.args() {
        if let Some(clif_ty) = sonatina_type_to_clif(arg_ty) {
            clif_sig.params.push(clif::AbiParam::new(clif_ty));
        }
    }

    if returns_indirect(ctx, sig) {
        // Indirect return via hidden sret pointer: no Cranelift return values.
    } else {
        for &ret_ty in sig.ret_tys() {
            if let Some(clif_ty) = sonatina_type_to_clif(ret_ty) {
                clif_sig.returns.push(clif::AbiParam::new(clif_ty));
            }
        }
    }
    clif_sig
}

fn sonatina_type_to_clif(ty: Type) -> Option<clif::Type> {
    match ty {
        Type::Unit => None,
        Type::I1 => Some(clif::types::I8),
        Type::I8 => Some(clif::types::I8),
        Type::I16 => Some(clif::types::I16),
        Type::I32 => Some(clif::types::I32),
        Type::I64 => Some(clif::types::I64),
        Type::I128 => Some(clif::types::I128),
        // I256: represent as pointer to 32 bytes on stack
        Type::I256 => Some(clif::types::I64),
        // Compound types (objref, constref, ptr) → native pointer
        Type::Compound(_) => Some(clif::types::I64),
        _ => None,
    }
}

fn sonatina_type_to_clif_or_err(ty: Type) -> Result<clif::Type, String> {
    sonatina_type_to_clif(ty).ok_or_else(|| format!("unsupported type for cranelift: {ty:?}"))
}

fn translate_function(
    module: &Module,
    function: &Function,
    func_ref: FuncRef,
    func_id: FuncId,
    func_id_map: &HashMap<FuncRef, FuncId>,
    clif_module: &mut impl ClifModule,
) -> Result<(), String> {
    let mut ctx = clif_module.make_context();
    let sig = module.ctx.func_sig(func_ref, |sig| -> Result<_, String> {
        validate_cranelift_signature(&module.ctx, sig)?;
        Ok(sonatina_sig_to_clif(&module.ctx, sig, clif_module))
    })?;
    ctx.func.signature = sig;

    let mut builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

    let mut cfg = ControlFlowGraph::default();
    cfg.compute(function);
    let mut block_order: Vec<_> = cfg.post_order().collect();
    block_order.reverse();

    let mut block_map: HashMap<BlockId, clif::Block> = HashMap::new();
    let mut value_map: HashMap<ValueId, clif::Value> = HashMap::new();
    for &block in &block_order {
        let clif_block = builder.create_block();
        block_map.insert(block, clif_block);
    }

    let has_sret = module
        .ctx
        .func_sig(func_ref, |sig| returns_indirect(&module.ctx, sig));

    let entry = function.layout.entry_block().ok_or("no entry block")?;
    let clif_entry = block_map[&entry];
    builder.append_block_params_for_function_params(clif_entry);
    builder.switch_to_block(clif_entry);

    let sret_ptr = if has_sret {
        Some(builder.block_params(clif_entry)[0])
    } else {
        None
    };

    let arg_offset = if has_sret { 1 } else { 0 };
    for (idx, &arg_value) in function.arg_values.iter().enumerate() {
        let param = builder.block_params(clif_entry)[idx + arg_offset];
        value_map.insert(arg_value, param);
    }

    let inst_set = function.inst_set();

    for &block in &block_order {
        let clif_block = block_map[&block];
        for inst_id in function.layout.iter_inst(block) {
            let inst_data = function.dfg.inst(inst_id);
            if <&sonatina_ir::inst::control_flow::Phi as sonatina_ir::InstDowncast>::downcast(
                inst_set, inst_data,
            )
            .is_some()
            {
                let result = function
                    .dfg
                    .inst_result(inst_id)
                    .ok_or("phi has no result")?;
                let ty = function.dfg.value_ty(result);
                let clif_ty = sonatina_type_to_clif_or_err(ty)?;
                let param = builder.append_block_param(clif_block, clif_ty);
                value_map.insert(result, param);
            } else {
                break;
            }
        }
    }

    // No blanket ISA rejection — the translator handles each instruction
    // individually, emitting intrinsic calls for EVM-specific operations
    // (addmod, mulmod) and errors for truly unsupported ones.

    for &block in &block_order {
        let clif_block = block_map[&block];
        if block != entry {
            builder.switch_to_block(clif_block);
        }

        for inst_id in function.layout.iter_inst(block) {
            let inst_data = function.dfg.inst(inst_id);

            if <&sonatina_ir::inst::control_flow::Phi as sonatina_ir::InstDowncast>::downcast(
                inst_set, inst_data,
            )
            .is_some()
            {
                continue;
            }

            if let Some(add) = <&sonatina_ir::inst::arith::Add as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*add.lhs()) == Type::I256 {
                    emit_i256_add(function, *add.lhs(), *add.rhs(), &value_map, &mut builder)?
                } else {
                    let lhs = resolve_value(function, *add.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *add.rhs(), &value_map, &mut builder)?;
                    builder.ins().iadd(lhs, rhs)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(sub) = <&sonatina_ir::inst::arith::Sub as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*sub.lhs()) == Type::I256 {
                    emit_i256_sub(function, *sub.lhs(), *sub.rhs(), &value_map, &mut builder)?
                } else {
                    let lhs = resolve_value(function, *sub.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *sub.rhs(), &value_map, &mut builder)?;
                    builder.ins().isub(lhs, rhs)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(mul) = <&sonatina_ir::inst::arith::Mul as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*mul.lhs()) == Type::I256 {
                    emit_i256_mul(function, *mul.lhs(), *mul.rhs(), &value_map, &mut builder)?
                } else {
                    let lhs = resolve_value(function, *mul.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *mul.rhs(), &value_map, &mut builder)?;
                    builder.ins().imul(lhs, rhs)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(neg) = <&sonatina_ir::inst::arith::Neg as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*neg.arg()) == Type::I256 {
                    emit_i256_neg(function, *neg.arg(), &value_map, &mut builder)?
                } else {
                    let val = resolve_value(function, *neg.arg(), &value_map, &mut builder)?;
                    builder.ins().ineg(val)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(div) = <&sonatina_ir::inst::arith::Udiv as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*div.lhs()) == Type::I256 {
                    emit_i256_div_rem(
                        function,
                        *div.lhs(),
                        *div.rhs(),
                        I256DivRemKind::Udiv,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *div.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *div.rhs(), &value_map, &mut builder)?;
                    builder.ins().udiv(lhs, rhs)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(div) = <&sonatina_ir::inst::arith::Sdiv as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*div.lhs()) == Type::I256 {
                    emit_i256_div_rem(
                        function,
                        *div.lhs(),
                        *div.rhs(),
                        I256DivRemKind::Sdiv,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *div.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *div.rhs(), &value_map, &mut builder)?;
                    builder.ins().sdiv(lhs, rhs)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(rem) = <&sonatina_ir::inst::arith::Umod as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*rem.lhs()) == Type::I256 {
                    emit_i256_div_rem(
                        function,
                        *rem.lhs(),
                        *rem.rhs(),
                        I256DivRemKind::Umod,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *rem.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *rem.rhs(), &value_map, &mut builder)?;
                    builder.ins().urem(lhs, rhs)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(rem) = <&sonatina_ir::inst::arith::Smod as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*rem.lhs()) == Type::I256 {
                    emit_i256_div_rem(
                        function,
                        *rem.lhs(),
                        *rem.rhs(),
                        I256DivRemKind::Smod,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *rem.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *rem.rhs(), &value_map, &mut builder)?;
                    builder.ins().srem(lhs, rhs)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(shl) = <&sonatina_ir::inst::arith::Shl as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*shl.value()) == Type::I256 {
                    emit_i256_shift(
                        function,
                        *shl.value(),
                        *shl.bits(),
                        I256ShiftKind::Shl,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let val = resolve_value(function, *shl.value(), &value_map, &mut builder)?;
                    let bits = resolve_value(function, *shl.bits(), &value_map, &mut builder)?;
                    builder.ins().ishl(val, bits)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(shr) = <&sonatina_ir::inst::arith::Shr as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*shr.value()) == Type::I256 {
                    emit_i256_shift(
                        function,
                        *shr.value(),
                        *shr.bits(),
                        I256ShiftKind::Shr,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let val = resolve_value(function, *shr.value(), &value_map, &mut builder)?;
                    let bits = resolve_value(function, *shr.bits(), &value_map, &mut builder)?;
                    builder.ins().ushr(val, bits)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(sar) = <&sonatina_ir::inst::arith::Sar as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*sar.value()) == Type::I256 {
                    emit_i256_shift(
                        function,
                        *sar.value(),
                        *sar.bits(),
                        I256ShiftKind::Sar,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let val = resolve_value(function, *sar.value(), &value_map, &mut builder)?;
                    let bits = resolve_value(function, *sar.bits(), &value_map, &mut builder)?;
                    builder.ins().sshr(val, bits)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(and) = <&sonatina_ir::inst::logic::And as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*and.lhs()) == Type::I256 {
                    emit_i256_bitwise(
                        function,
                        *and.lhs(),
                        *and.rhs(),
                        I256BitwiseOp::And,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *and.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *and.rhs(), &value_map, &mut builder)?;
                    builder.ins().band(lhs, rhs)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(or) = <&sonatina_ir::inst::logic::Or as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*or.lhs()) == Type::I256 {
                    emit_i256_bitwise(
                        function,
                        *or.lhs(),
                        *or.rhs(),
                        I256BitwiseOp::Or,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *or.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *or.rhs(), &value_map, &mut builder)?;
                    builder.ins().bor(lhs, rhs)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(xor) = <&sonatina_ir::inst::logic::Xor as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*xor.lhs()) == Type::I256 {
                    emit_i256_bitwise(
                        function,
                        *xor.lhs(),
                        *xor.rhs(),
                        I256BitwiseOp::Xor,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *xor.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *xor.rhs(), &value_map, &mut builder)?;
                    builder.ins().bxor(lhs, rhs)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(not) = <&sonatina_ir::inst::logic::Not as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*not.arg()) == Type::I256 {
                    emit_i256_not(function, *not.arg(), &value_map, &mut builder)?
                } else {
                    let val = resolve_value(function, *not.arg(), &value_map, &mut builder)?;
                    builder.ins().bnot(val)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(lt) = <&sonatina_ir::inst::cmp::Lt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::UnsignedLessThan, *lt.lhs(), *lt.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(gt) = <&sonatina_ir::inst::cmp::Gt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::UnsignedGreaterThan, *gt.lhs(), *gt.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(le) = <&sonatina_ir::inst::cmp::Le as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::UnsignedLessThanOrEqual, *le.lhs(), *le.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(ge) = <&sonatina_ir::inst::cmp::Ge as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::UnsignedGreaterThanOrEqual, *ge.lhs(), *ge.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(slt) = <&sonatina_ir::inst::cmp::Slt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::SignedLessThan, *slt.lhs(), *slt.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(sgt) = <&sonatina_ir::inst::cmp::Sgt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::SignedGreaterThan, *sgt.lhs(), *sgt.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(eq) = <&sonatina_ir::inst::cmp::Eq as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::Equal, *eq.lhs(), *eq.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(ne) = <&sonatina_ir::inst::cmp::Ne as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::NotEqual, *ne.lhs(), *ne.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(is_zero) = <&sonatina_ir::inst::cmp::IsZero as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val_ty = function.dfg.value_ty(*is_zero.lhs());
                let result_val = if val_ty == Type::I256 {
                    let val = resolve_value(function, *is_zero.lhs(), &value_map, &mut builder)?;
                    emit_i256_is_zero(val, &mut builder)
                } else {
                    let val = resolve_value(function, *is_zero.lhs(), &value_map, &mut builder)?;
                    let clif_ty = sonatina_type_to_clif(val_ty).unwrap_or(clif::types::I64);
                    let zero = builder.ins().iconst(clif_ty, 0);
                    builder.ins().icmp(IntCC::Equal, val, zero)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(sle) = <&sonatina_ir::inst::cmp::Sle as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::SignedLessThanOrEqual, *sle.lhs(), *sle.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(sge) = <&sonatina_ir::inst::cmp::Sge as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                translate_icmp(IntCC::SignedGreaterThanOrEqual, *sge.lhs(), *sge.rhs(), inst_id, module, function, &mut value_map, &mut builder)?;
            } else if let Some(sext) = <&sonatina_ir::inst::cast::Sext as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *sext.from(), &value_map, &mut builder)?;
                let to_ty = sonatina_type_to_clif_or_err(*sext.ty())?;
                let result_val = if *sext.ty() == Type::I256 {
                    materialize_scalar_as_i256(
                        val,
                        function.dfg.value_ty(*sext.from()),
                        true,
                        &mut builder,
                    )
                } else if function.dfg.value_ty(*sext.from()) == Type::I1 {
                    bool_to_int_value(val, to_ty, &mut builder)
                } else {
                    resize_int_value(val, to_ty, true, &mut builder)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(zext) = <&sonatina_ir::inst::cast::Zext as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *zext.from(), &value_map, &mut builder)?;
                let to_ty = sonatina_type_to_clif_or_err(*zext.ty())?;
                let result_val = if *zext.ty() == Type::I256 {
                    materialize_scalar_as_i256(
                        val,
                        function.dfg.value_ty(*zext.from()),
                        false,
                        &mut builder,
                    )
                } else if function.dfg.value_ty(*zext.from()) == Type::I1 {
                    bool_to_int_value(val, to_ty, &mut builder)
                } else {
                    resize_int_value(val, to_ty, false, &mut builder)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(trunc) = <&sonatina_ir::inst::cast::Trunc as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let from_ty = function.dfg.value_ty(*trunc.from());
                let val = resolve_value(function, *trunc.from(), &value_map, &mut builder)?;
                let to_ty = sonatina_type_to_clif_or_err(*trunc.ty())?;
                let result_val = if from_ty == Type::I256 {
                    // i256 values are pointers — load the target-sized value from the pointer
                    builder.ins().load(to_ty, MemFlagsData::new(), val, 0)
                } else {
                    builder.ins().ireduce(to_ty, val)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(bitcast) = <&sonatina_ir::inst::cast::Bitcast as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *bitcast.from(), &value_map, &mut builder)?;
                let to_ty = sonatina_type_to_clif_or_err(*bitcast.ty())?;
                let result_val = translate_bitcast(val, to_ty, &mut builder)?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(cast) = <&sonatina_ir::inst::cast::IntToPtr as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *cast.from(), &value_map, &mut builder)?;
                let to_ty = sonatina_type_to_clif_or_err(*cast.ty())?;
                let result_val = if function.dfg.value_ty(*cast.from()) == Type::I256 {
                    let scalar = load_i256_limb(val, 0, &mut builder);
                    resize_int_value(scalar, to_ty, false, &mut builder)
                } else {
                    resize_int_value(val, to_ty, false, &mut builder)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(cast) = <&sonatina_ir::inst::cast::PtrToInt as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let val = resolve_value(function, *cast.from(), &value_map, &mut builder)?;
                let to_ty = sonatina_type_to_clif_or_err(*cast.ty())?;
                let result_val = if *cast.ty() == Type::I256 {
                    materialize_scalar_as_i256(
                        val,
                        function.dfg.value_ty(*cast.from()),
                        false,
                        &mut builder,
                    )
                } else {
                    resize_int_value(val, to_ty, false, &mut builder)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(jump) = <&sonatina_ir::inst::control_flow::Jump as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let dest = block_map[jump.dest()];
                let phi_args = collect_phi_args_for_block(function, *jump.dest(), block, inst_set, &value_map, &mut builder)?;
                builder.ins().jump(dest, &phi_args);
            } else if let Some(br) = <&sonatina_ir::inst::control_flow::Br as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let cond = resolve_value(function, *br.cond(), &value_map, &mut builder)?;
                let nz_block = block_map[br.nz_dest()];
                let z_block = block_map[br.z_dest()];
                let nz_args = collect_phi_args_for_block(function, *br.nz_dest(), block, inst_set, &value_map, &mut builder)?;
                let z_args = collect_phi_args_for_block(function, *br.z_dest(), block, inst_set, &value_map, &mut builder)?;
                builder.ins().brif(cond, nz_block, &nz_args, z_block, &z_args);
            } else if let Some(ret) = <&sonatina_ir::inst::control_flow::Return as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if let Some(sret) = sret_ptr {
                    for &val_id in ret.args().as_slice() {
                        let val = resolve_value(function, val_id, &value_map, &mut builder)?;
                        let val_ty = function.dfg.value_ty(val_id);
                        copy_bytes(val, sret, compute_alloc_size(val_ty, &module.ctx), &mut builder);
                    }
                    builder.ins().return_(&[]);
                } else {
                    let args: Result<Vec<_>, _> = ret
                        .args()
                        .as_slice()
                        .iter()
                        .map(|v| resolve_value(function, *v, &value_map, &mut builder))
                        .collect();
                    let args = args?;
                    builder.ins().return_(&args);
                }
            } else if let Some(call) = <&sonatina_ir::inst::control_flow::Call as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let callee = *call.callee();
                let runtime_intrinsic = module.ctx.func_sig(callee, |sig| {
                    runtime_u256_intrinsic_for_external_name(
                        sig.name(),
                        module.ctx.func_linkage(callee),
                    )
                });
                if let Some(intrinsic_name) = runtime_intrinsic {
                    let args: Result<Vec<_>, _> = call.args()
                        .iter()
                        .map(|v| resolve_value(function, *v, &value_map, &mut builder))
                        .collect();
                    // Args are already pointers to 32-byte u256 buffers
                    // (from obj.load passthrough or emit_i256_immediate stack slots)
                    let result_val = emit_u256_intrinsic_call(
                        clif_module, &mut builder, intrinsic_name, &args?, true,
                    )?;
                    let ir_results = function.dfg.inst_results(inst_id);
                    if !ir_results.is_empty() {
                        value_map.insert(ir_results[0], result_val);
                    }
                } else {
                    let clif_func_id = func_id_map.get(&callee)
                        .ok_or_else(|| format!("unknown callee {:?}", callee))?;
                    let clif_func_ref = clif_module.declare_func_in_func(*clif_func_id, builder.func);
                    if uses_static_external_calls(module, callee) {
                        builder.func.dfg.ext_funcs[clif_func_ref].colocated = true;
                    }
                    let ir_results = function.dfg.inst_results(inst_id);
                    let callee_returns_indirect = ir_results.len() == 1
                        && uses_indirect_return_abi(&module.ctx, function.dfg.value_ty(ir_results[0]));

                    let mut call_args: Vec<clif::Value> = Vec::new();
                    let sret_slot = if callee_returns_indirect {
                        let result_ty = function.dfg.value_ty(ir_results[0]);
                        let slot = builder.create_sized_stack_slot(
                            cranelift_codegen::ir::StackSlotData::new(
                                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                compute_alloc_size(result_ty, &module.ctx),
                                0,
                            ),
                        );
                        let addr = builder.ins().stack_addr(clif::types::I64, slot, 0);
                        call_args.push(addr);
                        Some(addr)
                    } else {
                        None
                    };

                    let args: Result<Vec<_>, _> = call.args()
                        .iter()
                        .map(|v| resolve_value(function, *v, &value_map, &mut builder))
                        .collect();
                    call_args.extend(args?);

                    let clif_call = builder.ins().call(clif_func_ref, &call_args);

                    if let Some(sret_addr) = sret_slot {
                        // Result is in the sret buffer we allocated
                        if !ir_results.is_empty() {
                            value_map.insert(ir_results[0], sret_addr);
                        }
                    } else {
                        let results = builder.inst_results(clif_call).to_vec();
                        for (ir_result, clif_result) in ir_results.iter().zip(results.iter()) {
                            value_map.insert(*ir_result, *clif_result);
                        }
                    }
                }
            } else if let Some(uaddo) = <&sonatina_ir::inst::arith::Uaddo as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if function.dfg.value_ty(*uaddo.lhs()) == Type::I256 {
                    let (result_val, overflow) = emit_i256_uaddo(
                        function,
                        *uaddo.lhs(),
                        *uaddo.rhs(),
                        &value_map,
                        &mut builder,
                    )?;
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                } else {
                    let lhs = resolve_value(function, *uaddo.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *uaddo.rhs(), &value_map, &mut builder)?;
                    let (result_val, overflow) = builder.ins().uadd_overflow(lhs, rhs);
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                }
            } else if let Some(saddo) = <&sonatina_ir::inst::arith::Saddo as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if function.dfg.value_ty(*saddo.lhs()) == Type::I256 {
                    let (result_val, overflow) = emit_i256_saddo(
                        function,
                        *saddo.lhs(),
                        *saddo.rhs(),
                        &value_map,
                        &mut builder,
                    )?;
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                } else {
                    let lhs = resolve_value(function, *saddo.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *saddo.rhs(), &value_map, &mut builder)?;
                    let (result_val, overflow) = builder.ins().sadd_overflow(lhs, rhs);
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                }
            } else if let Some(usubo) = <&sonatina_ir::inst::arith::Usubo as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if function.dfg.value_ty(*usubo.lhs()) == Type::I256 {
                    let (result_val, overflow) = emit_i256_usubo(
                        function,
                        *usubo.lhs(),
                        *usubo.rhs(),
                        &value_map,
                        &mut builder,
                    )?;
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                } else {
                    let lhs = resolve_value(function, *usubo.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *usubo.rhs(), &value_map, &mut builder)?;
                    let (result_val, overflow) = builder.ins().usub_overflow(lhs, rhs);
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                }
            } else if let Some(ssubo) = <&sonatina_ir::inst::arith::Ssubo as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if function.dfg.value_ty(*ssubo.lhs()) == Type::I256 {
                    let (result_val, overflow) = emit_i256_ssubo(
                        function,
                        *ssubo.lhs(),
                        *ssubo.rhs(),
                        &value_map,
                        &mut builder,
                    )?;
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                } else {
                    let lhs = resolve_value(function, *ssubo.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *ssubo.rhs(), &value_map, &mut builder)?;
                    let (result_val, overflow) = builder.ins().ssub_overflow(lhs, rhs);
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                }
            } else if let Some(umulo) = <&sonatina_ir::inst::arith::Umulo as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if function.dfg.value_ty(*umulo.lhs()) == Type::I256 {
                    let (result_val, overflow) = emit_i256_umulo(
                        function,
                        *umulo.lhs(),
                        *umulo.rhs(),
                        &value_map,
                        &mut builder,
                    )?;
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                } else {
                    let lhs = resolve_value(function, *umulo.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *umulo.rhs(), &value_map, &mut builder)?;
                    let (result_val, overflow) = builder.ins().umul_overflow(lhs, rhs);
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                }
            } else if let Some(smulo) = <&sonatina_ir::inst::arith::Smulo as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if function.dfg.value_ty(*smulo.lhs()) == Type::I256 {
                    let (result_val, overflow) = emit_i256_smulo(
                        function,
                        *smulo.lhs(),
                        *smulo.rhs(),
                        &value_map,
                        &mut builder,
                    )?;
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                } else {
                    let lhs = resolve_value(function, *smulo.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *smulo.rhs(), &value_map, &mut builder)?;
                    let (result_val, overflow) = builder.ins().smul_overflow(lhs, rhs);
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                }
            } else if let Some(snego) = <&sonatina_ir::inst::arith::Snego as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if function.dfg.value_ty(*snego.arg()) == Type::I256 {
                    let (result_val, overflow) =
                        emit_i256_snego(function, *snego.arg(), &value_map, &mut builder)?;
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                } else {
                    let val = resolve_value(function, *snego.arg(), &value_map, &mut builder)?;
                    let result_val = builder.ins().ineg(val);
                    let ty = builder.func.dfg.value_type(val);
                    let min = signed_min_value(ty, &mut builder)?;
                    let overflow = builder.ins().icmp(IntCC::Equal, val, min);
                    insert_clif_results(function, inst_id, [result_val, overflow], &mut value_map);
                }
            } else if let Some(uaddsat) = <&sonatina_ir::inst::arith::Uaddsat as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*uaddsat.lhs()) == Type::I256 {
                    emit_i256_saturating_binary(
                        function,
                        *uaddsat.lhs(),
                        *uaddsat.rhs(),
                        I256SaturatingOp::Uadd,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *uaddsat.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *uaddsat.rhs(), &value_map, &mut builder)?;
                    let (raw, overflow) = builder.ins().uadd_overflow(lhs, rhs);
                    let max = unsigned_max_value(builder.func.dfg.value_type(lhs), &mut builder);
                    builder.ins().select(overflow, max, raw)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(saddsat) = <&sonatina_ir::inst::arith::Saddsat as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*saddsat.lhs()) == Type::I256 {
                    emit_i256_saturating_binary(
                        function,
                        *saddsat.lhs(),
                        *saddsat.rhs(),
                        I256SaturatingOp::Sadd,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *saddsat.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *saddsat.rhs(), &value_map, &mut builder)?;
                    let (raw, overflow) = builder.ins().sadd_overflow(lhs, rhs);
                    let ty = builder.func.dfg.value_type(lhs);
                    let zero = builder.ins().iconst(ty, 0);
                    let lhs_neg = builder.ins().icmp(IntCC::SignedLessThan, lhs, zero);
                    let min = signed_min_value(ty, &mut builder)?;
                    let max = signed_max_value(ty, &mut builder)?;
                    let sat = builder.ins().select(lhs_neg, min, max);
                    builder.ins().select(overflow, sat, raw)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(usubsat) = <&sonatina_ir::inst::arith::Usubsat as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*usubsat.lhs()) == Type::I256 {
                    emit_i256_saturating_binary(
                        function,
                        *usubsat.lhs(),
                        *usubsat.rhs(),
                        I256SaturatingOp::Usub,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *usubsat.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *usubsat.rhs(), &value_map, &mut builder)?;
                    let (raw, overflow) = builder.ins().usub_overflow(lhs, rhs);
                    let ty = builder.func.dfg.value_type(lhs);
                    let zero = builder.ins().iconst(ty, 0);
                    builder.ins().select(overflow, zero, raw)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(ssubsat) = <&sonatina_ir::inst::arith::Ssubsat as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*ssubsat.lhs()) == Type::I256 {
                    emit_i256_saturating_binary(
                        function,
                        *ssubsat.lhs(),
                        *ssubsat.rhs(),
                        I256SaturatingOp::Ssub,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *ssubsat.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *ssubsat.rhs(), &value_map, &mut builder)?;
                    let (raw, overflow) = builder.ins().ssub_overflow(lhs, rhs);
                    let ty = builder.func.dfg.value_type(lhs);
                    let zero = builder.ins().iconst(ty, 0);
                    let lhs_neg = builder.ins().icmp(IntCC::SignedLessThan, lhs, zero);
                    let min = signed_min_value(ty, &mut builder)?;
                    let max = signed_max_value(ty, &mut builder)?;
                    let sat = builder.ins().select(lhs_neg, min, max);
                    builder.ins().select(overflow, sat, raw)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(umulsat) = <&sonatina_ir::inst::arith::Umulsat as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*umulsat.lhs()) == Type::I256 {
                    emit_i256_saturating_binary(
                        function,
                        *umulsat.lhs(),
                        *umulsat.rhs(),
                        I256SaturatingOp::Umul,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *umulsat.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *umulsat.rhs(), &value_map, &mut builder)?;
                    let (raw, overflow) = builder.ins().umul_overflow(lhs, rhs);
                    let max = unsigned_max_value(builder.func.dfg.value_type(lhs), &mut builder);
                    builder.ins().select(overflow, max, raw)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(smulsat) = <&sonatina_ir::inst::arith::Smulsat as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let result_val = if function.dfg.value_ty(*smulsat.lhs()) == Type::I256 {
                    emit_i256_saturating_binary(
                        function,
                        *smulsat.lhs(),
                        *smulsat.rhs(),
                        I256SaturatingOp::Smul,
                        &value_map,
                        &mut builder,
                    )?
                } else {
                    let lhs = resolve_value(function, *smulsat.lhs(), &value_map, &mut builder)?;
                    let rhs = resolve_value(function, *smulsat.rhs(), &value_map, &mut builder)?;
                    let (raw, overflow) = builder.ins().smul_overflow(lhs, rhs);
                    let ty = builder.func.dfg.value_type(lhs);
                    let zero = builder.ins().iconst(ty, 0);
                    let lhs_neg = builder.ins().icmp(IntCC::SignedLessThan, lhs, zero);
                    let rhs_neg = builder.ins().icmp(IntCC::SignedLessThan, rhs, zero);
                    let same_sign = builder.ins().icmp(IntCC::Equal, lhs_neg, rhs_neg);
                    let min = signed_min_value(ty, &mut builder)?;
                    let max = signed_max_value(ty, &mut builder)?;
                    let sat = builder.ins().select(same_sign, max, min);
                    builder.ins().select(overflow, sat, raw)
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(obj_load) = <&sonatina_ir::inst::data::ObjLoad as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let addr = resolve_value(function, *obj_load.object(), &value_map, &mut builder)?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let result_ty = function.dfg.value_ty(result);
                    if result_ty == Type::I256 || matches!(result_ty, Type::Compound(_)) {
                        // For i256/struct: passthrough the pointer
                        value_map.insert(result, addr);
                    } else {
                        let clif_ty = sonatina_type_to_clif_or_err(result_ty)?;
                        let loaded = builder.ins().load(clif_ty, MemFlagsData::new(), addr, 0);
                        value_map.insert(result, loaded);
                    }
                }
            } else if let Some(extract) = <&sonatina_ir::inst::data::ExtractValue as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let base = resolve_value(function, *extract.dest(), &value_map, &mut builder)?;
                let idx = constant_value_index(function, *extract.idx(), "extract_value")?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let result_ty = function.dfg.value_ty(result);
                    let dest_ty = function.dfg.value_ty(*extract.dest());
                    let (offset, elem_ty) = aggregate_elem_offset(&module.ctx, dest_ty, idx)?;
                    if result_ty != elem_ty {
                        return Err(format!(
                            "extract_value element type mismatch: expected {elem_ty:?}, got {result_ty:?}"
                        ));
                    }
                    if result_ty == Type::I256 || matches!(result_ty, Type::Compound(_)) {
                        let addr = builder.ins().iadd_imm(base, offset as i64);
                        value_map.insert(result, addr);
                    } else {
                        let clif_ty = sonatina_type_to_clif_or_err(result_ty)?;
                        let loaded = builder.ins().load(clif_ty, MemFlagsData::new(), base, offset);
                        value_map.insert(result, loaded);
                    }
                }
            } else if let Some(insert) = <&sonatina_ir::inst::data::InsertValue as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let Some(result) = function.dfg.inst_result(inst_id) else {
                    continue;
                };
                let result_ty = function.dfg.value_ty(result);
                let result_addr = create_stack_slot_for_type(result_ty, &module.ctx, &mut builder);
                if !is_undef_value(function, *insert.dest()) {
                    let source = resolve_value(function, *insert.dest(), &value_map, &mut builder)?;
                    copy_bytes(source, result_addr, compute_alloc_size(result_ty, &module.ctx), &mut builder);
                }

                let idx = constant_value_index(function, *insert.idx(), "insert_value")?;
                let (offset, elem_ty) = aggregate_elem_offset(&module.ctx, result_ty, idx)?;
                let value_ty = function.dfg.value_ty(*insert.value());
                if value_ty != elem_ty {
                    return Err(format!(
                        "insert_value element type mismatch: expected {elem_ty:?}, got {value_ty:?}"
                    ));
                }

                let value = resolve_value(function, *insert.value(), &value_map, &mut builder)?;
                if value_ty == Type::I256 || matches!(value_ty, Type::Compound(_)) {
                    let field_addr = builder.ins().iadd_imm(result_addr, i64::from(offset));
                    copy_bytes(value, field_addr, compute_alloc_size(value_ty, &module.ctx), &mut builder);
                } else {
                    builder.ins().store(MemFlagsData::new(), value, result_addr, offset);
                }
                value_map.insert(result, result_addr);
            } else if <&sonatina_ir::inst::data::Alloca as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data).is_some() {
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 32, 0,
                        ),
                    );
                    let addr = builder.ins().stack_addr(clif::types::I64, slot, 0);
                    value_map.insert(result, addr);
                }
            } else if let Some(mstore) = <&sonatina_ir::inst::data::Mstore as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let addr = resolve_address_value(module, function, *mstore.addr(), &value_map, &mut builder)?;
                let val = resolve_value(function, *mstore.value(), &value_map, &mut builder)?;
                let store_ty = function.dfg.value_ty(*mstore.value());
                if store_ty == Type::I256 {
                    copy_i256(val, addr, &mut builder);
                } else {
                    builder.ins().store(MemFlagsData::new(), val, addr, 0);
                }
            } else if let Some(mload) = <&sonatina_ir::inst::data::Mload as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let addr = resolve_address_value(module, function, *mload.addr(), &value_map, &mut builder)?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let result_ty = function.dfg.value_ty(result);
                    if result_ty == Type::I256 {
                        let result_addr = create_i256_slot(&mut builder);
                        copy_i256(addr, result_addr, &mut builder);
                        value_map.insert(result, result_addr);
                    } else {
                        let clif_ty = sonatina_type_to_clif_or_err(result_ty)?;
                        let loaded = builder.ins().load(clif_ty, MemFlagsData::new(), addr, 0);
                        value_map.insert(result, loaded);
                    }
                }
            } else if let Some(addmod) = <&sonatina_ir::inst::evm::EvmAddMod as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let a = resolve_value(function, *addmod.lhs(), &value_map, &mut builder)?;
                let b = resolve_value(function, *addmod.rhs(), &value_map, &mut builder)?;
                let m = resolve_value(function, *addmod.modulus(), &value_map, &mut builder)?;
                let result_val = emit_u256_intrinsic_call(
                    clif_module, &mut builder, "__u256_addmod",
                    &[a, b, m], true,
                )?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(mulmod) = <&sonatina_ir::inst::evm::EvmMulMod as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let a = resolve_value(function, *mulmod.lhs(), &value_map, &mut builder)?;
                let b = resolve_value(function, *mulmod.rhs(), &value_map, &mut builder)?;
                let m = resolve_value(function, *mulmod.modulus(), &value_map, &mut builder)?;
                let result_val = emit_u256_intrinsic_call(
                    clif_module, &mut builder, "__u256_mulmod",
                    &[a, b, m], true,
                )?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if let Some(obj_store) = <&sonatina_ir::inst::data::ObjStore as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let dest = resolve_value(function, *obj_store.object(), &value_map, &mut builder)?;
                let val = resolve_value(function, *obj_store.value(), &value_map, &mut builder)?;
                let val_ty = function.dfg.value_ty(*obj_store.value());
                if val_ty == Type::I256 {
                    copy_i256(val, dest, &mut builder);
                } else {
                    builder.ins().store(MemFlagsData::new(), val, dest, 0);
                }
            } else if <&sonatina_ir::inst::data::ObjAlloc as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data).is_some() {
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let result_ty = function.dfg.value_ty(result);
                    let alloc_size = compute_alloc_size(result_ty, &module.ctx);
                    let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot, alloc_size, 0,
                    ));
                    let addr = builder.ins().stack_addr(clif::types::I64, slot, 0);
                    value_map.insert(result, addr);
                }
            } else if let Some(obj_init_const) = <&sonatina_ir::inst::data::ObjInitConst as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let object = resolve_value(function, *obj_init_const.object(), &value_map, &mut builder)?;
                let value = resolve_value(function, *obj_init_const.value(), &value_map, &mut builder)?;
                let object_ty = function.dfg.value_ty(*obj_init_const.object());
                copy_bytes(
                    value,
                    object,
                    compute_alloc_size(object_ty, &module.ctx),
                    &mut builder,
                );
            } else if let Some(obj_proj) = <&sonatina_ir::inst::data::ObjProj as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let addr = translate_aggregate_projection(
                        &module.ctx,
                        function,
                        obj_proj.values(),
                        "obj.proj",
                        &value_map,
                        &mut builder,
                    )?;
                    value_map.insert(result, addr);
                }
            } else if let Some(obj_index) = <&sonatina_ir::inst::data::ObjIndex as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let base = resolve_value(function, *obj_index.object(), &value_map, &mut builder)?;
                let index_val_id = *obj_index.index();
                let index_ty = function.dfg.value_ty(index_val_id);
                let index = if index_ty == Type::I256 {
                    if let Some(imm) = function.dfg.value_imm(index_val_id) {
                        let idx_i64 = match imm {
                            Immediate::I256(v) => {
                                let u = v.to_u256();
                                u.low_u64() as i64
                            }
                            _ => 0,
                        };
                        builder.ins().iconst(clif::types::I64, idx_i64)
                    } else {
                        let raw = resolve_value(function, index_val_id, &value_map, &mut builder)?;
                        builder.ins().load(clif::types::I64, MemFlagsData::new(), raw, 0)
                    }
                } else {
                    resolve_scalar_value(module, function, index_val_id, &value_map, &mut builder)?
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let obj_ty = function.dfg.value_ty(*obj_index.object());
                    let elem_size = compute_element_size(obj_ty, &module.ctx);
                    let stride = builder.ins().iconst(clif::types::I64, elem_size as i64);
                    let offset = builder.ins().imul(index, stride);
                    let addr = builder.ins().iadd(base, offset);
                    value_map.insert(result, addr);
                }
            } else if let Some(evm_umod) = <&sonatina_ir::inst::evm::EvmUmod as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let lhs = resolve_value(function, *evm_umod.lhs(), &value_map, &mut builder)?;
                let rhs = resolve_value(function, *evm_umod.rhs(), &value_map, &mut builder)?;
                let result_val = builder.ins().urem(lhs, rhs);
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    value_map.insert(result, result_val);
                }
            } else if <&sonatina_ir::inst::evm::EvmRevert as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data).is_some() {
                builder.ins().trap(cranelift_codegen::ir::TrapCode::user(2).unwrap());
            } else if <&sonatina_ir::inst::evm::EvmStop as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data).is_some() {
                builder.ins().return_(&[]);
            } else if let Some(const_ref) = <&sonatina_ir::inst::data::ConstRef as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let gv_ref = const_ref.global().gv();
                    let result_ty = function.dfg.value_ty(result);
                    let data_size = compute_alloc_size(result_ty, &module.ctx);
                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, data_size, 0,
                        ),
                    );
                    let addr = builder.ins().stack_addr(clif::types::I64, slot, 0);
                    let init_data = module.ctx.with_gv_store(|store| store.init_data(gv_ref).cloned());
                    if let Some(init) = init_data {
                        let gv_ty = module.ctx.with_gv_store(|store| store.ty(gv_ref));
                        materialize_gv_initializer(&init, gv_ty, addr, 0, &module.ctx, &mut builder);
                    }
                    value_map.insert(result, addr);
                }
            } else if let Some(const_proj) = <&sonatina_ir::inst::data::ConstProj as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let addr = translate_aggregate_projection(
                        &module.ctx,
                        function,
                        const_proj.values(),
                        "const.proj",
                        &value_map,
                        &mut builder,
                    )?;
                    value_map.insert(result, addr);
                }
            } else if let Some(const_index) = <&sonatina_ir::inst::data::ConstIndex as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let base = resolve_value(function, *const_index.object(), &value_map, &mut builder)?;
                let index_val_id = *const_index.index();
                let index_ty = function.dfg.value_ty(index_val_id);
                let index = if index_ty == Type::I256 {
                    if let Some(imm) = function.dfg.value_imm(index_val_id) {
                        let idx = match imm {
                            Immediate::I256(v) => v.to_u256().low_u64() as i64,
                            _ => 0,
                        };
                        builder.ins().iconst(clif::types::I64, idx)
                    } else {
                        let raw = resolve_value(function, index_val_id, &value_map, &mut builder)?;
                        builder.ins().load(clif::types::I64, MemFlagsData::new(), raw, 0)
                    }
                } else {
                    resolve_scalar_value(module, function, index_val_id, &value_map, &mut builder)?
                };
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let obj_ty = function.dfg.value_ty(*const_index.object());
                    let elem_size = compute_element_size(obj_ty, &module.ctx);
                    let stride = builder.ins().iconst(clif::types::I64, elem_size as i64);
                    let offset = builder.ins().imul(index, stride);
                    let ptr = builder.ins().iadd(base, offset);
                    value_map.insert(result, ptr);
                }
            } else if let Some(const_load) = <&sonatina_ir::inst::data::ConstLoad as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data) {
                let addr = resolve_value(function, *const_load.object(), &value_map, &mut builder)?;
                if let Some(result) = function.dfg.inst_result(inst_id) {
                    let result_ty = function.dfg.value_ty(result);
                    if result_ty == Type::I256 || matches!(result_ty, Type::Compound(_)) {
                        value_map.insert(result, addr);
                    } else {
                        let clif_ty = sonatina_type_to_clif_or_err(result_ty)?;
                        let loaded = builder.ins().load(clif_ty, MemFlagsData::new(), addr, 0);
                        value_map.insert(result, loaded);
                    }
                }
            } else if <&sonatina_ir::inst::control_flow::Unreachable as sonatina_ir::InstDowncast>::downcast(inst_set, inst_data).is_some() {
                builder.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
            } else {
                let mut text = Vec::new();
                let _ = InstStatement(inst_id).write(
                    &mut text,
                    &FuncWriteCtx::new(function, func_ref),
                );
                let text = String::from_utf8_lossy(&text);
                return Err(format!(
                    "unsupported instruction for CraneliftBackend: {:?}: {}",
                    inst_data.kind(),
                    text.trim(),
                ));
            }
        }
    }

    builder.seal_all_blocks();
    builder.finalize();

    if std::env::var("DUMP_CLIF").is_ok() {
        let name = module.ctx.func_sig(func_ref, |sig| sig.name().to_string());
        eprintln!("[cranelift] CLIF IR for {name}:\n{}", ctx.func.display());
    }

    if let Err(e) = clif_module.define_function(func_id, &mut ctx) {
        eprintln!("[cranelift] CLIF IR (error):\n{}", ctx.func.display());
        return Err(format!("cranelift define_function failed: {e}"));
    }

    Ok(())
}

fn uses_static_external_calls(module: &Module, callee: FuncRef) -> bool {
    let triple = module.ctx.triple;
    module.ctx.func_linkage(callee).is_external()
        && matches!(
            (triple.architecture, triple.vendor, triple.operating_system),
            (
                Architecture::Riscv32im | Architecture::Riscv64im,
                Vendor::Succinct,
                OperatingSystem::ZkvmElf
            )
        )
}

fn runtime_u256_intrinsic_for_external_name(
    name: &str,
    linkage: SonatinaLinkage,
) -> Option<&'static str> {
    if !linkage.is_external() {
        return None;
    }

    match runtime_intrinsic_symbol_basename(name) {
        "addmod" => Some("__u256_addmod"),
        "mulmod" => Some("__u256_mulmod"),
        _ => None,
    }
}

fn runtime_intrinsic_symbol_basename(name: &str) -> &str {
    let component = name.rsplit("__").next().unwrap_or(name);
    if let Some((base, suffix)) = component.rsplit_once('_')
        && !suffix.is_empty()
        && suffix.chars().all(|ch| ch.is_ascii_hexdigit())
    {
        return base;
    }
    component
}

fn resolve_scalar_value(
    module: &Module,
    function: &Function,
    value_id: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let ty = function.dfg.value_ty(value_id);
    let val = resolve_value(function, value_id, value_map, builder)?;
    if ty.is_obj_ref(&module.ctx)
        && let Some(sonatina_ir::types::CompoundType::ObjRef(elem)) =
            ty.resolve_compound(&module.ctx)
        && let Some(clif_ty) = sonatina_type_to_clif(elem)
    {
        return Ok(builder.ins().load(clif_ty, MemFlagsData::new(), val, 0));
    }
    Ok(val)
}

fn resolve_address_value(
    module: &Module,
    function: &Function,
    value_id: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let val = resolve_scalar_value(module, function, value_id, value_map, builder)?;
    if function.dfg.value_ty(value_id) == Type::I256 {
        Ok(load_i256_limb(val, 0, builder))
    } else {
        Ok(val)
    }
}

fn create_i256_slot(builder: &mut FunctionBuilder) -> clif::Value {
    let slot =
        builder.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 32, 0));
    builder.ins().stack_addr(clif::types::I64, slot, 0)
}

fn create_stack_slot_for_type(
    ty: Type,
    ctx: &ModuleCtx,
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let slot = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        compute_alloc_size(ty, ctx),
        0,
    ));
    builder.ins().stack_addr(clif::types::I64, slot, 0)
}

fn load_i256_limb(addr: clif::Value, limb: usize, builder: &mut FunctionBuilder) -> clif::Value {
    builder.ins().load(
        clif::types::I64,
        MemFlagsData::new(),
        addr,
        (limb * 8) as i32,
    )
}

fn store_i256_limb(
    addr: clif::Value,
    limb: usize,
    value: clif::Value,
    builder: &mut FunctionBuilder,
) {
    builder
        .ins()
        .store(MemFlagsData::new(), value, addr, (limb * 8) as i32);
}

fn copy_i256(src: clif::Value, dst: clif::Value, builder: &mut FunctionBuilder) {
    for limb in 0..I256_LIMBS {
        let value = load_i256_limb(src, limb, builder);
        store_i256_limb(dst, limb, value, builder);
    }
}

fn copy_bytes(src: clif::Value, dst: clif::Value, size: u32, builder: &mut FunctionBuilder) {
    let mut offset = 0;
    for (chunk, ty) in [
        (8, clif::types::I64),
        (4, clif::types::I32),
        (2, clif::types::I16),
        (1, clif::types::I8),
    ] {
        while offset + chunk <= size {
            let value = builder
                .ins()
                .load(ty, MemFlagsData::new(), src, offset as i32);
            builder
                .ins()
                .store(MemFlagsData::new(), value, dst, offset as i32);
            offset += chunk;
        }
    }
}

fn materialize_scalar_as_i256(
    value: clif::Value,
    source_ty: Type,
    signed: bool,
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let result = create_i256_slot(builder);
    let value_ty = builder.func.dfg.value_type(value);
    let low = if !value_ty.is_int() {
        bool_to_int_value(value, clif::types::I64, builder)
    } else {
        resize_int_value(value, clif::types::I64, signed, builder)
    };
    store_i256_limb(result, 0, low, builder);

    let zero = builder.ins().iconst(clif::types::I64, 0);
    let fill = if signed {
        let sign = if !value_ty.is_int() || source_ty == Type::I1 {
            bool_const(false, builder)
        } else {
            let sign_zero = builder.ins().iconst(value_ty, 0);
            builder.ins().icmp(IntCC::SignedLessThan, value, sign_zero)
        };
        let minus_one = builder.ins().iconst(clif::types::I64, -1);
        builder.ins().select(sign, minus_one, zero)
    } else {
        zero
    };

    for limb in 1..I256_LIMBS {
        store_i256_limb(result, limb, fill, builder);
    }
    result
}

fn load_i256_limbs(value: clif::Value, builder: &mut FunctionBuilder) -> [clif::Value; I256_LIMBS] {
    [
        load_i256_limb(value, 0, builder),
        load_i256_limb(value, 1, builder),
        load_i256_limb(value, 2, builder),
        load_i256_limb(value, 3, builder),
    ]
}

fn store_i256_limbs(
    limbs: [clif::Value; I256_LIMBS],
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let result = create_i256_slot(builder);
    for (limb_idx, limb) in limbs.into_iter().enumerate() {
        store_i256_limb(result, limb_idx, limb, builder);
    }
    result
}

fn zero_i256_limbs(builder: &mut FunctionBuilder) -> [clif::Value; I256_LIMBS] {
    let zero = builder.ins().iconst(clif::types::I64, 0);
    [zero; I256_LIMBS]
}

fn unsigned_max_i256_limbs(builder: &mut FunctionBuilder) -> [clif::Value; I256_LIMBS] {
    let all_ones = builder.ins().iconst(clif::types::I64, -1);
    [all_ones; I256_LIMBS]
}

fn signed_min_i256_limbs(builder: &mut FunctionBuilder) -> [clif::Value; I256_LIMBS] {
    let zero = builder.ins().iconst(clif::types::I64, 0);
    let high = builder.ins().iconst(clif::types::I64, i64::MIN);
    [zero, zero, zero, high]
}

fn signed_max_i256_limbs(builder: &mut FunctionBuilder) -> [clif::Value; I256_LIMBS] {
    let all_ones = builder.ins().iconst(clif::types::I64, -1);
    let high = builder.ins().iconst(clif::types::I64, i64::MAX);
    [all_ones, all_ones, all_ones, high]
}

fn select_i256_limbs(
    condition: clif::Value,
    if_true: [clif::Value; I256_LIMBS],
    if_false: [clif::Value; I256_LIMBS],
    builder: &mut FunctionBuilder,
) -> [clif::Value; I256_LIMBS] {
    std::array::from_fn(|limb| {
        builder
            .ins()
            .select(condition, if_true[limb], if_false[limb])
    })
}

fn bool_xor(lhs: clif::Value, rhs: clif::Value, builder: &mut FunctionBuilder) -> clif::Value {
    let not_rhs = bool_not(rhs, builder);
    builder.ins().select(lhs, not_rhs, rhs)
}

fn bool_eq(lhs: clif::Value, rhs: clif::Value, builder: &mut FunctionBuilder) -> clif::Value {
    let different = bool_xor(lhs, rhs, builder);
    bool_not(different, builder)
}

fn i256_sign_bit(limbs: [clif::Value; I256_LIMBS], builder: &mut FunctionBuilder) -> clif::Value {
    let zero = builder.ins().iconst(clif::types::I64, 0);
    builder
        .ins()
        .icmp(IntCC::SignedLessThan, limbs[I256_LIMBS - 1], zero)
}

fn add_i256_limbs(
    lhs: [clif::Value; I256_LIMBS],
    rhs: [clif::Value; I256_LIMBS],
    builder: &mut FunctionBuilder,
) -> ([clif::Value; I256_LIMBS], clif::Value) {
    let zero = builder.ins().iconst(clif::types::I64, 0);
    let one = builder.ins().iconst(clif::types::I64, 1);
    let mut carry = zero;
    let mut result = [zero; I256_LIMBS];

    for limb in 0..I256_LIMBS {
        let (sum, carry_from_sum) = builder.ins().uadd_overflow(lhs[limb], rhs[limb]);
        let (sum, carry_from_carry) = builder.ins().uadd_overflow(sum, carry);
        result[limb] = sum;
        let carry_from_sum = builder.ins().select(carry_from_sum, one, zero);
        let carry_from_carry = builder.ins().select(carry_from_carry, one, zero);
        carry = builder.ins().bor(carry_from_sum, carry_from_carry);
    }

    let overflow = builder.ins().icmp(IntCC::NotEqual, carry, zero);
    (result, overflow)
}

fn sub_i256_limbs(
    lhs: [clif::Value; I256_LIMBS],
    rhs: [clif::Value; I256_LIMBS],
    builder: &mut FunctionBuilder,
) -> ([clif::Value; I256_LIMBS], clif::Value) {
    let zero = builder.ins().iconst(clif::types::I64, 0);
    let one = builder.ins().iconst(clif::types::I64, 1);
    let mut borrow = zero;
    let mut result = [zero; I256_LIMBS];

    for limb in 0..I256_LIMBS {
        let (diff, borrow_from_diff) = builder.ins().usub_overflow(lhs[limb], rhs[limb]);
        let (diff, borrow_from_borrow) = builder.ins().usub_overflow(diff, borrow);
        result[limb] = diff;
        let borrow_from_diff = builder.ins().select(borrow_from_diff, one, zero);
        let borrow_from_borrow = builder.ins().select(borrow_from_borrow, one, zero);
        borrow = builder.ins().bor(borrow_from_diff, borrow_from_borrow);
    }

    let overflow = builder.ins().icmp(IntCC::NotEqual, borrow, zero);
    (result, overflow)
}

fn neg_i256_limbs(
    value: [clif::Value; I256_LIMBS],
    builder: &mut FunctionBuilder,
) -> [clif::Value; I256_LIMBS] {
    sub_i256_limbs(zero_i256_limbs(builder), value, builder).0
}

fn abs_i256_limbs(
    value: [clif::Value; I256_LIMBS],
    builder: &mut FunctionBuilder,
) -> [clif::Value; I256_LIMBS] {
    let negative = i256_sign_bit(value, builder);
    let negated = neg_i256_limbs(value, builder);
    select_i256_limbs(negative, negated, value, builder)
}

fn add_to_wide_limbs(
    limbs: &mut [clif::Value; I256_PRODUCT_LIMBS],
    start: usize,
    value: clif::Value,
    builder: &mut FunctionBuilder,
) {
    let zero = builder.ins().iconst(clif::types::I64, 0);
    let one = builder.ins().iconst(clif::types::I64, 1);
    let (sum, carry) = builder.ins().uadd_overflow(limbs[start], value);
    limbs[start] = sum;
    let mut carry = builder.ins().select(carry, one, zero);

    for limb in &mut limbs[start + 1..] {
        let (sum, next_carry) = builder.ins().uadd_overflow(*limb, carry);
        *limb = sum;
        carry = builder.ins().select(next_carry, one, zero);
    }
}

fn mul_i256_limbs_full(
    lhs: [clif::Value; I256_LIMBS],
    rhs: [clif::Value; I256_LIMBS],
    builder: &mut FunctionBuilder,
) -> [clif::Value; I256_PRODUCT_LIMBS] {
    let zero = builder.ins().iconst(clif::types::I64, 0);
    let mut result = [zero; I256_PRODUCT_LIMBS];

    for (lhs_idx, lhs_limb) in lhs.into_iter().enumerate() {
        for (rhs_idx, rhs_limb) in rhs.into_iter().enumerate() {
            let result_idx = lhs_idx + rhs_idx;
            let product_low = builder.ins().imul(lhs_limb, rhs_limb);
            let product_high = builder.ins().umulhi(lhs_limb, rhs_limb);
            add_to_wide_limbs(&mut result, result_idx, product_low, builder);
            add_to_wide_limbs(&mut result, result_idx + 1, product_high, builder);
        }
    }

    result
}

fn low_i256_limbs(limbs: [clif::Value; I256_PRODUCT_LIMBS]) -> [clif::Value; I256_LIMBS] {
    [limbs[0], limbs[1], limbs[2], limbs[3]]
}

fn wide_i256_high_nonzero(
    limbs: [clif::Value; I256_PRODUCT_LIMBS],
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let zero = builder.ins().iconst(clif::types::I64, 0);
    let mut result = bool_const(false, builder);
    for limb in limbs.into_iter().skip(I256_LIMBS) {
        let nonzero = builder.ins().icmp(IntCC::NotEqual, limb, zero);
        result = bool_or(result, nonzero, builder);
    }
    result
}

fn emit_i256_add(
    function: &Function,
    lhs: ValueId,
    rhs: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let lhs = resolve_value(function, lhs, value_map, builder)?;
    let rhs = resolve_value(function, rhs, value_map, builder)?;
    let result = add_i256_limbs(
        load_i256_limbs(lhs, builder),
        load_i256_limbs(rhs, builder),
        builder,
    )
    .0;
    Ok(store_i256_limbs(result, builder))
}

fn emit_i256_sub(
    function: &Function,
    lhs: ValueId,
    rhs: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let lhs = resolve_value(function, lhs, value_map, builder)?;
    let rhs = resolve_value(function, rhs, value_map, builder)?;
    let result = sub_i256_limbs(
        load_i256_limbs(lhs, builder),
        load_i256_limbs(rhs, builder),
        builder,
    )
    .0;
    Ok(store_i256_limbs(result, builder))
}

fn emit_i256_mul(
    function: &Function,
    lhs: ValueId,
    rhs: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let lhs = resolve_value(function, lhs, value_map, builder)?;
    let rhs = resolve_value(function, rhs, value_map, builder)?;
    let result = low_i256_limbs(mul_i256_limbs_full(
        load_i256_limbs(lhs, builder),
        load_i256_limbs(rhs, builder),
        builder,
    ));
    Ok(store_i256_limbs(result, builder))
}

fn emit_i256_neg(
    function: &Function,
    value: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let value = resolve_value(function, value, value_map, builder)?;
    Ok(store_i256_limbs(
        neg_i256_limbs(load_i256_limbs(value, builder), builder),
        builder,
    ))
}

enum I256DivRemKind {
    Udiv,
    Sdiv,
    Umod,
    Smod,
}

fn i256_limb_bit(
    limbs: [clif::Value; I256_LIMBS],
    bit: usize,
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let limb = limbs[bit / I256_LIMB_BITS as usize];
    let shifted = builder
        .ins()
        .ushr_imm(limb, (bit % I256_LIMB_BITS as usize) as i64);
    let one = builder.ins().iconst(clif::types::I64, 1);
    let bit = builder.ins().band(shifted, one);
    let zero = builder.ins().iconst(clif::types::I64, 0);
    builder.ins().icmp(IntCC::NotEqual, bit, zero)
}

fn i256_shl_one_with_bit(
    limbs: [clif::Value; I256_LIMBS],
    bit: clif::Value,
    builder: &mut FunctionBuilder,
) -> [clif::Value; I256_LIMBS] {
    let zero = builder.ins().iconst(clif::types::I64, 0);
    let one = builder.ins().iconst(clif::types::I64, 1);
    let mut carry = builder.ins().select(bit, one, zero);
    std::array::from_fn(|limb_idx| {
        let shifted = builder.ins().ishl_imm(limbs[limb_idx], 1);
        let result = builder.ins().bor(shifted, carry);
        carry = builder.ins().ushr_imm(limbs[limb_idx], I256_LIMB_BITS - 1);
        result
    })
}

fn i256_set_bit_if(
    mut limbs: [clif::Value; I256_LIMBS],
    bit: usize,
    condition: clif::Value,
    builder: &mut FunctionBuilder,
) -> [clif::Value; I256_LIMBS] {
    let limb_idx = bit / I256_LIMB_BITS as usize;
    let mask = 1u64 << (bit % I256_LIMB_BITS as usize);
    let mask = builder.ins().iconst(clif::types::I64, mask as i64);
    let with_bit = builder.ins().bor(limbs[limb_idx], mask);
    limbs[limb_idx] = builder.ins().select(condition, with_bit, limbs[limb_idx]);
    limbs
}

fn unsigned_div_rem_i256_limbs(
    numerator: [clif::Value; I256_LIMBS],
    denominator: [clif::Value; I256_LIMBS],
    builder: &mut FunctionBuilder,
) -> ([clif::Value; I256_LIMBS], [clif::Value; I256_LIMBS]) {
    let mut quotient = zero_i256_limbs(builder);
    let mut remainder = zero_i256_limbs(builder);

    for bit in (0..I256_BITS as usize).rev() {
        let next_bit = i256_limb_bit(numerator, bit, builder);
        remainder = i256_shl_one_with_bit(remainder, next_bit, builder);
        let remainder_lt_denominator = emit_i256_unsigned_lt_limbs(remainder, denominator, builder);
        let should_subtract = bool_not(remainder_lt_denominator, builder);
        let subtracted = sub_i256_limbs(remainder, denominator, builder).0;
        remainder = select_i256_limbs(should_subtract, subtracted, remainder, builder);
        quotient = i256_set_bit_if(quotient, bit, should_subtract, builder);
    }

    (quotient, remainder)
}

fn emit_i256_div_rem(
    function: &Function,
    lhs: ValueId,
    rhs: ValueId,
    kind: I256DivRemKind,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let lhs = load_i256_limbs(resolve_value(function, lhs, value_map, builder)?, builder);
    let rhs = load_i256_limbs(resolve_value(function, rhs, value_map, builder)?, builder);
    let result = match kind {
        I256DivRemKind::Udiv | I256DivRemKind::Umod => {
            let (quotient, remainder) = unsigned_div_rem_i256_limbs(lhs, rhs, builder);
            match kind {
                I256DivRemKind::Udiv => quotient,
                I256DivRemKind::Umod => remainder,
                I256DivRemKind::Sdiv | I256DivRemKind::Smod => unreachable!(),
            }
        }
        I256DivRemKind::Sdiv | I256DivRemKind::Smod => {
            let lhs_negative = i256_sign_bit(lhs, builder);
            let rhs_negative = i256_sign_bit(rhs, builder);
            let lhs_abs = abs_i256_limbs(lhs, builder);
            let rhs_abs = abs_i256_limbs(rhs, builder);
            let (quotient, remainder) = unsigned_div_rem_i256_limbs(lhs_abs, rhs_abs, builder);
            let quotient_negative = bool_xor(lhs_negative, rhs_negative, builder);
            let quotient = select_i256_limbs(
                quotient_negative,
                neg_i256_limbs(quotient, builder),
                quotient,
                builder,
            );
            let remainder = select_i256_limbs(
                lhs_negative,
                neg_i256_limbs(remainder, builder),
                remainder,
                builder,
            );
            match kind {
                I256DivRemKind::Sdiv => quotient,
                I256DivRemKind::Smod => remainder,
                I256DivRemKind::Udiv | I256DivRemKind::Umod => unreachable!(),
            }
        }
    };
    Ok(store_i256_limbs(result, builder))
}

fn emit_i256_uaddo(
    function: &Function,
    lhs: ValueId,
    rhs: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<(clif::Value, clif::Value), String> {
    let lhs = load_i256_limbs(resolve_value(function, lhs, value_map, builder)?, builder);
    let rhs = load_i256_limbs(resolve_value(function, rhs, value_map, builder)?, builder);
    let (result, overflow) = add_i256_limbs(lhs, rhs, builder);
    Ok((store_i256_limbs(result, builder), overflow))
}

fn emit_i256_saddo(
    function: &Function,
    lhs: ValueId,
    rhs: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<(clif::Value, clif::Value), String> {
    let lhs = load_i256_limbs(resolve_value(function, lhs, value_map, builder)?, builder);
    let rhs = load_i256_limbs(resolve_value(function, rhs, value_map, builder)?, builder);
    let (result, _) = add_i256_limbs(lhs, rhs, builder);
    let lhs_negative = i256_sign_bit(lhs, builder);
    let rhs_negative = i256_sign_bit(rhs, builder);
    let result_negative = i256_sign_bit(result, builder);
    let same_sign = bool_eq(lhs_negative, rhs_negative, builder);
    let sign_changed = bool_xor(result_negative, lhs_negative, builder);
    let overflow = bool_and(same_sign, sign_changed, builder);
    Ok((store_i256_limbs(result, builder), overflow))
}

fn emit_i256_usubo(
    function: &Function,
    lhs: ValueId,
    rhs: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<(clif::Value, clif::Value), String> {
    let lhs = load_i256_limbs(resolve_value(function, lhs, value_map, builder)?, builder);
    let rhs = load_i256_limbs(resolve_value(function, rhs, value_map, builder)?, builder);
    let (result, overflow) = sub_i256_limbs(lhs, rhs, builder);
    Ok((store_i256_limbs(result, builder), overflow))
}

fn emit_i256_ssubo(
    function: &Function,
    lhs: ValueId,
    rhs: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<(clif::Value, clif::Value), String> {
    let lhs = load_i256_limbs(resolve_value(function, lhs, value_map, builder)?, builder);
    let rhs = load_i256_limbs(resolve_value(function, rhs, value_map, builder)?, builder);
    let (result, _) = sub_i256_limbs(lhs, rhs, builder);
    let lhs_negative = i256_sign_bit(lhs, builder);
    let rhs_negative = i256_sign_bit(rhs, builder);
    let result_negative = i256_sign_bit(result, builder);
    let different_sign = bool_xor(lhs_negative, rhs_negative, builder);
    let sign_changed = bool_xor(result_negative, lhs_negative, builder);
    let overflow = bool_and(different_sign, sign_changed, builder);
    Ok((store_i256_limbs(result, builder), overflow))
}

fn emit_i256_umulo(
    function: &Function,
    lhs: ValueId,
    rhs: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<(clif::Value, clif::Value), String> {
    let lhs = load_i256_limbs(resolve_value(function, lhs, value_map, builder)?, builder);
    let rhs = load_i256_limbs(resolve_value(function, rhs, value_map, builder)?, builder);
    let product = mul_i256_limbs_full(lhs, rhs, builder);
    let overflow = wide_i256_high_nonzero(product, builder);
    Ok((store_i256_limbs(low_i256_limbs(product), builder), overflow))
}

fn emit_i256_smulo(
    function: &Function,
    lhs: ValueId,
    rhs: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<(clif::Value, clif::Value), String> {
    let lhs = load_i256_limbs(resolve_value(function, lhs, value_map, builder)?, builder);
    let rhs = load_i256_limbs(resolve_value(function, rhs, value_map, builder)?, builder);
    let raw = low_i256_limbs(mul_i256_limbs_full(lhs, rhs, builder));
    let lhs_negative = i256_sign_bit(lhs, builder);
    let rhs_negative = i256_sign_bit(rhs, builder);
    let product_negative = bool_xor(lhs_negative, rhs_negative, builder);
    let abs_product = mul_i256_limbs_full(
        abs_i256_limbs(lhs, builder),
        abs_i256_limbs(rhs, builder),
        builder,
    );
    let high_nonzero = wide_i256_high_nonzero(abs_product, builder);
    let low_abs_product = low_i256_limbs(abs_product);
    let positive_limit = signed_max_i256_limbs(builder);
    let negative_limit = signed_min_i256_limbs(builder);
    let limit = select_i256_limbs(product_negative, negative_limit, positive_limit, builder);
    let over_limit = emit_i256_unsigned_lt_limbs(limit, low_abs_product, builder);
    let overflow = bool_or(high_nonzero, over_limit, builder);
    Ok((store_i256_limbs(raw, builder), overflow))
}

fn emit_i256_snego(
    function: &Function,
    value: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<(clif::Value, clif::Value), String> {
    let value = load_i256_limbs(resolve_value(function, value, value_map, builder)?, builder);
    let result = neg_i256_limbs(value, builder);
    let overflow = emit_i256_eq_limbs(value, signed_min_i256_limbs(builder), builder);
    Ok((store_i256_limbs(result, builder), overflow))
}

enum I256SaturatingOp {
    Uadd,
    Sadd,
    Usub,
    Ssub,
    Umul,
    Smul,
}

fn emit_i256_saturating_binary(
    function: &Function,
    lhs: ValueId,
    rhs: ValueId,
    op: I256SaturatingOp,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let lhs_value = load_i256_limbs(resolve_value(function, lhs, value_map, builder)?, builder);
    let rhs_value = load_i256_limbs(resolve_value(function, rhs, value_map, builder)?, builder);
    let (raw, overflow, saturated) = match op {
        I256SaturatingOp::Uadd => {
            let (raw, overflow) = add_i256_limbs(lhs_value, rhs_value, builder);
            (raw, overflow, unsigned_max_i256_limbs(builder))
        }
        I256SaturatingOp::Sadd => {
            let (raw, _) = add_i256_limbs(lhs_value, rhs_value, builder);
            let lhs_negative = i256_sign_bit(lhs_value, builder);
            let rhs_negative = i256_sign_bit(rhs_value, builder);
            let result_negative = i256_sign_bit(raw, builder);
            let same_sign = bool_eq(lhs_negative, rhs_negative, builder);
            let sign_changed = bool_xor(result_negative, lhs_negative, builder);
            let overflow = bool_and(same_sign, sign_changed, builder);
            let saturated = select_i256_limbs(
                lhs_negative,
                signed_min_i256_limbs(builder),
                signed_max_i256_limbs(builder),
                builder,
            );
            (raw, overflow, saturated)
        }
        I256SaturatingOp::Usub => {
            let (raw, overflow) = sub_i256_limbs(lhs_value, rhs_value, builder);
            (raw, overflow, zero_i256_limbs(builder))
        }
        I256SaturatingOp::Ssub => {
            let (raw, _) = sub_i256_limbs(lhs_value, rhs_value, builder);
            let lhs_negative = i256_sign_bit(lhs_value, builder);
            let rhs_negative = i256_sign_bit(rhs_value, builder);
            let result_negative = i256_sign_bit(raw, builder);
            let different_sign = bool_xor(lhs_negative, rhs_negative, builder);
            let sign_changed = bool_xor(result_negative, lhs_negative, builder);
            let overflow = bool_and(different_sign, sign_changed, builder);
            let saturated = select_i256_limbs(
                lhs_negative,
                signed_min_i256_limbs(builder),
                signed_max_i256_limbs(builder),
                builder,
            );
            (raw, overflow, saturated)
        }
        I256SaturatingOp::Umul => {
            let product = mul_i256_limbs_full(lhs_value, rhs_value, builder);
            (
                low_i256_limbs(product),
                wide_i256_high_nonzero(product, builder),
                unsigned_max_i256_limbs(builder),
            )
        }
        I256SaturatingOp::Smul => {
            let raw = low_i256_limbs(mul_i256_limbs_full(lhs_value, rhs_value, builder));
            let lhs_negative = i256_sign_bit(lhs_value, builder);
            let rhs_negative = i256_sign_bit(rhs_value, builder);
            let product_negative = bool_xor(lhs_negative, rhs_negative, builder);
            let abs_product = mul_i256_limbs_full(
                abs_i256_limbs(lhs_value, builder),
                abs_i256_limbs(rhs_value, builder),
                builder,
            );
            let high_nonzero = wide_i256_high_nonzero(abs_product, builder);
            let low_abs_product = low_i256_limbs(abs_product);
            let limit = select_i256_limbs(
                product_negative,
                signed_min_i256_limbs(builder),
                signed_max_i256_limbs(builder),
                builder,
            );
            let over_limit = emit_i256_unsigned_lt_limbs(limit, low_abs_product, builder);
            let overflow = bool_or(high_nonzero, over_limit, builder);
            let saturated = select_i256_limbs(
                product_negative,
                signed_min_i256_limbs(builder),
                signed_max_i256_limbs(builder),
                builder,
            );
            (raw, overflow, saturated)
        }
    };
    Ok(store_i256_limbs(
        select_i256_limbs(overflow, saturated, raw, builder),
        builder,
    ))
}

enum I256BitwiseOp {
    And,
    Or,
    Xor,
}

fn emit_i256_bitwise(
    function: &Function,
    lhs: ValueId,
    rhs: ValueId,
    op: I256BitwiseOp,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let lhs = resolve_value(function, lhs, value_map, builder)?;
    let rhs = resolve_value(function, rhs, value_map, builder)?;
    let result = create_i256_slot(builder);

    for limb in 0..I256_LIMBS {
        let lhs_limb = load_i256_limb(lhs, limb, builder);
        let rhs_limb = load_i256_limb(rhs, limb, builder);
        let value = match op {
            I256BitwiseOp::And => builder.ins().band(lhs_limb, rhs_limb),
            I256BitwiseOp::Or => builder.ins().bor(lhs_limb, rhs_limb),
            I256BitwiseOp::Xor => builder.ins().bxor(lhs_limb, rhs_limb),
        };
        store_i256_limb(result, limb, value, builder);
    }

    Ok(result)
}

fn emit_i256_not(
    function: &Function,
    value: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let value = resolve_value(function, value, value_map, builder)?;
    let result = create_i256_slot(builder);

    for limb in 0..I256_LIMBS {
        let limb_value = load_i256_limb(value, limb, builder);
        store_i256_limb(result, limb, builder.ins().bnot(limb_value), builder);
    }

    Ok(result)
}

enum I256ShiftKind {
    Shl,
    Shr,
    Sar,
}

fn emit_i256_shift(
    function: &Function,
    value: ValueId,
    bits: ValueId,
    kind: I256ShiftKind,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let value_addr = resolve_value(function, value, value_map, builder)?;
    let limbs = [
        load_i256_limb(value_addr, 0, builder),
        load_i256_limb(value_addr, 1, builder),
        load_i256_limb(value_addr, 2, builder),
        load_i256_limb(value_addr, 3, builder),
    ];
    let (shift, too_large) = resolve_i256_shift_amount(function, bits, value_map, builder)?;
    let result = create_i256_slot(builder);
    let zero = builder.ins().iconst(clif::types::I64, 0);
    let limb_mask = builder.ins().iconst(clif::types::I64, I256_LIMB_BITS - 1);
    let limb_shift = builder.ins().ushr_imm(shift, 6);
    let bit_shift = builder.ins().band(shift, limb_mask);
    let bit_shift_is_zero = builder.ins().icmp(IntCC::Equal, bit_shift, zero);
    let limb_bits = builder.ins().iconst(clif::types::I64, I256_LIMB_BITS);
    let inverse_shift = builder.ins().isub(limb_bits, bit_shift);
    let high_limb_is_negative =
        builder
            .ins()
            .icmp(IntCC::SignedLessThan, limbs[I256_LIMBS - 1], zero);
    let all_ones = builder.ins().iconst(clif::types::I64, -1);
    let sign_fill = builder.ins().select(high_limb_is_negative, all_ones, zero);
    let overshift_fill = match kind {
        I256ShiftKind::Sar => sign_fill,
        I256ShiftKind::Shl | I256ShiftKind::Shr => zero,
    };

    for result_limb_idx in 0..I256_LIMBS {
        let shifted = match kind {
            I256ShiftKind::Shl => emit_i256_shl_limb(
                result_limb_idx,
                limbs,
                limb_shift,
                bit_shift,
                inverse_shift,
                bit_shift_is_zero,
                builder,
            ),
            I256ShiftKind::Shr => emit_i256_right_shift_limb(
                result_limb_idx,
                limbs,
                limb_shift,
                bit_shift,
                inverse_shift,
                bit_shift_is_zero,
                zero,
                builder,
            ),
            I256ShiftKind::Sar => emit_i256_right_shift_limb(
                result_limb_idx,
                limbs,
                limb_shift,
                bit_shift,
                inverse_shift,
                bit_shift_is_zero,
                sign_fill,
                builder,
            ),
        };
        let shifted = builder.ins().select(too_large, overshift_fill, shifted);
        store_i256_limb(result, result_limb_idx, shifted, builder);
    }

    Ok(result)
}

fn resolve_i256_shift_amount(
    function: &Function,
    bits: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<(clif::Value, clif::Value), String> {
    let zero = builder.ins().iconst(clif::types::I64, 0);
    let max_shift = builder.ins().iconst(clif::types::I64, I256_BITS);
    let bits_ty = function.dfg.value_ty(bits);

    if bits_ty == Type::I256 {
        let bits_addr = resolve_value(function, bits, value_map, builder)?;
        let shift = load_i256_limb(bits_addr, 0, builder);
        let mut upper_nonzero = bool_const(false, builder);
        for limb in 1..I256_LIMBS {
            let limb_value = load_i256_limb(bits_addr, limb, builder);
            let limb_nonzero = builder.ins().icmp(IntCC::NotEqual, limb_value, zero);
            upper_nonzero = bool_or(upper_nonzero, limb_nonzero, builder);
        }
        let shift_too_large =
            builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, shift, max_shift);
        return Ok((shift, bool_or(upper_nonzero, shift_too_large, builder)));
    }

    let raw = resolve_value(function, bits, value_map, builder)?;
    let raw_ty = builder.func.dfg.value_type(raw);
    let shift = resize_int_value(raw, clif::types::I64, false, builder);
    let shift_too_large = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThanOrEqual, shift, max_shift);
    if raw_ty.bits() <= I256_LIMB_BITS as u32 {
        return Ok((shift, shift_too_large));
    }

    let high_bits = builder.ins().ushr_imm(raw, I256_LIMB_BITS);
    let zero_raw = builder.ins().iconst(raw_ty, 0);
    let high_bits_nonzero = builder.ins().icmp(IntCC::NotEqual, high_bits, zero_raw);
    Ok((shift, bool_or(high_bits_nonzero, shift_too_large, builder)))
}

fn emit_i256_shl_limb(
    result_limb_idx: usize,
    limbs: [clif::Value; I256_LIMBS],
    limb_shift: clif::Value,
    bit_shift: clif::Value,
    inverse_shift: clif::Value,
    bit_shift_is_zero: clif::Value,
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let zero = builder.ins().iconst(clif::types::I64, 0);
    let mut result = zero;

    for shifted_limb_count in 0..=result_limb_idx {
        let low_idx = result_limb_idx - shifted_limb_count;
        let low = builder.ins().ishl(limbs[low_idx], bit_shift);
        let high = if low_idx > 0 {
            let high = builder.ins().ushr(limbs[low_idx - 1], inverse_shift);
            builder.ins().select(bit_shift_is_zero, zero, high)
        } else {
            zero
        };
        let combined = builder.ins().bor(low, high);
        let count = builder
            .ins()
            .iconst(clif::types::I64, shifted_limb_count as i64);
        let matches_count = builder.ins().icmp(IntCC::Equal, limb_shift, count);
        result = builder.ins().select(matches_count, combined, result);
    }

    result
}

#[allow(clippy::too_many_arguments)]
fn emit_i256_right_shift_limb(
    result_limb_idx: usize,
    limbs: [clif::Value; I256_LIMBS],
    limb_shift: clif::Value,
    bit_shift: clif::Value,
    inverse_shift: clif::Value,
    bit_shift_is_zero: clif::Value,
    fill: clif::Value,
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let mut result = fill;

    for shifted_limb_count in 0..I256_LIMBS {
        let low_idx = result_limb_idx + shifted_limb_count;
        let low_source = limbs.get(low_idx).copied().unwrap_or(fill);
        let high_source = limbs.get(low_idx + 1).copied().unwrap_or(fill);
        let low = builder.ins().ushr(low_source, bit_shift);
        let high = builder.ins().ishl(high_source, inverse_shift);
        let zero = builder.ins().iconst(clif::types::I64, 0);
        let high = builder.ins().select(bit_shift_is_zero, zero, high);
        let combined = builder.ins().bor(low, high);
        let count = builder
            .ins()
            .iconst(clif::types::I64, shifted_limb_count as i64);
        let matches_count = builder.ins().icmp(IntCC::Equal, limb_shift, count);
        result = builder.ins().select(matches_count, combined, result);
    }

    result
}

fn bool_const(value: bool, builder: &mut FunctionBuilder) -> clif::Value {
    let zero = builder.ins().iconst(clif::types::I8, 0);
    let rhs = builder.ins().iconst(clif::types::I8, (!value) as i64);
    builder.ins().icmp(IntCC::Equal, zero, rhs)
}

fn bool_not(value: clif::Value, builder: &mut FunctionBuilder) -> clif::Value {
    let yes = bool_const(true, builder);
    let no = bool_const(false, builder);
    builder.ins().select(value, no, yes)
}

fn bool_and(lhs: clif::Value, rhs: clif::Value, builder: &mut FunctionBuilder) -> clif::Value {
    let no = bool_const(false, builder);
    builder.ins().select(lhs, rhs, no)
}

fn bool_or(lhs: clif::Value, rhs: clif::Value, builder: &mut FunctionBuilder) -> clif::Value {
    let yes = bool_const(true, builder);
    builder.ins().select(lhs, yes, rhs)
}

fn emit_i256_eq_limbs(
    lhs: [clif::Value; I256_LIMBS],
    rhs: [clif::Value; I256_LIMBS],
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let mut result = bool_const(true, builder);
    for limb in 0..I256_LIMBS {
        let limbs_equal = builder.ins().icmp(IntCC::Equal, lhs[limb], rhs[limb]);
        result = bool_and(result, limbs_equal, builder);
    }
    result
}

fn emit_i256_eq(lhs: clif::Value, rhs: clif::Value, builder: &mut FunctionBuilder) -> clif::Value {
    emit_i256_eq_limbs(
        load_i256_limbs(lhs, builder),
        load_i256_limbs(rhs, builder),
        builder,
    )
}

fn emit_i256_unsigned_lt_limbs(
    lhs: [clif::Value; I256_LIMBS],
    rhs: [clif::Value; I256_LIMBS],
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let mut result = bool_const(false, builder);
    let mut equal_prefix = bool_const(true, builder);
    for limb in (0..I256_LIMBS).rev() {
        let limb_lt = builder
            .ins()
            .icmp(IntCC::UnsignedLessThan, lhs[limb], rhs[limb]);
        let limb_eq = builder.ins().icmp(IntCC::Equal, lhs[limb], rhs[limb]);
        result = builder.ins().select(equal_prefix, limb_lt, result);
        equal_prefix = bool_and(equal_prefix, limb_eq, builder);
    }
    result
}

fn emit_i256_unsigned_lt(
    lhs: clif::Value,
    rhs: clif::Value,
    builder: &mut FunctionBuilder,
) -> clif::Value {
    emit_i256_unsigned_lt_limbs(
        load_i256_limbs(lhs, builder),
        load_i256_limbs(rhs, builder),
        builder,
    )
}

fn emit_i256_signed_lt(
    lhs: clif::Value,
    rhs: clif::Value,
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let lhs_high = load_i256_limb(lhs, 3, builder);
    let rhs_high = load_i256_limb(rhs, 3, builder);
    let high_lt = builder
        .ins()
        .icmp(IntCC::SignedLessThan, lhs_high, rhs_high);
    let high_eq = builder.ins().icmp(IntCC::Equal, lhs_high, rhs_high);
    let lower_lt = emit_i256_unsigned_lt(lhs, rhs, builder);
    let equal_high_lower_lt = bool_and(high_eq, lower_lt, builder);
    bool_or(high_lt, equal_high_lower_lt, builder)
}

fn emit_i256_icmp(
    cc: IntCC,
    lhs: clif::Value,
    rhs: clif::Value,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let result = match cc {
        IntCC::Equal => emit_i256_eq(lhs, rhs, builder),
        IntCC::NotEqual => {
            let equal = emit_i256_eq(lhs, rhs, builder);
            bool_not(equal, builder)
        }
        IntCC::UnsignedLessThan => emit_i256_unsigned_lt(lhs, rhs, builder),
        IntCC::UnsignedGreaterThan => emit_i256_unsigned_lt(rhs, lhs, builder),
        IntCC::UnsignedLessThanOrEqual => {
            let greater = emit_i256_unsigned_lt(rhs, lhs, builder);
            bool_not(greater, builder)
        }
        IntCC::UnsignedGreaterThanOrEqual => {
            let less = emit_i256_unsigned_lt(lhs, rhs, builder);
            bool_not(less, builder)
        }
        IntCC::SignedLessThan => emit_i256_signed_lt(lhs, rhs, builder),
        IntCC::SignedGreaterThan => emit_i256_signed_lt(rhs, lhs, builder),
        IntCC::SignedLessThanOrEqual => {
            let greater = emit_i256_signed_lt(rhs, lhs, builder);
            bool_not(greater, builder)
        }
        IntCC::SignedGreaterThanOrEqual => {
            let less = emit_i256_signed_lt(lhs, rhs, builder);
            bool_not(less, builder)
        }
    };
    Ok(result)
}

fn emit_i256_is_zero(value: clif::Value, builder: &mut FunctionBuilder) -> clif::Value {
    let zero = builder.ins().iconst(clif::types::I64, 0);
    let mut result = bool_const(true, builder);
    for limb in 0..4 {
        let limb = load_i256_limb(value, limb, builder);
        let limb_is_zero = builder.ins().icmp(IntCC::Equal, limb, zero);
        result = bool_and(result, limb_is_zero, builder);
    }
    result
}

fn resize_int_value(
    value: clif::Value,
    to_ty: clif::Type,
    signed: bool,
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let from_ty = builder.func.dfg.value_type(value);
    if !from_ty.is_int() && to_ty.is_int() {
        let zero = builder.ins().iconst(to_ty, 0);
        let one = builder.ins().iconst(to_ty, 1);
        return builder.ins().select(value, one, zero);
    }
    match from_ty.bits().cmp(&to_ty.bits()) {
        Ordering::Equal => value,
        Ordering::Less if signed => builder.ins().sextend(to_ty, value),
        Ordering::Less => builder.ins().uextend(to_ty, value),
        Ordering::Greater => builder.ins().ireduce(to_ty, value),
    }
}

fn bool_to_int_value(
    value: clif::Value,
    to_ty: clif::Type,
    builder: &mut FunctionBuilder,
) -> clif::Value {
    let value_ty = builder.func.dfg.value_type(value);
    let cond = if value_ty.is_int() {
        let zero = builder.ins().iconst(value_ty, 0);
        builder.ins().icmp(IntCC::NotEqual, value, zero)
    } else {
        value
    };
    let zero = builder.ins().iconst(to_ty, 0);
    let one = builder.ins().iconst(to_ty, 1);
    builder.ins().select(cond, one, zero)
}

fn translate_bitcast(
    value: clif::Value,
    to_ty: clif::Type,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    let from_ty = builder.func.dfg.value_type(value);
    if from_ty == to_ty {
        Ok(value)
    } else if from_ty.bits() == to_ty.bits() {
        Ok(builder.ins().bitcast(to_ty, MemFlagsData::new(), value))
    } else {
        Err(format!(
            "cannot bitcast Cranelift value from {from_ty} to {to_ty}"
        ))
    }
}

fn insert_clif_results(
    function: &Function,
    inst_id: sonatina_ir::inst::InstId,
    values: impl IntoIterator<Item = clif::Value>,
    value_map: &mut HashMap<ValueId, clif::Value>,
) {
    for (ir_result, clif_result) in function.dfg.inst_results(inst_id).iter().zip(values) {
        value_map.insert(*ir_result, clif_result);
    }
}

fn unsigned_max_value(ty: clif::Type, builder: &mut FunctionBuilder) -> clif::Value {
    builder.ins().iconst(ty, -1)
}

fn signed_min_value(ty: clif::Type, builder: &mut FunctionBuilder) -> Result<clif::Value, String> {
    let bits = ty.bits();
    if bits > i64::BITS {
        return Err(format!(
            "signed minimum constant for {bits}-bit Cranelift integers is unsupported"
        ));
    }
    let value = if bits == i64::BITS {
        i64::MIN
    } else {
        1i64 << (bits - 1)
    };
    Ok(builder.ins().iconst(ty, value))
}

fn signed_max_value(ty: clif::Type, builder: &mut FunctionBuilder) -> Result<clif::Value, String> {
    let bits = ty.bits();
    if bits > i64::BITS {
        return Err(format!(
            "signed maximum constant for {bits}-bit Cranelift integers is unsupported"
        ));
    }
    let value = if bits == i64::BITS {
        i64::MAX
    } else {
        (1i64 << (bits - 1)) - 1
    };
    Ok(builder.ins().iconst(ty, value))
}

fn resolve_value(
    function: &Function,
    value_id: ValueId,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<clif::Value, String> {
    if let Some(&clif_val) = value_map.get(&value_id) {
        return Ok(clif_val);
    }
    // Check if there's a Variable for this (phi values in loops)
    // Variables are looked up via the FunctionBuilder's SSA system

    let value = function.dfg.value(value_id);
    match value {
        Value::Immediate { imm, ty } => {
            if matches!(imm, Immediate::I256(_) | Immediate::I128(_)) {
                match imm {
                    Immediate::I256(i256_val) => Ok(emit_i256_immediate(i256_val, builder)),
                    _ => Err(format!("unsupported large immediate: {imm:?}")),
                }
            } else {
                let clif_ty = sonatina_type_to_clif_or_err(*ty)?;
                let i64_val = imm_to_i64(imm)?;
                let val = builder.ins().iconst(clif_ty, i64_val);
                Ok(val)
            }
        }
        _ => Err(format!("unresolved value v{}", value_id.0)),
    }
}

fn imm_to_i64(imm: &Immediate) -> Result<i64, String> {
    match imm {
        Immediate::I1(b) => Ok(*b as i64),
        Immediate::I8(v) => Ok(*v as i64),
        Immediate::I16(v) => Ok(*v as i64),
        Immediate::I32(v) => Ok(*v as i64),
        Immediate::I64(v) => Ok(*v),
        _ => Err(format!("unsupported immediate type for cranelift: {imm:?}")),
    }
}

fn emit_i256_immediate(imm: &sonatina_ir::I256, builder: &mut FunctionBuilder) -> clif::Value {
    let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
        32,
        0,
    ));
    let addr = builder.ins().stack_addr(clif::types::I64, slot, 0);

    let u256 = imm.to_u256();
    let bytes = u256.to_little_endian();
    for i in 0..4 {
        let limb = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
        let val = builder.ins().iconst(clif::types::I64, limb as i64);
        builder
            .ins()
            .store(MemFlagsData::new(), val, addr, (i * 8) as i32);
    }

    addr
}

#[allow(clippy::too_many_arguments)]
fn translate_icmp(
    cc: IntCC,
    lhs: ValueId,
    rhs: ValueId,
    inst_id: sonatina_ir::inst::InstId,
    module: &Module,
    function: &Function,
    value_map: &mut HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<(), String> {
    let lhs_ty = function.dfg.value_ty(lhs);
    let rhs_ty = function.dfg.value_ty(rhs);
    let result_val = if lhs_ty == Type::I256 || rhs_ty == Type::I256 {
        if lhs_ty != Type::I256 || rhs_ty != Type::I256 {
            return Err(format!(
                "cannot compare mismatched i256 and scalar values: {lhs_ty:?}, {rhs_ty:?}"
            ));
        }
        let lhs_val = resolve_value(function, lhs, value_map, builder)?;
        let rhs_val = resolve_value(function, rhs, value_map, builder)?;
        emit_i256_icmp(cc, lhs_val, rhs_val, builder)?
    } else {
        let lhs_val = resolve_scalar_value(module, function, lhs, value_map, builder)?;
        let rhs_val = resolve_scalar_value(module, function, rhs, value_map, builder)?;
        builder.ins().icmp(cc, lhs_val, rhs_val)
    };
    if let Some(result) = function.dfg.inst_result(inst_id) {
        value_map.insert(result, result_val);
    }
    Ok(())
}

fn emit_u256_intrinsic_call(
    clif_module: &mut impl ClifModule,
    builder: &mut FunctionBuilder,
    name: &str,
    args: &[clif::Value],
    has_result: bool,
) -> Result<clif::Value, String> {
    let ptr_ty = clif::types::I64;

    // Build the intrinsic signature: all args are pointers, optional result pointer
    let mut sig = clif_module.make_signature();
    for _ in args {
        sig.params.push(clif::AbiParam::new(ptr_ty));
    }
    if has_result {
        sig.params.push(clif::AbiParam::new(ptr_ty)); // result pointer
    }

    let func_id = clif_module
        .declare_function(name, Linkage::Import, &sig)
        .map_err(|e| format!("failed to declare {name}: {e}"))?;
    let func_ref = clif_module.declare_func_in_func(func_id, builder.func);

    if has_result {
        // Allocate 32-byte stack slot for the result
        let result_slot =
            builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                32,
                0,
            ));
        let result_addr = builder.ins().stack_addr(ptr_ty, result_slot, 0);

        let mut call_args: Vec<clif::Value> = args.to_vec();
        call_args.push(result_addr);
        builder.ins().call(func_ref, &call_args);

        Ok(result_addr)
    } else {
        builder.ins().call(func_ref, args);
        Ok(builder.ins().iconst(ptr_ty, 0))
    }
}

fn translate_aggregate_projection(
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
        builder.ins().iadd_imm(base, offset)
    })
}

fn materialize_gv_initializer(
    init: &sonatina_ir::global_variable::GvInitializer,
    ty: Type,
    base: clif::Value,
    offset: i32,
    ctx: &sonatina_ir::module::ModuleCtx,
    builder: &mut FunctionBuilder,
) {
    use sonatina_ir::global_variable::GvInitializer;
    match init {
        GvInitializer::Immediate(imm) => match imm {
            Immediate::I8(v) => {
                let val = builder.ins().iconst(clif::types::I8, *v as i64);
                builder.ins().store(MemFlagsData::new(), val, base, offset);
            }
            Immediate::I16(v) => {
                let val = builder.ins().iconst(clif::types::I16, *v as i64);
                builder.ins().store(MemFlagsData::new(), val, base, offset);
            }
            Immediate::I32(v) => {
                let val = builder.ins().iconst(clif::types::I32, *v as i64);
                builder.ins().store(MemFlagsData::new(), val, base, offset);
            }
            Immediate::I64(v) => {
                let val = builder.ins().iconst(clif::types::I64, *v);
                builder.ins().store(MemFlagsData::new(), val, base, offset);
            }
            Immediate::I256(v) => {
                let u = v.to_u256();
                let bytes = u.to_little_endian();
                for i in 0..4 {
                    let limb = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
                    let val = builder.ins().iconst(clif::types::I64, limb as i64);
                    builder
                        .ins()
                        .store(MemFlagsData::new(), val, base, offset + (i * 8) as i32);
                }
            }
            _ => {}
        },
        GvInitializer::Array(elems) => {
            if let Some(
                sonatina_ir::types::CompoundType::Array { elem, .. }
                | sonatina_ir::types::CompoundType::ConstRef(elem),
            ) = ty.resolve_compound(ctx)
            {
                let elem_size = ctx.size_of_unchecked(elem) as i32;
                for (i, elem_init) in elems.iter().enumerate() {
                    materialize_gv_initializer(
                        elem_init,
                        elem,
                        base,
                        offset + i as i32 * elem_size,
                        ctx,
                        builder,
                    );
                }
            }
        }
        GvInitializer::Struct(fields) => {
            if let Some(sonatina_ir::types::CompoundType::Struct(s)) = ty.resolve_compound(ctx) {
                for (idx, (field_init, &field_ty)) in fields.iter().zip(s.fields.iter()).enumerate()
                {
                    let Some((field_offset, _)) = ctx.aggregate_elem_offset(ty, idx) else {
                        continue;
                    };
                    materialize_gv_initializer(
                        field_init,
                        field_ty,
                        base,
                        offset + field_offset as i32,
                        ctx,
                        builder,
                    );
                }
            }
        }
    }
}

fn compute_alloc_size(ty: Type, ctx: &sonatina_ir::module::ModuleCtx) -> u32 {
    if let Type::Compound(_) = ty
        && let Some(cmpd) = ty.resolve_compound(ctx)
    {
        match cmpd {
            sonatina_ir::types::CompoundType::ObjRef(inner)
            | sonatina_ir::types::CompoundType::ConstRef(inner) => {
                return compute_alloc_size(inner, ctx);
            }
            _ => {}
        }
    }
    let size = ctx.size_of_unchecked(ty);
    size.max(8) as u32
}

fn aggregate_elem_offset(
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
    if let Some(
        sonatina_ir::types::CompoundType::ObjRef(inner)
        | sonatina_ir::types::CompoundType::ConstRef(inner),
    ) = aggregate_ty.resolve_compound(ctx)
    {
        return aggregate_elem_offset(ctx, inner, idx);
    }
    Err(format!(
        "cannot compute aggregate element offset for {aggregate_ty:?}[{idx}]"
    ))
}

fn constant_value_index(
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

fn is_undef_value(function: &Function, value_id: ValueId) -> bool {
    matches!(function.dfg.value(value_id), Value::Undef { .. })
}

fn compute_element_size(obj_ty: Type, ctx: &sonatina_ir::module::ModuleCtx) -> usize {
    if let Some(cmpd) = obj_ty.resolve_compound(ctx) {
        match cmpd {
            sonatina_ir::types::CompoundType::Array { elem, .. } => {
                return ctx.size_of_unchecked(elem);
            }
            sonatina_ir::types::CompoundType::ObjRef(inner)
            | sonatina_ir::types::CompoundType::ConstRef(inner) => {
                return compute_element_size(inner, ctx);
            }
            _ => {}
        }
    }
    // Fallback: 32 bytes (i256 size)
    32
}

fn collect_phi_args_for_block(
    function: &Function,
    target_block: BlockId,
    source_block: BlockId,
    inst_set: &dyn sonatina_ir::InstSetBase,
    value_map: &HashMap<ValueId, clif::Value>,
    builder: &mut FunctionBuilder,
) -> Result<Vec<BlockArg>, String> {
    let mut args = Vec::new();
    for inst_id in function.layout.iter_inst(target_block) {
        let inst_data = function.dfg.inst(inst_id);
        if let Some(phi) =
            <&sonatina_ir::inst::control_flow::Phi as sonatina_ir::InstDowncast>::downcast(
                inst_set, inst_data,
            )
        {
            for &(value, from_block) in phi.args() {
                if from_block == source_block {
                    let clif_val = resolve_value(function, value, value_map, builder)?;
                    args.push(BlockArg::Value(clif_val));
                    break;
                }
            }
        } else {
            break;
        }
    }
    Ok(args)
}
