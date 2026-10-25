mod int;
mod memory;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::smallvec;
use sonatina_ir::{
    BlockId, ControlFlowGraph, Function, GlobalVariableRef, Immediate, InstId, Linkage, Module,
    Type, Value, ValueId,
    inst::{control_flow::Phi, inst_set::InstSetExt, native::inst_set::NativeInstKind},
    isa::native::inst_set as native_inst_set,
    module::{FuncRef, ModuleCtx},
};
use waffle::{
    BlockTarget, Export, ExportKind, Func, FuncDecl, FunctionBody, Global, GlobalData, Import,
    ImportKind, MemoryData, Module as WaffleModule, Operator, Terminator, Type as WType,
    Value as WValue,
};

use self::int::{Arith, CmpKind, OverflowOp, SaturatingOp};
use super::{
    abi::{
        IntRepr, indirect_return_storage, int_repr, is_indirect, returns_indirect, signature_data,
        value_types,
    },
    builder::{Builder, Repr},
    runtime::{Helper, MemoryLayout, Runtime, RuntimeSymbol, define_helper},
    wide::ShiftKind,
};
use crate::{
    isa::native_layout::value_storage_size,
    transform::aggregate::{compute_object_effect_summaries, object_abi::native_heap_object_roots},
};

/// Module-wide state shared by function translation.
struct ModuleEnv {
    funcs: FxHashMap<FuncRef, Func>,
    globals: FxHashMap<GlobalVariableRef, u32>,
    stack_pointer: Global,
    stack_base: u32,
    runtime: Option<Runtime>,
}

/// Arithmetic helper functions, defined once each when first used.
#[derive(Default)]
struct Helpers {
    funcs: FxHashMap<(Helper, usize), Func>,
}

pub(super) fn translate_module(
    module: &Module,
    stack_size: u32,
) -> Result<WaffleModule<'static>, String> {
    let ctx = &module.ctx;
    let layout = MemoryLayout::new(ctx, stack_size)?;
    let mut wmod = WaffleModule::empty();
    let memory = wmod.memories.push(MemoryData {
        initial_pages: layout.initial_pages(),
        maximum_pages: None,
        segments: layout.segments.clone(),
    });
    wmod.exports.push(Export {
        name: "memory".to_string(),
        kind: ExportKind::Memory(memory),
    });
    let stack_pointer = wmod.globals.push(GlobalData {
        ty: WType::I32,
        value: Some(u64::from(layout.stack_top)),
        mutable: true,
    });

    let funcs = module.funcs();
    let mut func_map = FxHashMap::default();
    let mut bindings = Vec::new();
    // Imports take the lowest function indices.
    for &func in &funcs {
        if !ctx.func_linkage(func).is_external() {
            continue;
        }
        let sig = ctx.get_sig(func).expect("function signature exists");
        let data = signature_data(ctx, &sig)?;
        if let Some(symbol) = RuntimeSymbol::from_name(sig.name()) {
            symbol.check_signature(&data)?;
            bindings.push((func, symbol));
            continue;
        }
        let signature = wmod.signatures.push(data);
        let name = sig.name().to_string();
        let import = wmod.funcs.push(FuncDecl::Import(signature, name.clone()));
        wmod.imports.push(Import {
            module: "env".to_string(),
            name,
            kind: ImportKind::Func(import),
        });
        func_map.insert(func, import);
    }

    let runtime =
        (!bindings.is_empty() || allocates(module)).then(|| Runtime::declare(&mut wmod, &layout));
    for (func, symbol) in bindings {
        let runtime = runtime
            .as_ref()
            .expect("runtime is declared for its symbols");
        func_map.insert(func, runtime.func(symbol));
    }

    let mut defined = Vec::new();
    for &func in &funcs {
        let sig = ctx.get_sig(func).expect("function signature exists");
        let has_body = module
            .func_store
            .try_view(func, |function| function.layout.entry_block().is_some())
            .unwrap_or(false);
        if sig.linkage().is_external() || !has_body {
            continue;
        }
        let signature = wmod.signatures.push(signature_data(ctx, &sig)?);
        let name = sig.name().to_string();
        let decl = wmod.funcs.push(FuncDecl::Body(
            signature,
            name.clone(),
            FunctionBody::default(),
        ));
        if sig.linkage() == Linkage::Public {
            if name == "memory" {
                return Err("public function `memory` collides with the memory export".into());
            }
            wmod.exports.push(Export {
                name: name.clone(),
                kind: ExportKind::Func(decl),
            });
        }
        func_map.insert(func, decl);
        defined.push((func, decl, signature, name));
    }

    let env = ModuleEnv {
        funcs: func_map,
        globals: layout.globals.clone(),
        stack_pointer,
        stack_base: layout.stack_base,
        runtime,
    };
    let effects = compute_object_effect_summaries(module);
    let mut helpers = Helpers::default();
    for (func, decl, signature, name) in defined {
        let body = module
            .func_store
            .view(func, |function| -> Result<FunctionBody, String> {
                let heap_roots = native_heap_object_roots(function, &effects);
                let b = Builder::new(&wmod, signature);
                FunctionTranslator {
                    ctx,
                    func,
                    function,
                    env: &env,
                    helpers: &mut helpers,
                    wmod: &mut wmod,
                    b,
                    heap_roots,
                    blocks: FxHashMap::default(),
                    values: FxHashMap::default(),
                    aggregate_phis: FxHashMap::default(),
                    frame_size: 0,
                    frame: None,
                    exits: Vec::new(),
                    sret: None,
                }
                .translate()
            })
            .map_err(|error| format!("failed to translate function {name}: {error}"))?;
        wmod.funcs[decl] = FuncDecl::Body(signature, name, body);
    }

    for ((helper, limbs), func) in helpers.funcs {
        define_helper(&mut wmod, func, helper, limbs);
    }
    if let Some(runtime) = &env.runtime {
        runtime.define(&mut wmod, &layout);
    }
    Ok(wmod)
}

/// Whether any function allocates from the heap.
fn allocates(module: &Module) -> bool {
    module.funcs().into_iter().any(|func| {
        module
            .func_store
            .try_view(func, |function| {
                function.layout.iter_all_insts().any(|inst| {
                    matches!(
                        native_inst_set().resolve_inst(function.dfg.inst(inst)),
                        NativeInstKind::MemAllocDynamic(_) | NativeInstKind::ObjMaterializeHeap(_)
                    )
                })
            })
            .unwrap_or(false)
    })
}

/// An aggregate phi: its value lives in `slot`, and each incoming edge first
/// copies its value to `incoming`, so no copy overwrites another phi's source.
#[derive(Clone, Copy)]
struct AggregatePhi {
    result: ValueId,
    size: u32,
    slot: u32,
    incoming: u32,
}

struct FunctionTranslator<'a> {
    ctx: &'a ModuleCtx,
    func: FuncRef,
    function: &'a Function,
    env: &'a ModuleEnv,
    helpers: &'a mut Helpers,
    wmod: &'a mut WaffleModule<'static>,
    b: Builder,
    heap_roots: FxHashSet<ValueId>,
    blocks: FxHashMap<BlockId, waffle::Block>,
    values: FxHashMap<ValueId, Repr>,
    aggregate_phis: FxHashMap<BlockId, Vec<AggregatePhi>>,
    /// Bytes of linear-memory stack this function uses.
    frame_size: u32,
    /// The frame's base address, defined in the prologue once the frame size
    /// is known.
    frame: Option<WValue>,
    /// Blocks that return, which restore the stack pointer.
    exits: Vec<waffle::Block>,
    /// The caller's buffer for an indirect result.
    sret: Option<WValue>,
}

impl FunctionTranslator<'_> {
    fn translate(mut self) -> Result<FunctionBody, String> {
        let mut cfg = ControlFlowGraph::default();
        cfg.compute(self.function);
        let mut order: Vec<_> = cfg.post_order().collect();
        order.reverse();
        let entry = self
            .function
            .layout
            .entry_block()
            .ok_or("function has no entry block")?;
        let prologue = self.b.block;

        let (returns_indirect, exported) = self.ctx.func_sig(self.func, |sig| {
            (
                returns_indirect(self.ctx, sig),
                sig.linkage() == Linkage::Public,
            )
        });
        let mut params = self.b.params().into_iter();
        if returns_indirect {
            self.sret = params.next();
        }
        for &arg in &self.function.arg_values {
            let ty = self.function.dfg.value_ty(arg);
            let count = value_types(self.ctx, ty)?.len();
            let mut repr: Repr = params.by_ref().take(count).collect();
            // Internal callers pass canonical values; hosts need not.
            if exported && let Some(IntRepr::Narrow(bits)) = int_repr(self.ctx, ty) {
                repr[0] = self.canonical(repr[0], bits);
            }
            self.values.insert(arg, repr);
        }

        for &block in &order {
            let wblock = self.b.add_block();
            self.blocks.insert(block, wblock);
        }
        for &block in &order {
            let wblock = self.blocks[&block];
            for inst in self.function.layout.iter_inst(block) {
                if self.function.dfg.downcast::<&Phi>(inst).is_none() {
                    break;
                }
                let result = self
                    .function
                    .dfg
                    .inst_result(inst)
                    .ok_or("phi has no result")?;
                let ty = self.function.dfg.value_ty(result);
                if is_indirect(self.ctx, ty) {
                    let size = value_storage_size(ty, self.ctx)?;
                    let align = self.alignment(ty)?;
                    let slot = self.reserve(size, align)?;
                    let incoming = self.reserve(size, align)?;
                    self.aggregate_phis
                        .entry(block)
                        .or_default()
                        .push(AggregatePhi {
                            result,
                            size,
                            slot,
                            incoming,
                        });
                } else {
                    let repr = value_types(self.ctx, ty)?
                        .into_iter()
                        .map(|ty| self.b.body.add_blockparam(wblock, ty))
                        .collect();
                    self.values.insert(result, repr);
                }
            }
        }

        for &block in &order {
            self.translate_block(block)?;
        }

        self.b.switch_to(prologue);
        let saved_sp = self.finish_frame();
        self.b.jump(self.blocks[&entry], vec![]);
        if let Some(saved_sp) = saved_sp {
            for &exit in &self.exits {
                self.b.body.add_op(
                    exit,
                    Operator::GlobalSet {
                        global_index: self.env.stack_pointer,
                    },
                    &[saved_sp],
                    &[],
                );
            }
        }
        Ok(self.b.body)
    }

    /// Allocates the frame in the prologue and returns the caller's stack
    /// pointer, or nothing when the function needs no frame.
    fn finish_frame(&mut self) -> Option<WValue> {
        let frame = self.frame?;
        let size = self.frame_size.next_multiple_of(16);
        let stack_pointer = self.env.stack_pointer;
        let caller = self.b.op(
            Operator::GlobalGet {
                global_index: stack_pointer,
            },
            &[],
            WType::I32,
        );
        let base = self.b.i32(self.env.stack_base as i32);
        let available = self.b.op(Operator::I32Sub, &[caller, base], WType::I32);
        let size = self.b.i32(size as i32);
        let overflow = self.b.op(Operator::I32LtU, &[available, size], WType::I32);
        let address = self.b.op(Operator::I32Sub, &[caller, size], WType::I32);
        self.b.trap_if(overflow);
        self.b.effect(
            Operator::GlobalSet {
                global_index: stack_pointer,
            },
            &[address],
        );
        self.b.body.set_alias(frame, address);
        Some(caller)
    }

    /// Reserves frame bytes and returns their offset from the frame base.
    fn reserve(&mut self, size: u32, align: u32) -> Result<u32, String> {
        if align > 16 {
            return Err(format!(
                "alignment {align} exceeds the wasm stack alignment"
            ));
        }
        let offset = self.frame_size.next_multiple_of(align.max(1));
        self.frame_size = offset
            .checked_add(size.max(1))
            .ok_or("stack frame exceeds 4 GiB")?;
        if self.frame.is_none() {
            self.frame = Some(self.b.body.add_placeholder(WType::I32));
        }
        Ok(offset)
    }

    fn frame_address(&mut self, offset: u32) -> WValue {
        let frame = self.frame.expect("frame offsets follow a reservation");
        self.offset(frame, offset)
    }

    /// A fresh frame slot holding a value of `ty`.
    fn slot(&mut self, ty: Type) -> Result<WValue, String> {
        let size = value_storage_size(ty, self.ctx)?;
        let align = self.alignment(ty)?;
        let offset = self.reserve(size, align)?;
        Ok(self.frame_address(offset))
    }

    fn alignment(&self, ty: Type) -> Result<u32, String> {
        let align = self
            .ctx
            .align_of(ty)
            .map_err(|error| format!("cannot align type {ty:?}: {error:?}"))?;
        Ok(align as u32)
    }

    fn offset(&mut self, address: WValue, offset: u32) -> WValue {
        if offset == 0 {
            address
        } else {
            let offset = self.b.i32(offset as i32);
            self.b.op(Operator::I32Add, &[address, offset], WType::I32)
        }
    }

    fn ty(&self, value: ValueId) -> Type {
        self.function.dfg.value_ty(value)
    }

    fn define(&mut self, inst: InstId, repr: Repr) {
        if let Some(result) = self.function.dfg.inst_result(inst) {
            self.values.insert(result, repr);
        }
    }

    fn define1(&mut self, inst: InstId, value: WValue) {
        self.define(inst, smallvec![value]);
    }

    fn resolve(&mut self, value: ValueId) -> Result<Repr, String> {
        if let Some(repr) = self.values.get(&value) {
            return Ok(repr.clone());
        }
        match self.function.dfg.value(value) {
            Value::Immediate { imm, .. } => Ok(self.immediate(*imm)),
            Value::Undef { ty } => {
                let ty = *ty;
                if is_indirect(self.ctx, ty) {
                    let slot = self.slot(ty)?;
                    let size = value_storage_size(ty, self.ctx)?;
                    let size = self.b.i32(size as i32);
                    self.fill_zero(slot, size);
                    Ok(smallvec![slot])
                } else {
                    // A surviving undef is zero; optimizations do not promise it.
                    Ok(value_types(self.ctx, ty)?
                        .into_iter()
                        .map(|ty| self.b.constant(ty, 0))
                        .collect())
                }
            }
            Value::Global { gv, .. } => {
                let address = self
                    .env
                    .globals
                    .get(gv)
                    .copied()
                    .ok_or_else(|| format!("undeclared global {gv:?}"))?;
                Ok(smallvec![self.b.i32(address as i32)])
            }
            value @ (Value::Arg { .. } | Value::Inst { .. }) => {
                Err(format!("unresolved SSA value {value:?}"))
            }
        }
    }

    fn resolve1(&mut self, value: ValueId) -> Result<WValue, String> {
        let repr = self.resolve(value)?;
        match repr.as_slice() {
            [single] => Ok(*single),
            _ => Err(format!(
                "value of type {:?} is not a single wasm value",
                self.ty(value)
            )),
        }
    }

    fn immediate(&mut self, imm: Immediate) -> Repr {
        match imm {
            Immediate::I1(value) => smallvec![self.b.i32(value as i32)],
            Immediate::I8(value) => smallvec![self.b.i32(value as u8 as i32)],
            Immediate::I16(value) => smallvec![self.b.i32(value as u16 as i32)],
            Immediate::I32(value) => smallvec![self.b.i32(value)],
            Immediate::I64(value) => smallvec![self.b.i64(value)],
            Immediate::I128(value) => {
                smallvec![self.b.i64(value as i64), self.b.i64((value >> 64) as i64)]
            }
            Immediate::I256(value) | Immediate::EnumTag { value, .. } => value
                .to_u256()
                .0
                .into_iter()
                .map(|limb| self.b.i64(limb as i64))
                .collect(),
        }
    }

    fn translate_block(&mut self, block: BlockId) -> Result<(), String> {
        self.b.switch_to(self.blocks[&block]);
        for phi in self.aggregate_phis.get(&block).cloned().unwrap_or_default() {
            let slot = self.frame_address(phi.slot);
            let incoming = self.frame_address(phi.incoming);
            self.copy(slot, incoming, phi.size);
            self.values.insert(phi.result, smallvec![slot]);
        }
        for inst in self.function.layout.iter_inst(block) {
            self.translate_inst(block, inst)?;
        }
        Ok(())
    }

    /// The target for an edge into `to`, passing its phi values.
    fn edge(&mut self, from: BlockId, to: BlockId) -> Result<BlockTarget, String> {
        let mut args = Vec::new();
        let mut copies = Vec::new();
        let aggregate_phis = self.aggregate_phis.get(&to).cloned().unwrap_or_default();
        for inst in self.function.layout.iter_inst(to) {
            let Some(phi) = self.function.dfg.downcast::<&Phi>(inst) else {
                break;
            };
            let result = self
                .function
                .dfg
                .inst_result(inst)
                .ok_or("phi has no result")?;
            let &(value, _) = phi
                .args()
                .iter()
                .find(|(_, pred)| *pred == from)
                .ok_or_else(|| format!("phi in {to:?} has no value from {from:?}"))?;
            if let Some(phi) = aggregate_phis.iter().find(|phi| phi.result == result) {
                copies.push((value, *phi));
            } else {
                args.extend(self.resolve(value)?);
            }
        }
        let target = self.blocks[&to];
        if copies.is_empty() {
            return Ok(BlockTarget {
                block: target,
                args,
            });
        }
        let current = self.b.block;
        let edge = self.b.add_block();
        self.b.switch_to(edge);
        for (value, phi) in copies {
            let source = self.resolve1(value)?;
            let incoming = self.frame_address(phi.incoming);
            self.copy(incoming, source, phi.size);
        }
        self.b.jump(target, args);
        self.b.switch_to(current);
        Ok(BlockTarget {
            block: edge,
            args: vec![],
        })
    }

    /// The truth of a branch condition of any integer type.
    fn truth(&mut self, value: ValueId) -> Result<WValue, String> {
        let repr = self.resolve(value)?;
        Ok(match self.int_repr(value)? {
            IntRepr::Narrow(_) => repr[0],
            IntRepr::I64 => {
                let zero = self.b.i64(0);
                self.b.op(Operator::I64Ne, &[repr[0], zero], WType::I32)
            }
            IntRepr::Wide(_) => {
                let zero = super::wide::is_zero(&mut self.b, &repr);
                self.b.op(Operator::I32Eqz, &[zero], WType::I32)
            }
        })
    }

    fn translate_inst(&mut self, block: BlockId, inst: InstId) -> Result<(), String> {
        let inst_data = self.function.dfg.inst(inst);
        match native_inst_set().resolve_inst(inst_data) {
            NativeInstKind::Phi(_) => {}
            NativeInstKind::Add(op) => self.arith(inst, Arith::Add, *op.lhs(), *op.rhs())?,
            NativeInstKind::Sub(op) => self.arith(inst, Arith::Sub, *op.lhs(), *op.rhs())?,
            NativeInstKind::Mul(op) => self.arith(inst, Arith::Mul, *op.lhs(), *op.rhs())?,
            NativeInstKind::And(op) => self.arith(inst, Arith::And, *op.lhs(), *op.rhs())?,
            NativeInstKind::Or(op) => self.arith(inst, Arith::Or, *op.lhs(), *op.rhs())?,
            NativeInstKind::Xor(op) => self.arith(inst, Arith::Xor, *op.lhs(), *op.rhs())?,
            NativeInstKind::Udiv(op) => self.div_rem(inst, false, false, *op.lhs(), *op.rhs())?,
            NativeInstKind::Sdiv(op) => self.div_rem(inst, true, false, *op.lhs(), *op.rhs())?,
            NativeInstKind::Umod(op) => self.div_rem(inst, false, true, *op.lhs(), *op.rhs())?,
            NativeInstKind::Smod(op) => self.div_rem(inst, true, true, *op.lhs(), *op.rhs())?,
            NativeInstKind::Shl(op) => self.shift(inst, ShiftKind::Shl, *op.value(), *op.bits())?,
            NativeInstKind::Shr(op) => self.shift(inst, ShiftKind::Shr, *op.value(), *op.bits())?,
            NativeInstKind::Sar(op) => self.shift(inst, ShiftKind::Sar, *op.value(), *op.bits())?,
            NativeInstKind::Neg(op) => self.neg(inst, *op.arg())?,
            NativeInstKind::Not(op) => self.not(inst, *op.arg())?,
            NativeInstKind::Bswap(op) => self.bswap(inst, *op.arg())?,
            NativeInstKind::Lt(op) => self.cmp(inst, CmpKind::Lt, false, *op.lhs(), *op.rhs())?,
            NativeInstKind::Gt(op) => self.cmp(inst, CmpKind::Gt, false, *op.lhs(), *op.rhs())?,
            NativeInstKind::Le(op) => self.cmp(inst, CmpKind::Le, false, *op.lhs(), *op.rhs())?,
            NativeInstKind::Ge(op) => self.cmp(inst, CmpKind::Ge, false, *op.lhs(), *op.rhs())?,
            NativeInstKind::Slt(op) => self.cmp(inst, CmpKind::Lt, true, *op.lhs(), *op.rhs())?,
            NativeInstKind::Sgt(op) => self.cmp(inst, CmpKind::Gt, true, *op.lhs(), *op.rhs())?,
            NativeInstKind::Sle(op) => self.cmp(inst, CmpKind::Le, true, *op.lhs(), *op.rhs())?,
            NativeInstKind::Sge(op) => self.cmp(inst, CmpKind::Ge, true, *op.lhs(), *op.rhs())?,
            NativeInstKind::Eq(op) => self.cmp(inst, CmpKind::Eq, false, *op.lhs(), *op.rhs())?,
            NativeInstKind::Ne(op) => self.cmp(inst, CmpKind::Ne, false, *op.lhs(), *op.rhs())?,
            NativeInstKind::IsZero(op) => self.is_zero(inst, *op.lhs())?,
            NativeInstKind::Uaddo(op) => {
                self.overflow(inst, OverflowOp::Uadd, *op.lhs(), *op.rhs())?
            }
            NativeInstKind::Saddo(op) => {
                self.overflow(inst, OverflowOp::Sadd, *op.lhs(), *op.rhs())?
            }
            NativeInstKind::Usubo(op) => {
                self.overflow(inst, OverflowOp::Usub, *op.lhs(), *op.rhs())?
            }
            NativeInstKind::Ssubo(op) => {
                self.overflow(inst, OverflowOp::Ssub, *op.lhs(), *op.rhs())?
            }
            NativeInstKind::Umulo(op) => {
                self.overflow(inst, OverflowOp::Umul, *op.lhs(), *op.rhs())?
            }
            NativeInstKind::Smulo(op) => {
                self.overflow(inst, OverflowOp::Smul, *op.lhs(), *op.rhs())?
            }
            NativeInstKind::Snego(op) => self.snego(inst, *op.arg())?,
            NativeInstKind::Uaddsat(op) => {
                self.saturating(inst, SaturatingOp::Uadd, *op.lhs(), *op.rhs())?
            }
            NativeInstKind::Saddsat(op) => {
                self.saturating(inst, SaturatingOp::Sadd, *op.lhs(), *op.rhs())?
            }
            NativeInstKind::Usubsat(op) => {
                self.saturating(inst, SaturatingOp::Usub, *op.lhs(), *op.rhs())?
            }
            NativeInstKind::Ssubsat(op) => {
                self.saturating(inst, SaturatingOp::Ssub, *op.lhs(), *op.rhs())?
            }
            NativeInstKind::Umulsat(op) => {
                self.saturating(inst, SaturatingOp::Umul, *op.lhs(), *op.rhs())?
            }
            NativeInstKind::Smulsat(op) => {
                self.saturating(inst, SaturatingOp::Smul, *op.lhs(), *op.rhs())?
            }
            NativeInstKind::Sext(op) => self.extend(inst, *op.from(), *op.ty(), true)?,
            NativeInstKind::Zext(op) => self.extend(inst, *op.from(), *op.ty(), false)?,
            NativeInstKind::Trunc(op) => self.trunc(inst, *op.from(), *op.ty())?,
            NativeInstKind::Bitcast(op) => self.bitcast(inst, *op.from(), *op.ty())?,
            NativeInstKind::IntToPtr(op) => {
                let address = self.address(*op.from())?;
                self.define1(inst, address);
            }
            NativeInstKind::PtrToInt(op) => self.trunc(inst, *op.from(), *op.ty())?,
            NativeInstKind::Jump(jump) => {
                let target = self.edge(block, *jump.dest())?;
                self.b.terminate(Terminator::Br { target });
            }
            NativeInstKind::Br(br) => {
                let cond = self.truth(*br.cond())?;
                let if_true = self.edge(block, *br.nz_dest())?;
                let if_false = self.edge(block, *br.z_dest())?;
                self.b.terminate(Terminator::CondBr {
                    cond,
                    if_true,
                    if_false,
                });
            }
            NativeInstKind::BrTable(br_table) => {
                let scrutinee = *br_table.scrutinee();
                let table = br_table.table().to_vec();
                let default = *br_table.default();
                for (case, dest) in table {
                    let matches = self.equal(scrutinee, case)?;
                    let next = self.b.add_block();
                    let if_true = self.edge(block, dest)?;
                    self.b.terminate(Terminator::CondBr {
                        cond: matches,
                        if_true,
                        if_false: BlockTarget {
                            block: next,
                            args: vec![],
                        },
                    });
                    self.b.switch_to(next);
                }
                match default {
                    Some(default) => {
                        let target = self.edge(block, default)?;
                        self.b.terminate(Terminator::Br { target });
                    }
                    None => self.b.terminate(Terminator::Unreachable),
                }
            }
            NativeInstKind::Return(ret) => {
                let args = ret.args().as_slice().to_vec();
                let values = if let Some(sret) = self.sret {
                    let [value] = args.as_slice() else {
                        return Err("indirect return requires exactly one value".into());
                    };
                    let source = self.resolve1(*value)?;
                    let storage = indirect_return_storage(self.ctx, self.ty(*value))
                        .ok_or("indirect return value has no storage type")?;
                    let size = value_storage_size(storage, self.ctx)?;
                    self.copy(sret, source, size);
                    vec![]
                } else {
                    let mut values = Vec::new();
                    for value in args {
                        values.extend(self.resolve(value)?);
                    }
                    values
                };
                self.exits.push(self.b.block);
                self.b.terminate(Terminator::Return { values });
            }
            NativeInstKind::Unreachable(_) => self.b.terminate(Terminator::Unreachable),
            NativeInstKind::Call(call) => {
                let callee = *call.callee();
                let args = call.args().to_vec();
                self.call(inst, callee, &args)?;
            }
            NativeInstKind::Mload(mload) => {
                let address = self.address(*mload.addr())?;
                let ty = *mload.ty();
                let value = self.load(ty, address, 0, true)?;
                self.define(inst, value);
            }
            NativeInstKind::Mstore(mstore) => {
                let address = self.address(*mstore.addr())?;
                let value = self.resolve(*mstore.value())?;
                self.store(*mstore.ty(), &value, address, 0)?;
            }
            NativeInstKind::Memzero(memzero) => {
                let dest = self.address(*memzero.dest())?;
                let len = self.address(*memzero.len())?;
                self.fill_zero(dest, len);
            }
            NativeInstKind::Gep(gep) => {
                let values = gep.values().to_vec();
                let address = self.gep(&values)?;
                self.define1(inst, address);
            }
            NativeInstKind::Alloca(alloca) => {
                let slot = self.slot(*alloca.ty())?;
                self.define1(inst, slot);
            }
            NativeInstKind::ObjAlloc(alloc) => {
                let ty = *alloc.ty();
                let result = self
                    .function
                    .dfg
                    .inst_result(inst)
                    .ok_or("obj.alloc has no result")?;
                let address = if self.heap_roots.contains(&result) {
                    let size = value_storage_size(ty, self.ctx)?;
                    let size = self.b.i32(size as i32);
                    self.allocate(size)?
                } else {
                    self.slot(ty)?
                };
                self.define1(inst, address);
            }
            NativeInstKind::ObjProj(proj) => {
                let values = proj.values().to_vec();
                let address = self.projection(&values, "obj.proj")?;
                self.define1(inst, address);
            }
            NativeInstKind::ConstProj(proj) => {
                let values = proj.values().to_vec();
                let address = self.projection(&values, "const.proj")?;
                self.define1(inst, address);
            }
            NativeInstKind::ObjIndex(index) => {
                let address = self.element(*index.object(), *index.index())?;
                self.define1(inst, address);
            }
            NativeInstKind::ConstIndex(index) => {
                let address = self.element(*index.object(), *index.index())?;
                self.define1(inst, address);
            }
            NativeInstKind::ObjLoad(load) => {
                let address = self.resolve1(*load.object())?;
                let ty = self.result_ty(inst)?;
                let value = self.load(ty, address, 0, true)?;
                self.define(inst, value);
            }
            NativeInstKind::ConstLoad(load) => {
                // Constant storage never changes, so aggregates can stay in place.
                let address = self.resolve1(*load.object())?;
                let ty = self.result_ty(inst)?;
                let value = self.load(ty, address, 0, false)?;
                self.define(inst, value);
            }
            NativeInstKind::ObjStore(store) => {
                let address = self.resolve1(*store.object())?;
                let ty = self.ty(*store.value());
                let value = self.resolve(*store.value())?;
                self.store(ty, &value, address, 0)?;
            }
            NativeInstKind::ConstRef(const_ref) => {
                let gv = const_ref.global().gv();
                let address = self
                    .env
                    .globals
                    .get(&gv)
                    .copied()
                    .ok_or_else(|| format!("undeclared global {gv:?}"))?;
                let address = self.b.i32(address as i32);
                self.define1(inst, address);
            }
            NativeInstKind::ObjMaterializeStack(materialize) => {
                let object = self.resolve(*materialize.object())?;
                self.define(inst, object);
            }
            NativeInstKind::ObjMaterializeHeap(materialize) => {
                let object = self.resolve(*materialize.object())?;
                self.define(inst, object);
            }
            NativeInstKind::MemAllocDynamic(alloc) => {
                let size = self.size(*alloc.size())?;
                let address = self.allocate(size)?;
                self.define1(inst, address);
            }
            NativeInstKind::InsertValue(insert) => {
                self.insert_value(block, inst, *insert.dest(), *insert.idx(), *insert.value())?
            }
            NativeInstKind::ExtractValue(extract) => {
                self.extract_value(inst, *extract.dest(), *extract.idx())?
            }
            NativeInstKind::EnumMake(_)
            | NativeInstKind::EnumTag(_)
            | NativeInstKind::EnumIsVariant(_)
            | NativeInstKind::EnumAssertVariant(_)
            | NativeInstKind::EnumAssertVariantRef(_)
            | NativeInstKind::EnumExtract(_)
            | NativeInstKind::EnumSetTag(_)
            | NativeInstKind::EnumWriteVariant(_)
            | NativeInstKind::EnumGetTag(_)
            | NativeInstKind::EnumProj(_) => {
                return Err(format!(
                    "enum instruction {:?} survived EnumLowerToProduct",
                    inst_data.kind()
                ));
            }
            NativeInstKind::GetFunctionPtr(_)
            | NativeInstKind::SymAddr(_)
            | NativeInstKind::SymSize(_) => {
                return Err(format!(
                    "symbolic instruction {:?} is not supported by the wasm backend",
                    inst_data.kind()
                ));
            }
        }
        Ok(())
    }

    fn result_ty(&self, inst: InstId) -> Result<Type, String> {
        let result = self
            .function
            .dfg
            .inst_result(inst)
            .ok_or("instruction has no result")?;
        Ok(self.ty(result))
    }

    fn call(&mut self, inst: InstId, callee: FuncRef, args: &[ValueId]) -> Result<(), String> {
        let func = *self
            .env
            .funcs
            .get(&callee)
            .ok_or_else(|| format!("unknown callee {callee:?}"))?;
        let results = self.function.dfg.inst_results(inst).to_vec();
        let storage = match results.as_slice() {
            [result] => indirect_return_storage(self.ctx, self.ty(*result)),
            _ => None,
        };
        let mut call_args = Vec::new();
        let sret = storage.map(|storage| self.slot(storage)).transpose()?;
        call_args.extend(sret);
        for &arg in args {
            call_args.extend(self.resolve(arg)?);
        }
        let mut result_tys = Vec::new();
        if sret.is_none() {
            for &result in &results {
                result_tys.extend(value_types(self.ctx, self.ty(result))?);
            }
        }
        let outputs = self.b.results(
            Operator::Call {
                function_index: func,
            },
            &call_args,
            &result_tys,
        );
        if let Some(sret) = sret {
            self.values.insert(results[0], smallvec![sret]);
            return Ok(());
        }
        let imported = self.ctx.func_linkage(callee).is_external();
        let mut outputs = outputs.into_iter();
        for result in results {
            let ty = self.ty(result);
            let count = value_types(self.ctx, ty)?.len();
            let mut repr: Repr = outputs.by_ref().take(count).collect();
            // Host functions need not keep narrow results canonical.
            if imported && let Some(IntRepr::Narrow(bits)) = int_repr(self.ctx, ty) {
                repr[0] = self.canonical(repr[0], bits);
            }
            self.values.insert(result, repr);
        }
        Ok(())
    }

    /// Allocates `size` bytes from the heap, trapping when memory runs out.
    fn allocate(&mut self, size: WValue) -> Result<WValue, String> {
        let runtime = self
            .env
            .runtime
            .as_ref()
            .ok_or("heap allocation requires the wasm runtime")?;
        let address = self.b.op(
            Operator::Call {
                function_index: runtime.func(RuntimeSymbol::Malloc),
            },
            &[size],
            WType::I32,
        );
        let failed = self.b.op(Operator::I32Eqz, &[address], WType::I32);
        self.b.trap_if(failed);
        Ok(address)
    }

    /// A byte count as an `i32`, trapping when it does not fit.
    fn size(&mut self, value: ValueId) -> Result<WValue, String> {
        let repr = self.resolve(value)?;
        if let IntRepr::Narrow(_) = self.int_repr(value)? {
            return Ok(repr[0]);
        }
        let thirty_two = self.b.i64(32);
        let mut overflow = self
            .b
            .op(Operator::I64ShrU, &[repr[0], thirty_two], WType::I64);
        for &limb in &repr[1..] {
            overflow = self.b.op(Operator::I64Or, &[overflow, limb], WType::I64);
        }
        let zero = self.b.i64(0);
        let overflow = self.b.op(Operator::I64Ne, &[overflow, zero], WType::I32);
        self.b.trap_if(overflow);
        Ok(self.b.op(Operator::I32WrapI64, &[repr[0]], WType::I32))
    }

    /// The low 32 bits of an integer or pointer value, as an address.
    fn address(&mut self, value: ValueId) -> Result<WValue, String> {
        let repr = self.resolve(value)?;
        Ok(match self.int_repr(value)? {
            IntRepr::Narrow(_) => repr[0],
            IntRepr::I64 | IntRepr::Wide(_) => {
                self.b.op(Operator::I32WrapI64, &[repr[0]], WType::I32)
            }
        })
    }

    fn int_repr(&self, value: ValueId) -> Result<IntRepr, String> {
        let ty = self.ty(value);
        int_repr(self.ctx, ty).ok_or_else(|| format!("expected an integer, found {ty:?}"))
    }
}
