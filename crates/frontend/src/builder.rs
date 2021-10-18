use sonatina_codegen::ir::{
    function::{CursorLocation, FunctionCursor},
    insn::{BinaryOp, BranchOp, CastOp, ImmediateOp, InsnData, JumpOp},
    types::U256,
    DataFlowGraph, Function, Layout, Signature,
};

use super::{Block, Type, Value};

pub trait Context {
    fn hash_type(&self) -> Type;
    fn address_type(&self) -> Type;
    fn balance_type(&self) -> Type;
    fn gas_type(&self) -> Type;
}

pub struct FunctionBuilder<'a> {
    ctxt: &'a dyn Context,
    func: Function,
    loc: CursorLocation,
}

macro_rules! impl_immediate_insn {
    ($name:ident, $code:path, $ty:ty) => {
        pub fn $name(&mut self, value: $ty) -> Value {
            let insn_data = InsnData::Immediate { code: $code(value) };

            self.insert_insn(insn_data).unwrap()
        }
    };
}

macro_rules! impl_binary_insn {
    ($name:ident, $code:path) => {
        pub fn $name(&mut self, lhs: Value, rhs: Value) -> Value {
            let insn_data = InsnData::Binary {
                code: $code,
                args: [lhs, rhs],
            };

            self.insert_insn(insn_data).unwrap()
        }
    };
}

macro_rules! impl_cast_insn {
    ($name:ident, $code:path) => {
        pub fn $name(&mut self, lhs: Value, ty: Type) -> Value {
            let insn_data = InsnData::Cast {
                code: $code,
                args: [lhs],
                ty,
            };

            self.insert_insn(insn_data).unwrap()
        }
    };
}

macro_rules! impl_branch_insn {
    ($name:ident, $code:path) => {
        pub fn $name(&mut self, dest: Block, cond: Value, params: Vec<Value>) {
            let insn_data = InsnData::Branch {
                code: $code,
                args: [cond],
                dest,
                params,
            };
            self.insert_insn(insn_data);
        }
    };
}

impl<'a> FunctionBuilder<'a> {
    pub fn new(name: String, sig: Signature, ctxt: &'a dyn Context) -> Self {
        let func = Function {
            name,
            sig,
            dfg: DataFlowGraph::default(),
            layout: Layout::default(),
        };

        Self {
            ctxt,
            func,
            loc: CursorLocation::NoWhere,
        }
    }

    pub fn make_entry_block(&mut self) -> (Block, Vec<Value>) {
        let block = self.append_block();
        let args = self.func.sig.args();
        let mut params = Vec::with_capacity(args.len());
        for param in args {
            let param_value = self.func.dfg.append_block_param(block, param.clone());
            params.push(param_value);
        }

        (block, params)
    }

    pub fn append_block(&mut self) -> Block {
        self.cursor().append_block()
    }

    pub fn append_block_param(&mut self, block: Block, ty: Type) -> Value {
        self.func.dfg.append_block_param(block, ty)
    }

    pub fn switch_to_block_top(&mut self, block: Block) {
        self.loc = CursorLocation::BlockTop(block)
    }

    pub fn switch_to_block_bottom(&mut self, block: Block) {
        self.loc = CursorLocation::BlockBottom(block);
    }

    impl_immediate_insn!(imm_i8, ImmediateOp::I8, i8);
    impl_immediate_insn!(imm_i16, ImmediateOp::I16, i16);
    impl_immediate_insn!(imm_i32, ImmediateOp::I32, i32);
    impl_immediate_insn!(imm_i64, ImmediateOp::I64, i64);
    impl_immediate_insn!(imm_i128, ImmediateOp::I128, i128);
    impl_immediate_insn!(imm_u8, ImmediateOp::U8, u8);
    impl_immediate_insn!(imm_u16, ImmediateOp::U16, u16);
    impl_immediate_insn!(imm_u32, ImmediateOp::U32, u32);
    impl_immediate_insn!(imm_u64, ImmediateOp::U64, u64);
    impl_immediate_insn!(imm_u128, ImmediateOp::U128, u128);
    impl_immediate_insn!(imm_u256, ImmediateOp::U256, U256);

    impl_binary_insn!(add, BinaryOp::Add);
    impl_binary_insn!(sub, BinaryOp::Sub);
    impl_binary_insn!(mul, BinaryOp::Mul);
    impl_binary_insn!(div, BinaryOp::Div);
    impl_binary_insn!(lt, BinaryOp::Lt);
    impl_binary_insn!(gt, BinaryOp::Gt);
    impl_binary_insn!(slt, BinaryOp::Slt);
    impl_binary_insn!(sgt, BinaryOp::Sgt);
    impl_binary_insn!(eq, BinaryOp::Eq);
    impl_binary_insn!(and, BinaryOp::And);
    impl_binary_insn!(or, BinaryOp::Or);

    impl_cast_insn!(sext, CastOp::Sext);
    impl_cast_insn!(zext, CastOp::Zext);
    impl_cast_insn!(trunc, CastOp::Trunc);

    pub fn load(&mut self, addr: Value, ty: Type) -> Value {
        let insn_data = InsnData::Load { args: [addr], ty };
        self.insert_insn(insn_data).unwrap()
    }

    pub fn store(&mut self, addr: Value, data: Value) {
        let insn_data = InsnData::Store { args: [addr, data] };
        self.insert_insn(insn_data);
    }

    pub fn jump(&mut self, dest: Block, params: Vec<Value>) {
        let insn_data = InsnData::Jump {
            code: JumpOp::Jump,
            dest,
            params,
        };
        self.insert_insn(insn_data);
    }

    impl_branch_insn!(brz, BranchOp::Brz);
    impl_branch_insn!(brnz, BranchOp::Brnz);

    pub fn build(self) -> Function {
        self.func
    }

    //TODO:
    // pub fn build_with_verifier(self, verifier: IrVerifier) -> Self {
    // todo!()
    //}

    pub fn type_of(&self, value: Value) -> &Type {
        self.func.dfg.value_ty(value)
    }

    pub fn hash_type(&self) -> Type {
        self.ctxt.hash_type()
    }

    pub fn address_type(&self) -> Type {
        self.ctxt.address_type()
    }

    pub fn balance_type(&self) -> Type {
        self.ctxt.balance_type()
    }

    pub fn gas_type(&self) -> Type {
        self.ctxt.gas_type()
    }

    fn cursor(&mut self) -> FunctionCursor {
        FunctionCursor::new(&mut self.func, self.loc)
    }

    fn insert_insn(&mut self, insn_data: InsnData) -> Option<Value> {
        let (insn, value) = self.cursor().insert_insn(insn_data);
        self.loc = CursorLocation::At(insn);
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use sonatina_codegen::ir::ir_writer::FuncWriter;

    pub struct TestContext {}

    impl Context for TestContext {
        fn hash_type(&self) -> Type {
            Type::I256
        }

        fn address_type(&self) -> Type {
            Type::make_array(Type::I8, 20)
        }

        fn balance_type(&self) -> Type {
            Type::I256
        }

        fn gas_type(&self) -> Type {
            Type::I256
        }
    }

    #[test]
    fn entry_block() {
        let sig = Signature::default();
        let ctxt = TestContext {};
        let mut builder = FunctionBuilder::new("test_func".into(), sig, &ctxt);

        let (entry_block, _) = builder.make_entry_block();
        builder.switch_to_block_top(entry_block);
        let v0 = builder.imm_i8(1);
        let v1 = builder.imm_i8(2);
        let v2 = builder.add(v0, v1);
        builder.sub(v2, v0);

        let func = builder.build();
        let mut writer = FuncWriter::new(&func);
        let result = writer.dump_string().unwrap();
        assert_eq!(
            result,
            "func %test_func():
    block0():
        %v0.i8 = imm.i8 1
        %v1.i8 = imm.i8 2
        %v2.i8 = add %v0.i8 %v1.i8
        %v3.i8 = sub %v2.i8 %v0.i8
"
        );
    }

    #[test]
    fn entry_block_with_args() {
        let mut sig = Signature::default();
        sig.append_arg(Type::I32);
        sig.append_arg(Type::I64);
        let ctxt = TestContext {};
        let mut builder = FunctionBuilder::new("test_func".into(), sig, &ctxt);

        let (entry_block, args) = builder.make_entry_block();
        builder.switch_to_block_top(entry_block);
        let v3 = builder.sext(args[0], Type::I64);
        builder.mul(v3, args[1]);

        let func = builder.build();
        let mut writer = FuncWriter::new(&func);
        let result = writer.dump_string().unwrap();
        assert_eq!(
            result,
            "func %test_func(i32, i64):
    block0(%v0.i32, %v1.i64):
        %v2.i64 = sext %v0.i32
        %v3.i64 = mul %v2.i64 %v1.i64
"
        );
    }
}