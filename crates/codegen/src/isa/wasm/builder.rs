use smallvec::{SmallVec, smallvec};
use waffle::{
    Block, BlockTarget, FunctionBody, Memory, MemoryArg, Module as WaffleModule, Operator,
    Signature, Terminator, Type as WType, Value as WValue, ValueDef, entity::EntityRef,
};

/// The WAFFLE values carrying one Sonatina value: none for `unit`, one per
/// scalar, pointer or aggregate address, and one `i64` limb per 64 bits of
/// `i128` and `i256`.
pub(super) type Repr = SmallVec<[WValue; 4]>;

pub(super) fn memarg(align: u32, offset: u32) -> MemoryArg {
    MemoryArg {
        align,
        offset,
        memory: Memory::new(0),
    }
}

/// A function body under construction and its current block.
pub(super) struct Builder {
    pub(super) body: FunctionBody,
    pub(super) block: Block,
}

impl Builder {
    pub(super) fn new(module: &WaffleModule, sig: Signature) -> Self {
        let body = FunctionBody::new(module, sig);
        let block = body.entry;
        Self { body, block }
    }

    pub(super) fn params(&self) -> Vec<WValue> {
        self.body.blocks[self.body.entry]
            .params
            .iter()
            .map(|&(_, value)| value)
            .collect()
    }

    pub(super) fn op(&mut self, op: Operator, args: &[WValue], ty: WType) -> WValue {
        self.body.add_op(self.block, op, args, &[ty])
    }

    pub(super) fn effect(&mut self, op: Operator, args: &[WValue]) {
        self.body.add_op(self.block, op, args, &[]);
    }

    /// Adds an operator with any number of results.
    pub(super) fn results(&mut self, op: Operator, args: &[WValue], tys: &[WType]) -> Repr {
        match tys {
            [] => {
                self.effect(op, args);
                Repr::new()
            }
            [ty] => smallvec![self.op(op, args, *ty)],
            _ => {
                let value = self.body.add_op(self.block, op, args, tys);
                tys.iter()
                    .enumerate()
                    .map(|(index, &ty)| {
                        let pick =
                            self.body
                                .add_value(ValueDef::PickOutput(value, index as u32, ty));
                        self.body.append_to_block(self.block, pick);
                        pick
                    })
                    .collect()
            }
        }
    }

    pub(super) fn i32(&mut self, value: i32) -> WValue {
        self.op(
            Operator::I32Const {
                value: value as u32,
            },
            &[],
            WType::I32,
        )
    }

    pub(super) fn i64(&mut self, value: i64) -> WValue {
        self.op(
            Operator::I64Const {
                value: value as u64,
            },
            &[],
            WType::I64,
        )
    }

    pub(super) fn constant(&mut self, ty: WType, value: i64) -> WValue {
        match ty {
            WType::I32 => self.i32(value as i32),
            _ => self.i64(value),
        }
    }

    /// `if_true` when `cond` is nonzero, otherwise `if_false`.
    pub(super) fn select(
        &mut self,
        cond: WValue,
        if_true: WValue,
        if_false: WValue,
        ty: WType,
    ) -> WValue {
        self.op(Operator::Select, &[if_true, if_false, cond], ty)
    }

    pub(super) fn add_block(&mut self) -> Block {
        self.body.add_block()
    }

    pub(super) fn switch_to(&mut self, block: Block) {
        self.block = block;
    }

    pub(super) fn terminate(&mut self, terminator: Terminator) {
        self.body.set_terminator(self.block, terminator);
    }

    pub(super) fn jump(&mut self, block: Block, args: Vec<WValue>) {
        self.terminate(Terminator::Br {
            target: BlockTarget { block, args },
        });
    }

    pub(super) fn branch(&mut self, cond: WValue, if_true: Block, if_false: Block) {
        self.terminate(Terminator::CondBr {
            cond,
            if_true: BlockTarget {
                block: if_true,
                args: vec![],
            },
            if_false: BlockTarget {
                block: if_false,
                args: vec![],
            },
        });
    }

    /// Traps when `cond` is nonzero and continues in a new block otherwise.
    pub(super) fn trap_if(&mut self, cond: WValue) {
        let trap = self.add_block();
        let next = self.add_block();
        self.branch(cond, trap, next);
        self.body.set_terminator(trap, Terminator::Unreachable);
        self.switch_to(next);
    }
}
