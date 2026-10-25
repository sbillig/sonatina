use smallvec::smallvec;
use sonatina_ir::{BlockId, InstId, Type, ValueId, inst::data::InsertValue, types::CompoundType};
use waffle::{Memory, Operator, Type as WType, Value as WValue, entity::EntityRef};

use super::FunctionTranslator;
use crate::isa::{
    native_layout::{
        aggregate_elem_offset, compute_element_size, constant_value_index, value_storage_size,
    },
    wasm::{
        abi::{IntRepr, is_indirect, is_reference},
        builder::{Repr, memarg},
    },
};

/// Aggregates up to this size copy through registers rather than
/// `memory.copy`.
const INLINE_COPY_BYTES: u32 = 64;

impl FunctionTranslator<'_> {
    /// Loads a value of `ty` from `address + offset`. With `snapshot`, an
    /// aggregate is copied to a fresh slot so later stores do not change it;
    /// otherwise it stays in place.
    pub(super) fn load(
        &mut self,
        ty: Type,
        address: WValue,
        offset: u32,
        snapshot: bool,
    ) -> Result<Repr, String> {
        if is_indirect(self.ctx, ty) {
            let source = self.offset(address, offset);
            if !snapshot {
                return Ok(smallvec![source]);
            }
            let slot = self.slot(ty)?;
            let size = value_storage_size(ty, self.ctx)?;
            self.copy(slot, source, size);
            return Ok(smallvec![slot]);
        }
        Ok(match ty {
            Type::Unit => smallvec![],
            Type::I1 => {
                let value = self.b.op(
                    Operator::I32Load8U {
                        memory: memarg(0, offset),
                    },
                    &[address],
                    WType::I32,
                );
                smallvec![self.canonical(value, 1)]
            }
            Type::I8 => smallvec![self.b.op(
                Operator::I32Load8U {
                    memory: memarg(0, offset),
                },
                &[address],
                WType::I32,
            )],
            Type::I16 => smallvec![self.b.op(
                Operator::I32Load16U {
                    memory: memarg(1, offset),
                },
                &[address],
                WType::I32,
            )],
            Type::I32 => smallvec![self.b.op(
                Operator::I32Load {
                    memory: memarg(2, offset),
                },
                &[address],
                WType::I32,
            )],
            Type::I64 | Type::I128 | Type::I256 => {
                let limbs = value_storage_size(ty, self.ctx)? / 8;
                (0..limbs)
                    .map(|limb| {
                        self.b.op(
                            Operator::I64Load {
                                memory: memarg(3, offset + limb * 8),
                            },
                            &[address],
                            WType::I64,
                        )
                    })
                    .collect()
            }
            _ if is_reference(self.ctx, ty) => smallvec![self.b.op(
                Operator::I32Load {
                    memory: memarg(2, offset),
                },
                &[address],
                WType::I32,
            )],
            _ => return Err(format!("cannot load a value of type {ty:?}")),
        })
    }

    pub(super) fn store(
        &mut self,
        ty: Type,
        value: &Repr,
        address: WValue,
        offset: u32,
    ) -> Result<(), String> {
        if is_indirect(self.ctx, ty) {
            let dest = self.offset(address, offset);
            let size = value_storage_size(ty, self.ctx)?;
            self.copy(dest, value[0], size);
            return Ok(());
        }
        match ty {
            Type::Unit => {}
            Type::I1 | Type::I8 => self.b.effect(
                Operator::I32Store8 {
                    memory: memarg(0, offset),
                },
                &[address, value[0]],
            ),
            Type::I16 => self.b.effect(
                Operator::I32Store16 {
                    memory: memarg(1, offset),
                },
                &[address, value[0]],
            ),
            Type::I64 | Type::I128 | Type::I256 => {
                for (limb, &value) in value.iter().enumerate() {
                    self.b.effect(
                        Operator::I64Store {
                            memory: memarg(3, offset + limb as u32 * 8),
                        },
                        &[address, value],
                    );
                }
            }
            _ if ty == Type::I32 || is_reference(self.ctx, ty) => self.b.effect(
                Operator::I32Store {
                    memory: memarg(2, offset),
                },
                &[address, value[0]],
            ),
            _ => return Err(format!("cannot store a value of type {ty:?}")),
        }
        Ok(())
    }

    /// Copies `size` bytes. Small copies read every chunk before writing, so
    /// they stay correct for overlapping ranges, as `memory.copy` does.
    pub(super) fn copy(&mut self, dest: WValue, source: WValue, size: u32) {
        if size > INLINE_COPY_BYTES {
            let len = self.b.i32(size as i32);
            self.b.effect(
                Operator::MemoryCopy {
                    dst_mem: Memory::new(0),
                    src_mem: Memory::new(0),
                },
                &[dest, source, len],
            );
            return;
        }
        let chunks: Vec<_> = storage_chunks(size)
            .map(|(offset, width)| {
                let (op, ty) = match width {
                    1 => (
                        Operator::I32Load8U {
                            memory: memarg(0, offset),
                        },
                        WType::I32,
                    ),
                    2 => (
                        Operator::I32Load16U {
                            memory: memarg(0, offset),
                        },
                        WType::I32,
                    ),
                    4 => (
                        Operator::I32Load {
                            memory: memarg(0, offset),
                        },
                        WType::I32,
                    ),
                    _ => (
                        Operator::I64Load {
                            memory: memarg(0, offset),
                        },
                        WType::I64,
                    ),
                };
                (offset, width, self.b.op(op, &[source], ty))
            })
            .collect();
        for (offset, width, value) in chunks {
            let op = match width {
                1 => Operator::I32Store8 {
                    memory: memarg(0, offset),
                },
                2 => Operator::I32Store16 {
                    memory: memarg(0, offset),
                },
                4 => Operator::I32Store {
                    memory: memarg(0, offset),
                },
                _ => Operator::I64Store {
                    memory: memarg(0, offset),
                },
            };
            self.b.effect(op, &[dest, value]);
        }
    }

    pub(super) fn fill_zero(&mut self, dest: WValue, len: WValue) {
        let zero = self.b.i32(0);
        self.b.effect(
            Operator::MemoryFill {
                mem: Memory::new(0),
            },
            &[dest, zero, len],
        );
    }

    /// An index as an `i32`, sign-extending narrow signed indices.
    fn index(&mut self, value: ValueId, signed: bool) -> Result<WValue, String> {
        let repr = self.int_repr(value)?;
        let index = self.resolve(value)?;
        Ok(match repr {
            IntRepr::Narrow(bits) if signed => self.signed32(index[0], bits),
            IntRepr::Narrow(_) => index[0],
            IntRepr::I64 | IntRepr::Wide(_) => {
                self.b.op(Operator::I32WrapI64, &[index[0]], WType::I32)
            }
        })
    }

    fn scaled(&mut self, address: WValue, index: WValue, size: u32) -> WValue {
        let size = self.b.i32(size as i32);
        let offset = self.b.op(Operator::I32Mul, &[index, size], WType::I32);
        self.b.op(Operator::I32Add, &[address, offset], WType::I32)
    }

    pub(super) fn gep(&mut self, values: &[ValueId]) -> Result<WValue, String> {
        let (&base, indices) = values.split_first().ok_or("gep requires a base pointer")?;
        let mut address = self.resolve1(base)?;
        let mut current = self.ty(base);
        for &index in indices {
            match current.resolve_compound(self.ctx) {
                Some(CompoundType::Ptr(elem) | CompoundType::Array { elem, .. }) => {
                    let size = value_storage_size(elem, self.ctx)?;
                    let index = self.index(index, true)?;
                    address = self.scaled(address, index, size);
                    current = elem;
                }
                Some(CompoundType::Struct(_)) => {
                    let field = constant_value_index(self.function, index, "gep")?;
                    let (offset, field_ty) = aggregate_elem_offset(self.ctx, current, field)?;
                    address = self.offset(address, offset as u32);
                    current = field_ty;
                }
                compound => return Err(format!("cannot index through {compound:?}")),
            }
        }
        Ok(address)
    }

    /// The address of a field reached through constant indices.
    pub(super) fn projection(&mut self, values: &[ValueId], name: &str) -> Result<WValue, String> {
        let (&base, indices) = values
            .split_first()
            .ok_or_else(|| format!("{name} requires a base value"))?;
        let address = self.resolve1(base)?;
        let mut offset = 0;
        let mut current = self.ty(base);
        for &index in indices {
            let index = constant_value_index(self.function, index, name)?;
            let (field_offset, field_ty) = aggregate_elem_offset(self.ctx, current, index)?;
            offset += field_offset as u32;
            current = field_ty;
        }
        Ok(self.offset(address, offset))
    }

    /// The address of an array element.
    pub(super) fn element(&mut self, object: ValueId, index: ValueId) -> Result<WValue, String> {
        let address = self.resolve1(object)?;
        let size = compute_element_size(self.ty(object), self.ctx)?;
        let index = self.index(index, false)?;
        Ok(self.scaled(address, index, size as u32))
    }

    pub(super) fn extract_value(
        &mut self,
        inst: InstId,
        dest: ValueId,
        idx: ValueId,
    ) -> Result<(), String> {
        let base = self.resolve1(dest)?;
        let idx = constant_value_index(self.function, idx, "extract_value")?;
        let (offset, elem_ty) = aggregate_elem_offset(self.ctx, self.ty(dest), idx)?;
        let result_ty = self.result_ty(inst)?;
        if result_ty != elem_ty {
            return Err(format!(
                "extract_value element type mismatch: expected {elem_ty:?}, got {result_ty:?}"
            ));
        }
        // Values never change, so an aggregate field can stay in its parent.
        let value = self.load(elem_ty, base, offset as u32, false)?;
        self.define(inst, value);
        Ok(())
    }

    pub(super) fn insert_value(
        &mut self,
        block: BlockId,
        inst: InstId,
        dest: ValueId,
        idx: ValueId,
        value: ValueId,
    ) -> Result<(), String> {
        let result_ty = self.result_ty(inst)?;
        let source = self.resolve1(dest)?;
        // An insert result owns its storage, so its sole consumer in the same
        // block can continue construction there. Stay in the block: a use in a
        // nested loop can run repeatedly for one definition.
        let reuse = self
            .function
            .dfg
            .value_inst(dest)
            .is_some_and(|source_inst| {
                self.function.layout.inst_block(source_inst) == block
                    && self.function.dfg.users(dest).copied().eq([inst])
                    && self
                        .function
                        .dfg
                        .downcast::<&InsertValue>(source_inst)
                        .is_some()
            });
        let result = if reuse {
            source
        } else {
            let slot = self.slot(result_ty)?;
            let size = value_storage_size(result_ty, self.ctx)?;
            self.copy(slot, source, size);
            slot
        };
        let idx = constant_value_index(self.function, idx, "insert_value")?;
        let (offset, elem_ty) = aggregate_elem_offset(self.ctx, result_ty, idx)?;
        let value_ty = self.ty(value);
        if value_ty != elem_ty {
            return Err(format!(
                "insert_value element type mismatch: expected {elem_ty:?}, got {value_ty:?}"
            ));
        }
        let value = self.resolve(value)?;
        self.store(elem_ty, &value, result, offset as u32)?;
        self.define1(inst, result);
        Ok(())
    }
}

/// `(offset, width)` chunks covering `size` bytes, widest first.
fn storage_chunks(size: u32) -> impl Iterator<Item = (u32, u32)> {
    let mut offset = 0;
    std::iter::from_fn(move || {
        let width = match size - offset {
            0 => return None,
            1 => 1,
            2..=3 => 2,
            4..=7 => 4,
            _ => 8,
        };
        let chunk = (offset, width);
        offset += width;
        Some(chunk)
    })
}
