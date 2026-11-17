//! A record of how the EVM backend lowered a section, kept when compilation
//! asks for it (`CompileOptions::emit_lowering`): the IR machine lowering
//! read, the machine IR the section's code is emitted from, and what
//! connects that machine IR to the bytes. Recording it changes no byte.
//!
//! Functions and symbols are named as the IR text of the section's module
//! names them, values and blocks by their ids in the text of their module
//! (`v12`, `block3`).

use sonatina_ir::{BlockId, Immediate, ValueId};

use super::SymbolDef;

#[derive(Debug, Clone)]
pub struct SectionLowering {
    /// The section's module after the backend's IR pipeline: what machine
    /// lowering reads.
    pub source_ir: String,
    /// The machine module the section's code is emitted from.
    pub machine_ir: String,
    /// The bytes the code takes; data and embeds follow it.
    pub code_size: u32,
    pub arena_base: u32,
    pub dyn_base: u32,
    /// The free pointer the section entry stores at `0x40` before anything
    /// else, if it stores one.
    pub entry_free_ptr: Option<u32>,
    pub funcs: Vec<FuncLowering>,
    /// The offset and size of each code unit the late merges outlined.
    pub units: Vec<(String, u32, u32)>,
    /// Each label the late merges redirected, as `(old, new)`: code that
    /// jumped or called to `old` now goes to `new`.
    pub redirects: Vec<(CodeLabel, CodeLabel)>,
    /// The section's symbol table, by offset, each symbol as the IR text
    /// writes it (`%func`, `$global`, `&embed`, `.` for the section). The
    /// pipeline adds functions and globals to the section's module, which the
    /// compiled module does not name.
    pub symbols: Vec<(String, SymbolDef)>,
}

#[derive(Debug, Clone)]
pub struct FuncLowering {
    pub name: String,
    /// Where the function's code starts; `None` if its code is another's.
    pub offset: Option<u32>,
    pub blocks: Vec<BlockLowering>,
    /// Where each value the stack does not hold everywhere is kept.
    pub spills: Vec<SpillLowering>,
    /// The moves between the stack and memory that the function's code makes
    /// for the stack allocator, at each point that has any, with the moves at
    /// a point in the order the code makes them. The points of a block come in
    /// the order its code reaches them: function entry, then for each
    /// instruction, before it, before each of its `br_table` case compares,
    /// and after it; blocks in the machine IR's order. A function without code
    /// of its own has none.
    pub spill_actions: Vec<(CodePoint, Vec<SpillAction>)>,
    pub frame: FrameLowering,
    /// The placement of each `evm_malloc` of the source IR, by its result.
    pub mallocs: Vec<(ValueId, MallocLowering)>,
}

#[derive(Debug, Clone)]
pub struct BlockLowering {
    /// The block of the machine IR.
    pub block: BlockId,
    /// The blocks of the source IR it comes from.
    pub sources: Vec<BlockId>,
    /// Where its code starts, if it is laid out.
    pub offset: Option<u32>,
    /// The block whose code it is emitted as: a jump-only block is.
    pub alias: Option<BlockId>,
    /// The stack the block starts with, top first. Below it the stack is
    /// the caller's or unused. Empty unless the block is planned.
    pub entry_stack: Vec<StackSlot>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StackSlot {
    Value(ValueId),
    Imm(Immediate),
    /// The address the function returns to.
    FuncReturn,
    /// The address an internal call returns to, pushed for the call.
    CallReturn,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpillLowering {
    pub value: ValueId,
    pub location: SpillLocation,
}

/// A move between the stack and memory that a function's code makes for the
/// stack allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpillAction {
    /// Pop the stack top into the location of the spilled value: the value
    /// itself after its definition or, for an argument, at function entry,
    /// and for a phi its operand on the edge, before the predecessor's
    /// terminator.
    Store(ValueId),
    /// Push the spilled value from its location.
    Reload(ValueId),
    /// Before a call, copy the word at address `word`, which holds a scratch
    /// object of the caller that the callee may overwrite, to `shadow`.
    Save { word: u32, shadow: SpillLocation },
    /// After the call, copy `shadow` back to the word at address `word`.
    Restore { word: u32, shadow: SpillLocation },
}

/// A point of a function's machine IR at which its code runs the stack
/// allocator's moves. `index` is the position of an instruction in its block,
/// phis included. A function without code of its own
/// (`FuncLowering::offset` is `None`) has no points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodePoint {
    /// Function entry, before the entry block's first instruction.
    Entry,
    /// Before the instruction's opcodes: where its operands are prepared, and
    /// for a terminator, the moves into its successors.
    Before { block: BlockId, index: u32 },
    /// After the instruction's opcodes, where its results are stored.
    After { block: BlockId, index: u32 },
    /// Before the comparison of case `case` of the `br_table` instruction.
    Case {
        block: BlockId,
        index: u32,
        case: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpillLocation {
    /// The word at this memory address.
    Memory(u32),
    /// This word slot of the function's dynamic frame: while the frame is
    /// entered, the word `32 * (stable_words + 1 - slot)` bytes below the
    /// dynamic stack pointer, which is the word at `0x80`.
    Frame(u32),
    /// Recomputed, not stored.
    Rematerialized,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameLowering {
    /// No stable objects, objects at `base_word`, or a dynamic frame.
    pub stable: StableLowering,
    pub scratch_words: u32,
    pub stable_words: u32,
    /// The function's code enters and leaves a dynamic frame.
    pub lazy_frame: bool,
    /// The function's entry initializes the dynamic stack pointer at `0x80`:
    /// always, or only when it is unset (`checked`).
    pub dyn_sp_entry_init: Option<DynSpInitLowering>,
    /// The calls before which the code initializes the dynamic stack pointer.
    pub dyn_sp_init_calls: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynSpInitLowering {
    Always,
    Checked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableLowering {
    None,
    Abs { base_word: u32 },
    DynamicFrame,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MallocLowering {
    /// At this constant address.
    Fixed { base: u32 },
    /// At the free pointer, raised to `min_base`, or at `exact_base` when
    /// the free pointer is known; `update_free_ptr` if it moves the pointer.
    Heap {
        min_base: u32,
        exact_base: Option<u32>,
        update_free_ptr: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodeLabel {
    Block {
        func: String,
        block: BlockId,
    },
    Func(String),
    Unit(String),
    /// An instruction of the emitted code, which no IR names.
    Inst,
}
