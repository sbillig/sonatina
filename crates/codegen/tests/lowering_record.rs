//! The lowering record (`EvmCompile::with_lowering_record`): recording it
//! changes no byte, and what it says agrees with the bytes and with itself.

use dir_test::{Fixture, dir_test};
use sonatina_codegen::{
    EvmCompile, OptLevel,
    isa::evm::opcode::OpCode,
    object::{
        ObjectArtifact, encode_gv_initializer_to_bytes,
        lowering::{
            CodeLabel, CodePoint, FuncLowering, SectionLowering, SpillAction, SpillLocation,
            SpillLowering, StableLowering, StackSlot,
        },
    },
};
use sonatina_ir::{BlockId, ValueId};
use sonatina_parser::parse_module;
use std::collections::{BTreeMap, BTreeSet, HashSet};

fn try_compile(source: &str, opt: OptLevel, record: bool) -> Option<Vec<ObjectArtifact>> {
    EvmCompile::new(parse_module(source).expect("fixture parses").module)
        .with_opt_level(opt)
        .with_lowering_record(record)
        .compile()
        .ok()
}

fn compile(source: &str, opt: OptLevel, record: bool) -> Vec<ObjectArtifact> {
    try_compile(source, opt, record).expect("fixture compiles")
}

/// The offsets at which instructions start, decoding from the start of the
/// section's code.
fn instruction_starts(bytes: &[u8], code_end: usize) -> BTreeSet<u32> {
    let mut starts = BTreeSet::new();
    let mut pc = 0;
    while pc < code_end {
        starts.insert(pc as u32);
        let op = bytes[pc];
        let push = OpCode::PUSH1 as u8..=OpCode::PUSH32 as u8;
        pc += 1 + if push.contains(&op) {
            usize::from(op - OpCode::PUSH1 as u8 + 1)
        } else {
            0
        };
    }
    starts
}

fn check_record(lowering: &SectionLowering, bytes: &[u8]) {
    let source = parse_module(&lowering.source_ir)
        .expect("the source IR parses")
        .module;
    parse_module(&lowering.machine_ir).expect("the machine IR parses");

    let source_funcs: BTreeSet<_> = source
        .funcs()
        .into_iter()
        .map(|func| source.ctx.func_sig(func, |sig| format!("%{}", sig.name())))
        .collect();
    for (symbol, def) in &lowering.symbols {
        let (start, end) = (def.offset as usize, (def.offset + def.size) as usize);
        assert!(end <= bytes.len(), "{symbol} ends past the section");
        if let Some(name) = symbol.strip_prefix('$') {
            let gv = source
                .ctx
                .with_gv_store(|store| store.lookup_gv(name))
                .unwrap_or_else(|| panic!("{symbol} is not a global of the source IR"));
            let data = encode_gv_initializer_to_bytes(&source.ctx, gv).expect("data encodes");
            assert!(
                start >= lowering.code_size as usize,
                "{symbol} is in the code"
            );
            assert_eq!(bytes[start..end], data, "the bytes of {symbol}");
        } else if symbol.starts_with('%') {
            assert!(
                source_funcs.contains(symbol),
                "{symbol} is not a function of the source IR"
            );
            assert!(
                end <= lowering.code_size as usize,
                "{symbol} is past the code"
            );
        } else if symbol == "." {
            assert_eq!((start, end), (0, bytes.len()));
        }
    }

    let starts = instruction_starts(bytes, lowering.code_size as usize);
    for func in &lowering.funcs {
        for block in &func.blocks {
            if let Some(offset) = block.offset {
                assert!(
                    starts.contains(&offset),
                    "{}: block{} starts at {offset}, inside an instruction",
                    func.name,
                    block.block.0
                );
            }
        }
    }

    let blocks: BTreeSet<_> = lowering
        .funcs
        .iter()
        .flat_map(|func| {
            func.blocks
                .iter()
                .map(|block| (func.name.as_str(), block.block))
        })
        .collect();
    let funcs: BTreeSet<_> = lowering
        .funcs
        .iter()
        .map(|func| func.name.as_str())
        .collect();
    let units: BTreeSet<_> = lowering
        .units
        .iter()
        .map(|(name, ..)| name.as_str())
        .collect();
    for (old, new) in &lowering.redirects {
        for label in [old, new] {
            match label {
                CodeLabel::Block { func, block } => assert!(
                    blocks.contains(&(func.as_str(), *block)) || units.contains(func.as_str()),
                    "redirect names unknown block{} of {func}",
                    block.0
                ),
                CodeLabel::Func(func) => assert!(funcs.contains(func.as_str())),
                CodeLabel::Unit(unit) => assert!(units.contains(unit.as_str())),
                CodeLabel::Inst => {}
            }
        }
    }

    for func in &lowering.funcs {
        if func.frame.dyn_sp_init_calls != 0 || func.frame.dyn_sp_entry_init.is_some() {
            assert!(
                lowering.funcs.iter().any(|f| f.frame.lazy_frame),
                "{}: initializes the dynamic stack pointer, but no function has a frame",
                func.name
            );
        }
    }
    check_spill_actions(lowering, bytes, &starts).unwrap();
    for func in &lowering.funcs {
        let Some(entry) = func.blocks.first() else {
            continue;
        };
        let returns = entry.entry_stack.last() == Some(&StackSlot::FuncReturn);
        assert!(
            entry.entry_stack[..entry.entry_stack.len() - usize::from(returns)]
                .iter()
                .all(|slot| matches!(slot, StackSlot::Value(_) | StackSlot::Imm(_))),
            "{}: the entry stack holds only arguments above the return address",
            func.name
        );
    }
}

/// A load (`false`) or store (`true`) of a word of memory, as the code makes
/// it.
type Access = (bool, SpillLocation);

/// The accesses of a recorded spill action, in the order the code makes them.
fn action_accesses(func: &FuncLowering, action: SpillAction) -> Vec<Access> {
    let location = |value: ValueId| {
        func.spills
            .iter()
            .find(|spill| spill.value == value)
            .map(|spill| spill.location)
            .expect("a spill action moves a spilled value")
    };
    match action {
        SpillAction::Store(value) => vec![(true, location(value))],
        SpillAction::Reload(value) => vec![(false, location(value))],
        SpillAction::Save { word, shadow } => {
            vec![(false, SpillLocation::Memory(word)), (true, shadow)]
        }
        SpillAction::Restore { word, shadow } => {
            vec![(false, shadow), (true, SpillLocation::Memory(word))]
        }
    }
}

/// Where a point is in its block's code, by the position of its instruction:
/// function entry, then before an instruction, the instruction's own opcodes
/// (`phase` 1), its `br_table` case compares, and after it.
fn point_order(point: CodePoint) -> (u32, u32, u32) {
    match point {
        CodePoint::Entry => (0, 0, 0),
        CodePoint::Before { index, .. } => (index + 1, 0, 0),
        CodePoint::Case { index, case, .. } => (index + 1, 2, case),
        CodePoint::After { index, .. } => (index + 1, 3, 0),
    }
}

/// The constant-address `evm_mstore` (`true`) or `evm_mload` of each
/// instruction of the machine IR, by function, block and position in the
/// block (phis included), from its text.
fn machine_ir_accesses(machine_ir: &str) -> BTreeMap<(String, BlockId, u32), (bool, u32)> {
    let mut accesses = BTreeMap::new();
    let (mut func, mut block, mut index) = (String::new(), BlockId(0), 0);
    for line in machine_ir.lines().map(str::trim) {
        if let Some((_, rest)) = line
            .strip_prefix("func ")
            .and_then(|header| header.split_once('%'))
        {
            func = rest.split('(').next().unwrap().to_string();
        } else if let Some(n) = line
            .strip_prefix("block")
            .and_then(|label| label.strip_suffix(':'))
        {
            (block, index) = (BlockId(n.parse().unwrap()), 0);
        } else if !line.is_empty() && line != "}" {
            if let Some((store, operands)) = line
                .strip_prefix("evm_mstore ")
                .map(|operands| (true, operands))
                .or_else(|| {
                    line.split_once("= evm_mload ")
                        .map(|(_, operands)| (false, operands))
                })
                && let Some((addr, _)) = operands.split_once(".i256")
                && let Ok(addr) = addr.parse()
            {
                accesses.insert((func.clone(), block, index), (store, addr));
            }
            index += 1;
        }
    }
    accesses
}

/// The accesses of the code from `start` to `end` to `words`: `PUSH w`
/// followed by `MLOAD` or `MSTORE` for a word of memory, and for a slot of a
/// dynamic frame of `frame_words` words, its address below the dynamic stack
/// pointer, `PUSH 32 * (frame_words + 1 - slot); PUSH1 0x80; MLOAD; SUB`,
/// followed by `MLOAD` or `MSTORE`.
fn code_accesses(
    bytes: &[u8],
    starts: &BTreeSet<u32>,
    (start, end): (u32, u32),
    words: &HashSet<SpillLocation>,
    frame_words: Option<u32>,
) -> Vec<Access> {
    let insts: Vec<(u8, Option<u32>)> = starts
        .range(start..end)
        .map(|&pc| {
            let (pc, op) = (pc as usize, bytes[pc as usize]);
            let push = match op {
                op if op == OpCode::PUSH0 as u8 => Some(0),
                op if (OpCode::PUSH1 as u8..=OpCode::PUSH4 as u8).contains(&op) => {
                    Some(usize::from(op - OpCode::PUSH1 as u8 + 1))
                }
                _ => None,
            };
            let imm = push.map(|width| {
                bytes[pc + 1..=pc + width]
                    .iter()
                    .fold(0u32, |word, &byte| word << 8 | u32::from(byte))
            });
            (op, imm)
        })
        .collect();
    let access = |op| match op {
        op if op == OpCode::MSTORE as u8 => Some(true),
        op if op == OpCode::MLOAD as u8 => Some(false),
        _ => None,
    };
    let mut accesses = Vec::new();
    let mut rest = insts.as_slice();
    while let Some(((_, imm), tail)) = rest.split_first() {
        rest = tail;
        let Some(imm) = *imm else {
            continue;
        };
        if let Some(frame_words) = frame_words
            && let [(_, Some(0x80)), (mload, None), (sub, None), (op, None), ..] = tail
            && *mload == OpCode::MLOAD as u8
            && *sub == OpCode::SUB as u8
            && let Some(store) = access(*op)
            && imm % 32 == 0
            && let Some(slot) = (frame_words + 1).checked_sub(imm / 32)
        {
            rest = &tail[4..];
            if words.contains(&SpillLocation::Frame(slot)) {
                accesses.push((store, SpillLocation::Frame(slot)));
            }
        } else if let [(op, None), ..] = tail
            && let Some(store) = access(*op)
            && words.contains(&SpillLocation::Memory(imm))
        {
            rest = &tail[1..];
            accesses.push((store, SpillLocation::Memory(imm)));
        }
    }
    accesses
}

/// The recorded spill actions are the code's, in order: in each block, the
/// accesses of the code to the words the function's actions name
/// (`code_accesses`) are the accesses of the block's actions, point by point
/// in the order the code reaches the points and in each point in the order
/// recorded, with among them the machine IR's own loads and stores at those
/// words, each at its instruction. Sections whose code the late merges moved
/// are left out: a block's code is no longer the bytes from its offset to the
/// next. Returns the number of recorded accesses checked.
fn check_spill_actions(
    lowering: &SectionLowering,
    bytes: &[u8],
    starts: &BTreeSet<u32>,
) -> Result<usize, String> {
    if !lowering.units.is_empty() || !lowering.redirects.is_empty() {
        return Ok(0);
    }
    let mut block_starts: BTreeSet<u32> = lowering
        .funcs
        .iter()
        .flat_map(|func| {
            func.offset
                .into_iter()
                .chain(func.blocks.iter().filter_map(|b| b.offset))
        })
        .collect();
    block_starts.insert(lowering.code_size);
    let ir_accesses = machine_ir_accesses(&lowering.machine_ir);
    let mut checked = 0;
    for func in &lowering.funcs {
        let blocks: BTreeMap<BlockId, usize> = func
            .blocks
            .iter()
            .zip(0..)
            .map(|(block, order)| (block.block, order))
            .collect();
        let point_key = |point| {
            let block = match point {
                CodePoint::Entry => 0,
                CodePoint::Before { block, .. }
                | CodePoint::After { block, .. }
                | CodePoint::Case { block, .. } => blocks[&block],
            };
            (block, point_order(point))
        };
        if !func
            .spill_actions
            .windows(2)
            .all(|pair| point_key(pair[0].0) < point_key(pair[1].0))
        {
            return Err(format!(
                "{}: the points are not in the order the code reaches them",
                func.name
            ));
        }
        let words: HashSet<SpillLocation> = func
            .spill_actions
            .iter()
            .flat_map(|(_, actions)| actions)
            .flat_map(|&action| action_accesses(func, action))
            .map(|(_, word)| word)
            .collect();
        let frame_words =
            (func.frame.stable == StableLowering::DynamicFrame).then_some(func.frame.stable_words);
        for block in &func.blocks {
            // Each access the code may make in the block, with whether it may
            // be left out: the recorded ones may not, the machine IR's may,
            // since its address need not be pushed right before the opcode.
            let mut expected: Vec<((u32, u32, u32), Access, bool)> = func
                .spill_actions
                .iter()
                .filter(|(point, _)| point_key(*point).0 == blocks[&block.block])
                .flat_map(|&(point, ref actions)| {
                    actions
                        .iter()
                        .flat_map(|&action| action_accesses(func, action))
                        .map(move |access| (point_order(point), access, false))
                })
                .collect();
            checked += expected.len();
            expected.extend(
                ir_accesses
                    .range(
                        (func.name.clone(), block.block, 0)
                            ..=(func.name.clone(), block.block, u32::MAX),
                    )
                    .filter(|(_, (_, addr))| words.contains(&SpillLocation::Memory(*addr)))
                    .map(|(&(_, _, index), &(store, addr))| {
                        (
                            (index + 1, 1, 0),
                            (store, SpillLocation::Memory(addr)),
                            true,
                        )
                    }),
            );
            expected.sort_by_key(|(order, ..)| *order);
            let Some(start) = block.offset else {
                if expected.iter().any(|&(.., optional)| !optional) {
                    return Err(format!(
                        "{}: block{} has spill actions but no code",
                        func.name, block.block.0
                    ));
                }
                continue;
            };
            let end = *block_starts.range(start + 1..).next().unwrap();
            let found = code_accesses(bytes, starts, (start, end), &words, frame_words);
            // Whether the expected accesses from each on can be the code's from
            // each on, the optional ones left out or not: `rest[j]` for the
            // code's from `j`.
            let mut rest: Vec<bool> = (0..=found.len()).map(|j| j == found.len()).collect();
            for &(_, access, optional) in expected.iter().rev() {
                rest = (0..=found.len())
                    .map(|j| {
                        (optional && rest[j]) || (found.get(j) == Some(&access) && rest[j + 1])
                    })
                    .collect();
            }
            if !rest[0] {
                return Err(format!(
                    "{}: block{} makes the accesses {found:?}, but the record says {:?}",
                    func.name,
                    block.block.0,
                    expected
                        .iter()
                        .map(|&(_, access, optional)| (access, optional))
                        .collect::<Vec<_>>()
                ));
            }
        }
    }
    Ok(checked)
}

/// Compiles `source` at each of `levels` with and without the record: the
/// bytes are the same, and the record agrees with them.
fn check_fixture(source: &str, levels: &[OptLevel]) {
    for &opt in levels {
        let Some(plain) = try_compile(source, opt, false) else {
            assert!(try_compile(source, opt, true).is_none());
            continue;
        };
        let recorded = compile(source, opt, true);
        assert_eq!(plain.len(), recorded.len());
        for (plain, recorded) in plain.iter().zip(&recorded) {
            for ((name, plain), (_, recorded)) in plain.sections.iter().zip(&recorded.sections) {
                assert_eq!(plain.bytes, recorded.bytes, "{name:?} at {opt:?}");
                assert!(plain.lowering.is_none());
                let lowering = recorded
                    .lowering
                    .as_ref()
                    .expect("a recorded compile has a lowering record");
                check_record(lowering, &recorded.bytes);
            }
        }
    }
}

#[dir_test(dir: "$CARGO_MANIFEST_DIR/tests/fixtures", glob: "*.sntn")]
fn lowering_record_is_metadata_and_agrees_with_bytes(fixture: Fixture<&str>) {
    if fixture.content().contains("\nobject ") {
        check_fixture(
            fixture.content(),
            &[OptLevel::O0, OptLevel::O1, OptLevel::Os],
        );
    }
}

/// The EVM fixtures, at O0: late merges, which leave spills out of
/// `check_spill_actions`, are off.
#[dir_test(dir: "$CARGO_MANIFEST_DIR/test_files/evm", glob: "*.sntn")]
fn lowering_record_of_evm_fixtures_agrees_with_bytes(fixture: Fixture<&str>) {
    if fixture.content().contains("\nobject ") {
        check_fixture(fixture.content(), &[OptLevel::O0]);
    }
}

/// `%runtime` reads a table at an index from calldata through `%get`. The
/// backend's pipeline specializes `%get` for the table and moves the table to
/// a word blob: a function and a global the compiled module does not have,
/// named in the record as the section's IR names them. The runtime section's
/// module gives the new function the id `%init` has in the compiled module.
#[test]
fn lowering_record_names_symbols_of_the_section_module() {
    let source = r#"
target = "evm-ethereum-osaka"

global private const [i256; 6] $table = [5, 17, 2, 99, 1234567, 8];

func inline(never) private %get(v0.constref<[i256; 6]>, v1.i256) -> i256 {
    block0:
        v2.constref<i256> = const.index v0 v1;
        v3.i256 = const.load v2;
        return v3;
}

func public %runtime() {
    block0:
        v0.i256 = evm_calldata_load 0.i256;
        v1.constref<[i256; 6]> = const.ref $table;
        v2.i256 = call %get v1 v0;
        mstore 0.i256 v2 i256;
        evm_return 0.i256 32.i256;
}

func public %init() {
    block0:
        v0.i256 = sym_addr &runtime;
        v1.i256 = sym_size &runtime;
        evm_code_copy 0.i256 v0 v1;
        evm_return 0.i256 v1;
}

object @Table {
  section init {
    entry %init;
    embed .runtime as &runtime;
  }
  section runtime {
    entry %runtime;
  }
}
"#;
    let input = parse_module(source).expect("fixture parses").module;
    let input_symbols: BTreeSet<_> = input
        .funcs()
        .into_iter()
        .map(|func| input.ctx.func_sig(func, |sig| format!("%{}", sig.name())))
        .chain(["$table".to_string(), ".".to_string()])
        .collect();
    for opt in [OptLevel::O0, OptLevel::O1, OptLevel::Os] {
        let artifacts = compile(source, opt, true);
        for section in artifacts[0].sections.values() {
            check_record(
                section.lowering.as_ref().expect("a lowering record"),
                &section.bytes,
            );
        }
        let (_, runtime) = artifacts[0]
            .sections
            .iter()
            .find(|(name, _)| name.0.as_str() == "runtime")
            .expect("a runtime section");
        let lowering = runtime.lowering.as_ref().expect("a lowering record");
        let new: Vec<_> = lowering
            .symbols
            .iter()
            .map(|(symbol, _)| symbol.as_str())
            .filter(|symbol| !input_symbols.contains(*symbol))
            .collect();
        for prefix in ['%', '$'] {
            assert!(
                new.iter().any(|symbol| symbol.starts_with(prefix)),
                "{opt:?}: no `{prefix}` symbol of the section's own in {new:?}"
            );
        }
    }
}

/// The O0 record of the fixture `file`'s only section, checked against the
/// bytes, and the function `name` of it.
fn recorded_func(file: &str, name: &str) -> (SectionLowering, Vec<u8>, FuncLowering) {
    let path = format!("{}/test_files/evm/{file}", env!("CARGO_MANIFEST_DIR"));
    let source = std::fs::read_to_string(path).expect("fixture reads");
    let artifacts = compile(&source, OptLevel::O0, true);
    let section = artifacts[0].sections.values().next().expect("a section");
    let lowering = section.lowering.clone().expect("a lowering record");
    let starts = instruction_starts(&section.bytes, lowering.code_size as usize);
    assert!(
        check_spill_actions(&lowering, &section.bytes, &starts).unwrap() > 0,
        "the spill actions are checked against the bytes"
    );
    let func = lowering
        .funcs
        .iter()
        .find(|func| func.name == name)
        .expect("the function is recorded")
        .clone();
    (lowering, section.bytes.clone(), func)
}

/// The spill actions of `func`, a point to a line.
fn spill_action_lines(func: &FuncLowering) -> Vec<String> {
    let point = |point| match point {
        CodePoint::Entry => "entry".to_string(),
        CodePoint::Before { block, index } => format!("before block{}:{index}", block.0),
        CodePoint::After { block, index } => format!("after block{}:{index}", block.0),
        CodePoint::Case { block, index, case } => {
            format!("case {case} of block{}:{index}", block.0)
        }
    };
    let action = |action| match action {
        SpillAction::Store(value) => format!("store v{}", value.as_u32()),
        SpillAction::Reload(value) => format!("reload v{}", value.as_u32()),
        SpillAction::Save { word, shadow } => format!("save {word} to {shadow:?}"),
        SpillAction::Restore { word, shadow } => format!("restore {word} from {shadow:?}"),
    };
    func.spill_actions
        .iter()
        .map(|&(at, ref actions)| {
            let actions: Vec<_> = actions.iter().map(|&a| action(a)).collect();
            format!("{}: {}", point(at), actions.join(", "))
        })
        .collect()
}

/// At O0 `%spill` keeps its arguments `v1` to `v7` in memory across the
/// calls of its loop: it stores each at entry, reloads them all for the first
/// call (`block2:1`, after the phi), last argument first, and `v7` again for
/// the `add` at `block2:4`.
#[test]
fn lowering_record_lists_argument_stores_and_reloads() {
    let (_, _, func) = recorded_func("spill.sntn", "spill");
    assert_eq!(
        func.spills,
        (1..=7)
            .map(|value| SpillLowering {
                value: ValueId(value),
                location: SpillLocation::Memory(32 * (value - 1)),
            })
            .collect::<Vec<_>>()
    );
    assert_eq!(
        spill_action_lines(&func),
        [
            "entry: store v1, store v2, store v3, store v4, store v5, store v6, store v7",
            "before block2:1: reload v7, reload v6, reload v5, reload v4, reload v3, reload v2, \
             reload v1",
            "before block2:4: reload v7",
        ]
    );
}

/// At O0 `%rotate`'s twenty phis rotate left on the backedge from `block2`,
/// `v1` taking `v2`, ..., `v20` taking `v1`, and all are spilled. Before the
/// backedge's `jump` the code stores the counter `v21` from the stack, holds
/// the old `v1` for `v20`, then reloads each phi's operand just before it
/// stores the phi it is the operand of: the stores of the edge come
/// interleaved with the reloads of their operands.
#[test]
fn lowering_record_orders_phi_edge_stores_and_reloads() {
    let (_, _, func) = recorded_func("spilled_phi_edge_rotation.sntn", "rotate");
    let backedge: Vec<_> = ["store v21".to_string(), "reload v1".to_string()]
        .into_iter()
        .chain((2..=20).flat_map(|k| [format!("reload v{k}"), format!("store v{}", k - 1)]))
        .chain(["store v20".to_string()])
        .collect();
    let stores: Vec<_> = (1..=21).map(|k| format!("store v{k}")).collect();
    assert_eq!(
        spill_action_lines(&func),
        [
            "entry: store v0".to_string(),
            format!("before block0:0: {}", stores.join(", ")),
            "before block1:21: reload v0, reload v21".to_string(),
            "before block2:0: reload v21".to_string(),
            format!("before block2:1: {}", backedge.join(", ")),
            "before block3:0: reload v20".to_string(),
        ]
    );
}

/// The check of the record against the bytes sees the order of the actions
/// at a point: with the reload of `v2` and the store of `v1` on `%rotate`'s
/// backedge swapped, the record no longer agrees with the bytes.
#[test]
fn spill_action_check_rejects_another_order() {
    let (mut lowering, bytes, _) = recorded_func("spilled_phi_edge_rotation.sntn", "rotate");
    let starts = instruction_starts(&bytes, lowering.code_size as usize);
    let rotate = lowering
        .funcs
        .iter_mut()
        .find(|func| func.name == "rotate")
        .expect("the function is recorded");
    let (_, backedge) = rotate
        .spill_actions
        .iter_mut()
        .find(|(point, _)| {
            *point
                == CodePoint::Before {
                    block: BlockId(2),
                    index: 1,
                }
        })
        .expect("the backedge has spill actions");
    assert_eq!(
        backedge[2..4],
        [
            SpillAction::Reload(ValueId(2)),
            SpillAction::Store(ValueId(1))
        ]
    );
    backedge.swap(2, 3);
    let error = check_spill_actions(&lowering, &bytes, &starts).unwrap_err();
    assert!(error.contains("rotate: block2"), "{error}");
}
