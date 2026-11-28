//! The lowering record (`EvmCompile::with_lowering_record`): recording it
//! changes no byte, and what it says agrees with the bytes and with itself.

use dir_test::{Fixture, dir_test};
use sonatina_codegen::{
    EvmCompile, OptLevel,
    isa::evm::{EvmBackend, opcode::OpCode},
    object::{
        CompileOptions, ObjectArtifact, compile_all_objects, encode_gv_initializer_to_bytes,
        lowering::{
            CodeLabel, CodePoint, FixedMallocRule, FuncLowering, MallocLowering, ObjLocLowering,
            SectionLowering, SpillLocation, StableLowering, StackSlot,
        },
    },
};
use sonatina_ir::{BlockId, isa::evm::Evm};
use sonatina_parser::parse_module;
use std::collections::{BTreeMap, BTreeSet};

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
        for (block, _) in &func.frame.call_preserves {
            assert!(
                blocks.contains(&(func.name.as_str(), *block)),
                "{}: a call preserve in unknown block{}",
                func.name,
                block.0
            );
        }
        if func.frame.dyn_sp_init_calls != 0 || func.frame.dyn_sp_entry_init.is_some() {
            assert!(
                lowering.funcs.iter().any(|f| f.frame.lazy_frame),
                "{}: initializes the dynamic stack pointer, but no function has a frame",
                func.name
            );
        }
    }
    check_spill_accesses(lowering, bytes, &starts);
    check_placement(lowering);
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

/// The block of `func` a point is in.
fn point_block(func: &FuncLowering, point: CodePoint) -> BlockId {
    match point {
        CodePoint::Entry => func.blocks[0].block,
        CodePoint::Before { block, .. }
        | CodePoint::After { block, .. }
        | CodePoint::Case { block, .. } => block,
    }
}

/// The constant addresses each block of each function of the machine IR
/// stores to with `evm_mstore` (`true`) and loads from with `evm_mload`, from
/// its text.
fn machine_ir_accesses(machine_ir: &str) -> BTreeMap<(String, u32, bool), Vec<u32>> {
    let mut accesses: BTreeMap<_, Vec<u32>> = BTreeMap::new();
    let (mut func, mut block) = (String::new(), 0);
    for line in machine_ir.lines().map(str::trim) {
        if let Some(rest) = line.split_once('%') {
            if line.starts_with("func ") {
                func = rest.1.split('(').next().unwrap().to_string();
            }
        } else if let Some(n) = line.strip_prefix("block").and_then(|l| l.strip_suffix(':')) {
            block = n.parse().unwrap();
        } else if let Some((store, operands)) = line
            .strip_prefix("evm_mstore ")
            .map(|operands| (true, operands))
            .or_else(|| {
                line.split_once("= evm_mload ")
                    .map(|(_, operands)| (false, operands))
            })
            && let Some((addr, _)) = operands.split_once(".i256")
            && let Ok(addr) = addr.parse()
        {
            accesses
                .entry((func.clone(), block, store))
                .or_default()
                .push(addr);
        }
    }
    accesses
}

/// The recorded stores and reloads of memory spills are the code's: in each
/// block, every `PUSH w; MSTORE` (`PUSH w; MLOAD`) of a spill word `w` of the
/// function is a store (reload) the record lists there or a store (load) of
/// the machine IR at `w`, and every one listed is in the code. Sections whose
/// code the late merges moved, and functions whose code saves words around
/// calls or addresses a dynamic frame, access spill words in other ways and
/// are left out. Returns the number of stores and reloads checked.
fn check_spill_accesses(lowering: &SectionLowering, bytes: &[u8], starts: &BTreeSet<u32>) -> usize {
    if !lowering.units.is_empty() || !lowering.redirects.is_empty() {
        return 0;
    }
    let mut checked = 0;
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
    for func in &lowering.funcs {
        if func.offset.is_none()
            || !func.frame.call_preserves.is_empty()
            || func.frame.lazy_frame
            || func.frame.stable == StableLowering::DynamicFrame
        {
            continue;
        }
        let mut recorded: BTreeMap<(BlockId, u32, bool), usize> = BTreeMap::new();
        let mut words = BTreeSet::new();
        for spill in &func.spills {
            let SpillLocation::Memory(word) = spill.location else {
                continue;
            };
            words.insert(word);
            for (points, store) in [(&spill.stores, true), (&spill.reloads, false)] {
                for &point in points {
                    *recorded
                        .entry((point_block(func, point), word, store))
                        .or_default() += 1;
                }
            }
        }
        for block in &func.blocks {
            let Some(start) = block.offset else {
                assert!(
                    !recorded.keys().any(|(b, ..)| *b == block.block),
                    "{}: block{} has spill stores or reloads but no code",
                    func.name,
                    block.block.0
                );
                continue;
            };
            let end = *block_starts.range(start + 1..).next().unwrap();
            let mut found: BTreeMap<(u32, bool), usize> = BTreeMap::new();
            for pc in starts.range(start..end) {
                let (pc, op) = (*pc as usize, bytes[*pc as usize]);
                let width = match op {
                    op if op == OpCode::PUSH0 as u8 => 0,
                    op if (OpCode::PUSH1 as u8..=OpCode::PUSH4 as u8).contains(&op) => {
                        usize::from(op - OpCode::PUSH1 as u8 + 1)
                    }
                    _ => continue,
                };
                let word = bytes[pc + 1..=pc + width]
                    .iter()
                    .fold(0u32, |word, &byte| word << 8 | u32::from(byte));
                let store = match bytes.get(pc + 1 + width) {
                    Some(&op) if op == OpCode::MSTORE as u8 => true,
                    Some(&op) if op == OpCode::MLOAD as u8 => false,
                    _ => continue,
                };
                if words.contains(&word) {
                    *found.entry((word, store)).or_default() += 1;
                }
            }
            for &word in &words {
                for store in [true, false] {
                    let listed = recorded
                        .get(&(block.block, word, store))
                        .copied()
                        .unwrap_or(0);
                    let code = found.get(&(word, store)).copied().unwrap_or(0);
                    let ir = ir_accesses
                        .get(&(func.name.clone(), block.block.0, store))
                        .map_or(0, |ir| ir.iter().filter(|&&addr| addr == word).count());
                    checked += listed;
                    assert!(
                        listed <= code && code <= listed + ir,
                        "{}: block{} {} word {word} {code} times, with {listed} recorded and \
                         {ir} of the machine IR",
                        func.name,
                        block.block.0,
                        if store { "stores" } else { "loads" }
                    );
                }
            }
        }
    }
    checked
}

/// The text of each function of `ir`, by name.
fn func_texts(ir: &str) -> BTreeMap<&str, &str> {
    let mut texts = BTreeMap::new();
    let mut rest = ir;
    while let Some(start) = rest.find("func ") {
        let text = &rest[start..];
        let end = text.find("\n}").map_or(text.len(), |end| end + 2);
        let name = text.split_once('%').unwrap().1.split('(').next().unwrap();
        texts.insert(name, &text[..end]);
        rest = &text[end..];
    }
    texts
}

/// The values of `text` defined by `inst` (`v3.*i256 = alloca i256;`).
fn defined_by(text: &str, inst: &str) -> BTreeSet<u32> {
    text.lines()
        .filter_map(|line| {
            let (def, rhs) = line.trim().split_once(" = ")?;
            let value = def.strip_prefix('v')?.split('.').next()?;
            rhs.starts_with(inst).then(|| value.parse().unwrap())
        })
        .collect()
}

/// The address of each recorded `alloca` and malloc of `func` that the code
/// places at a constant address, by value.
fn constant_addresses(lowering: &SectionLowering, func: &FuncLowering) -> BTreeMap<u32, u32> {
    let allocas = func.allocas.iter().filter_map(|(value, alloca)| {
        let word = match alloca.loc {
            _ if alloca.words == 0 => return None,
            ObjLocLowering::ScratchAbs(word) | ObjLocLowering::StableAbs(word) => word,
            ObjLocLowering::StableFrame(_) => return None,
        };
        Some((value.as_u32(), lowering.arena_base + 32 * word))
    });
    let mallocs = func.mallocs.iter().filter_map(|(value, malloc)| {
        let base = match *malloc {
            MallocLowering::Fixed { base, .. } => base,
            MallocLowering::Heap { exact_base, .. } => exact_base?,
        };
        Some((value.as_u32(), base))
    });
    allocas.chain(mallocs).collect()
}

/// The placement the record gives agrees with the source IR, with the
/// machine IR lowering built and with itself:
/// - every `alloca` and `evm_malloc` of the source IR is recorded, and
///   every source block with the lowered block it starts in: the `k`-th
///   block of the function's layout is lowered `blockk`, and the blocks of
///   the machine IR come from the source blocks the map takes to them;
/// - each `alloca` lies in its function's frame, outside its spill reserves,
///   and the reserves in the frame;
/// - each memory spill word is in a reserve of its function or in a fixed
///   slot, which only a section with fixed-slot spills uses;
/// - an `i256` store of an immediate through an `alloca` or malloc at a
///   constant address is an `evm_mstore` at that address in the lowered
///   machine IR;
/// - a heap malloc reads the free pointer, and the dynamic stack pointer if
///   it needs the clamp, and raises its base to `min_base` unless the
///   free-pointer floor makes that redundant;
/// - a function that restores the free pointer loads it at entry and stores
///   it back before each `return`.
///
/// Returns the number of stores checked against the machine IR.
fn check_placement(lowering: &SectionLowering) -> usize {
    let sources = func_texts(&lowering.source_ir);
    let lowered: BTreeMap<_, _> = lowering
        .machine_steps
        .lowered
        .iter()
        .map(|snap| (snap.func.as_str(), snap.text.as_str()))
        .collect();
    let mut checked = 0;
    for func in &lowering.funcs {
        let name = func.name.as_str();
        let (source, machine) = (sources[name], lowered[name]);
        assert_eq!(
            func.allocas
                .iter()
                .map(|(v, _)| v.as_u32())
                .collect::<BTreeSet<_>>(),
            defined_by(source, "alloca "),
            "{name}: the recorded allocas"
        );
        assert_eq!(
            func.mallocs
                .iter()
                .map(|(v, _)| v.as_u32())
                .collect::<BTreeSet<_>>(),
            defined_by(source, "evm_malloc "),
            "{name}: the recorded mallocs"
        );

        let source_blocks: Vec<_> = source
            .lines()
            .filter_map(|line| line.trim().strip_prefix("block")?.strip_suffix(':'))
            .map(|block| block.parse::<u32>().unwrap())
            .collect();
        assert_eq!(
            func.lowered_blocks
                .iter()
                .map(|(source, lowered)| (source.0, lowered.0))
                .collect::<Vec<_>>(),
            source_blocks.into_iter().zip(0..).collect::<Vec<_>>(),
            "{name}: the lowered block of each source block"
        );
        for block in &func.blocks {
            let from: Vec<_> = func
                .lowered_blocks
                .iter()
                .filter(|(_, lowered)| *lowered == block.block)
                .map(|(source, _)| *source)
                .collect();
            assert_eq!(
                block.sources, from,
                "{name}: the sources of block{}",
                block.block.0
            );
        }

        let frame = &func.frame;
        let scratch = (0, frame.scratch_words);
        let stable = match frame.stable {
            StableLowering::Abs { base_word } => Some((base_word, base_word + frame.stable_words)),
            StableLowering::None | StableLowering::DynamicFrame => None,
        };
        let reserves: Vec<_> = [Some(func.spill_reserve.scratch), func.spill_reserve.stable]
            .into_iter()
            .flatten()
            .map(|range| (range.start, range.start + range.words))
            .collect();
        let within = |(start, end): (u32, u32), (lo, hi): (u32, u32)| lo <= start && end <= hi;
        assert!(
            within(reserves[0], scratch),
            "{name}: the scratch reserve {reserves:?} is in the scratch overlay {scratch:?}"
        );
        assert_eq!(
            func.spill_reserve.stable.is_some(),
            stable.is_some(),
            "{name}: a stable reserve iff an absolute stable frame"
        );
        if let (Some(stable), Some(&reserve)) = (stable, reserves.get(1)) {
            assert_eq!(
                reserve.1, stable.1,
                "{name}: the stable reserve tops the frame"
            );
            assert!(
                within(reserve, stable),
                "{name}: the stable reserve is in the frame"
            );
        }
        for (value, alloca) in &func.allocas {
            let (start, region) = match (alloca.loc, stable) {
                (ObjLocLowering::ScratchAbs(word), _) => (word, scratch),
                (ObjLocLowering::StableAbs(word), Some(stable)) => (word, stable),
                (ObjLocLowering::StableFrame(_), None) => {
                    assert_eq!(frame.stable, StableLowering::DynamicFrame, "{name}");
                    continue;
                }
                (loc, _) => panic!(
                    "{name}: v{} at {loc:?} in {:?}",
                    value.as_u32(),
                    frame.stable
                ),
            };
            let range = (start, start + alloca.words);
            assert!(
                alloca.words == 0
                    || within(range, region)
                        && reserves
                            .iter()
                            .all(|&(lo, hi)| range.1 <= lo || hi <= range.0),
                "{name}: v{} at words {range:?}, in {region:?}, outside the reserves {reserves:?}",
                value.as_u32()
            );
        }
        for spill in &func.spills {
            let SpillLocation::Memory(addr) = spill.location else {
                continue;
            };
            let word = addr.wrapping_sub(lowering.arena_base) / 32;
            assert!(
                addr % 32 == 0
                    && (addr >= lowering.arena_base
                        && reserves.iter().any(|&(lo, hi)| lo <= word && word < hi)
                        || lowering.fixed_slot_spills && addr < 64),
                "{name}: v{} spilled at {addr}, outside the reserves {reserves:?}",
                spill.value.as_u32()
            );
        }

        for (value, addr) in constant_addresses(lowering, func) {
            for line in source.lines().map(str::trim) {
                if let Some(imm) = line
                    .strip_prefix(&format!("mstore v{value} "))
                    .and_then(|rest| rest.strip_suffix(".i256 i256;"))
                    .filter(|imm| imm.bytes().all(|b| b.is_ascii_digit()))
                {
                    let store = format!("evm_mstore {addr}.i256 {imm}.i256;");
                    assert!(machine.contains(&store), "{name}: {store} for {line}");
                    checked += 1;
                }
            }
        }
        for (_, malloc) in &func.mallocs {
            let MallocLowering::Heap {
                min_base,
                exact_base: None,
                needs_dyn_sp_clamp,
                free_ptr_floor,
                ..
            } = *malloc
            else {
                continue;
            };
            assert!(
                machine.contains("= evm_mload 64.i256;")
                    && (!needs_dyn_sp_clamp || machine.contains("= evm_mload 128.i256;")),
                "{name}: reads the free pointer, and the dynamic stack pointer to clamp to it"
            );
            if min_base != 0 && free_ptr_floor.is_none_or(|floor| floor < min_base) {
                let clamp = format!(" {}.i256;", min_base - 1);
                assert!(
                    machine
                        .lines()
                        .any(|l| l.contains("= gt ") && l.ends_with(&clamp)),
                    "{name}: raises the base to {min_base}"
                );
            }
        }
        if func.restores_free_ptr {
            let entry = machine
                .lines()
                .find_map(|line| line.trim().strip_suffix(" = evm_mload 64.i256;"))
                .expect("the entry loads the free pointer");
            let entry = entry.split('.').next().unwrap();
            let lines: Vec<_> = machine.lines().map(str::trim).collect();
            for (i, _) in lines
                .iter()
                .enumerate()
                .filter(|(_, l)| l.starts_with("return"))
            {
                assert_eq!(
                    lines[i - 1],
                    format!("evm_mstore 64.i256 {entry};"),
                    "{name}: restores the free pointer before returning"
                );
            }
        }
    }
    checked
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
/// `check_spill_accesses`, are off.
#[dir_test(dir: "$CARGO_MANIFEST_DIR/test_files/evm", glob: "*.sntn")]
fn lowering_record_of_evm_fixtures_agrees_with_bytes(fixture: Fixture<&str>) {
    if fixture.content().contains("\nobject ") {
        check_fixture(fixture.content(), &[OptLevel::O0]);
    }
}

/// `%runtime` reads a table at an index from calldata through `%get`. The
/// backend's pipeline moves the table to a word blob and, but at O0,
/// specializes `%get` for the table: a global and a function the compiled
/// module does not have, named in the record as the section's IR names them.
/// The compiled module has no function with the new function's id.
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
        let prefixes = if opt == OptLevel::O0 { "$" } else { "%$" };
        for prefix in prefixes.chars() {
            assert!(
                new.iter().any(|symbol| symbol.starts_with(prefix)),
                "{opt:?}: no `{prefix}` symbol of the section's own in {new:?}"
            );
        }
    }
}

/// The source of the EVM fixture `file`.
fn evm_fixture(file: &str) -> String {
    let path = format!("{}/test_files/evm/{file}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(path).expect("fixture reads")
}

/// The stores and reloads of each spill of the function `name` of the first
/// section of `artifacts`, as `(value, location, stores, reloads)`.
fn spill_points(artifacts: &[ObjectArtifact], name: &str) -> Vec<String> {
    let section = artifacts[0].sections.values().next().expect("a section");
    let lowering = section.lowering.as_ref().expect("a lowering record");
    let starts = instruction_starts(&section.bytes, lowering.code_size as usize);
    assert!(
        check_spill_accesses(lowering, &section.bytes, &starts) > 0,
        "the stores and reloads are checked against the bytes"
    );
    let func = lowering
        .funcs
        .iter()
        .find(|func| func.name == name)
        .expect("the function is recorded");
    let point = |point: &CodePoint| match *point {
        CodePoint::Entry => "entry".to_string(),
        CodePoint::Before { block, index } => format!("before block{}:{index}", block.0),
        CodePoint::After { block, index } => format!("after block{}:{index}", block.0),
        CodePoint::Case { block, index, case } => {
            format!("case {case} of block{}:{index}", block.0)
        }
    };
    let points = |points: &[CodePoint]| points.iter().map(point).collect::<Vec<_>>().join(", ");
    func.spills
        .iter()
        .map(|spill| {
            format!(
                "v{} {:?} stores [{}] reloads [{}]",
                spill.value.as_u32(),
                spill.location,
                points(&spill.stores),
                points(&spill.reloads)
            )
        })
        .collect()
}

/// At O0 `%spill` keeps its arguments `v1` to `v7` in memory across the
/// calls of its loop: each is stored at entry and reloaded for the first call
/// (`block2:1`, after the phi), and `v7` again for the `add` at `block2:4`.
#[test]
fn lowering_record_lists_argument_stores_and_reloads() {
    let word = |value: u32| {
        let reloads = if value == 7 {
            "before block2:1, before block2:4"
        } else {
            "before block2:1"
        };
        format!(
            "v{value} Memory({}) stores [entry] reloads [{reloads}]",
            32 * (value - 1)
        )
    };
    assert_eq!(
        spill_points(
            &compile(&evm_fixture("spill.sntn"), OptLevel::O0, true),
            "spill"
        ),
        (1..=7).map(word).collect::<Vec<_>>()
    );
}

/// At O0 `%rotate`'s twenty rotating phis are stored on the edge from
/// `block0` and on the backedge from `block2`, before each block's `jump`,
/// and reloaded on the backedge for the rotation. The counter's increment
/// `v24` is not spilled: the backedge takes it from the stack, and only
/// reloaded values are stored.
#[test]
fn lowering_record_lists_phi_edge_stores_and_reloads() {
    let spills = spill_points(
        &compile(
            &evm_fixture("spilled_phi_edge_rotation.sntn"),
            OptLevel::O0,
            true,
        ),
        "rotate",
    );
    for expected in [
        "v0 Memory(0) stores [entry] reloads [before block1:21]",
        "v1 Memory(32) stores [before block0:0, before block2:1] reloads [before block2:1]",
        "v20 Memory(640) stores [before block0:0, before block2:1] \
         reloads [before block2:1, before block3:0]",
        "v21 Memory(672) stores [before block0:0, before block2:1] \
         reloads [before block1:21, before block2:0]",
    ] {
        assert!(
            spills.iter().any(|spill| spill == expected),
            "{expected} in {spills:#?}"
        );
    }
    assert_eq!(spills.len(), 22, "{spills:#?}");
}

/// `evm_return_code`, `evm_mload_bytes` and `trunc_nuw` reach the record
/// only in the source IR. Machine lowering makes `evm_return_code` an
/// `evm_code_copy` to address 0 and an `evm_return` of it, which ends the
/// init section, makes `evm_mload_bytes` an `evm_mload` and a shift, behind a
/// branch on a dynamic length, and replaces the boolean `trunc_nuw` with its
/// operand. The three blocks the dynamic-length load adds have no source
/// block.
#[test]
fn lowering_record_of_return_code_mload_bytes_and_trunc_nuw() {
    let source = r#"
target = "evm-ethereum-osaka"

func public %init() {
    block0:
        v0.i256 = sym_addr &runtime;
        v1.i256 = sym_size &runtime;
        evm_return_code v0 v1;
}

func public %runtime() {
    block0:
        v0.i256 = evm_calldata_load 0.i256;
        v1.i256 = evm_calldata_load 32.i256;
        v2.i1 = lt v1 33.i256;
        br v2 block1 block2;

    block1:
        v3.i256 = evm_mload_bytes v0 v1;
        v4.i256 = evm_mload_bytes v0 4.i256;
        v5.i256 = and v0 1.i256;
        v6.i1 = trunc_nuw v5 i1;
        v7.i256 = zext v6 i256;
        v8.i256 = add v3 v4;
        v9.i256 = add v8 v7;
        mstore 0.i256 v9 i256;
        evm_return 0.i256 32.i256;

    block2:
        evm_revert 0.i256 0.i256;
}

object @Contract {
  section init {
    entry %init;
    embed .runtime as &runtime;
  }
  section runtime {
    entry %runtime;
  }
}
"#;
    let levels = [OptLevel::O0, OptLevel::O1, OptLevel::Os];
    check_fixture(source, &levels);
    for opt in levels {
        let artifacts = compile(source, opt, true);
        let lowering = |section: &str| {
            artifacts[0]
                .sections
                .iter()
                .find(|(name, _)| name.0.as_str() == section)
                .and_then(|(_, section)| section.lowering.clone())
                .expect("a lowering record")
        };
        let init = lowering("init");
        assert!(init.source_ir.contains("evm_return_code v0 v1;"), "{opt:?}");
        assert!(
            init.machine_ir
                .contains("evm_code_copy 0.i256 v0 v1;\n        evm_return 0.i256 v1;\n}"),
            "{opt:?}: {}",
            init.machine_ir
        );
        let runtime = lowering("runtime");
        for inst in ["evm_mload_bytes", "trunc_nuw"] {
            assert!(runtime.source_ir.contains(inst), "{opt:?}: {inst}");
        }
        for inst in ["evm_return_code", "evm_mload_bytes", "trunc_nuw"] {
            assert!(
                !init.machine_ir.contains(inst) && !runtime.machine_ir.contains(inst),
                "{opt:?}: {inst} in the machine IR"
            );
        }
        let sourceless = runtime.funcs[0]
            .blocks
            .iter()
            .filter(|block| block.sources.is_empty())
            .count();
        assert_eq!(sourceless, 3, "{opt:?}: {}", runtime.machine_ir);
    }
}

/// At stack reach 4, `%rotate`'s phis `v2` to `v16` share the words of their
/// sources on the entry edge, `v25` to `v53`, which die there. The edge would
/// reload each source only to store it back as its phi, and final spills drop
/// both: the record lists each source's store and no reload, and no store of
/// its phi on the entry edge.
#[test]
fn lowering_record_omits_stores_back_to_a_shared_word() {
    let module = parse_module(&evm_fixture("spilled_phi_shares_source_word.sntn"))
        .expect("fixture parses")
        .module;
    let compile = |record| {
        let backend = EvmBackend::new(Evm::new(module.ctx.triple))
            .with_stackify_reach_depth(4)
            .with_lowering_record(record);
        let opts = CompileOptions {
            emit_lowering: record,
            ..CompileOptions::default()
        };
        compile_all_objects(&module, &backend, &opts).expect("fixture compiles")
    };
    let (plain, recorded) = (compile(false), compile(true));
    let section = recorded[0].sections.values().next().expect("a section");
    let plain = plain[0].sections.values().next().expect("a section");
    assert_eq!(plain.bytes, section.bytes);
    check_record(
        section.lowering.as_ref().expect("a lowering record"),
        &section.bytes,
    );
    let spills = spill_points(&recorded, "rotate");
    for expected in [
        "v2 Memory(96) stores [before block2:1] reloads [before block2:1]",
        "v25 Memory(96) stores [after block0:1] reloads []",
        "v16 Memory(544) stores [before block2:1] reloads [before block2:1]",
        "v53 Memory(544) stores [after block0:15] reloads []",
    ] {
        assert!(
            spills.iter().any(|spill| spill == expected),
            "{expected} in {spills:#?}"
        );
    }
    assert_eq!(
        spills
            .iter()
            .filter(|spill| spill.ends_with("reloads []"))
            .count(),
        15,
        "{spills:#?}"
    );
}

/// The lowering records of the sections of `source`, compiled at O0 and
/// checked.
fn o0_records(source: &str) -> Vec<SectionLowering> {
    check_fixture(source, &[OptLevel::O0]);
    compile(source, OptLevel::O0, true)
        .iter()
        .flat_map(|artifact| artifact.sections.values())
        .map(|section| section.lowering.clone().expect("a lowering record"))
        .collect()
}

/// The recorded function `name` of `records`.
fn recorded_func<'a>(records: &'a [SectionLowering], name: &str) -> &'a FuncLowering {
    records
        .iter()
        .flat_map(|lowering| &lowering.funcs)
        .find(|func| func.name == name)
        .unwrap_or_else(|| panic!("{name} is recorded"))
}

/// `%entry`'s `alloca` lives across the call that reads it, so it is in
/// `%entry`'s stable frame; `%keep`'s is in the scratch overlay. The code
/// stores each marker at the address the record gives.
#[test]
fn lowering_record_places_scratch_and_stable_allocas() {
    let source = r#"
target = "evm-ethereum-osaka"

func inline(never) private %keep(v0.*i256) -> i256 {
    block0:
        v1.i256 = mload v0 i256;
        v2.*i256 = alloca i256;
        mstore v2 1001.i256 i256;
        v3.i256 = mload v2 i256;
        v4.i256 = add v1 v3;
        return v4;
}

func public %entry() {
    block0:
        v0.*i256 = alloca i256;
        mstore v0 1002.i256 i256;
        v1.i256 = call %keep v0;
        mstore 0.i256 v1 i256;
        evm_return 0.i256 32.i256;
}

object @Allocas {
  section runtime {
    entry %entry;
  }
}
"#;
    let records = o0_records(source);
    assert_eq!(check_placement(&records[0]), 2, "both markers are checked");
    let loc = |name| recorded_func(&records, name).allocas[0].1;
    assert!(
        matches!(loc("keep").loc, ObjLocLowering::ScratchAbs(_)),
        "{:?}",
        loc("keep")
    );
    let entry = recorded_func(&records, "entry");
    let (ObjLocLowering::StableAbs(word), StableLowering::Abs { base_word }) =
        (entry.allocas[0].1.loc, entry.frame.stable)
    else {
        panic!("{entry:?}");
    };
    assert!(base_word <= word && word < base_word + entry.frame.stable_words);
}

/// Each rule of a malloc's placement is recorded: the fixed rules but
/// `ExactEntry`, which no fixture places, and of heap mallocs the clamp to
/// the dynamic stack pointer and a free-pointer floor that drops the clamp
/// to the minimum base, or none.
#[test]
fn lowering_record_gives_each_malloc_rule() {
    for (fixture, kind) in [
        ("fe_wide_abi32_sum.sntn", "terminal"),
        ("malloc_transient_symbolic_size_fixed_base.sntn", "program"),
        (
            "free_ptr_restore_skips_raw_allocator.sntn",
            "private static",
        ),
        (
            "lazy_frame_enter_over_transient_malloc.sntn",
            "dynamic stack pointer clamp",
        ),
        ("callee_heap_escape_free_ptr.sntn", "floor drops the clamp"),
        (
            "heap_malloc_free_ptr_entry_reentry.sntn",
            "clamp to the minimum base",
        ),
    ] {
        let is = |malloc: &MallocLowering| match (kind, *malloc) {
            ("terminal", MallocLowering::Fixed { rule, .. }) => rule == FixedMallocRule::Terminal,
            ("program", MallocLowering::Fixed { rule, .. }) => rule == FixedMallocRule::Program,
            ("private static", MallocLowering::Fixed { rule, .. }) => {
                rule == FixedMallocRule::PrivateStatic
            }
            (
                "dynamic stack pointer clamp",
                MallocLowering::Heap {
                    needs_dyn_sp_clamp, ..
                },
            ) => needs_dyn_sp_clamp,
            (
                "floor drops the clamp" | "clamp to the minimum base",
                MallocLowering::Heap {
                    exact_base: None,
                    min_base,
                    free_ptr_floor,
                    ..
                },
            ) => {
                let drops = free_ptr_floor.is_some_and(|floor| floor >= min_base);
                drops == (kind == "floor drops the clamp") && min_base != 0
            }
            _ => false,
        };
        let records = o0_records(&evm_fixture(fixture));
        let mallocs: Vec<_> = records
            .iter()
            .flat_map(|lowering| &lowering.funcs)
            .flat_map(|func| &func.mallocs)
            .map(|(_, malloc)| *malloc)
            .collect();
        assert!(
            mallocs.iter().any(is),
            "{fixture}: no {kind} malloc in {mallocs:?}"
        );
    }
}

/// A function that releases the heap it allocated, which only optimized
/// output does, is recorded as restoring the free pointer, and
/// `check_placement` finds the save and the restores in its machine IR.
/// `%spill` of `arena_alloc_spills.sntn` spills to both of its reserves and
/// to the fixed slots.
#[test]
fn lowering_record_gives_restores_reserves_and_fixed_slots() {
    let source = evm_fixture("free_ptr_restore_releases_callee_heap.sntn");
    check_fixture(&source, &[OptLevel::O1]);
    assert!(
        compile(&source, OptLevel::O1, true)
            .iter()
            .flat_map(|artifact| artifact.sections.values())
            .flat_map(|section| &section.lowering.as_ref().expect("a record").funcs)
            .any(|func| func.restores_free_ptr)
    );
    let records = o0_records(&evm_fixture("arena_alloc_spills.sntn"));
    assert!(records.iter().any(|lowering| lowering.fixed_slot_spills));
    let funcs: Vec<_> = records
        .iter()
        .flat_map(|lowering| &lowering.funcs)
        .collect();
    assert!(
        funcs
            .iter()
            .any(|func| func.spill_reserve.scratch.words > 0)
    );
    assert!(funcs.iter().any(|func| {
        func.spill_reserve
            .stable
            .is_some_and(|stable| stable.words > 0)
    }));
}
