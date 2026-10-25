//! Memory, aggregates, objects, globals and the heap.

use super::{Instance, LEVELS, compile, try_compile};

fn words(instance: &Instance, address: u32, count: usize) -> Vec<i64> {
    instance
        .read(address, count * 8)
        .chunks(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn aggregate_values_are_snapshots() {
    let source = r#"
type @Pair = { i64, i64 };
type @Outer = { @Pair, i64 };
func public %snapshots(v0.i64, v1.i64, v2.*[i64; 6]) {
block0:
    v3.@Pair = insert_value undef.@Pair 0.i64 v0;
    v4.@Pair = insert_value v3 1.i64 v1;
    v5.@Outer = insert_value undef.@Outer 0.i64 v4;
    v6.@Outer = insert_value v5 1.i64 7.i64;
    v7.@Pair = extract_value v6 0.i64;
    v8.@Pair = insert_value v7 0.i64 91.i64;
    v9.@Outer = insert_value v6 0.i64 v8;
    v10.@Outer = insert_value v9 1.i64 93.i64;
    v11.@Pair = extract_value v6 0.i64;
    v12.i64 = extract_value v11 0.i64;
    v13.i64 = extract_value v11 1.i64;
    v14.@Pair = extract_value v10 0.i64;
    v15.i64 = extract_value v14 0.i64;
    v16.i64 = extract_value v14 1.i64;
    v17.i64 = extract_value v6 1.i64;
    v18.i64 = extract_value v10 1.i64;
    v19.[i64; 6] = insert_value undef.[i64; 6] 0.i64 v12;
    v20.[i64; 6] = insert_value v19 1.i64 v13;
    v21.[i64; 6] = insert_value v20 2.i64 v15;
    v22.[i64; 6] = insert_value v21 3.i64 v16;
    v23.[i64; 6] = insert_value v22 4.i64 v17;
    v24.[i64; 6] = insert_value v23 5.i64 v18;
    mstore v2 v24 [i64; 6];
    return;
}
"#;
    for level in LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        let out = instance.scratch(48);
        instance
            .call::<_, ()>("snapshots", (11i64, 22i64, out as i32))
            .unwrap();
        assert_eq!(words(&instance, out, 6), [11, 22, 91, 22, 7, 93]);
    }
}

#[test]
fn aggregate_phis_swap_without_clobbering() {
    let source = r#"
func public %swap(v0.i64, v1.*[i64; 4]) {
block0:
    v2.[i64; 2] = insert_value undef.[i64; 2] 0.i64 1.i64;
    v3.[i64; 2] = insert_value v2 1.i64 2.i64;
    v4.[i64; 2] = insert_value undef.[i64; 2] 0.i64 3.i64;
    v5.[i64; 2] = insert_value v4 1.i64 4.i64;
    jump block1;
block1:
    v6.[i64; 2] = phi (v3 block0) (v7 block1);
    v7.[i64; 2] = phi (v5 block0) (v6 block1);
    v8.i64 = phi (0.i64 block0) (v9 block1);
    v9.i64 = add v8 1.i64;
    v10.i1 = lt v9 v0;
    br v10 block1 block2;
block2:
    v11.i64 = extract_value v6 0.i64;
    v12.i64 = extract_value v6 1.i64;
    v13.i64 = extract_value v7 0.i64;
    v14.i64 = extract_value v7 1.i64;
    v15.*i64 = gep v1 0.i64 0.i64;
    mstore v15 v11 i64;
    v16.*i64 = gep v1 0.i64 1.i64;
    mstore v16 v12 i64;
    v17.*i64 = gep v1 0.i64 2.i64;
    mstore v17 v13 i64;
    v18.*i64 = gep v1 0.i64 3.i64;
    mstore v18 v14 i64;
    return;
}
"#;
    for level in LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        let out = instance.scratch(32);
        for (iterations, expected) in [(1i64, [1, 2, 3, 4]), (2, [3, 4, 1, 2]), (5, [1, 2, 3, 4])] {
            instance
                .call::<_, ()>("swap", (iterations, out as i32))
                .unwrap();
            assert_eq!(words(&instance, out, 4), expected, "{iterations} {level:?}");
        }
    }
}

#[test]
fn aggregate_arguments_and_results_copy_whole_values() {
    let source = r#"
type @Big = { i256, [i64; 3], i8 };
func private %make(v0.i64) -> @Big {
block0:
    v1.i256 = zext v0 i256;
    v2.i256 = shl 200.i256 v1;
    v3.@Big = insert_value undef.@Big 0.i64 v2;
    v4.[i64; 3] = insert_value undef.[i64; 3] 0.i64 v0;
    v5.[i64; 3] = insert_value v4 2.i64 9.i64;
    v6.@Big = insert_value v3 1.i64 v5;
    v7.@Big = insert_value v6 2.i64 -1.i8;
    return v7;
}
func private %bump(v0.@Big) -> @Big {
block0:
    v1.[i64; 3] = extract_value v0 1.i64;
    v2.i64 = extract_value v1 2.i64;
    v3.i64 = add v2 1.i64;
    v4.[i64; 3] = insert_value v1 2.i64 v3;
    v5.@Big = insert_value v0 1.i64 v4;
    return v5;
}
func public %run(v0.i64, v1.*@Big, v2.*@Big) {
block0:
    v3.@Big = call %make v0;
    v4.@Big = call %bump v3;
    mstore v1 v3 @Big;
    mstore v2 v4 @Big;
    return;
}
"#;
    for level in LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        let out = instance.scratch(128);
        instance
            .call::<_, ()>("run", (5i64, out as i32, out as i32 + 64))
            .unwrap();
        for (base, last) in [(out, 9), (out + 64, 10)] {
            let fields = words(&instance, base, 7);
            assert_eq!(fields[..4], [0, 0, 0, 5 << 8], "{level:?}");
            assert_eq!(fields[4..], [5, 0, last], "{level:?}");
            assert_eq!(instance.read(base + 56, 1), [0xff]);
        }
    }
}

#[test]
fn objects_hold_state_across_calls() {
    let source = r#"
func private %add_to(v0.objref<i64>, v1.i64) {
block0:
    v2.i64 = obj.load v0;
    v3.i64 = add v2 v1;
    obj.store v0 v3;
    return;
}
func public %accumulate(v0.i64) -> i64 {
block0:
    v1.objref<[i64; 4]> = obj.alloc [i64; 4];
    v2.objref<i64> = obj.index v1 2.i64;
    obj.store v2 0.i64;
    jump block1;
block1:
    v3.i64 = phi (0.i64 block0) (v4 block1);
    v4.i64 = add v3 1.i64;
    call %add_to v2 v4;
    v5.i1 = lt v4 v0;
    br v5 block1 block2;
block2:
    v6.i64 = obj.load v2;
    return v6;
}
"#;
    for level in LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        assert_eq!(instance.call::<_, i64>("accumulate", 10i64).unwrap(), 55);
    }
}

#[test]
fn globals_live_in_linear_memory() {
    let source = r#"
type @Record = { i8, i128, [i16; 3], i256 };
global public i64 $counter = 41;
global private i64 $zero;
global public const @Record $data = { 1, 2, [3, 4, 5], 6 };
func public %bump(v0.i64) -> i64 {
block0:
    v1.i64 = mload $counter i64;
    v2.i64 = add v1 v0;
    mstore $counter v2 i64;
    return v2;
}
func public %read(v0.i8) -> i64 {
block0:
    v1.i1 = is_zero v0;
    br v1 block1 block2;
block1:
    jump block3;
block2:
    jump block3;
block3:
    v2.*i64 = phi ($counter block1) ($zero block2);
    v3.i64 = mload v2 i64;
    return v3;
}
func public %field() -> i64 {
block0:
    v0.constref<@Record> = const.ref $data;
    v1.constref<[i16; 3]> = const.proj v0 2.i64;
    v2.constref<i16> = const.index v1 2.i64;
    v3.i16 = const.load v2;
    v4.constref<i256> = const.proj v0 3.i64;
    v5.i256 = const.load v4;
    v6.i64 = zext v3 i64;
    v7.i64 = trunc v5 i64;
    v8.i64 = mul v6 v7;
    return v8;
}
"#;
    for level in LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        assert_eq!(instance.call::<_, i64>("bump", 1i64).unwrap(), 42);
        assert_eq!(instance.call::<_, i64>("bump", 2i64).unwrap(), 44);
        assert_eq!(instance.call::<_, i64>("read", 0).unwrap(), 44);
        assert_eq!(instance.call::<_, i64>("read", 1).unwrap(), 0);
        assert_eq!(instance.call::<_, i64>("field", ()).unwrap(), 30);
    }
}

#[test]
fn heap_allocations_are_distinct_and_reused() {
    let source = r#"
declare external %malloc(i32) -> *i8;
declare external %free(*i8);
declare external %memmove(*i8, *i8, i32) -> *i8;
func public %alloc(v0.i32) -> *i8 {
block0:
    v1.*i8 = call %malloc v0;
    return v1;
}
func public %release(v0.*i8) {
block0:
    call %free v0;
    return;
}
func public %dynamic(v0.i256) -> *i8 {
block0:
    v1.*i8 = mem.alloc_dynamic v0;
    return v1;
}
func public %move(v0.*i8, v1.*i8, v2.i32) -> *i8 {
block0:
    v3.*i8 = call %memmove v0 v1 v2;
    return v3;
}
"#;
    for level in LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        let a: i32 = instance.call("alloc", 24).unwrap();
        let b: i32 = instance.call("alloc", 24).unwrap();
        let empty: i32 = instance.call("alloc", 0).unwrap();
        assert_ne!(a, 0);
        assert_eq!(a % 16, 0);
        assert!(b >= a + 32, "{a} {b}");
        assert!(empty != a && empty != b && empty != 0);
        instance.call::<_, ()>("release", a).unwrap();
        assert_eq!(instance.call::<_, i32>("alloc", 30).unwrap(), a);
        // Allocations grow memory as needed.
        let large: i32 = instance.call("alloc", 10 << 20).unwrap();
        instance.write(large as u32 + (10 << 20) - 1, &[1]);
        assert_eq!(instance.call::<_, i32>("alloc", (1 << 30) + 1).unwrap(), 0);
        let c: i32 = instance
            .call("dynamic", (100i64, 0i64, 0i64, 0i64))
            .unwrap();
        assert_ne!(c, 0);
        assert!(
            instance
                .call::<_, i32>("dynamic", (1i64, 0i64, 1i64, 0i64))
                .is_err()
        );

        instance.write(c as u32, b"abcdefgh");
        let moved: i32 = instance.call("move", (c + 2, c, 6)).unwrap();
        assert_eq!(moved, c + 2);
        assert_eq!(instance.read(c as u32, 8), b"ababcdef");
    }
}

#[test]
fn runtime_symbols_require_c_signatures() {
    let source = r#"
declare external %malloc(i64) -> *i8;
func public %alloc(v0.i64) -> *i8 {
block0:
    v1.*i8 = call %malloc v0;
    return v1;
}
"#;
    let errors = try_compile(source, sonatina_codegen::OptLevel::O0).unwrap_err();
    assert!(
        format!("{errors:?}").contains("wasm32 C signature"),
        "{errors:?}"
    );
}

#[test]
fn heap_objects_outlive_their_frames() {
    let source = r#"
func private %make(v0.i64) -> *[i64; 2] {
block0:
    v1.objref<[i64; 2]> = obj.alloc [i64; 2];
    v2.objref<i64> = obj.index v1 0.i64;
    obj.store v2 v0;
    v3.objref<i64> = obj.index v1 1.i64;
    v4.i64 = add v0 1.i64;
    obj.store v3 v4;
    v5.*[i64; 2] = obj.materialize.heap v1;
    return v5;
}
func public %sum(v0.i64) -> i64 {
block0:
    v1.*[i64; 2] = call %make v0;
    v2.*[i64; 2] = call %make 100.i64;
    v3.*i64 = gep v1 0.i64 1.i64;
    v4.i64 = mload v3 i64;
    v5.*i64 = gep v2 0.i64 0.i64;
    v6.i64 = mload v5 i64;
    v7.i64 = add v4 v6;
    return v7;
}
"#;
    for level in LEVELS {
        let mut instance = Instance::new(&compile(source, level));
        assert_eq!(instance.call::<_, i64>("sum", 5i64).unwrap(), 106);
    }
}
