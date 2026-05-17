#![allow(clippy::crosspointer_transmute)]

use sonatina_codegen::{Backend, Compile, OptLevel, isa::cranelift::CraneliftBackend};
use sonatina_ir::{
    I256, Immediate, Linkage, Signature, Type, U256,
    builder::ModuleBuilder,
    func_cursor::InstInserter,
    global_variable::{GlobalVariableData, GvInitializer},
    inst::{arith, cast, cmp, control_flow, data, logic},
    isa::{Isa, native::Native},
    module::ModuleCtx,
};
use sonatina_triple::{Architecture, OperatingSystem, TargetTriple, Vendor};

fn native_triple() -> TargetTriple {
    let arch = if cfg!(target_arch = "x86_64") {
        Architecture::X86_64
    } else if cfg!(target_arch = "aarch64") {
        Architecture::Aarch64
    } else {
        panic!("unsupported host architecture for cranelift tests");
    };
    TargetTriple::new(arch, Vendor::Unknown, OperatingSystem::Native)
}

fn native_isa() -> Native {
    Native::new(native_triple())
}

fn native_module_builder() -> ModuleBuilder {
    let isa = native_isa();
    let ctx = ModuleCtx::new(&isa);
    ModuleBuilder::new(ctx)
}

#[test]
fn cranelift_add_two_i64s() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };

    let sig = Signature::new_single(
        "add_i64",
        Linkage::Public,
        &[Type::I64, Type::I64],
        Type::I64,
    );
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let a = fb.args()[0];
    let b = fb.args()[1];
    let sum = fb.insert_inst(arith::Add::new(is, a, b), Type::I64);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let add_fn: fn(i64, i64) -> i64 = unsafe {
        let ptr = artifact
            .get_func_ptr::<fn(i64, i64) -> i64>("add_i64")
            .unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(add_fn(3, 4), 7);
    assert_eq!(add_fn(-10, 25), 15);
    assert_eq!(add_fn(0, 0), 0);
}

#[test]
fn cranelift_arithmetic_chain() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };

    let sig = Signature::new_single("arith", Linkage::Public, &[Type::I32, Type::I32], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let a = fb.args()[0];
    let b = fb.args()[1];
    let sum = fb.insert_inst(arith::Add::new(is, a, b), Type::I32);
    let diff = fb.insert_inst(arith::Sub::new(is, a, b), Type::I32);
    let product = fb.insert_inst(arith::Mul::new(is, sum, diff), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, product));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn(i32, i32) -> i32 = unsafe {
        let ptr = artifact
            .get_func_ptr::<fn(i32, i32) -> i32>("arith")
            .unwrap();
        std::mem::transmute(ptr)
    };

    // (5+3) * (5-3) = 8 * 2 = 16
    assert_eq!(f(5, 3), 16);
    // (10+7) * (10-7) = 17 * 3 = 51
    assert_eq!(f(10, 7), 51);
}

#[test]
fn cranelift_native_casts_round_trip_pointer_values() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();
    let objref_ty = mb.objref_type(Type::I64);

    let sig = Signature::new_single("cast_round_trip", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let slot = fb.insert_inst(data::ObjAlloc::new(is, Type::I64), objref_ty);
    let addr = fb.insert_inst(cast::PtrToInt::new(is, slot, Type::I64), Type::I64);
    let cast_addr = fb.insert_inst(cast::Bitcast::new(is, addr, Type::I64), Type::I64);
    let ptr = fb.insert_inst(cast::IntToPtr::new(is, cast_addr, objref_ty), objref_ty);
    let value = fb.make_imm_value(123i64);
    fb.insert_inst_no_result(data::ObjStore::new(is, ptr, value));
    let loaded = fb.insert_inst(data::ObjLoad::new(is, ptr), Type::I64);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, loaded));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn() -> i64 = unsafe {
        let ptr = artifact
            .get_func_ptr::<fn() -> i64>("cast_round_trip")
            .unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(), 123);
}

#[test]
fn cranelift_native_i256_address_casts_and_memory_round_trip() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();
    let ptr_ty = mb.ptr_type(Type::I32);

    let sig = Signature::new_single("i256_memory_round_trip", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let slot = fb.insert_inst(data::Alloca::new(is, Type::I32), ptr_ty);
    let zero = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(data::Mstore::new(is, slot, zero, Type::I32));

    let slot_addr = fb.insert_inst(cast::PtrToInt::new(is, slot, Type::I256), Type::I256);
    let loaded_word = fb.insert_inst(data::Mload::new(is, slot_addr, Type::I256), Type::I256);
    let loaded_i32 = fb.insert_inst(cast::Trunc::new(is, loaded_word, Type::I32), Type::I32);
    let five = fb.make_imm_value(5i32);
    let scalar_sum = fb.insert_inst(arith::Add::new(is, loaded_i32, five), Type::I32);
    let widened = fb.insert_inst(cast::Sext::new(is, scalar_sum, Type::I256), Type::I256);
    let seven = fb.make_imm_value(Immediate::from_i256(I256::from(7u8), Type::I256));
    let word_sum = fb.insert_inst(arith::Add::new(is, widened, seven), Type::I256);
    let twenty = fb.make_imm_value(Immediate::from_i256(I256::from(20u8), Type::I256));
    let is_below_twenty = fb.insert_inst(cmp::Lt::new(is, word_sum, twenty), Type::I1);

    fb.insert_inst_no_result(data::Mstore::new(is, slot_addr, word_sum, Type::I256));
    let stored_low = fb.insert_inst(data::Mload::new(is, slot, Type::I32), Type::I32);
    let flag = fb.insert_inst(cast::Zext::new(is, is_below_twenty, Type::I32), Type::I32);
    let result = fb.insert_inst(arith::Add::new(is, stored_low, flag), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, result));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn() -> i32 = unsafe {
        let ptr = artifact
            .get_func_ptr::<fn() -> i32>("i256_memory_round_trip")
            .unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(), 13);
}

#[test]
fn cranelift_native_i256_bool_sext_is_zero_one() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    let sig = Signature::new_single("i256_bool_sext", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let true_value = fb.make_imm_value(true);
    let widened = fb.insert_inst(cast::Sext::new(is, true_value, Type::I256), Type::I256);
    let narrowed = fb.insert_inst(cast::Trunc::new(is, widened, Type::I32), Type::I32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, narrowed));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn() -> i32 = unsafe {
        let ptr = artifact
            .get_func_ptr::<fn() -> i32>("i256_bool_sext")
            .unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(), 1);
}

#[test]
fn cranelift_native_i256_feasible_integer_ops() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    let sig = Signature::new_single("i256_integer_ops", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    macro_rules! word {
        ($value:expr) => {
            fb.make_imm_value(Immediate::from_i256($value, Type::I256))
        };
    }

    let three_hi = word!(I256::from(U256::from(3u8) << 64));
    let one_hi = word!(I256::from(U256::one() << 64));
    let three = word!(I256::from(3u8));
    let product = fb.insert_inst(arith::Mul::new(is, one_hi, three), Type::I256);
    let product_ok = fb.insert_inst(cmp::Eq::new(is, product, three_hi), Type::I1);

    let five = word!(I256::from(5u8));
    let neg_five = fb.insert_inst(arith::Neg::new(is, five), Type::I256);
    let neg_five = fb.insert_inst(cast::Trunc::new(is, neg_five, Type::I32), Type::I32);

    let zero_word = word!(I256::zero());
    let not_zero = fb.insert_inst(logic::Not::new(is, zero_word), Type::I256);
    let not_zero = fb.insert_inst(cast::Trunc::new(is, not_zero, Type::I32), Type::I32);

    let mask_lhs = word!(I256::from(0xf0u16));
    let mask_rhs = word!(I256::from(0xccu16));
    let mask = fb.insert_inst(logic::And::new(is, mask_lhs, mask_rhs), Type::I256);
    let or_rhs = word!(I256::from(0x03u8));
    let masked = fb.insert_inst(logic::Or::new(is, mask, or_rhs), Type::I256);
    let xor_rhs = word!(I256::from(0x0fu8));
    let xored = fb.insert_inst(logic::Xor::new(is, masked, xor_rhs), Type::I256);
    let xored = fb.insert_inst(cast::Trunc::new(is, xored, Type::I32), Type::I32);

    let one = word!(I256::from(1u8));
    let shift_65 = word!(I256::from(65u8));
    let shift_64 = word!(I256::from(64u8));
    let shifted_left = fb.insert_inst(arith::Shl::new(is, shift_65, one), Type::I256);
    let shifted_right = fb.insert_inst(arith::Shr::new(is, shift_64, shifted_left), Type::I256);
    let shifted_right = fb.insert_inst(cast::Trunc::new(is, shifted_right, Type::I32), Type::I32);

    let negative_eight = word!(I256::from(-8i8));
    let shift_1 = word!(I256::from(1u8));
    let shifted_arithmetic =
        fb.insert_inst(arith::Sar::new(is, shift_1, negative_eight), Type::I256);
    let shifted_arithmetic = fb.insert_inst(
        cast::Trunc::new(is, shifted_arithmetic, Type::I32),
        Type::I32,
    );

    let product_ok = fb.insert_inst(cast::Zext::new(is, product_ok, Type::I32), Type::I32);
    let mut acc = product_ok;
    for value in [neg_five, not_zero, xored, shifted_right, shifted_arithmetic] {
        acc = fb.insert_inst(arith::Add::new(is, acc, value), Type::I32);
    }

    fb.insert_inst_no_result(control_flow::Return::new_single(is, acc));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = CraneliftBackend::new()
        .compile_module(&module)
        .expect("compilation failed");

    let f: fn() -> i32 = unsafe {
        let ptr = artifact
            .get_func_ptr::<fn() -> i32>("i256_integer_ops")
            .unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(), 197);

    let object = CraneliftBackend::new()
        .compile_module_to_object(&module)
        .expect("object compilation failed");
    assert!(!object.bytes.is_empty());
    assert!(object.func_map.contains_key("i256_integer_ops"));
}

#[test]
fn cranelift_native_i256_div_checked_and_saturating_ops() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    let sig = Signature::new_single("i256_div_checked_sat_ops", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    macro_rules! word {
        ($value:expr) => {
            fb.make_imm_value(Immediate::from_i256($value, Type::I256))
        };
    }

    macro_rules! add_bool {
        ($acc:ident, $value:expr) => {{
            let value = fb.insert_inst(cast::Zext::new(is, $value, Type::I32), Type::I32);
            $acc = fb.insert_inst(arith::Add::new(is, $acc, value), Type::I32);
        }};
    }

    macro_rules! add_word_eq {
        ($acc:ident, $value:expr, $expected:expr) => {{
            let expected = word!($expected);
            let matches = fb.insert_inst(cmp::Eq::new(is, $value, expected), Type::I1);
            add_bool!($acc, matches);
        }};
    }

    let zero_i32 = fb.make_imm_value(0i32);
    let mut acc = zero_i32;

    let big = (U256::one() << 130) + U256::from(123u8);
    let seven_u = U256::from(7u8);
    let big_word = word!(I256::from(big));
    let seven_word = word!(I256::from(seven_u));
    let udiv = fb.insert_inst(arith::Udiv::new(is, big_word, seven_word), Type::I256);
    let umod = fb.insert_inst(arith::Umod::new(is, big_word, seven_word), Type::I256);
    add_word_eq!(acc, udiv, I256::from(big / seven_u));
    add_word_eq!(acc, umod, I256::from(big % seven_u));

    let negative_hundred = word!(I256::from(-100i8));
    let seven = word!(I256::from(7u8));
    let sdiv = fb.insert_inst(arith::Sdiv::new(is, negative_hundred, seven), Type::I256);
    let smod = fb.insert_inst(arith::Smod::new(is, negative_hundred, seven), Type::I256);
    add_word_eq!(acc, sdiv, I256::from(-14i8));
    add_word_eq!(acc, smod, I256::from(-2i8));

    let signed_min = I256::from(U256::one() << 255);
    let signed_max = I256::from((U256::one() << 255) - U256::one());
    let unsigned_max = I256::all_one();
    let min_word = word!(signed_min);
    let negative_one = word!(I256::from(-1i8));
    let min_div = fb.insert_inst(arith::Sdiv::new(is, min_word, negative_one), Type::I256);
    let min_mod = fb.insert_inst(arith::Smod::new(is, min_word, negative_one), Type::I256);
    add_word_eq!(acc, min_div, signed_min);
    add_word_eq!(acc, min_mod, I256::zero());

    let zero = word!(I256::zero());
    let one = word!(I256::one());
    let two = word!(I256::from(2u8));
    let unsigned_max_word = word!(unsigned_max);
    let signed_max_word = word!(signed_max);
    let signed_min_word = word!(signed_min);

    let [uadd_raw, uadd_overflow] = fb.insert_uaddo(unsigned_max_word, one);
    add_word_eq!(acc, uadd_raw, I256::zero());
    add_bool!(acc, uadd_overflow);

    let [sadd_raw, sadd_overflow] = fb.insert_saddo(signed_max_word, one);
    add_word_eq!(acc, sadd_raw, signed_min);
    add_bool!(acc, sadd_overflow);

    let [usub_raw, usub_overflow] = fb.insert_usubo(zero, one);
    add_word_eq!(acc, usub_raw, unsigned_max);
    add_bool!(acc, usub_overflow);

    let [ssub_raw, ssub_overflow] = fb.insert_ssubo(signed_min_word, one);
    add_word_eq!(acc, ssub_raw, signed_max);
    add_bool!(acc, ssub_overflow);

    let high_a = word!(I256::from(U256::one() << 128));
    let high_b = word!(I256::from(U256::one() << 128));
    let [umul_raw, umul_overflow] = fb.insert_umulo(high_a, high_b);
    add_word_eq!(acc, umul_raw, I256::zero());
    add_bool!(acc, umul_overflow);

    let [smul_raw, smul_overflow] = fb.insert_smulo(signed_max_word, two);
    add_word_eq!(acc, smul_raw, I256::from(-2i8));
    add_bool!(acc, smul_overflow);

    let [sneg_raw, sneg_overflow] = fb.insert_snego(signed_min_word);
    add_word_eq!(acc, sneg_raw, signed_min);
    add_bool!(acc, sneg_overflow);

    let uaddsat = fb.insert_uaddsat(unsigned_max_word, one);
    add_word_eq!(acc, uaddsat, unsigned_max);

    let saddsat = fb.insert_saddsat(signed_max_word, one);
    add_word_eq!(acc, saddsat, signed_max);

    let usubsat = fb.insert_usubsat(zero, one);
    add_word_eq!(acc, usubsat, I256::zero());

    let ssubsat = fb.insert_ssubsat(signed_min_word, one);
    add_word_eq!(acc, ssubsat, signed_min);

    let high_a = word!(I256::from(U256::one() << 128));
    let high_b = word!(I256::from(U256::one() << 128));
    let umulsat = fb.insert_umulsat(high_a, high_b);
    add_word_eq!(acc, umulsat, unsigned_max);

    let smulsat_hi = fb.insert_smulsat(signed_max_word, two);
    add_word_eq!(acc, smulsat_hi, signed_max);

    let smulsat_lo = fb.insert_smulsat(signed_min_word, two);
    add_word_eq!(acc, smulsat_lo, signed_min);

    fb.insert_inst_no_result(control_flow::Return::new_single(is, acc));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = CraneliftBackend::new()
        .compile_module(&module)
        .expect("compilation failed");

    let f: fn() -> i32 = unsafe {
        let ptr = artifact
            .get_func_ptr::<fn() -> i32>("i256_div_checked_sat_ops")
            .unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(), 27);

    let object = CraneliftBackend::new()
        .compile_module_to_object(&module)
        .expect("object compilation failed");
    assert!(!object.bytes.is_empty());
    assert!(object.func_map.contains_key("i256_div_checked_sat_ops"));
}

#[test]
fn cranelift_native_integer_edge_ops() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    let sig = Signature::new_single("integer_edge_ops", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let ten = fb.make_imm_value(10i64);
    let three = fb.make_imm_value(3i64);
    let neg_ten = fb.make_imm_value(-10i64);
    let urem = fb.insert_inst(arith::Umod::new(is, ten, three), Type::I64);
    let srem = fb.insert_inst(arith::Smod::new(is, neg_ten, three), Type::I64);

    let i32_max = fb.make_imm_value(i32::MAX);
    let i32_min = fb.make_imm_value(i32::MIN);
    let i32_all_ones = fb.make_imm_value(-1i32);
    let i32_one = fb.make_imm_value(1i32);
    let i32_two = fb.make_imm_value(2i32);
    let i32_zero = fb.make_imm_value(0i32);
    let [_, uadd_overflow] = fb.insert_uaddo(i32_all_ones, i32_one);
    let [_, sadd_overflow] = fb.insert_saddo(i32_max, i32_one);
    let [_, usub_overflow] = fb.insert_usubo(i32_zero, i32_one);
    let [_, ssub_overflow] = fb.insert_ssubo(i32_min, i32_one);
    let [_, umul_overflow] = fb.insert_umulo(i32_all_ones, i32_two);
    let [_, smul_overflow] = fb.insert_smulo(i32_max, i32_two);
    let [_, neg_overflow] = fb.insert_snego(i32_min);

    let i32_neg_six = fb.make_imm_value(-6i32);
    let i32_ten = fb.make_imm_value(10i32);
    let i32_five = fb.make_imm_value(5i32);
    let i32_65536 = fb.make_imm_value(65_536i32);

    let uaddsat = fb.insert_uaddsat(i32_neg_six, i32_ten);
    let saddsat = fb.insert_saddsat(i32_max, i32_one);
    let usubsat = fb.insert_usubsat(i32_five, i32_ten);
    let ssubsat = fb.insert_ssubsat(i32_min, i32_one);
    let umulsat = fb.insert_umulsat(i32_65536, i32_65536);
    let smulsat_hi = fb.insert_smulsat(i32_max, i32_two);
    let smulsat_lo = fb.insert_smulsat(i32_min, i32_two);

    let mut acc = fb.insert_inst(arith::Add::new(is, urem, srem), Type::I64);
    for value in [
        uadd_overflow,
        sadd_overflow,
        usub_overflow,
        ssub_overflow,
        umul_overflow,
        smul_overflow,
        neg_overflow,
    ] {
        let extended = fb.insert_inst(cast::Zext::new(is, value, Type::I64), Type::I64);
        acc = fb.insert_inst(arith::Add::new(is, acc, extended), Type::I64);
    }
    for value in [uaddsat, usubsat, umulsat] {
        let extended = fb.insert_inst(cast::Zext::new(is, value, Type::I64), Type::I64);
        acc = fb.insert_inst(arith::Add::new(is, acc, extended), Type::I64);
    }
    for value in [saddsat, ssubsat, smulsat_hi, smulsat_lo] {
        let extended = fb.insert_inst(cast::Sext::new(is, value, Type::I64), Type::I64);
        acc = fb.insert_inst(arith::Add::new(is, acc, extended), Type::I64);
    }

    fb.insert_inst_no_result(control_flow::Return::new_single(is, acc));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn() -> i64 = unsafe {
        let ptr = artifact
            .get_func_ptr::<fn() -> i64>("integer_edge_ops")
            .unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(), 8_589_934_595);
}

#[test]
fn cranelift_through_generic_compile_pipeline() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };

    let sig = Signature::new_single("identity", Linkage::Public, &[Type::I64], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let a = fb.args()[0];
    fb.insert_inst_no_result(control_flow::Return::new_single(is, a));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let compile = Compile::new(module, backend).with_opt_level(OptLevel::O0);
    let artifact = compile.compile().expect("Compile<CraneliftBackend> failed");

    let f: fn(i64) -> i64 = unsafe {
        let ptr = artifact.get_func_ptr::<fn(i64) -> i64>("identity").unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(42), 42);
    assert_eq!(f(-1), -1);
}

#[test]
fn cranelift_emits_native_object_for_main() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = {
        let ctx = ModuleCtx::new(&isa);
        ModuleBuilder::new(ctx)
    };

    let sig = Signature::new_single("main", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let status = fb.make_imm_value(42i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, status));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = CraneliftBackend::new()
        .compile_module_to_object(&module)
        .expect("object compilation failed");

    assert!(!artifact.bytes.is_empty());
    assert!(artifact.func_map.contains_key("main"));
}

#[test]
fn cranelift_emits_native_object_with_external_import_call() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    let putchar_sig = Signature::new_single("putchar", Linkage::External, &[Type::I32], Type::I32);
    let putchar = mb.declare_function(putchar_sig).unwrap();
    let main_sig = Signature::new_single("main", Linkage::Public, &[], Type::I32);
    let main = mb.declare_function(main_sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(main);
    let entry = fb.append_block();
    fb.switch_to_block(entry);
    let ch = fb.make_imm_value(65i32);
    let _ = fb.insert_inst(
        control_flow::Call::new(is, putchar, smallvec::smallvec![ch]),
        Type::I32,
    );
    let status = fb.make_imm_value(0i32);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, status));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = CraneliftBackend::new()
        .compile_module_to_object(&module)
        .expect("object compilation failed");

    assert!(!artifact.bytes.is_empty());
    assert!(artifact.func_map.contains_key("main"));
    assert!(artifact.func_map.contains_key("putchar"));
}

#[test]
fn cranelift_insert_value_builds_byte_array() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();
    let arr_ty = mb.declare_array_type(Type::I8, 4);

    let sig = Signature::new_single("byte_array_sum", Linkage::Public, &[], Type::I32);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let mut arr = fb.make_undef_value(arr_ty);
    for (idx, byte) in [80i8, 85, 83, 72].into_iter().enumerate() {
        let idx = fb.make_imm_value(idx as i64);
        let byte = fb.make_imm_value(byte);
        arr = fb.insert_inst(data::InsertValue::new(is, arr, idx, byte), arr_ty);
    }

    let mut sum = fb.make_imm_value(0i32);
    for idx in 0..4 {
        let idx = fb.make_imm_value(idx as i64);
        let byte = fb.insert_inst(data::ExtractValue::new(is, arr, idx), Type::I8);
        let widened = fb.insert_inst(cast::Zext::new(is, byte, Type::I32), Type::I32);
        sum = fb.insert_inst(arith::Add::new(is, sum, widened), Type::I32);
    }

    fb.insert_inst_no_result(control_flow::Return::new_single(is, sum));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = CraneliftBackend::new()
        .compile_module_to_object(&module)
        .expect("object compilation failed");

    assert!(!artifact.bytes.is_empty());
    assert!(artifact.func_map.contains_key("byte_array_sum"));
}

#[test]
fn cranelift_aggregate_offsets_cover_aligned_struct_fields() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();
    let arr_ty = mb.declare_array_type(Type::I8, 16);
    let struct_ty = mb.declare_struct_type("Input", &[arr_ty, Type::I256], false);
    let struct_ref_ty = mb.objref_type(struct_ty);
    let arr_ref_ty = mb.objref_type(arr_ty);
    let byte_ref_ty = mb.objref_type(Type::I8);
    let i256_ref_ty = mb.objref_type(Type::I256);

    let sig = Signature::new_single("aligned_struct_fields", Linkage::Public, &[], Type::I8);
    let func_ref = mb.declare_function(sig).unwrap();

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let object = fb.insert_inst(data::ObjAlloc::new(is, struct_ty), struct_ref_ty);
    let idx0 = fb.make_imm_value(0i64);
    let idx1 = fb.make_imm_value(1i64);
    let bytes = fb.insert_inst(
        data::ObjProj::new(is, smallvec::smallvec![object, idx0]),
        arr_ref_ty,
    );
    let first_byte = fb.insert_inst(data::ObjIndex::new(is, bytes, idx0), byte_ref_ty);
    let byte = fb.make_imm_value(42i8);
    fb.insert_inst_no_result(data::ObjStore::new(is, first_byte, byte));

    let len = fb.insert_inst(
        data::ObjProj::new(is, smallvec::smallvec![object, idx1]),
        i256_ref_ty,
    );
    let len_value = fb.make_imm_value(Immediate::from_i256(I256::from(3u8), Type::I256));
    fb.insert_inst_no_result(data::ObjStore::new(is, len, len_value));

    let loaded = fb.insert_inst(data::ObjLoad::new(is, first_byte), Type::I8);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, loaded));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let artifact = CraneliftBackend::new()
        .compile_module_to_object(&module)
        .expect("object compilation failed");

    assert!(!artifact.bytes.is_empty());
    assert!(artifact.func_map.contains_key("aligned_struct_fields"));
}

#[test]
fn cranelift_sret_uses_full_aggregate_size() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();
    let arr_ty = mb.declare_array_type(Type::I8, 16);
    let struct_ty = mb.declare_struct_type("InputReturn", &[arr_ty, Type::I256], false);

    let make_sig = Signature::new_single("make_input", Linkage::Private, &[], struct_ty);
    let make_ref = mb.declare_function(make_sig).unwrap();
    let caller_sig = Signature::new_single("read_len", Linkage::Public, &[], Type::I32);
    let caller_ref = mb.declare_function(caller_sig).unwrap();

    {
        let mut fb = mb.func_builder::<InstInserter>(make_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);

        let idx0 = fb.make_imm_value(0i64);
        let first = fb.make_imm_value(1i8);
        let bytes = fb.make_undef_value(arr_ty);
        let bytes = fb.insert_inst(data::InsertValue::new(is, bytes, idx0, first), arr_ty);
        let len = fb.make_imm_value(Immediate::from_i256(I256::from(3u8), Type::I256));
        let idx1 = fb.make_imm_value(1i64);
        let mut value = fb.make_undef_value(struct_ty);
        value = fb.insert_inst(data::InsertValue::new(is, value, idx0, bytes), struct_ty);
        value = fb.insert_inst(data::InsertValue::new(is, value, idx1, len), struct_ty);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, value));
        fb.seal_all();
        fb.finish();
    }

    {
        let mut fb = mb.func_builder::<InstInserter>(caller_ref);
        let entry = fb.append_block();
        fb.switch_to_block(entry);

        let value = fb.insert_inst(
            control_flow::Call::new(is, make_ref, smallvec::smallvec![]),
            struct_ty,
        );
        let idx1 = fb.make_imm_value(1i64);
        let len = fb.insert_inst(data::ExtractValue::new(is, value, idx1), Type::I256);
        let len = fb.insert_inst(cast::Trunc::new(is, len, Type::I32), Type::I32);
        fb.insert_inst_no_result(control_flow::Return::new_single(is, len));
        fb.seal_all();
        fb.finish();
    }

    let module = mb.build();
    let artifact = CraneliftBackend::new()
        .compile_module_to_object(&module)
        .expect("object compilation failed");

    assert!(!artifact.bytes.is_empty());
    assert!(artifact.func_map.contains_key("read_len"));
}

#[test]
fn cranelift_array_index() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    // Build: fn get_elem(idx: i64) -> i64 {
    //   let arr: [i64; 3] = [10, 20, 30]
    //   return arr[idx]
    // }
    let sig = Signature::new_single("get_elem", Linkage::Public, &[Type::I64], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let arr_ty = mb.declare_array_type(Type::I64, 3);
    let arr_objref_ty = mb.objref_type(arr_ty);
    let elem_objref_ty = mb.objref_type(Type::I64);

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let idx = fb.args()[0];

    // obj.alloc [i64; 3]
    let arr = fb.insert_inst(data::ObjAlloc::new(is, arr_ty), arr_objref_ty);

    // Store elements: arr[0]=10, arr[1]=20, arr[2]=30
    let imm0 = fb.make_imm_value(0i64);
    let imm1 = fb.make_imm_value(1i64);
    let imm2 = fb.make_imm_value(2i64);

    let val10 = fb.make_imm_value(10i64);
    let val20 = fb.make_imm_value(20i64);
    let val30 = fb.make_imm_value(30i64);

    let p0 = fb.insert_inst(data::ObjIndex::new(is, arr, imm0), elem_objref_ty);
    fb.insert_inst_no_result(data::ObjStore::new(is, p0, val10));

    let p1 = fb.insert_inst(data::ObjIndex::new(is, arr, imm1), elem_objref_ty);
    fb.insert_inst_no_result(data::ObjStore::new(is, p1, val20));

    let p2 = fb.insert_inst(data::ObjIndex::new(is, arr, imm2), elem_objref_ty);
    fb.insert_inst_no_result(data::ObjStore::new(is, p2, val30));

    // Dynamic index: val = arr[idx]
    let pi = fb.insert_inst(data::ObjIndex::new(is, arr, idx), elem_objref_ty);
    let val = fb.insert_inst(data::ObjLoad::new(is, pi), Type::I64);

    fb.insert_inst_no_result(control_flow::Return::new_single(is, val));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn(i64) -> i64 = unsafe {
        let ptr = artifact.get_func_ptr::<fn(i64) -> i64>("get_elem").unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(0), 10);
    assert_eq!(f(1), 20);
    assert_eq!(f(2), 30);
}

#[test]
fn cranelift_array_sum() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    // Build: fn sum_arr() -> i64 {
    //   let arr: [i64; 4] = [100, 200, 300, 400]
    //   return arr[0] + arr[1] + arr[2] + arr[3]
    // }
    let sig = Signature::new_single("sum_arr", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let arr_ty = mb.declare_array_type(Type::I64, 4);
    let arr_objref_ty = mb.objref_type(arr_ty);
    let elem_objref_ty = mb.objref_type(Type::I64);

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let arr = fb.insert_inst(data::ObjAlloc::new(is, arr_ty), arr_objref_ty);

    // Store 4 elements
    for (i, val) in [100i64, 200, 300, 400].iter().enumerate() {
        let idx = fb.make_imm_value(i as i64);
        let imm_val = fb.make_imm_value(*val);
        let p = fb.insert_inst(data::ObjIndex::new(is, arr, idx), elem_objref_ty);
        fb.insert_inst_no_result(data::ObjStore::new(is, p, imm_val));
    }

    // Load and sum
    let mut acc = {
        let idx = fb.make_imm_value(0i64);
        let p = fb.insert_inst(data::ObjIndex::new(is, arr, idx), elem_objref_ty);
        fb.insert_inst(data::ObjLoad::new(is, p), Type::I64)
    };
    for i in 1..4 {
        let idx = fb.make_imm_value(i as i64);
        let p = fb.insert_inst(data::ObjIndex::new(is, arr, idx), elem_objref_ty);
        let elem = fb.insert_inst(data::ObjLoad::new(is, p), Type::I64);
        acc = fb.insert_inst(arith::Add::new(is, acc, elem), Type::I64);
    }

    fb.insert_inst_no_result(control_flow::Return::new_single(is, acc));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn() -> i64 = unsafe {
        let ptr = artifact.get_func_ptr::<fn() -> i64>("sum_arr").unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(), 1000);
}

#[test]
fn cranelift_const_ref_array() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    // Declare a global constant array: [i64; 4] = [10, 20, 30, 40]
    let arr_ty = mb.declare_array_type(Type::I64, 4);
    let gv = mb.declare_gv(GlobalVariableData::constant(
        "ROUND_CONSTANTS".to_string(),
        arr_ty,
        Linkage::Private,
        GvInitializer::Array(vec![
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(10)),
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(20)),
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(30)),
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(40)),
        ]),
    ));

    // fn sum_consts(idx: i64) -> i64 { ROUND_CONSTANTS[idx] }
    let sig = Signature::new_single("get_const", Linkage::Public, &[Type::I64], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let constref_ty = mb.make_compound(sonatina_ir::types::CompoundType::ConstRef(arr_ty));
    let constref_type = Type::Compound(constref_ty);
    let elem_constref_ty = mb.make_compound(sonatina_ir::types::CompoundType::ConstRef(Type::I64));
    let elem_constref_type = Type::Compound(elem_constref_ty);

    let mut fb = mb.func_builder::<InstInserter>(func_ref);
    let entry = fb.append_block();
    fb.switch_to_block(entry);

    let idx = fb.args()[0];

    // const.ref → pointer to global array
    let arr_ref = fb.insert_inst(data::ConstRef::new(is, gv.into()), constref_type);
    // const.index → pointer to element
    let elem_ref = fb.insert_inst(data::ConstIndex::new(is, arr_ref, idx), elem_constref_type);
    // const.load → load the element value
    let val = fb.insert_inst(data::ConstLoad::new(is, elem_ref), Type::I64);

    fb.insert_inst_no_result(control_flow::Return::new_single(is, val));
    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn(i64) -> i64 = unsafe {
        let ptr = artifact
            .get_func_ptr::<fn(i64) -> i64>("get_const")
            .unwrap();
        std::mem::transmute(ptr)
    };

    assert_eq!(f(0), 10);
    assert_eq!(f(1), 20);
    assert_eq!(f(2), 30);
    assert_eq!(f(3), 40);
}

#[test]
fn cranelift_poseidon_loop_with_const_array() {
    let isa = native_isa();
    let is = isa.inst_set();
    let mb = native_module_builder();

    // Global constant round array: [i64; 4] = [3, 5, 7, 11]
    let arr_ty = mb.declare_array_type(Type::I64, 4);
    let gv = mb.declare_gv(GlobalVariableData::constant(
        "ROUND_CONSTS".to_string(),
        arr_ty,
        Linkage::Private,
        GvInitializer::Array(vec![
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(3)),
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(5)),
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(7)),
            GvInitializer::Immediate(sonatina_ir::Immediate::I64(11)),
        ]),
    ));

    // fn poseidon_sum() -> i64 {
    //   let C = ROUND_CONSTS;  // const array
    //   let mut acc: i64 = 1;
    //   for i in 0..4 {
    //     let c = C[i];
    //     acc = (acc + c) * (acc + c) + (acc + c);  // sigma(acc + c)
    //   }
    //   return acc;
    // }
    let sig = Signature::new_single("poseidon_sum", Linkage::Public, &[], Type::I64);
    let func_ref = mb.declare_function(sig).unwrap();

    let constref_ty = mb.make_compound(sonatina_ir::types::CompoundType::ConstRef(arr_ty));
    let constref_type = Type::Compound(constref_ty);
    let elem_constref_ty = mb.make_compound(sonatina_ir::types::CompoundType::ConstRef(Type::I64));
    let elem_constref_type = Type::Compound(elem_constref_ty);

    let mut fb = mb.func_builder::<InstInserter>(func_ref);

    let entry = fb.append_block();
    let loop_header = fb.append_block();
    let loop_body = fb.append_block();
    let exit = fb.append_block();

    // entry: jump to loop with (acc=1, i=0)
    fb.switch_to_block(entry);
    let init_acc = fb.make_imm_value(1i64);
    let init_i = fb.make_imm_value(0i64);
    fb.insert_inst_no_result(control_flow::Jump::new(is, loop_header));

    // loop_header: phi(acc, i), check i < 4
    fb.switch_to_block(loop_header);
    let acc_phi = fb.insert_inst(
        control_flow::Phi::new(is, vec![(init_acc, entry)]),
        Type::I64,
    );
    let i_phi = fb.insert_inst(control_flow::Phi::new(is, vec![(init_i, entry)]), Type::I64);
    let four = fb.make_imm_value(4i64);
    let cond = fb.insert_inst(cmp::Lt::new(is, i_phi, four), Type::I1);
    fb.insert_inst_no_result(control_flow::Br::new(is, cond, loop_body, exit));

    // loop_body: c = C[i], sigma(acc + c), i++
    fb.switch_to_block(loop_body);
    let arr_ref = fb.insert_inst(data::ConstRef::new(is, gv.into()), constref_type);
    let elem_ref = fb.insert_inst(
        data::ConstIndex::new(is, arr_ref, i_phi),
        elem_constref_type,
    );
    let c = fb.insert_inst(data::ConstLoad::new(is, elem_ref), Type::I64);
    let sum = fb.insert_inst(arith::Add::new(is, acc_phi, c), Type::I64);
    let sq = fb.insert_inst(arith::Mul::new(is, sum, sum), Type::I64);
    let new_acc = fb.insert_inst(arith::Add::new(is, sq, sum), Type::I64);
    let one = fb.make_imm_value(1i64);
    let new_i = fb.insert_inst(arith::Add::new(is, i_phi, one), Type::I64);

    // Update phis and jump back
    fb.append_phi_arg(acc_phi, new_acc, loop_body);
    fb.append_phi_arg(i_phi, new_i, loop_body);
    fb.insert_inst_no_result(control_flow::Jump::new(is, loop_header));

    // exit: return acc
    fb.switch_to_block(exit);
    fb.insert_inst_no_result(control_flow::Return::new_single(is, acc_phi));

    fb.seal_all();
    fb.finish();

    let module = mb.build();
    let backend = CraneliftBackend::new();
    let artifact = backend.compile_module(&module).expect("compilation failed");

    let f: fn() -> i64 = unsafe {
        let ptr = artifact
            .get_func_ptr::<fn() -> i64>("poseidon_sum")
            .unwrap();
        std::mem::transmute(ptr)
    };

    // Manual computation:
    // i=0: acc=1, c=3, sum=4, sq=16, new_acc=20
    // i=1: acc=20, c=5, sum=25, sq=625, new_acc=650
    // i=2: acc=650, c=7, sum=657, sq=431649, new_acc=432306
    // i=3: acc=432306, c=11, sum=432317, sq=186897988489, new_acc=186898420806
    let result = f();
    assert_eq!(result, 186898420806, "poseidon_sum with const array rounds");
}
