use sonatina_parser::parse_module;
use sonatina_verifier::{VerificationLevel, VerifierConfig, verify_module};

#[test]
fn memory_guard_requires_a_u32_immediate_size_and_full_word_result() {
    for (size, ty, valid) in [
        ("0.i256", "i256", true),
        ("128.i256", "i256", true),
        ("129.i32", "i256", true),
        ("v0", "i256", false),
        ("-1.i256", "i256", false),
        ("4294967296.i256", "i256", false),
        ("128.i256", "i8", false),
        ("128.i256", "*i256", false),
    ] {
        let source = format!(
            "target = \"evm-ethereum-osaka\"\nfunc public %guard(v0.i256) -> {ty} {{\nblock0:\n v1.{ty} = evm_memory_guard {size};\n return v1;\n}}"
        );
        let parsed = parse_module(&source).unwrap();
        let report = verify_module(
            &parsed.module,
            &VerifierConfig::for_level(VerificationLevel::Full),
        );
        assert_eq!(report.is_ok(), valid, "{size}, {ty}: {report}");
    }
}
