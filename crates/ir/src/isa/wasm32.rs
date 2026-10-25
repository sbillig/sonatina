use sonatina_triple::TargetTriple;

use super::{
    Isa, TypeLayout,
    native::{NativeAddressSpaces, NativeTypeLayout, inst_set},
};
use crate::{AddressSpaceInfo, inst::native::inst_set::NativeInstSet};

/// WebAssembly with 32-bit linear-memory addresses.
///
/// It accepts the target-neutral native instruction set. Memory is
/// little-endian with the native scalar and struct layout; pointers and
/// references take four bytes.
#[derive(Debug, Clone, Copy)]
pub struct Wasm32 {
    triple: TargetTriple,
}

impl Wasm32 {
    pub fn new(triple: TargetTriple) -> Self {
        assert_eq!(triple, TargetTriple::WASM32);
        Self { triple }
    }
}

impl Default for Wasm32 {
    fn default() -> Self {
        Self::new(TargetTriple::WASM32)
    }
}

impl Isa for Wasm32 {
    type InstSet = NativeInstSet;

    fn triple(&self) -> TargetTriple {
        self.triple
    }

    fn type_layout(&self) -> &'static dyn TypeLayout {
        const LAYOUT: NativeTypeLayout = NativeTypeLayout { pointer_size: 4 };
        &LAYOUT
    }

    fn address_spaces(&self) -> &'static dyn AddressSpaceInfo {
        static SPACES: NativeAddressSpaces = NativeAddressSpaces;
        &SPACES
    }

    fn inst_set(&self) -> &'static Self::InstSet {
        inst_set()
    }
}

#[cfg(test)]
mod tests {
    use super::Wasm32;
    use crate::{Module, Type, isa::Isa};

    #[test]
    fn wasm32_layout_uses_32_bit_pointers() {
        let isa = Wasm32::default();
        assert_eq!(isa.type_layout().pointer_repl(), Type::I32);
        let module = Module::new(&isa);
        let (pointer, structure) = module.ctx.with_ty_store_mut(|store| {
            let pointer = store.make_ptr(Type::I8);
            let structure = store.make_struct("Pair", &[Type::I8, pointer, Type::I64], false);
            (pointer, structure)
        });
        assert_eq!(module.ctx.size_of_unchecked(pointer), 4);
        assert_eq!(module.ctx.align_of_unchecked(pointer), 4);
        assert_eq!(module.ctx.size_of_unchecked(structure), 16);
        assert_eq!(module.ctx.size_of_unchecked(Type::I256), 32);
        assert_eq!(module.ctx.align_of_unchecked(Type::I256), 16);
    }
}
