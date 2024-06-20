use super::syntax::Node;
use crate::syntax::{FromSyntax, Parser, Rule};
use annotate_snippets::{Level, Renderer, Snippet};
use either::Either;
use hex::FromHex;
pub use ir::{
    insn::{BinaryOp, CastOp, UnaryOp},
    DataLocationKind, Immediate, Linkage,
};
use ir::{I256, U256};
use pest::Parser as _;
use smol_str::SmolStr;
pub use sonatina_triple::{InvalidTriple, TargetTriple};
use std::{io, ops::Range, str::FromStr};

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Error {
    NumberOutOfBounds(Range<usize>),
    InvalidTarget(InvalidTriple, Range<usize>),
    SyntaxError(pest::error::Error<Rule>),
}

pub fn parse(input: &str) -> Result<Module, Vec<Error>> {
    pest::set_error_detail(true); // xxx

    match Parser::parse(Rule::module, input) {
        Err(err) => Err(vec![Error::SyntaxError(err)]),
        Ok(mut pairs) => {
            let pair = pairs.next().unwrap();
            debug_assert_eq!(pair.as_rule(), Rule::module);
            let mut node = Node::new(pair);

            let module = Module::from_syntax(&mut node);

            if node.errors.is_empty() {
                Ok(module)
            } else {
                Err(node.errors)
            }
        }
    }
}

#[derive(Debug)]
pub struct Module {
    pub target: Option<TargetTriple>,
    pub declared_functions: Vec<FuncDeclaration>,
    pub functions: Vec<Func>,
    pub comments: Vec<String>,
}

impl FromSyntax<Error> for Module {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        let target = match node
            .get_opt(Rule::target_triple)
            .map(|p| TargetTriple::parse(p.as_str()))
        {
            Some(Ok(t)) => Some(t),
            Some(Err(e)) => {
                node.error(Error::InvalidTarget(e, node.span.clone()));
                None
            }
            None => None,
        };

        let module_comments = node.map_while(|p| {
            if p.as_rule() == Rule::COMMENT && p.as_str().starts_with("#!") {
                Either::Right(p.as_str().into())
            } else {
                Either::Left(p)
            }
        });

        let mut declared_functions = vec![];
        let mut functions = vec![];
        loop {
            let comments = node.map_while(|p| {
                if p.as_rule() == Rule::COMMENT {
                    Either::Right(p.as_str().to_string())
                } else {
                    Either::Left(p)
                }
            });

            if let Some(func) = node.single_opt(Rule::function_declaration) {
                declared_functions.push(func);
            } else {
                match node.single_opt::<Func>(Rule::function) {
                    Some(mut func) => {
                        func.comments = comments;
                        functions.push(func);
                    }
                    None => break,
                }
            }
        }
        Module {
            target,
            declared_functions,
            functions,
            comments: module_comments,
        }
    }
}

#[derive(Debug)]
pub struct Func {
    pub signature: FuncSignature,
    pub blocks: Vec<Block>,
    pub comments: Vec<String>,
}

impl FromSyntax<Error> for Func {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        Func {
            signature: node.single(Rule::function_signature),
            blocks: node.multi(Rule::block),
            comments: vec![],
        }
    }
}

#[derive(Debug)]
pub struct FuncSignature {
    pub linkage: Linkage,
    pub name: FunctionName,
    pub params: Vec<ValueDeclaration>,
    pub ret_type: Option<Type>,
}

impl FromSyntax<Error> for FuncSignature {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        let linkage = node
            .parse_str_opt(Rule::function_linkage)
            .unwrap_or(Linkage::Private);

        FuncSignature {
            linkage,
            name: node.single(Rule::function_identifier),
            params: node.descend_into(Rule::function_params, |n| n.multi(Rule::value_declaration)),
            ret_type: node.descend_into_opt(Rule::function_ret_type, |n| n.single(Rule::type_name)),
        }
    }
}

#[derive(Debug)]
pub struct FuncDeclaration {
    pub linkage: Linkage,
    pub name: FunctionName,
    pub params: Vec<Type>,
    pub ret_type: Option<Type>,
}

impl FromSyntax<Error> for FuncDeclaration {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        let linkage = node
            .parse_str_opt(Rule::function_linkage)
            .unwrap_or(Linkage::Private);

        FuncDeclaration {
            linkage,
            name: node.single(Rule::function_identifier),
            params: node.descend_into(Rule::function_param_type_list, |n| n.multi(Rule::type_name)),
            ret_type: node.descend_into_opt(Rule::function_ret_type, |n| n.single(Rule::type_name)),
        }
    }
}

#[derive(Debug)]
pub struct Block {
    pub id: BlockId,
    pub stmts: Vec<Stmt>,
}

impl FromSyntax<Error> for Block {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        Self {
            id: node.single(Rule::block_ident),
            stmts: node.multi(Rule::stmt),
        }
    }
}

#[derive(Debug)]
pub struct BlockId(pub Option<u32>);

impl FromSyntax<Error> for BlockId {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        node.descend();
        debug_assert_eq!(node.rule, Rule::block_number);
        BlockId(node.txt.parse().ok())
    }
}

#[derive(Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    // pub comments: Vec<SmolStr>,
}

impl FromSyntax<Error> for Stmt {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        node.descend();
        let kind = match node.rule {
            Rule::define_stmt => StmtKind::Define(
                node.single(Rule::value_declaration),
                node.single(Rule::expr),
            ),
            Rule::store_stmt => StmtKind::Store(
                node.parse_str(Rule::location),
                node.single(Rule::value),
                node.single(Rule::value),
            ),
            Rule::return_stmt => StmtKind::Return(node.single_opt(Rule::value)),
            Rule::jump_stmt => StmtKind::Jump(node.single(Rule::block_ident)),
            Rule::br_stmt => StmtKind::Branch(
                node.single(Rule::value),
                node.single(Rule::block_ident),
                node.single(Rule::block_ident),
            ),
            Rule::br_table_stmt => StmtKind::BranchTable(
                node.single(Rule::value),
                node.single_opt(Rule::block_ident),
                node.multi(Rule::br_table_case),
            ),
            _ => unreachable!(),
        };
        Stmt { kind }
    }
}

#[derive(Debug)]
pub enum StmtKind {
    Define(ValueDeclaration, Expr),
    Store(DataLocationKind, Value, Value),
    Return(Option<Value>),
    Jump(BlockId),
    Branch(Value, BlockId, BlockId),
    BranchTable(Value, Option<BlockId>, Vec<(Value, BlockId)>),
    Call(Call),
}

impl FromSyntax<Error> for (Value, BlockId) {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        (node.single(Rule::value), node.single(Rule::block_ident))
    }
}

#[derive(Debug)]
pub enum Type {
    Int(IntType),
    Ptr(Box<Type>),
    Array(Box<Type>, usize),
    Void,
    Error,
}

impl FromSyntax<Error> for Type {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        node.descend();
        match node.rule {
            Rule::primitive_type => Type::Int(IntType::from_str(node.txt).unwrap()),
            Rule::ptr_type => Type::Ptr(Box::new(node.single(Rule::type_name))),
            Rule::array_type => {
                let Ok(size) = usize::from_str(node.get(Rule::array_size).as_str()) else {
                    node.error(Error::NumberOutOfBounds(node.span.clone()));
                    return Type::Error;
                };
                Type::Array(Box::new(node.single(Rule::type_name)), size)
            }
            Rule::void_type => Type::Void,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum IntType {
    I1,
    I8,
    I16,
    I32,
    I64,
    I128,
    I256,
}

impl From<IntType> for ir::Type {
    fn from(value: IntType) -> Self {
        match value {
            IntType::I1 => ir::Type::I1,
            IntType::I8 => ir::Type::I8,
            IntType::I16 => ir::Type::I16,
            IntType::I32 => ir::Type::I32,
            IntType::I64 => ir::Type::I64,
            IntType::I128 => ir::Type::I128,
            IntType::I256 => ir::Type::I256,
        }
    }
}

#[derive(Debug)]
pub enum Expr {
    Binary(BinaryOp, Value, Value),
    Unary(UnaryOp, Value),
    Cast(CastOp, Value),
    Load(DataLocationKind, Value),
    Alloca(Type),
    Call(Call),
    Gep(Vec<Value>),
    Phi(Vec<(Value, BlockId)>),
}

impl FromSyntax<Error> for Expr {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        node.descend();
        match node.rule {
            Rule::bin_expr => Expr::Binary(
                node.parse_str(Rule::bin_op),
                node.single(Rule::value),
                node.single(Rule::value),
            ),
            Rule::una_expr => Expr::Unary(node.parse_str(Rule::una_op), node.single(Rule::value)),
            Rule::alloca_expr => Expr::Alloca(node.single(Rule::type_name)),
            Rule::call_expr => Expr::Call(Call(
                node.single(Rule::function_identifier),
                node.multi(Rule::value),
            )),
            Rule::cast_expr => Expr::Cast(node.parse_str(Rule::cast_op), node.single(Rule::value)),

            Rule::gep_expr => Expr::Gep(node.multi(Rule::value)),
            Rule::load_expr => Expr::Load(node.parse_str(Rule::location), node.single(Rule::value)),
            Rule::phi_expr => Expr::Phi(node.multi(Rule::phi_value)),
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct Call(pub FunctionName, pub Vec<Value>);

/// Doesn't include `%` prefix.
#[derive(Debug)]
pub struct FunctionName(pub SmolStr);

impl FromSyntax<Error> for FunctionName {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        FunctionName(node.parse_str(Rule::function_name))
    }
}

#[derive(Debug)]
pub struct ValueName(pub SmolStr);

impl FromSyntax<Error> for ValueName {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        Self(node.txt.into())
    }
}

#[derive(Debug)]
pub struct ValueDeclaration(pub ValueName, pub Type);

impl FromSyntax<Error> for ValueDeclaration {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        ValueDeclaration(node.single(Rule::value_name), node.single(Rule::type_name))
    }
}

#[derive(Debug)]
pub enum Value {
    Immediate(Immediate),
    Named(ValueName),
    Error,
}

impl FromSyntax<Error> for Value {
    fn from_syntax(node: &mut Node<Error>) -> Self {
        node.descend();
        match node.rule {
            Rule::value_name => Value::Named(ValueName(node.txt.into())),
            Rule::imm_number => {
                let ty: IntType = node.parse_str(Rule::primitive_type);
                node.descend();
                let mut txt = node.txt;
                match node.rule {
                    Rule::decimal => match ty {
                        IntType::I1 => imm_or_err(node, || {
                            let b = match u8::from_str(txt).ok()? {
                                0 => false,
                                1 => true,
                                _ => return None,
                            };
                            Some(Immediate::I1(b))
                        }),
                        IntType::I8 => imm_or_err(node, || Some(Immediate::I8(txt.parse().ok()?))),
                        IntType::I16 => {
                            imm_or_err(node, || Some(Immediate::I16(txt.parse().ok()?)))
                        }
                        IntType::I32 => {
                            imm_or_err(node, || Some(Immediate::I32(txt.parse().ok()?)))
                        }
                        IntType::I64 => {
                            imm_or_err(node, || Some(Immediate::I64(txt.parse().ok()?)))
                        }
                        IntType::I128 => {
                            imm_or_err(node, || Some(Immediate::I128(txt.parse().ok()?)))
                        }
                        IntType::I256 => {
                            let s = txt.strip_prefix('-');
                            let is_negative = s.is_some();
                            txt = s.unwrap_or(txt);

                            imm_or_err(node, || {
                                let mut i256 = U256::from_dec_str(txt).ok()?.into();
                                if is_negative {
                                    i256 = I256::zero().overflowing_sub(i256).0;
                                }
                                Some(Immediate::I256(i256))
                            })
                        }
                    },

                    Rule::hex => match ty {
                        IntType::I1 => {
                            node.error(Error::NumberOutOfBounds(node.span.clone()));
                            Value::Error
                        }
                        IntType::I8 => imm_or_err(node, || {
                            Some(Immediate::I8(i8::from_be_bytes(hex_bytes(txt)?)))
                        }),
                        IntType::I16 => imm_or_err(node, || {
                            Some(Immediate::I16(i16::from_be_bytes(hex_bytes(txt)?)))
                        }),
                        IntType::I32 => imm_or_err(node, || {
                            Some(Immediate::I32(i32::from_be_bytes(hex_bytes(txt)?)))
                        }),
                        IntType::I64 => imm_or_err(node, || {
                            Some(Immediate::I64(i64::from_be_bytes(hex_bytes(txt)?)))
                        }),
                        IntType::I128 => imm_or_err(node, || {
                            Some(Immediate::I128(i128::from_be_bytes(hex_bytes(txt)?)))
                        }),
                        IntType::I256 => {
                            let s = txt.strip_prefix('-');
                            let is_negative = s.is_some();
                            txt = s.unwrap_or(txt);

                            imm_or_err(node, || {
                                let mut i256 = U256::from_big_endian(&hex_bytes::<32>(txt)?).into();
                                if is_negative {
                                    i256 = I256::zero().overflowing_sub(i256).0;
                                }
                                Some(Immediate::I256(i256))
                            })
                        }
                    },
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
    }
}

impl FromStr for IntType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "i1" => Ok(Self::I1),
            "i8" => Ok(Self::I8),
            "i16" => Ok(Self::I16),
            "i32" => Ok(Self::I32),
            "i64" => Ok(Self::I64),
            "i128" => Ok(Self::I128),
            "i256" => Ok(Self::I256),
            _ => Err(()),
        }
    }
}

impl Error {
    pub fn span(&self) -> Range<usize> {
        match self {
            Error::NumberOutOfBounds(span) => span.clone(),
            Error::InvalidTarget(_, span) => span.clone(),
            Error::SyntaxError(err) => match err.location {
                pest::error::InputLocation::Pos(p) => p..p,
                pest::error::InputLocation::Span((s, e)) => s..e,
            },
        }
    }

    pub fn print(&self, mut w: impl io::Write, path: &str, content: &str) -> io::Result<()> {
        let label = match self {
            Error::NumberOutOfBounds(_) => "number out of bounds".into(),
            Error::InvalidTarget(err, _) => err.to_string(),
            Error::SyntaxError(err) => err.to_string(),
        };
        let snippet = Level::Error.title("parse error").snippet(
            Snippet::source(content)
                .line_start(0)
                .origin(path)
                .fold(true)
                .annotation(Level::Error.span(self.span()).label(&label)),
        );
        let rend = Renderer::styled();
        let disp = rend.render(snippet);
        write!(w, "{}", disp)
    }

    pub fn print_to_string(&self, path: &str, content: &str) -> String {
        let mut v = vec![];
        self.print(&mut v, path, content).unwrap();
        String::from_utf8(v).unwrap()
    }
}

fn imm_or_err<F>(node: &mut Node<Error>, f: F) -> Value
where
    F: Fn() -> Option<Immediate>,
{
    let Some(imm) = f() else {
        let span = node.span.clone();
        node.error(Error::NumberOutOfBounds(span));
        return Value::Error;
    };
    Value::Immediate(imm)
}

fn hex_bytes<const N: usize>(mut s: &str) -> Option<[u8; N]> {
    s = s.strip_prefix("0x").unwrap();
    let bytes = Vec::<u8>::from_hex(s).unwrap();

    if bytes.len() > N {
        return None;
    }

    let mut out = [0; N];
    out[N - bytes.len()..].copy_from_slice(&bytes);
    Some(out)
}

// xxx remove
// pub fn parse_immediate<L, T>(
//     val: &str,
//     loc: Range<usize>,
// ) -> Result<Value, ParseError<L, T, Error>> {
//     let mut chunks = val.split('.');
//     let num = chunks.next().unwrap();
//     let t = chunks.next().unwrap();

//     let imm = match t {
//         "i1" => Immediate::I1(parse_num(num, loc)?),
//         "i8" => Immediate::I8(parse_num(num, loc)?),
//         "i16" => Immediate::I16(parse_num(num, loc)?),
//         "i32" => Immediate::I32(parse_num(num, loc)?),
//         "i64" => Immediate::I64(parse_num(num, loc)?),
//         "i128" => Immediate::I128(parse_num(num, loc)?),
//         "i256" => todo!(),
//         _ => {
//             unreachable!()
//         }
//     };
//     Ok(Value::Immediate(imm))
// }

// pub fn parse_num<T, E, Loc, Tok>(
//     s: &str,
//     loc: Range<usize>,
// ) -> Result<T, ParseError<Loc, Tok, Error>>
// where
//     T: FromStr<Err = E>,
// {
//     T::from_str(s).map_err(|_| ParseError::User {
//         error: Error::NumberOutOfBounds(loc),
//     })
// }