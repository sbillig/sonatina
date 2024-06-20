use std::{ops::Range, str::FromStr};

use either::Either;
use pest::iterators::Pair;

#[derive(pest_derive::Parser)]
#[grammar = "sonatina.pest"]
pub struct Parser;

pub trait FromSyntax<E> {
    fn from_syntax(node: &mut Node<E>) -> Self;
}

pub struct Node<'i, E> {
    pub rule: Rule,
    pub txt: &'i str,
    pub span: Range<usize>,
    pairs: Vec<Option<Pair<'i, Rule>>>,
    pub errors: Vec<E>,
    child: Option<Box<Self>>,
}

impl<'i, E> Node<'i, E> {
    pub fn new(pair: Pair<'i, Rule>) -> Self {
        let mut n = Self::default();
        n.set_pair(pair);
        n
    }

    fn set_pair(&mut self, pair: Pair<'i, Rule>) {
        self.rule = pair.as_rule();
        self.txt = pair.as_str();
        let s = pair.as_span();
        self.span = s.start()..s.end();
        self.pairs.clear();
        self.pairs.extend(pair.into_inner().map(Some));
        debug_assert!(self.errors.is_empty());
    }

    fn reset<F>(&mut self, pair: Pair<'i, Rule>, with_errors: F)
    where
        F: FnMut(E),
    {
        self.clear(with_errors);
        self.set_pair(pair);
    }

    fn clear<F>(&mut self, with_errors: F)
    where
        F: FnMut(E),
    {
        self.errors.drain(..).for_each(with_errors);
        self.pairs.clear();
    }

    fn with_child<F, T>(&mut self, pair: Pair<'i, Rule>, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        let mut child = self.child.take().unwrap_or_default();
        child.set_pair(pair);
        let r = f(&mut child);

        child.clear(|err| self.errors.push(err));
        self.child = Some(child);
        r
    }

    pub fn error(&mut self, err: E) {
        self.errors.push(err);
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
            || (self.pairs.len() == 1 && self.pairs[0].as_ref().unwrap().as_rule() == Rule::EOI)
    }

    pub fn descend(&mut self) {
        debug_assert_eq!(self.pairs.len(), 1);
        let p = self.pairs.remove(0).unwrap();
        self.set_pair(p);
    }

    pub fn descend_into<F, T>(&mut self, rule: Rule, f: F) -> T
    where
        F: FnOnce(&mut Self) -> T,
    {
        self.descend_into_opt(rule, f).unwrap()
    }

    pub fn descend_into_opt<F, T>(&mut self, rule: Rule, f: F) -> Option<T>
    where
        F: FnOnce(&mut Self) -> T,
    {
        let p = self.get_opt(rule)?;
        Some(self.with_child(p, f))
    }

    pub fn single<T: FromSyntax<E>>(&mut self, rule: Rule) -> T {
        self.single_opt(rule).unwrap()
    }

    pub fn single_opt<T: FromSyntax<E>>(&mut self, rule: Rule) -> Option<T> {
        let p = self.get_opt(rule)?;
        Some(self.with_child(p, T::from_syntax))
    }

    pub fn multi<T: FromSyntax<E>>(&mut self, rule: Rule) -> Vec<T> {
        let mut child = self.child.take().unwrap_or_default();
        let mut errors = vec![];

        // `take` the pairs that match the `rule`, and convert them to T
        let r = self
            .pairs
            .iter_mut()
            .filter_map(|p| {
                if p.as_ref().unwrap().as_rule() == rule {
                    let p = p.take().unwrap();
                    child.reset(p, |err| errors.push(err));
                    Some(T::from_syntax(&mut child))
                } else {
                    None
                }
            })
            .collect();

        // remove the pairs that were taken
        self.pairs.retain(|p| p.is_some());

        self.errors.append(&mut errors);
        child.clear(|e| self.errors.push(e));
        self.child = Some(child);
        r
    }

    pub fn get(&mut self, rule: Rule) -> Pair<'i, Rule> {
        let r = self.get_opt(rule);
        debug_assert!(
            r.is_some(),
            "Failed to get {rule:?} inside {:?}, with pairs: {:?}",
            self.rule,
            self.pairs
        );
        r.unwrap()
    }

    pub fn get_opt(&mut self, rule: Rule) -> Option<Pair<'i, Rule>> {
        let pos = self
            .pairs
            .iter()
            .position(|p| p.as_ref().unwrap().as_rule() == rule)?;
        Some(self.pairs.remove(pos).unwrap())
    }

    pub fn parse_str<T>(&mut self, rule: Rule) -> T
    where
        T: FromStr,
        T::Err: std::fmt::Debug,
    {
        self.parse_str_opt(rule).unwrap()
    }

    pub fn parse_str_opt<T>(&mut self, rule: Rule) -> Option<T>
    where
        T: FromStr,
        T::Err: std::fmt::Debug,
    {
        self.get_opt(rule).map(|p| p.as_str().parse().unwrap())
    }

    pub fn map_while<T, F>(&mut self, mut f: F) -> Vec<T>
    where
        F: FnMut(Pair<'i, Rule>) -> Either<Pair<'i, Rule>, T>,
    {
        let mut out = vec![];
        for p in self.pairs.iter_mut() {
            match f(p.take().unwrap()) {
                Either::Left(pp) => {
                    *p = Some(pp);
                    break;
                }
                Either::Right(r) => {
                    out.push(r);
                }
            }
        }
        self.pairs.retain(|p| p.is_some());
        out
    }
}

impl<'i, E> std::default::Default for Node<'i, E> {
    fn default() -> Self {
        Self {
            rule: Rule::EOI,
            txt: Default::default(),
            span: Default::default(),
            pairs: vec![],
            errors: vec![],
            child: None,
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_with_module_comment() {
//         let input = "
//             #! Module comment 1
//             #! Module comment 2

//             target = \"evm-ethereum-london\"

//             # f1 start 1
//             # f1 start 2
//             func private %f1() -> i32 {
//                 block0:
//                     return 311.i32;
//             }

//             # f2 start 1
//             # f2 start 2
//             func public %f2() -> i32 {
//                 block0:
//                     return 311.i32;
//             }";

//         let parsed_module = parse_module2(input).unwrap();
//         let module_comments = parsed_module.module_comments;
//         assert_eq!(module_comments[0], "#! Module comment 1");
//         assert_eq!(module_comments[1], "#! Module comment 2");

//         let module = parsed_module.module;
//         let mut funcs = module.iter_functions();
//         let func1 = funcs.next().unwrap();
//         let func1_comment = &parsed_module.func_comments[func1];
//         assert_eq!(func1_comment.len(), 2);
//         assert_eq!(func1_comment[0], "# f1 start 1");
//         assert_eq!(func1_comment[1], "# f1 start 2");

//         let func2 = funcs.next().unwrap();
//         let func2_comment = &parsed_module.func_comments[func2];
//         assert_eq!(func2_comment.len(), 2);
//         assert_eq!(func2_comment[0], "# f2 start 1");
//         assert_eq!(func2_comment[1], "# f2 start 2");
//     }
// }