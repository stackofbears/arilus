#![feature(iterator_try_reduce)]
#![feature(min_specialization)]

mod bytecode;
mod compile;
mod lex;
mod ops;
mod parse;
mod prim;
mod util;
mod val;
mod vm;

pub use crate::compile::compile_string;
pub use crate::vm::Mem;
