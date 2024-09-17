#![feature(iterator_try_reduce)]
#![feature(min_specialization)]

pub mod bytecode;
pub mod compile;
pub mod lex;
pub mod ops;
pub mod parse;
pub mod prim;
pub mod util;
pub mod val;
pub mod vm;

pub use crate::compile::Compiler;
pub use crate::vm::Mem;
