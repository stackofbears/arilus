#![feature(iterator_try_reduce)]
#![feature(min_specialization)]
#![feature(variant_count)]

pub mod bytecode;
pub mod compile;
pub mod lex;
pub mod ops;
pub mod parse;
pub mod prim;
pub mod util;
pub mod val;
pub mod vm;

use std::{fs, path::Path};
use crate::{compile::Compiler, util::Res, vm::Mem};

pub fn run_file<P: AsRef<Path> + std::fmt::Display>(path: P) -> Res<()> {
    let code = &fs::read_to_string(&path)
        .map_err(|e| format!("Error reading file {path}: {e}"))?;
    run_program(code)
}

pub fn run_program(text: &str) -> Res<()> {
    let code = compile::compile_string(text)?;
    let mut mem = vm::Mem::new();
    mem.code = code;
    mem.execute_from_toplevel(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_string(code: &str) -> Res<val::Val> {
        let mut compiler = Compiler::new();
        let code_start = compiler.code.len();

        compiler.compile_string(code)?;
        
        let mut mem = Mem::new();
        std::mem::swap(&mut mem.code, &mut compiler.code);
        mem.execute_from_toplevel(code_start)?;

        Ok(mem.stack.pop().unwrap())
    }

    #[test]
    fn run_tests() -> Res<()> {
        run_file("tests/tests.arl")
    }
}
