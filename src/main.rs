// TODO
//   make refcounts atomic; read https://doc.rust-lang.org/nomicon/atomics.html, links at https://stackoverflow.com/questions/30407121/which-stdsyncatomicordering-to-use

// Grammar
// expr:
//  small (verb small?)+ | name ":" expr
// small:
//  lit | name | "(" block ")" | indexed
//  # | projection
// verb:
//  prim | Name | "{"("("args")")? block "}" | "(" verb ")" | adv any | any conj any
//  # | Name ":" verb
// any:
//   small | verb | (name | Name) ":" any
// block:
//  expr sepby ";"
// indexed:
//  # block instead?
//  small "[" expr "]"
// lit:
//  num | "{"("(" args ")")? block "}"

mod bytecode;
mod compile;
mod lex;
mod parse;
mod ptr_range;
mod vm;

use std::fs;

fn main() -> std::io::Result<()> {
    // Read command line
    //   No options? Start repl (one-line tokenize, parse -> if in delayed spot (fn def, branch, line continuation, nested in parens) then wait for more, else run)
    //   File given? Run file (whole-file tokenize, parse, run)
    //     Maybe includes other files, though! File <-> line
    // General form: (env, string) -> (mutated env, val result?)

    if let Err(s) = go(&fs::read_to_string("in.txt")?) {
        eprintln!("{s}");
    }
    Ok(())
}

fn go(text: &str) -> Result<(), String> {
    let tokens = lex::tokenize(text)?;
    dbg!(&tokens);
    let exprs = parse::parse(tokens)?;
    dbg!(&exprs);
    let code = compile::compile(&exprs)?;
    dbg!(&code);
    let mut mem = vm::Mem::new(512);
    let status = mem.execute(&code)?;  // TODO use exit status
    Ok(())
}
