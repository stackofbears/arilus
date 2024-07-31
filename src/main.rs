#![feature(iterator_try_reduce)]
#![feature(min_specialization)]

// mutable references; boolean type; regex; output formatting; idiom recognition; train syntax; make dyad cases fail when called as monad (or pass ()?); error messages (incl. argument errors for commutative primitives, which may output argument info in the wrong order)
//
// FIX: [] shows as "", []#n is just []
//
// Possible primitive symbol changes:
//   ^ head/take, Pow/** for pow, $ last/drop, ^: prefixes/windows overlapping (adverb), $: suffixes/windows non-overlapping (adverb), x#y filter
//   =: for match, >: for gt, <: for lt, !: for not/not-equal (what about `!'? abs?), |: for or (short-circuit), &: for and (short-circuit) (how do these work?), /: for integer-divide
//
// Still need: cut, each-atom, each-list
//
// Think about whether arrays should always be indexed when called - right now
// x@y indexes x, while x I and x .i treat i as a constant function.
//
// possible match conjunction (also usable as monad/dyad) (after arg syntax)
// {(xpat [;ypat] ["&"["&"] exprs]} ) stuff}::more::more
//
// e.g.
// Add:{(x) x + 1}::{(x;y) x + y}
// MoveFirstToEnd:{([x;..rest]) rest,[x]}::{x}

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
mod ops;
mod parse;
mod prim;
mod util;
mod val;
mod vm;

use std::{
    io::{self, Write},
    fs,
    mem::swap,
};

use crate::util::{cold, err};

fn main() -> io::Result<()> {
    // Read command line
    //   No options? Start repl (one-line tokenize, parse -> if in delayed spot (fn def, branch, line continuation, nested in parens) then wait for more, else run)
    //   File given? Run file (whole-file tokenize, parse, run)
    //     Maybe includes other files, though! File <-> line
    // General form: (env, string) -> (mutated env, val result?)

    let args: Vec<String> = std::env::args().collect();
    if let Err(s) = match &args[..] {
        [_] => run_repl(),
        [_, file] => go(&fs::read_to_string(&file)?),
        _ => err!("Too many command-line options"),
    } {
        eprintln!("{s}");
    }

    Ok(())
}

struct ReplSession {
    line: String,
    tokens: Vec<lex::Token>,
    compiler: compile::Compiler,
    mem: vm::Mem,
}

impl ReplSession {
    fn new() -> Self {
        let mut mem = vm::Mem::new();
        let print_and_pop_result = vec![
            bytecode::Instr::CallPrimFunc1 { prim: bytecode::PrimFunc::Verb(lex::PrimVerb::DebugPrint) },
            bytecode::Instr::Pop,
        ];
        mem.code = print_and_pop_result;

        Self {
            line: String::new(),
            tokens: Vec::new(),
            compiler: compile::Compiler::new(lex::Lexer::new()),
            mem,
        }
    }

    fn run_line(&mut self) -> Result<(), String> {
        let token_start = self.tokens.len();
        let mut nesting: i32 = 0;
        // TODO multiline strings
        //
        // TODO do we want to run code if we enter nesting level 0 and then nest
        // again on the same line? e.g. in
        //  1  (
        //  2   code 1...
        //  3  ); (
        //  4   code 2...
        // we can run code 1 after the ) on line 3, but here we'll continue to
        // wait until a line exits on level 0.
        loop {
            // TODO support pasting a bunch of lines into the repl without
            // printing the continuation prompt multiple times (or at all, if
            // the lines close all their open parens).
            if nesting > 0 {
                print!("↪ ");
            } else {
                print!("  ");
            }

            if let Err(err) = io::stdout().flush() { cold(()); return Err(err.to_string()) }
            if let Err(err) = io::stdin().read_line(&mut self.line) { cold(()); return Err(err.to_string()) }
            let line_start = self.tokens.len();

            // TODO use line length to guess token count
            if let err@Err(_) = self.compiler.lexer.tokenize(&self.line, &mut self.tokens) {
                cold(());
                self.tokens.truncate(token_start);
                self.line.clear();
                return err
            }
            self.line.clear();

            nesting += count_nesting(&self.tokens[line_start..]);
            if nesting <= 0 { break }  // < 0 will raise a parse error
        }
        let exprs = parse::parse(&self.tokens[token_start..])?;
        if exprs.is_empty() { return Ok(()) }

        let code_start = self.compiler.code.len();
        self.compiler.compile(&exprs)?;

        let is_assignment = matches!(exprs.last(),
                                     Some(
                                         parse::Expr::Noun(parse::Noun::LowerAssign(_, _)) |
                                         parse::Expr::Verb(parse::Verb::UpperAssign(_, _)) |
                                         parse::Expr::Noun(parse::Noun::ModifyingAssign(_, _))
                                     ));
        if !is_assignment {
            self.compiler.code.push(bytecode::Instr::CallPrimFunc1 {
                prim: bytecode::PrimFunc::Verb(lex::PrimVerb::DebugPrint)
            });
        }
        self.compiler.code.push(bytecode::Instr::Pop);

        swap(&mut self.mem.code, &mut self.compiler.code);
        let result = self.mem.execute_from_toplevel(code_start);
        swap(&mut self.mem.code, &mut self.compiler.code);

        self.compiler.code.pop();
        if !is_assignment {
            self.compiler.code.pop();
        }

        result
    }
}

fn run_repl() -> Result<(), String> {
    let mut session = ReplSession::new();
    loop {
        if let Err(err) = session.run_line() {
            cold(());
            eprintln!("{}", err)
        }
    }
}

fn count_nesting(tokens: &[lex::Token]) -> i32 {
    use lex::Token::*;
    let mut level = 0;
    for token in tokens {
        match token {
            LParen | LBracket{..} | LBrace => level += 1,
            RParen | RBracket | RBrace => level -= 1,
            _ => (),
        }
    }
    level
}

fn go(text: &str) -> Result<(), String> {
    let code = compile::compile_string(text)?;
    let mut mem = vm::Mem::new();
    mem.code = code;
    mem.execute_from_toplevel(0)?;
    Ok(())
}
