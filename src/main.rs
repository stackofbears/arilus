extern crate arilus;

use std::{
    io::{self, Write},
    mem::swap,
};

use arilus::*;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    match &args[..] {
        [_] => run_repl(),
        [_, file] => run_file(&file),
        _ => err!("Too many command-line options"),
    }
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
            bytecode::Instr::CallPrimFunc1 { prim: bytecode::PrimFunc::DebugPrint },
            bytecode::Instr::Pop,
        ];
        mem.code = print_and_pop_result;

        Self {
            line: String::new(),
            tokens: Vec::new(),
            compiler: compile::Compiler::new(),
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

            io::stdout().flush().map_err(|e| e.to_string())?;
            io::stdin().read_line(&mut self.line).map_err(|e| e.to_string())?;

            let line_start = self.tokens.len();
            if let err@Err(_) = self.compiler.lexer.tokenize(&self.line, &mut self.tokens) {
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
                prim: bytecode::PrimFunc::DebugPrint
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
