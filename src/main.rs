// next: unpacking assignment; refactor closures to avoid unsafe code; mutable references; boolean type; tail call elimination; branching; regex; output formatting; arg syntax; idiom recognition; train syntax
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

//mod explore;
mod bytecode;
mod compile;
mod lex;
mod parse;
mod util;
mod vm;

use std::{
    io::{self, Write},
    fs,
    mem::swap,
};

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
        _ => Err("Too many command-line options".to_string()),
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
            bytecode::Instr::CallPrimVerb1 { prim: lex::PrimVerb::DebugPrint },
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
            print!("  ");
            if let Err(err) = io::stdout().flush() { return Err(err.to_string()) }
            if let Err(err) = io::stdin().read_line(&mut self.line) { return Err(err.to_string()) }
            let line_start = self.tokens.len();
            // TODO use line length to guess token count
            lex::tokenize(&self.line, &mut self.tokens)?;
            self.line.clear();

            nesting += count_nesting(&self.tokens[line_start..]);
            if nesting <= 0 { break }  // < 0 will raise a parse error
        }
        let exprs = parse::parse(&self.tokens[token_start..])?;
        if exprs.is_empty() { return Ok(()) }

        let code_start = self.compiler.code.len();
        self.compiler.compile_block(&exprs)?;
        self.compiler.code.push(bytecode::Instr::CallPrimVerb1 { prim: lex::PrimVerb::DebugPrint });
        self.compiler.code.push(bytecode::Instr::Pop);

        swap(&mut self.mem.code, &mut self.compiler.code);
        let result = self.mem.execute(code_start);
        swap(&mut self.mem.code, &mut self.compiler.code);

        self.compiler.code.pop();
        self.compiler.code.pop();

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
            LParen | LBracket | LBrace => level += 1,
            RParen | RBracket | RBrace => level -= 1,
            _ => (),
        }
    }
    level
}

fn compile_string(text: &str) -> Result<Vec<bytecode::Instr>, String> {
    let tokens = lex::tokenize_to_vec(text)?;
    dbg!(&tokens);
    let exprs = parse::parse(&tokens)?;
    dbg!(&exprs);
    let code = compile::compile(&exprs)?;
    dbg!(&code);
    Ok(code)
}

fn go(text: &str) -> Result<(), String> {
    let code = compile_string(text)?;
    let mut mem = vm::Mem::new();
    mem.code = code;
    mem.execute(0)?;
    Ok(())
}
