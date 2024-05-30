use std::{
    collections::HashMap,
    fs,
};

use crate::bytecode::*;
use crate::lex::{self, *};
use crate::parse::*;
use crate::util::*;

#[derive(Clone, Copy)]
enum Arity { One, Two }

// TODO better error than string
pub fn compile(lexer: Lexer, exprs: &[Expr]) -> Result<Vec<Instr>, String> {
    let mut compiler = Compiler::new(lexer);
    compiler.compile_block(exprs)?;
    Ok(compiler.code)
}

// Invariants: `scopes` non-empty after `new`.
pub struct Compiler {
    pub code: Vec<Instr>,
    pub lexer: Lexer,

    scopes: Vec<HashMap<String, Var>>,
    
    // (code index right after LoadModule, module vars)
    module_cache: HashMap<String, (usize, HashMap<String, Var>)>,
}

// TODO be able to compile globals for repl
// TODO adverb nounification (') (for ultimate adverb verbification)
impl Compiler {
    pub fn new(lexer: Lexer) -> Self {
        let mut globals = HashMap::new();
        for (i, name) in get_stdlib_names().iter().enumerate() {
            globals.insert(name.to_string(), Var { place: Place::Local, slot: i });
        }

        Self { code: vec![],
               lexer,
               scopes: vec![globals],
               module_cache: HashMap::new() }
    }

    pub fn compile(&mut self, exprs: &[Expr]) -> Result<(), String> {
        let result = self.compile_block(exprs);
        self.scopes.truncate(1);  // Move back to global scope
        result
    }

    fn parse_file(&self, filepath: &str) -> Result<Vec<Expr>, String> {
        let file_contents = fs::read_to_string(filepath).map_err(|err| err.to_string())?;
        let tokens = self.lexer.tokenize_to_vec(&file_contents)?;
        parse(&tokens)
    }
    
    fn compile_with_fresh_scopes(&mut self, exprs: &[Expr]) -> Result<HashMap<String, Var>, String> {
        let mut saved_scopes = vec![HashMap::new()];
        std::mem::swap(&mut self.scopes, &mut saved_scopes);
        self.compile(&exprs)?;
        std::mem::swap(&mut self.scopes, &mut saved_scopes);
        assert!(saved_scopes.len() == 1);
        Ok(saved_scopes.into_iter().next().unwrap())
    }

    fn compile_block(&mut self, exprs: &[Expr]) -> Result<(), String> {
        if exprs.is_empty() { todo!("Compile ()"); }

        self.compile_expr(&exprs[0])?;
        for expr in &exprs[1..] {
            self.code.push(Instr::Pop);
            self.compile_expr(expr)?;
        }
        Ok(())
    }

    fn push(&mut self, instr: Instr) -> usize {
        self.code.push(instr);
        self.code.len() - 1
    }

    fn ensure_module_loaded<'a>(&'a mut self, mod_name: &str) -> Result<(usize, &'a HashMap<String, Var>), String> {
        // This double-lookup is a workaround for problem case 3 in the NLL RFC:
        // https://github.com/rust-lang/rfcs/blob/master/text/2094-nll.md#problem-case-3-conditional-control-flow-across-functions. We
        // could use the entry API, but that would mean a clone of `mod_name`
        // even when it's already in the cache.
        let (code_index, scope) = if self.module_cache.contains_key(mod_name) {
            self.module_cache.get(mod_name).unwrap()
        } else {
            let mod_start_index = self.push(Instr::Nop);
            let mod_exprs = self.parse_file(mod_name)?;
            let mod_scope = self.compile_with_fresh_scopes(&mod_exprs)?;
            let mod_end_index = self.push(Instr::ModuleEnd);
            self.code[mod_start_index] = Instr::ModuleStart {
                num_instructions: mod_end_index - mod_start_index
            };
            self.module_cache.entry(mod_name.to_string())
                // The code index, used in LoadModule, points right after
                // ModuleStart.
                .or_insert((mod_start_index + 1, mod_scope))
        };
        Ok((*code_index, scope))
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<(), String> {
        match expr {
            Expr::Noun(noun) => self.compile_noun(noun),
            Expr::Verb(verb) => self.compile_verb(verb, None),
            Expr::PragmaLoad(mod_name) => {
                let load_module_index = self.push(Instr::Nop);
                let mut local_scope = self.scopes.pop().unwrap();
                let mod_locals_base_slot = next_local_slot(&local_scope);
                let (code_index, mod_scope) = self.ensure_module_loaded(mod_name)?;
                for (name, mut var) in mod_scope.iter().map(|(name, var)| (name.clone(), *var)) {
                    var.slot += mod_locals_base_slot;
                    match var.place {
                        Place::Local => local_scope.insert(name, var),
                        Place::ClosureEnv => panic!("Error: closure env var {name} in toplevel of module {mod_name}"),
                    };
                }
                self.scopes.push(local_scope);
                self.code[load_module_index] = Instr::LoadModule { code_index };
                self.code.push(Instr::PushLiteralInteger(0));  // TODO return module/unit/something
                Ok(())
            }
        }
    }

    fn compile_verb(&mut self, verb: &Verb, arity: Option<Arity>) -> Result<(), String> {
        match verb {
            Verb::UpperAssign(name, rhs) => {
                self.compile_verb(rhs, None)?;
                let dst = self.fetch_var_in_current_scope(name);
                self.code.push(Instr::StoreTo { dst });
            }
            Verb::SmallVerb(small_verb) => self.compile_small_verb(small_verb)?,
            Verb::AmbivalentCases(monadic, dyadic) => match arity {
                Some(Arity::One) => self.compile_verb(&*monadic, arity)?,
                Some(Arity::Two) => self.compile_verb(&*monadic, arity)?,
                None => {
                    self.compile_verb(monadic, Some(Arity::One))?;
                    self.compile_small_verb(dyadic)?;
                    self.code.push(Instr::CollectVerbAlternatives);
                }
            },
            Verb::Atop(f, g) => {
                self.compile_verb(f.as_ref(), arity)?;
                self.compile_verb(g.as_ref(), Some(Arity::One))?;
                self.code.push(Instr::MakeAtopFunc);
            }
            Verb::Bind(f, y) => {
                self.compile_verb(f.as_ref(), Some(Arity::Two))?;
                self.compile_small_noun(y.as_ref())?;
                self.code.push(Instr::MakeBoundFunc);
            }
            Verb::Fork(f, h, g) => {
                self.compile_verb(f.as_ref(), arity)?;
                self.compile_small_verb(h.as_ref())?;
                self.compile_small_verb(g.as_ref())?;
                self.code.push(Instr::MakeForkFunc);
            }
        }
        Ok(())
    }

    fn compile_unpacking_assignment(&mut self, pat: &Pattern, keep_top: bool) {
        match pat {
            Pattern::Name(name) => {
                let dst = self.fetch_var_in_current_scope(name);
                self.code.push(Instr::StoreTo { dst });
                if !keep_top { self.code.push(Instr::Pop) }
            }
            Pattern::As(pat, as_pat) => {
                self.compile_unpacking_assignment(&*pat, true);
                self.compile_unpacking_assignment(&*as_pat, keep_top);
            }
            Pattern::Array(pats) => {
                if keep_top { self.code.push(Instr::Dup) }
                self.code.push(Instr::Splat { count: pats.len() });
                for pat in pats {
                    self.compile_unpacking_assignment(pat, false);
                }
            }
        }
    }

    // TODO compile_small_verb should expect an arity (may be decided by a particular adverb)
    fn compile_small_verb(&mut self, small_verb: &SmallVerb) -> Result<(), String> {
        match small_verb {
            SmallVerb::UpperName(name) => {
                let src = self.fetch_var(name)?;
                self.code.push(Instr::PushVar { src });
            }
            SmallVerb::PrimVerb(PrimVerb::C0) => self.code.push(Instr::LiteralBytes { bytes: [0; 8] }),
            &SmallVerb::PrimVerb(prim) => self.code.push(Instr::PushPrimVerb { prim }),
            SmallVerb::Lambda(explicit_args, exprs) => {
                let make_func_index = self.push(Instr::Nop);
                // TODO alloc locals?

                let mut scope = HashMap::new();
                if let None = explicit_args {
                    scope.insert("x".to_string(), local_var(0));
                    scope.insert("y".to_string(), local_var(1));  // Might not be used!
                }
                self.scopes.push(scope);

                if let Some(ExplicitArgs { x, y }) = explicit_args {
                    if y.is_some() {
                        self.code.push(Instr::PushVar { src: local_var(1) });
                    }
                    self.code.push(Instr::PushVar { src: local_var(0) });
                    self.compile_unpacking_assignment(x, false);

                    if let Some(y_pat) = y {
                        self.compile_unpacking_assignment(y_pat, false);
                    }
                }

                self.compile_block(exprs)?;
                self.code.push(Instr::Return);

                let num_instructions = self.code.len() - (make_func_index + 1);
                self.code[make_func_index] = Instr::MakeFunc { num_instructions };

                let final_scope = self.scopes.pop().unwrap();

                // Tell the outer scope how to populate the closure environment.
                let mut closure_vars: Vec<(&String, &Var)> =
                    final_scope.iter().filter(|(_, v)| v.place == Place::ClosureEnv).collect();

                if closure_vars.len() > 0 {
                    self.code.push(Instr::MakeClosure { num_closure_vars: closure_vars.len() });
                    closure_vars.sort_unstable_by_key(|(_, var)| var.slot);
                    for (name, _) in &closure_vars {
                        let src = self.fetch_var(name)?;
                        self.code.push(Instr::PushVar { src });
                    }
                }
            }
            SmallVerb::Adverb(prim, small_expr_box) => {
                match small_expr_box.as_ref() {
                    SmallExpr::Verb(small_verb) => {
                        self.compile_small_verb(small_verb)?;
                        if !matches!(prim, PrimAdverb::Dot) {
                            self.code.push(Instr::CallPrimAdverb { prim: *prim });
                        }
                    }
                    SmallExpr::Noun(small_noun) => {
                        self.compile_small_noun(small_noun)?;
                        self.code.push(Instr::CallPrimAdverb { prim: *prim });
                    }
                }
            }
        }
        Ok(())
    }

    fn get_local_scope_mut(&mut self) -> &mut HashMap<String, Var> {
        self.scopes.last_mut().unwrap()
    }

    fn get_local_scope(&self) -> &HashMap<String, Var> {
        self.scopes.last().unwrap()
    }

    // Leaves the compiled noun's value in subject1
    fn compile_noun(&mut self, noun: &Noun) -> Result<(), String> {
        match noun {
            Noun::SmallNoun(small_noun) => self.compile_small_noun(small_noun)?,
            Noun::LowerAssign(pat, rhs) => {
                self.compile_noun(rhs)?;
                self.compile_unpacking_assignment(pat, true);
            }
            Noun::Sentence(small_noun, predicates) => {
                self.compile_small_noun(small_noun)?;
                for predicate in predicates {
                    self.compile_predicate(predicate)?;
                }
            }
        }
        Ok(())
    }

    // Before calling, add the code to put the x argument on the stack
    fn compile_predicate(&mut self, predicate: &Predicate) -> Result<(), String> {
        match predicate {
            Predicate::VerbCall(verb, maybe_y_arg) => {
                if !matches!(verb, Verb::SmallVerb(SmallVerb::PrimVerb(_))) {
                    self.compile_verb(verb, Some(if maybe_y_arg.is_some() { Arity::Two }
                                                 else { Arity::One }))?;
                }
                if let Some(y) = maybe_y_arg {
                    self.compile_small_noun(y)?;
                }
                self.code.push(match verb {
                    &Verb::SmallVerb(SmallVerb::PrimVerb(prim)) =>
                        if maybe_y_arg.is_some() { Instr::CallPrimVerb2 { prim } }
                    else { Instr::CallPrimVerb1 { prim } },
                    _ =>
                        if maybe_y_arg.is_some() { Instr::Call2 }
                    else { Instr::Call1 }
                });
            }
            Predicate::ForwardAssignment(pat) =>
                self.compile_unpacking_assignment(pat, true),
        }
        Ok(())
    }

    fn fetch_var_in_current_scope(&mut self, name: &str) -> Var {
        let local_scope = self.get_local_scope_mut();
        if let Some(var) = local_scope.get(name) {
            return *var;
        }
        let var = local_var(next_local_slot(local_scope));
        local_scope.insert(name.to_string(), var);
        var
    }

    // If name isn't defined in the current scope, updates all intervening
    // scopes to include it in their closure environment.
    fn fetch_var(&mut self, name: &str) -> Result<Var, String> {
        for i in (0 .. self.scopes.len()).rev() {
            if self.scopes[i].contains_key(name) {
                for j in (i+1)..self.scopes.len() {
                    add_closure_var(name, &mut self.scopes[j]);
                }
                break;
            }
        }

        if let Some(var) = self.get_local_scope().get(name) {
            return Ok(*var);
        }
        err!("Undefined name: `{name}'")
    }

    fn compile_small_noun(&mut self, small_noun: &SmallNoun) -> Result<(), String> {
        // TODO currently V N pushes N but doesn't pop it unless another
        // expression follows (e.g. the program "+ 3" is push prim; pop verb; push literal)
        use SmallNoun::*;
        match small_noun {
            PrimNoun(prim) => {
                let verb = match prim {
                    lex::PrimNoun::Exit => PrimVerb::Exit,
                    lex::PrimNoun::Show => PrimVerb::Show,
                    lex::PrimNoun::Print => PrimVerb::Print,
                    lex::PrimNoun::ReadFile => PrimVerb::ReadFile,
                    lex::PrimNoun::Rand => PrimVerb::Rand,
                    lex::PrimNoun::Rec => PrimVerb::Rec,
                    lex::PrimNoun::Type => PrimVerb::Type,
                    lex::PrimNoun::C0 => {
                        self.code.push(Instr::LiteralBytes { bytes: [0; 8] });
                        return Ok(())
                    }
                };
                self.code.push(Instr::PushPrimVerb { prim: verb });
            }
            LowerName(name) => {
                let src = self.fetch_var(name)?;
                self.code.push(Instr::PushVar { src });
            }
            Block(exprs) => self.compile_block(exprs)?,
            IntLiteral(int) => self.code.push(Instr::PushLiteralInteger(*int)),
            FloatLiteral(float) => self.code.push(Instr::PushLiteralFloat(*float)),
            CharLiteral(byte) => {
                let mut bytes = [0; 8];
                bytes[0] = *byte;
                self.code.push(Instr::LiteralBytes { bytes });
            }
            StringLiteral(s) => {
                self.code.push(Instr::MakeString { num_bytes: s.len() });
                for i in (0..s.len()).step_by(8) {
                    let mut bytes = [0; 8];
                    for j in i..(i+8).min(s.len()) {
                        bytes[j-i] = s.as_bytes()[j];
                    }
                    self.code.push(Instr::LiteralBytes { bytes });
                }
            }
            ArrayLiteral(exprs) => {
                for expr in exprs {
                    self.compile_expr(expr)?;
                }
                self.code.push(Instr::CollectToArray { num_elems: exprs.len() });
            }
        }
        Ok(())
    }
}

fn local_var(slot: usize) -> Var {
    Var { place: Place::Local, slot }
}

fn closure_var(slot: usize) -> Var {
    Var { place: Place::ClosureEnv, slot }
}

// Note that the next local slot isn't always the same as the number of local
// slots in the scope; loading a module can add a var to the scope that has the
// same name but a different slot number, shadowing the current local.
//
// For example:
//
//   \ module mod
//   a: 10
//
//   \ main program
//   a: 5
//   load mod
//   a Print
//
// After `a: 5` executes, the scope is {"a": Local(0)}. Then `mod` is loaded;
// inside, `a` is local slot 0, but when it's imported to the program, it's
// offset by 1 to put it after the current locals. The scope is
// {"a": Local(1)}. The next local slot would be 2, even though there's only one
// local in scope.
fn next_local_slot(scope: &HashMap<String, Var>) -> usize {
    scope.values()
        .filter_map(|v| if v.place == Place::Local { Some(v.slot) } else { None })
        .max()
        .map(|slot| slot + 1)
        .unwrap_or(0)
}

fn get_num_closure_vars(scope: &mut HashMap<String, Var>) -> usize {
    scope.values().filter(|v| v.place == Place::ClosureEnv).count()
}

// TODO track number of closure vars per scope so we don't have to scan
fn add_closure_var(name: &str, scope: &mut HashMap<String, Var>) {
    let slot = get_num_closure_vars(scope);
    scope.insert(name.to_string(), closure_var(slot));
}

fn get_stdlib_names() -> &'static [&'static str] {
    &[
        
    ]
}
