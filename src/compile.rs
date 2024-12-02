use std::{
    collections::HashMap,
    fs,
};

use crate::bytecode::*;
use crate::lex::*;
use crate::parse::*;
use crate::util::*;

enum NameValue {
    Var(Var),
    Prim(PrimFunc),
}

pub fn compile_string(text: &str) -> Res<Vec<Instr>> {
    let mut compiler = Compiler::new();
    compiler.compile_string(text)?;
    Ok(compiler.code)
}

// Invariants: `scopes` non-empty after `new`.
#[derive(Debug)]
pub struct Compiler {
    pub code: Vec<Instr>,
    pub lexer: Lexer,

    scopes: Vec<HashMap<String, Var>>,
    primitive_identifiers: HashMap<&'static str, PrimFunc>,

    // (code index of module code (right after LoadModule), module vars)
    //
    // TODO in REPL use, allow cache entries to become dirty if a module or one of its transitive
    // imports changes.
    module_cache: HashMap<String, (usize, HashMap<String, Var>)>,
}

// TODO be able to compile globals for repl
// TODO adverb nounification (') (for ultimate adverb verbification)
impl Compiler {
    pub fn new() -> Self {
        Self::with_lexer(Lexer::new())
    }

    pub fn with_lexer(lexer: Lexer) -> Self {
        let mut globals = HashMap::new();
        for (i, name) in get_stdlib_names().iter().enumerate() {
            globals.insert(name.to_string(), Var { place: Place::Local, slot: i });
        }

        Self { code: vec![],
               lexer,
               scopes: vec![globals],
               primitive_identifiers: make_primitive_identifier_map(),
               module_cache: HashMap::new() }
    }

    // Compile program text.
    pub fn compile_string(&mut self, text: &str) -> Res<()> {
        let tokens = self.lexer.tokenize_to_vec(text)?;
        if tokens.is_empty() { return Ok(()) }
        let exprs = parse(&tokens)?;
        self.compile(&exprs)
    }

    pub fn compile(&mut self, exprs: &[Expr]) -> Res<()> {
        let start_of_new_code = self.code.len();
        let result = self.compile_block(exprs, true);
        if result.is_ok() {
            self.eliminate_unconditional_jump_chains();
        } else {
            // Clean up everything compiled during this invocation.
            self.scopes.truncate(1);  // Reset to global scope
            self.code.truncate(start_of_new_code);
            self.module_cache.retain(|_, (code_index, _)| *code_index < start_of_new_code);
        }
        result
    }

    fn eliminate_unconditional_jump_chains(&mut self) {
        use Instr::*;

        // `i` is the index of the instruction that contains the offset.
        fn next_ip(i: usize, offset: i64) -> usize {
            // Note that we add 1 here because when we run instruction `i`, `ip`
            // is actually equal to `i + 1`; the offset is from `i + 1`, not
            // `i`.
            (i as i64 + 1 + offset) as usize
        }

        // When we see a chain of unconditional jumps that eventually lead to a return, return
        // directly instead.
        //
        // This can happen in nested ifs, as in:
        //     { if(c1; if(c2; t2; e2); e1) }
        // which translates to
        //     eval c1
        //     if false, jump to L3
        //     eval c2
        //     if false, jump to L1
        //     eval t2
        //     jump to L2
        // L1: eval e2
        // L2: jump to L4
        // L3: eval e1
        // L4: return
        //
        // If c1 and c2 are true, then after evaluating t2 we want to jump to the instruction
        // immediately after e2. That instruction itself is a jump to the instruction immediately
        // after e1. Put another way, the else-skipping jump of the outer `if` jumps to the
        // else-skipping jump of the inner `if`.
        //
        // Eliminiating these chains is an optimization, but the interpreter also assumes that it's
        // been done when identifying tail calls made from within `if` expressions; since we do
        // this, it can just check if the next instruction is a return instead of having to follow
        // the chains itself.
        for i in (0..self.code.len()).rev() {
            match self.code[i] {
                JumpRelative { offset } => match self.code.get(next_ip(i, offset)) {
                    Some(Return) => self.code[i] = Return,

                    // 0: Jump(2)  2 = 3-1
                    // 1:
                    // 2:
                    // 3: Jump(1)  1 = 5-4
                    // 4:
                    // 5: (target)
                    //
                    // So Jump(2) should be adjusted to Jump(4) (4 = 5-1 = 2+1+1)
                    Some(JumpRelative { offset: offset2 }) =>
                        self.code[i] = JumpRelative { offset: offset + offset2 + 1 },

                    Some(JumpRelativeUnless { offset: offset2 }) =>
                        self.code[i] = JumpRelativeUnless { offset: offset + offset2 + 1 },

                    _ => {}
                }

                JumpRelativeUnless { offset } => match self.code.get(next_ip(i, offset)) {
                    Some(JumpRelative { offset: offset2 }) =>
                        self.code[i] = JumpRelativeUnless { offset: offset + offset2 + 1 },
                    _ => {}
                }

                _ => {}
            }
        }
    }

    fn parse_file(&self, filepath: &str) -> Res<Vec<Expr>> {
        let file_contents = fs::read_to_string(filepath).map_err(
            |err| cold(format!("\"{filepath}\": {err}"))
        )?;
        let tokens = self.lexer.tokenize_to_vec(&file_contents)?;
        parse(&tokens)
    }

    fn compile_with_fresh_scopes(&mut self, exprs: &[Expr]) -> Res<HashMap<String, Var>> {
        let mut saved_scopes = vec![HashMap::new()];
        std::mem::swap(&mut self.scopes, &mut saved_scopes);
        let res = self.compile(&exprs);
        std::mem::swap(&mut self.scopes, &mut saved_scopes);
        res?;
        assert!(saved_scopes.len() == 1);
        Ok(saved_scopes.into_iter().next().unwrap())
    }

    fn compile_block(&mut self, exprs: &[Expr], keep: bool) -> Res<()> {
        match exprs.split_last() {
            None => todo!("Compile ()"),
            Some((last, init)) => {
                for expr in init {
                    self.compile_expr(expr, false)?;
                }
                self.compile_expr(last, keep)
            }
        }
    }

    fn push(&mut self, instr: Instr) -> usize {
        self.code.push(instr);
        self.code.len() - 1
    }

    // If this returns Ok, then module_cache.contains_key(mod_name) is guaranteed.
    fn ensure_module_loaded(&mut self, mod_name: &str) -> Res<()> {
        // This double-lookup is a workaround for problem case 3 in the NLL RFC:
        // https://github.com/rust-lang/rfcs/blob/master/text/2094-nll.md#problem-case-3-conditional-control-flow-across-functions. We
        // could use the entry API, but that would mean a clone of `mod_name`
        // even when it's already in the cache.
        if self.module_cache.contains_key(mod_name) { return Ok(()); }

        let mod_start_index = self.push(Instr::Nop);
        let mod_exprs = self.parse_file(mod_name)?;
        let mod_scope = self.compile_with_fresh_scopes(&mod_exprs)?;
        let mod_end_index = self.push(Instr::ModuleEnd);
        self.code[mod_start_index] = Instr::ModuleStart {
            num_instructions: mod_end_index - mod_start_index
        };

        // The code index, used in LoadModule, points right after ModuleStart.
        self.module_cache.insert(mod_name.to_string(), (mod_start_index + 1, mod_scope));
        Ok(())
    }

    fn compile_expr_with_arity(&mut self, expr: &Expr, arity: Option<u32>, keep: bool) -> Res<()> {
        if let Expr::Verb(verb) = expr { self.compile_verb(verb, arity, keep) }
        else { self.compile_expr(expr, keep) }
    }

    fn compile_expr(&mut self, expr: &Expr, keep: bool) -> Res<()> {
        match expr {
            Expr::Noun(noun) => self.compile_noun(noun, keep),
            Expr::Verb(verb) => self.compile_verb(verb, None, keep),
            Expr::PragmaLoad(mod_name) => {
                self.ensure_module_loaded(mod_name)?;

                let mut local_scope = self.scopes.pop().unwrap();
                let mod_locals_base_slot = next_local_slot(&local_scope);

                let (code_index, mod_scope) = self.module_cache.get(mod_name).unwrap();
                for (name, mut var) in mod_scope.iter().map(|(name, var)| (name.clone(), *var)) {
                    var.slot += mod_locals_base_slot;
                    match var.place {
                        Place::Local => local_scope.insert(name, var),
                        Place::ClosureEnv => panic!("Error: closure env var {name} in toplevel of module {mod_name}"),
                    };
                }
                self.scopes.push(local_scope);
                self.code.push(Instr::LoadModule { code_index: *code_index });
                self.code.push(Instr::PushLiteralInteger(0));  // TODO return module/unit/something
                Ok(())
            }
        }
    }

    fn compile_verb(&mut self, verb: &Verb, arity: Option<u32>, keep: bool) -> Res<()> {
        match verb {
            Verb::UpperAssign(name, rhs) => {
                self.compile_verb(rhs, arity, true)?;
                match self.fetch_name_in_current_scope(name) {
                    NameValue::Prim(prim) => return cold_err!("Invalid assignment to primitive `{prim}'"),
                    NameValue::Var(dst) => {
                        if keep { self.code.push(Instr::Dup) }
                        self.code.push(Instr::StoreTo { dst });
                    }
                }
            }
            Verb::SmallVerb(small_verb) => self.compile_small_verb(small_verb, arity, keep)?,
            Verb::Train(f, rest) => self.compile_train_as_explicit(f, rest, arity, keep)?,
        }
        Ok(())
    }

    fn compile_train_as_explicit(&mut self, f: &SmallVerb, parts: &[TrainPart], arity: Option<u32>, keep: bool) -> Res<()> {
        self.compile_small_verb(f, arity, true)?;

        let mut last_slot_called_on_args = 0;
        let mut total_closure_slots = 1;
        for part in parts {
            match part {
                TrainPart::Fork(mid, rhs) => {
                    total_closure_slots += 2;
                    self.compile_small_verb(mid, Some(2), true)?;
                    last_slot_called_on_args = total_closure_slots - 1;
                    self.compile_small_verb(rhs, arity, true)?;
                }
                TrainPart::Atop(small_verb) => {
                    total_closure_slots += 1;
                    self.compile_small_verb(small_verb, Some(1), true)?;
                }
            }
        }

        let make_func_index = self.push(Instr::Nop);

        if last_slot_called_on_args != 0 {
            self.code.push(Instr::CopyArgs);
        }
        self.code.push(Instr::CallOnArgs { var: closure_var(0) });

        for (slot, part) in (1..).step_by(2).zip(parts) {
            match part {
                TrainPart::Fork(_, _) => {
                    let rhs_slot = slot + 1;
                    // Each branch sets up x f y on the stack
                    if rhs_slot != last_slot_called_on_args {
                        self.code.push(Instr::PushVar { src: closure_var(slot) });
                        self.code.push(Instr::CopyArgs);
                        self.code.push(Instr::CallOnArgs { var: closure_var(rhs_slot) });
                    } else {
                        self.code.push(Instr::StoreTo { dst: local_var(0) });
                        self.code.push(Instr::CallOnArgs { var: closure_var(rhs_slot) });
                        self.code.push(Instr::TuckVarLastUse { src: local_var(0) });
                        self.code.push(Instr::TuckVar { src: closure_var(slot) });
                    }
                    self.code.push(Instr::Call2);
                }
                TrainPart::Atop(_) => {
                    self.code.push(Instr::PushVar { src: closure_var(slot) });
                    self.code.push(Instr::Call1);
                }
            }
        }

        self.code.push(Instr::Return);

        self.code[make_func_index] = Instr::MakeFunc {
            num_instructions: self.code.len() - (make_func_index + 1)
        };
        self.code.push(Instr::MakeClosureFromStack { num_closure_vars: total_closure_slots });

        if !keep { self.code.push(Instr::Pop); }

        Ok(())
    }

    fn compile_unpacking_assignment(&mut self, pat: &Pattern, keep: bool) -> Res<()> {
        match pat {
            Pattern::View(small_verb, pat) => {
                let predicate = Predicate::VerbCall(Verb::SmallVerb(small_verb.clone()), None);
                self.compile_predicate(&predicate, true)?;
                self.compile_unpacking_assignment(pat, keep)?;
            }
            Pattern::Constant(literal) => {
                if keep { self.code.push(Instr::Dup) }
                self.compile_small_noun(&SmallNoun::Constant(literal.clone()), true)?;
                if literal.is_atom() {
                    self.code.push(Instr::CallPrimFunc2 { prim: PrimFunc::Equal });
                } else {
                    self.code.push(Instr::CallPrimFunc2 { prim: PrimFunc::Match });
                }
                self.code.push(Instr::Assert);
            }
            Pattern::Wildcard => if !keep { self.code.push(Instr::Pop) }
            Pattern::Name(name) => match self.fetch_name_in_current_scope(name) {
                NameValue::Prim(prim) => return cold_err!("Can't use primitive `{prim}' as a pattern"),
                NameValue::Var(dst) => {
                    if keep { self.code.push(Instr::Dup) }
                    self.code.push(Instr::StoreTo { dst });
                }
            }
            Pattern::As(pat, as_pat) => {
                self.compile_unpacking_assignment(&*pat, true)?;
                self.compile_unpacking_assignment(&*as_pat, keep)?;
            }
            Pattern::Array(pats) => {
                if keep { self.code.push(Instr::Dup) }
                let splat_index = self.push(Instr::Nop);

                enum Splicing { NoSplice, Splice { i: usize, keep_splice: bool } }
                let mut splicing = Splicing::NoSplice;

                for i in 0..pats.len() {
                    match &pats[i] {
                        PatternElem::Pattern(pat) => self.compile_unpacking_assignment(pat, false)?,
                        PatternElem::Subarray(name) if matches!(splicing, Splicing::NoSplice) => match name {
                            Some(name) => {
                                self.compile_unpacking_assignment(&Pattern::Name(name.clone()), false)?;
                                splicing = Splicing::Splice { i, keep_splice: true };
                            }
                            None => splicing = Splicing::Splice { i, keep_splice: false },
                        }
                        PatternElem::Subarray(_) => return cold_err!("Only one `..' is allowed per array-matching pattern."),
                    }
                }

                self.code[splat_index] = match splicing {
                    Splicing::NoSplice => Instr::SplatReverse { count: pats.len() },
                    Splicing::Splice {i, keep_splice} => Instr::SplatReverseWithSplice {
                        prefix_count: i as u32,
                        suffix_count: (pats.len() - i - 1) as u32,
                        keep_splice
                    }
                };
            }
        }

        Ok(())
    }

    fn compile_small_verb(&mut self, small_verb: &SmallVerb, arity: Option<u32>, keep: bool) -> Res<()> {
        if let Some(prim) = small_verb.into_prim_func(arity, &self.primitive_identifiers)? {
            if keep { self.code.push(Instr::PushPrimFunc { prim }); }
            return Ok(());
        }

        match small_verb {
            &SmallVerb::PrimVerb(prim) => {
                if prim == PrimFunc::Rec && self.scopes.len() < 2 {
                    return cold_err!("Can't use `Rec` outside of an explicit definition")
                }
                if keep { self.code.push(Instr::PushPrimFunc { prim }) }
            }
            SmallVerb::UpperName(name) => match self.fetch_name(name)? {
                NameValue::Prim(prim) => {
                    if prim == PrimFunc::Rec && self.scopes.len() < 2 {
                        return cold_err!("Can't use `Rec` outside of an explicit definition")
                    }
                    if keep { self.code.push(Instr::PushPrimFunc { prim }) }
                }
                NameValue::Var(src) => if keep { self.code.push(Instr::PushVar { src }) }
            }
            SmallVerb::VerbBlock(exprs, verb) => {
                if !exprs.is_empty() {
                    self.compile_block(exprs, false)?;
                }
                self.compile_verb(&*verb, arity, keep)?;
            }
            SmallVerb::Lambda(lambda) => {
                let make_func_index = self.push(Instr::Nop);
                // TODO alloc locals?

                match lambda {
                    Lambda::Short(exprs) => self.compile_short_lambda(exprs, arity)?,
                    Lambda::Cases(cases) => self.compile_lambda_cases(cases)?,
                }

                let num_instructions = self.code.len() - (make_func_index + 1);
                self.code[make_func_index] = Instr::MakeFunc { num_instructions };

                let final_scope = self.scopes.pop().unwrap();
                self.mark_last_local_uses(make_func_index, next_local_slot(&final_scope));

                // Tell the outer scope how to populate the closure environment.
                let mut closure_vars: Vec<(&String, &Var)> =
                    final_scope.iter().filter(|(_, v)| v.place == Place::ClosureEnv).collect();

                if closure_vars.len() > 0 {
                    self.code.push(Instr::MakeClosure { num_closure_vars: closure_vars.len() });
                    closure_vars.sort_unstable_by_key(|(_, var)| var.slot);
                    for (name, _) in &closure_vars {
                        irrefutable!(self.fetch_name(name)?,
                                     NameValue::Var(src) => self.code.push(Instr::PushVar { src }));
                    }
                }
                if !keep { self.code.push(Instr::Pop) }
            }
            SmallVerb::PrimAdverbCall(prim, small_expr_box) => {
                match small_expr_box.as_ref() {
                    SmallExpr::Verb(small_verb) => {
                        self.compile_small_verb(small_verb, get_prim_adverb_operand_arity(*prim, arity), true)?;
                        if !matches!(prim, PrimAdverb::Dot) {
                            self.code.push(Instr::CallPrimAdverb { prim: *prim });
                        }
                    }
                    SmallExpr::Noun(small_noun) => {
                        self.compile_small_noun(small_noun, true)?;
                        self.code.push(Instr::CallPrimAdverb { prim: *prim });
                    }
                }
                if !keep { self.code.push(Instr::Pop) }
            }
            SmallVerb::NamedAdverbCall(small_verb, elems) => match (self.get_prim_adverb_from_small_verb(small_verb), &elems[..]) {
                (Some(adverb), [Elem::Expr(expr)]) => {
                    self.compile_expr_with_arity(&expr, get_prim_adverb_operand_arity(adverb, arity), true)?;
                    self.code.push(Instr::CallPrimAdverb { prim: adverb });
                    if !keep { self.code.push(Instr::Pop) }
                }
                _ => {
                    if elems.len() > ArgSpec::MAX_ARITY as usize {
                        return cold_err!("Functions can't take more than {} arguments.", ArgSpec::MAX_ARITY);
                    }

                    self.compile_small_verb(small_verb, Some(elems.len() as u32), true)?;
                    for elem in elems {
                        if matches!(elem, Elem::Spliced(_)) {
                            todo!("Support splices in argument lists.");
                        }
                        self.compile_elem(elem)?;
                    }

                    let arg_spec = ArgSpec::new(elems.iter().map(|arg| !matches!(arg, Elem::MissingArg)));
                    match arg_spec {
                        Some(arg_spec) => self.code.push(Instr::CallSpec { arg_spec }),
                        None => return cold_err!(
                            "Can't call a function or index an array with more than 63 arguments."
                        ),
                    }
                    
                    if !keep { self.code.push(Instr::Pop) }
                }
            }
        }
        Ok(())
    }

    fn get_prim_adverb_from_small_verb(&self, small_verb: &SmallVerb) -> Option<PrimAdverb> {
        let prim_func = if let SmallVerb::UpperName(name) = small_verb {
            self.primitive_identifiers.get(name.as_str())?
        } else {
            return None
        };

        Some(match prim_func {
            PrimFunc::Runs => PrimAdverb::Runs,
            _ => return None,
        })
    }

    fn compile_short_lambda(&mut self, exprs: &[Expr], arity: Option<u32>) -> Res<()> {
        // If we don't know the arity, assume the function is dyadic and then look at the generated
        // code to see which arguments were actually accessed.
        //
        // Note that we don't need an ArgCheck at all if we know the arity this function will be
        // called at.
        let arg_check_index = if arity.is_none() {
            Some(self.push(Instr::ArgCheck { arg_spec: ArgSpec::saturated(2) }))
        } else {
            None
        };

        let assumed_arity = arity.unwrap_or(2);
        if assumed_arity > 2 {
            return cold_err!("Can't compile attempt to use implicit-arg lambda at arity {assumed_arity} > 2");
        }
        
        let mut scope = HashMap::new();

        if assumed_arity >= 1 {
            scope.insert("x".to_string(), local_var(0));
            // TODO eliminate StoreTo(var); PushVarLastUse(var)
            self.code.push(Instr::StoreTo { dst: local_var(0) });
        }

        // y_start is the start of the instructions that should be replaced with Nops if y is never
        // actually accessed.
        let y_start = if assumed_arity == 2 {
            scope.insert("y".to_string(), local_var(1));
            Some(self.push(Instr::StoreTo { dst: local_var(1) }))
        } else {
            None
        };

        self.scopes.push(scope);

        let body_start = self.code.len();
        self.compile_block(exprs, true)?;
        self.code.push(Instr::Return);

        if let (None, Some(y_start), Some(arg_check_index)) = (arity, y_start, arg_check_index) {
            // We compiled at arity 2, but we don't actually know what arity this function will be
            // called at. Scan the body to see what args were actually mentioned, and replace the
            // instructions that prepare unmentioned args with Nops.
            //
            // TODO consider allowing but discarding extra args
            // 
            // TODO eliminate nops (would need to adjust offsets)
            //
            // TODO looking at generated code won't work if we skip generating push instrs for
            // completely unused args, e.g. {y; x + 1} should be dyadic, but we might not generate
            // code for pushing y.
            if !accessed(&self.code[body_start..], local_var(1)) {
                decrement_locals(&mut self.code[body_start..], 1);
                if !accessed(&self.code[body_start..], local_var(0)) {
                    decrement_locals(&mut self.code[body_start..], 0);
                    self.code[arg_check_index] = Instr::ArgCheck { arg_spec: ArgSpec::saturated(0) };
                    self.fill_with_nop(arg_check_index+1..body_start);
                } else {
                    self.code[arg_check_index] = Instr::ArgCheck { arg_spec: ArgSpec::saturated(1) };
                    self.fill_with_nop(y_start..body_start);
                }
            }
        }

        Ok(())
    }

    fn compile_lambda_cases(&mut self, cases: &[LambdaCase]) -> Res<()> {
        self.scopes.push(HashMap::new());
        for i in 0..cases.len() {
            let LambdaCase(ExplicitArgs(pats), exprs) = &cases[i];
            let header_index = if i == cases.len() - 1 {
                None
            } else {
                Some(self.push(Instr::Nop))
            };

            let arg_check_index = self.push(Instr::Nop);
            for pat in pats {
                if let Some(pat) = pat {
                    self.compile_unpacking_assignment(pat, false)?;
                }
            }

            let arg_spec = ArgSpec::new(
                pats.iter().map(|pat| pat.is_some())
            ).ok_or_else(
                || format!("Can't define a function case that takes more than {} arguments.",
                           ArgSpec::MAX_ARITY)
            )?;
            self.code[arg_check_index] = Instr::ArgCheck { arg_spec };

            if header_index.is_some() {
                self.code.push(Instr::HeaderPassed);
            }

            self.compile_block(exprs, true)?;
            let case_end = self.push(Instr::Return);

            if let Some(header_index) = header_index {
                self.code[header_index] = Instr::Header {
                    next_case_offset: (case_end - header_index) as i64
                };
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
    fn compile_noun(&mut self, noun: &Noun, keep: bool) -> Res<()> {
        match noun {
            Noun::SmallNoun(small_noun) => self.compile_small_noun(small_noun, keep)?,
            Noun::LowerAssign(pat, rhs) => {
                self.compile_noun(rhs, true)?;
                self.compile_unpacking_assignment(pat, keep)?;
            }
            Noun::ModifyingAssign(pattern, predicates) => {
                self.compile_small_noun(&pattern_to_small_noun(pattern)?, true)?;
                for predicate in predicates {
                    self.compile_predicate(predicate, true)?;
                }
                self.compile_unpacking_assignment(pattern, keep)?;
            }
            Noun::Sentence(small_noun, predicates) => {
                self.compile_small_noun(small_noun, true)?;
                if let Some((last, init)) = predicates.split_last() {
                    for predicate in init {
                        self.compile_predicate(predicate, true)?;
                    }
                    self.compile_predicate(last, keep)?;
                }
            }
        }
        Ok(())
    }

    // Before calling, add the code to put the x argument on the stack
    fn compile_predicate(&mut self, predicate: &Predicate, keep: bool) -> Res<()> {
        match predicate {
            Predicate::VerbCall(verb, maybe_y_arg) => {
                let prim_func = verb.into_prim_func(Some(1 + maybe_y_arg.is_some() as u32),
                                                    &self.primitive_identifiers)?;
                if prim_func.is_none() {
                    let arity = 1 + maybe_y_arg.is_some() as u32;
                    self.compile_verb(verb, Some(arity), true)?;
                }

                match maybe_y_arg {
                    None => self.code.push(match prim_func {
                        Some(prim) => Instr::CallPrimFunc1 { prim },
                        None => Instr::Call1,
                    }),
                    Some(y) => {
                        self.compile_small_noun(y, true)?;
                        self.code.push(match prim_func {
                            Some(prim) => Instr::CallPrimFunc2 { prim },
                            None => Instr::Call2,
                        });
                    }
                }

                if !keep { self.code.push(Instr::Pop) }
            }

            Predicate::ForwardAssignment(pat) =>
                self.compile_unpacking_assignment(pat, keep)?,

            Predicate::If2(then, else_) => self.compile_if(&*then, &*else_, keep)?,
        }
        Ok(())
    }

    fn compile_if(&mut self, then: &Expr, else_: &Expr, keep: bool) -> Res<()> {
        // s: start;  t: #then;  e: #else     offset = target      - (index+1)
        //
        // s          JumpUnless(t+1) --,     t+1    = (s+t+2)     - (ip=s+1)
        // s+1        (then start)      |
        // s+t        (then end)        |
        // s+t+1      Jump(e+1) --------|--,  e+1    = (s+t+2+e+1) - (ip=s+t+2)
        // s+t+2      (else start) <----'  |
        // s+t+2+e    (else end)           |
        // s+t+2+e+1  (after) <------------'

        let jump_unless_index = self.push(Instr::Nop);

        // TODO keep local names that are defined in both branches? They'd need
        // to get the same slot.
        {
            let in_branch_locals_start = next_local_slot(self.get_local_scope());
            self.compile_expr(then, keep)?;
            self.cull_locals_at_and_above(in_branch_locals_start);
        }

        let jump_index = self.push(Instr::Nop);

        self.code[jump_unless_index] = Instr::JumpRelativeUnless {
            offset: self.code.len() as i64 - (jump_unless_index as i64 + 1)
        };

        {
            let in_branch_locals_start = next_local_slot(self.get_local_scope());
            self.compile_expr(else_, keep)?;
            self.cull_locals_at_and_above(in_branch_locals_start);
        }

        self.code[jump_index] = Instr::JumpRelative {
            offset: self.code.len() as i64 - (jump_index as i64 + 1)
        };

        Ok(())
    }

    fn fetch_name_in_current_scope(&mut self, name: &str) -> NameValue {
        if let Some(prim) = self.primitive_identifiers.get(name) {
            return NameValue::Prim(*prim)
        }

        let local_scope = self.get_local_scope_mut();
        if let Some(var) = local_scope.get(name) {
            return NameValue::Var(*var);
        }
        let var = local_var(next_local_slot(local_scope));
        local_scope.insert(name.to_string(), var);
        NameValue::Var(var)
    }

    // If name isn't defined in the current scope, updates all intervening
    // scopes to include it in their closure environment.
    fn fetch_name(&mut self, name: &str) -> Res<NameValue> {
        if let Some(prim) = self.primitive_identifiers.get(name) {
            return Ok(NameValue::Prim(*prim))
        }

        for i in (0 .. self.scopes.len()).rev() {
            if self.scopes[i].contains_key(name) {
                for j in (i+1)..self.scopes.len() {
                    add_closure_var(name, &mut self.scopes[j]);
                }
                break;
            }
        }

        if let Some(var) = self.get_local_scope().get(name) {
            return Ok(NameValue::Var(*var));
        }
        cold_err!("Undefined name: `{name}'")
    }

    fn compile_small_noun(&mut self, small_noun: &SmallNoun, keep: bool) -> Res<()> {
        // TODO currently V N pushes N but doesn't pop it unless another
        // expression follows (e.g. the program "+ 3" is push prim; pop verb; push literal)
        use SmallNoun::*;
        match small_noun {
            If3(cond, then, else_) => {
                self.compile_expr(&*cond, true)?;
                self.compile_if(&*then, &*else_, keep)?;
            }
            LowerName(name) => {
                match self.fetch_name(name)? {
                    NameValue::Prim(prim) => {
                        if prim == PrimFunc::Rec && self.scopes.len() < 2 {
                            return cold_err!("Can't use `rec` outside of an explicit definition")
                        }
                        if keep { self.code.push(Instr::PushPrimFunc { prim }) }
                    }
                    NameValue::Var(src) => if keep { self.code.push(Instr::PushVar { src }) }
                }
            }
            NounBlock(exprs, last) => {
                if !exprs.is_empty() {
                    self.compile_block(exprs, false)?;
                }
                self.compile_noun(&*last, keep)?;
            }
            Underscored(small_expr) => self.compile_small_expr(small_expr.as_ref(), keep)?,
            Constant(Literal::Int(int)) => if keep { self.code.push(Instr::PushLiteralInteger(*int)) }
            Constant(Literal::Float(float)) => if keep { self.code.push(Instr::PushLiteralFloat(*float)) }
            Constant(Literal::Char(byte)) => if keep {
                let mut bytes = [0; 8];
                bytes[0] = *byte;
                self.code.push(Instr::LiteralBytes { bytes });
            }
            Constant(Literal::String(s)) => if keep {
                self.code.push(Instr::MakeString { num_bytes: s.len() });
                for i in (0..s.len()).step_by(8) {
                    let mut bytes = [0; 8];
                    for j in i..(i+8).min(s.len()) {
                        bytes[j-i] = s.as_bytes()[j];
                    }
                    self.code.push(Instr::LiteralBytes { bytes });
                }
            }
            ArrayLiteral(elems) => {
                let num_missing_args =
                    elems.iter().filter(|elem| matches!(elem, Elem::MissingArg)).count();
                if num_missing_args == 0 {
                    self.code.push(Instr::MarkStack);
                    for elem in elems {
                        self.compile_elem(elem)?;
                    }
                    self.code.push(Instr::CollectMarkedToArray);
                } else {
                    for elem in elems {
                        self.compile_elem(elem)?;
                    }
                    let make_func_index = self.push(Instr::Nop);
                    self.push(Instr::ArgCheck { arg_spec: ArgSpec::saturated(num_missing_args as u64) });
                    for i in 0..num_missing_args {
                        self.code.push(Instr::StoreTo { dst: Var { place: Place::Local, slot: i }});
                    }

                    self.code.push(Instr::MarkStack);

                    let mut locals_used = 0;
                    let mut closure_vars_used = 0;
                    for elem in elems {
                        let (place, slot) = match elem {
                            Elem::MissingArg => (Place::Local, &mut locals_used),
                            Elem::Expr(_) | Elem::Spliced(_) => (Place::ClosureEnv, &mut closure_vars_used),
                        };
                        self.code.push(Instr::PushVar { src: Var { place, slot: *slot } });
                        if let Elem::Spliced(_) = elem { self.code.push(Instr::Splat); }
                        *slot += 1;
                    }

                    self.code.push(Instr::CollectMarkedToArray);
                    self.code.push(Instr::Return);
                    let num_instructions = self.code.len() - (make_func_index + 1);
                    self.code[make_func_index] = Instr::MakeFunc { num_instructions };
                    self.code.push(Instr::MakeClosureFromStack {
                        num_closure_vars: elems.len() - num_missing_args
                    });
                }
                if !keep { self.code.push(Instr::Pop) }
            }
            Indexed(small_noun, args) => {
                self.compile_small_noun(&*small_noun, true)?;
                for elem in args {
                    if matches!(elem, Elem::Spliced(_)) {
                        todo!("Support splices in argument lists.")
                    }
                    self.compile_elem(elem)?;
                }
                let arg_spec =
                    ArgSpec::new(args.iter().map(|arg| !matches!(arg, Elem::MissingArg)));
                match arg_spec {
                    Some(arg_spec) => self.code.push(Instr::CallSpec { arg_spec }),
                    None => return cold_err!(
                        "Can't call a function or index an array with more than 63 arguments."
                    ),
                }

                if !keep { self.code.push(Instr::Pop) }
            }
        }
        Ok(())
    }

    // Caller must handle Elem::MissingArg, which doesn't compile to anything.
    fn compile_elem(&mut self, elem: &Elem) -> Res<()> {
        match elem {
            Elem::MissingArg => Ok(()),
            Elem::Expr(expr) => self.compile_expr(expr, true),
            Elem::Spliced(small_expr) => {
                self.compile_small_expr(small_expr, true)?;
                self.code.push(Instr::Splat);
                Ok(())
            }
        }
    }

    fn compile_small_expr(&mut self, expr: &SmallExpr, keep: bool) -> Res<()> {
        match expr {
            SmallExpr::Verb(small_verb) => self.compile_small_verb(small_verb, None, keep),
            SmallExpr::Noun(small_noun) => self.compile_small_noun(small_noun, keep),
        }
    }

    fn fill_with_nop(&mut self, range: std::ops::Range<usize>) {
        for i in range { self.code[i] = Instr::Nop }
    }

    fn cull_locals_at_and_above(&mut self, too_high: usize) {
        let scope = self.scopes.last_mut().unwrap();
        scope.retain(|_, var| var.place == Place::ClosureEnv || var.slot < too_high);
    }

    // Mark the last uses of locals in the function that was just compiled. This starts at the end
    // of self.code and works backwards.
    // 
    // We may need to take special care to make sure this works if we add imperative loops.
    fn mark_last_local_uses(&mut self, make_func_index: usize, num_local_vars_in_scope: usize) {
        // We start at the end of a function and go backwards, marking the first use we encounter of
        // each local. The bodies of inner function definitions have already been processed, so we
        // need to know where their ends are and how far back to go to skip to their start.
        struct Span {
            start: usize,  // Index of MakeFunc
            end: usize,    // Index of instruction just after Return
        }
        let mut inner_definition_spans = Vec::new();
        let mut i = make_func_index + 1;
        while i < self.code.len() {
            if let Instr::MakeFunc { num_instructions } = self.code[i] {
                inner_definition_spans.push(Span { start: i, end: i + num_instructions + 1 });
                i += num_instructions;  // This points i at Return.
            }
            i += 1;
        }

        // True if we've already enountered the last use of a local (remember that we go backwards,
        // so the last use is actually the first time we see it).
        //
        // This may need to grow beyond its initial length. `num_local_vars_in_scope' denotes the
        // number of active locals at the function's end, but variables can enter and exit scope
        // over the course of a function (e.g. due to an `if` branch), resulting in more locals to
        // account for somewhere in the middle than at the end.
        let mut last_use_seen = vec![false; num_local_vars_in_scope];

        // Returns the old value.
        fn set(v: &mut Vec<bool>, slot: usize, val: bool) -> bool {
            if slot >= v.len() {
                v.resize(slot + 1, false);
            }
            std::mem::replace(&mut v[slot], val)
        }

        // Returns true if it's the first time we've seen this local.
        fn first_encounter(v: &mut Vec<bool>, slot: usize) -> bool {
            !set(v, slot, true)
        }

        // The last element is the list of slots last seen in the `else` alternative of this `then`
        // branch.
        let mut else_branch_last_uses: Vec<Vec<usize>> = vec![];

        let mut i = self.code.len() - 1;
        while i > make_func_index {
            match self.code[i] {
                Instr::StoreTo { dst } if dst.is_local() => { set(&mut last_use_seen, dst.slot, false); }

                Instr::PushVar { src } if src.is_local() && first_encounter(&mut last_use_seen, src.slot) =>
                    self.code[i] = Instr::PushVarLastUse { src },

                Instr::TuckVar { src } if src.is_local() && first_encounter(&mut last_use_seen, src.slot) =>
                    self.code[i] = Instr::TuckVarLastUse { src },

                // There's not currently a LastUse variant for CallOnArgs.
                Instr::CallOnArgs { var } if var.is_local() => { set(&mut last_use_seen, var.slot, true); }

                Instr::JumpRelative { offset } if offset > 0 => {
                    // Scan the `else` branch we just went through, record the slots last seen in
                    // it, and unsee them so we can mark last uses in the `then` branch.
                    let mut last_uses_in_else = vec![];
                    for j in (i+1)..(i+offset as usize) {
                        if let Instr::PushVarLastUse { src: Var { slot, .. } } = self.code[j] {
                            last_use_seen[slot] = false;
                            last_uses_in_else.push(slot);
                        }
                    }
                    else_branch_last_uses.push(last_uses_in_else);
                }

                Instr::JumpRelativeUnless { offset } if offset > 0 => {
                    // Now that we're exiting the `then` branch, re-see all the slots last used in
                    // the `else` branch. Some of them might have been seen in the `then`, but
                    // that's not necessarily true.
                    let last_uses_in_else = else_branch_last_uses.pop().unwrap();
                    for slot in last_uses_in_else {
                        last_use_seen[slot] = true;
                    }
                }

                _ => {}
            }

            // If an inner function definition ends here, skip its body. (TODO consider rearranging
            // this loop into nested loops so we don't need to check this every iteration).
            if let Some(&Span{start, end}) = inner_definition_spans.last() {
                if i == end {
                    i = start;  // i now points at MakeFunc
                    inner_definition_spans.pop();
                }
            }

            i -= 1;
        }
    }
}

trait TryIntoPrimFunc {
    // Sometimes, things can be replaced with primitives if:
    //   - We know the call arity of a function in advance
    //   - An expression is the result of a recognized pattern (e.g., adverb application)
    //
    // Returns
    //   Ok(Some(prim)) if `self` at `arity` is equivalent to `prim`.
    //   Ok(None) if `self` doesn't resolve to a smaller primitive and should be compiled as `self`.
    //   Err if `self` can't possibly be applied at arity `arity`.
    fn into_prim_func(
        &self, arity: Option<u32>, prim_identifiers: &HashMap<&str, PrimFunc>
    ) -> Res<Option<PrimFunc>>;
}

impl TryIntoPrimFunc for Verb {
    fn into_prim_func(
        &self, arity: Option<u32>, prim_identifiers: &HashMap<&str, PrimFunc>
    ) -> Res<Option<PrimFunc>> {
        match self {
            Verb::SmallVerb(small_verb) => small_verb.into_prim_func(arity, prim_identifiers),
            _ => Ok(None),
        }
    }
}

impl TryIntoPrimFunc for SmallVerb {
    fn into_prim_func(
        &self, arity: Option<u32>, prim_identifiers: &HashMap<&str, PrimFunc>
    ) -> Res<Option<PrimFunc>> {
        use PrimFunc::*;
        let prim = match self {
            &SmallVerb::PrimVerb(Verb(prim)) => match arity {
                Some(arity) => Some(PrimFunc::resolve_at_arity(prim, arity)?),
                None => Some(PrimFunc::Verb(prim)),
            }
            &SmallVerb::PrimVerb(prim_func) => Some(prim_func),
            SmallVerb::PrimAdverbCall(PrimAdverb::Backslash, expr_box) =>
                match expr_box.into_prim_func(Some(2), prim_identifiers)? {
                    Some(PrimFunc::Add) => Some(PrimFunc::Sum),
                    // TODO instead of always resolving to a PrimFunc, allow resolving to SmallVerb
                    // so we can match Some(small_verb) here: knowing arity can specialize
                    // explicits, too.
                    _ => None,
                }
            SmallVerb::PrimAdverbCall(PrimAdverb::Dot, expr_box) =>
                expr_box.into_prim_func(arity, prim_identifiers)?,
            SmallVerb::UpperName(name) =>
                prim_identifiers.get(name.as_str()).copied(),
            _ => None,
        };
        Ok(prim)
    }
}

impl TryIntoPrimFunc for SmallNoun {
    fn into_prim_func(
        &self, arity: Option<u32>, prim_identifiers: &HashMap<&str, PrimFunc>
    ) -> Res<Option<PrimFunc>> {
        match self {
            SmallNoun::LowerName(name) => Ok(prim_identifiers.get(name.as_str()).copied()),
            SmallNoun::Underscored(expr_box) => expr_box.into_prim_func(arity, prim_identifiers),
            _ => Ok(None),
        }
    }
}

impl TryIntoPrimFunc for SmallExpr {
    fn into_prim_func(
        &self, arity: Option<u32>, prim_identifiers: &HashMap<&str, PrimFunc>
    ) -> Res<Option<PrimFunc>> {
        match self {
            SmallExpr::Noun(small_noun) => small_noun.into_prim_func(arity, prim_identifiers),
            SmallExpr::Verb(small_verb) => small_verb.into_prim_func(arity, prim_identifiers),
        }
    }
}

fn local_var(slot: usize) -> Var {
    Var { place: Place::Local, slot }
}

fn closure_var(slot: usize) -> Var {
    Var { place: Place::ClosureEnv, slot }
}

// Note that the next local slot isn't always the same as the number of local slots in the scope;
// loading a module can add a var to the scope that has the same name but a different slot number,
// shadowing the current local.
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
// After `a: 5` executes, the scope is {"a": Local(0)}. Then `mod` is loaded; inside, `a` is local
// slot 0, but when it's imported to the program, it's offset by 1 to put it after the current
// locals. The scope is {"a": Local(1)}. The next local slot would be 2, even though there's only
// one local in scope.
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

fn pattern_elem_to_elem(pat_elem: &PatternElem) -> Res<Elem> {
    Ok(match pat_elem {
        PatternElem::Pattern(pat) => Elem::Expr(Expr::Noun(Noun::SmallNoun(pattern_to_small_noun(pat)?))),
        PatternElem::Subarray(Some(name)) => Elem::Spliced(SmallExpr::Noun(SmallNoun::LowerName(name.clone()))),
        PatternElem::Subarray(None) => return cold_err!("Unnamed `..' wildcards can't be converted to expressions."),
    })
}

fn pattern_to_small_noun(pat: &Pattern) -> Res<SmallNoun> {
    Ok(match pat {
        Pattern::Constant(literal) => SmallNoun::Constant(literal.clone()),
        Pattern::As(pat1, _) => pattern_to_small_noun(pat1)?,
        Pattern::Name(name) => SmallNoun::LowerName(name.clone()),
        Pattern::Array(pat_elems) => SmallNoun::ArrayLiteral(
            pat_elems.iter()
                .map(|pat_elem| pattern_elem_to_elem(pat_elem))
                .collect::<Res<Vec<_>>>()?
        ),
        // TODO lens?
        Pattern::View(_, _) => return cold_err!("View patterns can't be converted to expressions."),
        Pattern::Wildcard => return cold_err!("`_' wildcards can't be converted to expressions."),
    })
}

fn get_prim_adverb_operand_arity(adverb: PrimAdverb, derived_verb_arity: Option<u32>) -> Option<u32> {
    use PrimAdverb::*;
    match adverb {
        Underscore {..} => None,  // TODO is this right?
        AtColon => Some(1),
        Tilde | Backslash | BackslashColon | Runs => Some(2),
        Dot | SingleQuote | Backtick | BacktickColon | P | Q => derived_verb_arity,
    }
}

fn accessed(code: &[Instr], var: Var) -> bool {
    let mut i = 0;
    while i < code.len() {
        match code[i] {
            Instr::PushVar { src } | Instr::PushVarLastUse { src } |
            Instr::TuckVar { src } | Instr::TuckVarLastUse { src } if src == var => return true,

            Instr::CallOnArgs { var: called } if called == var => return true,

            Instr::StoreTo { dst } if dst == var => return true,

            Instr::MakeFunc { num_instructions } => i += num_instructions,
            _ => {}
        }
        i += 1;
    }
    false
}

fn decrement_locals(code: &mut [Instr], above_slot: usize) {
    use Place::*;
    let mut i = 0;
    while i < code.len() {
        match &mut code[i] {
            Instr::PushVar        { src: Var { place: Local, slot } } |
            Instr::PushVarLastUse { src: Var { place: Local, slot } } |
            Instr::TuckVar        { src: Var { place: Local, slot } } |
            Instr::TuckVarLastUse { src: Var { place: Local, slot } } |
            Instr::CallOnArgs     { var: Var { place: Local, slot } } |
            Instr::StoreTo        { dst: Var { place: Local, slot } } if *slot > above_slot => *slot -= 1,

            Instr::MakeFunc { num_instructions } => i += *num_instructions,
            _ => {}
        }
        i += 1;
    }
}

fn make_primitive_identifier_map() -> HashMap<&'static str, PrimFunc> {
    use PrimFunc::*;
    HashMap::from([
        ("rec", Rec),
        ("ints", Ints),
        ("rev", Rev),
        ("where", Where),
        ("nub", Nub),
        ("identity", Identity),
        ("asc", Asc),
        ("desc", Desc),
        ("sort", Sort),
        ("sortDesc", SortDesc),
        ("inits", Inits),
        ("tails", Tails),
        ("not", Not),
        ("ravel", Ravel),
        ("floor", Floor),
        ("ceil", Ceil),
        ("length", Length),
        ("exit", Exit),
        ("show", Show),
        ("print", Print),
        ("getLine", GetLine),
        ("readFile", ReadFile),
        ("rand", Rand),
        ("type", Type),
        ("printBytecode", PrintBytecode),
        ("take", Take),
        ("drop", Drop),
        ("remove", Remove),
        ("rot", Rot),
        ("find", Find),
        ("findAll", FindAll),
        ("findSubseq", FindSubseq),
        ("has", Has),
        ("in", In),
        ("copy", Copy),
        ("identityLeft", IdentityLeft),
        ("identityRight", IdentityRight),
        ("shiftLeft", ShiftLeft),
        ("shiftRight", ShiftRight),
        ("and", And),
        ("or", Or),
        ("equal", Equal),
        ("notEqual", NotEqual),
        ("match", Match),
        ("notMatch", NotMatch),
        ("greaterThan", GreaterThan),
        ("greaterThanEqual", GreaterThanEqual),
        ("lessThan", LessThan),
        ("lessThanEqual", LessThanEqual),
        ("index", Index),
        ("pick", Pick),
        ("append", Append),
        ("cons", Cons),
        ("snoc", Snoc),
        ("cut", Cut),
        ("add", Add),
        ("sub", Sub),
        ("neg", Neg),
        ("abs", Abs),
        ("mul", Mul),
        ("div", Div),
        ("intDiv", IntDiv),
        ("mod", Mod),
        ("pow", Pow),
        ("log", Log),
        ("min", Min),
        ("max", Max),
        ("windows", Windows),
        ("chunks", Chunks),
        ("groupBy", GroupBy),
        ("groupIndices", GroupIndices),
        ("sendToIndex", SendToIndex),
        ("runs", Runs),
        ("or", Or),
    ])
}
