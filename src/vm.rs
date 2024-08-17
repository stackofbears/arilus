use std::{
    cell::RefCell,
    cmp::Ordering,
    mem,
    rc::Rc,
    fmt::Write,
};

use crate::bytecode::*;
use crate::lex::*;
use crate::prim;
use crate::util::*;
use crate::val::*;

// When looking for a tail call, we'll either tell `Mem::execute` to jump to a
// code index, or we'll simply call a function without eliminating the tail call
// (if the function is a primitive, for example) and return the value to be
// pushed.
enum ChasedTail {
    GoTo(usize),
    Push(Val),
}

// Used in val printing.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum PrecedenceContext {
    Toplevel,

    // Comments are cumulative; if parentheses are needed for one variant,
    // they're also needed for the variants that follow.

    // Parentheses needed for trains.
    Small,

    // Parentheses needed for stranded lists.
    AdverbOperand,

    // Parentheses needed for adverb and conjunction applications.
    ConjunctionLeftOperand,
}

#[derive(Debug)]
struct StackFrame {
    // The first instruction of this function.
    code_index: usize,

    closure_env: Rc<RefCell<Vec<Val>>>,

    // Index into locals_stack. Local slots are offsets from this index.
    locals_start: usize,
}

pub struct Mem {
    pub code: Vec<Instr>,

    pub stack: Vec<Val>,

    // Stack of local scopes
    locals_stack: Vec<Val>,

    // Details about the explicit function (or global scope) we're currently
    // in. Never empty!
    stack_frames: Vec<StackFrame>,
}

impl Mem {
    pub fn new() -> Self {
        Self {
            code: vec![],
            stack: vec![],
            locals_stack: vec![],  // TODO stdlib?
            stack_frames: vec![StackFrame {
                closure_env: Rc::new(RefCell::new(vec![])),
                locals_start: 0,
                code_index: usize::MAX,
            }],
        }
    }

    pub fn execute_from_toplevel(&mut self, ip: usize) -> Result<(), String> {
        let result = self.execute(ip);
        if result.is_err() {
            self.stack.clear();
            let mut leftover_frames = self.stack_frames.drain(1..);
            if let Some(second_frame) = leftover_frames.next() {
                self.locals_stack.truncate(second_frame.locals_start)
            }
        }
        result
    }

    fn execute(&mut self, mut ip: usize) -> Result<(), String> {
        use Instr::*;

        while ip < self.code.len() {
            ip += 1;
            match self.code[ip - 1] {
                Nop => {}
                LoadModule { code_index } => {
                    let top_of_locals = self.locals_stack.len();
                    let frame = self.current_frame_mut();
                    let old_locals_start = frame.locals_start;
                    frame.locals_start = top_of_locals;
                    self.execute(code_index)?;
                    self.current_frame_mut().locals_start = old_locals_start;
                }
                ModuleStart { num_instructions } => {
                    ip += num_instructions;
                }
                ModuleEnd => return Ok(()),
                Dup => self.push(self.stack.last().unwrap().clone()),
                MakeClosure{..} => panic!("Malformed code at ip {}: reached MakeClosure not immediately following a MakeFunc's Return.", ip - 1),
                MakeFunc { num_instructions } => {
                    let code_index = ip;
                    ip += num_instructions;

                    let closure_data = match self.code.get(ip) {
                        Some(MakeClosure { num_closure_vars }) => {
                            ip += 1;
                            let mut closure_data = Vec::with_capacity(*num_closure_vars);
                            for _ in 0..*num_closure_vars {
                                match self.code[ip] {
                                    PushVar { src } => {
                                        ip += 1;
                                        // TODO use e.g. List::I64s if closure vals are all ints
                                        closure_data.push(self.load(src));
                                    }
                                    PushVarLastUse { src } => {
                                        ip += 1;
                                        closure_data.push(self.consuming_load(src));
                                    }
                                    _ => panic!("Malformed code at ip {ip}: expected PushVar after MakeClosure, but found {:?}",
                                                self.code[ip]),
                                }
                            }
                            closure_data
                        }
                        _ => vec![],
                    };

                    self.push(Val::Function(Rc::new(Func::Explicit {
                        code_index,
                        closure_env: Rc::new(RefCell::new(closure_data)),
                    })));
                }
                Return => {
                    let frame = self.stack_frames.pop().unwrap();
                    self.locals_stack.truncate(frame.locals_start);
                    return Ok(())
                }
                JumpRelative { offset } => ip = (ip as i64 + offset) as usize,
                JumpRelativeUnless { offset } =>
                    if self.pop().is_falsy() { ip = (ip as i64 + offset) as usize },
                PushLiteralInteger(value) => self.push(Val::Int(value)),
                PushLiteralFloat(value) => self.push(Val::Float(value)),
                PushVar { src } => {
                    let val = self.load(src);
                    self.push(val);
                }
                PushVarLastUse { src } => {
                    let val = self.consuming_load(src);
                    self.push(val);
                }
                PushPrimFunc { prim: PrimFunc::Rec } => {
                    let frame = self.current_frame();
                    // TODO can we copy the actual function that was called instead?
                    self.push(Val::Function(Rc::new(Func::Explicit {
                        code_index: frame.code_index,
                        closure_env: frame.closure_env.clone(),
                    })));
                }
                PushPrimFunc { prim } => self.push(Val::Function(Rc::new(Func::Prim(prim)))),

                // TODO eliminate tail calls for call instructions.
                Call1 => {
                    let func = self.pop();
                    let x = self.pop();

                    if self.is_tail_call(ip) {
                        match self.chase_tail(func, x, None)? {
                            ChasedTail::GoTo(code_index) => ip = code_index,
                            ChasedTail::Push(result) => self.push(result),
                        }
                    } else {
                        let result = self.call_val(func, x, None)?;
                        self.push(result);
                    }
                }
                Call2 => {
                    let y = self.pop();
                    let func = self.pop();
                    let x = self.pop();

                    if self.is_tail_call(ip) {
                        match self.chase_tail(func, x, Some(y))? {
                            ChasedTail::GoTo(code_index) => ip = code_index,
                            ChasedTail::Push(result) => self.push(result),
                        }
                    } else {
                        let result = self.call_val(func, x, Some(y))?;
                        self.push(result);
                    }
                }
                CallN { num_args } => {
                    // TODO - drain to locals stack?
                    let args = self.stack.drain((self.stack.len() - num_args)..);
                    self.locals_stack.extend(args);
                    let f = self.pop();
                    let result = self.index_or_call_from_locals(f, num_args)?;
                    self.push(result);
                }
                CallPrimFunc1 { prim } => {
                    let x = self.pop();
                    let result = self.call_prim_monad(prim, x)?;
                    self.push(result);
                }
                CallPrimFunc2 { prim } => {
                    let y = self.pop();
                    let x = self.pop();
                    if prim == PrimFunc::Verb(PrimVerb::At) && x.is_func() && self.is_tail_call(ip) {
                        match self.chase_tail(x, y, None)? {
                            ChasedTail::GoTo(code_index) => ip = code_index,
                            ChasedTail::Push(result) => self.push(result),
                        }
                    } else {
                        let result = self.call_prim_dyad(prim, x, y)?;
                        self.push(result);
                    }
                }
                Pop => { self.pop(); }
                StoreTo { dst } => self.store(dst, self.stack.last().unwrap().clone()),
                Splat { count } => {
                    // TODO repetition
                    match self.pop().as_val() {
                        a@atom!() => return cold_err!("Array unpacking failed; expected {count} elements, got atom {:?}", a),
                        Val::U8s(cs) => {
                            if cs.len() != count {
                                return cold_err!("Array unpacking failed; expected {count} elements, got {}", cs.len())
                            }
                            self.stack.extend(cs.iter().rev().map(|c| Val::Char(*c)))
                        }
                        Val::I64s(is) => {
                            if is.len() != count {
                                return cold_err!("Array unpacking failed; expected {count} elements, got {}", is.len())
                            }
                            self.stack.extend(is.iter().rev().map(|i| Val::Int(*i)))
                        }
                        Val::F64s(fs) =>  {
                            if fs.len() != count {
                                return cold_err!("Array unpacking failed; expected {count} elements, got {}", fs.len())
                            }
                            self.stack.extend(fs.iter().rev().map(|f| Val::Float(*f)))
                        }
                        Val::Vals(vs) => {
                            if vs.len() != count {
                                return cold_err!("Array unpacking failed; expected {count} elements, got {}", vs.len())
                            }
                            self.stack.extend(vs.iter().rev().map(|v| v.clone()))
                        }
                    }
                }
                CallPrimAdverb { prim: adverb } => {
                    let operand = self.pop();

                    fn apply_adverb(adverb: PrimAdverb, operand: Val) -> Val {
                        if adverb == PrimAdverb::P {
                            if let Val::Function(rc) = &operand {
                                if let Func::Ambivalent(monad, _) = &**rc {
                                    return monad.clone()
                                }
                            }
                        }
                        Val::Function(Rc::new(Func::AdverbDerived { adverb, operand }))
                    }

                    self.push(apply_adverb(adverb, operand));
                }
                CollectVerbAlternatives => {
                    fn get_dyad_case(val: Val) -> Val {
                        if let Val::Function(rc) = &val {
                            if let Func::Ambivalent(_, inner_dyad) = &**rc {
                                return inner_dyad.clone()
                            }
                        }
                        val
                    }
                    let dyad_case = get_dyad_case(self.pop());

                    fn get_monad_case(val: Val) -> Val {
                        if let Val::Function(rc) = &val {
                            if let Func::Ambivalent(inner_monad, _) = &**rc {
                                return inner_monad.clone()
                            }
                        }
                        val
                    }
                    let monad_case = get_monad_case(self.pop());

                    self.push(Val::Function(Rc::new(Func::Ambivalent(monad_case, dyad_case))));
                }
                MakeAtopFunc => {
                    let g_func = self.pop();
                    let f_func = self.pop();
                    self.push(Val::Function(Rc::new(Func::Atop { f_func, g_func })));
                }
                MakeBoundFunc => {
                    let y = self.pop();
                    let func = self.pop();
                    self.push(Val::Function(Rc::new(Func::Bound { func, y })));
                }
                MakeForkFunc => {
                    let g_func = self.pop();
                    let h_func = self.pop();
                    let f_func = self.pop();
                    self.push(Val::Function(Rc::new(Func::Fork { f_func, h_func, g_func })));
                }
                MakeString { num_bytes } => {
                    let mut s = Vec::with_capacity(num_bytes);
                    for i in (0..num_bytes).step_by(8) {
                        irrefutable!(self.code[ip], LiteralBytes { bytes } => {
                            ip += 1;
                            for j in i..(i+8).min(num_bytes) {
                                s.push(bytes[j-i]);
                            }
                        });
                    }
                    self.push(Val::U8s(Rc::new(s)));
                }
                LiteralBytes { bytes } => self.push(Val::Char(bytes[0])),
                CollectToArray { num_elems } => {
                    let mut all_chars = true;
                    let mut all_ints = true;
                    let mut all_floats = true;
                    for elem in &self.stack[(self.stack.len() - num_elems)..] {
                        all_chars &= matches!(elem, Val::Char(_));
                        all_ints &= matches!(elem, Val::Int(_));
                        all_floats &= matches!(elem, Val::Float(_) | Val::Int(_));
                    }

                    let elems = self.stack.drain((self.stack.len() - num_elems)..);
                    let list_val = if all_chars {
                        Val::U8s(Rc::new(map(elems, |elem| irrefutable!(elem, Val::Char(ch) => ch))))
                    } else if all_ints {
                        Val::I64s(Rc::new(map(elems, |elem| irrefutable!(elem, Val::Int(int) => int))))
                    } else if all_floats {
                        Val::F64s(Rc::new(map(elems, |elem| match elem {
                            Val::Float(f) => f,
                            Val::Int(i) => i as f64,
                            _ => unreachable!(),
                        })))
                    } else {
                        Val::Vals(Rc::new(elems.collect()))
                    };
                    self.push(list_val)
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn push(&mut self, val: Val) {
        self.stack.push(val)
    }

    #[inline]
    fn pop(&mut self) -> Val {
        self.stack.pop().unwrap()
    }

    #[inline]
    fn current_frame(&self) -> &StackFrame {
        self.stack_frames.last().unwrap()
    }

    #[inline]
    fn current_frame_mut(&mut self) -> &mut StackFrame {
        self.stack_frames.last_mut().unwrap()
    }

    fn load(&mut self, var: Var) -> Val {
        // TODO do all local/closure vars point to non-lists (instead to slices)?
        let frame = self.current_frame();
        match var.place {
            Place::Local => self.locals_stack[frame.locals_start + var.slot].clone(),
            Place::ClosureEnv => frame.closure_env.borrow()[var.slot].clone(),
        }
    }

    fn consuming_load(&mut self, var: Var) -> Val {
        let mut ret = Val::Int(0);  // TODO
        let frame = self.current_frame_mut();
        match var.place {
            Place::Local => {
                let absolute_slot = frame.locals_start + var.slot;
                mem::swap(&mut ret, &mut self.locals_stack[absolute_slot])
            }
            Place::ClosureEnv =>
                mem::swap(&mut ret, &mut frame.closure_env.borrow_mut()[var.slot]),
        }
        ret
    }

    fn store(&mut self, dst: Var, val: Val) {
        let frame = self.current_frame();
        match dst.place {
            Place::Local => {
                let absolute_slot = frame.locals_start + dst.slot;
                if absolute_slot >= self.locals_stack.len() {
                    self.locals_stack.resize(absolute_slot + 1, Val::Int(0));  // TODO is zero right?
                }
                self.locals_stack[absolute_slot] = val;
            }
            Place::ClosureEnv => frame.closure_env.borrow_mut()[dst.slot] = val,
        }
    }

    // `ip` should point to the next instruction to execute.
    #[inline]
    fn is_tail_call(&self, ip: usize) -> bool {
        matches!(self.code.get(ip), Some(Instr::Return))
    }

    fn chase_tail(&mut self, mut func: Val, mut x: Val, mut y: Option<Val>) -> Result<ChasedTail, String> {
        loop {
            let function = match &func {
                Val::Function(rc) => &**rc,
                _ => return Ok(ChasedTail::Push(self.call_val(func, x, y)?)),
            };

            match function {
                Func::Explicit{code_index, closure_env} => {
                    let frame = self.current_frame_mut();
                    frame.code_index = *code_index;
                    frame.closure_env = closure_env.clone();
                    let locals_start = frame.locals_start;
                    self.locals_stack.truncate(locals_start);
                    self.locals_stack.push(x);
                    self.locals_stack.push(y.unwrap_or_else(|| Val::Int(0)));
                    return Ok(ChasedTail::GoTo(*code_index));
                }
                Func::Ambivalent(monad, dyad) =>
                    func = if y.is_none() { monad.clone() } else { dyad.clone() },
                Func::Atop { f_func, g_func } => {
                    x = self.call_val(f_func.clone(), x, None)?;
                    func = g_func.clone();
                }
                Func::Bound { func: bound_func, y: bound_y } => {
                    y = Some(bound_y.clone());
                    func = bound_func.clone();
                }
                Func::Fork { f_func, h_func, g_func } => {
                    let new_x = self.call_val(f_func.clone(), x.clone(), y.clone())?;
                    let new_y = Some(self.call_val(g_func.clone(), x, y)?);
                    x = new_x;
                    y = new_y;
                    func = h_func.clone();
                }
                Func::Prim(PrimFunc::Verb(PrimVerb::At)) if x.is_func() && y.is_some() => {
                    func = x;
                    x = y.unwrap();
                    y = None;
                }
                Func::AdverbDerived { adverb: PrimAdverb::P, operand } => {
                    func = operand.clone();
                    y = None;
                }
                Func::AdverbDerived { adverb: PrimAdverb::Q, operand } => {
                    func = operand.clone();
                    if let Some(y_val) = y.take() {
                        x = y_val;
                    }
                }
                Func::AdverbDerived { adverb: PrimAdverb::AtColon, operand } => {
                    let index = self.call_val(operand.clone(), x.clone(), None)?;
                    func = self.prim_index(&y.expect("TODO: monadic @:v"), &index)?;
                    y = None;
                }
                Func::AdverbDerived { adverb: PrimAdverb::Dot, operand } => func = operand.clone(),
                Func::AdverbDerived { adverb: PrimAdverb::Tilde, operand } => {
                    match &mut y {
                        Some(y_val) => std::mem::swap(&mut x, y_val),
                        None => y = Some(x.clone()),
                    }
                    func = operand.clone();
                }
                Func::AdverbDerived { adverb: PrimAdverb::Backslash, operand } => {
                    match y {
                        Some(y_val) if x.len().unwrap_or(1) == 1 => {
                            func = operand.clone();
                            // Remember, the fold seed becomes the first x argument to the operand.
                            y = Some(index_or_cycle_val(&x, 0).unwrap());
                            x = y_val;
                        }
                        None if x.len().unwrap_or(2) == 2 => {
                            func = operand.clone();
                            y = Some(index_or_cycle_val(&x, 1).unwrap());
                            x = index_or_cycle_val(&x, 0).unwrap();
                        }
                        _ => return Ok(ChasedTail::Push(self.fold_val(operand.clone(), x, y)?)),
                    }
                }
                _ => return Ok(ChasedTail::Push(self.call_val(func, x, y)?)),
            }
        }
    }

    // Calls `val` with arguments `x` and, if present, `y`. If `val` is a
    // function, run the function; otherwise, `val` is treated as a constant
    // function and this call results in `val`.
    fn call_val(&mut self, val: Val, x: Val, y: Option<Val>) -> Result<Val, String> {
        let result = match val.as_val() {
            Val::Char(_) | Val::Int(_) | Val::Float(_) |
            Val::U8s(_) | Val::I64s(_) | Val::F64s(_) | Val::Vals(_) => val,
            Val::Function(func) => match &**func {
                &Func::Explicit { ref closure_env, code_index } => {
                    let frame = StackFrame {
                        code_index,
                        closure_env: closure_env.clone(),
                        locals_start: self.locals_stack.len(),
                    };
                    self.stack_frames.push(frame);
                    self.locals_stack.push(x);

                    // TODO functions can access this y when they shouldn't be able to
                    //
                    //     F:{y}
                    //     []F
                    //   0
                    //
                    // functions that mention y should always be called dyadically?
                    self.locals_stack.push(y.unwrap_or(Val::Int(0)));
                    self.execute(code_index)?;
                    self.pop()
                }
                Func::AdverbDerived { adverb, operand } =>
                    self.call_prim_adverb(*adverb, operand.clone(), x, y)?,
                Func::Ambivalent(monad, dyad) =>
                    self.call_val(if y.is_some() { dyad } else { monad }.clone(), x, y)?,
                Func::Atop { f_func, g_func } => {
                    let f_result = self.call_val(f_func.clone(), x, y)?;
                    self.call_val(g_func.clone(), f_result, None)?
                }
                Func::Bound { func, y } => self.call_val(func.clone(), x, Some(y.clone()))?,
                Func::Fork { f_func, h_func, g_func } => {
                    let f_result = self.call_val(f_func.clone(), x.clone(), y.clone())?;
                    let g_result = self.call_val(g_func.clone(), x, y)?;
                    self.call_val(h_func.clone(), f_result, Some(g_result))?
                }
                &Func::Prim(prim) => if let Some(y) = y {
                    self.call_prim_dyad(prim, x, y)?
                } else {
                    self.call_prim_monad(prim, x)?
                },
            },
        };
        Ok(result)
    }

    fn call_prim_adverb(&mut self,
                        adverb: PrimAdverb,
                        operand: Val,
                        x: Val,
                        maybe_y: Option<Val>) -> Result<Val, String> {
        use PrimAdverb::*;
        let result = match adverb {
            AtColon => {
                let index = self.call_val(operand, x.clone(), None)?;
                let elem = self.prim_index(&maybe_y.expect("TODO: monadic @:v"), &index)?;
                self.call_val(elem, x, None)?
            }
            Dot => self.call_val(operand, x, maybe_y)?,
            P => self.call_val(operand, x, None)?,
            Q => match maybe_y {
                Some(y) => self.call_val(operand, y, None)?,
                None => self.call_val(operand, x, None)?,
            },
            SingleQuote => match maybe_y {
                None => for_each(self, operand, x)?,
                Some(y) => match zip_vals(&x, &y) {
                    None => self.call_val(operand, x, Some(y))?,
                    Some(iter) => collect_list(
                        iter?.map(|(x_val, y_val)| self.call_val(operand.clone(), x_val, Some(y_val)))
                    )?,
                }
            }
            Backtick => match maybe_y {
                None => for_each(self, operand, x)?,
                Some(y) => match iter_val(&x) {
                    None => self.call_val(operand.clone(), x, Some(y))?,
                    Some(iter) => collect_list(
                        iter.map(|x_val| self.call_val(operand.clone(), x_val, Some(y.clone())))
                    )?,
                }
            }
            BacktickColon => match maybe_y {
                None => match iter_val(&x) {
                    None => self.call_val(operand, x, None)?,
                    Some(iter) => collect_list(
                        iter.map(|val| self.call_val(operand.clone(), val, None))
                    )?,
                }
                Some(y) => match iter_val(&y) {
                    None => self.call_val(operand.clone(), x, Some(y))?,
                    Some(iter) => collect_list(
                        iter.map(|y_val| self.call_val(operand.clone(), x.clone(), Some(y_val)))
                    )?,
                }
            }
            Tilde => match maybe_y {
                None => self.call_val(operand, x.clone(), Some(x))?,
                Some(y) => self.call_val(operand, y, Some(x))?,
            }
            Backslash => self.fold_val(operand, x, maybe_y)?,
        };
        Ok(result)
    }

    fn call_prim_monad(&mut self, v: PrimFunc, x: Val) -> Result<Val, String> {
        use PrimFunc::*;
        let result = match v {
            IdentityLeft | Verb(PrimVerb::P) | IdentityRight | Verb(PrimVerb::Q) => Ok(x),
            Neg | Verb(PrimVerb::Minus) => prim_negate(x),
            Show => prim_show(x),
            GetLine => prim_get_line(),
            Print => self.prim_to_string(&x)
                .inspect(|s| println!("{s}"))
                .map(|_| x),
            DebugPrint => self.prim_to_debug_string(&x)
                .inspect(|s| println!("{s}"))
                .map(|_| x),
            PrintBytecode => self.prim_print_bytecode(x.as_val()).map(|_| x),
            ReadFile => prim_read_file(x.as_val()),
            Length | Verb(PrimVerb::Hash) => Ok(Val::Int(x.len().unwrap_or(1) as i64)),
            Ints | Verb(PrimVerb::Slash) => Ok(iota(&x)),
            Rev | Verb(PrimVerb::Pipe) => Ok(prim_reverse(x)),
            Ravel | Verb(PrimVerb::Comma) => Ok(prim_ravel(&x)),
            Inits | Verb(PrimVerb::Caret) => Ok(Val::Vals(Rc::new(prim_prefixes(&x)))),
            Tails | Verb(PrimVerb::Dollar) => Ok(Val::Vals(Rc::new(prim_suffixes(&x)))),
            Where | Verb(PrimVerb::Question) => prim_where(&x),
            Sort | Verb(PrimVerb::LessThan) => Ok(prim_sort(x, false)),
            SortDesc | Verb(PrimVerb::GreaterThan) => Ok(prim_sort(x, true)),
            Asc | Verb(PrimVerb::LessThanColon) => Ok(prim_grade(&x, false)),
            Desc | Verb(PrimVerb::GreaterThanColon) => Ok(prim_grade(&x, true)),
            Type => Ok(Val::U8s(Rc::new(prim_type(x.as_val())))),
            GroupIndices => prim::group_indices(x),
            Exit => prim_exit(&x),

            Sum => prim::sum(x, None),

            _ => todo!("{x:?} {v:?}")
        };
        result.map_err(|err| cold(format!("Error in `{v}': {err}")))
    }

    fn call_prim_dyad(&mut self, v: PrimFunc, x: Val, y: Val) -> Result<Val, String> {
        use PrimFunc::*;
        let result = match v {
            IdentityLeft | Verb(PrimVerb::P) => Ok(x),
            IdentityRight | Verb(PrimVerb::Q) => Ok(y),
            Add | Verb(PrimVerb::Plus) => prim::add(x, y),
            Sub | Verb(PrimVerb::Minus) => prim::subtract(x, y),
            Mul | Verb(PrimVerb::Asterisk) => prim::multiply(x, y),
            Div | Verb(PrimVerb::Slash) => prim::divide(x, y),
            IntDiv | Verb(PrimVerb::DoubleSlash) => prim::int_divide(x, y),
            Mod | Verb(PrimVerb::Percent) => prim::int_mod(x, y),
            Pow | Verb(PrimVerb::Caret) => prim::pow(x, y),
            Take | Verb(PrimVerb::Hash) => prim_take(x, &y),
            Copy | Verb(PrimVerb::HashColon) => prim_copy(&x, &y),
            Append | Verb(PrimVerb::Comma) => prim_append(x, y),
            Match | Verb(PrimVerb::DoubleEquals) => prim_match(&x, &y),
            // TODO take Val instead of &
            Equal | Verb(PrimVerb::Equals) => prim_compare(&x, &y, |ord| ord == Ordering::Equal),
            NotEqual | Verb(PrimVerb::EqualBang) => prim_compare(&x, &y, |ord| ord != Ordering::Equal),
            GreaterThan | Verb(PrimVerb::GreaterThan) => prim_compare(&x, &y, |ord| ord > Ordering::Equal),
            GreaterThanEqual | Verb(PrimVerb::GreaterThanEquals) => prim_compare(&x, &y, |ord| ord >= Ordering::Equal),
            LessThan | Verb(PrimVerb::LessThan) => prim_compare(&x, &y, |ord| ord < Ordering::Equal),
            LessThanEqual | Verb(PrimVerb::LessThanEquals) => prim_compare(&x, &y, |ord| ord <= Ordering::Equal),
            Min | Verb(PrimVerb::LessThanColon) => prim_choose_atoms(&x, &y, Val::le),
            Max | Verb(PrimVerb::GreaterThanColon) => prim_choose_atoms(&x, &y, Val::ge),
            Index | Verb(PrimVerb::At) => self.prim_index(&x, &y),
            Find | Verb(PrimVerb::Question) => Ok(Val::Int(prim_find(x.as_val(), y.as_val()))),
            FindSubseq | Verb(PrimVerb::QuestionColon) => Ok(Val::I64s(Rc::new(prim_subsequence_starts(x.as_val(), y.as_val())))),

            Sum => prim::sum(x, Some(y)), // self.fold_val(Val::Function(Rc::new(Func::Prim(PrimFunc::Verb(PrimVerb::Plus)))), x, Some(y)), // TODO prim::sum(x, Some(y)),

            _ => todo!("{x:?} {v:?} {y:?}"),
        };
        result.map_err(|err| cold(format!("Error in `{v}': {err}")))
    }

    // TODO output formatting (take indent as arg)
    fn prim_fmt(&self, prec: PrecedenceContext, x: &Val, out: &mut String) -> Result<(), String> {
        macro_rules! write_or {
            ($($arg:tt)*) => {
                write!($($arg)*).map_err(|err| cold(err.to_string()))
            };
        }

        use PrecedenceContext::*;

        fn parenthesized_if<F>(cond: bool, out: &mut String, f: F) -> Result<(), String>
        where F: FnOnce(&mut String) -> Result<(), String> {
            if cond { write_or!(out, "(")?; }
            f(out)?;
            if cond { write_or!(out, ")")?; }
            Ok(())
        }

        match x {
            Val::Char(c) => write_or!(out, "{}", char::from_u32(*c as u32).unwrap())?,
            Val::Int(i) => write_or!(out, "{i}")?,
            Val::Float(float) => write_or!(out, "{float}")?,
            Val::U8s(cs) => match std::str::from_utf8(cs) {  // TODO unicode
                Ok(s) => write_or!(out, "{s}")?,
                Err(err) => return cold(Err(err.to_string())),
            },
            Val::I64s(is) => {
                if is.is_empty() {
                    write_or!(out, "[]")
                } else {
                    parenthesized_if(prec >= AdverbOperand, out, |out| {
                        write_or!(out, "{}", is[0])?;
                        for int in &is[1..] { write_or!(out, " {int}")? }
                        Ok(())
                    })
                }?
            }
            Val::F64s(fs) => {
                if fs.is_empty() {
                    write_or!(out, "[]")
                } else {
                    parenthesized_if(prec >= AdverbOperand, out, |out| {
                        write_or!(out, "{}", fs[0])?;
                        for float in &fs[1..] { write_or!(out, " {float}")?; }
                        Ok(())
                    })
                }?
            }
            Val::Vals(vs) => {
                if vs.is_empty() {
                    write_or!(out, "[]")?
                } else {
                    let nested_list = vs.iter().any(|val| val.len().is_some());
                    write_or!(out, "[")?;
                    self.prim_debug_fmt(Toplevel, &vs[0], out)?;
                    for val in &vs[1..] {
                        if nested_list { write_or!(out, "\n ")? }
                        else { write_or!(out, "; ")? }
                        self.prim_debug_fmt(Toplevel, val, out)?;
                    }
                    write_or!(out, "]")?
                }
            }
            Val::Function(rc) => match &**rc {
                Func::Prim(prim) => write_or!(out, "{prim}")?,
                Func::AdverbDerived { adverb, operand } => parenthesized_if(
                    prec >= ConjunctionLeftOperand, out, |out| {
                        write_or!(out, "{adverb}")?;
                        self.prim_fmt(AdverbOperand, operand.as_val(), out)?;
                        Ok(())
                    }
                )?,
                Func::Atop { f_func, g_func } => parenthesized_if(
                    prec >= Small, out, |out| {
                        self.prim_fmt(Toplevel, f_func, out)?;
                        write_or!(out, " ")?;
                        || -> Result<(), String> {
                            if let Val::Function(rc) = g_func {
                                if let Func::Bound { func, y } = &**rc {
                                    self.prim_fmt(Small, func, out)?;
                                    write_or!(out, " ")?;
                                    self.prim_fmt(Small, y, out)?;
                                    return Ok(());
                                }
                            }
                            self.prim_fmt(Small, g_func, out)
                        }()?;
                        Ok(())
                    }
                )?,
                Func::Bound { func, y } => parenthesized_if(prec >= Small, out, |out| {
                    self.prim_fmt(Small, func, out)?;
                    write_or!(out, " ")?;
                    self.prim_fmt(Small, y, out)?;
                    Ok(())
                })?,
                Func::Fork { f_func, h_func, g_func } => parenthesized_if(
                    prec >= Small, out, |out| {
                        self.prim_fmt(Toplevel, f_func, out)?;
                        write_or!(out, " ")?;
                        self.prim_fmt(Small, h_func, out)?;
                        write_or!(out, " ")?;
                        self.prim_fmt(Small, g_func, out)?;
                        Ok(())
                    }
                )?,
                Func::Ambivalent(monad, dyad) => parenthesized_if(
                    prec >= ConjunctionLeftOperand, out, |out| {
                        self.prim_fmt(ConjunctionLeftOperand, monad.as_val(), out)?;
                        write_or!(out, " : ")?;
                        self.prim_fmt(AdverbOperand, dyad.as_val(), out)?;
                        Ok(())
                    }
                )?,
                Func::Explicit { .. } => {
                    // map code index -> tokens?
                    write_or!(out, "{{explicit func}}")?
                }
            }
        }
        Ok(())
    }

    // TODO output formatting
    fn prim_debug_fmt(&self, prec: PrecedenceContext, x: &Val, out: &mut String) -> Result<(), String> {
        macro_rules! write_or {
            ($($arg:tt)*) => {
                write!($($arg)*).map_err(|err| cold(err.to_string()))
            };
        }

        match x {
            Val::Char(c) => write_or!(out, "{:?}", char::from_u32(*c as u32).unwrap())?,
            Val::U8s(cs) => match std::str::from_utf8(cs) {  // TODO unicode
                Ok(s) => write_or!(out, "{s:?}")?,
                Err(err) => return cold(Err(err.to_string())),
            },
            _ => self.prim_fmt(prec, x, out)?,
        }
        Ok(())
    }

    fn prim_to_string(&self, x: &Val) -> Result<String, String> {
        let mut s = String::new();
        self.prim_fmt(PrecedenceContext::Toplevel, x, &mut s)?;
        Ok(s)
    }

    fn prim_to_debug_string(&self, x: &Val) -> Result<String, String> {
        let mut s = String::new();
        self.prim_debug_fmt(PrecedenceContext::Toplevel, x, &mut s)?;
        Ok(s)
    }

    fn prim_print_bytecode(&self, x: &Val) -> Result<(), String> {
        if let Val::Function(rc) = x {
            if let Func::Explicit { code_index, closure_env } = &**rc {
                if closure_env.borrow().is_empty() {
                    println!("Env: []");
                } else {
                    println!("Env: [");
                    for (i, val) in closure_env.borrow().iter().enumerate() {
                        print!("  {i}: ");
                        print!("{}", self.prim_to_debug_string(val.as_val())?);
                        println!();
                    }
                    println!("]");
                }
                let len = irrefutable!(self.code[*code_index - 1],
                                       Instr::MakeFunc { num_instructions: n } => n);

                println!("Code: [");
                for instr in &self.code[*code_index .. *code_index+len] {
                    println!("  {instr}");
                }
                println!("]");
                return Ok(())
            }
        }
        cold_err!("domain\nx is not an explicit function, so it has no bytecode. x is: {}",
                  self.prim_to_debug_string(x)?)
    }

    fn fold_val(&mut self, f: Val, x: Val, maybe_y: Option<Val>) -> Result<Val, String> {
        let (mut seed, start) = match maybe_y {
            Some(y) => (y, 0),
            None => match index_or_cycle_val(&x, 0) {
                Some(first) => (first, 1),
                None => return cold_err!("Error: fold with no input"),
            }
        };

        for i in start..x.len().unwrap_or(1) {
            seed = self.call_val(f.clone(), seed, index_or_cycle_val(&x, i))?;
        }

        Ok(seed)
    }

    fn index_or_call_from_locals(&mut self, x: Val, num_args: usize) -> Result<Val, String> {
        use Val::*;
        match x {
            Function(f) => match &*f {
                &Func::Explicit { ref closure_env, code_index } => {
                    let frame = StackFrame {
                        code_index,
                        closure_env: closure_env.clone(),
                        locals_start: self.locals_stack.len() - num_args,
                    };
                    self.stack_frames.push(frame);
                    // TODO arity mismatch (too few args, too many args).  If
                    // this returns an array on arity args, the (num_args-arity)
                    // args can be used to index the array and so forth.
                    //
                    // This is a problem because explicits truncate the locals
                    // stack on Return. Can this be helped if function locals
                    // are on the stack in opposite order? (can't add new ones
                    // during execution, but first ones can be popped off, also
                    // don't need to track locals start)
                    self.execute(code_index)?;
                    Ok(self.pop())
                }
                _ => todo!(),
            }
            _ => {
                let result = self.progressive_index_from_locals(x, num_args);
                self.locals_stack.truncate(self.locals_stack.len() - num_args);
                result
            }
        }
    }

    fn progressive_index_from_locals(&mut self, x: Val, num_args: usize) -> Result<Val, String> {
        if num_args == 0 { return Ok(x) }
        for_each_atom_retaining_shape(
            &self.locals_stack[self.locals_stack.len() - num_args].clone(),
            |i| {
                let elem = self.prim_index(&x, i)?;
                self.progressive_index_from_locals(elem, num_args - 1)
            }
        )
    }

    // TODO index by float when int-convertible.
    fn prim_index(&mut self, x: &Val, y: &Val) -> Result<Val, String> {
        #[cold]
        fn oob(i: i64, len: usize) -> String {
            format!("index out of bounds\nRequested index {i}, but length is {len}")
        }
        fn to_index(i: i64, len: usize) -> usize {
            if i >= 0 { i as usize } else { (len as i64 + i) as usize }  // TODO overflow?
        }
        fn index<A>(slice: &[A], i: i64) -> Result<&A, String> {
            slice.get(to_index(i, slice.len())).ok_or_else(|| oob(i, slice.len()))
        }
        fn index_atom<A: Clone>(atom: &A, i: i64) -> Result<A, String> {
            let len = 1;
            if to_index(i, len) == 0 { Ok(atom.clone()) } else { Err(oob(i, len)) }
        }

        use Val::*;
        let val = match (x.as_val(), y.as_val()) {
            (Int(_) | Char(_) | Float(_), &Int(i)) => return index_atom(x, i),
            (Char(ch), I64s(is)) => U8s(Rc::new(traverse(&**is, |i| index_atom(ch, *i))?)),
            (Int(int), I64s(is)) => I64s(Rc::new(traverse(&**is, |i| index_atom(int, *i))?)),
            (Float(float), I64s(is)) => F64s(Rc::new(traverse(&**is, |i| index_atom(float, *i))?)),
            (U8s(cs), &Int(i)) => Char(*index(cs, i)?),
            (U8s(cs), I64s(is)) => U8s(Rc::new(traverse(&**is, |i| index(cs, *i).copied())?)),
            (I64s(is), &Int(i)) => Int(*index(is, i)?),
            (I64s(xs), I64s(is)) => I64s(Rc::new(traverse(&**is, |i| index(xs, *i).copied())?)),
            (F64s(is), &Int(i)) => Float(*index(is, i)?),
            (F64s(xs), I64s(is)) => F64s(Rc::new(traverse(&**is, |i| index(xs, *i).copied())?)),
            (Vals(vs), &Int(i)) => return Ok(index(vs, i)?.clone()),
            (Vals(vs), I64s(is)) => collect_list(is.iter().map(|i| index(vs, *i).cloned()))?,
            (Char(_) | Int(_) | Float(_) | U8s(_) | I64s(_) | F64s(_) | Vals(_), Vals(is)) => collect_list(
                is.iter().map(|i| self.prim_index(x, i))
            )?,
            _ => return self.call_val(x.clone(), y.clone(), None),
        };
        Ok(val)
    }
}

fn for_each_atom_retaining_shape<F>(x: &Val, mut f: F) -> Result<Val, String>
where F: FnMut(&Val) -> Result<Val, String> {
    match x {
        atom!() => f(x),
        Val::U8s(x) => collect_list(x.as_slice().iter().map(|x| f(&Val::Char(*x)))),
        Val::I64s(x) => collect_list(x.as_slice().iter().map(|x| f(&Val::Int(*x)))),
        Val::F64s(x) => collect_list(x.as_slice().iter().map(|x| f(&Val::Float(*x)))),
        Val::Vals(x) => collect_list(x.as_slice().iter().map(f)),
    }
}

// Primitives

fn prim_get_line() -> Result<Val, String> {
    let mut line = String::new();
    if let Err(err) = std::io::stdin().read_line(&mut line) {
        return Err(cold(err.to_string()))
    }
    Ok(Val::U8s(Rc::new(line.into_bytes())))
}

fn prim_type(x: &Val) -> Vec<u8> {
    x.type_name().as_bytes().to_vec()
}

fn prim_show(x: Val) -> Result<Val, String> {
    fn as_bytes<A: ToString>(a: A) -> Val {
        Val::U8s(Rc::new(a.to_string().into_bytes()))
    }

    let ret = match x {
        Val::Char(c) => as_bytes(char::from_u32(c as u32).unwrap()),
        Val::Int(i) => as_bytes(i),
        Val::Float(f) => as_bytes(f),
        Val::U8s(cs) => Val::Vals(Rc::new(map_rc(cs, |c| as_bytes(char::from_u32(*c as u32).unwrap())))),
        Val::I64s(is) => Val::Vals(Rc::new(map_rc(is, |i| as_bytes(*i)))),
        Val::F64s(fs) => Val::Vals(Rc::new(map_rc(fs, |f| as_bytes(*f)))),
        Val::Vals(vs) => Val::Vals(Rc::new(traverse_rc(vs, |v| prim_show(v.clone()))?)),
        _ => return cold_err!("domain\nUnable to show {x:?}"), //TODO actually we can!
    };
    Ok(ret)
}

fn prim_exit(x: &Val) -> Result<Val, String> {
    use std::process::exit;

    match x.as_val() {
        Val::Int(i) => exit(*i as i32),
        Val::Float(f) if *f == f.trunc() => exit(*f as i32),
        bad => return cold_err!("domain\nExpected integer exit code, got {bad:?}"),
    }
}

fn prim_prefixes(x: &Val) -> Vec<Val> {
    fn get_prefixes<A: Clone, F: Fn(Rc<Vec<A>>) -> Val>(xs: &Vec<A>, f: F) -> Vec<Val> {
        map(1..=xs.len(), |i| f(Rc::new(xs[..i].to_vec())))
    }

    match x.as_val() {
        Val::Char(c) if true => vec![Val::U8s(Rc::new(vec![*c]))],
        Val::Int(i) if true => vec![Val::I64s(Rc::new(vec![*i]))],
        Val::Float(f) if true => vec![Val::F64s(Rc::new(vec![*f]))],
        atom!() => vec![x.clone()],
        Val::U8s(xs) => get_prefixes(&**xs, |rc_vec| Val::U8s(rc_vec)),
        Val::I64s(xs) => get_prefixes(&**xs, |rc_vec| Val::I64s(rc_vec)),
        Val::F64s(xs) => get_prefixes(&**xs, |rc_vec| Val::F64s(rc_vec)),
        Val::Vals(xs) => get_prefixes(&**xs, |rc_vec| Val::Vals(rc_vec)),
    }
}

fn prim_suffixes(x: &Val) -> Vec<Val> {
    fn get_suffixes<A: Clone, F: Fn(Rc<Vec<A>>) -> Val>(xs: &Vec<A>, f: F) -> Vec<Val> {
        map(0..xs.len(), |i| f(Rc::new(xs[i..].to_vec())))
    }

    match x.as_val() {
        Val::Char(c) if true => vec![Val::U8s(Rc::new(vec![*c]))],
        Val::Int(i) if true => vec![Val::I64s(Rc::new(vec![*i]))],
        Val::Float(f) if true => vec![Val::F64s(Rc::new(vec![*f]))],
        atom!() => vec![x.clone()],
        Val::U8s(xs) => get_suffixes(xs, |rc_vec| Val::U8s(rc_vec)),
        Val::I64s(xs) => get_suffixes(xs, |rc_vec| Val::I64s(rc_vec)),
        Val::F64s(xs) => get_suffixes(xs, |rc_vec| Val::F64s(rc_vec)),
        Val::Vals(xs) => get_suffixes(xs, |rc_vec| Val::Vals(rc_vec)),
    }
}

fn prim_choose_atoms<F>(x: &Val, y: &Val, f: F) -> Result<Val, String>
where F: Copy + Fn(&Val, &Val) -> bool {
    Ok(match zip_vals(x, y) {
        None => if f(x, y) { x.clone() } else { y.clone() },
        Some(iter) => collect_list(iter?.map(|(x, y)| prim_choose_atoms(&x, &y, f)))?
    })
}

// Attempts to find the whole of y as an element of x.
// TODO flip argument order?
fn prim_find(x: &Val, y: &Val) -> i64 {
    use Val::*;
    match (x, y) {
        (atom!(), _) => if x == y { 0 } else { 1 },
        (U8s(xs), Char(c)) => index_of(&**xs, c),
        (I64s(xs), Int(i)) => index_of(&**xs, i),
        (I64s(xs), Float(f)) =>
            float_as_int(*f).map(|i| index_of(&**xs, &i)).unwrap_or(xs.len() as i64),
        (F64s(xs), Float(f)) => index_of(&**xs, f),
        (F64s(xs), Int(i)) => index_of(&**xs, &(*i as f64)),
        (Vals(xs), _) => index_of(xs.iter().map(|rc_val| rc_val.as_val()), y),
        _ => x.len().unwrap_or(1) as i64,
    }
}

fn prim_where(x: &Val) -> Result<Val, String> {
    use Val::*;
    let val = match x {
        Int(i) => Val::I64s(Rc::new(replicate_with_i64(0, *i)?.collect())),
        Float(f) => Val::I64s(Rc::new(replicate_with_float(0, *f)?.collect())),
        I64s(xs) => {
            let mut vec = vec![];
            for (i, n) in xs.iter().enumerate() {
                vec.extend(replicate_with_i64(i as i64, *n)?)
            }
            Val::I64s(Rc::new(vec))
        }
        F64s(xs) => {
            let mut vec = vec![];
            for (i, f) in xs.iter().enumerate() {
                vec.extend(replicate_with_float(i as i64, *f)?)
            }
            Val::I64s(Rc::new(vec))
        }
        Vals(xs) => Val::Vals(
            Rc::new(traverse(&**xs, |val| prim_where(val))?)
        ),
        _ => return cold_err!("domain\nExpected integers, got {x:?}"),
    };
    Ok(val)
}

fn prim_subsequence_starts(x: &Val, y: &Val) -> Vec<i64> {
    use Val::*;

    // TODO linear time impl
    fn subsequence_starts_by<A, B, F: Fn(&A, &B) -> bool>(
        text: &[A], pat: &[B], pred: F
    ) -> Vec<i64> {
        if pat.is_empty() { return replicate(1, text.len()).collect() }

        let mut out = Vec::with_capacity(text.len());
        for i in 0..=(text.len() - pat.len()) {
            let matches = text[i..].iter().zip(pat).all(|(t, p)| pred(t, p));
            out.push(matches as i64);
        }
        out.extend(replicate(0, pat.len() - 1));
        out
    }

    fn subsequence_starts<A: PartialEq<B>, B>(text: &[A], pat: &[B]) -> Vec<i64> {
        subsequence_starts_by(text, pat, PartialEq::eq)
    }

    let starts = match (x, y) {
        (atom!(), atom!()) => vec![(x == y) as i64],
        (U8s(xs), Char(y)) => subsequence_starts(xs, &[*y]),
        (U8s(xs), U8s(ys)) => subsequence_starts(xs, ys),
        (U8s(xs), Vals(ys)) => subsequence_starts_by(xs, ys, |x, y| y.as_val() == x),
        (U8s(xs), _) => replicate(0, xs.len()).collect(),

        (I64s(xs), Int(y)) => xs.iter().map(|x| (x == y) as i64).collect(),
        (I64s(xs), I64s(ys)) => subsequence_starts(xs, ys),
        (I64s(xs), Vals(ys)) => subsequence_starts_by(xs, ys, |x, y| y.as_val() == x),
        (I64s(xs), _) => replicate(0, xs.len()).collect(),

        (F64s(xs), Float(y)) => xs.iter().map(|x| (x == y) as i64).collect(),
        (F64s(xs), F64s(ys)) => subsequence_starts(xs, ys),
        (F64s(xs), Vals(ys)) => subsequence_starts_by(xs, ys, |x, y| y.as_val() == x),
        (F64s(xs), _) => replicate(0, xs.len()).collect(),

        (Vals(xs), atom!()) => map(&**xs, |x| (x == y) as i64),

        (Vals(xs), U8s(ys)) => subsequence_starts_by(xs, ys, |x, y| x.as_val() == y),
        (Vals(xs), I64s(ys)) => subsequence_starts_by(xs, ys, |x, y| x.as_val() == y),
        (Vals(xs), F64s(ys)) => subsequence_starts_by(
            xs, ys,
            |x, y| match x.as_val() {
                Float(f) => f == y,
                Int(i) => *i as f64 == *y,
                _ => false,
            }
        ),
        (Vals(xs), Vals(ys)) => subsequence_starts(xs, ys),
        _ => todo!(),
    };

    starts
}

fn prim_read_file(x: &Val) -> Result<Val, String> {
    let mut byte: [u8; 1] = [0; 1];
    let path = std::str::from_utf8(
        match x {
            Val::Char(c) => { byte[0] = *c; &byte }
            Val::U8s(cs) => cs,
            _ => return cold_err!("expected string filepath, got {x:?}"),
        }
    ).map_err(|err| err.to_string())?;
    match std::fs::read_to_string(path) {
        Ok(contents) => Ok(Val::U8s(Rc::new(contents.into_bytes()))),
        Err(err) => Err(cold(err.to_string())),
    }
}

fn prim_reverse(x: Val) -> Val {
    use Val::*;
    fn reverse_iter<A: Clone>(xs: &[A]) -> Vec<A> {
        xs.iter().cloned().rev().collect()
    }

    match x {
        atom!() => x,
        // TODO inplace
        U8s(xs) => U8s(Rc::new(reverse_iter(xs.as_slice()))),
        I64s(xs) => I64s(Rc::new(xs.iter().cloned().rev().collect())),
        F64s(xs) => F64s(Rc::new(xs.iter().cloned().rev().collect())),
        Vals(xs) => Vals(Rc::new(xs.iter().cloned().rev().collect())),
    }
}

// TODO reshape on list y
fn prim_take(x: Val, y: &Val) -> Result<Val, String> {
    let count = match y {
        &Val::Int(i) => i,
        _ => return cold_err!("Invalid right argument {y:?}"),
    };

    fn take_from_slice<A: Clone>(count: i64, xs: &[A]) -> Rc<Vec<A>> {
        let (start, count) = if count < 0 {
            let abs_count = (-count) as usize;
            (xs.len() - abs_count % xs.len(), abs_count)
        } else {
            (0, count as usize)
        };
        Rc::new(xs.iter().cloned().cycle().skip(start).take(count).collect())
    }

    let result = match x.as_val() {
        Val::U8s(cs) => Val::U8s(take_from_slice(count, cs)),
        Val::I64s(is) => Val::I64s(take_from_slice(count, is)),
        Val::F64s(fs) => Val::F64s(take_from_slice(count, fs)),
        Val::Vals(vals) => Val::Vals(take_from_slice(count, vals)),
        Val::Char(c) => Val::U8s(Rc::new(vec![*c; count.abs() as usize])),
        Val::Int(int) => Val::I64s(Rc::new(vec![*int; count.abs() as usize])),
        Val::Float(float) => Val::F64s(Rc::new(vec![*float; count.abs() as usize])),
        _ => Val::Vals(Rc::new(vec![x; count.abs() as usize])),
    };
    Ok(result)
}

fn prim_copy(x: &Val, y: &Val) -> Result<Val, String> {
    use Val::*;
    #[cold]
    fn unexpected_y(y: &Val) -> String {
        format!("domain\nExpected non-negative integer, got {y:?}")
    }

    fn replicate_all<A: Clone>(xs: &[A], y: usize) -> Vec<A> {
        // TODO see if vec![0; count] -> vec[i] = ... is faster
        let mut vec = Vec::with_capacity(xs.len() * y);
        for x in xs {
            vec.extend(replicate(x, y).cloned())
        }
        vec
    }

    fn replicate_each<A: Clone, Y: ExactSizeIterator<Item=usize>>(
        xs: &[A], ys: Y, count: usize
    ) -> Result<Vec<A>, String> {
        match_lengths(xs.len(), ys.len())?;
        let mut vec = Vec::with_capacity(count);
        for (x, y) in xs.iter().zip(ys) {
            vec.extend(replicate(x.clone(), y))
        }
        Ok(vec)
    }

    fn run_one(x: &Val, y: usize) -> Val {
        match x.as_val() {
             Char(x) if true =>  U8s(Rc::new(replicate(*x, y).collect())),
              Int(x) if true => I64s(Rc::new(replicate(*x, y).collect())),
            Float(x) if true => F64s(Rc::new(replicate(*x, y).collect())),
             atom!() => Vals(Rc::new(replicate(x.clone(), y).collect())),
             U8s(xs) =>  U8s(Rc::new(replicate_all(xs, y))),
            I64s(xs) => I64s(Rc::new(replicate_all(xs, y))),
            F64s(xs) => F64s(Rc::new(replicate_all(xs, y))),
            Vals(xs) => Vals(Rc::new(replicate_all(xs, y))),
        }
    }

    fn run_many<Y>(x: &Val, y: Y) -> Result<Val, String>
    where Y: Clone + ExactSizeIterator<Item=usize> {
        let count = y.clone().sum();
        let val = match x.as_val() {
             Char(x) if true =>  U8s(Rc::new(replicate(*x, count).collect())),
              Int(x) if true => I64s(Rc::new(replicate(*x, count).collect())),
            Float(x) if true => F64s(Rc::new(replicate(*x, count).collect())),
             atom!() => Vals(Rc::new(replicate(x.clone(), count).collect())),
            U8s(xs) => U8s(Rc::new(replicate_each(xs, y, count)?)),
            I64s(xs) => I64s(Rc::new(replicate_each(xs, y, count)?)),
            F64s(xs) => F64s(Rc::new(replicate_each(xs, y, count)?)),
            Vals(xs) => Vals(Rc::new(replicate_each(xs, y, count)?)),
        };
        Ok(val)
    }

    fn int_as_usize(i: i64) -> Result<usize, String> {
        i.try_into().map_err(|_| unexpected_y(&Val::Int(i)))
    }

    fn float_as_usize(f: f64) -> Result<usize, String> {
        if f >= 0.0 && f.trunc() == f {
            Ok(f as usize)
        } else {
            Err(unexpected_y(&Val::Float(f)))
        }
    }

    let val = match y.as_val() {
        Int(i) => run_one(x, int_as_usize(*i)?),
        Float(f) => run_one(x, float_as_usize(*f)?),
        I64s(is) => {
            for i in is.iter() {
                if *i < 0 {
                    return Err(unexpected_y(&Val::Int(*i)))
                }
            }
            run_many(x, is.iter().map(|i| *i as usize))?
        }
        F64s(fs) => {
            for f in fs.iter() {
                if *f < 0.0 || *f != f.trunc() {
                    return Err(unexpected_y(&Val::Float(*f)))
                }
            }
            run_many(x, fs.iter().map(|f| *f as usize))?
        }
        // TODO y may still have depth 1 if e.g. int, float
        Vals(_) => todo!("#: for Vals y"),
        // zip_vals(x, y).unwrap()?.map(
        //     |x, y|
        //     None => unreachable!(),
        //     Some(iter) =>
        _ => return Err(unexpected_y(y)),
    };

    Ok(val)
}

fn cmp<A: Ord>(down: bool, a: &A, b: &A) -> Ordering {
    let up = a.cmp(b);
    if down { up.reverse() } else { up }
}

fn cmp_floats(down: bool, a: &f64, b: &f64) -> Ordering {
    let up = a.total_cmp(b);
    if down { up.reverse() } else { up }
}

// TODO
// fn min_max(down: bool, x: &RcVal, y: &RcVal) -> Result<RcVal, String> {
//     match zip_vals(x.as_val(), y.as_val()) {
//         None => match cmp(x.as_val(), y.as_val()) {
//             Ordering::Less => y.clone(),
//             _ => x.clone(),
//         }

//         Some(iter) => iter.map(
//         (Val::U8s(cs), Val::Char(c)) => Val::I64s(
//             cs.iter().
//         ),
//     }
// }

fn prim_grade(x: &Val, down: bool) -> Val {
    let indices = match x.as_val() {
        atom!() => vec![0],
        Val::U8s(cs) => {  // TODO unicode
            let mut indices: Vec<i64> = (0..cs.len() as i64).collect();
            indices.sort_by(|i, j| cmp(down, &cs[*i as usize], &cs[*j as usize]));
            indices
        }
        Val::I64s(ints) => {
            let mut indices: Vec<i64> = (0..ints.len() as i64).collect();
            indices.sort_by(|i, j| cmp(down, &ints[*i as usize], &ints[*j as usize]));
            indices
        }
        Val::F64s(fs) => {
            let mut indices: Vec<i64> = (0..fs.len() as i64).collect();
            indices.sort_by(|i, j| cmp_floats(down, &fs[*i as usize], &fs[*j as usize]));
            indices
        }
        Val::Vals(vals) => {
            let mut indices: Vec<i64> = (0..vals.len() as i64).collect();
            indices.sort_by(|i, j| cmp(down, &vals[*i as usize], &vals[*j as usize]));
            indices
        }
    };
    Val::I64s(Rc::new(indices))
}

fn prim_sort(x: Val, down: bool) -> Val {
    match x {
        Val::Char(c) if true => Val::U8s(Rc::new(vec![c])),
        Val::Int(i) if true => Val::I64s(Rc::new(vec![i])),
        Val::Float(f) if true => Val::F64s(Rc::new(vec![f])),
        atom!() => Val::Vals(Rc::new(vec![x])),
        Val::U8s(cs) => {
            let mut sorted = Rc::unwrap_or_clone(cs);
            sorted.sort_unstable_by(|a, b| cmp(down, a, b));
            Val::U8s(Rc::new(sorted))
        }
        Val::I64s(is) => {
            let mut sorted = Rc::unwrap_or_clone(is);
            sorted.sort_unstable_by(|a, b| cmp(down, a, b));
            Val::I64s(Rc::new(sorted))
        }
        Val::F64s(fs) => {
            let mut sorted = Rc::unwrap_or_clone(fs);
            sorted.sort_unstable_by(|a, b| cmp_floats(down, a, b));
            Val::F64s(Rc::new(sorted))
        }
        Val::Vals(vals) => {
            let mut sorted = Rc::unwrap_or_clone(vals);
            sorted.sort_by(|a, b| cmp(down, a, b));
            Val::Vals(Rc::new(sorted))
        }
    }
}

fn prim_append(x: Val, y: Val) -> Result<Val, String> {
    use std::iter::once;
    use Val::*;

    fn one_then_many<A, I>(x: A, ys: I) -> Rc<Vec<A>>
    where I: IntoIterator<Item=A> {
        Rc::new(once(x).chain(ys.into_iter()).collect())
    }

    fn many_then_one<A, I>(xs: I, y: A) -> Rc<Vec<A>>
    where I: IntoIterator<Item=A> {
        Rc::new(xs.into_iter().chain(once(y)).collect())
    }

    fn many_then_many<A, I, J>(xs: I, ys: J) -> Rc<Vec<A>>
    where I: IntoIterator<Item=A>,
          J: IntoIterator<Item=A> {
        Rc::new(xs.into_iter().chain(ys.into_iter()).collect())
    }

    fn clones<'a, A: Clone>(x: &'a [A]) -> impl ExactSizeIterator<Item=A> + 'a {
        x.iter().cloned()
    }

    fn floats<'a>(x: &'a [i64]) -> impl ExactSizeIterator<Item=f64> + 'a {
        x.iter().map(|i| *i as f64)
    }

    fn push_or_clone<A: Clone>(mut x: Rc<Vec<A>>, y: A) -> Rc<Vec<A>> {
        match Rc::get_mut(&mut x) {
            Some(vec) => { vec.push(y); x }
            None => many_then_one(clones(&x), y),
        }
    }

    fn extend_or_clone<A: Clone, Y: IntoIterator<IntoIter: ExactSizeIterator<Item=A>>>(mut x: Rc<Vec<A>>, y: Y) -> Rc<Vec<A>> {
        match Rc::get_mut(&mut x) {
            Some(vec) => { vec.extend(y.into_iter()); x }
            None => many_then_many(clones(&x), y.into_iter()),
        }
    }

    let val = match (x, y) {
        (Char(x), Char(y)) => U8s(Rc::new(vec![x, y])),
        (Char(x), U8s(y)) => U8s(one_then_many(x, clones(&y))),
        (U8s(x), Char(y)) => U8s(push_or_clone(x, y)),
        (U8s(x), U8s(y)) => U8s(extend_or_clone(x, clones(&y))),

        (Int(x), Int(y)) => I64s(Rc::new(vec![x, y])),
        (Int(x), I64s(y)) => I64s(one_then_many(x, y.iter().copied())),
        (I64s(x), Int(y)) => I64s(push_or_clone(x, y)),
        (I64s(x), I64s(y)) => I64s(extend_or_clone(x, clones(&y))),

        (Float(x), Float(y)) => F64s(Rc::new(vec![x, y])),
        (Float(x), F64s(y)) => F64s(one_then_many(x, y.iter().copied())),
        (F64s(x), Float(y)) => F64s(push_or_clone(x, y)),
        (F64s(x), F64s(y)) => F64s(extend_or_clone(x, clones(&y))),

        (Int(x), Float(y)) => F64s(Rc::new(vec![x as f64, y])),
        (Float(x), Int(y)) => F64s(Rc::new(vec![x, y as f64])),

        (Int(x), F64s(y)) =>  F64s(one_then_many(x as f64, clones(&y))),
        (F64s(x), Int(y)) =>  F64s(push_or_clone(x, y as f64)),

        (I64s(x), Float(y)) => F64s(many_then_one(floats(&x), y)),
        (Float(x), I64s(y)) => F64s(one_then_many(x, floats(&y))),

        (I64s(x), F64s(y)) => F64s(many_then_many(floats(&x), clones(&y))),
        (F64s(x), I64s(y)) => F64s(extend_or_clone(x, floats(&y))),

        (Vals(x), y@atom!()) => Vals(push_or_clone(x, y)),
        (Vals(x), Vals(y)) => Vals(extend_or_clone(x, clones(&y))),

        (x, y) => match (iter_val(&x), iter_val(&y)) {
            (None, None) => Vals(Rc::new(vec![x, y])),
            (Some(iter), None) => Vals(many_then_one(iter, y)),
            (None, Some(iter)) => Vals(one_then_many(x, iter)),
            (Some(x), Some(y)) => Vals(many_then_many(x, y)),
        }
    };
    Ok(val)
}

fn prim_ravel(x: &Val) -> Val {
    match x.as_val() {
        atom!() => Val::Vals(Rc::new(vec![x.clone()])),
        Val::U8s(_) | Val::I64s(_) | Val::F64s(_) => x.clone(),

        // TODO do something about this mess
        Val::Vals(_) => collect_list(
            ValIter { x: x.clone(), i: 0 }
            .flat_map(|val| ValIter { x: prim_ravel(&val), i: 0 })
            .map(|val| -> Result<Val, ()> { Ok(val) })
        ).unwrap(),
    }
}

fn prim_compare<F: Fn(Ordering) -> bool + Copy>(x: &Val, y: &Val, op: F) -> Result<Val, String> {
    use Val::*;
    let result = match zip_vals(&x, &y) {
        None => Int(op(x.as_val().cmp(y.as_val())) as i64),
        // TODO we already know this will consist of bits
        Some(iter) => collect_list(iter?.map(|(x, y)| prim_compare(&x, &y, op)))?
    };
    Ok(result)
}

fn iota(x: &Val) -> Val {
    use Val::*;
    match x {
        &Int(i) => Val::I64s(Rc::new(if i >= 0 { 0..i } else { i..0 }.collect())),
        _ => todo!("Implement x/ on non-ints"),
    }
}

// TODO should [] == "" be 1?
fn prim_match(x: &Val, y: &Val) -> Result<Val, String> {
    Ok(Val::Int((x == y) as i64))
}

fn index_of<'a, A: 'a + PartialEq, I: IntoIterator<Item=&'a A>>(x: I, y: &A) -> i64 {
    let mut i = 0i64;
    let mut iter = x.into_iter();
    while let Some(next) = iter.next() {
        if next == y { break }
        i += 1
    }
    return i
}

fn replicate<A: Clone>(a: A, n: usize) -> impl Iterator<Item=A> {
    std::iter::repeat(a).take(n)
}

fn replicate_with_float<A: Clone>(a: A, f: f64) -> Result<impl Iterator<Item=A>, String> {
    match float_as_int(f) {
        Some(n) => replicate_with_i64(a, n),
        _ => cold_err!("domain\nExpected non-negative integer, got {f}"),
    }
}

fn replicate_with_i64<A: Clone>(a: A, n: i64) -> Result<impl Iterator<Item=A>, String> {
    if n >= 0 {
        Ok(replicate(a, n as usize))
    } else {
        cold_err!("domain\nExpected non-negative integer, got {n}")
    }
}

fn map<A, I, F>(xs: I, f: F) -> Vec<A>
where I: IntoIterator,
      F: FnMut(I::Item) -> A {
    Vec::from_iter(xs.into_iter().map(f))
}


fn traverse<A, E, I, F>(xs: I, f: F) -> Result<Vec<A>, E>
where I: IntoIterator,
      F: FnMut(I::Item) -> Result<A, E> {
    Result::from_iter(xs.into_iter().map(f))
}

// TODO for map_rc/traverse_rc, allow f to take Val or &Val
fn map_rc<A, X, F>(xs: Rc<Vec<X>>, mut f: F) -> Vec<A>
where F: FnMut(&X) -> A {
    match Rc::try_unwrap(xs) {
        Ok(xs) => xs.into_iter().map(|x| f(&x)).collect(),
        Err(rc) => rc.iter().map(f).collect(),
    }
}

fn traverse_rc<A, E, X, F>(xs: Rc<Vec<X>>, mut f: F) -> Result<Vec<A>, E>
where F: FnMut(&X) -> Result<A, E> {
    match Rc::try_unwrap(xs) {
        Ok(xs) => xs.into_iter().map(|x| f(&x)).collect::<Result<Vec<_>, _>>(),
        Err(xs) => xs.iter().map(f).collect::<Result<Vec<_>, _>>(),
    }
}

#[derive(Clone)]
struct ValIter {
    x: Val,
    i: usize,
}

impl Iterator for ValIter {
    type Item = Val;
    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        index_or_cycle_val(&self.x, self.i - 1)
    }
}

struct ZippedVals {
    x: Val,
    y: Val,
    i: usize,
}

impl Iterator for ZippedVals {
    type Item = (Val, Val);
    fn next(&mut self) -> Option<Self::Item> {
        let x_val = index_or_cycle_val(&self.x, self.i)?;
        let y_val = index_or_cycle_val(&self.y, self.i)?;
        self.i += 1;
        Some((x_val, y_val))
    }
}

// TODO dyadic
fn for_each(mem: &mut Mem, f: Val, x: Val) -> Result<Val, String> {
    use crate::ops::{SingleValConsumer, IsVal};

    struct ForEach<'a> { mem: &'a mut Mem, f: Val }
    impl<'a> SingleValConsumer for ForEach<'a> {
        fn eat_val(&mut self, val: Val) -> Result<Val, String> {
            self.mem.call_val(self.f.clone(), val, None)
        }
        fn eat_val_ref(&mut self, val: &Val) -> Result<Val, String> {
            self.mem.call_val(self.f.clone(), val.clone(), None)
        }
    }
    x.dispatch_for_each(ForEach { mem, f })
}

fn iter_val(x: &Val) -> Option<ValIter> {
    if x.len().is_none() { return None }
    Some(ValIter { x: x.clone(), i: 0 })
}

fn zip_vals(x: &Val, y: &Val) -> Option<Result<ZippedVals, String>> {
    match (x.len(), y.len()) {
        (None, None) => return None,
        (Some(xlen), Some(ylen)) => if let Err(err) = match_lengths(xlen, ylen) { return Some(Err(err)) },
        _ => (),
    }
    Some(Ok(ZippedVals { x: x.clone(), y: y.clone(), i: 0 }))
}

fn prim_negate(x: Val) -> Result<Val, String> {
    use Val::*;
    Ok(match x {
        Int(x) => Int(-x),
        Float(x) => Float(-x),
        I64s(x) => match Rc::try_unwrap(x) {
            Ok(x) => I64s(Rc::new(x.into_iter().map(|x| -x).collect())),
            Err(x) => I64s(Rc::new(x.as_slice().iter().map(|x| -*x).collect())),
        }
        F64s(x) => match Rc::try_unwrap(x) {
            Ok(x) => F64s(Rc::new(x.into_iter().map(|x| -x).collect())),
            Err(x) => F64s(Rc::new(x.as_slice().iter().map(|x| -*x).collect())),
        }
        Vals(x) => match Rc::try_unwrap(x) {
            Ok(x) => Vals(Rc::new(x.into_iter().map(prim_negate).collect::<Result<_, _>>()?)),
            Err(x) => Vals(Rc::new(x.as_slice().iter().cloned().map(prim_negate).collect::<Result<_, _>>()?)),
        }
        x => return cold_err!("domain\nUnsupported argument: {}\nExpected a numeric value",
                              x.type_name()),
    })
}
