use std::{
    cell::RefCell,
    cmp::Ordering,
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
    Push(RcVal),
}

// Used in printing vals.
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

    // Always points to Val::Vals.
    closure_env: Rc<RefCell<Vec<RcVal>>>,

    // Index into locals_stack. Local slots are offsets from this index.
    locals_start: usize,
}

pub struct Mem {
    pub code: Vec<Instr>,

    pub stack: Vec<RcVal>,

    // Stack of local scopes
    locals_stack: Vec<RcVal>,

    // Details about the explicit function (or global scope) we're currently
    // in. Never empty!
    stack_frames: Vec<StackFrame>,

    // TODO intern small ints (actually not necessary if we do Small | Rc<Big>)
    zero: RcVal,
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

            zero: Rc::new(Val::Int(0)),
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
                                if let PushVar { src } = self.code[ip] {
                                    ip += 1;
                                    // TODO use e.g. List::I64s if closure vals are all ints
                                    closure_data.push(self.load(src));
                                } else {
                                    panic!("Malformed code at ip {ip}: expected PushVar after MakeClosure, but found {:?}", self.code[ip]);
                                }
                            }
                            closure_data
                        }
                        _ => vec![],
                    };

                    self.push(RcVal::new(Val::Function(Func::Explicit {
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
                PushLiteralInteger(value) => self.push(RcVal::new(Val::Int(value))),
                PushLiteralFloat(value) => self.push(RcVal::new(Val::Float(value))),
                PushVar { src } => {
                    let val = self.load(src);
                    self.push(val);
                }
                PushPrimVerb { prim: PrimVerb::Rec } => {
                    let frame = self.current_frame();
                    // TODO can we copy the actual function that was called instead?
                    self.push(RcVal::new(Val::Function(Func::Explicit {
                        code_index: frame.code_index,
                        closure_env: frame.closure_env.clone(),
                    })));
                }
                PushPrimVerb { prim } => self.push(RcVal::new(Val::Function(Func::Prim(prim)))),

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
                CallPrimVerb1 { prim } => {
                    let x = self.pop();
                    let result = self.call_prim_monad(prim, x)?;
                    self.push(result);
                }
                CallPrimVerb2 { prim } => {
                    let y = self.pop();
                    let x = self.pop();
                    if prim == PrimVerb::At && x.is_func() && self.is_tail_call(ip) {
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
                        a@atom!() => return err!("Array unpacking failed; expected {count} elements, got atom {:?}", a),
                        Val::U8s(cs) => {
                            if cs.len() != count {
                                return err!("Array unpacking failed; expected {count} elements, got {}", cs.len())
                            }
                            self.stack.extend(cs.iter().rev().map(|c| RcVal::new(Val::Char(*c))))
                        }
                        Val::I64s(is) => {
                            if is.len() != count {
                                return err!("Array unpacking failed; expected {count} elements, got {}", is.len())
                            }
                            self.stack.extend(is.iter().rev().map(|i| RcVal::new(Val::Int(*i))))
                        }
                        Val::F64s(fs) =>  {
                            if fs.len() != count {
                                return err!("Array unpacking failed; expected {count} elements, got {}", fs.len())
                            }
                            self.stack.extend(fs.iter().rev().map(|f| RcVal::new(Val::Float(*f))))
                        }
                        Val::Vals(vs) => {
                            if vs.len() != count {
                                return err!("Array unpacking failed; expected {count} elements, got {}", vs.len())
                            }
                            self.stack.extend(vs.iter().rev().map(|v| v.clone()))
                        }
                    }
                }
                CallPrimAdverb { prim: adverb } => {
                    let operand = self.pop();
                    match operand.as_val() {
                        Val::Function(Func::Ambivalent(monad, _)) if matches!(adverb, PrimAdverb::P) => self.push(monad.clone()),
                        
                        _ => self.push(RcVal::new(Val::Function(Func::AdverbDerived { adverb, operand }))),
                    }
                }
                CollectVerbAlternatives => {
                    let mut dyad = self.pop();
                    if let Val::Function(Func::Ambivalent(_, inner_dyad)) = dyad.as_val() {
                        dyad = inner_dyad.clone();
                    }
                    let mut monad = self.pop();
                    if let Val::Function(Func::Ambivalent(inner_monad, _)) = monad.as_val() {
                        monad = inner_monad.clone();
                    }
                    self.push(RcVal::new(Val::Function(Func::Ambivalent(monad, dyad))));
                }
                MakeAtopFunc => {
                    let g_func = self.pop();
                    let f_func = self.pop();
                    self.push(RcVal::new(Val::Function(Func::Atop { f_func, g_func })));
                }
                MakeBoundFunc => {
                    let y = self.pop();
                    let func = self.pop();
                    self.push(RcVal::new(Val::Function(Func::Bound { func, y })));
                }
                MakeForkFunc => {
                    let g_func = self.pop();
                    let h_func = self.pop();
                    let f_func = self.pop();
                    self.push(RcVal::new(Val::Function(Func::Fork { f_func, h_func, g_func })));
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
                    self.push(RcVal::new(Val::U8s(s)));
                }
                LiteralBytes { bytes } => self.push(RcVal::new(Val::Char(bytes[0]))),
                CollectToArray { num_elems } => {
                    let mut all_chars = true;
                    let mut all_ints = true;
                    let mut all_floats = true;
                    for elem in &self.stack[(self.stack.len() - num_elems)..] {
                        all_chars &= matches!(&**elem, Val::Char(_));
                        all_ints &= matches!(&**elem, Val::Int(_));
                        all_floats &= matches!(&**elem, Val::Float(_) | Val::Int(_));
                    }

                    let elems = self.stack.drain((self.stack.len() - num_elems)..);
                    let list_val = if all_chars {
                        Val::U8s(map(elems, |elem| irrefutable!(*elem, Val::Char(ch) => ch)))
                    } else if all_ints {
                        Val::I64s(map(elems, |elem| irrefutable!(*elem, Val::Int(int) => int)))
                    } else if all_floats {
                        Val::F64s(map(elems, |elem| match *elem {
                            Val::Float(f) => f,
                            Val::Int(i) => i as f64,
                            _ => unreachable!(),
                        }))
                    } else {
                        Val::Vals(elems.collect())
                    };
                    self.push(RcVal::new(list_val))
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn push(&mut self, val: RcVal) {
        self.stack.push(val)
    }

    #[inline]
    fn pop(&mut self) -> RcVal {
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


    fn load(&mut self, var: Var) -> RcVal {
        // TODO do all local/closure vars point to non-lists (instead to slices)?
        let frame = self.current_frame();
        match var.place {
            Place::Local => self.locals_stack[frame.locals_start + var.slot].clone(),
            Place::ClosureEnv => frame.closure_env.borrow()[var.slot].clone(),
        }
    }

    fn store(&mut self, dst: Var, val: RcVal) {
        let frame = self.current_frame();
        match dst.place {
            Place::Local => {
                let absolute_slot = frame.locals_start + dst.slot;
                if absolute_slot >= self.locals_stack.len() {
                    self.locals_stack.resize(absolute_slot + 1, self.zero.clone());  // TODO is zero right?
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

    fn chase_tail(&mut self, mut func: RcVal, mut x: RcVal, mut y: Option<RcVal>) -> Result<ChasedTail, String> {
        loop {
            let function = match func.as_val() {
                Val::Function(f) => f,
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
                    self.locals_stack.push(y.unwrap_or_else(|| self.zero.clone()));
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
                Func::Prim(PrimVerb::At) if x.is_func() && y.is_some() => {
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
    fn call_val(&mut self, val: RcVal, x: RcVal, y: Option<RcVal>) -> Result<RcVal, String> {
        let result = match val.as_val() {
            Val::Char(_) | Val::Int(_) | Val::Float(_) |
            Val::U8s(_) | Val::I64s(_) | Val::F64s(_) | Val::Vals(_) => val,
            Val::Function(func) => match func {
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
                    self.locals_stack.push(y.unwrap_or(self.zero.clone()));
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
                        operand: RcVal,
                        x: RcVal,  // TODO take &Val?
                        maybe_y: Option<RcVal>) -> Result<RcVal, String> {
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
                None => match iter_val(&x) {
                    None => self.call_val(operand, x, None)?,
                    Some(iter) => RcVal::new(collect_list(
                        iter.map(|val| self.call_val(operand.clone(), val, None))
                    )?),
                }
                Some(y) => match zip_vals(&x, &y) {
                    None => self.call_val(operand, x, Some(y))?,
                    Some(iter) => RcVal::new(collect_list(
                        iter?.map(|(x_val, y_val)| self.call_val(operand.clone(), x_val, Some(y_val)))
                    )?),
                }
            }
            Backtick => match maybe_y {
                None => match iter_val(&x) {
                    None => self.call_val(operand, x, None)?,
                    Some(iter) => RcVal::new(collect_list(
                        iter.map(|val| self.call_val(operand.clone(), val, None))
                    )?),
                }
                Some(y) => match iter_val(&x) {
                    None => self.call_val(operand.clone(), x, Some(y))?,
                    Some(iter) => RcVal::new(collect_list(
                        iter.map(|x_val| self.call_val(operand.clone(), x_val, Some(y.clone())))
                    )?),
                }
            }
            BacktickColon => match maybe_y {
                None => match iter_val(&x) {
                    None => self.call_val(operand, x, None)?,
                    Some(iter) => RcVal::new(collect_list(
                        iter.map(|val| self.call_val(operand.clone(), val, None))
                    )?),
                }
                Some(y) => match iter_val(&y) {
                    None => self.call_val(operand.clone(), x, Some(y))?,
                    Some(iter) => RcVal::new(collect_list(
                        iter.map(|y_val| self.call_val(operand.clone(), x.clone(), Some(y_val)))
                    )?),
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

    fn call_prim_monad(&mut self, v: PrimVerb, x: RcVal) -> Result<RcVal, String> {
        use PrimVerb::*;
        let result = match v {
            P | Q => Ok(x),
            Show => prim_show(x.as_val()).map(RcVal::new),
            GetLine => prim_get_line(),
            Print => self.prim_to_string(&x)
                .inspect(|s| println!("{s}"))
                .map(|_| x),
            DebugPrint => self.prim_to_debug_string(&x)
                .inspect(|s| println!("{s}"))
                .map(|_| x),
            PrintBytecode => self.prim_print_bytecode(x.as_val()).map(|_| x),
            ReadFile => prim_read_file(x.as_val()).map(RcVal::new),
            Hash => Ok(RcVal::new(Val::Int(x.len().unwrap_or(1) as i64))),
            Slash => Ok(RcVal::new(iota(&*x))),
            Pipe => Ok(prim_reverse(&x)),
            Comma => Ok(prim_ravel(&x)),
            Caret => Ok(RcVal::new(Val::Vals(prim_prefixes(&x)))),
            Dollar => Ok(RcVal::new(Val::Vals(prim_suffixes(&x)))),
            Question => prim_where(x.as_val()).map(RcVal::new),
            LessThan => Ok(RcVal::new(prim_sort(&x, false))),
            GreaterThan => Ok(RcVal::new(prim_sort(&x, true))),
            LessThanColon => Ok(RcVal::new(prim_grade(&x, false))),
            GreaterThanColon => Ok(RcVal::new(prim_grade(&x, true))),
            Type => Ok(RcVal::new(Val::U8s(prim_type(x.as_val())))),
            Exit => prim_exit(&x),
            _ => todo!("{x:?} {v:?}")
        };
        result.map_err(|err| format!("Error in `{v}': {err}"))
    }

    fn call_prim_dyad(&mut self, v: PrimVerb, x: RcVal, y: RcVal) -> Result<RcVal, String> {
        use PrimVerb::*;
        let result = match v {
            P => Ok(x),
            Q => Ok(y),
            Plus => prim::add(x.as_val(), y.as_val()),
            Minus => prim::subtract(x.as_val(), y.as_val()),
            Asterisk => prim::multiply(x.as_val(), y.as_val()),
            Slash => prim::divide(x.as_val(), y.as_val()),
            DoubleSlash => prim::int_divide(x.as_val(), y.as_val()),
            Percent => prim::int_mod(x.as_val(), y.as_val()),
            Caret => prim::pow(x.as_val(), y.as_val()),
            Hash => prim_take(x, y.as_val()),
            HashColon => prim_copy(&x, &y),
            Comma => prim_append(x, y),
            DoubleEquals => prim_match(x.as_val(), y.as_val()),
            Equals => prim_compare(&x, &y, |ord| ord == Ordering::Equal),
            EqualBang => prim_compare(&x, &y, |ord| ord != Ordering::Equal),
            GreaterThan => prim_compare(&x, &y, |ord| ord > Ordering::Equal),
            GreaterThanEquals => prim_compare(&x, &y, |ord| ord >= Ordering::Equal),
            LessThan => prim_compare(&x, &y, |ord| ord < Ordering::Equal),
            LessThanEquals => prim_compare(&x, &y, |ord| ord <= Ordering::Equal),
            LessThanColon => prim_choose_atoms(&x, &y, Val::le),
            GreaterThanColon => prim_choose_atoms(&x, &y, Val::ge),
            At => self.prim_index(&x, &y),
            Question => Ok(RcVal::new(Val::Int(prim_find(x.as_val(), y.as_val())))),
            QuestionColon => Ok(RcVal::new(Val::I64s(prim_subsequence_starts(x.as_val(), y.as_val())))),
            _ => todo!("{x:?} {v:?} {y:?}"),
        };
        result.map_err(|err| format!("Error in `{v}': {err}"))
    }

    // TODO output formatting (take indent as arg)
    fn prim_fmt(&self, prec: PrecedenceContext, x: &Val, out: &mut String) -> Result<(), String> {
        macro_rules! write_or {
            ($($arg:tt)*) => {
                write!($($arg)*).map_err(|err| err.to_string())
            };
        }

        use PrecedenceContext::*;

        fn parenthesized_if<F: FnOnce(&mut String) -> Result<(), String>>(
            cond: bool, out: &mut String, f: F
        ) -> Result<(), String> {
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
                Err(err) => return Err(err.to_string()),
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
            Val::Function(f) => match f {
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
                        if let Val::Function(Func::Bound { func, y }) = g_func.as_val() {
                            self.prim_fmt(Small, func, out)?;
                            write_or!(out, " ")?;
                            self.prim_fmt(Small, y, out)?;
                        } else {
                            self.prim_fmt(Small, g_func, out)?;
                        }
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
                write!($($arg)*).map_err(|err| err.to_string())
            };
        }

        match x {
            Val::Char(c) => write_or!(out, "{:?}", char::from_u32(*c as u32).unwrap())?,
            Val::U8s(cs) => match std::str::from_utf8(cs) {  // TODO unicode
                Ok(s) => write_or!(out, "{s:?}")?,
                Err(err) => return Err(err.to_string()),
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
        match x {
            Val::Function(Func::Explicit { code_index, closure_env }) => {
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
                Ok(())
            }
            _ => err!("domain\nx is not an explicit function, so it has no bytecode. x is: {}", self.prim_to_debug_string(x)?),
        }
    }

    fn fold_val(&mut self, f: RcVal, x: RcVal, maybe_y: Option<RcVal>) -> Result<RcVal, String> {
        let (mut seed, start) = match maybe_y {
            Some(y) => (y, 0),
            None => match index_or_cycle_val(&x, 0) {
                Some(first) => (first, 1),
                None => return err!("Error: fold with no input"),
            }
        };

        for i in start..x.len().unwrap_or(1) {
            seed = self.call_val(f.clone(), seed, index_or_cycle_val(&x, i))?;
        }

        Ok(seed)
    }

    fn prim_index(&mut self, x: &RcVal, y: &RcVal) -> Result<RcVal, String> {
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
            (Int(_) | Char(_), &Int(i)) => return index_atom(x, i),
            (Int(int), I64s(is)) => I64s(traverse(is, |i| index_atom(int, *i))?),
            (Char(ch), I64s(is)) => U8s(traverse(is, |i| index_atom(ch, *i))?),
            (I64s(is), &Int(i)) => Int(*index(is, i)?),
            (I64s(xs), I64s(is)) => I64s(traverse(is, |i| index(xs, *i).copied())?),
            (U8s(cs), &Int(i)) => Char(*index(cs, i)?),
            (U8s(cs), I64s(is)) => U8s(traverse(is, |i| index(cs, *i).copied())?),
            (Vals(vs), &Int(i)) => return Ok(index(vs, i)?.clone()),
            (Vals(vs), I64s(is)) => collect_list(is.iter().map(|i| index(vs, *i).cloned()))?,
            (Int(_) | Char(_) | I64s(_) | U8s(_) | Vals(_), Vals(is)) => collect_list(
                is.iter().map(|i| self.prim_index(x, i))
            )?,
            _ => return self.call_val(x.clone(), y.clone(), None),
        };
        Ok(RcVal::new(val))
    }
}

// Primitives

fn prim_get_line() -> Result<RcVal, String> {
    let mut line = String::new();
    if let Err(err) = std::io::stdin().read_line(&mut line) {
        return Err(err.to_string())
    }
    Ok(RcVal::new(Val::U8s(line.into_bytes())))
}

fn prim_type(x: &Val) -> Vec<u8> {
    x.type_name().as_bytes().to_vec()
}

fn prim_show(x: &Val) -> Result<Val, String> {
    fn as_bytes<A: ToString>(a: A) -> Val {
        Val::U8s(a.to_string().into_bytes())
    }

    let ret = match x {
        Val::Char(c) => as_bytes(char::from_u32(*c as u32).unwrap()),
        Val::Int(i) => as_bytes(i),
        Val::Float(f) => as_bytes(f),
        Val::U8s(cs) => Val::Vals(map(cs, |c| RcVal::new(as_bytes(char::from_u32(*c as u32).unwrap())))),
        Val::I64s(is) => Val::Vals(map(is, |i| RcVal::new(as_bytes(i)))),
        Val::F64s(fs) => Val::Vals(map(fs, |f| RcVal::new(as_bytes(f)))),
        Val::Vals(vs) => Val::Vals(traverse(vs, |v| prim_show(v.as_val()).map(RcVal::new))?),
        _ => return err!("domain\nUnable to show {x:?}"), //TODO actually we can!
    };
    Ok(ret)
}

fn prim_exit(x: &RcVal) -> Result<RcVal, String> {
    use std::process::exit;

    match x.as_val() {
        Val::Int(i) => exit(*i as i32),
        Val::Float(f) if *f == f.trunc() => exit(*f as i32),
        bad => return err!("domain\nExpected integer exit code, got {bad:?}"),
    }
}

fn prim_prefixes(x: &RcVal) -> Vec<RcVal> {
    fn get_prefixes<A: Clone, F: Fn(Vec<A>) -> Val>(xs: &Vec<A>, f: F) -> Vec<RcVal> {
        map(1..=xs.len(), |i| RcVal::new(f(xs[..i].to_vec())))
    }

    match x.as_val() {
        Val::Char(c) if true => vec![RcVal::new(Val::U8s(vec![*c]))],
        Val::Int(i) if true => vec![RcVal::new(Val::I64s(vec![*i]))],
        Val::Float(f) if true => vec![RcVal::new(Val::F64s(vec![*f]))],
        atom!() => vec![x.clone()],
        Val::U8s(xs) => get_prefixes(xs, |vec| Val::U8s(vec)),
        Val::I64s(xs) => get_prefixes(xs, |vec| Val::I64s(vec)),
        Val::F64s(xs) => get_prefixes(xs, |vec| Val::F64s(vec)),
        Val::Vals(xs) => get_prefixes(xs, |vec| Val::Vals(vec)),
    }
}

fn prim_suffixes(x: &RcVal) -> Vec<RcVal> {
    fn get_suffixes<A: Clone, F: Fn(Vec<A>) -> Val>(xs: &Vec<A>, f: F) -> Vec<RcVal> {
        map(0..xs.len(), |i| RcVal::new(f(xs[i..].to_vec())))
    }

    match x.as_val() {
        Val::Char(c) if true => vec![RcVal::new(Val::U8s(vec![*c]))],
        Val::Int(i) if true => vec![RcVal::new(Val::I64s(vec![*i]))],
        Val::Float(f) if true => vec![RcVal::new(Val::F64s(vec![*f]))],
        atom!() => vec![x.clone()],
        Val::U8s(xs) => get_suffixes(xs, |vec| Val::U8s(vec)),
        Val::I64s(xs) => get_suffixes(xs, |vec| Val::I64s(vec)),
        Val::F64s(xs) => get_suffixes(xs, |vec| Val::F64s(vec)),
        Val::Vals(xs) => get_suffixes(xs, |vec| Val::Vals(vec)),
    }
}

fn prim_choose_atoms<F>(x: &RcVal, y: &RcVal, f: F) -> Result<RcVal, String>
where F: Copy + Fn(&Val, &Val) -> bool {
    Ok(match zip_vals(x, y) {
        None => if f(x.as_val(), y.as_val()) { x.clone() } else { y.clone() },
        Some(iter) => RcVal::new(
            collect_list(iter?.map(|(x, y)| prim_choose_atoms(&x, &y, f)))?
        ),
    })
}

// Attempts to find the whole of y as an element of x.
// TODO flip argument order?
fn prim_find(x: &Val, y: &Val) -> i64 {
    use Val::*;
    match (x, y) {
        (atom!(), _) => if x == y { 0 } else { 1 },
        (U8s(xs), Char(c)) => index_of(xs, c),
        (I64s(xs), Int(i)) => index_of(xs, i),
        (I64s(xs), Float(f)) =>
            float_as_int(*f).map(|i| index_of(xs, &i)).unwrap_or(xs.len() as i64),
        (F64s(xs), Float(f)) => index_of(xs, f),
        (F64s(xs), Int(i)) => index_of(xs, &(*i as f64)),
        (Vals(xs), _) => index_of(xs.iter().map(|rc_val| rc_val.as_val()), y),
        _ => x.len().unwrap_or(1) as i64,
    }
}

fn prim_where(x: &Val) -> Result<Val, String> {
    use Val::*;
    let val = match x {
        Int(i) => Val::I64s(replicate_with_i64(0, *i)?.collect()),
        Float(f) => Val::I64s(replicate_with_float(0, *f)?.collect()),
        I64s(xs) => {
            let mut vec = vec![];
            for (i, n) in xs.iter().enumerate() {
                vec.extend(replicate_with_i64(i as i64, *n)?)
            }
            Val::I64s(vec)
        }
        F64s(xs) => {
            let mut vec = vec![];
            for (i, f) in xs.iter().enumerate() {
                vec.extend(replicate_with_float(i as i64, *f)?)
            }
            Val::I64s(vec)
        }
        Vals(xs) => Val::Vals(
            traverse(xs, |val| prim_where(val).map(RcVal::new))?
        ),
        _ => return err!("domain\nExpected integers, got {x:?}"),
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

        (Vals(xs), atom!()) => map(xs, |x| (x.as_val() == y) as i64),

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
            _ => return err!("expected string filepath, got {x:?}"),
        }
    ).map_err(|err| err.to_string())?;
    match std::fs::read_to_string(path) {
        Ok(contents) => Ok(Val::U8s(contents.into_bytes())),
        Err(err) => Err(err.to_string()),
    }
}

fn prim_reverse(x: &RcVal) -> RcVal {
    use Val::*;
    fn reverse_iter<A: Clone>(xs: &[A]) -> Vec<A> {
        xs.iter().cloned().rev().collect()
    }

    match x.as_val() {
        atom!() => x.clone(),
        U8s(xs) => RcVal::new(U8s(reverse_iter(xs))),
        I64s(xs) => RcVal::new(I64s(xs.iter().cloned().rev().collect())),
        F64s(xs) => RcVal::new(F64s(xs.iter().cloned().rev().collect())),
        Vals(xs) => RcVal::new(Vals(xs.iter().cloned().rev().collect())),
    }
}

// TODO reshape on list y
fn prim_take(x: RcVal, y: &Val) -> Result<RcVal, String> {
    let count = match y {
        &Val::Int(i) => i,
        _ => return err!("Invalid right argument {y:?}"),
    };

    fn take_from_slice<A: Clone>(count: i64, xs: &[A]) -> Vec<A> {
        let (start, count) = if count < 0 {
            let abs_count = (-count) as usize;
            (xs.len() - abs_count % xs.len(), abs_count)
        } else {
            (0, count as usize)
        };
        xs.iter().cloned().cycle().skip(start).take(count).collect()
    }

    let result = match x.as_val() {
        Val::U8s(cs) => Val::U8s(take_from_slice(count, cs)),
        Val::I64s(is) => Val::I64s(take_from_slice(count, is)),
        Val::F64s(fs) => Val::F64s(take_from_slice(count, fs)),
        Val::Vals(vals) => Val::Vals(take_from_slice(count, vals)),
        Val::Char(c) => Val::U8s(vec![*c; count.abs() as usize]),
        Val::Int(int) => Val::I64s(vec![*int; count.abs() as usize]),
        Val::Float(float) => Val::F64s(vec![*float; count.abs() as usize]),
        _ => Val::Vals(vec![x; count.abs() as usize]),
    };
    Ok(RcVal::new(result))
}

fn prim_copy(x: &RcVal, y: &RcVal) -> Result<RcVal, String> {
    use Val::*;
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
        match_length(xs.len(), ys.len())?;
        let mut vec = Vec::with_capacity(count);
        for (x, y) in xs.iter().zip(ys) {
            vec.extend(replicate(x.clone(), y))
        }
        Ok(vec)
    }

    fn run_one(x: &RcVal, y: usize) -> Val {
        match x.as_val() {
             Char(x) if true =>  U8s(replicate(*x, y).collect()),
              Int(x) if true => I64s(replicate(*x, y).collect()),
            Float(x) if true => F64s(replicate(*x, y).collect()),
             atom!() => Vals(replicate(x.clone(), y).collect()),
             U8s(xs) =>  U8s(replicate_all(xs, y)),
            I64s(xs) => I64s(replicate_all(xs, y)),
            F64s(xs) => F64s(replicate_all(xs, y)),
            Vals(xs) => Vals(replicate_all(xs, y)),
        }
    }

    fn run_many<Y>(x: &RcVal, y: Y) -> Result<Val, String>
    where Y: Clone + ExactSizeIterator<Item=usize> {
        let count = y.clone().sum();
        let val = match x.as_val() {
             Char(x) if true =>  U8s(replicate(*x, count).collect()),
              Int(x) if true => I64s(replicate(*x, count).collect()),
            Float(x) if true => F64s(replicate(*x, count).collect()),
             atom!() => Vals(replicate(x.clone(), count).collect()),
            U8s(xs) => U8s(replicate_each(xs, y, count)?),
            I64s(xs) => I64s(replicate_each(xs, y, count)?),
            F64s(xs) => F64s(replicate_each(xs, y, count)?),
            Vals(xs) => Vals(replicate_each(xs, y, count)?),
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
            for i in is {
                if *i < 0 {
                    return Err(unexpected_y(&Val::Int(*i)))
                }
            }
            run_many(x, is.iter().map(|i| *i as usize))?
        }
        F64s(fs) => {
            for f in fs {
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

    Ok(RcVal::new(val))
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

fn prim_grade(x: &RcVal, down: bool) -> Val {
    let mut indices: Vec<i64> = vec![];
    match x.as_val() {
        atom!() => indices.push(0),
        Val::U8s(cs) => {  // TODO unicode
            indices = (0..cs.len() as i64).collect();
            indices.sort_unstable_by(|i, j| cmp(down, &cs[*i as usize], &cs[*j as usize]));
        }
        Val::I64s(ints) => {
            indices = (0..ints.len() as i64).collect();
            indices.sort_unstable_by(|i, j| cmp(down, &ints[*i as usize], &ints[*j as usize]));
        }
        Val::F64s(fs) => {
            indices = (0..fs.len() as i64).collect();
            indices.sort_unstable_by(|i, j| cmp_floats(down, &fs[*i as usize], &fs[*j as usize]));
        }
        Val::Vals(vals) => {
            indices = (0..vals.len() as i64).collect();
            indices.sort_by(|i, j| cmp(down, &vals[*i as usize], &vals[*j as usize]));
        }
    };
    Val::I64s(indices)
}

fn prim_sort(x: &RcVal, down: bool) -> Val {
    match x.as_val() {
        Val::Char(c) if true => Val::U8s(vec![*c]),
        Val::Int(i) if true => Val::I64s(vec![*i]),
        Val::Float(f) if true => Val::F64s(vec![*f]),
        atom!() => Val::Vals(vec![x.clone()]),
        Val::U8s(cs) => {
            let mut sorted = cs.clone();
            sorted.sort_unstable_by(|a, b| cmp(down, a, b));
            Val::U8s(sorted)
        }
        Val::I64s(is) => {
            let mut sorted = is.clone();
            sorted.sort_unstable_by(|a, b| cmp(down, a, b));
            Val::I64s(sorted)
        }
        Val::F64s(fs) => {
            let mut sorted = fs.clone();
            sorted.sort_unstable_by(|a, b| cmp_floats(down, a, b));
            Val::F64s(sorted)
        }
        Val::Vals(vals) => {
            let mut sorted = vals.clone();
            sorted.sort_by(|a, b| cmp(down, a, b));
            Val::Vals(sorted)
        }

    }
}

fn prim_append(x: RcVal, y: RcVal) -> Result<RcVal, String> {
    use std::iter::once;
    use Val::*;

    fn one_then_many<A, I>(x: A, ys: I) -> Vec<A>
    where I: IntoIterator<Item=A> {
        once(x).chain(ys.into_iter()).collect()
    }

    fn many_then_one<A, I>(xs: I, y: A) -> Vec<A>
    where I: IntoIterator<Item=A> {
        xs.into_iter().chain(once(y)).collect()
    }

    fn many_then_many<A, I, J>(xs: I, ys: J) -> Vec<A>
    where I: IntoIterator<Item=A>,
          J: IntoIterator<Item=A> {
        xs.into_iter().chain(ys.into_iter()).collect()
    }

    fn copies<'a, A: Copy>(x: &'a [A]) -> impl Iterator<Item=A> + 'a {
        x.iter().copied()
    }

    fn floats<'a>(x: &'a [i64]) -> impl Iterator<Item=f64> + 'a {
        x.iter().map(|i| *i as f64)
    }

    let val = match (x.as_val(), y.as_val()) {
        (Char(x), Char(y)) => U8s(vec![*x, *y]),
        (Char(x), U8s(y)) => U8s(one_then_many(*x, copies(y))),
        (U8s(x), Char(y)) => U8s(many_then_one(copies(x), *y)),
        (U8s(x), U8s(y)) => U8s(many_then_many(copies(x), copies(y))),

        (Int(x), Int(y)) => I64s(vec![*x, *y]),
        (Int(x), I64s(y)) => I64s(one_then_many(*x, y.iter().copied())),
        (I64s(x), Int(y)) => I64s(many_then_one(copies(x), *y)),
        (I64s(x), I64s(y)) => I64s(many_then_many(copies(x), copies(y))),

        (Float(x), Float(y)) => F64s(vec![*x, *y]),
        (Float(x), F64s(y)) => F64s(one_then_many(*x, y.iter().copied())),
        (F64s(x), Float(y)) => F64s(many_then_one(copies(x), *y)),
        (F64s(x), F64s(y)) => F64s(many_then_many(copies(x), copies(y))),

        (Int(x), Float(y)) => F64s(vec![*x as f64, *y]),
        (Float(x), Int(y)) => F64s(vec![*x, *y as f64]),

        (Int(x), F64s(y)) =>  F64s(one_then_many(*x as f64, copies(y))),
        (F64s(x), Int(y)) =>  F64s(many_then_one(copies(x), *y as f64)),

        (I64s(x), Float(y)) => F64s(many_then_one(floats(x), *y)),
        (Float(x), I64s(y)) => F64s(one_then_many(*x, floats(y))),

        (I64s(x), F64s(y)) => F64s(many_then_many(floats(x), copies(y))),
        (F64s(x), I64s(y)) => F64s(many_then_many(copies(x), floats(y))),

        _ => match (iter_val(&x), iter_val(&y)) {
            (None, None) => Vals(vec![x, y]),
            (Some(iter), None) => Vals(many_then_one(iter, y)),
            (None, Some(iter)) => Vals(one_then_many(x, iter)),
            (Some(x), Some(y)) => Vals(many_then_many(x, y)),
        }
    };
    Ok(RcVal::new(val))
}

fn prim_ravel(x: &RcVal) -> RcVal {
    match x.as_val() {
        atom!() => RcVal::new(Val::Vals(vec![x.clone()])),
        Val::U8s(_) | Val::I64s(_) | Val::F64s(_) => x.clone(),

        // TODO do something about this mess
        Val::Vals(_) => RcVal::new(collect_list(
            ValIter { x: x.clone(), i: 0 }
            .flat_map(|val| ValIter { x: prim_ravel(&val), i: 0 })
            .map(|val| -> Result<RcVal, ()> { Ok(val) })
        ).unwrap()),
    }
}

fn prim_compare<F: Fn(Ordering) -> bool + Copy>(x: &RcVal, y: &RcVal, op: F) -> Result<RcVal, String> {
    use Val::*;
    let result = match zip_vals(&x, &y) {
        None => Int(op(x.as_val().cmp(y.as_val())) as i64),
        // TODO we already know this will consist of bits
        Some(iter) => collect_list(iter?.map(|(x, y)| prim_compare(&x, &y, op)))?
    };
    Ok(RcVal::new(result))
}

fn iota(x: &Val) -> Val {
    use Val::*;
    match x {
        &Int(i) => Val::I64s(if i >= 0 { 0..i } else { i..0 }.collect()),
        _ => todo!("Implement x/ on non-ints"),
    }
}

// TODO should [] == "" be 1?
fn prim_match(x: &Val, y: &Val) -> Result<RcVal, String> {
    Ok(RcVal::new(Val::Int((x == y) as i64)))
}

fn collect_list<E, I: Iterator<Item=Result<RcVal, E>>>(mut it: I) -> Result<Val, E> {
    enum List {
        U8s(Vec<u8>),
        I64s(Vec<i64>),
        F64s(Vec<f64>),
        Vals(Vec<RcVal>),
    }

    let cap = match it.size_hint() {
        (lower, None) => lower,
        (_, Some(upper)) => upper,
    };

    let mut list = match it.next() {
        None => return Ok(Val::I64s(vec![])),
        Some(Err(err)) => return Err(err),
        Some(Ok(rc)) => match rc.as_val() {
            Val::Char(c) => {
                let mut vec = Vec::with_capacity(cap);
                vec.push(*c);
                List::U8s(vec)
            }
            Val::Int(i) => {
                let mut vec = Vec::with_capacity(cap);
                vec.push(*i);
                List::I64s(vec)
            }
            // TODO should we do this, or just go floats?
            Val::Float(f) => match float_as_int(*f) {
                None => {
                    let mut vec = Vec::with_capacity(cap);
                    vec.push(*f);
                    List::F64s(vec)
                }
                Some(i) => {
                    let mut vec = Vec::with_capacity(cap);
                    vec.push(i);
                    List::I64s(vec)
                }
            },
            _ => {
                let mut vec = Vec::with_capacity(cap);
                vec.push(rc);
                List::Vals(vec)
            }
        }
    };

    // TODO optimize
    for val_or in it {
        let val = val_or?;
        match &mut list {
            List::I64s(ints) => match val.as_val() {
                Val::Int(i) => ints.push(*i),
                Val::Float(f) => match float_as_int(*f) {
                    Some(i) => ints.push(i),
                    None => {
                        let mut fs: Vec<f64> = ints.drain(..).map(|i| i as f64).collect();
                        fs.reserve(cap - fs.len());
                        fs.push(*f);
                        list = List::F64s(fs);
                    }
                }
                _ => {
                    let mut vals: Vec<RcVal> =
                        ints.drain(..).map(|i| RcVal::new(Val::Int(i))).collect();
                    vals.reserve(cap - vals.len());
                    vals.push(val);
                    list = List::Vals(vals);
                }
            },
            List::F64s(fs) => match val.as_val() {
                Val::Float(f) => fs.push(*f),
                Val::Int(i) => fs.push(*i as f64),
                _ => {
                    let mut vals: Vec<RcVal> =
                        fs.drain(..).map(|f| RcVal::new(Val::Float(f))).collect();
                    vals.reserve(cap - vals.len());
                    vals.push(val);
                    list = List::Vals(vals);
                }
            },
            List::U8s(cs) => match val.as_val() {
                Val::Char(c) => cs.push(*c),
                _ => {
                    let mut vals: Vec<RcVal> =
                        cs.drain(..).map(|c| RcVal::new(Val::Char(c))).collect();
                    vals.reserve(cap - vals.len());
                    vals.push(val);
                    list = List::Vals(vals);
                }
            },
            List::Vals(vals) => vals.push(val),
        }
    }

    Ok(match list {
        List::U8s(cs) => Val::U8s(cs),
        List::I64s(ints) => Val::I64s(ints),
        List::F64s(fs) => Val::F64s(fs),
        List::Vals(vals) => Val::Vals(vals),
    })
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
        _ => err!("domain\nExpected non-negative integer, got {f}"),
    }
}

fn replicate_with_i64<A: Clone>(a: A, n: i64) -> Result<impl Iterator<Item=A>, String> {
    if n >= 0 {
        Ok(replicate(a, n as usize))
    } else {
        err!("domain\nExpected non-negative integer, got {n}")
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

#[derive(Clone)]
struct ValIter {
    x: RcVal,
    i: usize,
}

impl Iterator for ValIter {
    type Item = RcVal;
    fn next(&mut self) -> Option<Self::Item> {
        self.i += 1;
        index_or_cycle_val(&self.x, self.i - 1)
    }
}

struct ZippedVals {
    x: RcVal,
    y: RcVal,
    i: usize,
}

impl Iterator for ZippedVals {
    type Item = (RcVal, RcVal);
    fn next(&mut self) -> Option<Self::Item> {
        let x_val = index_or_cycle_val(&self.x, self.i)?;
        let y_val = index_or_cycle_val(&self.y, self.i)?;
        self.i += 1;
        Some((x_val, y_val))
    }
}

fn iter_val(x: &RcVal) -> Option<ValIter> {
    if x.len().is_none() { return None }
    Some(ValIter { x: x.clone(), i: 0 })
}

fn zip_vals(x: &RcVal, y: &RcVal) -> Option<Result<ZippedVals, String>> {
    match (x.len(), y.len()) {
        (None, None) => return None,
        (Some(xlen), Some(ylen)) => if let Err(err) = match_length(xlen, ylen) { return Some(Err(err)) },
        _ => (),
    }
    Some(Ok(ZippedVals { x: x.clone(), y: y.clone(), i: 0 }))
}

fn index_or_cycle_val(val: &RcVal, i: usize) -> Option<RcVal> {
    use Val::*;
    Some(match val.as_val() {
        atom!() => val.clone(),
        I64s(is) => RcVal::new(Val::Int(*is.get(i)?)),
        F64s(fs) => RcVal::new(Val::Float(*fs.get(i)?)),
        U8s(cs) => RcVal::new(Val::Char(*cs.get(i)?)),
        Vals(vs) => vs.get(i)?.clone(),
    })
}

fn match_length(xlen: usize, ylen: usize) -> Result<(), String> {
    if xlen == ylen { return Ok(()); }
    // TODO include name/position of verb
    err!("length mismatch: {xlen} vs {ylen}")
}
