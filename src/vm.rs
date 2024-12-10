use std::{
    cell::RefCell,
    cmp::Ordering,
    iter,
    mem,
    rc::Rc,
    fmt::Write,
};

use crate::bytecode::*;
use crate::lex::*;
use crate::prim;
use crate::util::*;
use crate::val::*;

// Used in val printing.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum PrecedenceContext {
    Toplevel,

    // These comments are cumulative; if parentheses are needed for one variant, they're also needed
    // for the variants that follow.

    // Parentheses needed for trains.
    Small,

    // Parentheses needed for stranded lists.
    AdverbOperand,

    // Parentheses needed for adverb and conjunction applications.
    ConjunctionLeftOperand,
}

#[derive(Debug)]
struct StackFrame {
    explicit: ExplicitFunc,

    // The index of the next header to try if this one fails, or None if there are no headers left
    // to try.
    //
    // TODO: use Option<NonZero>
    next_header: Option<usize>,

    // The ArgSpec this explicit function was called with.
    arg_spec: ArgSpec,

    // Index into `stack`.
    args_start: usize,

    // Index into `locals_stack`. Local slots are offsets from this index.
    locals_start: usize,

    // The function Yield dispatches to. This is G in F<-G. None if this function isn't being
    // treated as a target.
    yielding_to: Option<Val>,
}

// The VM is called `Mem` because, as data, it's just the machine's storage.
pub struct Mem {
    pub code: Vec<Instr>,

    pub stack: Vec<Val>,

    // Indices into `stack` that mark the start of a sequence of related stack elements; for
    // example, the elements of an array literal. We don't statically know the length of the result
    // since it may include splices, so we set a marker, continue execution, and then collect the
    // elements at or above the marker at the end. Stack markers are created by MarkStack and
    // consumed as needed by other instructions.
    stack_markers: Vec<usize>,

    // Stack of local scopes.
    locals_stack: Vec<Val>,

    // Details about the explicit function (or global scope) we're currently in. Never empty!
    stack_frames: Vec<StackFrame>,
}

impl Mem {
    pub fn new() -> Self {
        Self {
            code: vec![],
            stack: vec![],
            stack_markers: vec![],
            locals_stack: vec![],  // TODO stdlib?
            stack_frames: vec![StackFrame {
                explicit: ExplicitFunc {
                    code_index: usize::MAX,
                    closure_env: Rc::new(RefCell::new(vec![])),
                },
                next_header: None,
                arg_spec: ArgSpec::saturated(0),
                args_start: 0,
                locals_start: 0,
                yielding_to: None,
            }],
        }
    }

    pub fn execute_from_toplevel(&mut self, ip: usize) -> Res<()> {
        let result = self.execute(ip);
        if result.is_err() {
            self.stack.clear();
            self.stack_markers.clear();
            let mut leftover_frames = self.stack_frames.drain(1..);
            if let Some(second_frame) = leftover_frames.next() {
                self.locals_stack.truncate(second_frame.locals_start)
            }
        }
        result
    }

    fn execute(&mut self, mut ip: usize) -> Res<()> {
        use Instr::*;

        while ip < self.code.len() {
            ip += 1;
            match self.code[ip - 1] {
                Nop => {}
                Assert => {
                    let val = self.pop();
                    if val.is_falsy() {
                        // TODO better (+custom) messages
                        ip = self.jump_to_next_case_or_fail(|| "Assertion failed".to_string())?;
                    }
                }
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
                Dup => self.dup(),
                MakeClosure{..} => panic!(concat!("Malformed code at ip {}: reached MakeClosure ",
                                                  "not immediately following a MakeFunc's Return."),
                                          ip - 1),
                MakeClosureFromStack{..} =>
                    panic!(concat!("Malformed code at ip {}: reached MakeClosureFromStack not ",
                                   "immediately following a MakeFunc's Return."),
                           ip - 1),
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
                                    _ => panic!(
                                        concat!("Malformed code at ip {}: expected PushVar ",
                                                "after MakeClosure, but found {:?}"),
                                        ip, self.code[ip]
                                    ),
                                }
                            }
                            closure_data
                        }
                        Some(MakeClosureFromStack { num_closure_vars }) => {
                            ip += 1;
                            self.stack.drain(self.stack.len()-num_closure_vars ..).collect()
                        }
                        _ => vec![],
                    };

                    let func = Func::Unapplied(UnappliedFunc::Explicit(ExplicitFunc {
                        code_index,
                        closure_env: Rc::new(RefCell::new(closure_data)),
                    }));
                    self.push(Val::Function(Rc::new(func)));
                }
                Header { next_case_offset } => {
                    enum CaseTreatment { Enter, Curry, GoToNextCase }
                    use CaseTreatment::*;

                    let provided_arg_spec = self.current_frame().arg_spec;
                    let action = match self.code[ip] {
                        ArgCheck { arg_spec } =>
                            if provided_arg_spec == arg_spec { Enter }
                            else if arg_spec.is_satisfied_by(provided_arg_spec) { Curry }
                            else { GoToNextCase }
                        _ => GoToNextCase,
                    };

                    match action {
                        Enter => {
                            // We may need to try the next case after this one, so we have to keep a
                            // copy of the args underneath the args this function case actually
                            // works with. There can also be some args that weren't consumed by the
                            // previous function case, so we only have to copy the ones that have
                            // been consumed (note that args are on the stack in reverse order):
                            //
                            //   ...|a3|a2|a1|a0|a3|a2|
                            //      ^ args_start
                            //      ^-----------^ arg_count=4
                            //                  ^-----^ have=2
                            //            ^-----^ arg_start+have .. args_start+arg_count
                            //
                            // Here, we'd copy a1 and a0 to give us a full set of args over the
                            // copied ones:
                            // 
                            //   ...|a3|a2|a1|a0|a3|a2|a1|a0|
                            let args_start = self.current_frame().args_start;
                            let arg_count = provided_arg_spec.count_args() as usize;
                            let have = self.stack.len() - (args_start + arg_count);
                            if have < arg_count {
                                self.stack.extend_from_within(args_start+have .. args_start+arg_count);
                            }
                            let next_header_index = offset_by(ip, next_case_offset);
                            self.current_frame_mut().next_header = Some(next_header_index);
                            ip += 1;  // Skip ArgCheck instruction
                        }
                        Curry => {
                            let frame = self.stack_frames.pop().unwrap();
                            let explicit = UnappliedFunc::Explicit(ExplicitFunc {
                                code_index: ip - 1, // Come back to this header
                                closure_env: frame.explicit.closure_env,
                            });
                            let bound_args = {
                                let arg_count = provided_arg_spec.count_args() as usize;
                                // Pop the unconsumed args from the last function case, if any
                                self.stack.truncate(frame.args_start + arg_count);
                                self.stack.drain(frame.args_start..).collect()
                            };
                            let func = Func::PartiallyApplied {
                                func: explicit, bound_arg_spec: provided_arg_spec, bound_args
                            };
                            self.push(Val::Function(Rc::new(func)));
                            return Ok(())
                        }
                        GoToNextCase => ip = offset_by(ip, next_case_offset),
                    }
                }
                HeaderPassed => {
                    let frame = self.current_frame_mut();
                    frame.next_header = None;
                    let args_start = frame.args_start;
                    self.stack.truncate(args_start);
                }
                ArgCheck { arg_spec } => {
                    let provided_arg_spec = self.current_frame().arg_spec;
                    if provided_arg_spec == arg_spec {
                        let frame = self.current_frame();
                        self.stack.truncate(frame.args_start + arg_spec.count_args() as usize);
                    } else if arg_spec.is_satisfied_by(provided_arg_spec) {
                        // TODO deduplicate with header code
                        self.pop_locals();
                        let frame = self.stack_frames.pop().unwrap();
                        let explicit = UnappliedFunc::Explicit(ExplicitFunc {
                            code_index: ip - 1, // Come back to this header
                            closure_env: frame.explicit.closure_env,
                        });
                        let bound_args = {
                            let arg_count = provided_arg_spec.count_args();
                            // Pop the unconsumed args from the last function case, if any.
                            self.stack.truncate(frame.args_start + arg_count as usize);
                            self.stack.drain(frame.args_start..).collect()
                        };
                        let func = Func::PartiallyApplied {
                            func: explicit, bound_arg_spec: provided_arg_spec, bound_args
                        };
                        self.push(Val::Function(Rc::new(func)));
                        return Ok(())
                    } else {
                        // We don't need to try jumping to the next case on failure because this
                        // instruction is only executed directly if this is the last or only case;
                        // otherwise, Header would have taken care of it and skipped this
                        // instruction.
                        return cold_err!("Argument mismatch: function has no case matching {}",
                                         provided_arg_spec.describe()?);
                    }
                }
                // ArgCheckEq { count } => {
                //     let arg_count = self.current_frame().arg_spec.count_args() as usize;
                //     if arg_count != count {
                //         // We don't need to try jumping to the next case on failure because this
                //         // instruction is only executed directly if this is the last or only case;
                //         // otherwise, Header would have taken care of it and skipped this
                //         // instruction.
                //         return cold_err!("Arity mismatch; expected {count} args, got {arg_count}")
                //     }
                //     let args_start = self.current_frame().args_start;
                //     self.stack.truncate(args_start + arg_count);
                // }
                CollectArgs { suffix_count, keep } => {
                    let frame = self.current_frame();
                    let arg_count = frame.arg_spec.arity() as usize;
                    let have_saved_args = frame.next_header.is_some() as usize;
                    let vals_start = frame.args_start + suffix_count as usize + arg_count * have_saved_args;
                    if keep {
                        let array = collect_list(
                            self.stack.drain(vals_start..).map(Ok::<Val, Empty>).rev()
                        ).unwrap();
                        self.push(array);
                    } else {
                        self.stack.truncate(vals_start);
                    }
                }
                CopyArgs => {
                    let arg_count = self.current_frame().arg_spec.count_args() as usize;
                    let args_start = self.current_frame().args_start;
                    self.stack.extend_from_within(args_start .. args_start+arg_count);
                }
                Return => {
                    self.pop_locals();
                    self.stack_frames.pop();
                    return Ok(())
                }
                JumpRelative { offset } => ip = offset_by(ip, offset),
                JumpRelativeUnless { offset } =>
                    if self.pop().is_falsy() { ip = offset_by(ip, offset) }
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
                TuckVar { src } => {
                    let val = self.load(src);
                    self.tuck(val);
                }
                TuckVarLastUse { src } => {
                    let val = self.consuming_load(src);
                    self.tuck(val);
                }
                PushPrimFunc { prim: PrimFunc::Rec } => {
                    let frame = self.current_frame();
                    // TODO can we copy the actual function that was called instead?
                    self.push(Val::explicit_func(frame.explicit.clone()));
                }
                PushPrimFunc { prim } => self.push(Val::prim_func(prim)),
                Call1 => {
                    let f = self.pop();
                    if let Some(code_index) = self.call_or_get_jump_target(ip, f, ArgSpec::saturated(1))? {
                        ip = code_index;
                    }
                }
                Call2 => {
                    self.swap();         // x f y -> x y f
                    let f = self.pop();  // x y f -> x y
                    self.swap();         // x y -> y x
                    if let Some(code_index) = self.call_or_get_jump_target(ip, f, ArgSpec::saturated(2))? {
                        ip = code_index;
                    }
                }
                CallSpec { arg_spec } => {
                    // - 1 for function.
                    let call_start = self.stack.len() - arg_spec.count_args() as usize - 1;
                    self.stack[call_start..].reverse();
                    let f = self.pop();
                    if let Some(code_index) = self.call_or_get_jump_target(ip, f, arg_spec)? {
                        ip = code_index;
                    }
                }
                CallOnArgs { var } => {
                    let f = self.load(var);
                    let arg_spec = self.current_frame().arg_spec;
                    if let Some(code_index) = self.call_or_get_jump_target(ip, f, arg_spec)? {
                        ip = code_index;
                    }
                }
                CallPrimFunc1 { prim } =>
                    if prim == PrimFunc::Rec {
                        let frame = self.current_frame();
                        if let Some(code_index) = self.call_explicit_or_get_jump_target(
                            frame.explicit.clone(), ArgSpec::saturated(1), ip
                        )? {
                            ip = code_index;
                        }
                    } else {
                        let x = self.pop();
                        let result = self.call_prim_monad(prim, x)?;
                        self.push(result);
                    }
                CallPrimFunc2 { prim } =>
                    if prim == PrimFunc::Rec {
                        self.swap();
                        let frame = self.current_frame();
                        if let Some(code_index) = self.call_explicit_or_get_jump_target(
                            frame.explicit.clone(), ArgSpec::saturated(2), ip
                        )? {
                            ip = code_index;
                        }
                    } else if matches!(prim, PrimFunc::Verb(PrimVerb::At) | PrimFunc::Index) &&
                           self.stack[self.stack.len() - 2].is_func() &&
                           self.is_tail_call(ip) {
                        let x = self.stack.swap_remove(self.stack.len() - 2);
                        let arg_spec = ArgSpec::saturated(1);
                        if let Some(code_index) = self.call_or_get_jump_target(ip, x, arg_spec)? {
                            ip = code_index;
                        }
                    } else {
                        let y = self.pop();
                        let x = self.pop();
                        let result = self.call_prim_dyad(prim, x, y)?;
                        self.push(result);
                    }
                MarkStack => self.push_marker(self.stack.len()),
                Pop => { self.pop(); }
                StoreTo { dst } => {
                    let val = self.pop();
                    self.store(dst, val);
                }
                Splat => {
                    match iter_non_atom(self.pop()) {
                        // TODO just push the atom?
                        None => return cold_err!("Splice failed; expected array, got atom."),
                        Some(val_iter) => self.stack.extend(val_iter),
                    }
                }
                SplatReverse { count } => {
                    let x = self.pop();
                    let actual_count = match x.clone() {
                        atom!() => None,
                        Val::U8s(cs) => {
                            if cs.len() == count {
                                self.stack.extend(cs.iter().rev().map(|c| Val::Char(*c)))
                            }
                            Some(cs.len())
                        }
                        Val::I64s(is) => {
                            if is.len() == count {
                                self.stack.extend(is.iter().rev().map(|i| Val::Int(*i)))
                            }
                            Some(is.len())
                        }
                        Val::F64s(fs) =>  {
                            if fs.len() == count {
                                self.stack.extend(fs.iter().rev().map(|f| Val::Float(*f)))
                            }
                            Some(fs.len())
                        }
                        Val::Vals(vs) => {
                            if vs.len() == count {
                                self.stack.extend(vs.iter().rev().cloned())
                            }
                            Some(vs.len())
                        }
                    };

                    let success = actual_count.is_some_and(|actual| actual == count);
                    if !success {
                        self.push(x);
                        ip = self.jump_to_next_case_or_fail(
                            || format!("Array unpacking failed; expected {count} elements, got {}",
                                       to_string_or(actual_count, "atom"))
                        )?;
                    }
                }
                SplatReverseWithSplice { prefix_count, suffix_count, keep_splice } => {
                    let actual_count = self.top().len();

                    let (prefix_count, suffix_count) = (prefix_count as usize, suffix_count as usize);
                    let min_expected_count = prefix_count + suffix_count;

                    let success = actual_count.is_some_and(|actual| actual >= min_expected_count);
                    if !success {
                        ip = self.jump_to_next_case_or_fail(
                            || format!("Array unpacking failed; expected at least {} elements, got {}",
                                       min_expected_count, to_string_or(actual_count, "atom"))
                        )?;
                    } else {
                        let x = self.pop();  // Success: consume the value.
                        let len = actual_count.unwrap_or(1);

                        self.stack.reserve(min_expected_count + keep_splice as usize);
                        if suffix_count > 0 {
                            self.stack.extend(
                                ValIter { x: x.clone(), i: len - suffix_count, len }.rev()
                            );
                        }
                        if keep_splice {
                            self.stack.push(
                                collect_list(ValIter {
                                    x: x.clone(), i: prefix_count, len: len - suffix_count
                                }.map(Ok::<Val, Empty>)).unwrap()
                            )
                        }
                        if prefix_count > 0 {
                            self.stack.extend(ValIter { x, i: 0, len: prefix_count }.rev());
                        }
                    }
                }
                CallPrimAdverb { prim } => {
                    let operand = self.pop();
                    self.push(Val::adverb_derived_func(prim, operand));
                }
                CallPrimConjunction { prim } => {
                    let operand2 = self.pop();
                    let operand1 = self.pop();
                    self.push(Val::conjunction_derived_func(prim, operand1, operand2));
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
                CollectMarkedToArray => {
                    let mut all_chars = true;
                    let mut all_ints = true;
                    let mut all_floats = true;

                    let start_index = self.pop_marker();
                    for elem in &self.stack[start_index..] {
                        all_chars &= matches!(elem, Val::Char(_));
                        all_ints &= matches!(elem, Val::Int(_));
                        all_floats &= matches!(elem, Val::Float(_) | Val::Int(_));
                    }

                    let elems = self.stack.drain(start_index..);
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

    // Pops the second stack element from the top.
    #[inline]
    fn nip(&mut self) -> Val {
        self.swap();
        self.pop()
    }

    // Places `val` under the top stack element.
    #[inline]
    fn tuck(&mut self, mut val: Val) {
        mem::swap(&mut val, self.stack.last_mut().unwrap());
        self.push(val);
    }

    #[inline]
    fn swap(&mut self) {
        let len = self.stack.len();
        self.stack.swap(len-1, len-2);
    }

    #[inline]
    fn dup(&mut self) {
        self.push(self.stack[self.stack.len()-1].clone())
    }

    #[inline]
    fn top(&self) -> &Val {
        self.stack.last().unwrap()
    }

    #[inline]
    fn pop(&mut self) -> Val {
        self.stack.pop().unwrap()
    }

    #[inline]
    fn push_marker(&mut self, marker: usize) {
        self.stack_markers.push(marker);
    }

    #[inline]
    fn pop_marker(&mut self) -> usize {
        self.stack_markers.pop().unwrap()
    }

    #[inline]
    fn current_frame(&self) -> &StackFrame {
        self.stack_frames.last().unwrap()
    }

    #[inline]
    fn current_frame_mut(&mut self) -> &mut StackFrame {
        self.stack_frames.last_mut().unwrap()
    }

    #[inline]
    fn pop_locals(&mut self) {
        let frame = self.current_frame();
        let locals_start = frame.locals_start;
        self.locals_stack.truncate(locals_start);
    }

    fn jump_to_next_case_or_fail<F: FnOnce() -> String>(&mut self, error_message: F) -> Res<usize> {
        match self.current_frame_mut().next_header.take() {
            None => cold(Err(error_message())),
            Some(next_header_index) => {
                self.pop_locals();
                Ok(next_header_index)
            }
        }
    }

    fn load(&mut self, var: Var) -> Val {
        // TODO do all local/closure vars point to non-lists (instead to slices)?
        let frame = self.current_frame();
        match var.place {
            Place::Local => self.locals_stack[frame.locals_start + var.slot].clone(),
            Place::ClosureEnv => frame.explicit.closure_env.borrow()[var.slot].clone(),
        }
    }

    fn consuming_load(&mut self, var: Var) -> Val {
        let mut ret = Val::Int(0);  // TODO
        let frame = self.current_frame_mut();
        match var.place {
            Place::Local => {
                let absolute_slot = frame.locals_start + var.slot;
                mem::swap(&mut ret, &mut self.locals_stack[absolute_slot]);
            }
            Place::ClosureEnv =>
                mem::swap(&mut ret, &mut frame.explicit.closure_env.borrow_mut()[var.slot]),
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
            Place::ClosureEnv => frame.explicit.closure_env.borrow_mut()[dst.slot] = val,
        }
    }

    // `ip` should point to the next instruction to execute.
    #[inline]
    fn is_tail_call(&self, ip: usize) -> bool {
        matches!(self.code.get(ip), Some(Instr::Return))
    }

    // Returns None, meaning `calling` has been called and its return value is on the stack, or
    // Some(code_index), meaning the caller should jump directly to code_index.
    fn call_or_get_jump_target(
        &mut self, ip: usize, calling: Val, arg_spec: ArgSpec
    ) -> Result<Option<usize>, String> {
        match self.set_up_call(calling, arg_spec)? {
            None => Ok(None),
            Some((explicit, arg_spec)) => self.call_explicit_or_get_jump_target(explicit, arg_spec, ip),
        }
    }

    fn call_explicit(&mut self, explicit: ExplicitFunc, arg_spec: ArgSpec) -> Res<()> {
        let code_index = explicit.code_index;
        self.stack_frames.push(StackFrame {
            explicit,
            args_start: self.stack.len() - arg_spec.count_args() as usize,
            arg_spec,
            next_header: None,
            locals_start: self.locals_stack.len(),
            yielding_to: None,
        });
        self.execute(code_index)
    }

    fn call_explicit_or_get_jump_target(
        &mut self, explicit: ExplicitFunc, arg_spec: ArgSpec, ip: usize
    ) -> Res<Option<usize>> {
        if self.is_tail_call(ip) {
            self.pop_locals();
            let locals_start = self.locals_stack.len();
            let code_index = explicit.code_index;
            *self.current_frame_mut() = StackFrame {
                explicit,
                arg_spec,
                args_start: self.stack.len() - arg_spec.count_args() as usize,
                locals_start,
                next_header: None,
                yielding_to: None,
            };
            Ok(Some(code_index))
        } else {
            self.call_explicit(explicit, arg_spec).map(|_| None)
        }
    }

    // Prepare to call `calling`. This will either
    //   - set up the current stack frame and return the code index to jump to, or
    //   - execute a primitive and push its result, returning None.
    //
    // Arguments should be on the stack in reverse order (first argument on top).
    fn set_up_call(
        &mut self, mut calling: Val, mut arg_spec: ArgSpec
    ) -> Res<Option<(ExplicitFunc, ArgSpec)>> {
        loop {
            let function = match &calling {
                Val::Function(rc) => &**rc,
                _ => {
                    let val = self.progressive_index(&calling, arg_spec)?;
                    self.push(val);
                    return Ok(None);
                }
            };

            use UnappliedFunc::*;
            match function {
                Func::Unapplied(Explicit(explicit)) => return Ok(Some((explicit.clone(), arg_spec))),

                Func::Unapplied(Prim(PrimFunc::Rec)) => {
                    let frame = self.current_frame();
                    return Ok(Some((frame.explicit.clone(), arg_spec)));
                }

                // TODO arg specs other than 2. e.g., @[f;...] should return a Partial with f fixed
                // to arity - 1 and the provided args.
                Func::Unapplied(Prim(PrimFunc::Verb(PrimVerb::At) | PrimFunc::Index))
                if arg_spec.is_saturated_at_arity(2) => {
                    arg_spec = ArgSpec::saturated(1);
                    calling = self.pop();
                }

                Func::Unapplied(AdverbDerived { adverb: PrimAdverb::P, operand })
                if arg_spec.is_saturated() => {
                    let arg_count = arg_spec.arity();
                    if arg_count == 0 || arg_count > 2 {
                        // TODO support?
                        return cold_err!("Attempt to call result of adverb `p' on {arg_count} arguments");
                    }
                    if arg_count == 2 {
                        // Take y out of the picture.
                        self.stack.swap_remove(self.stack.len() - 2);
                        arg_spec = ArgSpec::saturated(1);
                    }
                    calling = operand.clone();
                }

                Func::Unapplied(AdverbDerived { adverb: PrimAdverb::Q, operand })
                if arg_spec.is_saturated() => {
                    let arg_count = arg_spec.arity();
                    if arg_count == 0 || arg_count > 2 {
                        // TODO support?
                        return cold_err!("Attempt to call result of adverb `q' on {arg_count} arguments");
                    }
                    if arg_count == 2 {
                        // Take y out of the picture.
                        self.pop();
                        arg_spec = ArgSpec::saturated(1);
                    }
                    calling = operand.clone();
                }

                Func::Unapplied(AdverbDerived { adverb: PrimAdverb::AtColon, .. })
                if arg_spec.raw() == 0b1_1 => return cold_err!("TODO: monadic @:v"),
                Func::Unapplied(AdverbDerived { adverb: PrimAdverb::AtColon, operand })
                if arg_spec.raw() == 0b1_11 => {
                    self.dup();
                    let index = self.index_or_call(&operand, ArgSpec::saturated(1))?;
                    let y = self.nip();

                    // TODO progressive index?
                    calling = self.prim_index(&y, &index)?;
                    arg_spec = ArgSpec::saturated(1);
                }

                Func::Unapplied(AdverbDerived { adverb: PrimAdverb::Dot, operand }) =>
                    calling = operand.clone(),

                Func::Unapplied(AdverbDerived { adverb: PrimAdverb::Tilde, operand })
                if arg_spec.is_saturated_at_arity(1) => {
                    self.dup();
                    arg_spec = arg_spec.pushed(true);
                    calling = operand.clone();
                }
                Func::Unapplied(AdverbDerived { adverb: PrimAdverb::Tilde, operand })
                if arg_spec.is_saturated_at_arity(2) => {
                    self.swap();
                    calling = operand.clone();
                }

                Func::Unapplied(AdverbDerived { adverb: PrimAdverb::Backslash, operand }) 
                if arg_spec.is_saturated_at_arity(2) && self.top().len().unwrap_or(1) == 1 => {
                    let x = self.pop();
                    // Remember, the fold seed becomes the first x argument to the operand.
                    self.tuck(index_or_cycle_val(&x, 0).unwrap());
                    calling = operand.clone();
                }
                Func::Unapplied(AdverbDerived { adverb: PrimAdverb::Backslash, operand })
                if arg_spec.is_saturated_at_arity(1) && matches!(self.top().len(), Some(2)) => {
                    arg_spec = arg_spec.pushed(true);
                    calling = operand.clone();
                    let x = self.pop();
                    self.push(index_or_cycle_val(&x, 1).unwrap());
                    self.push(index_or_cycle_val(&x, 0).unwrap());
                }
                Func::Unapplied(AdverbDerived { adverb: PrimAdverb::Backslash, operand })
                if arg_spec.is_saturated() => {
                    let x = self.pop();
                    let y = if arg_spec.arity() == 2 { Some(self.pop()) } else { None };
                    let result = self.fold_val(&operand, x, y)?;
                    self.push(result);
                    return Ok(None);
                }

                _ => {
                    let val = self.call_func(function, arg_spec)?;
                    self.push(val);
                    return Ok(None);
                }
            }
        }
    }

    fn call_val(&mut self, val: &Val, x: Val, y: Option<Val>) -> Res<Val> {
        if let Some(y) = y { self.call_dyad(val, x, y) } else { self.call_monad(val, x) }
    }

    fn call_monad(&mut self, val: &Val, x: Val) -> Res<Val> {
        self.push(x);
        self.index_or_call(val, ArgSpec::saturated(1))
    }

    fn call_dyad(&mut self, val: &Val, x: Val, y: Val) -> Res<Val> {
        self.push(y);
        self.push(x);
        self.index_or_call(val, ArgSpec::saturated(2))
    }

    fn call_prim_adverb(
        &mut self, adverb: PrimAdverb, operand: &Val, x: Val, maybe_y: Option<Val>
    ) -> Res<Val> {
        use PrimAdverb::*;
        let result = match adverb {
            Runs => match maybe_y {
                Some(y) => self.prim_runs_dyad(operand, x, y)?,
                None => self.prim_runs_monad(operand, x)?,
            }
            Underscore {..} => operand.clone(),
            AtColon => {
                let index = self.call_monad(operand, x.clone())?;
                let elem = self.prim_index(&maybe_y.expect("TODO: monadic @:v"), &index)?;
                self.call_monad(&elem, x)?
            }
            Dot => self.call_val(operand, x, maybe_y)?,
            P => self.call_monad(operand, x)?,
            Q => match maybe_y {
                Some(y) => self.call_monad(operand, y)?,
                None => self.call_monad(operand, x)?,
            }
            SingleQuote => match maybe_y {
                None => for_each(self, operand, x)?,
                Some(y) => match zip_vals(x, y) {
                    Err((x, y)) => self.call_dyad(operand, x, y)?,
                    Ok(iter) => collect_list(
                        iter?.map(|(x_val, y_val)| self.call_dyad(operand, x_val, y_val))
                    )?,
                }
            }
            Backtick => match maybe_y {
                None => for_each(self, operand, x)?,
                Some(y) => match iter_non_atom(x.clone()) {
                    None => self.call_dyad(operand, x, y)?,
                    Some(iter) => collect_list(
                        iter.map(|x_val| self.call_dyad(operand, x_val, y.clone()))
                    )?,
                }
            }
            BacktickColon => match maybe_y {
                None => match iter_non_atom(x.clone()) {
                    None => self.call_monad(operand, x)?,
                    Some(iter) => collect_list(
                        iter.map(|val| self.call_monad(operand, val))
                    )?,
                }
                Some(y) => match iter_non_atom(y.clone()) {
                    None => self.call_dyad(operand, x, y)?,
                    Some(iter) => collect_list(
                        iter.map(|y_val| self.call_dyad(operand, x.clone(), y_val))
                    )?,
                }
            }
            // TODO should ~f just swap f's first two arguments?
            Tilde => match maybe_y {
                None => self.call_dyad(operand, x.clone(), x)?,
                Some(y) => self.call_dyad(operand, y, x)?,
            }
            Backslash => self.fold_val(operand, x, maybe_y)?,
            BackslashColon => self.scan_val(operand, x, maybe_y)?,
        };
        Ok(result)
    }

    fn call_prim_conjunction(
        &mut self,
        prim: PrimConjunction, operand1: &Val, operand2: &Val,
        x: Val, maybe_y: Option<Val>
    ) -> Res<Val> {
        use PrimConjunction::*;
        match prim {
            Compose => {
                let res = self.call_val(operand1, x, maybe_y)?;
                self.call_monad(operand2, res)
            }
            LeftArrow => match operand1.get_explicit() {
                Some(explicit) => {
                    let args_start = self.stack.len();
                    let arg_spec = ArgSpec::saturated(if let Some(y) = maybe_y { self.push(y); 2 }
                                                      else { 1 });
                    self.push(x);
                    self.stack_frames.push(StackFrame {
                        args_start, arg_spec,
                        explicit: explicit.clone(),
                        next_header: None,
                        locals_start: self.locals_stack.len(),
                        yielding_to: Some(operand2.clone()),
                    });
                    self.execute(explicit.code_index)?;
                    Ok(self.pop())
                }
                None => {
                    let res = self.call_val(operand1, x.clone(), maybe_y)?;
                    let new_val = self.call_monad(operand2, res)?;
                    self.invert_f(operand1, x, new_val)
                }
            }
        }
    }

    fn invert_f(&mut self, f: &Val, x: Val, new_val: Val) -> Res<Val> {
        match f {
            Val::Function(func) => match func.as_ref() {
                Func::Unapplied(_) => todo!(),
                Func::PartiallyApplied { func, bound_arg_spec, bound_args } => match func {
                    UnappliedFunc::Prim(PrimFunc::Add) => match bound_arg_spec.raw() {
                        0b1_01 | 0b1_10 => prim::subtract(new_val, &bound_args[0]),
                        _ => todo!(),
                    }
                    _ => todo!("Partially applied inverses"),
                }
            }
            _ => todo!("Inverses for other kinds of vals??"),
        }
    }

    // Args should be on the stack in reverse order
    fn call_prim_func(&mut self, prim: PrimFunc, arg_spec: ArgSpec) -> Res<Val> {
        let result_prim = match prim {
            PrimFunc::Verb(verb) => PrimFunc::resolve_at_arity(verb, arg_spec.arity()),
            _ => Ok(prim),
        };

        // TODO allow primitives to dispatch on partial application
        let result = result_prim.and_then(|prim|
            match arg_spec.raw() {
                // TODO: extend to arbitrary cases
                0b1_1 => {
                    let x = self.pop();
                    self.call_prim_monad(prim, x)
                }
                0b1_11 => {
                    let x = self.pop();
                    let y = self.pop();
                    self.call_prim_dyad(prim, x, y)
                }
                // TODO make sure none of `resolve_at_arity`'s return values can have various
                // arities for consistency.
                _ if arg_spec.has_no_args() => Ok(Val::prim_func(prim)),
                _ => {
                    let args_start = self.stack.len() - arg_spec.count_args() as usize;
                    let bound_args = self.stack.drain(args_start..).rev().collect();
                    let func = Func::PartiallyApplied {
                        func: UnappliedFunc::Prim(prim),
                        bound_arg_spec: arg_spec,
                        bound_args,
                    };
                    Ok(Val::Function(Rc::new(func)))
                }
            }
        );

        result.map_err(|err| format!("Error in `{prim}': {err}"))
    }

    // `prim` should not match Verb(_).
    fn call_prim_monad(&mut self, prim: PrimFunc, x: Val) -> Res<Val> {
        use PrimFunc::*;
        match prim {
            Identity | IdentityLeft | IdentityRight => Ok(x),
            ParseInt => prim::parse_int(&x),
            ParseFloat => prim::parse_float(&x),
            Neg => prim::negate(x),
            Abs => prim::abs(x),
            Not => prim::not(x),
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
            Length => Ok(Val::Int(x.len().unwrap_or(1) as i64)),
            Ints => Ok(iota(&x)),
            Rev => Ok(prim_reverse(x)),
            Ravel => Ok(prim_ravel(x)),
            Inits => Ok(Val::Vals(Rc::new(prim_prefixes(&x)))),
            Tails => Ok(Val::Vals(Rc::new(prim_suffixes(&x)))),
            Where => prim_where(&x),
            Sort => Ok(prim_sort(x, false)),
            SortDesc => Ok(prim_sort(x, true)),
            Asc => Ok(prim_grade(&x, false)),
            Desc => Ok(prim_grade(&x, true)),
            Type => Ok(Val::U8s(Rc::new(prim_type(x.as_val())))),
            GroupIndices => prim::group_indices(x),
            Exit => prim_exit(&x),
            Sum => prim::sum(x, None),
            Rec => {
                self.push(x);
                let explicit = self.current_frame().explicit.clone();
                self.call_explicit(explicit, ArgSpec::saturated(1))?;
                Ok(self.pop())
            }
            Yield => match &self.current_frame().yielding_to {
                None => Ok(Val::prim_func(PrimFunc::Identity)),
                Some(yielding_to) => Ok(Val::conjunction_derived_func(
                    PrimConjunction::Compose, yielding_to.clone(), x
                )),
            }

            _ => todo!("{x:?} {prim:?}")
        }
    }

    // `prim` should not match Verb(_).
    fn call_prim_dyad(&mut self, prim: PrimFunc, x: Val, y: Val) -> Res<Val> {
        use PrimFunc::*;
        match prim {
            Identity | IdentityLeft => Ok(x),
            IdentityRight => Ok(y),
            Or => prim::or(x, y),
            Add => prim::add(x, y),
            Sub => prim::subtract(x, y),
            Mul => prim::multiply(x, y),
            Div => prim::divide(x, y),
            IntDiv => prim::int_divide(x, y),
            Mod => prim::int_mod(x, y),
            Pow => prim::pow(x, y),
            Take => prim_take(x, &y),
            Drop => prim_drop(x, &y),
            Remove => Ok(prim::remove(x, &y)),
            Copy => prim_copy(&x, &y),
            Append => prim_append(x, y),
            Windows => prim::windows(&x, &y),
            Chunks => prim::chunks(&x, &y),
            Match => prim_match(&x, &y),
            NotMatch => prim_not_match(&x, &y),
            // TODO take Val instead of &
            Equal => prim_compare(x, y, |ord| ord == Ordering::Equal),
            NotEqual => prim_compare(x, y, |ord| ord != Ordering::Equal),
            GreaterThan => prim_compare(x, y, |ord| ord > Ordering::Equal),
            GreaterThanEqual => prim_compare(x, y, |ord| ord >= Ordering::Equal),
            LessThan => prim_compare(x, y, |ord| ord < Ordering::Equal),
            LessThanEqual => prim_compare(x, y, |ord| ord <= Ordering::Equal),
            Min => prim_choose_atoms(x, y, Val::le),
            Max => prim_choose_atoms(x, y, Val::ge),
            Index => self.prim_index(&x, &y),
            Find => Ok(Val::Int(prim::find(x.as_val(), y.as_val()))),
            Has => Ok(Val::Int(prim::has(x, y) as i64)),
            In => Ok(Val::Int(prim::has(y, x) as i64)),
            FindSubseq => Ok(Val::I64s(Rc::new(prim_subsequence_starts(x.as_val(), y.as_val())))),

            Sum => prim::sum(x, Some(y)),
            Rec => {
                self.push(y);
                self.push(x);
                let explicit = self.current_frame().explicit.clone();
                self.call_explicit(explicit, ArgSpec::saturated(2))?;
                Ok(self.pop())
            }
            _ => todo!("{x:?} {prim:?} {y:?}"),
        }
    }

    // TODO output formatting (take indent as arg)
    // TODO singleton lists print incorrectly ([1] prints as 1).
    fn prim_fmt(&self, prec: PrecedenceContext, x: &Val, out: &mut String) -> Res<()> {
        use PrecedenceContext::*;
        use UnappliedFunc::*;

        fn parenthesized_if<F>(cond: bool, out: &mut String, f: F) -> Res<()>
        where F: FnOnce(&mut String) -> Res<()> {
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
                Ok(s) => if s.len() == 1 {
                    write_or!(out, "[{s}]")?
                } else {
                    write_or!(out, "{s}")?
                }
                Err(err) => return cold(Err(err.to_string())),
            }
            Val::I64s(is) => {
                if is.is_empty() {
                    write_or!(out, "[]")
                } else if is.len() == 1 {
                    write_or!(out, "[{}]", is[0])
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
                } else if fs.len() == 1 {
                    write_or!(out, "[{}]", fs[0])
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
                Func::Unapplied(Prim(prim)) => write_or!(out, "{prim}")?,
                Func::Unapplied(AdverbDerived { adverb, operand }) => parenthesized_if(
                    prec >= ConjunctionLeftOperand, out, |out| {
                        write_or!(out, "{adverb}")?;
                        self.prim_fmt(AdverbOperand, operand, out)?;
                        Ok(())
                    }
                )?,

                Func::Unapplied(ConjunctionDerived { conjunction, operand1, operand2 }) =>
                    parenthesized_if(prec >= ConjunctionLeftOperand, out, |out| {
                        self.prim_fmt(ConjunctionLeftOperand, operand1, out)?;
                        write_or!(out, "{conjunction}")?;
                        self.prim_fmt(AdverbOperand, operand2, out)?;
                        Ok(())
                    }
                )?,

                // Func::Atop { f_func, g_func } => parenthesized_if(
                //     prec >= Small, out, |out| {
                //         self.prim_fmt(Toplevel, f_func, out)?;
                //         write_or!(out, " ")?;
                //         || -> Res<()> {
                //             if let Val::Function(rc) = g_func {
                //                 if let Func::Bound { func, y } = &**rc {
                //                     self.prim_fmt(Small, func, out)?;
                //                     write_or!(out, " ")?;
                //                     self.prim_fmt(Small, y, out)?;
                //                     return Ok(());
                //                 }
                //             }
                //             self.prim_fmt(Small, g_func, out)
                //         }()?;
                //         Ok(())
                //     }
                // )?,
                // Func::Bound { func, y } => parenthesized_if(prec >= Small, out, |out| {
                //     self.prim_fmt(Small, func, out)?;
                //     write_or!(out, " ")?;
                //     self.prim_fmt(Small, y, out)?;
                //     Ok(())
                // })?,
                // Func::Fork { f_func, h_func, g_func } => parenthesized_if(
                //     prec >= Small, out, |out| {
                //         self.prim_fmt(Toplevel, f_func, out)?;
                //         write_or!(out, " ")?;
                //         self.prim_fmt(Small, h_func, out)?;
                //         write_or!(out, " ")?;
                //         self.prim_fmt(Small, g_func, out)?;
                //         Ok(())
                //     }
                // )?,
                Func::Unapplied(Explicit { .. }) => {
                    // map code index -> tokens?
                    write_or!(out, "{{explicit func}}")?
                }

                Func::PartiallyApplied { func, bound_arg_spec, bound_args } => {
                    self.prim_fmt(ConjunctionLeftOperand,
                                  &Val::Function(Rc::new(Func::Unapplied(func.clone()))),
                                  out)?;
                    write_or!(out, "[")?;
                    let mut args_i = 0;
                    let mut print_one = |has_arg: bool, out: &mut String| -> Res<()> {
                        if has_arg {
                            self.prim_fmt(Toplevel, &bound_args[args_i], out)?;
                            args_i += 1;
                        } else {
                            write_or!(out, "_")?;
                        }
                        Ok(())
                    };

                    let mut iter = bound_arg_spec.into_iter();
                    if let Some(has_arg) = iter.next() {
                        print_one(has_arg, out)?;
                    }
                    for has_arg in iter {
                        write_or!(out, "; ")?;
                        print_one(has_arg, out)?;
                    }
                    write_or!(out, "]")?;
                }
            }
        }
        Ok(())
    }

    // TODO output formatting
    fn prim_debug_fmt(&self, prec: PrecedenceContext, x: &Val, out: &mut String) -> Res<()> {
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
            }
            _ => self.prim_fmt(prec, x, out)?,
        }
        Ok(())
    }

    fn prim_to_string(&self, x: &Val) -> Res<String> {
        let mut s = String::new();
        self.prim_fmt(PrecedenceContext::Toplevel, x, &mut s)?;
        Ok(s)
    }

    fn prim_to_debug_string(&self, x: &Val) -> Res<String> {
        let mut s = String::new();
        self.prim_debug_fmt(PrecedenceContext::Toplevel, x, &mut s)?;
        Ok(s)
    }

    fn prim_print_bytecode(&self, x: &Val) -> Res<()> {
        use UnappliedFunc::*;
        if let Val::Function(rc) = x {
            if let Func::Unapplied(Explicit(ExplicitFunc { code_index, closure_env })) = &**rc {
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
                println!("Code: [");
                let mut return_indices = vec![];
                for i in *code_index..self.code.len() {
                    let instr = &self.code[i];
                    println!("  {}", instr);
                    match instr {
                        Instr::MakeFunc { num_instructions } => return_indices.push(i + *num_instructions),
                        Instr::Return => if return_indices.last().is_some_and(|j| *j == i) {
                            return_indices.pop();
                        } else if let Some(Instr::Header{..} |
                                           Instr::ArgCheck{..}) = self.code.get(i + 1) {
                            continue;
                        } else {
                            break;
                        }
                        _ => {}
                    }
                }
                println!("]");
                return Ok(())
            }
        }
        cold_err!("domain\nx is not an explicit function, so it has no bytecode. x is: {}",
                  self.prim_to_debug_string(x)?)
    }

    pub fn prim_runs_monad(&mut self, f: &Val, x: Val) -> Res<Val> {
        struct Runs<'a> {
            mem: &'a mut Mem,
            f: &'a Val,
            x: Val,
            i: usize,
        }

        impl Iterator for Runs<'_> {
            type Item = Res<Val>;
            fn next(&mut self) -> Option<Self::Item> {
                let current = index_or_cycle_val(&self.x, self.i)?;
                let start = self.i;
                self.i += 1;
                while let Some(next) = index_or_cycle_val(&self.x, self.i) {
                    if next != current {
                        break;
                    }
                    self.i += 1;
                }
                let count = (self.i - start) as i64;
                Some(self.mem.call_dyad(self.f, current, Val::Int(count)))
            }
        }

        collect_list(Runs { mem: self, f, x, i: 0 })
    }

    // TODO returns a singleton list of x and y are atoms; think about this?
    pub fn prim_runs_dyad(&mut self, f: &Val, x: Val, y: Val) -> Res<Val> {
        use Val::*;
        match_lengths(x.len().unwrap_or(1), y.len().unwrap_or(1))?;
        
        fn rc_vec<A: Clone>(vec: &Rc<Vec<A>>, start: usize, one_past_end: usize) -> Rc<Vec<A>> {
            Rc::new(vec.as_slice()[start..one_past_end].to_vec())
        }

        fn sublist(x: &Val, start: usize, one_past_end: usize) -> Val {
            match x {
                atom!() => x.clone(),
                U8s(x) => Val::U8s(rc_vec(x, start, one_past_end)),
                I64s(x) => Val::I64s(rc_vec(x, start, one_past_end)),
                F64s(x) => Val::F64s(rc_vec(x, start, one_past_end)),
                Vals(x) => Val::Vals(rc_vec(x, start, one_past_end)),
            }
        }

        struct Runs<'a> {
            mem: &'a mut Mem,
            f: &'a Val,
            x: Val,
            iter_y: ValIter,
            current: Option<Val>,
            x_start: usize,
        }

        impl Iterator for Runs<'_> {
            type Item = Res<Val>;
            fn next(&mut self) -> Option<Self::Item> {
                let current = self.current.take()?;
                let x_start = self.x_start;
                let mut end = self.iter_y.len;
                while let Some(next) = self.iter_y.next() {
                    if next != current {
                        self.current = Some(next);
                        // iter_y.i is the index after `next`.
                        end = self.iter_y.i - 1;
                        break
                    }
                }
                self.x_start = end;
                Some(self.mem.call_dyad(self.f, sublist(&self.x, x_start, end), current))
            }
        }

        let mut iter_y = iter_val(y);
        let current = iter_y.next();
        collect_list(Runs { mem: self, f, x, iter_y, current, x_start: 0 })
    }

    fn fold_val(&mut self, f: &Val, x: Val, maybe_y: Option<Val>) -> Res<Val> {
        let (mut seed, start) = match maybe_y {
            Some(y) => (y, 0),
            None => match index_or_cycle_val(&x, 0) {
                Some(first) => (first, 1),
                None => return cold_err!("Error: fold with no input"),
            }
        };

        for i in start..x.len().unwrap_or(1) {
            seed = self.call_val(f, seed, index_or_cycle_val(&x, i))?;
        }

        Ok(seed)
    }

    // TODO bad code
    fn scan_val(&mut self, f: &Val, x: Val, maybe_y: Option<Val>) -> Res<Val> {
        let mut seed = match maybe_y {
            Some(y) => match index_or_cycle_val(&x, 0) {
                None => return Ok(y),
                Some(first) => self.call_dyad(f, y, first)?,
            }
            None => match index_or_cycle_val(&x, 0) {
                Some(first) => first,
                None => return Ok(Val::empty_list_of_same_type(&x)),
            }
        };

        collect_list(
            (1 .. x.len().unwrap_or(1)+1)
                .map(|i| match index_or_cycle_val(&x, i) {
                    Some(val) => {
                        let mut new = self.call_dyad(f, seed.clone(), val)?;
                        std::mem::swap(&mut seed, &mut new);
                        Ok(new)
                    }
                    None => Ok(seed.clone())
                })
        )
    }

    // Args should be on the stack in reverse order (first argument on top).
    fn index_or_call(&mut self, f: &Val, arg_spec: ArgSpec) -> Res<Val> {
        use Val::*;
        if let Function(func) = f {
            self.call_func(func, arg_spec)
        } else {
            self.progressive_index(f, arg_spec)
        }
    }

    // Args should be on the stack in reverse order.
    fn call_func(&mut self, f: &Func, arg_spec: ArgSpec) -> Res<Val> {
        use UnappliedFunc::*;
        match f {
            Func::Unapplied(Explicit(explicit)) => {
                self.call_explicit(explicit.clone(), arg_spec)?;
                Ok(self.pop())
            }

            &Func::Unapplied(Prim(prim)) => self.call_prim_func(prim, arg_spec),
            
            // TODO reorganize the adverb stuff
            &Func::Unapplied(AdverbDerived { adverb, ref operand }) => match arg_spec.raw() {
                0b1_1 => {
                    let x = self.pop();
                    self.call_prim_adverb(adverb, operand, x, None)
                }
                0b1_11 => {
                    let x = self.pop();
                    let y = self.pop();
                    self.call_prim_adverb(adverb, operand, x, Some(y))
                }
                _ => {
                    let arity = arg_spec.arity();
                    if arity == 1 || arity == 2 {
                        let args_start = self.stack.len() - arg_spec.count_args() as usize;
                        let bound_args = self.stack.drain(args_start..).rev().collect();
                        let func = Func::PartiallyApplied {
                            func: UnappliedFunc::AdverbDerived { adverb, operand: operand.clone() },
                            bound_arg_spec: arg_spec,
                            bound_args,
                        };
                        Ok(Val::Function(Rc::new(func)))
                    } else {
                        cold_err!("Arity mismatch: expected 1 or 2 args, got {arity}")
                    }
                }
            }

            &Func::Unapplied(ConjunctionDerived { conjunction, ref operand1, ref operand2 }) =>
                match arg_spec.raw() {
                    0b1_1 => {
                        let x = self.pop();
                        self.call_prim_conjunction(conjunction, operand1, operand2, x, None)
                    }
                    0b1_11 => {
                        let x = self.pop();
                        let y = self.pop();
                        self.call_prim_conjunction(conjunction, operand1, operand2, x, Some(y))
                    }
                    _ => {
                        let arity = arg_spec.arity();
                        if arity == 1 || arity == 2 {
                            let args_start = self.stack.len() - arg_spec.count_args() as usize;
                            let bound_args = self.stack.drain(args_start..).rev().collect();
                            let func = Func::PartiallyApplied {
                                func: UnappliedFunc::ConjunctionDerived {
                                    conjunction, operand1: operand1.clone(), operand2: operand2.clone()
                                },
                                bound_arg_spec: arg_spec,
                                bound_args,
                            };
                            Ok(Val::Function(Rc::new(func)))
                        } else {
                            cold_err!("Arity mismatch: expected 1 or 2 args, got {arity}")
                        }
                    }
                }

            &Func::PartiallyApplied { ref func, bound_arg_spec, ref bound_args } => {
                let new_arg_spec = match bound_arg_spec.apply_more(arg_spec) {
                    Ok(new_arg_spec) => new_arg_spec,
                    Err(ArityMismatch { expected, actual }) =>
                        return cold_err!("Arity mismatch: expected {expected} args, got {actual}"),
                };

                // TODO fill stack in place
                let mut new_args = Vec::with_capacity(new_arg_spec.count_args() as usize);
                let mut bound_args_i = 0;
                for (already_bound, newly_bound) in std::iter::zip(bound_arg_spec, new_arg_spec) {
                    if already_bound {
                        new_args.push(bound_args[bound_args_i].clone());
                        bound_args_i += 1;
                    } else if newly_bound {
                        new_args.push(self.pop());
                    }
                }

                self.stack.extend(new_args.into_iter().rev());
                self.call_func(&Func::Unapplied(func.clone()), new_arg_spec)
            }
        }
    }

    // Args should be on the stack in reverse order.
    fn progressive_index(&mut self, x: &Val, arg_spec: ArgSpec) -> Res<Val> {
        let mut iter = arg_spec.into_iter();
        let arg_count = iter.len();
        let ret = self.progressive_index_loop(x, &mut iter, 0);
        self.stack.truncate(self.stack.len() - arg_count);
        ret
    }

    // `i` indexes from the top of the stack, so i=0 is stack.top().
    fn progressive_index_loop(&mut self, x: &Val, arg_spec: &mut ArgSpecIter, i: usize) -> Res<Val> {
        let arg_provided = match arg_spec.next() {
            None => return Ok(x.clone()),
            Some(arg_provided) => arg_provided,
        };

        if arg_provided {
            let arg = self.stack[self.stack.len() - 1 - i].clone();
            for_each_atom_retaining_shape(
                arg,
                |atom| {
                    let elem = self.prim_index(&x, atom)?;
                    self.progressive_index_loop(&elem, arg_spec, i + 1)
                }
            )
        } else {
            let vals = iter_val(x.clone()).map(
                |elem| self.progressive_index_loop(&elem, arg_spec, i)
            );
            collect_list(vals)
        }
    }

    // TODO index by float when int-convertible.
    fn prim_index(&mut self, x: &Val, y: &Val) -> Res<Val> {
        #[cold]
        fn oob(i: i64, len: usize) -> String {
            format!("index out of bounds\nRequested index {i}, but length is {len}")
        }
        fn to_index(i: i64, len: usize) -> usize {
            if i >= 0 { i as usize } else { offset_by(len, i) }  // TODO overflow?
        }
        fn index<A>(slice: &[A], i: i64) -> Res<&A> {
            slice.get(to_index(i, slice.len())).ok_or_else(|| oob(i, slice.len()))
        }
        fn index_atom<A: Clone>(atom: &A, i: i64) -> Res<A> {
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
            (Char(_) | Int(_) | Float(_) | U8s(_) | I64s(_) | F64s(_) | Vals(_), Vals(is)) =>
                collect_list(is.iter().map(|i| self.prim_index(x, i)))?,
            _ => return self.call_monad(&x, y.clone()),
        };
        Ok(val)
    }
}

fn for_each_atom_retaining_shape<F>(x: Val, mut f: F) -> Res<Val>
where F: FnMut(&Val) -> Res<Val> {
    match x {
        atom!() => f(&x),
        Val::U8s(x) => collect_list(x.as_slice().iter().map(|x| f(&Val::Char(*x)))),
        Val::I64s(x) => collect_list(x.as_slice().iter().map(|x| f(&Val::Int(*x)))),
        Val::F64s(x) => collect_list(x.as_slice().iter().map(|x| f(&Val::Float(*x)))),
        Val::Vals(x) => collect_list(x.as_slice().iter().map(f)),
    }
}

// Primitives

fn prim_get_line() -> Res<Val> {
    let mut line = String::new();
    if let Err(err) = std::io::stdin().read_line(&mut line) {
        return Err(cold(err.to_string()))
    }
    Ok(Val::U8s(Rc::new(line.into_bytes())))
}

fn prim_type(x: &Val) -> Vec<u8> {
    x.type_name().as_bytes().to_vec()
}

fn prim_show(x: Val) -> Res<Val> {
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
        _ => return cold_err!("domain\nUnable to show {x:?}"),  // TODO actually we can!
    };
    Ok(ret)
}

fn prim_exit(x: &Val) -> Res<Val> {
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

fn prim_choose_atoms<F>(x: Val, y: Val, f: F) -> Res<Val>
where F: Copy + Fn(&Val, &Val) -> bool {
    Ok(match zip_vals(x, y) {
        Err((x, y)) => if f(&x, &y) { x } else { y }
        Ok(iter) => collect_list(iter?.map(|(x, y)| prim_choose_atoms(x, y, f)))?
    })
}

fn prim_where(x: &Val) -> Res<Val> {
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
        match text.len().checked_sub(pat.len()) {
            None => replicate(0, text.len()).collect(),
            Some(diff) => {
                let mut out = Vec::with_capacity(text.len());
                for i in 0..=diff {
                    let matches = iter::zip(&text[i..], pat).all(|(t, p)| pred(t, p));
                    out.push(matches as i64);
                }
                out.extend(replicate(0, pat.len() - 1));
                out
            }
        }
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
        _ => todo!("{} ?: {}", Val::type_name(x), Val::type_name(y)),
    };

    starts
}

fn prim_read_file(x: &Val) -> Res<Val> {
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
fn prim_take(x: Val, y: &Val) -> Res<Val> {
    let count = match y {
        &Val::Int(i) => i,
        _ => return cold_err!("Invalid right argument {y:?}"),
    };
    take_with_i64(x, count)
}

fn take_with_i64(x: Val, count: i64) -> Res<Val> {
    if x.len().is_some_and(|len| len == 0) { return Ok(x) }

    fn take_iter_from_slice<A: Clone>(count: i64, xs: &[A]) -> impl Iterator<Item=A> + '_ {
        let (start, count) = if count < 0 {
            let abs_count = (-count) as usize;
            (xs.len() - abs_count % xs.len(), abs_count)
        } else {
            (0, count as usize)
        };
        xs.iter().cloned().cycle().skip(start).take(count)
    }

    fn take_vec_from_slice<A: Clone>(count: i64, xs: &[A]) -> Rc<Vec<A>> {
        Rc::new(take_iter_from_slice(count, xs).collect())
    }

    let result = match x.as_val() {
        Val::U8s(cs) => Val::U8s(take_vec_from_slice(count, cs)),
        Val::I64s(is) => Val::I64s(take_vec_from_slice(count, is)),
        Val::F64s(fs) => Val::F64s(take_vec_from_slice(count, fs)),
        Val::Vals(vals) => collect_list(take_iter_from_slice(count, vals).map(Ok::<Val, Empty>)).unwrap(),

        Val::Char(c) => Val::U8s(Rc::new(vec![*c; count.abs() as usize])),
        Val::Int(int) => Val::I64s(Rc::new(vec![*int; count.abs() as usize])),
        Val::Float(float) => Val::F64s(Rc::new(vec![*float; count.abs() as usize])),
        _ => Val::Vals(Rc::new(vec![x; count.abs() as usize])),
    };
    Ok(result)
}

// TODO list y?
fn prim_drop(x: Val, y: &Val) -> Res<Val> {
    let count = match y {
        &Val::Int(i) => i,
        _ => return cold_err!("Invalid right argument {y:?}"),
    };
    let len = x.len().unwrap_or(1) as i64;
    let take_count = if count < 0 {
        len + count
    } else {
        count - len
    };
    take_with_i64(x, take_count)
}

fn prim_copy(x: &Val, y: &Val) -> Res<Val> {
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
    ) -> Res<Vec<A>> {
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

    fn run_many<Y>(x: &Val, y: Y) -> Res<Val>
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

    fn int_as_usize(i: i64) -> Res<usize> {
        i.try_into().map_err(|_| unexpected_y(&Val::Int(i)))
    }

    fn float_as_usize(f: f64) -> Res<usize> {
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

fn prim_append(x: Val, y: Val) -> Res<Val> {
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

    fn extend_or_clone<A: Clone, Y>(mut x: Rc<Vec<A>>, y: Y) -> Rc<Vec<A>>
    where Y: IntoIterator<IntoIter: ExactSizeIterator<Item=A>> {
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

        (x, y) => match (iter_non_atom(x.clone()), iter_non_atom(y.clone())) {
            (None, None) => Vals(Rc::new(vec![x, y])),
            (Some(iter), None) => Vals(many_then_one(iter, y)),
            (None, Some(iter)) => Vals(one_then_many(x, iter)),
            (Some(x), Some(y)) => Vals(many_then_many(x, y)),
        }
    };
    Ok(val)
}

fn prim_ravel(x: Val) -> Val {
    match x {
        atom!() => Val::Vals(Rc::new(vec![x])),
        Val::U8s(_) | Val::I64s(_) | Val::F64s(_) => x,

        // TODO do something about this mess
        Val::Vals(vals) => collect_list(
            { let len = vals.len(); ValIter { x: Val::Vals(vals), i: 0, len } }
            .flat_map(|val| ValIter { x: prim_ravel(val.clone()), i: 0, len: val.len().unwrap_or(1) })
            .map(|val| -> Result<Val, ()> { Ok(val) })
        ).unwrap(),
    }
}

fn prim_compare<F: Fn(Ordering) -> bool + Copy>(x: Val, y: Val, op: F) -> Res<Val> {
    use Val::*;
    let result = match zip_vals(x, y) {
        Err((x, y)) => Int(op(x.cmp(&y)) as i64),
        // TODO we already know this will consist of bits
        Ok(iter) => collect_list(iter?.map(|(x, y)| prim_compare(x, y, op)))?
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
fn prim_match(x: &Val, y: &Val) -> Res<Val> {
    Ok(Val::Int((x == y) as i64))
}

fn prim_not_match(x: &Val, y: &Val) -> Res<Val> {
    Ok(Val::Int((x != y) as i64))
}

fn replicate<A: Clone>(a: A, n: usize) -> impl Iterator<Item=A> {
    std::iter::repeat(a).take(n)
}

fn replicate_with_float<A: Clone>(a: A, f: f64) -> Res<impl Iterator<Item=A>> {
    match float_as_int(f) {
        Some(n) => replicate_with_i64(a, n),
        _ => cold_err!("domain\nExpected non-negative integer, got {f}"),
    }
}

fn replicate_with_i64<A: Clone>(a: A, n: i64) -> Res<impl Iterator<Item=A>> {
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

// TODO dyadic
fn for_each(mem: &mut Mem, f: &Val, x: Val) -> Res<Val> {
    use crate::ops::{SingleValConsumer, IsVal};

    struct ForEach<'a> { mem: &'a mut Mem, f: &'a Val }
    impl<'a> SingleValConsumer for ForEach<'a> {
        fn eat_val(&mut self, val: Val) -> Res<Val> {
            self.mem.call_monad(&self.f, val)
        }
        fn eat_val_ref(&mut self, val: &Val) -> Res<Val> {
            self.mem.call_monad(&self.f, val.clone())
        }
    }
    x.dispatch_for_each(ForEach { mem, f })
}

fn to_string_or<A: ToString>(thing: Option<A>, default: &str) -> String {
    thing.map_or_else(|| default.to_string(), |x| x.to_string())
}

fn offset_by(n: usize, offset: i64) -> usize {
    (n as i64 + offset) as usize
}
