// TODO there are definitely some places here where we don't track the refcounts
// correctly
//
// TODO make refcounts atomic; read
// https://doc.rust-lang.org/nomicon/atomics.html, links at
// https://stackoverflow.com/questions/30407121/which-stdsyncatomicordering-to-use

use std::{
    rc::Rc,
};

use crate::bytecode::*;
use crate::lex::*;
use crate::util::*;

// Maybe we go back to moving lists to the heap when they're sliced (and in
// general moving vals to the heap when they're referenced).
// Does this work for e.g. x '# 2
//   Would want to increment rc when creating slices
//   Could re-traverse after whole operation and inc rcs?
//    But how would we know which refs were there before and which are new?
//    Could dec beforehand w/o deletion and then inc all after
//
// Preventing cycles: including element at equal or greater depth triggers copy?

// alt formulation: two heaps, one for most vals & one for array vecs
//
// alt: Slice becomes raw heap slot (for rc)+ptr to start+len. we just agree that vector elements can't be moved (or mutated?) through a non-unique reference

/*
problem: some vals need to go on heap for refs



o1: put everything on heap, just use heap indices as values
  pro
o2: move val to heap when ref is taken
  [1;2;3]
o3: arrays (ie dyn sized objects) always on heap, others off or on (default off, values move on when ref taken)
actually, maybe o2 since arrays aren't *actually* expensive to move
the difference is that capturing lists by value would always elicit a move to heap, unlike slices/ints/refs, so why not put them there in the first place?

[1;2;3]->a  \ a is Val:Ints(Vec)
a#2  \ Val:Ints moves to heap w/ rc 2, a becomes Val:Slice(0,len,slot), result is Val:Slice(0,2,slot)

a: retaking reference to array x

[1;2;3]->a
a#2->b
c:ref a

a:points to array
b:slice of a
c: 


question:
a:1
F:{a:a+1}
G:F
[]F  \ a=2
[]G  \ ? Does G share F's environment? If not, then you can't mutate closure envs by eg sending them to functions

f{[]X}

maybe closure envs are refs by default

*/

type RcVal = Rc<Val>;

// TODO Ref
// TODO write Clone instance manually and panic on List
#[derive(Debug, Clone)]
enum Val {
    Int(i64),  // TODO TwoInts, ThreeInts
    Char(u8),

    // TODO switch this out for the non-token type (some primitives won't have
    // token representations)
    PrimFunc(PrimVerb),

    // TODO decide if closures should refer to a shared environment or be value
    // types (copying copies environment. Currently, we hold a reference to the env.
    //   a:1; F:{a:a+1}; G:F
    //   []F  \ 2
    //   []G  \ Currently this returns 3. Should G have its own copy of the env, making this 2?
    ExplicitFunc {
        // Always points to Val::Vals.
        closure_env: RcVal,
        // The function's first instruction (ALWAYS points after MakeFunc, so
        // you can get the function's instruction count with
        // code[func.code_index-1]
        code_index: usize,
    },
    
    AdverbDerivedFunc {
        adverb: PrimAdverb,
        operand: RcVal,
    },

    I64s(Vec<i64>),
    U8s(Vec<u8>),
    Vals(Vec<RcVal>),
}

impl Val {
    fn as_val(&self) -> &Self { &self }

    fn len(&self) -> Option<usize> {
        use Val::*;
        match self {
            I64s(vec) => Some(vec.len()),
            U8s(vec) => Some(vec.len()),
            Vals(vec) => Some(vec.len()),
            Int(_) | Char(_) | PrimFunc(_) | ExplicitFunc {..} | AdverbDerivedFunc {..} => None,
        }
    }
    
    fn length_mismatch(&self, other: &Val) -> Option<(usize, usize)> {
        let len1 = self.len()?;
        let len2 = other.len()?;
        if len1 == len2 { None } else { Some((len1, len2)) }
    }
}

// TODO Refs
#[derive(Debug, Clone)]
enum List {
    I64s(Vec<i64>),
    U8s(Vec<u8>),
    Vals(Vec<Val>),
}    

impl List {
    fn len(&self) -> usize {
        match self {
            List::I64s(v) => v.len(),
            List::U8s(v) => v.len(),
            List::Vals(v) => v.len(),
        }
    }
}

#[derive(Debug)]
struct StackFrame {
    // Always points to Val::Vals
    closure_env: RcVal,

    // Index into locals_stack. Local slots are offsets from this index.
    locals_start: usize,

    // Code index after Call; or, if usize::MAX, return from the execute
    // loop. TODO replace with Option<NonZeroUsize>?
    ret_addr: usize,
}

pub struct Mem<'a> {
    code: &'a [Instr],

    // TODO can we merge locals_stack, subject1, subject2, and verb?
    subject1: Vec<RcVal>,
    verb: Vec<RcVal>,
    subject2: Option<RcVal>,
    
    // Stack of local scopes
    locals_stack: Vec<RcVal>,
    stack_frames: Vec<StackFrame>,

    // TODO intern small ints
    zero: RcVal,
}

impl<'a> Mem<'a> {
    pub fn new() -> Self {
        Self {
            code: &[],
            subject1: vec![],
            verb: vec![],
            subject2: None,
            locals_stack: vec![],  // TODO stdlib?
            stack_frames: vec![StackFrame {
                closure_env: Rc::new(Val::Vals(vec![])),
                locals_start: 0,
                ret_addr: usize::MAX,
            }],

            zero: Rc::new(Val::Int(0)),
        }
    }

    pub fn set_code(&mut self, code: &'a [Instr]) {
        self.code = code;
    }

    fn store(&mut self, dst: Var, val: RcVal) {
        let frame = self.stack_frames.last().unwrap();
        match dst.place {
            Place::Local => {
                let absolute_slot = frame.locals_start + dst.slot;
                if absolute_slot >= self.locals_stack.len() {
                    self.locals_stack.resize(absolute_slot + 1, self.zero.clone());  // TODO is zero right?
                }
                self.locals_stack[absolute_slot] = val;
            }
            Place::ClosureEnv => match &*frame.closure_env {
                // SAFETY this seems bad, but we want closures to share an
                // environment. We can restructure how closures work in a way
                // that Rust likes, but this "works" for now.
                Val::Vals(vals) => unsafe { (vals.as_ptr() as *mut RcVal).add(dst.slot).replace(val); }
                // TODO if a closure env consists of one value, it's fine not to
                // use a whole list.
                // TODO consolidate closure envs of all-matching types to
                // e.g. Int64s instead of Vals.
                _ => unreachable!(),  
            }
        }
    }

    pub fn execute(&mut self, mut ip: usize) -> Result<u8, String> {
        use Instr::*;
        while ip < self.code.len() {
            match self.code[ip] {
                Nop => ip += 1,
                Halt { exit_status } => return Ok(exit_status),
                MakeClosure { num_closure_vars } => {
                    ip += 1;
                    let num_instructions = match &self.code[ip] {
                        MakeFunc { num_instructions } => { ip += 1; num_instructions }
                        bad => panic!("Malformed code at ip {ip}: expected MakeFunc after MakeClosure, but found {bad:?}"),
                    };

                    let code_index = ip;  // First instruction of function body

                    ip += num_instructions;
                    let mut closure_data = Vec::with_capacity(num_closure_vars);
                    for _ in 0..num_closure_vars {
                        if let PushVar { src } = self.code[ip] {
                            ip += 1;
                            // TODO use e.g. List::I64s if closure vals are all ints
                            closure_data.push(self.load(src).clone());
                        } else {
                            panic!("Malformed code at ip {ip}: expected PushVar after MakeClosure, but found {:?}", self.code[ip]);
                        }
                    }
                    let closure_env = RcVal::new(Val::Vals(closure_data));
                    self.verb.push(RcVal::new(Val::ExplicitFunc { closure_env, code_index }));
                }
                MakeFunc { num_instructions } => {
                    self.subject1.push(RcVal::new(
                        Val::ExplicitFunc {
                            closure_env: RcVal::new(Val::Vals(vec![])),
                            code_index: ip + 1,
                        }
                    ));
                    ip += num_instructions;
                }
                AllocLocals { num_locals } => {
                    ip += 1;
                    // TODO is 0 the right thing to fill with here?
                    self.locals_stack.resize(self.locals_stack.len() + num_locals, self.zero.clone());
                }
                Return => {
                    let frame = self.stack_frames.pop().unwrap();
                    self.locals_stack.truncate(frame.locals_start);
                    self.verb.pop();

                    if frame.ret_addr == usize::MAX { return Ok(0); }
                    else { ip = frame.ret_addr; }
                }
                PushLiteralInteger(value) => {
                    ip += 1;
                    self.subject1.push(RcVal::new(Val::Int(value)));
                }
                PushVar { src } => {
                    ip += 1;
                    let val = self.load(src).clone();
                    self.subject1.push(val);
                }
                PushPrimVerb { prim } => {
                    ip += 1;
                    self.verb.push(RcVal::new(Val::PrimFunc(prim)));
                }
                PushVerb { src } => {
                    ip += 1;
                    let val = self.load(src).clone();
                    self.verb.push(val);
                }
                // TODO call1/call2
                Call => {
                    let verb = self.verb.pop().unwrap();
                    match &*verb {
                        Val::PrimFunc(prim) => {
                            ip += 1;
                            self.call_prim_verb(*prim)?;
                        }

                        Val::AdverbDerivedFunc { adverb, operand } => {
                            ip += 1;
                            self.call_adverb_derived_func(*adverb, operand.clone())?;
                        }

                        &Val::ExplicitFunc { ref closure_env, code_index } => {
                            self.stack_frames.push(StackFrame {
                                closure_env: closure_env.clone(),
                                locals_start: self.locals_stack.len(),
                                ret_addr: ip + 1,
                            });

                            // Keep in mind, Return pops the verb.
                            // TODO Return doesn't pop the verb & we put code_index into StackFrame
                            self.verb.push(verb);
                            let x = self.subject1.pop().unwrap();
                            self.locals_stack.push(x);
                            self.locals_stack.push(
                                self.subject2.take().unwrap_or(self.zero.clone())  // TODO monads won't even mention y, but locals slot 1 is always y
                            );
                            ip = code_index;
                        }

                        Val::Int(_) | Val::Char(_) | Val::I64s(_) | Val::U8s(_) | Val::Vals(_) => {
                            ip += 1;
                            self.subject1.pop();
                            self.subject2.take();
                            self.subject1.push(verb);
                        }
                    }
                },
                CallPrimVerb { prim } => {
                    ip += 1;
                    self.call_prim_verb(prim)?;
                }
                Pop => {
                    ip += 1;
                    self.subject1.pop();
                    
                    // TODO find out if we discard verb here
                    if self.subject2.is_some() {
                        todo!("Pop should discard subject2")
                    }
                }
                PopVerb => {
                    ip += 1;
                    self.verb.pop();
                } 
                PopToSubject2 => {
                    ip += 1;
                    self.subject2 = self.subject1.pop();
                }
                PopToVerb => {
                    ip += 1;
                    self.verb.push(self.subject1.pop().unwrap());
                }
                MoveVerbToSubject1 => {
                    ip += 1;
                    self.subject1.push(self.verb.pop().unwrap());
                }
                StoreTo { dst } => {
                    ip += 1;
                    self.store(dst, self.subject1.last().unwrap().clone());
                }
                StoreVerbTo { dst } => {
                    ip += 1;
                    self.store(dst, self.verb.last().unwrap().clone());
                }
                CallPrimAdverb { prim: adverb } => {
                    ip += 1;
                    let operand = self.verb.pop().unwrap();
                    self.verb.push(RcVal::new(Val::AdverbDerivedFunc { adverb, operand }));
                }
                MakeString { num_bytes } => todo!("string literals"),
                LiteralBytes { bytes } => todo!("char literals"),                
                CollectToArray { num_elems } => {
                    ip += 1;
                    let mut all_ints = true;
                    let mut all_chars = true;
                    for elem in &self.subject1[(self.subject1.len() - num_elems)..] {
                        all_ints &= matches!(&**elem, Val::Int(_));
                        all_chars &= matches!(&**elem, Val::Char(_));
                    }

                    let mut elems = self.subject1.drain((self.subject1.len() - num_elems)..);
                    let list_val = if all_ints {
                        Val::I64s(elems.map(|elem| irrefutable!(*elem, Val::Int(int) => int)).collect())
                    } else if all_chars {
                        Val::U8s(elems.map(|elem| irrefutable!(*elem, Val::Char(ch) => ch)).collect())
                    } else {
                        Val::Vals(elems.collect())
                    };
                    self.subject1.push(RcVal::new(list_val))
                }
            }
        }
        Ok(0)
    }

    fn call_adverb_derived_func(&mut self, adverb: PrimAdverb, operand: RcVal) -> Result<(), String> {
        let x = self.subject1.pop().unwrap();
        let y = self.subject2.take();
        let ret = self.call_prim_adverb(adverb, operand, x, y)?;
        self.subject1.push(ret);
        Ok(())
    }

    fn call_val(&mut self, val: RcVal, x: RcVal, y: Option<RcVal>) -> Result<RcVal, String> {
        let result = match &*val {
            &Val::ExplicitFunc { ref closure_env, code_index } => {
                let frame = StackFrame {
                    closure_env: closure_env.clone(),
                    locals_start: self.locals_stack.len(),
                    ret_addr: usize::MAX,
                };
                self.stack_frames.push(frame);
                self.verb.push(val);
                self.locals_stack.push(x);
                self.locals_stack.push(y.unwrap_or(self.zero.clone()));
                self.execute(code_index)?;  // TODO throwing away exit status?
                self.subject1.pop().unwrap()
            }
            Val::AdverbDerivedFunc { adverb, operand } =>
                self.call_prim_adverb(*adverb, operand.clone(), x, y)?,
            &Val::PrimFunc(prim) => if let Some(y) = y {
                self.call_prim_dyad(prim, x, y)?
            } else {
                self.call_prim_monad(prim, x)?
            },
            Val::Int(_) | Val::Char(_) | Val::I64s(_) | Val::U8s(_) | Val::Vals(_) => val,
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
            Dot => self.call_val(operand, x, maybe_y)?,
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
            Tilde => match maybe_y {
                None => self.call_val(operand, x.clone(), Some(x))?,
                Some(y) => self.call_val(operand, y, Some(x))?,
            }
            _ => todo!("Implement adverb {adverb}"),
        };
        Ok(result)
    }

    fn load(&mut self, var: Var) -> &RcVal {
        // TODO do all local/closure vars point to non-lists (instead to slices)?
        let frame = self.stack_frames.last().unwrap();
        match var.place {
            Place::Local => &self.locals_stack[frame.locals_start + var.slot],
            Place::ClosureEnv => match &*frame.closure_env {
                Val::Vals(vals) => &vals[var.slot],
                _ => unreachable!(),
            }
        }
    }

    fn call_prim_adverb_dyad(&mut self, adverb: PrimAdverb, operand_heap_slot: usize, x: Val, y: Val) -> Result<Val, String> {
        todo!()
    }

    fn call_prim_verb(&mut self, prim: PrimVerb) -> Result<(), String> {
        let x = self.subject1.pop().unwrap();  // TODO drop val?
        let ret = if let Some(y) = self.subject2.take() {  // TODO drop val?
            self.call_prim_dyad(prim, x, y)?
        } else {
            self.call_prim_monad(prim, x)?
        };
        self.subject1.push(ret);
        Ok(())
    }

    fn call_prim_monad(&mut self, v: PrimVerb, x: RcVal) -> Result<RcVal, String> {
        use PrimVerb::*;
        match v {
            Print => { self.print_val(&x)?; Ok(x) }
            _ => todo!("{x:?} {v:?}")
        }
    }

    fn call_prim_dyad(&mut self, v: PrimVerb, x: RcVal, y: RcVal) -> Result<RcVal, String> {
        use PrimVerb::*;
        match v {
            Plus => add_vals(&*x, &*y),  // TODO take x,y by ref above?
            Minus => subtract_vals(&*x, &*y),
            Asterisk => multiply_vals(&*x, &*y),
            Slash => integer_divide_vals(&*x, &*y),
            Divide => subtract_vals(&*x, &*y),
            Snoc => todo!(),
            _ => todo!("{x:?} {v:?} {y:?}"),
        }
    }
    
    // TODO output formatting
    fn print_val(&self, x: &Val) -> Result<(), String> {
        match x {
            Val::Int(i) => println!("{i}"),
            Val::Char(c) => println!("{c}"),
            Val::PrimFunc(prim) => println!("{prim}"),
            Val::I64s(ints) => println!("{ints:?}"),
            Val::U8s(chars) => match std::str::from_utf8(chars) {  // TODO unicode
                Ok(s) => println!("{s}"),
                Err(err) => return Err(err.to_string()),
            },
            Val::Vals(vals) => {
                println!("[");
                for val in vals { self.print_val(&*val)? }
                println!("]");
            }
            Val::AdverbDerivedFunc { adverb, operand } => {
                todo!("implement adverb-derived verb printing")
            }
            Val::ExplicitFunc { closure_env, code_index } => {
                todo!("implement explicit func printing")  // map code index -> tokens?
            }
        }
        
        Ok(())
    }

    // Slices and lists index normally, and atoms cycle.
    // fn get_nth_item(&self, x: &Val, i: usize) -> Option<&Val> {
    //     use Val::*;
    //     match x {
    //         Int(_) |
    //         Char(_) |
    //         PrimFunc(_) |
    //         ExplicitFunc {..} |
    //         AdverbDerivedFunc {..} => Some(x),
    //         List(list) => index_list_as_val(list.deref, i),
    //         Slice(handle) => irrefutable!(&self.heap[handle.heap_slot],
    //                                       List(list) => index_list_as_val(list.deref(), handle.start_offset + i)),
    //     }
    // }

    
    // TODO
    // fn index(&mut self, x: RcVal, y: RcVal) -> Result<RcVal, String> {
    //     match x {
    //         &PrimFunc(prim) => self.call_prim_dyad(prim, x, y),
    //         &ExplicitFunc { 
    //     }
    // }
}

// TODO
// fn as_vals<'a>(list: &'a List) -> impl Iterator<Item=Cow<'a, Val>> {
//     match list {
//         List::I64s(x) => x.iter().map(|i| Cow::Owned(Val::Int(*i))),
//         List::U8s(x) => x.iter().map(|ch| Cow::Owned(Val::Char(*ch))),
//         List::Vals(x) => x.iter().map(|val| Cow::Borrowed(val)),
//     }
// }

// fn index_list_as_val(list: &List, i: usize) -> Option<&Val> {
//     match list {
//         List::I64s(ints) => ints.get(i).copied().map(Val::Int),
//         List::U8s(chars) => Val::Char(chars.get(i)?),
//         List::Vals(vals) => vals.get(i),
//     }        
// }

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
        I64s(ints) => RcVal::new(Val::Int(*ints.get(i)?)),
        U8s(chars) => RcVal::new(Val::Char(*chars.get(i)?)),
        Vals(vals) => vals.get(i)?.clone(),
        Int(_) | Char(_) | PrimFunc(_) | ExplicitFunc{..} | AdverbDerivedFunc{..} => val.clone(),
    })
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

fn zip_exact<X: ExactSizeIterator, Y: ExactSizeIterator>(x: X, y: Y) -> Result<impl Iterator<Item=(X::Item, Y::Item)>, String> {
    let xlen = x.len();
    let ylen = y.len();
    if xlen == ylen {
        Ok(x.zip(y))
    } else {
        Err(format!("Length mismatch: {xlen} vs {ylen}"))
    }
}

fn match_length(xlen: usize, ylen: usize) -> Result<(), String> {
    if xlen == ylen { return Ok(()); }
    // TODO include name/position of verb
    Err(format!("Length mismatch: {xlen} vs {ylen}"))
}

// Primitives

fn add_vals(x: &Val, y: &Val) -> Result<RcVal, String> {
    use Val::*;
    let result = match (x, y) {
        (Int(i), Int(j)) => Int(i + j),
        (Int(i), Char(c)) | (Char(c), Int(i)) => Char((*c as i64 + i) as u8),
        (I64s(ints), Int(i)) | (Int(i), I64s(ints)) => I64s(
            ints.iter().map(|int| *int + *i).collect()
        ),
        (U8s(chars), Int(i)) | (Int(i), U8s(chars)) => U8s(
            chars.iter().map(|ch| (*ch as i64 + *i) as u8).collect()
        ),
        (I64s(x_ints), I64s(y_ints)) => I64s(
            zip_exact(x_ints.iter(), y_ints.iter())?.map(|(i, j)| *i + *j).collect()
        ),
        (U8s(chars), I64s(ints)) |
        (I64s(ints), U8s(chars)) => U8s(
            zip_exact(chars.iter(), ints.iter())?.map(|(ch, i)| (*ch as i64 + *i) as u8).collect()
        ),
        (Vals(vals), Int(i)) | (Int(i), Vals(vals)) => Vals(
            vals.iter().map(|rc_val| add_vals(&*rc_val, &Int(*i))).collect::<Result<_, _>>()?
        ),
        (Vals(vals), I64s(ints)) |
        (I64s(ints), Vals(vals)) => Vals(
            zip_exact(vals.iter(), ints.iter())?
                .map(|(rc_val, int)| add_vals(&*rc_val, &Val::Int(*int)))
                .collect::<Result<_, _>>()?
        ),
        (Vals(vals), U8s(chars)) |
        (U8s(chars), Vals(vals)) => Vals(
            zip_exact(vals.iter(), chars.iter())?
                .map(|(rc_val, ch)| add_vals(&*rc_val, &Val::Char(*ch)))
                .collect::<Result<_, _>>()?
        ),
        (Vals(x_vals), Vals(y_vals)) => Vals(
            zip_exact(x_vals.iter(), y_vals.iter())?
                .map(|(x_rc_val, y_rc_val)| add_vals(&*x_rc_val, &*y_rc_val))
                .collect::<Result<_, _>>()?
        ),
        _ => return Err(format!("Error in `+': Can't add {x:?} and {y:?}")),
    };
    Ok(RcVal::new(result))
}

fn subtract_vals(x: &Val, y: &Val) -> Result<RcVal, String> {
    use Val::*;
    let result = match (x, y) {
        (Int(i), Int(j)) => Int(i - j),
        (Char(c), Int(i)) => Char((*c as i64 - i) as u8),
        (Char(x_c), Char(y_c)) => Int(*x_c as i64 - *y_c as i64),
        (I64s(ints), Int(i)) => I64s(
            ints.iter().map(|int| *int - *i).collect()
        ),
        (Int(i), I64s(ints)) => I64s(
            ints.iter().map(|int| *i - *int).collect()
        ),
        (U8s(chars), Int(i)) => U8s(
            chars.iter().map(|ch| (*ch as i64 - *i) as u8).collect()
        ),
        (I64s(x_ints), I64s(y_ints)) => I64s(
            zip_exact(x_ints.iter(), y_ints.iter())?.map(|(i, j)| *i - *j).collect()
        ),
        (U8s(chars), I64s(ints)) => U8s(
            zip_exact(chars.iter(), ints.iter())?.map(|(ch, i)| (*ch as i64 - *i) as u8).collect()
        ),
        (U8s(x_chars), U8s(y_chars)) => I64s(
            zip_exact(x_chars.iter(), y_chars.iter())?.map(|(x, y)| *x as i64 - *y as i64).collect()
        ),
        (Vals(vals), Int(i)) => Vals(
            vals.iter().map(|rc_val| subtract_vals(&*rc_val, &Int(*i))).collect::<Result<_, _>>()?
        ),
        (Int(i), Vals(vals)) => Vals(
            vals.iter().map(|rc_val| subtract_vals(&Int(*i), &*rc_val)).collect::<Result<_, _>>()?
        ),
        (Vals(vals), I64s(ints)) => Vals(
            zip_exact(vals.iter(), ints.iter())?
                .map(|(rc_val, int)| subtract_vals(&*rc_val, &Val::Int(*int)))
                .collect::<Result<_, _>>()?
        ),
        (I64s(ints), Vals(vals)) => Vals(
            zip_exact(ints.iter(), vals.iter())?
                .map(|(int, rc_val)| subtract_vals(&Val::Int(*int), &*rc_val))
                .collect::<Result<_, _>>()?
        ),
        (Vals(vals), U8s(chars)) => Vals(
            zip_exact(vals.iter(), chars.iter())?
                .map(|(rc_val, ch)| subtract_vals(&*rc_val, &Val::Char(*ch)))
                .collect::<Result<_, _>>()?
        ),
        (U8s(chars), Vals(vals)) => Vals(
            zip_exact(chars.iter(), vals.iter())?
                .map(|(ch, rc_val)| subtract_vals(&Val::Char(*ch), &*rc_val))
                .collect::<Result<_, _>>()?
        ),
        (Vals(x_vals), Vals(y_vals)) => Vals(
            zip_exact(x_vals.iter(), y_vals.iter())?
                .map(|(x_rc_val, y_rc_val)| subtract_vals(&*x_rc_val, &*y_rc_val))
                .collect::<Result<_, _>>()?
        ),
        _ => return Err(format!("Error in `-': Can't subtract {x:?} and {y:?}")),
    };
    Ok(RcVal::new(result))
}

fn multiply_vals(x: &Val, y: &Val) -> Result<RcVal, String> {
    use Val::*;
    let result = match (x, y) {
        (Int(i), Int(j)) => Int(i * j),
        (Int(i), Char(c)) | (Char(c), Int(i)) => Char((*c as i64 * i) as u8),
        (I64s(ints), Int(i)) | (Int(i), I64s(ints)) => I64s(
            ints.iter().map(|int| *int * *i).collect()
        ),
        (I64s(x_ints), I64s(y_ints)) => I64s(
            zip_exact(x_ints.iter(), y_ints.iter())?.map(|(i, j)| *i * *j).collect()
        ),
        (Vals(vals), Int(i)) | (Int(i), Vals(vals)) => Vals(
            vals.iter().map(|rc_val| multiply_vals(&*rc_val, &Int(*i))).collect::<Result<_, _>>()?
        ),
        (Vals(vals), I64s(ints)) |
        (I64s(ints), Vals(vals)) => Vals(
            zip_exact(vals.iter(), ints.iter())?
                .map(|(rc_val, int)| multiply_vals(&*rc_val, &Val::Int(*int)))
                .collect::<Result<_, _>>()?
        ),
        (Vals(x_vals), Vals(y_vals)) => Vals(
            zip_exact(x_vals.iter(), y_vals.iter())?
                .map(|(x_rc_val, y_rc_val)| multiply_vals(&*x_rc_val, &*y_rc_val))
                .collect::<Result<_, _>>()?
        ),
        _ => return Err(format!("Error in `+': Can't add {x:?} and {y:?}")),
    };
    Ok(RcVal::new(result))
}

fn integer_divide_vals(x: &Val, y: &Val) -> Result<RcVal, String> {
    use Val::*;
    let result = match (x, y) {
        (Int(i), Int(j)) => Int(i / j),
        (I64s(ints), Int(i)) => I64s(
            ints.iter().map(|int| *int / *i).collect()
        ),
        (Int(i), I64s(ints)) => I64s(
            ints.iter().map(|int| *i / *int).collect()
        ),
        (Vals(vals), Int(i)) => Vals(
            vals.iter().map(|rc_val| integer_divide_vals(&*rc_val, &Int(*i))).collect::<Result<_, _>>()?
        ),
        (Int(i), Vals(vals)) => Vals(
            vals.iter().map(|rc_val| integer_divide_vals(&Int(*i), &*rc_val)).collect::<Result<_, _>>()?
        ),
        (I64s(x_ints), I64s(y_ints)) => I64s(
            zip_exact(x_ints.iter(), y_ints.iter())?.map(|(i, j)| *i / *j).collect()
        ),
        (Vals(vals), I64s(ints)) => Vals(
            zip_exact(vals.iter(), ints.iter())?
                .map(|(rc_val, int)| integer_divide_vals(&*rc_val, &Val::Int(*int)))
                .collect::<Result<_, _>>()?
        ),
        (I64s(ints), Vals(vals)) => Vals(
            zip_exact(ints.iter(), vals.iter())?
                .map(|(int, rc_val)| integer_divide_vals(&Val::Int(*int), &*rc_val))
                .collect::<Result<_, _>>()?
        ),
        (Vals(x_vals), Vals(y_vals)) => Vals(
            zip_exact(x_vals.iter(), y_vals.iter())?
                .map(|(x_rc_val, y_rc_val)| integer_divide_vals(&*x_rc_val, &*y_rc_val))
                .collect::<Result<_, _>>()?
        ),
        _ => return Err(format!("Error in `*': Can't multiply {x:?} and {y:?}")),
    };
    Ok(RcVal::new(result))
}

fn collect_list<E, I: Iterator<Item=Result<RcVal, E>>>(mut it: I) -> Result<Val, E> {
    enum List {
        I64s(Vec<i64>),
        U8s(Vec<u8>),
        Vals(Vec<RcVal>),
    }

    let cap = match it.size_hint() {
        (lower, None) => lower,
        (_, Some(upper)) => upper,
    };

    let mut list = match it.next() {
        None => return Ok(Val::I64s(vec![])),
        Some(Err(err)) => return Err(err),
        Some(Ok(rc_val)) => match rc_val.as_val() {
            Val::Int(i) => {
                let mut vec = Vec::with_capacity(cap);
                vec.push(*i);
                List::I64s(vec)
            }
            Val::Char(c) => {
                let mut vec = Vec::with_capacity(cap);
                vec.push(*c);
                List::U8s(vec)
            }
            _ => {
                let mut vec = Vec::with_capacity(cap);
                vec.push(rc_val);
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
                _ => {
                    let mut vals: Vec<RcVal> =
                        ints.drain(..).map(|i| RcVal::new(Val::Int(i))).collect();
                    vals.reserve(cap - vals.len());
                    vals.push(val);
                    list = List::Vals(vals);
                }
            },
            List::U8s(chars) => match &*val {
                Val::Char(c) => chars.push(*c),
                _ => {
                    let mut vals: Vec<RcVal> =
                        chars.drain(..).map(|c| RcVal::new(Val::Char(c))).collect();
                    vals.reserve(cap - vals.len());
                    vals.push(val);
                    list = List::Vals(vals);
                }
            },
            List::Vals(vals) => vals.push(val),
        }
    }

    Ok(match list {
        List::I64s(ints) => Val::I64s(ints),
        List::U8s(chars) => Val::U8s(chars),
        List::Vals(vals) => Val::Vals(vals),
    })
}
