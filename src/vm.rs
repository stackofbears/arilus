use std::{
    cmp::Ordering,
    ops::{Div, Mul, Sub},
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

// TODO save an alloc+indirection: move small values out so RcVal is Small | Rc<BigVal>
type RcVal = Rc<Val>;

// TODO Ref, Box to say functions shouldn't pervade
#[derive(Debug, Clone)]
pub enum Val {
    Char(u8),

    Int(i64),  // TODO TwoInts, ThreeInts

    Float(f64),

    // TODO switch this out for the non-token type (some primitives won't have
    // token representations)
    PrimFunc(PrimVerb),

    AdverbDerivedFunc {
        adverb: PrimAdverb,
        operand: RcVal,
    },

    // TODO decide if closures should refer to a shared environment or be value
    // types (copying copies environment. Currently, we hold a reference to the env.
    //   a:1; F:{a:a+1}; G:F
    //   []F  \ 2
    //   []G  \ Currently this returns 3. Should G have its own copy of the env, making this 2?
    ExplicitFunc {
        // The function's first instruction (ALWAYS points after MakeFunc, so
        // you can get the function's instruction count with
        // code[func.code_index-1]
        code_index: usize,

        // Always points to Val::Vals.
        closure_env: RcVal,
    },

    U8s(Vec<u8>),
    I64s(Vec<i64>),
    F64s(Vec<f64>),
    Vals(Vec<RcVal>),
}

macro_rules! atom {
    () => {
        Val::Char(_) | Val::Int(_) | Val::Float(_) | Val::PrimFunc(_) | Val::ExplicitFunc{..} | Val::AdverbDerivedFunc{..}
    }
}

// enum Vals<'a> {
//     Chars(Chars),
//     Ints(Ints),
//     Floats(Floats),
//     Vals(Vals),

//     CharsInts(CharsInts),
//     CharsFloats(CharsFoats),
//     CharsVals(CharsVals),

//     IntsFloats(IntsFloats),
//     IntsVals(IntsVals),
// }

// struct Chars<'a> {
//     x: &'a [u8],
//     i: usize,
// }

// trait Dispatch {
//     type Output;
//     fn on_atoms(x: Atom, y: Atom) -> Self::Output;
//     fn // TODO godbolt basically zippevalls, do we have to keep matching for every element?
// }

// enum ClassifiedVal<'a> {
//     Atom(Atom<'a>),
//     List(List<'a>),
// }

// impl<'a> ClassifiedVal<'a> {
//     fn data_atom(data: DataAtom) -> Self {
//         ClassifiedVal::Atom(Atom::Data(data))
//     }
//     fn func_atom(data: FuncAtom<'a>) -> Self {
//         ClassifiedVal::Atom(Atom::Func(data))
//     }
// }

// enum List<'a> {
//     U8s(&'a [u8]),
//     I64s(&'a [i64]),
//     F64s(&'a [f64]),
//     Vals(&'a [RcVal]),
// }

// #[derive(Clone, Copy)]
// enum Atom<'a> {
//     Data(DataAtom),
//     Func(FuncAtom<'a>),
// }

// #[derive(Clone, Copy)]
// enum DataAtom {
//     Char(u8),
//     Int(i64),
//     Float(f64),
// }

// #[derive(Clone, Copy)]
// enum FuncAtom<'a> {
//     PrimFunc(PrimVerb),

//     AdverbDerivedFunc {
//         adverb: PrimAdverb,
//         operand: &'a RcVal,
//     },

//     ExplicitFunc {
//         code_index: usize,
//         closure_env: &'a RcVal,
//     },
// }

impl Val {
    fn as_val(&self) -> &Self { &self }

    fn len(&self) -> Option<usize> {
        use Val::*;
        match self {
            atom!() => None,
            U8s(vec) => Some(vec.len()),
            I64s(vec) => Some(vec.len()),
            F64s(vec) => Some(vec.len()),
            Vals(vec) => Some(vec.len()),
        }
    }

    // fn classify<'a>(&'a self) -> ClassifiedVal<'a> {
    //     use Val::*;
    //     match self {
    //         &Char(x) => ClassifiedVal::data_atom(DataAtom::Char(x)),
    //         &Int(x) => ClassifiedVal::data_atom(DataAtom::Int(x)),
    //         &Float(x) => ClassifiedVal::data_atom(DataAtom::Float(x)),

    //         &PrimFunc(prim) => ClassifiedVal::func_atom(FuncAtom::PrimFunc(prim)),
    //         &AdverbDerivedFunc { adverb, ref operand } =>
    //             ClassifiedVal::func_atom(FuncAtom::AdverbDerivedFunc { adverb, operand }),
    //         &ExplicitFunc { code_index, ref closure_env } =>
    //             ClassifiedVal::func_atom(FuncAtom::ExplicitFunc { code_index, closure_env }),

    //         U8s(vec) => ClassifiedVal::List(List::U8s(vec)),
    //         I64s(vec) => ClassifiedVal::List(List::I64s(vec)),
    //         F64s(vec) => ClassifiedVal::List(List::F64s(vec)),
    //         Vals(vec) => ClassifiedVal::List(List::Vals(vec)),
    //     }
    // }
}

// Val instances for val sorting.

impl PartialEq for Val {
    fn eq(&self, other: &Val) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl PartialEq<u8> for Val {
    fn eq(&self, other: &u8) -> bool {
        matches!(self, Val::Char(c) if c == other)
    }
}

impl PartialEq<i64> for Val {
    fn eq(&self, other: &i64) -> bool {
        match self {
            Val::Int(i) => i == other,
            Val::Float(f) => float_as_int(*f) == Some(*other),
            _ => false,
        }
    }
}

impl PartialEq<f64> for Val {
    fn eq(&self, other: &f64) -> bool {
        match self {
            Val::Float(f) => f == other,
            Val::Int(i) => *i as f64 == *other,
            _ => false,
        }
    }
}

impl PartialOrd for Val {
    fn partial_cmp(&self, other: &Val) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Val {}

impl Ord for Val {
    fn cmp(&self, other: &Val) -> Ordering {
        use Val::*;

        fn key_variant(x: &Val) -> u32 {
            match x {
                Char(_) => 0,
                Int(_) => 1,
                Float(_) => 2,
                PrimFunc(_) => 3,
                AdverbDerivedFunc {..} => 4,
                ExplicitFunc {..} => 5,
                U8s(_) => 6,
                I64s(_) => 7,
                F64s(_) => 8,
                Vals(_) => 9,
            }
        }

        // TODO better int-float comparison? This is "inaccurate" if `i` can't
        // be represented in floating point.
        fn int_float_cmp(i: &i64, f: &f64) -> Ordering {
            let i_float = *i as f64;
            i_float.total_cmp(f)
        }

        fn ints_floats_cmp(is: &[i64], fs: &[f64]) -> Ordering {
            let len = is.len().min(fs.len());
            for i in 0..len {
                let cmp = int_float_cmp(&is[i], &fs[i]);
                if cmp != Ordering::Equal { return cmp }
            }
            is.len().cmp(&fs.len())
        }

        match (self, other) {
            (Char(x), Char(y)) => x.cmp(y),
            (Int(x), Int(y)) => x.cmp(y),
            (Float(x), Float(y)) => x.total_cmp(y),

            (Int(i), Float(f)) => int_float_cmp(i, f),
            (Float(f), Int(i)) => int_float_cmp(i, f).reverse(),

            (PrimFunc(_), PrimFunc(_)) => todo!("Sort primitives"),
            // TODO sort by closure envs instead? Would be cool, but they may
            // not have envs, and ideally there aren't semantic differences
            // between explicit and primitive functions
            (ExplicitFunc{code_index: x, ..}, ExplicitFunc{code_index: y, ..}) => x.cmp(y),
            (AdverbDerivedFunc{operand: x, ..}, AdverbDerivedFunc{operand: y, ..}) => x.cmp(y),
            (U8s(x), U8s(y)) => x.cmp(y),
            (I64s(x), I64s(y)) => x.cmp(y),

            (I64s(is), F64s(fs)) => ints_floats_cmp(is, fs),
            (F64s(fs), I64s(is)) => ints_floats_cmp(is, fs).reverse(),

            (Vals(x), Vals(y)) => x.cmp(y),
            _ => key_variant(self).cmp(&key_variant(other)),
        }
    }
}

#[derive(Debug)]
struct StackFrame {
    // The first instruction of this function.
    code_index: usize,

    // Always points to Val::Vals.
    closure_env: RcVal,

    // Index into locals_stack. Local slots are offsets from this index.
    locals_start: usize,
}

pub struct Mem {
    pub code: Vec<Instr>,

    pub stack: Vec<RcVal>,
    
    // Stack of local scopes
    locals_stack: Vec<RcVal>,

    // Details about the explicit function (or global scope) we're currently in.
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
                closure_env: Rc::new(Val::Vals(vec![])),
                locals_start: 0,
                code_index: usize::MAX,
            }],

            zero: Rc::new(Val::Int(0)),
        }
    }

    pub fn execute(&mut self, mut ip: usize) -> Result<(), String> {
        use Instr::*;
        while ip < self.code.len() {
            ip += 1;
            match self.code[ip - 1] {
                Nop => {}
                Dup => self.push(self.stack.last().unwrap().clone()),
                MakeClosure { num_closure_vars } => {
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
                    self.push(RcVal::new(Val::ExplicitFunc { closure_env, code_index }));
                }
                MakeFunc { num_instructions } => {
                    self.push(RcVal::new(
                        Val::ExplicitFunc {
                            closure_env: RcVal::new(Val::Vals(vec![])),
                            code_index: ip + 1,
                        }
                    ));
                    ip += num_instructions - 1;
                }
                AllocLocals { num_locals } => {
                    // TODO is 0 the right thing to fill with here?
                    self.locals_stack.resize(self.locals_stack.len() + num_locals, self.zero.clone());
                }
                Return => {
                    let frame = self.stack_frames.pop().unwrap();
                    self.locals_stack.truncate(frame.locals_start);
                    return Ok(())
                }
                PushLiteralInteger(value) => self.push(RcVal::new(Val::Int(value))),
                PushLiteralFloat(value) => self.push(RcVal::new(Val::Float(value))),
                PushVar { src } => {
                    let val = self.load(src).clone();
                    self.push(val);
                }
                PushPrimVerb { prim: PrimVerb::Rec } => {
                    let frame = self.current_frame();
                    // TODO can we copy the actual function that was called instead?
                    self.push(RcVal::new(Val::ExplicitFunc {
                        code_index: frame.code_index,
                        closure_env: frame.closure_env.clone(),
                    }));
                }
                PushPrimVerb { prim } => self.push(RcVal::new(Val::PrimFunc(prim))),

                // TODO eliminate tail calls for call instructions.
                Call1 => {
                    let func = self.pop();
                    let x = self.pop();
                    let result = self.call_val(func, x, None)?;
                    self.push(result);
                }
                Call2 => {
                    let y = self.pop();
                    let func = self.pop();
                    let x = self.pop();
                    let result = self.call_val(func, x, Some(y))?;
                    self.push(result);
                }
                CallPrimVerb1 { prim } => {
                    let x = self.pop();
                    let result = self.call_prim_monad(prim, x)?;
                    self.push(result);
                }
                CallPrimVerb2 { prim } => {
                    let y = self.pop();
                    let x = self.pop();
                    let result = self.call_prim_dyad(prim, x, y)?;
                    self.push(result)
                }
                Pop => { self.pop(); }
                StoreTo { dst } => self.store(dst, self.stack.last().unwrap().clone()),
                Splat { count } => {
                    // TODO repetition
                    match self.pop().as_val() {
                        a@atom!() => return Err(format!("Array unpacking failed; expected {count} elements, got atom {:?}", a)),
                        Val::U8s(cs) => {
                            if cs.len() != count {
                                return Err(format!("Array unpacking failed; expected {count} elements, got {}", cs.len()))
                            }
                            self.stack.extend(cs.iter().rev().map(|c| RcVal::new(Val::Char(*c))))
                        }
                        Val::I64s(is) => {
                            if is.len() != count {
                                return Err(format!("Array unpacking failed; expected {count} elements, got {}", is.len()))
                            }
                            self.stack.extend(is.iter().rev().map(|i| RcVal::new(Val::Int(*i))))
                        }
                        Val::F64s(fs) =>  {
                            if fs.len() != count {
                                return Err(format!("Array unpacking failed; expected {count} elements, got {}", fs.len()))
                            }
                            self.stack.extend(fs.iter().rev().map(|f| RcVal::new(Val::Float(*f))))
                        }
                        Val::Vals(vs) => {
                            if vs.len() != count {
                                return Err(format!("Array unpacking failed; expected {count} elements, got {}", vs.len()))
                            }
                            self.stack.extend(vs.iter().rev().map(|v| v.clone()))
                        }
                    }
                }
                CallPrimAdverb { prim: adverb } => {
                    let operand = self.pop();
                    self.push(RcVal::new(Val::AdverbDerivedFunc { adverb, operand }));
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
                    let mut all_ints = true;
                    let mut all_chars = true;
                    for elem in &self.stack[(self.stack.len() - num_elems)..] {
                        all_ints &= matches!(&**elem, Val::Int(_));
                        all_chars &= matches!(&**elem, Val::Char(_));
                    }

                    let elems = self.stack.drain((self.stack.len() - num_elems)..);
                    let list_val = if all_ints {
                        Val::I64s(map(elems, |elem| irrefutable!(*elem, Val::Int(int) => int)))
                    } else if all_chars {
                        Val::U8s(map(elems, |elem| irrefutable!(*elem, Val::Char(ch) => ch)))
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

    fn load(&mut self, var: Var) -> &RcVal {
        // TODO do all local/closure vars point to non-lists (instead to slices)?
        let frame = self.current_frame();
        match var.place {
            Place::Local => &self.locals_stack[frame.locals_start + var.slot],
            Place::ClosureEnv => match &*frame.closure_env {
                Val::Vals(vs) => &vs[var.slot],
                _ => unreachable!(),
            }
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
            Place::ClosureEnv => match &*frame.closure_env {
                // SAFETY this seems bad, but we want closures to share an
                // environment. We'll restructure how closures work in a way
                // that Rust likes (hold them in a member vec & access by
                // index), but this "works" for now.
                Val::Vals(vs) => unsafe { (vs.as_ptr() as *mut RcVal).add(dst.slot).replace(val); }
                // TODO if a closure env consists of one value, it's fine not to
                // use a whole list.
                // TODO consolidate closure envs of all-matching types to
                // e.g. Int64s instead of Vals.
                _ => unreachable!(),  
            }
        }
    }

    fn call_val(&mut self, val: RcVal, x: RcVal, y: Option<RcVal>) -> Result<RcVal, String> {
        let result = match val.as_val() {
            &Val::ExplicitFunc { ref closure_env, code_index } => {
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
            Val::AdverbDerivedFunc { adverb, operand } =>
                self.call_prim_adverb(*adverb, operand.clone(), x, y)?,
            &Val::PrimFunc(prim) => if let Some(y) = y {
                self.call_prim_dyad(prim, x, y)?
            } else {
                self.call_prim_monad(prim, x)?
            },
            Val::Char(_) | Val::Int(_) | Val::Float(_) |
            Val::U8s(_) | Val::I64s(_) | Val::F64s(_) | Val::Vals(_) => val,
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
            Backtick => match maybe_y {
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
            Print => { self.prim_print(&x)?; println!(); x }
            DebugPrint => { self.debug_print_val(&x)?; println!(); x }
            ReadFile => RcVal::new(prim_read_file(x.as_val())?),
            Hash => RcVal::new(Val::Int(x.len().unwrap_or(1) as i64)),
            Slash => RcVal::new(iota(&*x)),
            Pipe => prim_reverse(&x),
            Comma => prim_ravel(&x),
            Caret => RcVal::new(Val::Vals(prim_prefixes(&x))),
            Dollar => RcVal::new(Val::Vals(prim_suffixes(&x))),
            Question => RcVal::new(prim_where(x.as_val())?),
            LessThan => RcVal::new(prim_sort(&x, false)),
            GreaterThan => RcVal::new(prim_sort(&x, true)),
            LessThanColon => RcVal::new(prim_grade(&x, false)),
            GreaterThanColon => RcVal::new(prim_grade(&x, true)),
            Exit => prim_exit(&x)?,
            _ => todo!("{x:?} {v:?}")
        };
        Ok(result)
    }

    fn call_prim_dyad(&mut self, v: PrimVerb, x: RcVal, y: RcVal) -> Result<RcVal, String> {
        use PrimVerb::*;
        let result = match v {
            Plus => prim_add(x.as_val(), y.as_val()),
            Minus => prim_subtract(x.as_val(), y.as_val()),
            Asterisk => prim_multiply(x.as_val(), y.as_val()),
            Slash => prim_divide(x.as_val(), y.as_val()),
            DoubleSlash => prim_divmod(DivModOp::Div, x.as_val(), y.as_val()),
            Percent => prim_divmod(DivModOp::Mod, x.as_val(), y.as_val()),
            Caret => prim_pow(x.as_val(), y.as_val()),
            Hash => prim_take(x, y.as_val()),
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
    fn prim_print(&self, x: &Val) -> Result<(), String> {
        match x {
            Val::Char(c) => print!("{}", char::from_u32(*c as u32).unwrap()),
            Val::Int(i) => print!("{i}"),
            Val::Float(f) => print!("{f}"),
            Val::PrimFunc(prim) => print!("{prim}"),
            Val::U8s(cs) => match std::str::from_utf8(cs) {  // TODO unicode
                Ok(s) => print!("{s}"),
                Err(err) => return Err(err.to_string()),
            },
            Val::I64s(is) => {
                if is.is_empty() {
                    print!("[]")
                } else {
                    print!("{}", is[0]);
                    for int in &is[1..] { print!(" {int}") }
                }
            }
            Val::F64s(fs) => {
                if fs.is_empty() {
                    print!("[]")
                } else {
                    print!("{}", fs[0]);
                    for float in &fs[1..] { print!(" {float}") }
                }
            }
            Val::Vals(vs) => {
                if vs.is_empty() {
                    print!("[]")
                } else {
                    let nested_list = vs.iter().any(|val| val.len().is_some());
                    print!("[");
                    self.debug_print_val(&vs[0])?;
                    for val in &vs[1..] {
                        if nested_list { print!("\n ") }
                        else { print!("; ") }
                        self.debug_print_val(val)?;
                    }
                    print!("]");
                }
            }
            Val::AdverbDerivedFunc { adverb, operand } => {
                print!("{adverb}");
                self.prim_print(operand.as_val())?;
            }
            Val::ExplicitFunc { .. } => {
                // map code index -> tokens?
                print!("(explicit func; TODO: implement printing)")
            }
        }
        Ok(())
    }

    // TODO output formatting
    fn debug_print_val(&self, x: &Val) -> Result<(), String> {
        match x {
            Val::Char(c) => print!("{:?}", char::from_u32(*c as u32).unwrap()),
            Val::U8s(cs) => match std::str::from_utf8(cs) {  // TODO unicode
                Ok(s) => print!("{s:?}"),
                Err(err) => return Err(err.to_string()),
            },
            _ => self.prim_print(x)?,
        }
        Ok(())
    }

    fn fold_val(&mut self, f: RcVal, x: RcVal, maybe_y: Option<RcVal>) -> Result<RcVal, String> {
        let (mut seed, start) = match maybe_y {
            Some(y) => (y, 0),
            None => match index_or_cycle_val(&x, 0) {
                Some(first) => (first, 1),
                None => return Err(format!("Error: fold with no input")),
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

fn prim_exit(x: &RcVal) -> Result<RcVal, String> {
    use std::process::exit;

    match x.as_val() {
        Val::Int(i) => exit(*i as i32),
        Val::Float(f) if *f == f.trunc() => exit(*f as i32),
        bad => return Err(format!("domain\nExpected integer exit code, got {bad:?}")),
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
        _ => return Err(format!("Expected integers, got {x:?}")),
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
            _ => return Err(format!("expected string filepath, got {x:?}")),
        }
    ).map_err(|err| err.to_string())?;
    match std::fs::read_to_string(path) {
        Ok(contents) => Ok(Val::U8s(contents.into_bytes())),
        Err(err) => Err(err.to_string()),
    }
}

fn float_as_int(f: f64) -> Option<i64> {
    let trunc = f.trunc();
    if trunc == f { Some(trunc as i64) } else { None }
}

fn prim_add(x: &Val, y: &Val) -> Result<RcVal, String> {
    use Val::*;
    fn add_char_int(c: u8, i: i64) -> u8 {
        (c as i64 + i) as u8
    }

    fn as_int_or_fail(f: f64) -> Result<i64, String> {
        float_as_int(f).ok_or_else(
            || format!("domain\nCan't add floats to chars (expected integer, got {f})")
        )
    }

    fn add_char_float(c: u8, f: f64) -> Result<u8, String> {
        Ok(add_char_int(c, as_int_or_fail(f)?))
    }

    let val = match (x, y) {
        (Int(i), Int(j)) => Int(i + j),

        (Int(i),  Char(c)) |
        (Char(c), Int(i)) => Char(add_char_int(*c, *i)),

        (Int(i),   Float(f)) |
        (Float(f), Int(i)) => Float(*i as f64 + f),

        (Float(f), Char(c)) |
        (Char(c),  Float(f)) => Char(add_char_float(*c, *f)?),

        (I64s(is), Char(c)) |
        (Char(c),  I64s(is)) => U8s(map(is, |i| add_char_int(*c, *i))),

        (I64s(is), Int(int)) |
        (Int(int), I64s(is)) => I64s(map(is, |i| *i + int)),

        (U8s(cs), Int(i)) |
        (Int(i),  U8s(cs)) => U8s(map(cs, |c| add_char_int(*c, *i))),

        (U8s(cs), Float(f)) |
        (Float(f), U8s(cs)) => U8s({
            let i = as_int_or_fail(*f)?;
            map(cs, |c| add_char_int(*c, i))
        }),

        (I64s(xs), I64s(ys)) => I64s(zip_map(xs, ys, |x, y| *x + *y)?),

        (I64s(is), Float(f)) |
        (Float(f), I64s(is)) => F64s(map(is, |i| *i as f64 + *f)),

        (I64s(is), F64s(fs)) | (F64s(fs), I64s(is)) => {
            let zipped = zip_exact(is, fs)?;
            if fs.iter().all(|f| f.trunc() == *f) {
                I64s(map(zipped, |(i, f)| i + *f as i64))
            } else {
                F64s(map(zipped, |(i, f)| *i as f64 + f))
            }
        }

        (F64s(fs), U8s(cs)) |
        (U8s(cs),  F64s(fs)) => U8s(
            zip_traverse(cs, fs, |c, f| add_char_float(*c, *f))?
        ),

        (F64s(fs), Int(i)) |
        (Int(i), F64s(fs)) => F64s(map(fs, |f| f + *i as f64)),

        (F64s(xs), F64s(ys)) => F64s(zip_map(xs, ys, |x, y| *x + *y)?),

        (U8s(cs), I64s(is)) |
        (I64s(is), U8s(cs)) => U8s(
            zip_map(cs, is, |c, i| add_char_int(*c, *i))?
        ),
        
        (Vals(vs), Int(i)) | (Int(i), Vals(vs)) => Vals(
            traverse(vs, |rc| prim_add(rc.as_val(), &Int(*i)))?
        ),

        (Vals(vs), I64s(is)) |
        (I64s(is), Vals(vs)) => Vals(
            zip_traverse(vs, is, |rc, int| prim_add(rc.as_val(), &Val::Int(*int)))?
        ),

        (Vals(vs), F64s(fs)) |
        (F64s(fs), Vals(vs)) => Vals(
            zip_traverse(vs, fs, |rc, float| prim_add(rc.as_val(), &Val::Float(*float)))?
        ),

        (Vals(vs), U8s(cs)) |
        (U8s(cs), Vals(vs)) => Vals({
            let zipped = zip_exact(vs, cs)?;
            traverse(zipped, |(rc, ch)| prim_add(rc.as_val(), &Val::Char(*ch)))?
        }),
        (Vals(xs), Vals(ys)) => Vals({
            let zipped = zip_exact(xs, ys)?;
            traverse(zipped, |(x, y)| prim_add(x.as_val(), y.as_val()))?
        }),
        _ => return Err(format!("domain\nCan't add {x:?} and {y:?}")),
    };
    Ok(RcVal::new(val))
}

fn prim_subtract(x: &Val, y: &Val) -> Result<RcVal, String> {
    use Val::*;
    fn sub_char_int(c: u8, i: i64) -> u8 {
        (c as i64 - i) as u8
    }

    fn as_int_or_fail(f: f64) -> Result<i64, String> {
        float_as_int(f).ok_or_else(
            || format!("domain\nCan't subtract floats from chars (expected integer, got {f})")
        )
    }

    fn sub_char_float(c: u8, f: f64) -> Result<u8, String> {
        Ok(sub_char_int(c, as_int_or_fail(f)?))
    }

    let result = match (x, y) {
        (Int(i), Int(j)) => Int(i - j),
        (Float(f), Float(g)) => Float(f - g),
        (Char(c), Int(i)) => Char(sub_char_int(*c, *i)),
        (Char(c), Float(f)) => Char(sub_char_float(*c, *f)?),
        (Char(x), Char(y)) => Int(*x as i64 - *y as i64),
        (I64s(is), Int(i)) => I64s(
            is.iter().map(|int| *int - *i).collect()
        ),
        (Int(i), I64s(is)) => I64s(
            is.iter().map(|int| *i - *int).collect()
        ),
        (I64s(is), Float(f)) => F64s(map(is, |int| *int as f64 - *f)),
        (Float(f), I64s(is)) => F64s(map(is, |int| *f - *int as f64)),
        (U8s(cs), Int(i)) => U8s(map(cs, |c| sub_char_int(*c, *i))),
        (U8s(cs), Float(f)) => U8s({
            let i = as_int_or_fail(*f)?;
            map(cs, |c| sub_char_int(*c, i))
        }),
        (I64s(xs), I64s(ys)) => I64s(zip_map(xs, ys, <&i64>::sub)?),
        (F64s(xs), F64s(ys)) => F64s(zip_map(xs, ys, <&f64>::sub)?),
        (U8s(cs), I64s(is)) => U8s({
            let zipped = zip_exact(cs, is)?;
            map(zipped, |(c, i)| sub_char_int(*c, *i))
        }),
        (U8s(cs), F64s(fs)) => U8s({
            let zipped = zip_exact(cs, fs)?;
            traverse(zipped, |(c, f)| sub_char_float(*c, *f))?
        }),
        (U8s(cs), Char(c)) => I64s({
            map(cs, |ch| *ch as i64 - *c as i64)
        }),
        (U8s(xs), U8s(ys)) => I64s(zip_map(xs, ys, |x, y| *x as i64 - *y as i64)?),
        (Vals(vs), Int(i)) => Vals(
            traverse(vs, |rc| prim_subtract(rc.as_val(), &Int(*i)))?
        ),
        (Int(i), Vals(vs)) => Vals(
            traverse(vs, |rc| prim_subtract(&Int(*i), rc.as_val()))?
        ),
        (Vals(vs), Float(f)) => Vals(
            traverse(vs, |rc| prim_subtract(rc.as_val(), &Float(*f)))?
        ),
        (Float(f), Vals(vs)) => Vals(
            traverse(vs, |rc| prim_subtract(&Float(*f), rc.as_val()))?
        ),
        (Vals(vs), I64s(is)) => Vals(
            zip_traverse(vs, is, |rc, i| prim_subtract(rc.as_val(), &Val::Int(*i)))?
        ),
        (I64s(is), Vals(vs)) => Vals(
            zip_traverse(is, vs, |i, rc| prim_subtract(&Val::Int(*i), rc.as_val()))?
        ),
        (Vals(vs), F64s(fs)) => Vals(
            zip_traverse(vs, fs, |rc, f| prim_subtract(rc.as_val(), &Val::Float(*f)))?
        ),
        (F64s(fs), Vals(vals)) => Vals(
            zip_traverse(fs, vals, |f, rc| prim_subtract(&Val::Float(*f), rc.as_val()))?
        ),
        (Vals(vals), U8s(cs)) => Vals(
            zip_traverse(vals, cs, |rc, c| prim_subtract(rc.as_val(), &Val::Char(*c)))?
        ),
        (U8s(cs), Vals(vals)) => Vals(
            zip_traverse(cs, vals, |c, rc| prim_subtract(&Val::Char(*c), rc.as_val()))?
        ),
        (Vals(xs), Vals(ys)) => Vals(
            zip_traverse(xs, ys, |x, y| prim_subtract(x.as_val(), y.as_val()))?
        ),
        _ => return Err(format!("domain\nCan't subtract {x:?} and {y:?}")),
    };
    Ok(RcVal::new(result))
}

fn prim_multiply(x: &Val, y: &Val) -> Result<RcVal, String> {
    use Val::*;
    let result = match (x, y) {
        (Int(i), Int(j)) => Int(i * j),
        (Float(f), Float(g)) => Float(f * g),

        (Int(i), Float(f)) |
        (Float(f), Int(i)) => Float(f * *i as f64),

        (I64s(is), Int(int)) |
        (Int(int), I64s(is)) => I64s(map(is, |i| *i * *int)),

        (I64s(is), Float(f)) |
        (Float(f), I64s(is)) => F64s(map(is, |i| *i as f64 * *f)),

        (F64s(fs), Float(float)) |
        (Float(float), F64s(fs)) => F64s(map(fs, |f| *f * *float)),

        (F64s(fs), Int(i)) |
        (Int(i), F64s(fs)) => F64s({
            let float = *i as f64;
            map(fs, |f| *f * float)
        }),

        (I64s(xs), I64s(ys)) => I64s(zip_map(xs, ys, <&i64>::mul)?),
        (F64s(xs), F64s(ys)) => F64s(zip_map(xs, ys, <&f64>::mul)?),

        (Vals(vs), I64s(is)) |
        (I64s(is), Vals(vs)) => Vals(
            zip_traverse(vs, is, |rc, i| prim_multiply(rc.as_val(), &Val::Int(*i)))?
        ),

        (Vals(vs), F64s(fs)) |
        (F64s(fs), Vals(vs)) => Vals(
            zip_traverse(vs, fs, |rc, f| prim_multiply(rc.as_val(), &Val::Float(*f)))?
        ),
        (Vals(xs), Vals(ys)) => Vals(
            zip_traverse(xs, ys, |x, y| prim_multiply(x.as_val(), y.as_val()))?
        ),
        (Vals(vs), val) | (val, Vals(vs)) => Vals(
            traverse(vs, |rc| prim_multiply(rc.as_val(), val))?
        ),
        _ => return Err(format!("domain\nCan't add {x:?} and {y:?}")),
    };
    Ok(RcVal::new(result))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DivModOp { Div, Mod }

// TODO division by 0
fn prim_divmod(op: DivModOp, x: &Val, y: &Val) -> Result<RcVal, String> {
    macro_rules! op {
        ($a:expr, $b:expr) => {
            match op { DivModOp::Div => $a / $b,
                       DivModOp::Mod => $a % $b } as i64
        };
    }

    use Val::*;
    let result = match (x, y) {
        (Int(i), Int(j)) => Int(op!(*i, *j)),
        (Float(f), Float(g)) => Int(op!(*f, *g)),
        (I64s(ints), Int(i)) => I64s(
            ints.iter().map(|int| *int / *i).collect()
        ),
        (Int(i), I64s(ints)) => I64s(
            ints.iter().map(|int| *i / *int).collect()
        ),
        (I64s(ints), Float(f)) => {
            let i = *f as i64;
            I64s(ints.iter().map(|int| *int / i).collect())
        }
        (Int(i), F64s(fs)) => I64s(fs.iter().map(|f| *i / *f as i64).collect()),
        (F64s(fs), Int(i)) => I64s(fs.iter().map(|f| *f as i64 / *i).collect()),
        (F64s(xs), F64s(ys)) => I64s(zip_map(xs, ys, |x, y| (x / y) as i64)?),
        (F64s(fs), Float(float)) => I64s(map(fs, |f| (f / float) as i64)),
        (Float(float), F64s(fs)) => I64s(map(fs, |f| (float / f) as i64)),
        (I64s(xs), I64s(ys)) => I64s(zip_map(xs, ys, <&i64>::div)?),

        (Vals(vs), I64s(is)) => Vals(
            zip_traverse(vs, is, |v, i| prim_divmod(op, v.as_val(), &Val::Int(*i)))?
        ),

        (Vals(vs), F64s(fs)) => Vals(
            zip_traverse(vs, fs, |v, f| prim_divmod(op, v.as_val(), &Val::Float(*f)))?
        ),

        (F64s(fs), Vals(vs)) => Vals(
            zip_traverse(fs, vs, |f, v| prim_divmod(op, &Val::Float(*f), v.as_val()))?
        ),

        (I64s(is), Vals(vs)) => Vals(
            zip_traverse(is, vs, |i, v| prim_divmod(op, &Val::Int(*i), v.as_val()))?
        ),

        (Vals(xs), Vals(ys)) => Vals(
            zip_traverse(xs, ys, |x, y| prim_divmod(op, x.as_val(), y.as_val()))?
        ),
        (Vals(vals), v@atom!()) => collect_list(
            vals.iter().map(|rc| prim_divmod(op, rc.as_ref(), v))
        )?,
        (v@atom!(), Vals(vals)) => collect_list(
            vals.iter().map(|rc| prim_divmod(op, v, rc.as_ref()))
        )?,
        _ => {
            let op_str = match op { DivModOp::Div => &"divide",
                                    DivModOp::Mod => &"mod" };
            return Err(format!("domain\nCan't {op_str} {x:?} by {y:?}"))
        }
    };
    Ok(RcVal::new(result))
}

fn prim_divide(x: &Val, y: &Val) -> Result<RcVal, String> {
    use Val::*;
    let result = match (x, y) {
        (Int(i), Int(j)) => Float(*i as f64 / *j as f64),
        (Float(f), Float(g)) => Float(f / g),
        (Float(f), Int(i)) => Float(f / *i as f64),
        (Int(i), Float(f)) => Float(*i as f64 / f),
        (I64s(ints), Int(i)) => F64s({
            let f = *i as f64;
            map(ints, |int| *int as f64 / f)
        }),
        (Int(i), I64s(ints)) => F64s({
            let f = *i as f64;
            map(ints, |int| f / *int as f64)
        }),
        (Float(float), F64s(fs)) => F64s(map(fs, |f| f / *float)),
        (F64s(fs), Float(float)) => F64s(map(fs, |f| *f / float)),
        (F64s(fs), Int(i)) => F64s({
            let g = *i as f64;
            map(fs, |f| *f / g)
        }),
        (Int(i), F64s(fs)) => F64s({
            let f = *i as f64;
            map(fs, |g| f / *g)
        }),
        (F64s(xs), F64s(ys)) => F64s(zip_map(xs, ys, <&f64>::div)?),

        (I64s(xs), I64s(ys)) => F64s(
            zip_map(xs, ys, |x, y| *x as f64 / *y as f64)?
        ),

        (Vals(vs), I64s(is)) => collect_list(
            zip_exact(vs, is)?.map(|(v, i)| prim_divide(v.as_val(), &Val::Int(*i)))
        )?,

        (I64s(is), Vals(vs)) => collect_list(
            zip_exact(is, vs)?.map(|(i, v)| prim_divide(&Val::Int(*i), v.as_val()))
        )?,

        (Vals(xs), Vals(ys)) => collect_list(
            zip_exact(xs, ys)?.map(|(x, y)| prim_divide(x.as_val(), y.as_val()))
        )?,

        (Vals(vs), val@atom!()) => collect_list(vs.iter().map(|v| prim_divide(v.as_val(), val)))?,

        (val@atom!(), Vals(vs)) => collect_list(vs.iter().map(|v| prim_divide(val, v.as_val())))?,

        _ => return Err(format!("domain\nCan't divide {x:?} by {y:?}")),
    };
    Ok(RcVal::new(result))
}

fn prim_pow(x: &Val, y: &Val) -> Result<RcVal, String> {
    use Val::*;
    let result = match (x, y) {
        (Int(i), Int(j)) => if *j < 0 {
            Float((*i as f64).powi(*j as i32))
        } else { 
            Int(i.pow(*j as u32))
        },
        (Int(i), Float(f)) => Float((*i as f64).powf(*f)),
        (Float(f), Int(i)) => Float(f.powi(*i as i32)),
        (Float(f), Float(g)) => Float(f.powf(*g)),
        (I64s(ints), Int(pow)) => if *pow < 0 {
            F64s(ints.iter().map(|int| (*int as f64).powi(*pow as i32)).collect())
        } else {
            I64s(ints.iter().map(|int| int.pow(*pow as u32)).collect())
        },
        (Int(int), I64s(pows)) => {
            let mut vec: Vec<i64> = Vec::with_capacity(pows.len());
            for pow in pows {
                if *pow < 0 { break }
                vec.push(int.pow(*pow as u32));
            }

            if vec.len() == pows.len() {
                I64s(vec)
            } else {
                // TODO just do Vals here instead of switching everything to floats?
                let base = *int as f64;
                let mut fs: Vec<f64> = vec.drain(..).map(|i| i as f64).collect();
                for pow in &pows[fs.len()..] {
                    fs.push(base.powi(*pow as i32));
                }
                F64s(fs)
            }
        }
        (I64s(ints), Float(pow)) => F64s(
            ints.iter().map(|int| (*int as f64).powf(*pow)).collect()
        ),
        (I64s(is), F64s(fs)) => F64s(zip_map(is, fs, |i, f| (*i as f64).powf(*f))?),
        (F64s(fs), I64s(is)) => F64s(zip_map(fs, is, |f, i| f.powi(*i as i32))?),
        (F64s(xs), F64s(ys)) => F64s(zip_map(xs, ys, |x, y| x.powf(*y))?),
        (I64s(xs), I64s(ys)) => I64s(zip_map(xs, ys, |x, y| x.pow(*y as u32))?),
        (Vals(vs), I64s(is)) => Vals(
            zip_traverse(vs, is, |v, i| prim_pow(v.as_val(), &Val::Int(*i)))?
        ),
        (I64s(is), Vals(vs)) => Vals(
            zip_traverse(is, vs, |i, v| prim_pow(&Val::Int(*i), v.as_val()))?
        ),
        (Vals(vs), F64s(fs)) => Vals(
            zip_traverse(vs, fs, |v, f| prim_pow(v.as_val(), &Val::Float(*f)))?
        ),
        (F64s(fs), Vals(vs)) => Vals(
            zip_traverse(fs, vs, |f, v| prim_pow(&Val::Float(*f), v.as_val()))?
        ),
        (Vals(xs), Vals(ys)) => Vals(
            zip_traverse(xs, ys, |x, y| prim_pow(x.as_val(), y.as_val()))?
        ),
        (Vals(vs), val@atom!()) => collect_list(
            vs.iter().map(|v| prim_pow(v.as_val(), val))
        )?,
        (val@atom!(), Vals(vs)) => collect_list(
            vs.iter().map(|v| prim_pow(val, v.as_val()))
        )?,
        _ => return Err(format!("domain\nCan't raise {x:?} to power {y:?}")),
    };
    Ok(RcVal::new(result))
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
        _ => return Err(format!("Invalid right argument {y:?}")),
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
        Val::Char(c) => {
            let mut cs = vec![];
            cs.resize(count.abs() as usize, *c);
            Val::U8s(cs)
        }
        Val::Int(int) => {
            let mut ints = vec![];
            ints.resize(count.abs() as usize, *int);
            Val::I64s(ints)
        }
        Val::Float(float) => {
            let mut fs = vec![];
            fs.resize(count.abs() as usize, *float);
            Val::F64s(fs)
        }
        _ => {
            let mut vals = vec![];
            vals.resize(count.abs() as usize, x.clone());
            Val::Vals(vals)
        }
    };
    Ok(RcVal::new(result))
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
    fn one_then_many<A: Copy>(x: &A, ys: &[A]) -> Vec<A> {
        once(*x).chain(ys.iter().copied()).collect()
    }
    fn many_then_one<A: Copy>(xs: &[A], y: &A) -> Vec<A> {
        xs.iter().copied().chain(once(*y)).collect()
    }
    fn many_then_many<A: Copy>(xs: &[A], ys: &[A]) -> Vec<A> {
        xs.iter().chain(ys.iter()).copied().collect()
    }
    let val = match (x.as_val(), y.as_val()) {
        (Val::Char(x), Val::Char(y)) => Val::U8s(vec![*x, *y]),
        (Val::Char(x), Val::U8s(y)) => Val::U8s(one_then_many(x, y)),
        (Val::U8s(x), Val::Char(y)) => Val::U8s(many_then_one(x, y)),
        (Val::U8s(x), Val::U8s(y)) => Val::U8s(many_then_many(x, y)),

        (Val::Int(x), Val::Int(y)) => Val::I64s(vec![*x, *y]),
        (Val::Int(x), Val::I64s(y)) => Val::I64s(one_then_many(x, y)),
        (Val::I64s(x), Val::Int(y)) => Val::I64s(many_then_one(x, y)),
        (Val::I64s(x), Val::I64s(y)) => Val::I64s(many_then_many(x, y)),

        // TODO consolidate floats and ints?
        (Val::Float(x), Val::Float(y)) => Val::F64s(vec![*x, *y]),
        (Val::Float(x), Val::F64s(y)) => Val::F64s(one_then_many(x, y)),
        (Val::F64s(x), Val::Float(y)) => Val::F64s(many_then_one(x, y)),
        (Val::F64s(x), Val::F64s(y)) => Val::F64s(many_then_many(x, y)),

        (Val::Vals(x), Val::Vals(y)) => Val::Vals(x.iter().chain(y.iter()).cloned().collect()),
        (Val::Vals(x), _) => Val::Vals(x.iter().cloned().chain(once(y)).collect()),
        (_, Val::Vals(y)) => Val::Vals(once(x).chain(y.iter().cloned()).collect()),
        (_, _) => Val::Vals(vec![x, y]),
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

fn prim_match(x: &Val, y: &Val) -> Result<RcVal, String> {
    fn match_bool(x: &Val, y: &Val) -> bool {
        use Val::*;
        match (x, y) {
            (Int(i), Int(j)) => *i == *j,
            (Char(c), Char(d)) => *c == *d,
            (PrimFunc(p), PrimFunc(q)) => *p == *q,
            (ExplicitFunc { closure_env: x_env, code_index: x_code },
             ExplicitFunc { closure_env: y_env, code_index: y_code }) =>
                *x_code == *y_code && match_bool(x_env.as_val(), y_env.as_val()),
            (AdverbDerivedFunc { adverb: x_adverb, operand: x_operand },
             AdverbDerivedFunc { adverb: y_adverb, operand: y_operand }) =>
                *x_adverb == *y_adverb && match_bool(x_operand.as_val(), y_operand.as_val()),
            (I64s(xs), I64s(ys)) => xs == ys,
            (U8s(xs), U8s(ys)) => xs == ys,
            (Vals(xs), Vals(ys)) =>
                xs.len() == ys.len() &&
                xs.iter().zip(ys).all(|(x, y)| match_bool(x.as_val(), y.as_val())),
            _ => false,
        }
    }

    Ok(RcVal::new(Val::Int(match_bool(x, y) as i64)))
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
        _ => Err(format!("Expected integer, got float {f}")),
    }
}

fn replicate_with_i64<A: Clone>(a: A, n: i64) -> Result<impl Iterator<Item=A>, String> {
    if n >= 0 {
        Ok(replicate(a, n as usize))
    } else {
        Err(format!("Expected non-negative integer, got {n}"))
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

fn zip_map<A, X, Y, F>(x: X, y: Y, mut f: F) -> Result<Vec<A>, String>
where X: IntoIterator,
      X::IntoIter: ExactSizeIterator,
      Y: IntoIterator,
      Y::IntoIter: ExactSizeIterator,
      F: FnMut(X::Item, Y::Item) -> A {
    let zipped = zip_exact(x, y)?;
    Ok(map(zipped, |(a, b)| f(a, b)))
}

fn zip_traverse<A, X, Y, F>(x: X, y: Y, mut f: F) -> Result<Vec<A>, String>
where X: IntoIterator,
      X::IntoIter: ExactSizeIterator,
      Y: IntoIterator,
      Y::IntoIter: ExactSizeIterator,
      F: FnMut(X::Item, Y::Item) -> Result<A, String> {
    let zipped = zip_exact(x, y)?;
    traverse(zipped, |(a, b)| f(a, b))
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

fn zip_exact<X: IntoIterator, Y: IntoIterator>(x: X, y: Y) ->
    Result<impl Iterator<Item=(X::Item, Y::Item)>, String>
where X::IntoIter: ExactSizeIterator,
      Y::IntoIter: ExactSizeIterator {
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();
    let xlen = x_iter.len();
    let ylen = y_iter.len();
    if xlen == ylen {
        Ok(x_iter.zip(y_iter))
    } else {
        Err(format!("length mismatch: {xlen} vs {ylen}"))
    }
}

fn match_length(xlen: usize, ylen: usize) -> Result<(), String> {
    if xlen == ylen { return Ok(()); }
    // TODO include name/position of verb
    Err(format!("length mismatch: {xlen} vs {ylen}"))
}

