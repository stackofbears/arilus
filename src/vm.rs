use std::{
    cmp::Ordering,
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

        fn ints_floats_cmp(ints: &[i64], floats: &[f64]) -> Ordering {
            let len = ints.len().min(floats.len());
            for i in 0..len {
                let cmp = int_float_cmp(&ints[i], &floats[i]);
                if cmp != Ordering::Equal { return cmp }
            }
            ints.len().cmp(&floats.len())
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

            (I64s(ints), F64s(floats)) => ints_floats_cmp(ints, floats),
            (F64s(floats), I64s(ints)) => ints_floats_cmp(ints, floats).reverse(),

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
                Halt { exit_status } => std::process::exit(exit_status),
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
                        Val::I64s(elems.map(|elem| irrefutable!(*elem, Val::Int(int) => int)).collect())
                    } else if all_chars {
                        Val::U8s(elems.map(|elem| irrefutable!(*elem, Val::Char(ch) => ch)).collect())
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
                Val::Vals(vals) => &vals[var.slot],
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
                Val::Vals(vals) => unsafe { (vals.as_ptr() as *mut RcVal).add(dst.slot).replace(val); }
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
            Print => { self.print_val(&x)?; println!(); x }
            DebugPrint => { self.debug_print_val(&x)?; println!(); x }
            ReadFile => RcVal::new(read_file(x.as_val())?),
            Hash => RcVal::new(Val::Int(x.len().unwrap_or(1) as i64)),
            Slash => RcVal::new(iota(&*x)),
            Pipe => reverse(&x),
            Comma => ravel(&x),
            Caret => RcVal::new(Val::Vals(prim_prefixes(&x))),
            Dollar => RcVal::new(Val::Vals(prim_suffixes(&x))),
            Question => RcVal::new(prim_where(x.as_val())?),
            LessThan => RcVal::new(sort(&x, false)),
            GreaterThan => RcVal::new(sort(&x, true)),
            LessThanColon => RcVal::new(grade(&x, false)),
            GreaterThanColon => RcVal::new(grade(&x, true)),
            _ => todo!("{x:?} {v:?}")
        };
        Ok(result)
    }

    fn call_prim_dyad(&mut self, v: PrimVerb, x: RcVal, y: RcVal) -> Result<RcVal, String> {
        use PrimVerb::*;
        match v {
            Plus => add_vals(x.as_val(), y.as_val()),  // TODO take x,y by ref above?
            Minus => subtract_vals(x.as_val(), y.as_val()),
            Asterisk => multiply_vals(x.as_val(), y.as_val()),
            Slash => divide_vals(x.as_val(), y.as_val()),
            DoubleSlash => integer_divmod_vals(DivModOp::Div, x.as_val(), y.as_val()),
            Percent => integer_divmod_vals(DivModOp::Mod, x.as_val(), y.as_val()),
            Caret => exponentiate_vals(x.as_val(), y.as_val()),
            Hash => take(x, y.as_val()),
            Comma => append(x, y),
            DoubleEquals => match_vals(x.as_val(), y.as_val()),
            Equals => compare(&x, &y, |ord| ord == Ordering::Equal),
            EqualBang => compare(&x, &y, |ord| ord != Ordering::Equal),
            GreaterThan => compare(&x, &y, |ord| ord > Ordering::Equal),
            GreaterThanEquals => compare(&x, &y, |ord| ord >= Ordering::Equal),
            LessThan => compare(&x, &y, |ord| ord < Ordering::Equal),
            LessThanEquals => compare(&x, &y, |ord| ord <= Ordering::Equal),
            LessThanColon => choose_atoms(&x, &y, Val::le),
            GreaterThanColon => choose_atoms(&x, &y, Val::ge),
            At => self.index_val(&x, &y),
            Question => Ok(RcVal::new(Val::Int(prim_find(x.as_val(), y.as_val())))),
            QuestionColon => Ok(RcVal::new(Val::I64s(prim_subsequence_starts(x.as_val(), y.as_val())))),
            Snoc => todo!(),
            _ => todo!("{x:?} {v:?} {y:?}"),
        }
    }
    
    // TODO output formatting (take indent as arg)
    fn print_val(&self, x: &Val) -> Result<(), String> {
        match x {
            Val::Char(c) => print!("{}", char::from_u32(*c as u32).unwrap()),
            Val::Int(i) => print!("{i}"),
            Val::Float(f) => print!("{f}"),
            Val::PrimFunc(prim) => print!("{prim}"),
            Val::U8s(chars) => match std::str::from_utf8(chars) {  // TODO unicode
                Ok(s) => print!("{s}"),
                Err(err) => return Err(err.to_string()),
            },
            Val::I64s(ints) => {
                if ints.is_empty() {
                    print!("[]")
                } else {
                    print!("{}", ints[0]);
                    for int in &ints[1..] { print!(" {int}") }
                }
            }
            Val::F64s(floats) => {
                if floats.is_empty() {
                    print!("[]")
                } else {
                    print!("{}", floats[0]);
                    for float in &floats[1..] { print!(" {float}") }
                }
            }
            Val::Vals(vals) => {
                if vals.is_empty() {
                    print!("[]")
                } else {
                    let nested_list = vals.iter().any(|val| val.len().is_some());
                    print!("[");
                    self.debug_print_val(&vals[0])?;
                    for val in &vals[1..] {
                        if nested_list { print!("\n ") }
                        else { print!("; ") }
                        self.debug_print_val(val)?;
                    }
                    print!("]");
                }
            }
            Val::AdverbDerivedFunc { adverb, operand } => {
                print!("{adverb}");
                self.print_val(operand.as_val())?;
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
            Val::U8s(chars) => match std::str::from_utf8(chars) {  // TODO unicode
                Ok(s) => print!("{s:?}"),
                Err(err) => return Err(err.to_string()),
            },
            _ => self.print_val(x)?,
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

    fn index_val(&mut self, x: &RcVal, y: &RcVal) -> Result<RcVal, String> {
        fn oob(len: usize, i: i64) -> String {
            format!("Error in `@': index out of bounds (index {i}, but length is {len})")
        }
        fn to_index(len: usize, i: i64) -> usize {
            if i >= 0 { i as usize } else { (len as i64 + i) as usize }  // TODO overflow?
        }
        fn index<A>(slice: &[A], i: i64) -> Result<&A, String> {
            slice.get(to_index(slice.len(), i)).ok_or_else(|| oob(slice.len(), i))
        }

        use Val::*;
        let val = match (x.as_val(), y.as_val()) {
            (Int(_) | Char(_), &Int(i)) =>
                return if to_index(1, i) == 0 { Ok(x.clone()) } else { Err(oob(1, i)) },
            (&Int(int), I64s(is)) => I64s(
                is.iter()
                    .map(|i| if to_index(1, *i) == 0 { Ok(int) } else { Err(oob(1, *i)) })
                    .collect::<Result<_, _>>()?
            ),
            (&Char(ch), I64s(is)) => U8s(
                is.iter()
                    .map(|i| if to_index(1, *i) == 0 { Ok(ch) } else { Err(oob(1, *i)) })
                    .collect::<Result<_, _>>()?
            ),
            (I64s(ints), &Int(i)) => Int(*index(ints, i)?),
            (I64s(ints), I64s(is)) => I64s(
                is.iter().map(|i| index(ints, *i).copied()).collect::<Result<_, _>>()?
            ),
            (U8s(chars), &Int(i)) => Char(*index(chars, i)?),
            (U8s(chars), I64s(is)) => U8s(
                is.iter().map(|i| index(chars, *i).copied()).collect::<Result<_, _>>()?
            ),
            (Vals(vals), &Int(i)) => return Ok(index(vals, i)?.clone()),
            (Vals(vals), I64s(is)) => collect_list(is.iter().map(|i| index(vals, *i).cloned()))?,
            (Int(_) | Char(_) | I64s(_) | U8s(_) | Vals(_), Vals(is)) => collect_list(
                is.iter().map(|i| self.index_val(x, i))
            )?,

            _ => return self.call_val(x.clone(), y.clone(), None),
        };
        Ok(RcVal::new(val))
    }
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
        I64s(ints) => RcVal::new(Val::Int(*ints.get(i)?)),
        F64s(floats) => RcVal::new(Val::Float(*floats.get(i)?)),
        U8s(chars) => RcVal::new(Val::Char(*chars.get(i)?)),
        Vals(vals) => vals.get(i)?.clone(),
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
        Err(format!("Length mismatch: {xlen} vs {ylen}"))
    }
}

fn match_length(xlen: usize, ylen: usize) -> Result<(), String> {
    if xlen == ylen { return Ok(()); }
    // TODO include name/position of verb
    Err(format!("Length mismatch: {xlen} vs {ylen}"))
}

// Primitives

fn prim_prefixes(x: &RcVal) -> Vec<RcVal> {
    fn get_prefixes<A: Clone, F: Fn(Vec<A>) -> Val>(xs: &Vec<A>, f: F) -> Vec<RcVal> {
        (1..=xs.len()).map(|i| RcVal::new(f(xs[..i].to_vec()))).collect()
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
        (0..xs.len()).map(|i| RcVal::new(f(xs[i..].to_vec()))).collect()
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

fn choose_atoms<F: Copy + Fn(&Val, &Val) -> bool>(x: &RcVal, y: &RcVal, f: F) -> Result<RcVal, String> {
    Ok(match zip_vals(x, y) {
        None => if f(x.as_val(), y.as_val()) { x.clone() } else { y.clone() },
        Some(iter) => RcVal::new(
            collect_list(iter?.map(|(x, y)| choose_atoms(&x, &y, f)))?
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

// Attempts to find the whole of y as an element of x.
// TODO flip argument order?
fn prim_where(x: &Val) -> Result<Val, String> {
    use Val::*;
    match x {
        Int(i) => Ok(Val::I64s(replicate(0, *i).collect())),
        Float(f) => Ok(Val::I64s(replicate_with_float(0, *f)?.collect())),
        I64s(xs) => {
            let mut vec = vec![];
            for (i, n) in xs.iter().enumerate() {
                vec.extend(replicate(i, *n))
            }
            Ok(Val::I64s(vec))
        }
        F64s(xs) => {
            let mut vec = vec![];
            for (i, f) in xs.iter().enumerate() {
                vec.extend(replicate_with_float(i, *f)?)
            }
            Ok(Val::I64s(vec))
        }
        Vals(xs) => Ok(Val::Vals(xs.iter().map(|val| prim_where(val).map(RcVal::new)).collect::<Result<_, _>>()?)),
        _ => Err(format!("Error in `?': Expected integers, got {x:?}")),
    }
}

fn prim_subsequence_starts(x: &Val, y: &Val) -> Vec<i64> {
    use Val::*;

    // TODO linear time impl
    fn subsequence_starts_by<A, B, F: Fn(&A, &B) -> bool>(
        text: &[A], pat: &[B], pred: F
    ) -> Vec<i64> {
        if pat.is_empty() { return replicate(1, text.len() as i64).collect() }

        let mut out = Vec::with_capacity(text.len());
        for i in 0..=(text.len() - pat.len()) {
            let matches = text[i..].iter().zip(pat).all(|(t, p)| pred(t, p));
            out.push(matches as i64);
        }
        out.extend(replicate(0, (pat.len() - 1) as i64));
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
        (U8s(xs), _) => replicate(0, xs.len() as i64).collect(),

        (I64s(xs), Int(y)) => xs.iter().map(|x| (x == y) as i64).collect(),
        (I64s(xs), I64s(ys)) => subsequence_starts(xs, ys),
        (I64s(xs), Vals(ys)) => subsequence_starts_by(xs, ys, |x, y| y.as_val() == x),
        (I64s(xs), _) => replicate(0, xs.len() as i64).collect(),

        (F64s(xs), Float(y)) => xs.iter().map(|x| (x == y) as i64).collect(),
        (F64s(xs), F64s(ys)) => subsequence_starts(xs, ys),
        (F64s(xs), Vals(ys)) => subsequence_starts_by(xs, ys, |x, y| y.as_val() == x),
        (F64s(xs), _) => replicate(0, xs.len() as i64).collect(),

        (Vals(xs), atom!()) => xs.iter().map(|x| (x.as_val() == y) as i64).collect(),

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

fn read_file(x: &Val) -> Result<Val, String> {
    let mut byte: [u8; 1] = [0; 1];
    let path = std::str::from_utf8(
        match x {
            Val::Char(c) => { byte[0] = *c; &byte }
            Val::U8s(chars) => chars,
            _ => return Err(format!("Error in `ReadFile': expected string filepath, got {x:?}")),
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

fn add_vals(x: &Val, y: &Val) -> Result<RcVal, String> {
    use Val::*;
    fn add_char_int(c: u8, i: i64) -> u8 {
        (c as i64 + i) as u8
    }

    fn as_int_or_fail(f: f64) -> Result<i64, String> {
        float_as_int(f).ok_or_else(|| format!("Error in `+': Expected integer, got float {f}"))
    }

    fn add_char_float(c: u8, f: f64) -> Result<u8, String> {
        Ok(add_char_int(c, as_int_or_fail(f)?))
    }

    let result = match (x, y) {
        (Int(i), Int(j)) => Int(i + j),
        (Int(i), Char(c)) | (Char(c), Int(i)) => Char(add_char_int(*c, *i)),
        (Int(i), Float(f)) | (Float(f), Int(i)) => Float(*i as f64 + f),
        (Float(f), Char(c)) | (Char(c), Float(f)) => Char(add_char_float(*c, *f)?),
        (I64s(ints), Char(c)) | (Char(c), I64s(ints)) => U8s(
            ints.iter().map(|int| add_char_int(*c, *int)).collect()
        ),
        (I64s(ints), Int(i)) | (Int(i), I64s(ints)) => I64s(
            ints.iter().map(|int| *int + *i).collect()
        ),
        (U8s(chars), Int(i)) | (Int(i), U8s(chars)) => U8s(
            chars.iter().map(|c| add_char_int(*c, *i)).collect()
        ),
        (U8s(chars), Float(f)) | (Float(f), U8s(chars)) => {
            let i = as_int_or_fail(*f)?;
            U8s(chars.iter().map(|c| add_char_int(*c, i)).collect())
        }
        (I64s(x_ints), I64s(y_ints)) => I64s(
            zip_exact(x_ints, y_ints)?.map(|(i, j)| *i + *j).collect()
        ),
        (I64s(ints), Float(f)) | (Float(f), I64s(ints)) => F64s(
            ints.iter().map(|i| *i as f64 + *f).collect()
        ),
        (I64s(ints), F64s(floats)) | (F64s(floats), I64s(ints)) => {
            let zipped = zip_exact(ints, floats)?;
            if floats.iter().all(|f| f.trunc() == *f) {
                I64s(zipped.map(|(i, f)| i + *f as i64).collect())
            } else {
                F64s(zipped.map(|(i, f)| *i as f64 + f).collect())
            }
        }
        (F64s(floats), U8s(chars)) | (U8s(chars), F64s(floats)) => U8s(
            zip_exact(chars, floats)?
                .map(|(c, f)| add_char_float(*c, *f))
                .collect::<Result<_, _>>()?
        ),
        (F64s(floats), Int(i)) | (Int(i), F64s(floats)) => F64s(
            floats.iter().map(|f| f + *i as f64).collect()
        ),
        (F64s(x_floats), F64s(y_floats)) => F64s(
            zip_exact(x_floats, y_floats)?.map(|(f, g)| *f + *g).collect()
        ),
        (U8s(chars), I64s(ints)) |
        (I64s(ints), U8s(chars)) => U8s(
            zip_exact(chars, ints)?.map(|(c, i)| add_char_int(*c, *i)).collect()
        ),
        (Vals(vals), Int(i)) | (Int(i), Vals(vals)) => Vals(
            vals.iter().map(|rc_val| add_vals(rc_val.as_val(), &Int(*i))).collect::<Result<_, _>>()?
        ),
        (Vals(vals), I64s(ints)) |
        (I64s(ints), Vals(vals)) => Vals(
            zip_exact(vals, ints)?
                .map(|(rc_val, int)| add_vals(rc_val.as_val(), &Val::Int(*int)))
                .collect::<Result<_, _>>()?
        ),
        (Vals(vals), F64s(floats)) |
        (F64s(floats), Vals(vals)) => Vals(
            zip_exact(vals, floats)?
                .map(|(rc_val, float)| add_vals(rc_val.as_val(), &Val::Float(*float)))
                .collect::<Result<_, _>>()?
        ),
        (Vals(vals), U8s(chars)) |
        (U8s(chars), Vals(vals)) => Vals(
            zip_exact(vals, chars)?
                .map(|(rc_val, ch)| add_vals(rc_val.as_val(), &Val::Char(*ch)))
                .collect::<Result<_, _>>()?
        ),
        (Vals(x_vals), Vals(y_vals)) => Vals(
            zip_exact(x_vals, y_vals)?
                .map(|(x_rc_val, y_rc_val)| add_vals(x_rc_val.as_val(), y_rc_val.as_val()))
                .collect::<Result<_, _>>()?
        ),
        _ => return Err(format!("Error in `+': Can't add {x:?} and {y:?}")),
    };
    Ok(RcVal::new(result))
}

// fn add_vals2(x: &Val, y: &Val) -> Result<RcVal, String> {
//     use Val::*;
//     fn add_char_int(c: u8, i: i64) -> u8 {
//         (c as i64 + i) as u8
//     }

//     fn as_int_or_fail(f: f64) -> Result<i64, String> {
//         float_as_int(f).ok_or_else(|| format!("Error in `+': Expected integer, got float {f}"))
//     }

//     fn add_char_float(c: u8, f: f64) -> Result<u8, String> {
//         Ok(add_char_int(c, as_int_or_fail(f)?))
//     }

//     let result = match (x.classify(), y.classify()) {
//         (ClassifiedVal::Atom(Atom::Data(x)),
//          ClassifiedVal::Atom(Atom::Data(y))) => {
//             match (x, y) {
//                 (DataAtom::Char(x), DataAtom::Int(y)) |  =>
//             }
//         }
//         (ClassifiedVal::Atom(x), ClassifiedVal::List(ys)) => {}
//         (ClassifiedVal::List(xs), ClassifiedVal::Atom(y)) => {}
//         (ClassifiedVal::List(xs), ClassifiedVal::List(ys)) => {}
//     };

//     Ok(RcVal::new(result))
// }

fn subtract_vals(x: &Val, y: &Val) -> Result<RcVal, String> {
    use Val::*;
    fn sub_char_int(c: u8, i: i64) -> u8 {
        (c as i64 - i) as u8
    }

    fn as_int_or_fail(f: f64) -> Result<i64, String> {
        float_as_int(f).ok_or_else(|| format!("Error in `-': Expected integer, got float {f}"))
    }

    fn sub_char_float(c: u8, f: f64) -> Result<u8, String> {
        Ok(sub_char_int(c, as_int_or_fail(f)?))
    }

    let result = match (x, y) {
        (Int(i), Int(j)) => Int(i - j),
        (Float(f), Float(g)) => Float(f - g),
        (Char(c), Int(i)) => Char(sub_char_int(*c, *i)),
        (Char(c), Float(f)) => Char(sub_char_float(*c, *f)?),
        (Char(x_c), Char(y_c)) => Int(*x_c as i64 - *y_c as i64),
        (I64s(ints), Int(i)) => I64s(
            ints.iter().map(|int| *int - *i).collect()
        ),
        (Int(i), I64s(ints)) => I64s(
            ints.iter().map(|int| *i - *int).collect()
        ),
        (I64s(ints), Float(f)) => F64s(ints.iter().map(|int| *int as f64 - *f).collect()),
        (Float(f), I64s(ints)) => F64s(ints.iter().map(|int| *f - *int as f64).collect()),
        (U8s(chars), Int(i)) => U8s(
            chars.iter().map(|c| sub_char_int(*c, *i)).collect()
        ),
        (U8s(chars), Float(f)) => {
            let i = as_int_or_fail(*f)?;
            U8s(chars.iter().map(|c| sub_char_int(*c, i)).collect())
        }
        (I64s(x_ints), I64s(y_ints)) => I64s(
            zip_exact(x_ints, y_ints)?.map(|(i, j)| *i - *j).collect()
        ),
        (F64s(x_floats), F64s(y_floats)) => F64s(
            zip_exact(x_floats, y_floats)?.map(|(f, g)| *f - *g).collect()
        ),
        (U8s(chars), I64s(ints)) => U8s(
            zip_exact(chars, ints)?
                .map(|(c, i)| sub_char_int(*c, *i))
                .collect()
        ),
        (U8s(chars), F64s(floats)) => U8s(
            zip_exact(chars, floats)?
                .map(|(c, f)| sub_char_float(*c, *f))
                .collect::<Result<_, _>>()?
        ),
        (U8s(chars), Char(c)) => I64s(
            chars.iter().map(|ch| *ch as i64 - *c as i64).collect()
        ),
        (U8s(x_chars), U8s(y_chars)) => I64s(
            zip_exact(x_chars, y_chars)?.map(|(x, y)| *x as i64 - *y as i64).collect()
        ),
        (Vals(vals), Int(i)) => Vals(
            vals.iter()
                .map(|rc_val| subtract_vals(rc_val.as_val(), &Int(*i)))
                .collect::<Result<_, _>>()?
        ),
        (Int(i), Vals(vals)) => Vals(
            vals.iter()
                .map(|rc_val| subtract_vals(&Int(*i), rc_val.as_val()))
                .collect::<Result<_, _>>()?
        ),
        (Vals(vals), Float(f)) => Vals(
            vals.iter()
                .map(|rc_val| subtract_vals(rc_val.as_val(), &Float(*f)))
                .collect::<Result<_, _>>()?
        ),
        (Float(f), Vals(vals)) => Vals(
            vals.iter()
                .map(|rc_val| subtract_vals(&Float(*f), rc_val.as_val()))
                .collect::<Result<_, _>>()?
        ),
        (Vals(vals), I64s(ints)) => Vals(
            zip_exact(vals, ints)?
                .map(|(rc_val, int)| subtract_vals(rc_val.as_val(), &Val::Int(*int)))
                .collect::<Result<_, _>>()?
        ),
        (I64s(ints), Vals(vals)) => Vals(
            zip_exact(ints, vals)?
                .map(|(int, rc_val)| subtract_vals(&Val::Int(*int), rc_val.as_val()))
                .collect::<Result<_, _>>()?
        ),
        (Vals(vals), F64s(floats)) => Vals(
            zip_exact(vals, floats)?
                .map(|(rc_val, float)| subtract_vals(rc_val.as_val(), &Val::Float(*float)))
                .collect::<Result<_, _>>()?
        ),
        (F64s(floats), Vals(vals)) => Vals(
            zip_exact(floats, vals)?
                .map(|(float, rc_val)| subtract_vals(&Val::Float(*float), rc_val.as_val()))
                .collect::<Result<_, _>>()?
        ),
        (Vals(vals), U8s(chars)) => Vals(
            zip_exact(vals, chars)?
                .map(|(rc_val, ch)| subtract_vals(rc_val.as_val(), &Val::Char(*ch)))
                .collect::<Result<_, _>>()?
        ),
        (U8s(chars), Vals(vals)) => Vals(
            zip_exact(chars, vals)?
                .map(|(ch, rc_val)| subtract_vals(&Val::Char(*ch), rc_val.as_val()))
                .collect::<Result<_, _>>()?
        ),
        (Vals(x_vals), Vals(y_vals)) => Vals(
            zip_exact(x_vals, y_vals)?
                .map(|(x_rc_val, y_rc_val)| subtract_vals(x_rc_val.as_val(), y_rc_val.as_val()))
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
        (Float(f), Float(g)) => Float(f * g),
        (Int(i), Float(f)) | (Float(f), Int(i)) => Float(f * *i as f64),
        (I64s(ints), Int(i)) | (Int(i), I64s(ints)) => I64s(
            ints.iter().map(|int| *int * *i).collect()
        ),
        (I64s(ints), Float(f)) | (Float(f), I64s(ints)) => F64s(
            ints.iter().map(|int| *int as f64 * *f).collect()
        ),
        (F64s(floats), Float(f)) | (Float(f), F64s(floats)) => F64s(
            floats.iter().map(|float| *float * *f).collect()
        ),
        (F64s(floats), Int(i)) | (Int(i), F64s(floats)) => {
            let f = *i as f64;
            F64s(floats.iter().map(|float| *float * f).collect())
        }
        (I64s(x_ints), I64s(y_ints)) => I64s(
            zip_exact(x_ints, y_ints)?.map(|(i, j)| *i * *j).collect()
        ),
        (F64s(x_floats), F64s(y_floats)) => F64s(
            zip_exact(x_floats, y_floats)?.map(|(f, g)| *f * *g).collect()
        ),
        (Vals(vals), I64s(ints)) |
        (I64s(ints), Vals(vals)) => Vals(
            zip_exact(vals, ints)?
                .map(|(rc_val, int)| multiply_vals(rc_val.as_val(), &Val::Int(*int)))
                .collect::<Result<_, _>>()?
        ),
        (Vals(vals), F64s(floats)) |
        (F64s(floats), Vals(vals)) => Vals(
            zip_exact(vals, floats)?
                .map(|(rc_val, float)| multiply_vals(rc_val.as_val(), &Val::Float(*float)))
                .collect::<Result<_, _>>()?
        ),
        (Vals(x_vals), Vals(y_vals)) => Vals(
            zip_exact(x_vals, y_vals)?
                .map(|(x_rc_val, y_rc_val)| multiply_vals(x_rc_val.as_val(), y_rc_val.as_val()))
                .collect::<Result<_, _>>()?
        ),
        (Vals(vals), v) | (v, Vals(vals)) => Vals(
            vals.iter().map(|rc_val| multiply_vals(rc_val.as_val(), v)).collect::<Result<_, _>>()?
        ),
        _ => return Err(format!("Error in `+': Can't add {x:?} and {y:?}")),
    };
    Ok(RcVal::new(result))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DivModOp { Div, Mod }

// TODO division by 0
fn integer_divmod_vals(op_choice: DivModOp, x: &Val, y: &Val) -> Result<RcVal, String> {
    macro_rules! op {
        ($a:expr, $b:expr) => {
            match op_choice { DivModOp::Div => $a / $b,
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
        (Int(i), F64s(floats)) => I64s(floats.iter().map(|f| *i / *f as i64).collect()),
        (F64s(floats), Int(i)) => I64s(floats.iter().map(|f| *f as i64 / *i).collect()),
        (F64s(xs), F64s(ys)) => I64s(
            zip_exact(xs, ys)?.map(|(x, y)| (x / y) as i64).collect()
        ),
        (F64s(floats), Float(g)) => I64s(
            floats.iter().map(|float| (float / g) as i64).collect()
        ),
        (Float(f), F64s(floats)) => I64s(
            floats.iter().map(|g| (f / g) as i64).collect()
        ),
        (I64s(x_ints), I64s(y_ints)) => I64s(
            zip_exact(x_ints, y_ints)?.map(|(i, j)| *i / *j).collect()
        ),
        (Vals(vals), I64s(ints)) => Vals(
            zip_exact(vals, ints)?
                .map(|(rc_val, int)| integer_divmod_vals(op_choice, rc_val.as_val(), &Val::Int(*int)))
                .collect::<Result<_, _>>()?
        ),
        (Vals(vals), F64s(floats)) => Vals(
            zip_exact(vals, floats)?
                .map(|(rc_val, float)| integer_divmod_vals(op_choice, rc_val.as_val(), &Val::Float(*float)))
                .collect::<Result<_, _>>()?
        ),
        (F64s(floats), Vals(vals)) => Vals(
            zip_exact(vals, floats)?
                .map(|(rc_val, float)| integer_divmod_vals(op_choice, rc_val.as_val(), &Val::Float(*float)))
                .collect::<Result<_, _>>()?
        ),
        (I64s(ints), Vals(vals)) => Vals(
            zip_exact(ints, vals)?
                .map(|(int, rc_val)| integer_divmod_vals(op_choice, &Val::Int(*int), rc_val.as_val()))
                .collect::<Result<_, _>>()?
        ),
        (Vals(x_vals), Vals(y_vals)) => Vals(
            zip_exact(x_vals, y_vals)?
                .map(|(x_rc_val, y_rc_val)| integer_divmod_vals(op_choice, x_rc_val.as_val(), y_rc_val.as_val()))
                .collect::<Result<_, _>>()?
        ),
        (Vals(vals), v@atom!()) => collect_list(
            vals.iter().map(|rc_val| integer_divmod_vals(op_choice, rc_val.as_ref(), v))
        )?,
        (v@atom!(), Vals(vals)) => collect_list(
            vals.iter().map(|rc_val| integer_divmod_vals(op_choice, v, rc_val.as_ref()))
        )?,
        _ => {
            let op_str = match op_choice { DivModOp::Div => &"//",
                                           DivModOp::Mod => &"%" };
            return Err(format!("Error in `{op_str}': Can't divide {x:?} and {y:?}"))
        }
    };
    Ok(RcVal::new(result))
}

fn divide_vals(x: &Val, y: &Val) -> Result<RcVal, String> {
    use Val::*;
    let result = match (x, y) {
        (Int(i), Int(j)) => Float(*i as f64 / *j as f64),
        (Float(f), Float(g)) => Float(f / g),
        (Float(f), Int(i)) => Float(f / *i as f64),
        (Int(i), Float(f)) => Float(*i as f64 / f),
        (I64s(ints), Int(i)) => {
            let f = *i as f64;
            F64s(ints.iter().map(|int| *int as f64 / f).collect())
        }
        (Int(i), I64s(ints)) => {
            let f = *i as f64;
            F64s(ints.iter().map(|int| f / *int as f64).collect())
        }
        (Float(f), F64s(floats)) => F64s(floats.iter().map(|g| f / *g).collect()),
        (F64s(floats), Float(g)) => F64s(floats.iter().map(|f| *f / g).collect()),
        (F64s(floats), Int(i)) => {
            let g = *i as f64;
            F64s(floats.iter().map(|f| *f / g).collect())
        }
        (Int(i), F64s(floats)) => {
            let f = *i as f64;
            F64s(floats.iter().map(|g| f / *g).collect())
        }
        (F64s(xs), F64s(ys)) => F64s(
            zip_exact(xs, ys)?.map(|(x, y)| x / y).collect()
        ),
        (I64s(x_ints), I64s(y_ints)) => F64s(
            zip_exact(x_ints, y_ints)?
                .map(|(i, j)| *i as f64 / *j as f64)
                .collect()
        ),
        (Vals(vals), I64s(ints)) => collect_list(
            zip_exact(vals, ints)?
                .map(|(rc_val, int)| divide_vals(rc_val.as_val(), &Val::Int(*int)))
        )?,
        (I64s(ints), Vals(vals)) => collect_list(
            zip_exact(ints, vals)?
                .map(|(int, rc_val)| divide_vals(&Val::Int(*int), rc_val.as_val()))
        )?,
        (Vals(x_vals), Vals(y_vals)) => collect_list(
            zip_exact(x_vals, y_vals)?
                .map(|(x_rc_val, y_rc_val)| divide_vals(x_rc_val.as_val(), y_rc_val.as_val()))
        )?,
        (Vals(vals), v@atom!()) => collect_list(vals.iter().map(|rc_val| divide_vals(rc_val.as_val(), v)))?,
        (v@atom!(), Vals(vals)) => collect_list(vals.iter().map(|rc_val| divide_vals(v, rc_val.as_val())))?,
        _ => return Err(format!("Error in `/': Can't divide {x:?} and {y:?}")),
    };
    Ok(RcVal::new(result))
}

fn exponentiate_vals(x: &Val, y: &Val) -> Result<RcVal, String> {
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
                let mut floats: Vec<f64> = vec.drain(..).map(|i| i as f64).collect();
                for pow in &pows[floats.len()..] {
                    floats.push(base.powi(*pow as i32));
                }
                F64s(floats)
            }
        }
        (I64s(ints), Float(pow)) => F64s(
            ints.iter().map(|int| (*int as f64).powf(*pow)).collect()
        ),
        (I64s(ints), F64s(pows)) => F64s(
            zip_exact(ints, pows)?
                .map(|(base, pow)| (*base as f64).powf(*pow))
                .collect()
        ),
        (F64s(floats), I64s(pows)) => F64s(
            zip_exact(floats, pows)?
                .map(|(base, pow)| base.powi(*pow as i32))
                .collect()
        ),
        (F64s(floats), F64s(pows)) => F64s(
            zip_exact(floats, pows)?
                .map(|(base, pow)| base.powf(*pow))
                .collect()
        ),
        (I64s(x_ints), I64s(y_ints)) => I64s(
            zip_exact(x_ints, y_ints)?.map(|(i, j)| i.pow(*j as u32)).collect()
        ),
        (Vals(xs), I64s(ys)) => Vals(
            zip_exact(xs, ys)?
                .map(|(x, y)| exponentiate_vals(x.as_val(), &Val::Int(*y)))
                .collect::<Result<_, _>>()?
        ),
        (I64s(xs), Vals(ys)) => Vals(
            zip_exact(xs, ys)?
                .map(|(x, y)| exponentiate_vals(&Val::Int(*x), y.as_val()))
                .collect::<Result<_, _>>()?
        ),

        (Vals(xs), F64s(ys)) => Vals(
            zip_exact(xs, ys)?
                .map(|(x, y)| exponentiate_vals(x.as_val(), &Val::Float(*y)))
                .collect::<Result<_, _>>()?
        ),

        (F64s(xs), Vals(ys)) => Vals(
            zip_exact(xs, ys)?
                .map(|(x, y)| exponentiate_vals(&Val::Float(*x), y.as_val()))
                .collect::<Result<_, _>>()?
        ),

        (Vals(xs), Vals(ys)) => Vals(
            zip_exact(xs, ys)?
                .map(|(x, y)| exponentiate_vals(x.as_val(), y.as_val()))
                .collect::<Result<_, _>>()?
        ),

        (Vals(vals), v@atom!()) => collect_list(
            vals.iter().map(|rc_val| exponentiate_vals(rc_val.as_val(), v))
        )?,
        (v@atom!(), Vals(vals)) => collect_list(
            vals.iter().map(|rc_val| exponentiate_vals(v, rc_val.as_val()))
        )?,
        _ => return Err(format!("Error in `^': Can't raise {x:?} to power {y:?}")),
    };
    Ok(RcVal::new(result))
}

fn reverse(x: &RcVal) -> RcVal {
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
fn take(x: RcVal, y: &Val) -> Result<RcVal, String> {
    let count = match y {
        &Val::Int(i) => i,
        _ => return Err(format!("Error in `#': Invalid right argument {y:?}")),
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
        Val::U8s(chars) => Val::U8s(take_from_slice(count, chars)),
        Val::I64s(ints) => Val::I64s(take_from_slice(count, ints)),
        Val::F64s(floats) => Val::F64s(take_from_slice(count, floats)),
        Val::Vals(vals) => Val::Vals(take_from_slice(count, vals)),
        Val::Char(c) => {
            let mut chars = vec![];
            chars.resize(count.abs() as usize, *c);
            Val::U8s(chars)
        }
        Val::Int(int) => {
            let mut ints = vec![];
            ints.resize(count.abs() as usize, *int);
            Val::I64s(ints)
        }
        Val::Float(float) => {
            let mut floats = vec![];
            floats.resize(count.abs() as usize, *float);
            Val::F64s(floats)
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
//         (Val::U8s(chars), Val::Char(c)) => Val::I64s(
//             chars.iter().
//         ),
//     }
// }

fn grade(x: &RcVal, down: bool) -> Val {
    let mut indices: Vec<i64> = vec![];
    match x.as_val() {
        atom!() => indices.push(0),
        Val::U8s(chars) => {  // TODO unicode
            indices = (0..chars.len() as i64).collect();
            indices.sort_unstable_by(|i, j| cmp(down, &chars[*i as usize], &chars[*j as usize]));
        }
        Val::I64s(ints) => {
            indices = (0..ints.len() as i64).collect();
            indices.sort_unstable_by(|i, j| cmp(down, &ints[*i as usize], &ints[*j as usize]));
        }
        Val::F64s(floats) => {
            indices = (0..floats.len() as i64).collect();
            indices.sort_unstable_by(|i, j| cmp_floats(down, &floats[*i as usize], &floats[*j as usize]));
        }
        Val::Vals(vals) => {
            indices = (0..vals.len() as i64).collect();
            indices.sort_by(|i, j| cmp(down, &vals[*i as usize], &vals[*j as usize]));
        }
    };
    Val::I64s(indices)
}

fn sort(x: &RcVal, down: bool) -> Val {
    match x.as_val() {
        Val::Char(c) if true => Val::U8s(vec![*c]),
        Val::Int(i) if true => Val::I64s(vec![*i]),
        Val::Float(f) if true => Val::F64s(vec![*f]),
        atom!() => Val::Vals(vec![x.clone()]),
        Val::U8s(chars) => {
            let mut sorted = chars.clone();
            sorted.sort_unstable_by(|a, b| cmp(down, a, b));
            Val::U8s(sorted)
        }
        Val::I64s(ints) => {
            let mut sorted = ints.clone();
            sorted.sort_unstable_by(|a, b| cmp(down, a, b));
            Val::I64s(sorted)
        }
        Val::F64s(floats) => {
            let mut sorted = floats.clone();
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

fn append(x: RcVal, y: RcVal) -> Result<RcVal, String> {
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

fn ravel(x: &RcVal) -> RcVal {
    match x.as_val() {
        atom!() => RcVal::new(Val::Vals(vec![x.clone()])),
        Val::U8s(_) | Val::I64s(_) | Val::F64s(_) => x.clone(),
        Val::Vals(_) => RcVal::new(collect_list(
            ValIter { x: x.clone(), i: 0 }.flat_map(|val| ValIter { x: ravel(&val), i: 0 }).map(|val| -> Result<RcVal, ()> { Ok(val) })
        ).unwrap()),
    }
}

fn compare<F: Fn(Ordering) -> bool + Copy>(x: &RcVal, y: &RcVal, op: F) -> Result<RcVal, String> {
    use Val::*;
    let result = match zip_vals(&x, &y) {
        None => Int(op(x.as_val().cmp(y.as_val())) as i64),
        // TODO we already know this will consist of ints
        Some(iter) => collect_list(iter?.map(|(x, y)| compare(&x, &y, op)))?
    };
    Ok(RcVal::new(result))
}

fn iota(x: &Val) -> Val {
    use Val::*;
    match x {
        &Int(i) => Val::I64s(if i >= 0 { 0..i } else { i..0 }.collect()),
        _ => todo!("Implement / on non-ints"),
    }
}

fn match_vals(x: &Val, y: &Val) -> Result<RcVal, String> {
    Ok(RcVal::new(Val::Int(match_vals_bool(x, y) as i64)))
}

fn match_vals_bool(x: &Val, y: &Val) -> bool {
    use Val::*;
    match (x, y) {
        (Int(i), Int(j)) => *i == *j,
        (Char(c), Char(d)) => *c == *d,
        (PrimFunc(p), PrimFunc(q)) => *p == *q,
        (ExplicitFunc { closure_env: x_env, code_index: x_code },
         ExplicitFunc { closure_env: y_env, code_index: y_code }) =>
            *x_code == *y_code && match_vals_bool(x_env.as_val(), y_env.as_val()),
        (AdverbDerivedFunc { adverb: x_adverb, operand: x_operand },
         AdverbDerivedFunc { adverb: y_adverb, operand: y_operand }) =>
            *x_adverb == *y_adverb && match_vals_bool(x_operand.as_val(), y_operand.as_val()),
        (I64s(xs), I64s(ys)) => xs == ys,
        (U8s(xs), U8s(ys)) => xs == ys,
        (Vals(xs), Vals(ys)) =>
            xs.len() == ys.len() &&
            xs.iter().zip(ys.iter())
            .all(|(x, y)| match_vals_bool(x.as_val(), y.as_val())),
        _ => false,
    }
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
        Some(Ok(rc_val)) => match rc_val.as_val() {
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
                Val::Float(f) => match float_as_int(*f) {
                    Some(i) => ints.push(i),
                    None => {
                        let mut floats: Vec<f64> = ints.drain(..).map(|i| i as f64).collect();
                        floats.reserve(cap - floats.len());
                        floats.push(*f);
                        list = List::F64s(floats);
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
            List::F64s(floats) => match val.as_val() {
                Val::Float(f) => floats.push(*f),
                Val::Int(i) => floats.push(*i as f64),
                _ => {
                    let mut vals: Vec<RcVal> =
                        floats.drain(..).map(|f| RcVal::new(Val::Float(f))).collect();
                    vals.reserve(cap - vals.len());
                    vals.push(val);
                    list = List::Vals(vals);
                }
            },
            List::U8s(chars) => match val.as_val() {
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
        List::U8s(chars) => Val::U8s(chars),
        List::I64s(ints) => Val::I64s(ints),
        List::F64s(floats) => Val::F64s(floats),
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

fn replicate(i: usize, n: i64) -> impl Iterator<Item=i64> {
    std::iter::repeat(i as i64).take(n as usize)
}

fn replicate_with_float(i: usize, f: f64) -> Result<impl Iterator<Item=i64>, String> {
    match float_as_int(f) {
        Some(n) => Ok(replicate(i, n)),
        _ => Err(format!("Error in `?': Expected integer, got float {f}")),
    }
}

