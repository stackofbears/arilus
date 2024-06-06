use std::{
    cell::RefCell,
    cmp::Ordering,
    ops::{Div, Mul, Sub},
    rc::Rc,
    fmt::Write,
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

    // First element is the monadic case, second is the dyadic case
    AmbivalentFunc(RcVal, RcVal),

    // TODO this can probably just be a closure
    AtopFunc { f_func: RcVal, g_func: RcVal },
    BoundFunc { func: RcVal, y: RcVal },
    ForkFunc { f_func: RcVal, h_func: RcVal, g_func: RcVal },

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

        closure_env: Rc<RefCell<Vec<RcVal>>>,
    },

    U8s(Vec<u8>),
    I64s(Vec<i64>),
    F64s(Vec<f64>),
    Vals(Vec<RcVal>),
}

macro_rules! atom {
    () => {
        Val::Char(_) | Val::Int(_) | Val::Float(_) | Val::PrimFunc(_) | Val::AdverbDerivedFunc{..} | Val::AmbivalentFunc(_, _) | Val::AtopFunc{..} | Val::BoundFunc{..} | Val::ForkFunc{..} | Val::ExplicitFunc{..}
    }
}

enum ChasedTail {
    GoTo(usize),
    Push(RcVal),
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

    fn is_func(&self) -> bool {
        use Val::*;
        matches!(self,
                 PrimFunc(_) | AdverbDerivedFunc{..} | AmbivalentFunc(_, _) |
                 AtopFunc{..} | BoundFunc{..} | ForkFunc{..} | ExplicitFunc{..})
    }

    fn is_falsy(&self) -> bool {
        use Val::*;
        match self {
            Char(c) => *c == 0,
            Int(i) => *i == 0,
            Float(f) => *f == 0.0,
            U8s(cs) => cs.first() == Some(&0),
            I64s(is) => is.first() == Some(&0),
            F64s(fs) => fs.first() == Some(&0.0),
            Vals(vals) => vals.first().is_some_and(|val| val.is_falsy()),
            _ => false,
        }
    }

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
                AmbivalentFunc {..} => 5,
                AtopFunc{..} => 6,
                BoundFunc{..} => 7,
                ForkFunc{..} => 8,
                ExplicitFunc {..} => 9,
                U8s(_) => 10,
                I64s(_) => 11,
                F64s(_) => 12,
                Vals(_) => 13,
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

            (AmbivalentFunc(x_monad, x_dyad), AmbivalentFunc(y_monad, y_dyad)) =>
                x_monad.cmp(y_monad).then_with(|| x_dyad.cmp(y_dyad)),

            (AtopFunc { f_func: x_f_func, g_func: x_g_func },
             AtopFunc { f_func: y_f_func, g_func: y_g_func }) =>
                x_f_func.cmp(y_f_func).then_with(|| x_g_func.cmp(y_g_func)),

            (BoundFunc{func: x_func, y: x_y}, BoundFunc{func: y_func, y: y_y}) =>
                x_func.cmp(y_func).then_with(|| x_y.cmp(y_y)),

            (ForkFunc{f_func: x_f_func, h_func: x_h_func, g_func: x_g_func},
             ForkFunc{f_func: y_f_func, h_func: y_h_func, g_func: y_g_func}) =>
                x_f_func.cmp(y_f_func).then_with(|| x_h_func.cmp(y_h_func)).then_with(|| x_g_func.cmp(y_g_func)),

            (U8s(x), U8s(y)) => x.cmp(y),
            (I64s(x), I64s(y)) => x.cmp(y),

            (I64s(is), F64s(fs)) => ints_floats_cmp(is, fs),
            (F64s(fs), I64s(is)) => ints_floats_cmp(is, fs).reverse(),

            (Vals(x), Vals(y)) => x.cmp(y),
            _ => key_variant(self).cmp(&key_variant(other)),
        }
    }
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

                    self.push(RcVal::new(Val::ExplicitFunc {
                        code_index,
                        closure_env: Rc::new(RefCell::new(closure_data)),
                    }));
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
                        Val::AmbivalentFunc(monad, _) if matches!(adverb, PrimAdverb::P) => self.push(monad.clone()),
                        _ => self.push(RcVal::new(Val::AdverbDerivedFunc { adverb, operand })),
                    }
                }
                CollectVerbAlternatives => {
                    let mut dyad = self.pop();
                    if let Val::AmbivalentFunc(_, inner_dyad) = dyad.as_val() {
                        dyad = inner_dyad.clone();
                    }
                    let mut monad = self.pop();
                    if let Val::AmbivalentFunc(inner_monad, _) = monad.as_val() {
                        monad = inner_monad.clone();
                    }
                    self.push(RcVal::new(Val::AmbivalentFunc(monad, dyad)));
                }
                MakeAtopFunc => {
                    let g_func = self.pop();
                    let f_func = self.pop();
                    self.push(RcVal::new(Val::AtopFunc { f_func, g_func }));
                }
                MakeBoundFunc => {
                    let y = self.pop();
                    let func = self.pop();
                    self.push(RcVal::new(Val::BoundFunc { func, y }));
                }
                MakeForkFunc => {
                    let g_func = self.pop();
                    let h_func = self.pop();
                    let f_func = self.pop();
                    self.push(RcVal::new(Val::ForkFunc { f_func, h_func, g_func }));
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
            match func.as_val() {
                Val::ExplicitFunc{code_index, closure_env} => {
                    let frame = self.current_frame_mut();
                    frame.code_index = *code_index;
                    frame.closure_env = closure_env.clone();
                    let locals_start = frame.locals_start;
                    self.locals_stack.truncate(locals_start);
                    self.locals_stack.push(x);
                    self.locals_stack.push(y.unwrap_or_else(|| self.zero.clone()));
                    return Ok(ChasedTail::GoTo(*code_index));
                }
                Val::AmbivalentFunc(monad, dyad) =>
                    func = if y.is_none() { monad.clone() } else { dyad.clone() },
                Val::AtopFunc { f_func, g_func } => {
                    x = self.call_val(f_func.clone(), x, None)?;
                    func = g_func.clone();
                }
                Val::BoundFunc { func: bound_func, y: bound_y } => {
                    y = Some(bound_y.clone());
                    func = bound_func.clone();
                }
                Val::ForkFunc { f_func, h_func, g_func } => {
                    let new_x = self.call_val(f_func.clone(), x.clone(), y.clone())?;
                    let new_y = Some(self.call_val(g_func.clone(), x, y)?);
                    x = new_x;
                    y = new_y;
                    func = h_func.clone();
                }
                Val::PrimFunc(PrimVerb::At) if x.is_func() && y.is_some() => {
                    func = x;
                    x = y.unwrap();
                    y = None;
                }
                Val::AdverbDerivedFunc { adverb: PrimAdverb::P, operand } => {
                    func = operand.clone();
                    y = None;
                }
                Val::AdverbDerivedFunc { adverb: PrimAdverb::Q, operand } => {
                    func = operand.clone();
                    if let Some(y_val) = y.take() {
                        x = y_val;
                    }
                }
                Val::AdverbDerivedFunc { adverb: PrimAdverb::AtColon, operand } => {
                    let index = self.call_val(operand.clone(), x.clone(), None)?;
                    func = self.prim_index(&y.expect("TODO: monadic @:v"), &index)?;
                    y = None;
                }
                Val::AdverbDerivedFunc { adverb: PrimAdverb::Dot, operand } => func = operand.clone(),
                Val::AdverbDerivedFunc { adverb: PrimAdverb::Tilde, operand } => {
                    match &mut y {
                        Some(y_val) => std::mem::swap(&mut x, y_val),
                        None => y = Some(x.clone()),
                    }
                    func = operand.clone();
                }
                Val::AdverbDerivedFunc { adverb: PrimAdverb::Backslash, operand } => {
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
                _ => return Ok(ChasedTail::Push(self.call_val(func, x, y)?))
            }
        }
    }

    // Calls `val` with arguments `x` and, if present, `y`. If `val` is a
    // function, run the function; otherwise, `val` is treated as a constant
    // function and this call results in `val`.
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
            Val::AmbivalentFunc(monad, dyad) =>
                self.call_val(if y.is_some() { dyad } else { monad }.clone(), x, y)?,
            Val::AtopFunc { f_func, g_func } => {
                let f_result = self.call_val(f_func.clone(), x, y)?;
                self.call_val(g_func.clone(), f_result, None)?
            }
            Val::BoundFunc { func, y } => self.call_val(func.clone(), x, Some(y.clone()))?,
            Val::ForkFunc { f_func, h_func, g_func } => {
                let f_result = self.call_val(f_func.clone(), x.clone(), y.clone())?;
                let g_result = self.call_val(g_func.clone(), x, y)?;
                self.call_val(h_func.clone(), f_result, Some(g_result))?
            }
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
            Plus => prim_add(x.as_val(), y.as_val()),
            Minus => prim_subtract(x.as_val(), y.as_val()),
            Asterisk => prim_multiply(x.as_val(), y.as_val()),
            Slash => prim_divide(x.as_val(), y.as_val()),
            DoubleSlash => prim_divmod(DivModOp::Div, x.as_val(), y.as_val()),
            Percent => prim_divmod(DivModOp::Mod, x.as_val(), y.as_val()),
            Caret => prim_pow(x.as_val(), y.as_val()),
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
            Val::PrimFunc(prim) => write_or!(out, "{prim}")?,
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
            Val::AdverbDerivedFunc { adverb, operand } => parenthesized_if(
                prec >= ConjunctionLeftOperand, out, |out| {
                    write_or!(out, "{adverb}")?;
                    self.prim_fmt(AdverbOperand, operand.as_val(), out)?;
                    Ok(())
                }
            )?,
            Val::AtopFunc { f_func, g_func } => parenthesized_if(
                prec >= Small, out, |out| {
                    self.prim_fmt(Toplevel, f_func, out)?;
                    write_or!(out, " ")?;
                    if let Val::BoundFunc { func, y } = g_func.as_val() {
                        self.prim_fmt(Small, func, out)?;
                        write_or!(out, " ")?;
                        self.prim_fmt(Small, y, out)?;
                    } else {
                        self.prim_fmt(Small, g_func, out)?;
                    }
                    Ok(())
                }
            )?,
            Val::BoundFunc { func, y } => parenthesized_if(prec >= Small, out, |out| {
                self.prim_fmt(Small, func, out)?;
                write_or!(out, " ")?;
                self.prim_fmt(Small, y, out)?;
                Ok(())
            })?,
            Val::ForkFunc { f_func, h_func, g_func } => parenthesized_if(
                prec >= Small, out, |out| {
                    self.prim_fmt(Toplevel, f_func, out)?;
                    write_or!(out, " ")?;
                    self.prim_fmt(Small, h_func, out)?;
                    write_or!(out, " ")?;
                    self.prim_fmt(Small, g_func, out)?;
                    Ok(())
                }
            )?,
            Val::AmbivalentFunc(monad, dyad) => parenthesized_if(
                prec >= ConjunctionLeftOperand, out, |out| {
                    self.prim_fmt(ConjunctionLeftOperand, monad.as_val(), out)?;
                    write_or!(out, " : ")?;
                    self.prim_fmt(AdverbOperand, dyad.as_val(), out)?;
                    Ok(())
                }
            )?,
            Val::ExplicitFunc { .. } => {
                // map code index -> tokens?
                write_or!(out, "{{explicit func; TODO: implement printing}}")?
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
            Val::ExplicitFunc { code_index, closure_env } => {
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
    let s = match x {
        Val::Char(_) => &"char",
        Val::Int(_) => &"int",
        Val::Float(_) => &"float",
        Val::U8s(_) => &"string",
        Val::I64s(_) => &"int list",
        Val::F64s(_) => &"float list",
        Val::Vals(_) => &"val list",

        Val::PrimFunc(_) |
        Val::AdverbDerivedFunc{..} |
        Val::AmbivalentFunc(_, _) |
        Val::AtopFunc{..} |
        Val::BoundFunc{..} |
        Val::ForkFunc{..} |
        Val::ExplicitFunc{..} => &"function",
    };
    s.as_bytes().to_vec()
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
        _ => return err!("domain\nCan't add {x:?} and {y:?}"),
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
        _ => return err!("domain\nCan't subtract {x:?} and {y:?}"),
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
        _ => return err!("domain\nCan't add {x:?} and {y:?}"),
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
        (Int(i), Float(f)) => Int(op!(*i, *f as i64)),
        (Float(f), Int(i)) => Int(op!(*f as i64, *i)),
        (Float(f), Float(g)) => Int(op!(*f, *g)),
        (I64s(ints), Int(i)) => I64s(
            ints.iter().map(|int| op!(*int, *i)).collect()
        ),
        (Int(i), I64s(ints)) => I64s(
            ints.iter().map(|int| op!(*i, *int)).collect()
        ),
        (I64s(ints), Float(f)) => {
            let i = *f as i64;
            I64s(ints.iter().map(|int| op!(*int, i)).collect())
        }
        (Int(i), F64s(fs)) => I64s(fs.iter().map(|f| op!(*i, *f as i64)).collect()),
        (F64s(fs), Int(i)) => I64s(fs.iter().map(|f| op!(*f as i64, *i)).collect()),
        (F64s(xs), F64s(ys)) => I64s(zip_map(xs, ys, |x, y| op!(x, y) as i64)?),
        (F64s(fs), Float(float)) => I64s(map(fs, |f| op!(f, float) as i64)),
        (Float(float), F64s(fs)) => I64s(map(fs, |f| op!(float, f) as i64)),
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
            return err!("domain\nCan't {op_str} {x:?} by {y:?}")
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

        _ => return err!("domain\nCan't divide {x:?} by {y:?}"),
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
        _ => return err!("domain\nCan't raise {x:?} to power {y:?}"),
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

fn prim_copy(x: &RcVal, y: &RcVal) -> Result<RcVal, String> {
    use Val::*;
    fn unexpected_y(y: &Val) -> String {
        format!("domain\nExpected non-negative integer, got {y:?}")
    }

    fn replicate_all<A: Clone>(xs: &[A], y: usize) -> Vec<A> {
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
        err!("length mismatch: {xlen} vs {ylen}")
    }
}

fn match_length(xlen: usize, ylen: usize) -> Result<(), String> {
    if xlen == ylen { return Ok(()); }
    // TODO include name/position of verb
    err!("length mismatch: {xlen} vs {ylen}")
}
