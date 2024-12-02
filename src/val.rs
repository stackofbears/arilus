use std::{
    cell::RefCell,
    cmp::Ordering,
    rc::Rc,
};

use crate::lex::*;
use crate::util::{cold, float_as_int, match_lengths};
use crate::bytecode::{PrimFunc, ArgSpec};

// The Val representation isn't very efficient for now.
//
// TODO NaN-box
// TODO switch to custom reference counting so we can store the count inline with array data
// 
// TODO Ref
#[derive(Debug, Clone)]
pub enum Val {
    Char(u8),
    Int(i64),
    Float(f64),
    Function(Rc<Func>),
    U8s(Rc<Vec<u8>>),
    I64s(Rc<Vec<i64>>),
    F64s(Rc<Vec<f64>>),
    Vals(Rc<Vec<Val>>),
}

#[derive(Debug, Clone)]
pub enum Func {
    Unapplied(UnappliedFunc),

    // TODO supersede AdverbDerived?
    PartiallyApplied { 
        func: UnappliedFunc,

        bound_arg_spec: ArgSpec,

        // Items correspond with the 1s in provided_arg_spec.mask().
        bound_args: Vec<Val>,
    }
}

#[derive(Debug, Clone)]
pub enum UnappliedFunc {
    Prim(PrimFunc),

    AdverbDerived {
        adverb: PrimAdverb,
        operand: Val,
    },

    ConjunctionDerived {
        conjunction: PrimConjunction,
        operand1: Val,
        operand2: Val,
    },

    Explicit(ExplicitFunc),
}

#[derive(Debug, Clone)]
pub struct ExplicitFunc {
    pub code_index: usize,

    // TODO closure environments should be immutable, but right now they act like mutable references
    // (confusing).
    pub closure_env: Rc<RefCell<Vec<Val>>>,
}

// Pattern for matching atoms
macro_rules! atom {
    () => {
        Val::Char(_) | Val::Int(_) | Val::Float(_) | Val::Function(_)
    }
}
pub(crate) use atom;

impl Val {
    pub fn empty_list() -> Val {
        // TODO
        Val::I64s(Rc::new(vec![]))
    }

    pub fn empty_list_of_same_type(val: &Val) -> Val {
        use Val::*;
        match val {
            Char(_) | U8s(_) => U8s(Rc::new(vec![])),
            Int(_) | I64s(_) => I64s(Rc::new(vec![])),
            Float(_) | F64s(_) => F64s(Rc::new(vec![])),
            Function(_) | Vals(_) => Vals(Rc::new(vec![])),
        }
    }

    pub fn prim_func(prim: PrimFunc) -> Val {
        // This is too much!
        Val::Function(Rc::new(Func::Unapplied(UnappliedFunc::Prim(prim))))
    }

    pub fn explicit_func(explicit: ExplicitFunc) -> Val {
        // This is too much!
        Val::Function(Rc::new(Func::Unapplied(UnappliedFunc::Explicit(explicit))))
    }

    pub fn get_explicit(&self) -> Option<&ExplicitFunc> {
        if let Val::Function(func) = self {
            if let Func::Unapplied(UnappliedFunc::Explicit(explicit)) = func.as_ref() {
                return Some(explicit);
            }
        }
        None
    }

    pub fn adverb_derived_func(adverb: PrimAdverb, operand: Val) -> Val {
        // This is too much!
        Val::Function(Rc::new(Func::Unapplied(UnappliedFunc::AdverbDerived {
            adverb, operand
        })))
    }

    pub fn conjunction_derived_func(
        conjunction: PrimConjunction, operand1: Val, operand2: Val
    ) -> Val {
        // This is too much!
        Val::Function(Rc::new(Func::Unapplied(UnappliedFunc::ConjunctionDerived {
            conjunction, operand1, operand2
        })))
    }

    pub fn as_val(&self) -> &Self { &self }

    pub fn is_func(&self) -> bool {
        matches!(self, Val::Function(_))
    }

    pub fn is_constant_function(&self) -> bool {
        if let Val::Function(func) = self {
            func.is_constant_function()
        } else {
            false
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Val::Char(_) => &"char",
            Val::Int(_) => &"int",
            Val::Float(_) => &"float",
            Val::U8s(_) => &"string",
            Val::I64s(_) => &"int list",
            Val::F64s(_) => &"float list",
            Val::Vals(_) => &"val list",
            Val::Function(_) => &"function",
        }
    }

    pub fn is_falsy(&self) -> bool {
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

    pub fn len(&self) -> Option<usize> {
        use Val::*;
        match self {
            atom!() => None,
            U8s(vec) => Some(vec.len()),
            I64s(vec) => Some(vec.len()),
            F64s(vec) => Some(vec.len()),
            Vals(vec) => Some(vec.len()),
        }
    }
}

// impls for sorting.

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

impl PartialEq for Func {
    fn eq(&self, other: &Func) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Func {}

impl PartialOrd for Func {
    fn partial_cmp(&self, other: &Func) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


impl Func {
    pub fn is_constant_function(&self) -> bool {
        matches!(self,
                 Func::Unapplied(UnappliedFunc::AdverbDerived{adverb: PrimAdverb::Underscore {..}, ..}))
    }

    fn key_variant(&self) -> u32 {
        use Func::*;
        use UnappliedFunc::*;
        match self {
            Unapplied(Prim(_)) => 0,
            Unapplied(AdverbDerived{..}) => 1,
            Unapplied(ConjunctionDerived{..}) => 2,
            Unapplied(Explicit{..}) => 3,
            PartiallyApplied{..} => 4,
        }
    }
}

impl Ord for Func {
    fn cmp(&self, other: &Func) -> Ordering {
        use Func::*;
        use UnappliedFunc::*;
        match (self, other) {
            (Unapplied(Prim(_)),
             Unapplied(Prim(_))) => todo!("Sort primitives"),

            (Unapplied(Explicit(ExplicitFunc{code_index: x, ..})),
             Unapplied(Explicit(ExplicitFunc{code_index: y, ..}))) => x.cmp(y),

            (Unapplied(AdverbDerived{operand: x, ..}),
             Unapplied(AdverbDerived{operand: y, ..})) => x.cmp(y),

            (Unapplied(ConjunctionDerived{operand1: x1, operand2: x2, ..}),
             Unapplied(ConjunctionDerived{operand1: y1, operand2: y2, ..})) => x1.cmp(y1).then_with(|| x2.cmp(y2)),

            (PartiallyApplied { func: _, .. },
             PartiallyApplied { func: _, .. }) => todo!("Sort partially applied functions"),

            _ => self.key_variant().cmp(&other.key_variant()),
        }
    }
}

impl Ord for Val {
    fn cmp(&self, other: &Val) -> Ordering {
        use Val::*;

        fn key_variant(x: &Val) -> u32 {
            match x {
                Char(_) => 0,
                Int(_) => 1,
                Float(_) => 2,
                U8s(_) => 3,
                I64s(_) => 4,
                F64s(_) => 5,
                Vals(_) => 6,
                Function(func) => 7 + func.key_variant(),
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

            (U8s(x), U8s(y)) => x.cmp(y),
            (I64s(x), I64s(y)) => x.cmp(y),

            (I64s(is), F64s(fs)) => ints_floats_cmp(is, fs),
            (F64s(fs), I64s(is)) => ints_floats_cmp(is, fs).reverse(),

            (Vals(x), Vals(y)) => x.cmp(y),

            (Function(f), Function(g)) => f.cmp(g),

            _ => if let (Some(0), Some(0)) = (self.len(), other.len()) {
                Ordering::Equal
            } else {
                key_variant(self).cmp(&key_variant(other))
            }
        }
    }
}

// For the sake of safety, this type must never be instantiated. Ideally, we
// would use ! instead, but it isn't stabilized as a type.
#[derive(Clone, Copy)]
pub enum NoValEmptyEnum {}
impl Default for NoValEmptyEnum {
    fn default() -> Self { unsafe { std::hint::unreachable_unchecked() } }
}

pub fn index_or_cycle_val(val: &Val, i: usize) -> Option<Val> {
    use Val::*;
    Some(match val {
        atom!() => val.clone(),
        U8s(cs) => Val::Char(*cs.get(i)?),
        I64s(is) => Val::Int(*is.get(i)?),
        F64s(fs) => Val::Float(*fs.get(i)?),
        Vals(vs) => vs.get(i)?.clone(),
    })
}

pub fn collect_list<E, I: Iterator<Item=Result<Val, E>>>(mut it: I) -> Result<Val, E> {
    enum List {
        U8s(Vec<u8>),
        I64s(Vec<i64>),
        F64s(Vec<f64>),
        Vals(Vec<Val>),
    }

    let cap = match it.size_hint() {
        (lower, None) => lower,
        (_, Some(upper)) => upper,
    };

    let mut list = match it.next() {
        None => return Ok(Val::I64s(Rc::new(vec![]))),
        Some(Err(err)) => return cold(Err(err)),
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
                        fs.reserve(cap.checked_sub(fs.len()).unwrap_or(1));
                        fs.push(*f);
                        list = List::F64s(fs);
                    }
                }
                _ => {
                    let mut vals: Vec<Val> =
                        ints.drain(..).map(|i| Val::Int(i)).collect();
                    vals.reserve(cap.checked_sub(vals.len()).unwrap_or(1));
                    vals.push(val);
                    list = List::Vals(vals);
                }
            },
            List::F64s(fs) => match val.as_val() {
                Val::Float(f) => fs.push(*f),
                Val::Int(i) => fs.push(*i as f64),
                _ => {
                    let mut vals: Vec<Val> =
                        fs.drain(..).map(|f| Val::Float(f)).collect();
                    vals.reserve(cap.checked_sub(vals.len()).unwrap_or(1));
                    vals.push(val);
                    list = List::Vals(vals);
                }
            },
            List::U8s(cs) => match val.as_val() {
                Val::Char(c) => cs.push(*c),
                _ => {
                    let mut vals: Vec<Val> =
                        cs.drain(..).map(|c| Val::Char(c)).collect();
                    vals.reserve(cap.checked_sub(vals.len()).unwrap_or(1));
                    vals.push(val);
                    list = List::Vals(vals);
                }
            },
            List::Vals(vals) => vals.push(val),
        }
    }

    Ok(match list {
        List::U8s(cs) => Val::U8s(Rc::new(cs)),
        List::I64s(ints) => Val::I64s(Rc::new(ints)),
        List::F64s(fs) => Val::F64s(Rc::new(fs)),
        List::Vals(vals) => Val::Vals(Rc::new(vals)),
    })
}

#[derive(Clone)]
pub struct ValIter {
    pub x: Val,
    pub i: usize,
    pub len: usize,
}

pub fn iter_non_atom(x: Val) -> Option<ValIter> {
    let len = x.len()?;
    Some(ValIter { x, i: 0, len })
}

pub fn iter_val(x: Val) -> ValIter {
    let len = x.len().unwrap_or(1);
    ValIter { x, i: 0, len }
}

impl Iterator for ValIter {
    type Item = Val;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.len { return None }
        self.i += 1;
        index_or_cycle_val(&self.x, self.i - 1)
    }
}

impl ExactSizeIterator for ValIter {
    fn len(&self) -> usize {
        self.len - self.i
    }
}

impl DoubleEndedIterator for ValIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.i == self.len { return None }
        self.len -= 1;
        index_or_cycle_val(&self.x, self.len)
    }
}

pub struct ZippedVals {
    pub x: Val,
    pub y: Val,
    pub i: usize,
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

pub fn zip_vals(x: Val, y: Val) -> Result<Result<ZippedVals, String>, (Val, Val)> {
    match (x.len(), y.len()) {
        (None, None) => return Err((x, y)),
        (Some(xlen), Some(ylen)) => if let Err(err) = match_lengths(xlen, ylen) { return Ok(Err(err)) }
        _ => {}
    }
    Ok(Ok(ZippedVals { x, y, i: 0 }))
}
