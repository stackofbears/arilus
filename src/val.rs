use std::{
    borrow::Cow,
    cell::RefCell,
    cmp::Ordering,
    rc::Rc,
};

use crate::lex::*;
use crate::util;
use crate::bytecode::PrimFunc;

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
// Possible val format (nan-boxed):
// Val {
//    Char(u8),
//    Int(i64),
//    Float(f64),
//    Ints(ptr to [rc; N; elem1; elem2; ...; elemN]),
//    Prim(usize),
//    Function(ptr to [rc; code idx; N; closureVal1; closureVal2; ...; closureValN]),
// }
pub type RcVal = Rc<Val>;

// TODO Ref
// TODO Box to say functions shouldn't pervade
#[derive(Debug, Clone)]
pub enum Val {
    Char(u8),
    Int(i64),  // TODO TwoInts, ThreeInts
    Float(f64),
    Function(Func),
    U8s(Vec<u8>),
    I64s(Vec<i64>),
    F64s(Vec<f64>),
    Vals(Vec<RcVal>),
}

#[derive(Debug, Clone)]
pub enum Func {
    // TODO switch this out for the non-token type (some primitives won't have
    // token representations)
    Prim(PrimFunc),

    AdverbDerived {
        adverb: PrimAdverb,
        operand: RcVal,
    },

    // First element is the monadic case, second is the dyadic case
    Ambivalent(RcVal, RcVal),

    // TODO this can probably just be a closure
    Atop { f_func: RcVal, g_func: RcVal },
    Bound { func: RcVal, y: RcVal },
    Fork { f_func: RcVal, h_func: RcVal, g_func: RcVal },

    // TODO decide if closures should refer to a shared environment or be value
    // types (copying copies environment. Currently, we hold a reference to the env.
    //   a:1; F:{a:a+1}; G:F
    //   []F  \ 2
    //   []G  \ Currently this returns 3. Should G have its own copy of the env, making this 2?
    Explicit {
        // The function's first instruction (ALWAYS points after MakeFunc, so
        // you can get the function's instruction count with
        // code[func.code_index-1]
        code_index: usize,

        closure_env: Rc<RefCell<Vec<RcVal>>>,
    },
}

macro_rules! atom {
    () => {
        Val::Char(_) | Val::Int(_) | Val::Float(_) | Val::Function(_)
    }
}
pub(crate) use atom;

impl Val {
    pub fn as_val(&self) -> &Self { &self }

    // pub fn into_cow<'a>(self: &'a mut Rc<Val>) -> Cow<'a, Val> {
    //     match RcVal::try_unwrap(self) {
    //         Ok(value) => Cow::Owned(value),
    //         Err(reference) => Cow::Borrowed(reference.as_val()),
    //     }
    // }

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

    pub fn is_func(&self) -> bool {
        matches!(self, Val::Function(_))
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

// Val instances for sorting.

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
            Val::Float(f) => util::float_as_int(*f) == Some(*other),
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
        use Func::*;

        fn key_variant(x: &Val) -> u32 {
            match x {
                Char(_) => 0,
                Int(_) => 1,
                Float(_) => 2,
                Function(Prim(_)) => 3,
                Function(AdverbDerived {..}) => 4,
                Function(Ambivalent {..}) => 5,
                Function(Atop{..}) => 6,
                Function(Bound{..}) => 7,
                Function(Fork{..}) => 8,
                Function(Explicit {..}) => 9,
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

            (Function(f), Function(g)) => match (f, g) {
                (Prim(_), Prim(_)) => todo!("Sort primitives"),
                // TODO sort by closure envs instead? Would be cool, but they may
                // not have envs, and ideally there aren't semantic differences
                // between explicit and primitive functions
                (Explicit{code_index: x, ..}, Explicit{code_index: y, ..}) => x.cmp(y),
                (AdverbDerived{operand: x, ..}, AdverbDerived{operand: y, ..}) => x.cmp(y),

                (Ambivalent(x_monad, x_dyad), Ambivalent(y_monad, y_dyad)) =>
                    x_monad.cmp(y_monad).then_with(|| x_dyad.cmp(y_dyad)),

                (Atop { f_func: x_f_func, g_func: x_g_func },
                 Atop { f_func: y_f_func, g_func: y_g_func }) =>
                    x_f_func.cmp(y_f_func).then_with(|| x_g_func.cmp(y_g_func)),

                (Bound{func: x_func, y: x_y}, Bound{func: y_func, y: y_y}) =>
                    x_func.cmp(y_func).then_with(|| x_y.cmp(y_y)),

                (Fork{f_func: x_f_func, h_func: x_h_func, g_func: x_g_func},
                 Fork{f_func: y_f_func, h_func: y_h_func, g_func: y_g_func}) =>
                    x_f_func.cmp(y_f_func).then_with(|| x_h_func.cmp(y_h_func)).then_with(|| x_g_func.cmp(y_g_func)),
                _ => key_variant(self).cmp(&key_variant(other)),
            }
            (U8s(x), U8s(y)) => x.cmp(y),
            (I64s(x), I64s(y)) => x.cmp(y),

            (I64s(is), F64s(fs)) => ints_floats_cmp(is, fs),
            (F64s(fs), I64s(is)) => ints_floats_cmp(is, fs).reverse(),

            (Vals(x), Vals(y)) => x.cmp(y),
            _ => key_variant(self).cmp(&key_variant(other)),
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

pub fn index_or_cycle_val(val: &RcVal, i: usize) -> Option<RcVal> {
    use Val::*;
    Some(match val.as_val() {
        atom!() => val.clone(),
        I64s(is) => RcVal::new(Val::Int(*is.get(i)?)),
        F64s(fs) => RcVal::new(Val::Float(*fs.get(i)?)),
        U8s(cs) => RcVal::new(Val::Char(*cs.get(i)?)),
        Vals(vs) => vs.get(i)?.clone(),
    })
}

