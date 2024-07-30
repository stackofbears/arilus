use std::iter;
use std::rc::Rc;

use crate::util::{length_mismatch_error, match_lengths};
use crate::val::*;

type Res<A> = Result<A, String>;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AtomOp2 {
    Add,
    Mul,
    Sub,
    Div,
    IntDiv,
    Mod,
    Pow,
}

pub trait Op2<X, Y> {
    type Out: ToVal + VecToVal;
    fn op(x: X, y: Y) -> Res<Self::Out>;
}

trait Op2Val<X, Y> {
    fn op2val(x: X, y: Y) -> Res<Val>;
}

impl<Op, X: NamedType, Y: NamedType> Op2Val<X, Y> for Op {
    default fn op2val(_: X, _: Y) -> Res<Val> {
        Err(domain_error::<X, Y>(&""))
    }
}

// These should match Val::type_name
pub trait NamedType { const TYPE_NAME: &'static str; }
impl NamedType for u8 { const TYPE_NAME: &'static str = &"char"; }
impl NamedType for i64 { const TYPE_NAME: &'static str = &"int"; }
impl NamedType for f64 { const TYPE_NAME: &'static str = &"float"; }
impl NamedType for &Func { const TYPE_NAME: &'static str = &"function"; }

impl<A: NamedType> NamedType for Vec<A> {
    const TYPE_NAME: &'static str = A::TYPE_NAME;
}
impl<A: NamedType> NamedType for &[A] {
    const TYPE_NAME: &'static str = A::TYPE_NAME;
}

#[cold]
fn domain_error_concrete(a_type: &str, b_type: &str, detail: &str) -> String {
    format!("domain\nUnsupported arguments: {a_type} and {b_type}{}{detail}",
            &if detail.is_empty() { "" } else { "\n" })
}

#[cold]
pub fn domain_error<A: NamedType, B: NamedType>(detail: &str) -> String {
    domain_error_concrete(A::TYPE_NAME, B::TYPE_NAME, detail)
}

macro_rules! impl_op2 {
    ($self_type:ty, ($x:ident : $x_ty:ty, $y:ident : $y_ty:ty) -> Res<$out:ty> { $($body:tt)* } $($rest:tt)*) => {
        impl Op2<$x_ty, $y_ty> for $self_type {
            type Out = $out;
            #[inline]
            fn op($x: $x_ty, $y: $y_ty) -> Res<$out> {
                $($body)*
            }
        }

        impl Op2Val<$x_ty, $y_ty> for $self_type {
            #[inline]
            fn op2val(x: $x_ty, y: $y_ty) -> Res<Val> {
                Ok(Self::op(x, y)?.to_val())
            }
        }

        impl<'a> Op2Val<$x_ty, &'a [$y_ty]> for $self_type {
            #[inline]
            fn op2val(x: $x_ty, y: &'a [$y_ty]) -> Res<Val> {
                Ok(y.into_iter().copied().map(|y| Self::op(x, y)).collect::<Res<Vec<_>>>()?.to_val())
            }
        }

        impl Op2Val<$x_ty, Vec<$y_ty>> for $self_type {
            #[inline]
            fn op2val(x: $x_ty, y: Vec<$y_ty>) -> Res<Val> {
                Ok(y.into_iter().map(|y| Self::op(x, y)).collect::<Res<Vec<_>>>()?.to_val())
            }
        }

        impl<'a> Op2Val<&'a [$x_ty], $y_ty> for $self_type {
            #[inline]
            fn op2val(x: &'a [$x_ty], y: $y_ty) -> Res<Val> {
                Ok(x.into_iter().copied().map(|x| Self::op(x, y)).collect::<Res<Vec<_>>>()?.to_val())
            }
        }

        impl<'a, 'b> Op2Val<&'a [$x_ty], &'b [$y_ty]> for $self_type {
            #[inline]
            fn op2val(x: &'a [$x_ty], y: &'b [$y_ty]) -> Res<Val> {
                match_lengths(x.len(), y.len())?;
                Ok(iter::zip(x.iter().copied(), y.iter().copied()).map(|(x, y)| Self::op(x, y)).collect::<Res<Vec<_>>>()?.to_val())
            }
        }

        impl<'a> Op2Val<&'a [$x_ty], Vec<$y_ty>> for $self_type {
            #[inline]
            fn op2val(x: &'a [$x_ty], y: Vec<$y_ty>) -> Res<Val> {
                match_lengths(x.len(), y.len())?;
                Ok(iter::zip(x.iter().copied(), y).map(|(x, y)| Self::op(x, y)).collect::<Res<Vec<_>>>()?.to_val())
            }
        }

        impl Op2Val<Vec<$x_ty>, $y_ty> for $self_type {
            #[inline]
            fn op2val(x: Vec<$x_ty>, y: $y_ty) -> Res<Val> {
                Ok(x.into_iter().map(|x| Self::op(x, y)).collect::<Res<Vec<_>>>()?.to_val())
            }
        }

        impl<'a> Op2Val<Vec<$x_ty>, &'a [$y_ty]> for $self_type {
            #[inline]
            fn op2val(x: Vec<$x_ty>, y: &'a [$y_ty]) -> Res<Val> {
                match_lengths(x.len(), y.len())?;
                Ok(iter::zip(x, y.iter().copied()).map(|(x, y)| Self::op(x, y)).collect::<Res<Vec<_>>>()?.to_val())
            }
        }

        impl Op2Val<Vec<$x_ty>, Vec<$y_ty>> for $self_type {
            #[inline]
            fn op2val(x: Vec<$x_ty>, y: Vec<$y_ty>) -> Res<Val> {
                match_lengths(x.len(), y.len())?;
                Ok(iter::zip(x, y).map(|(x, y)| Self::op(x, y)).collect::<Res<Vec<_>>>()?.to_val())
            }
        }

        impl_op2!($self_type, $($rest)*);
    };

    ($self_type:ty,) => {};
}

enum Add {}
impl_op2!(
    Add,
    (x: u8, y: i64) -> Res<u8> { Ok((x as i64 + y) as u8) }
    (x: u8, y: f64) -> Res<u8> {
        if y.trunc() == y {
            Ok((x as i64 + y as i64) as u8)
        } else {
            Err(domain_error::<u8, f64>(&"(An int-convertible float would've worked)"))
        }
    }
    (x: i64, y: u8)  -> Res<u8>  { Self::op(y, x) }
    (x: i64, y: i64) -> Res<i64> { Ok(x + y) }
    (x: i64, y: f64) -> Res<f64> { Ok(x as f64 + y) }
    (x: f64, y: u8)  -> Res<u8>  { Self::op(y, x) }
    (x: f64, y: i64) -> Res<f64> { Self::op(y, x) }
    (x: f64, y: f64) -> Res<f64> { Ok(x + y) }
);

enum Sub {}
impl_op2!(
    Sub,
    (x: u8, y: u8) -> Res<i64> { Ok(x as i64 - y as i64) }
    (x: u8, y: i64) -> Res<u8> { Ok((x as i64 - y) as u8) }
    (x: u8, y: f64) -> Res<u8> {
        if y.trunc() == y {
            Ok((x as i64 - y as i64) as u8)
        } else {
            Err(domain_error::<u8, f64>(&"(An int-convertible float would've worked)"))
        }
    }
    (x: i64, y: i64) -> Res<i64> { Ok(x - y) }
    (x: i64, y: f64) -> Res<f64> { Ok(x as f64 - y) }
    (x: f64, y: i64) -> Res<f64> { Ok(x - y as f64) }
    (x: f64, y: f64) -> Res<f64> { Ok(x - y) }
);

enum Mul {}
impl_op2!(
    Mul,
    (x: i64, y: i64) -> Res<i64> { Ok(x * y) }
    (x: i64, y: f64) -> Res<f64> { Ok(x as f64 * y) }
    (x: f64, y: i64) -> Res<f64> { Ok(x * y as f64) }
    (x: f64, y: f64) -> Res<f64> { Ok(x * y) }
);

enum Div {}
impl_op2!(
    Div,
    (x: i64, y: i64) -> Res<f64> { Ok(x as f64 / y as f64) }
    (x: i64, y: f64) -> Res<f64> { Ok(x as f64 / y) }
    (x: f64, y: i64) -> Res<f64> { Ok(x / y as f64) }
    (x: f64, y: f64) -> Res<f64> { Ok(x / y) }
);

enum IntDiv {}
impl_op2!(
    IntDiv,
    (x: i64, y: i64) -> Res<i64> { Ok(x.div_euclid(y)) }
    (x: i64, y: f64) -> Res<i64> { Ok(x.div_euclid(y as i64)) }
    (x: f64, y: i64) -> Res<i64> { Ok((x.floor() as i64).div_euclid(y)) }
    (x: f64, y: f64) -> Res<i64> { Ok(x.div_euclid(y).floor() as i64) }
);

enum Mod {}
impl_op2!(
    Mod,
    (x: i64, y: i64) -> Res<i64> { Ok(x.rem_euclid(y)) }
    (x: i64, y: f64) -> Res<f64> { Ok((x as f64).rem_euclid(y)) }
    (x: f64, y: i64) -> Res<f64> { Ok(x.rem_euclid(y as f64)) }
    (x: f64, y: f64) -> Res<f64> { Ok(x.rem_euclid(y)) }
);

enum Pow {}
enum Either<A, B> { Left(A), Right(B) }

impl ToVal for Either<i64, f64> {
    #[inline]
    fn to_val(self) -> Val {
        match self {
            Either::Left(int) => Val::Int(int),
            Either::Right(float) => Val::Float(float),
        }
    }
}

impl VecToVal for Either<i64, f64> {
    #[inline]
    fn vec_to_val(v: Vec<Self>) -> Val {
        use Either::*;
        let mut ints: Vec<i64> = Vec::with_capacity(v.len());
        for i in 0..v.len() {
            match &v[i] {
                Left(int) => ints.push(*int),
                Right(float) => {
                    let mut floats: Vec<f64> = ints.drain(..).map(|int| int as f64).collect();
                    floats.push(*float);
                    for j in (i+1)..v.len() {
                        floats.push(match &v[j] {
                            Left(int) => *int as f64,
                            Right(float) => *float,
                        });
                    }
                    return Val::F64s(Rc::new(floats));
                }
            }
        }
        Val::I64s(Rc::new(ints))
    }
}

impl_op2!(
    Pow,
    (x: i64, y: i64) -> Res<Either<i64, f64>> {
        Ok(if y >= 0 {
            Either::Left(x.pow(y as u32)) 
        } else {
            Either::Right((x as f64).powi(y as i32))
        })
    }
    
    (x: i64, y: f64) -> Res<f64> { Ok((x as f64).powf(y)) }
    (x: f64, y: i64) -> Res<f64> { Ok(x.powi(y as i32)) }
    (x: f64, y: f64) -> Res<f64> { Ok(x.powf(y)) }
);

// TODO flip_for_error
fn monomorph<X: NamedType, Y: NamedType>(op: AtomOp2, flip: bool, x: X, y: Y) -> Res<Val> {
    match op {
        AtomOp2::Add => Add::op2val(x, y),
        AtomOp2::Mul => Mul::op2val(x, y),
        AtomOp2::Sub    => if !flip { Sub::op2val(x, y) }    else { Sub::op2val(y, x) },
        AtomOp2::Div    => if !flip { Div::op2val(x, y) }    else { Div::op2val(y, x) },
        AtomOp2::IntDiv => if !flip { IntDiv::op2val(x, y) } else { IntDiv::op2val(y, x) },
        AtomOp2::Mod    => if !flip { Mod::op2val(x, y) }    else { Mod::op2val(y, x) },
        AtomOp2::Pow    => if !flip { Pow::op2val(x, y) }    else { Pow::op2val(y, x) },
    }
}

pub trait Atom: Sized + Copy + NamedType {}

impl Atom for u8 {}
impl Atom for i64 {}
impl Atom for f64 {}
impl Atom for &Func {}

pub trait ToVal {
    fn to_val(self) -> Val;
}
impl ToVal for u8 {
    #[inline]
    fn to_val(self) -> Val { Val::Char(self) }
}
impl ToVal for i64 {
    #[inline]
    fn to_val(self) -> Val { Val::Int(self) }
}
impl ToVal for f64 {
    #[inline]
    fn to_val(self) -> Val { Val::Float(self) }
}
impl ToVal for Func {
    #[inline]
    fn to_val(self) -> Val { Val::Function(Rc::new(self)) }
}
impl ToVal for Val {
    #[inline]
    fn to_val(self) -> Val { self }
}
impl ToVal for Rc<Func> {
    #[inline]
    fn to_val(self) -> Val { Val::Function(self) }
}
impl ToVal for NoValEmptyEnum {
    #[inline]
    // SAFETY Self can't be instantiated, so this is never called.
    fn to_val(self) -> Val { unsafe { std::hint::unreachable_unchecked() } }
}

pub trait VecToVal: Sized {
    fn vec_to_val(x: Vec<Self>) -> Val;
}

impl VecToVal for u8 {
    #[inline]
    fn vec_to_val(v: Vec<Self>) -> Val { Val::U8s(Rc::new(v)) }
}
impl VecToVal for i64 {
    #[inline]
    fn vec_to_val(v: Vec<Self>) -> Val { Val::I64s(Rc::new(v)) }
}
impl VecToVal for f64 {
    #[inline]
    fn vec_to_val(v: Vec<Self>) -> Val { Val::F64s(Rc::new(v)) }
}
impl VecToVal for Val {
    #[inline]
    fn vec_to_val(v: Vec<Self>) -> Val { Val::Vals(Rc::new(v)) }
}
impl VecToVal for NoValEmptyEnum {
    #[inline]
    fn vec_to_val(_: Vec<NoValEmptyEnum>) -> Val { Val::I64s(Rc::new(vec![])) }
}

impl<A> ToVal for Vec<A> where A: VecToVal {
    fn to_val(self) -> Val { <A as VecToVal>::vec_to_val(self) }
}

fn match_lengths_flipped(xlen: usize, ylen: usize, flip: bool) -> Res<()> {
    if xlen != ylen { length_mismatch_error_flipped(xlen, ylen, flip) }
    else { Ok(()) }
}

#[cold]
fn length_mismatch_error_flipped(xlen: usize, ylen: usize, flip: bool) -> Res<()> {
    if !flip {
        length_mismatch_error(xlen, ylen)
    } else {
        length_mismatch_error(ylen, xlen)
    }
}

pub trait AtomConsumer<A: Atom> {
    fn eat_atom(self, a: A) -> Res<Val>;
    fn eat_atom_slice(self, a: &[A]) -> Res<Val>;
    fn eat_atom_vec(self, a: Vec<A>) -> Res<Val>;
}

pub trait ValConsumer
where Self: AtomConsumer<u8>,
      Self: AtomConsumer<i64>,
      Self: AtomConsumer<f64>,
      Self: for<'a> AtomConsumer<&'a Func> {
    fn eat_vals<Ys: ExactSizeIterator<Item: IsVal>>(self, ys: Ys) -> Res<Val>;
}

pub trait IsVal {
    fn dispatch<F: ValConsumer>(self, f: F) -> Res<Val>;
}

impl IsVal for Val {
    fn dispatch<F: ValConsumer>(self, f: F) -> Res<Val> {
        use Val::*;
        match self {
            Char(x) => f.eat_atom(x),
            Int(x) => f.eat_atom(x),
            Float(x) => f.eat_atom(x),
            Function(x) => f.eat_atom(x.as_ref()),
            U8s(x) => match Rc::try_unwrap(x) {
                Ok(x) => f.eat_atom_vec(x),
                Err(x) => f.eat_atom_slice(x.as_slice()),
            }
            I64s(x) => match Rc::try_unwrap(x) {
                Ok(x) => f.eat_atom_vec(x),
                Err(x) => f.eat_atom_slice(x.as_slice()),
            }
            F64s(x) => match Rc::try_unwrap(x) {
                Ok(x) => f.eat_atom_vec(x),
                Err(x) => f.eat_atom_slice(x.as_slice()),
            }
            Vals(x) => match Rc::try_unwrap(x) {
                Ok(x) => f.eat_vals(x.into_iter()),
                Err(x) => f.eat_vals(x.as_slice().into_iter()),
            }
        }
    }
}

impl IsVal for &Val {
    fn dispatch<F: ValConsumer>(self, f: F) -> Res<Val> {
        use Val::*;
        match self {
            Char(x) => f.eat_atom(*x),
            Int(x) => f.eat_atom(*x),
            Float(x) => f.eat_atom(*x),
            Function(x) => f.eat_atom(x.as_ref()),
            U8s(x) => f.eat_atom_slice(x.as_slice()),
            I64s(x) => f.eat_atom_slice(x.as_slice()),
            F64s(x) => f.eat_atom_slice(x.as_slice()),
            Vals(x) => f.eat_vals(x.as_slice().into_iter()),
        }
    }
}

#[inline]
pub fn dispatch_to_atoms<X: IsVal, Y: IsVal>(op: AtomOp2, x: X, y: Y) -> Res<Val> {
    y.dispatch(XVal::<X> { flip: false, op, x })
}

struct XVal<X> { flip: bool, op: AtomOp2, x: X }
impl<X: IsVal> ValConsumer for XVal<X> {
    #[inline]
    fn eat_vals<Ys: ExactSizeIterator<Item: IsVal>>(self, ys: Ys) -> Res<Val> {
        self.x.dispatch(XVals { flip: !self.flip, x: ys, op: self.op })
    }
}

impl<X: IsVal, Y: Atom> AtomConsumer<Y> for XVal<X> {
    #[inline]
    fn eat_atom(self, y: Y) -> Res<Val> {
        self.x.dispatch(XAtom { flip: !self.flip, x: y, op: self.op })
    }

    #[inline]
    fn eat_atom_slice(self, y: &[Y]) -> Res<Val> {
        self.x.dispatch(XAtomSlice { flip: self.flip, op: self.op, x: y })
    }

    #[inline]
    fn eat_atom_vec(self, y: Vec<Y>) -> Res<Val> {
        self.x.dispatch(XAtomVec { flip: !self.flip, op: self.op, x: y })
    }
}

// x is atom
#[derive(Clone, Copy)]
struct XAtom<X> { flip: bool, op: AtomOp2, x: X }

impl<X: Atom> ValConsumer for XAtom<X> {
    fn eat_vals<Ys: ExactSizeIterator<Item: IsVal>>(mut self, ys: Ys) -> Res<Val> {
        self.flip = !self.flip;
        let vec = ys.map(|y| y.dispatch(self)).collect::<Res<Vec<_>>>()?;
        Ok(vec.to_val())
    }
}

impl<X: Atom, Y: Atom> AtomConsumer<Y> for XAtom<X> {
    #[inline]
    fn eat_atom(self, y: Y) -> Res<Val> {
        monomorph(self.op, self.flip, self.x, y)
    }

    #[inline]
    fn eat_atom_slice(self, y: &[Y]) -> Res<Val> {
        monomorph(self.op, self.flip, self.x, y)
    }

    #[inline]
    fn eat_atom_vec(self, y: Vec<Y>) -> Res<Val> {
        monomorph(self.op, self.flip, self.x, y)
    }
}

// x is Vec<atom>
struct XAtomVec<X> { flip: bool, op: AtomOp2, x: Vec<X> }
impl<X: Atom> ValConsumer for XAtomVec<X> {
    fn eat_vals<Ys: ExactSizeIterator<Item: IsVal>>(self, ys: Ys) -> Res<Val> {
        match_lengths_flipped(self.x.len(), ys.len(), self.flip)?;
        let vec = iter::zip(self.x, ys)
            .map(#[inline] |(x, y)| y.dispatch(XAtom { flip: !self.flip, op: self.op, x }))
            .collect::<Res<Vec<_>>>()?;
        Ok(vec.to_val())
    }
}

impl<X: Atom, Y: Atom> AtomConsumer<Y> for XAtomVec<X> {
    #[inline]
    fn eat_atom(self, y: Y) -> Res<Val> {
        monomorph(self.op, self.flip, self.x, y)
    }

    #[inline]
    fn eat_atom_slice(self, y: &[Y]) -> Res<Val> {
        monomorph(self.op, self.flip, self.x, y)
    }

    #[inline]
    fn eat_atom_vec(self, y: Vec<Y>) -> Res<Val> {
        monomorph(self.op, self.flip, self.x, y)
    }
}

// x is &[atom]
struct XAtomSlice<'a, X> { flip: bool, op: AtomOp2, x: &'a [X] }
impl<'a, X: Atom> ValConsumer for XAtomSlice<'a, X> {
    fn eat_vals<Ys: ExactSizeIterator<Item: IsVal>>(self, ys: Ys) -> Res<Val> {
        match_lengths_flipped(self.x.len(), ys.len(), self.flip)?;
        let vec = iter::zip(self.x.iter().copied(), ys)
            .map(#[inline] |(x, y)| y.dispatch(XAtom { flip: !self.flip, op: self.op, x }))
            .collect::<Res<Vec<_>>>()?;
        Ok(vec.to_val())
    }
}

impl<'a, X: Atom, Y: Atom> AtomConsumer<Y> for XAtomSlice<'a, X> {
    #[inline]
    fn eat_atom(self, y: Y) -> Res<Val> {
        monomorph(self.op, self.flip, self.x, y)
    }

    #[inline]
    fn eat_atom_slice(self, y: &[Y]) -> Res<Val> {
        monomorph(self.op, self.flip, self.x, y)
    }

    #[inline]
    fn eat_atom_vec(self, y: Vec<Y>) -> Res<Val> {
        monomorph(self.op, self.flip, self.x, y)
    }
}

// x is Vec<Val>::IntoIter or &[Val]::IntoIter
struct XVals<Xs> { flip: bool, op: AtomOp2, x: Xs }

impl<Xs: ExactSizeIterator<Item: IsVal>> ValConsumer for XVals<Xs> {
    fn eat_vals<Ys: ExactSizeIterator<Item: IsVal>>(self, ys: Ys) -> Res<Val> {
        match_lengths_flipped(self.x.len(), ys.len(), self.flip)?;
        let vec = iter::zip(self.x, ys)
            .map(#[inline] |(x, y)|
                 y.dispatch(XVal::<Xs::Item> { flip: self.flip, op: self.op, x })
            ).collect::<Res<Vec<_>>>()?;
        Ok(vec.to_val())
    }
}

impl<Xs: ExactSizeIterator<Item: IsVal>, Y: Atom> AtomConsumer<Y> for XVals<Xs> {
    fn eat_atom(self, y: Y) -> Res<Val> {
        let x_atom = XAtom { flip: !self.flip, op: self.op, x: y };
        let vec = self.x.into_iter()
            .map(#[inline] |y| y.dispatch(x_atom))
            .collect::<Res<Vec<_>>>()?;
        Ok(vec.to_val())
    }

    fn eat_atom_slice(self, y: &[Y]) -> Res<Val> {
        match_lengths_flipped(self.x.len(), y.len(), self.flip)?;
        let vec = iter::zip(self.x, y)
            .map(#[inline] |(x, y)| x.dispatch(XAtom { flip: !self.flip, op: self.op, x: *y }))
            .collect::<Res<Vec<_>>>()?;
        Ok(vec.to_val())
    }

    fn eat_atom_vec(self, y: Vec<Y>) -> Res<Val> {
        match_lengths_flipped(self.x.len(), y.len(), self.flip)?;
        let vec = iter::zip(self.x, y)
            .map(#[inline] |(x, y)| x.dispatch(XAtom { flip: !self.flip, op: self.op, x: y }))
            .collect::<Res<Vec<_>>>()?;
        Ok(vec.to_val())
    }
}
