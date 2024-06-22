use std::iter;
use std::rc::Rc;
use std::marker::PhantomData;

use crate::val::*;

type Res<A> = Result<A, String>;

pub trait Atom: Sized + Copy {}
impl Atom for u8 {}
impl Atom for i64 {}
impl Atom for f64 {}
impl Atom for &Func {}

pub trait ToVal {
    fn to_val(self) -> Val;
}
impl ToVal for u8 {
    #[inline(always)]
    fn to_val(self) -> Val { Val::Char(self) }
}
impl ToVal for i64 {
    #[inline(always)]
    fn to_val(self) -> Val { Val::Int(self) }
}
impl ToVal for f64 {
    #[inline(always)]
    fn to_val(self) -> Val { Val::Float(self) }
}
impl ToVal for Func {
    #[inline(always)]
    fn to_val(self) -> Val { Val::Function(Rc::new(self)) }
}
impl ToVal for Val {
    #[inline(always)]
    fn to_val(self) -> Val { self }
}
impl ToVal for Rc<Func> {
    #[inline(always)]
    fn to_val(self) -> Val { Val::Function(self) }
}
impl ToVal for NoValEmptyEnum {
    #[inline(always)]
    // SAFETY Self can't be instantiated, so this is never called.
    fn to_val(self) -> Val { unsafe { std::hint::unreachable_unchecked() } }
}

// impl ToVal for Rc<Vec<u8>> {
//     #[inline(always)]
//     fn to_val(self) -> Val { Val::U8s(self) }
// }
// impl ToVal for Rc<Vec<i64>> {
//     #[inline(always)]
//     fn to_val(self) -> Val { Val::I64s(self) }
// }
// impl ToVal for Rc<Vec<f64>> {
//     #[inline(always)]
//     fn to_val(self) -> Val { Val::F64s(self) }
// }
// impl ToVal for Rc<Vec<Val>> {
//     #[inline(always)]
//     fn to_val(self) -> Val { Val::Vals(self) }
// }
// impl<A> ToVal for Vec<A> where Rc<Vec<A>>: ToVal {
//     #[inline(always)]
//     fn to_val(self) -> Val { Rc::new(self).to_val() }
// }

pub trait VecToVal: Sized {
    fn vec_to_val(x: Vec<Self>) -> Val;
}

impl VecToVal for u8 {
    #[inline(always)]
    fn vec_to_val(v: Vec<Self>) -> Val { Val::U8s(Rc::new(v)) }
}
impl VecToVal for i64 {
    #[inline(always)]
    fn vec_to_val(v: Vec<Self>) -> Val { Val::I64s(Rc::new(v)) }
}
impl VecToVal for f64 {
    #[inline(always)]
    fn vec_to_val(v: Vec<Self>) -> Val { Val::F64s(Rc::new(v)) }
}
impl VecToVal for Val {
    #[inline(always)]
    fn vec_to_val(v: Vec<Self>) -> Val { Val::Vals(Rc::new(v)) }
}
impl VecToVal for NoValEmptyEnum {
    #[inline(always)]
    fn vec_to_val(_: Vec<NoValEmptyEnum>) -> Val { Val::I64s(Rc::new(vec![])) }
}


// impl VecToVal for A where Vec<A>: ToVal {
//     fn to_val(x: Vec<A>) -> Val { x.to_val() }
// }
// other way
impl<A> ToVal for Vec<A> where A: VecToVal {
    fn to_val(self) -> Val { <A as VecToVal>::vec_to_val(self) }
}

#[cold]
pub fn length_mismatch_error(xlen: usize, ylen: usize) -> String {
    format!("length mismatch: {xlen} vs {ylen}")
}

// impl for atoms, val, val ref
pub trait Op2<X: Atom, Y: Atom> {
    // TODO + vectoval?
    type Out: ToVal + VecToVal;
    fn op(x: X, y: Y) -> Res<Self::Out>;
    // fn op_vec_bare(x: Rc<Vec<X>>, y: Y) -> Res<Vec<Self::Out>>;
    // fn op_bare_vec(x: X, y: Rc<Vec<Y>>) -> Res<Vec<Self::Out>>;
    // fn op_vec_vec(x: Rc<Vec<X>>, y: Rc<Vec<Y>>) -> Res<Vec<Self::Out>>;
}

// TODO valrefconsumer (doesn't take vec) superclass of val consumer (can take vec)
pub trait AtomConsumer<A: Atom> {
    type AtomRet;
    fn eat_atom(self, a: A) -> Self::AtomRet;
    fn eat_atom_slice(self, a: &[A]) -> Self::AtomRet;
    fn eat_atom_vec(self, a: Vec<A>) -> Self::AtomRet;
}

// TODO can accept any exact size iterator where Item: IsVal?
pub trait MultiValConsumer {
    type MultiValRet;
    fn eat_val_slice(self, a: &[Val]) -> Self::MultiValRet;
    fn eat_val_vec(self, a: Vec<Val>) -> Self::MultiValRet;
}

pub trait ValConsumer
where Self: AtomConsumer<u8, AtomRet=Self::Ret>,
      Self: AtomConsumer<i64, AtomRet=Self::Ret>,
      Self: AtomConsumer<f64, AtomRet=Self::Ret>,
      Self: for<'a> AtomConsumer<&'a Func, AtomRet=Self::Ret>,
      Self: MultiValConsumer<MultiValRet=Self::Ret> {
    type Ret;
}

impl<A, Ret> ValConsumer for A
where A: AtomConsumer<u8, AtomRet=Ret> +
         AtomConsumer<i64, AtomRet=Ret> +
         AtomConsumer<f64, AtomRet=Ret> +
         for<'a> AtomConsumer<&'a Func, AtomRet=Ret> +
         MultiValConsumer<MultiValRet=Ret> {
    type Ret = Ret;
}

pub trait IsVal {
    fn dispatch<Ret, F: ValConsumer<Ret=Ret>>(self, f: F) -> Ret;
}

impl IsVal for Val {
    fn dispatch<Ret, F: ValConsumer>(self, f: F) -> F::Ret {
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
                Ok(x) => f.eat_val_vec(x),
                Err(x) => f.eat_val_slice(x.as_slice()),
            }
        }
    }
}

impl IsVal for &Val {
    fn dispatch<Ret, F: ValConsumer<Ret=Ret>>(self, f: F) -> Ret {
        use Val::*;
        match self {
            Char(x) => f.eat_atom(*x),
            Int(x) => f.eat_atom(*x),
            Float(x) => f.eat_atom(*x),
            Function(x) => f.eat_atom(x.as_ref()),
            U8s(x) => f.eat_atom_slice(x.as_slice()),
            I64s(x) => f.eat_atom_slice(x.as_slice()),
            F64s(x) => f.eat_atom_slice(x.as_slice()),
            Vals(x) => f.eat_val_slice(x.as_slice()),
        }
    }
}

pub trait Op2Flippable<X: Atom, Y: Atom>: Op2<X, Y> + Op2<Y, X> {}
impl<Op: Op2<X, Y> + Op2<Y, X>, X: Atom, Y: Atom> Op2Flippable<X, Y> for Op {}

pub trait AtomOp2Fixed<A: Atom>:
    Op2Flippable<A, u8> +
    Op2Flippable<A, i64> +
    Op2Flippable<A, f64> +
    for<'a> Op2Flippable<A, &'a Func> {}

impl<A: Atom, Op> AtomOp2Fixed<A> for Op
where Op: Op2Flippable<A, u8> +
          Op2Flippable<A, i64> +
          Op2Flippable<A, f64> +
          for<'a> Op2Flippable<A, &'a Func> {}

pub trait AtomOp2: AtomOp2Fixed<u8> + AtomOp2Fixed<i64> + AtomOp2Fixed<f64> + for<'a> AtomOp2Fixed<&'a Func> {}

pub fn dispatch_to_atoms<Op: AtomOp2, XVal: IsVal, YVal: IsVal>(x: XVal, y: YVal) -> Res<Val> {
    x.dispatch(XDispatch::<Op, YVal> { flip: false, y, _dummy: PhantomData })
}

// Initial dispatch on x
struct XDispatch<Op, YVal>{ flip: bool, y: YVal, _dummy: PhantomData<Op> }

impl<X: Atom, YVal: IsVal, Op: AtomOp2Fixed<X>> AtomConsumer<X> for XDispatch<Op, YVal> {
    type AtomRet = Res<Val>;
    fn eat_atom(self, x: X) -> Self::AtomRet {
        self.y.dispatch(XAtom::<Op, X>::new(x, self.flip))
    }
    fn eat_atom_slice(self, x: &[X]) -> Self::AtomRet {
        self.y.dispatch(XAtomSlice::<Op, X>::new(x, self.flip))
    }
    fn eat_atom_vec(self, x: Vec<X>) -> Self::AtomRet {
        self.y.dispatch(XAtomVec::<Op, X>::new(x, self.flip))
    }
}

impl<Op: AtomOp2, YVal: IsVal> MultiValConsumer for XDispatch<Op, YVal> {
    type MultiValRet = Res<Val>;
    fn eat_val_slice(self, a: &[Val]) -> Self::MultiValRet {
        self.y.dispatch(XValSlice::<Op>::new(a, self.flip))
    }
    fn eat_val_vec(self, a: Vec<Val>) -> Self::MultiValRet {
        self.y.dispatch(XValVec::<Op>::new(a, self.flip))
    }
}

// x is atom
struct XAtom<Op, X> { flip: bool, x: X, _dummy: PhantomData<Op> }
impl<Op, X> XAtom<Op, X> {
    #[inline(always)]
    fn new(x: X, flip: bool) -> Self {
        Self { flip, x, _dummy: PhantomData::<Op> }
    }
}

impl<Op, X: Clone> Clone for XAtom<Op, X> {
    #[inline(always)]
    fn clone(&self) -> Self { XAtom::new(self.x.clone(), self.flip) }
}
impl<Op, X: Copy> Copy for XAtom<Op, X> {}

impl<X: Atom, Y: Atom, Op: Op2Flippable<X, Y>> AtomConsumer<Y> for XAtom<Op, X> {
    type AtomRet = Res<Val>;
    #[inline(always)]
    fn eat_atom(self, y: Y) -> Self::AtomRet {
        let val = if !self.flip {
            Op::op(self.x, y)?.to_val()
        } else {
            Op::op(y, self.x)?.to_val()
        };
        Ok(val)
    }
    #[inline(always)]
    fn eat_atom_slice(self, y: &[Y]) -> Self::AtomRet {
        let val = if !self.flip {
            y.iter()
                .map(#[inline(always)] |y| Op::op(self.x, *y))
                .collect::<Res<Vec<_>>>()?.to_val()
        } else {
            y.iter()
                .map(#[inline(always)] |y| Op::op(*y, self.x))
                .collect::<Res<Vec<_>>>()?.to_val()
        };
        Ok(val)
    }
    #[inline(always)]
    fn eat_atom_vec(self, y: Vec<Y>) -> Self::AtomRet {
        let val = if !self.flip {
            y.into_iter()
                .map(#[inline(always)] |y| Op::op(self.x, y))
                .collect::<Res<Vec<_>>>()?.to_val()
        } else {
            y.into_iter()
                .map(#[inline(always)] |y| Op::op(y, self.x))
                .collect::<Res<Vec<_>>>()?.to_val()
        };
        Ok(val)
    }
}

impl<X: Atom, Op: AtomOp2Fixed<X>> MultiValConsumer for XAtom<Op, X> {
    type MultiValRet = Res<Val>;
    #[inline(always)]
    fn eat_val_slice(self, a: &[Val]) -> Self::MultiValRet {
        let outs = a.iter().map(|y| y.dispatch(self)).collect::<Res<Vec<Val>>>()?;
        Ok(outs.to_val())
    }
    #[inline(always)]
    fn eat_val_vec(self, a: Vec<Val>) -> Self::MultiValRet {
        let outs = a.into_iter().map(|y| y.dispatch(self)).collect::<Res<Vec<Val>>>()?;
        Ok(outs.to_val())
    }
}

// x is &[atom]
struct XAtomSlice<'a, Op, X> { flip: bool, x: &'a [X], _dummy: PhantomData<Op> }
impl<'a, Op, X: Clone> Clone for XAtomSlice<'a, Op, X> {
    #[inline(always)]
    fn clone(&self) -> Self { XAtomSlice::new(self.x, self.flip) }
}
impl<'a, Op, X: Copy> Copy for XAtomSlice<'a, Op, X> {}

impl<'a, Op, X> XAtomSlice<'a, Op, X> {
    #[inline(always)]
    fn new(x: &'a [X], flip: bool) -> Self {
        Self { flip, x, _dummy: PhantomData::<Op>}
    }
}

impl<'a, X: Atom, Y: Atom, Op: Op2Flippable<X, Y>> AtomConsumer<Y> for XAtomSlice<'a, Op, X> {
    type AtomRet = Res<Val>;
    #[inline(always)]
    fn eat_atom(self, y: Y) -> Self::AtomRet {
        let val = if !self.flip {
            self.x.iter()
                .map(|x| Op::op(*x, y))
                .collect::<Res<Vec<_>>>()?.to_val()
        } else {
            self.x.iter()
                .map(|x| Op::op(y, *x))
                .collect::<Res<Vec<_>>>()?.to_val()
        };
        Ok(val)
    }
    #[inline(always)]
    fn eat_atom_slice(self, y: &[Y]) -> Self::AtomRet {
        if self.x.len() != y.len() {
            return Err(length_mismatch_error(self.x.len(), y.len()))
        }
        let zipped = iter::zip(self.x, y);
        let val = if !self.flip {
            zipped
                .map(|(x, y)| Op::op(*x, *y))
                .collect::<Res<Vec<_>>>()?.to_val()
        } else {
            zipped
                .map(|(x, y)| Op::op(*y, *x))
                .collect::<Res<Vec<_>>>()?.to_val()
        };
        Ok(val)
    }
    #[inline(always)]
    fn eat_atom_vec(self, y: Vec<Y>) -> Self::AtomRet {
        if self.x.len() != y.len() {
            return Err(length_mismatch_error(self.x.len(), y.len()))
        }
        let zipped = iter::zip(self.x, y);
        let val = if !self.flip {
            zipped
                .map(|(x, y)| Op::op(*x, y))
                .collect::<Res<Vec<_>>>()?.to_val()
        } else {
            zipped
                .map(|(x, y)| Op::op(y, *x))
                .collect::<Res<Vec<_>>>()?.to_val()
        };
        Ok(val)
    }
}

impl<'a, X: Atom, Op: AtomOp2Fixed<X>> MultiValConsumer for XAtomSlice<'a, Op, X> {
    type MultiValRet = Res<Val>;
    fn eat_val_slice(self, y: &[Val]) -> Self::MultiValRet {
        if self.x.len() != y.len() {
            return Err(length_mismatch_error(self.x.len(), y.len()))
        }
        let val = iter::zip(self.x, y)
            .map(|(x, y)| y.dispatch(XAtom::<Op, X>::new(*x, self.flip)))
            .collect::<Res<Vec<_>>>()?.to_val();
        Ok(val)
    }
    fn eat_val_vec(self, y: Vec<Val>) -> Self::MultiValRet {
        if self.x.len() != y.len() {
            return Err(length_mismatch_error(self.x.len(), y.len()))
        }
        let val = iter::zip(self.x, y)
            .map(|(x, y)| y.dispatch(XAtom::<Op, X>::new(*x, self.flip)))
            .collect::<Res<Vec<_>>>()?.to_val();
        Ok(val)
    }
}

// x is vec<atom>
struct XAtomVec<Op, X> { flip: bool, x: Vec<X>, _dummy: PhantomData<Op> }
impl<Op, X> XAtomVec<Op, X> {
    #[inline(always)]
    fn new(x: Vec<X>, flip: bool) -> Self {
        Self { flip, x, _dummy: PhantomData::<Op> }
    }
}

impl<X: Atom, Y: Atom, Op: Op2Flippable<X, Y>> AtomConsumer<Y> for XAtomVec<Op, X> {
    type AtomRet = Res<Val>;
    #[inline(always)]
    fn eat_atom(self, y: Y) -> Self::AtomRet {
        let val = if !self.flip {
            self.x.into_iter()
                .map(|x| Op::op(x, y))
                .collect::<Res<Vec<_>>>()?.to_val()
        } else {
            self.x.into_iter()
                .map(|x| Op::op(y, x))
                .collect::<Res<Vec<_>>>()?.to_val()
        };
        Ok(val)
    }
    #[inline(always)]
    fn eat_atom_slice(self, y: &[Y]) -> Self::AtomRet {
        if self.x.len() != y.len() {
            return Err(length_mismatch_error(self.x.len(), y.len()))
        }
        let zipped = iter::zip(self.x, y);
        let val = if !self.flip {
            zipped
                .map(|(x, y)| Op::op(x, *y))
                .collect::<Res<Vec<_>>>()?.to_val()
        } else {
            zipped
                .map(|(x, y)| Op::op(*y, x))
                .collect::<Res<Vec<_>>>()?.to_val()
        };
        Ok(val)
    }
    #[inline(always)]
    fn eat_atom_vec(self, y: Vec<Y>) -> Self::AtomRet {
        if self.x.len() != y.len() {
            return Err(length_mismatch_error(self.x.len(), y.len()))
        }
        let zipped = iter::zip(self.x, y);
        let val = if !self.flip {
            zipped
                .map(|(x, y)| Op::op(x, y))
                .collect::<Res<Vec<_>>>()?.to_val()
        } else {
            zipped
                .map(|(x, y)| Op::op(y, x))
                .collect::<Res<Vec<_>>>()?.to_val()
        };
        Ok(val)
    }
}

impl<X: Atom, Op: AtomOp2Fixed<X>> MultiValConsumer for XAtomVec<Op, X> {
    type MultiValRet = Res<Val>;
    fn eat_val_slice(self, a: &[Val]) -> Self::MultiValRet {
        if self.x.len() != a.len() {
            return Err(length_mismatch_error(self.x.len(), a.len()))
        }
        let flip = self.flip;
        let outs = iter::zip(self.x, a)
            .map(|(x, y)| y.dispatch(XAtom::<Op, X>::new(x, flip)))
            .collect::<Res<Vec<_>>>()?;
        Ok(outs.to_val())
    }
    fn eat_val_vec(self, a: Vec<Val>) -> Self::MultiValRet {
        if self.x.len() != a.len() {
            return Err(length_mismatch_error(self.x.len(), a.len()))
        }
        let flip = self.flip;
        let outs = iter::zip(self.x, a)
            .map(|(x, y)| y.dispatch(XAtom::<Op, X>::new(x, flip)))
            .collect::<Res<Vec<_>>>()?;
        Ok(outs.to_val())
    }
}

// x is &[val]
struct XValSlice<'a, Op> { flip: bool, x: &'a [Val], _dummy: PhantomData<Op> }
impl<'a, Op> XValSlice<'a, Op> {
    #[inline(always)]
    fn new(x: &'a [Val], flip: bool) -> Self {
        Self { flip, x, _dummy: PhantomData::<Op> }
    }
}

impl<'a, Op: AtomOp2Fixed<Y>, Y: Atom> AtomConsumer<Y> for XValSlice<'a, Op> {
    type AtomRet = Res<Val>;
    fn eat_atom(self, y: Y) -> Self::AtomRet {
        let y_atom = XAtom::<Op, Y>::new(y, !self.flip);
        let outs = self.x.iter()
            .map(|x| x.dispatch(y_atom))
            .collect::<Res<Vec<_>>>()?;
        Ok(outs.to_val())
    }
    fn eat_atom_slice(self, y: &[Y]) -> Self::AtomRet {
        if self.x.len() != y.len() {
            return Err(length_mismatch_error(self.x.len(), y.len()))
        }
        let outs = iter::zip(self.x, y)
            .map(|(x, y)| x.dispatch(XAtom::<Op, Y>::new(*y, !self.flip)))
            .collect::<Res<Vec<_>>>()?;
        Ok(outs.to_val())
    }
    fn eat_atom_vec(self, y: Vec<Y>) -> Self::AtomRet {
        if self.x.len() != y.len() {
            return Err(length_mismatch_error(self.x.len(), y.len()))
        }
        let outs = iter::zip(self.x, y)
            .map(|(x, y)| x.dispatch(XAtom::<Op, Y>::new(y, !self.flip)))
            .collect::<Res<Vec<_>>>()?;
        Ok(outs.to_val())
    }
}

impl<'a, Op: AtomOp2> MultiValConsumer for XValSlice<'a, Op> {
    type MultiValRet = Res<Val>;
    #[inline(always)]
    fn eat_val_slice(self, a: &[Val]) -> Self::MultiValRet {
        if self.x.len() != a.len() {
            return Err(length_mismatch_error(self.x.len(), a.len()))
        }
        let zipped = iter::zip(self.x, a);
        let val = if !self.flip {
            zipped.map(|(x, y)| dispatch_to_atoms::<Op, _, _>(x, y)).collect::<Res<Vec<_>>>()
        } else {
            zipped.map(|(x, y)| dispatch_to_atoms::<Op, _, _>(y, x)).collect::<Res<Vec<_>>>()
        }?.to_val();
        Ok(val)
    }
    #[inline(always)]
    fn eat_val_vec(self, a: Vec<Val>) -> Self::MultiValRet {
        if self.x.len() != a.len() {
            return Err(length_mismatch_error(self.x.len(), a.len()))
        }
        let zipped = iter::zip(self.x, a);
        let val = if !self.flip {
            zipped.map(|(x, y)| dispatch_to_atoms::<Op, _, _>(x, y)).collect::<Res<Vec<_>>>()
        } else {
            zipped.map(|(x, y)| dispatch_to_atoms::<Op, _, _>(y, x)).collect::<Res<Vec<_>>>()
        }?.to_val();
        Ok(val)
    }
}

// x is vec<val>
struct XValVec<Op> { flip: bool, x: Vec<Val>, _dummy: PhantomData<Op> }
impl<Op> XValVec<Op> {
    #[inline(always)]
    fn new(x: Vec<Val>, flip: bool) -> Self {
        Self { flip, x, _dummy: PhantomData::<Op> }
    }
}

impl<Op: AtomOp2Fixed<Y>, Y: Atom> AtomConsumer<Y> for XValVec<Op> {
    type AtomRet = Res<Val>;
    #[inline(always)]
    fn eat_atom(self, y: Y) -> Self::AtomRet {
        let x_atom = XAtom::<Op, Y>::new(y, !self.flip);
        let val = self.x.into_iter()
            .map(|y| y.dispatch(x_atom))
            .collect::<Res<Vec<_>>>()?.to_val();
        Ok(val)
    }
    #[inline(always)]
    fn eat_atom_slice(self, y: &[Y]) -> Self::AtomRet {
        if self.x.len() != y.len() {
            return Err(length_mismatch_error(self.x.len(), y.len()))
        }
        let flip = !self.flip;
        let val = iter::zip(self.x, y)
            .map(|(x, y)| x.dispatch(XAtom::<Op, Y>::new(*y, flip)))
            .collect::<Res<Vec<_>>>()?.to_val();
        Ok(val)
    }
    #[inline(always)]
    fn eat_atom_vec(self, y: Vec<Y>) -> Self::AtomRet {
        if self.x.len() != y.len() {
            return Err(length_mismatch_error(self.x.len(), y.len()))
        }
        let flip = !self.flip;
        let val = iter::zip(self.x, y)
            .map(|(x, y)| x.dispatch(XAtom::<Op, Y>::new(y, flip)))
            .collect::<Res<Vec<_>>>()?.to_val();
        Ok(val)
    }
}

impl<Op: AtomOp2> MultiValConsumer for XValVec<Op> {
    type MultiValRet = Res<Val>;
    #[inline(always)]
    fn eat_val_slice(self, a: &[Val]) -> Self::MultiValRet {
        if self.x.len() != a.len() {
            return Err(length_mismatch_error(self.x.len(), a.len()))
        }
        let zipped = iter::zip(self.x, a);
        let val = if !self.flip {
            zipped.map(|(x, y)| dispatch_to_atoms::<Op, _, _>(x, y)).collect::<Res<Vec<_>>>()
        } else {
            zipped.map(|(x, y)| dispatch_to_atoms::<Op, _, _>(y, x)).collect::<Res<Vec<_>>>()
        }?.to_val();
        Ok(val)
    }
    #[inline(always)]
    fn eat_val_vec(self, a: Vec<Val>) -> Self::MultiValRet {
        if self.x.len() != a.len() {
            return Err(length_mismatch_error(self.x.len(), a.len()))
        }
        let zipped = iter::zip(self.x, a);
        let val = if !self.flip {
            zipped.map(|(x, y)| dispatch_to_atoms::<Op, _, _>(x, y)).collect::<Res<Vec<_>>>()
        } else {
            zipped.map(|(x, y)| dispatch_to_atoms::<Op, _, _>(y, x)).collect::<Res<Vec<_>>>()
        }?.to_val();
        Ok(val)
    }
}
