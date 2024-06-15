use crate::val::*;

type Res<A> = Result<A, String>;

// An operation on two arguments.
pub trait Op2<X, Y> {
    type Out;
    fn op(x: &X, y: &Y) -> Res<Self::Out>;

    // TODO let ops return a type other than Vec<Out> in the traverse case
    // e.g. for Pow<i64, i64>,
    //   - Out(X, Y) = Either<i64, f64> since (y>=0 -> i, y<0 -> f)
    //   - Out([X], Y) = Either<Vec<i64>, Vec<f64>> (y still decides)
    //   - Out(X, [Y]) = Either<Vec<i64>, Vec<f64>> (multiple ys; first go to ints otherwise
    //                                               drain and convert to floats)
    //   - Out([X], [Y]) = Either<Vec<i64>, Vec<f64>> (same deal as above)

    #[inline(always)]
    fn op_traverse_x(xs: &[X], y: &Y) -> Res<Vec<Self::Out>> {
        let mut v = Vec::with_capacity(xs.len());
        for i in 0..xs.len() { v.push(Self::op(&xs[i], y)?) }
        Ok(v)
    }

    #[inline(always)]
    fn op_traverse_y(x: &X, ys: &[Y]) -> Res<Vec<Self::Out>> {
        let mut v = Vec::with_capacity(ys.len());
        for i in 0..ys.len() { v.push(Self::op(x, &ys[i])?); }
        Ok(v)
    }

    #[inline(always)]
    fn op_traverse_zip(xs: &[X], ys: &[Y]) -> Res<Vec<Self::Out>> {
        let xlen = xs.len();
        let ylen = ys.len();
        if xlen != ylen {
            return Err(length_mismatch_error(xlen, ylen))
        }
        let mut v = Vec::with_capacity(xlen);
        for i in 0..xlen {
            v.push(Self::op(&xs[i], &ys[i])?)
        }
        Ok(v)
    }
}

pub trait ToVal: Sized {
    fn to_val(self) -> Val;
    fn to_rc_val(self) -> RcVal { RcVal::new(self.to_val()) }
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
    fn to_val(self) -> Val { Val::Function(self) }
}
impl ToVal for NoValEmptyEnum {
    #[inline(always)]
    // SAFETY Self can't be instantiated, so this is never called.
    fn to_val(self) -> Val { unsafe { std::hint::unreachable_unchecked() } }
}

impl<A: VecToVal> ToVal for Vec<A> {
    #[inline(always)]
    fn to_val(self) -> Val { VecToVal::vec_to_val(self) }
}

// Ideally we would just implement ToVal for Vec<u8>, Vec<i64>, ..., but for
// functions to infer bounds from trait definitions, they need to be directly on
// Self or on an associated type; see
// https://github.com/rust-lang/rust/issues/20671.
//
// Instead, A: VecToVal indicates that Vec<A> can be turned into a Val. This
// way, we can write A::Out: VecToVal and have that information propagated to
// `dispatch_to_atoms`.
pub trait VecToVal: Sized {
    fn vec_to_val(v: Vec<Self>) -> Val;
}

impl VecToVal for u8 {
    #[inline(always)]
    fn vec_to_val(v: Vec<Self>) -> Val { Val::U8s(v) }
}
impl VecToVal for i64 {
    #[inline(always)]
    fn vec_to_val(v: Vec<Self>) -> Val { Val::I64s(v) }
}
impl VecToVal for f64 {
    #[inline(always)]
    fn vec_to_val(v: Vec<Self>) -> Val { Val::F64s(v) }
}
impl VecToVal for RcVal {
    #[inline(always)]
    fn vec_to_val(v: Vec<Self>) -> Val { Val::Vals(v) }
}
impl VecToVal for NoValEmptyEnum {
    // TODO this isn't actually unreachable
    #[inline(always)]
    fn vec_to_val(_: Vec<Self>) -> Val { unreachable!() }
}

pub trait AtomOp2:
  Op2<u8,  u8,  Out: ToVal + VecToVal> + Op2<u8,  i64,  Out: ToVal + VecToVal> +
  Op2<u8,  f64, Out: ToVal + VecToVal> + Op2<u8,  Func, Out: ToVal + VecToVal> +

  Op2<i64, u8,  Out: ToVal + VecToVal> + Op2<i64, i64,  Out: ToVal + VecToVal> +
  Op2<i64, f64, Out: ToVal + VecToVal> + Op2<i64, Func, Out: ToVal + VecToVal> +

  Op2<f64, u8,  Out: ToVal + VecToVal> + Op2<f64, i64,  Out: ToVal + VecToVal> +
  Op2<f64, f64, Out: ToVal + VecToVal> + Op2<f64, Func, Out: ToVal + VecToVal> +

  Op2<Func, u8,  Out: ToVal + VecToVal> + Op2<Func, i64,  Out: ToVal + VecToVal> +
  Op2<Func, f64, Out: ToVal + VecToVal> + Op2<Func, Func, Out: ToVal + VecToVal> {
}

// These should match Val::type_name
pub trait NamedType { const TYPE_NAME: &'static str; }
impl NamedType for u8 { const TYPE_NAME: &'static str = &"char"; }
impl NamedType for i64 { const TYPE_NAME: &'static str = &"int"; }
impl NamedType for f64 { const TYPE_NAME: &'static str = &"float"; }
impl NamedType for Func { const TYPE_NAME: &'static str = &"function"; }

#[cold]
pub fn domain_error<A: NamedType, B: NamedType>(detail: &str) -> String {
    format!("domain\nUnsupported arguments: {} and {}{}{detail}",
            A::TYPE_NAME, B::TYPE_NAME, &if detail.is_empty() { "" } else { "\n" })
}

macro_rules! impl_op2 {
    ($self_type:ty, ($x:ident : &$x_ty:ty, $y:ident : &$y_ty:ty) -> Res<$out:ty> { $($body:tt)* } $($rest:tt)*) => {
        impl Op2<$x_ty, $y_ty> for $self_type {
            type Out = $out;

            #[inline(always)]
            fn op($x: &$x_ty, $y: &$y_ty) -> Res<$out> {
                $($body)*
            }
        }
        impl_op2!($self_type, $($rest)*);
    };

    // $detail can be () or (literal)
    ($self_type:ty, $(($x_ty:ty, $y_ty:ty))|+ -> domain_error $detail:tt $($rest:tt)*) => {
        $(impl Op2<$x_ty, $y_ty> for $self_type {
            type Out = NoValEmptyEnum;

            #[cold]
            fn op(_: &$x_ty, _: &$y_ty) -> Res<Self::Out> {
                macro_rules! optional_detail {
                    (()) => { "" };
                    (($lit:literal)) => { $lit };
                }
                Err(domain_error::<$x_ty, $y_ty>(optional_detail!($detail)))
            }
        })+
        impl_op2!($self_type, $($rest)*);
    };

    ($self_type:ty,) => {};
}
pub(crate) use impl_op2;

#[inline]
pub fn dispatch_to_atoms<A: AtomOp2>(x: &Val, y: &Val) -> Res<RcVal> {
    use Val::*;
    let val = match (x, y) {
        (Char(x), _) => dispatch_to_atoms_fix_x::<A, _>(x, y)?,
        (Int(x), _) => dispatch_to_atoms_fix_x::<A, _>(x, y)?,
        (Float(x), _) => dispatch_to_atoms_fix_x::<A, _>(x, y)?,
        (Function(x), _) => dispatch_to_atoms_fix_x::<A, _>(x, y)?,

        (_, Char(y)) => dispatch_to_atoms_fix_y::<A, _>(x, y)?,
        (_, Int(y)) => dispatch_to_atoms_fix_y::<A, _>(x, y)?,
        (_, Float(y)) => dispatch_to_atoms_fix_y::<A, _>(x, y)?,
        (_, Function(y)) => dispatch_to_atoms_fix_y::<A, _>(x, y)?,

        (U8s(x), U8s(y)) => A::op_traverse_zip(x, y)?.to_rc_val(),
        (U8s(x), I64s(y)) => A::op_traverse_zip(x, y)?.to_rc_val(),
        (U8s(x), F64s(y)) => A::op_traverse_zip(x, y)?.to_rc_val(),
        (U8s(x), Vals(y)) => zip_traverse(x, y, #[inline] |x, y| dispatch_to_atoms_fix_x::<A, _>(x, y.as_val()))?.to_rc_val(),

        (I64s(x), U8s(y)) => A::op_traverse_zip(x, y)?.to_rc_val(),
        (I64s(x), I64s(y)) => A::op_traverse_zip(x, y)?.to_rc_val(),
        (I64s(x), F64s(y)) => A::op_traverse_zip(x, y)?.to_rc_val(),
        (I64s(x), Vals(y)) => zip_traverse(x, y, #[inline] |x, y| dispatch_to_atoms_fix_x::<A, _>(x, y.as_val()))?.to_rc_val(),

        (F64s(x), U8s(y)) => A::op_traverse_zip(x, y)?.to_rc_val(),
        (F64s(x), I64s(y)) => A::op_traverse_zip(x, y)?.to_rc_val(),
        (F64s(x), F64s(y)) => A::op_traverse_zip(x, y)?.to_rc_val(),
        (F64s(x), Vals(y)) => zip_traverse(x, y, #[inline] |x, y| dispatch_to_atoms_fix_x::<A, _>(x, y.as_val()))?.to_rc_val(),

        (Vals(x), U8s(y)) => zip_traverse(x, y, #[inline] |x, y| dispatch_to_atoms_fix_y::<A, _>(x.as_val(), y))?.to_rc_val(),
        (Vals(x), I64s(y)) => zip_traverse(x, y, #[inline] |x, y| dispatch_to_atoms_fix_y::<A, _>(x.as_val(), y))?.to_rc_val(),
        (Vals(x), F64s(y)) => zip_traverse(x, y, #[inline] |x, y| dispatch_to_atoms_fix_y::<A, _>(x.as_val(), y))?.to_rc_val(),
        (Vals(x), Vals(y)) => zip_traverse(x, y, #[inline] |x, y| dispatch_to_atoms::<A>(x.as_val(), y.as_val()))?.to_rc_val(),
    };
    Ok(val)
}

fn dispatch_to_atoms_fix_x<A, X>(x: &X, y: &Val) -> Res<RcVal>
where A: Op2<X, u8, Out: ToVal + VecToVal>,
      A: Op2<X, i64, Out: ToVal + VecToVal>,
      A: Op2<X, f64, Out: ToVal + VecToVal>,
      A: Op2<X, Func, Out: ToVal + VecToVal> {
    use Val::*;
    let val = match y {
        Char(y) => A::op(x, y)?.to_val(),
        Int(y) => A::op(x, y)?.to_val(),
        Float(y) => A::op(x, y)?.to_val(),
        Function(y) => A::op(x, y)?.to_val(),

        U8s(y) => A::op_traverse_y(x, y)?.to_val(),
        I64s(y) => A::op_traverse_y(x, y)?.to_val(),
        F64s(y) => A::op_traverse_y(x, y)?.to_val(),

        Vals(y) => traverse(y, #[inline] |y| dispatch_to_atoms_fix_x::<A, _>(x, y.as_val()))?.to_val(),
    };
    Ok(RcVal::new(val))
}

fn dispatch_to_atoms_fix_y<A, Y>(x: &Val, y: &Y) -> Res<RcVal>
where A: Op2<u8, Y, Out: ToVal + VecToVal>,
      A: Op2<i64, Y, Out: ToVal + VecToVal>,
      A: Op2<f64, Y, Out: ToVal + VecToVal>,
      A: Op2<Func, Y, Out: ToVal + VecToVal> {
    use Val::*;
    let val = match x {
        Char(x) => A::op(x, y)?.to_val(),
        Int(x) => A::op(x, y)?.to_val(),
        Float(x) => A::op(x, y)?.to_val(),
        Function(x) => A::op(x, y)?.to_val(),

        U8s(x) => A::op_traverse_x(x, y)?.to_val(),
        I64s(x) => A::op_traverse_x(x, y)?.to_val(),
        F64s(x) => A::op_traverse_x(x, y)?.to_val(),

        Vals(x) => traverse(x, #[inline] |x| dispatch_to_atoms_fix_y::<A, _>(x.as_val(), y))?.to_val(),
    };
    Ok(RcVal::new(val))
}

#[inline(always)]
fn traverse<A, X, F: Fn(&X) -> Res<A>>(xs: &[X], f: F) -> Res<Vec<A>> {
    let mut v = Vec::with_capacity(xs.len());
    for x in xs { v.push(f(x)?) }
    Ok(v)
}

#[inline(always)]
fn zip_traverse<A, X, Y, F: FnMut(&X, &Y) -> Res<A>>(
    xs: &[X], ys: &[Y], mut f: F
) -> Result<Vec<A>, String> {
    let xlen = xs.len();
    let ylen = ys.len();
    if xlen != ylen { return Err(length_mismatch_error(xlen, ylen)) }
    let mut v = Vec::with_capacity(xlen);
    for i in 0..xlen {
        v.push(f(&xs[i], &ys[i])?)
    }
    Ok(v)
}

#[inline(never)]
fn length_mismatch_error(xlen: usize, ylen: usize) -> String {
    format!("length mismatch: {xlen} vs {ylen}")
}
