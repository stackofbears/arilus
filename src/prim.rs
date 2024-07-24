use std::rc::Rc;

use crate::ops::*;
use crate::val::*;

type Res<A> = Result<A, String>;

// These should match Val::type_name
pub trait NamedType { const TYPE_NAME: &'static str; }
impl NamedType for u8 { const TYPE_NAME: &'static str = &"char"; }
impl NamedType for i64 { const TYPE_NAME: &'static str = &"int"; }
impl NamedType for f64 { const TYPE_NAME: &'static str = &"float"; }
impl NamedType for &Func { const TYPE_NAME: &'static str = &"function"; }

#[inline(never)]
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

            #[inline(always)]
            fn op($x: $x_ty, $y: $y_ty) -> Res<$out> {
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
            fn op(_: $x_ty, _: $y_ty) -> Res<Self::Out> {
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

enum Add {}
impl AtomOp2 for Add {}
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

    (u8, u8) | (&Func, &Func) | (u8, &Func) | (&Func, u8) |
    (f64, &Func) | (&Func, f64) | (i64, &Func) | (&Func, i64) -> domain_error()
);

pub fn add<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    dispatch_to_atoms::<Add, X, Y>(x, y)
}

pub fn subtract<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    enum Sub {}
    impl AtomOp2 for Sub {}

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

        (i64, u8) | (f64, u8) | (&Func, &Func) | (u8, &Func) | (&Func, u8) |
        (f64, &Func) | (&Func, f64) | (i64, &Func) | (&Func, i64) -> domain_error()
    );

    dispatch_to_atoms::<Sub, X, Y>(x, y)
}

pub fn multiply<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    enum Mul {}
    impl AtomOp2 for Mul {}

    impl_op2!(
        Mul,
        (x: i64, y: i64) -> Res<i64> { Ok(x * y) }
        (x: i64, y: f64) -> Res<f64> { Ok(x as f64 * y) }
        (x: f64, y: i64) -> Res<f64> { Ok(x * y as f64) }
        (x: f64, y: f64) -> Res<f64> { Ok(x * y) }

        (u8, u8) | (u8, i64) | (i64, u8) | (u8, f64) | (f64, u8) |
        (&Func, &Func) | (u8, &Func) | (&Func, u8) | (f64, &Func) |
        (&Func, f64) | (i64, &Func) | (&Func, i64) -> domain_error()
    );

    dispatch_to_atoms::<Mul, X, Y>(x, y)
}

pub fn divide<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    enum Div {}
    impl AtomOp2 for Div {}

    impl_op2!(
        Div,
        (x: i64, y: i64) -> Res<f64> { Ok(x as f64 / y as f64) }
        (x: i64, y: f64) -> Res<f64> { Ok(x as f64 / y) }
        (x: f64, y: i64) -> Res<f64> { Ok(x / y as f64) }
        (x: f64, y: f64) -> Res<f64> { Ok(x / y) }

        (u8, u8) | (u8, i64) | (i64, u8) | (u8, f64) | (f64, u8) |
        (&Func, &Func) | (u8, &Func) | (&Func, u8) | (f64, &Func) |
        (&Func, f64) | (i64, &Func) | (&Func, i64) -> domain_error()
    );

    dispatch_to_atoms::<Div, X, Y>(x, y)
}

pub fn int_divide<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    enum IntDiv {}
    impl AtomOp2 for IntDiv {}

    impl_op2!(
        IntDiv,
        (x: i64, y: i64) -> Res<i64> { Ok(x.div_euclid(y)) }
        (x: i64, y: f64) -> Res<i64> { Ok(x.div_euclid(y as i64)) }
        (x: f64, y: i64) -> Res<i64> { Ok((x.floor() as i64).div_euclid(y)) }
        (x: f64, y: f64) -> Res<i64> { Ok(x.div_euclid(y).floor() as i64) }

        (u8, u8) | (u8, i64) | (i64, u8) | (u8, f64) | (f64, u8) |
        (&Func, &Func) | (u8, &Func) | (&Func, u8) | (f64, &Func) |
        (&Func, f64) | (i64, &Func) | (&Func, i64) -> domain_error()
    );

    dispatch_to_atoms::<IntDiv, X, Y>(x, y)
}

pub fn int_mod<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    enum Mod {}
    impl AtomOp2 for Mod {}

    impl_op2!(
        Mod,
        (x: i64, y: i64) -> Res<i64> { Ok(x.rem_euclid(y)) }
        (x: i64, y: f64) -> Res<f64> { Ok((x as f64).rem_euclid(y)) }
        (x: f64, y: i64) -> Res<f64> { Ok(x.rem_euclid(y as f64)) }
        (x: f64, y: f64) -> Res<f64> { Ok(x.rem_euclid(y)) }

        (u8, u8) | (u8, i64) | (i64, u8) | (u8, f64) | (f64, u8) |
        (&Func, &Func) | (u8, &Func) | (&Func, u8) | (f64, &Func) |
        (&Func, f64) | (i64, &Func) | (&Func, i64) -> domain_error()
    );

    dispatch_to_atoms::<Mod, X, Y>(x, y)
}

pub fn pow<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    enum Pow {}
    impl AtomOp2 for Pow {}

    enum Either<A, B> { Left(A), Right(B) }

    impl ToVal for Either<i64, f64> {
        #[inline(always)]
        fn to_val(self) -> Val {
            match self {
                Either::Left(int) => Val::Int(int),
                Either::Right(float) => Val::Float(float),
            }
        }
    }

    impl VecToVal for Either<i64, f64> {
        #[inline(always)]
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

        (u8, u8) | (u8, i64) | (i64, u8) | (u8, f64) | (f64, u8) |
        (&Func, &Func) | (u8, &Func) | (&Func, u8) | (f64, &Func) |
        (&Func, f64) | (i64, &Func) | (&Func, i64) -> domain_error()
    );

    dispatch_to_atoms::<Pow, X, Y>(x, y)
}

// pub fn sum(x: Val, y: Option<Val>) -> Res<Val> {
//     match x.as_val() {
//         atom!() => return match y {
//             Some(y) => add(y, x),
//             None => Ok(x),
//         },
//         Val::I64s(x) => {
//             let x_sum = x.iter().sum();
//             match y {
//                 None => Ok(Val::new(Val::Int(x_sum))),
//                 Some(y) => match Val::try_unwrap(y) {
//                     Err(y) => dispatch_to_atoms_fix_y::<Add, _>(y.as_val(), &x_sum),
//                     Ok(y) => dispatch_to_atoms_take_x_fix_y::<Add, _>(y, &x_sum),
//                 }
//             }
//         }
//         Val::F64s(x) => {
//             let x_sum = x.iter().sum();
//             match y {
//                 None => Ok(Val::new(Val::Float(x_sum))),
//                 Some(y) => match Val::try_unwrap(y) {
//                     Err(y) => dispatch_to_atoms_fix_y::<Add, _>(y.as_val(), &x_sum),
//                     Ok(y) => dispatch_to_atoms_take_x_fix_y::<Add, _>(y, &x_sum),
//                 }
//             }
//         }
//         Val::Vals(x) => {
//             let (mut seed, start) = match y {
//                 Some(y) => (y, 0),
//                 None => match x.get(0) {
//                     Some(first) => (first.clone(), 1),
//                     None => return err!("Error: fold with no input"),
//                 }
//             };
//             for x_elem in &x[start..] {
//                 seed = dispatch_to_atoms_rc::<Add>(seed, x_elem.clone())?;
//             }
//             Ok(seed)
//         }
//         Val::U8s(x) => {
//             let mut seed = match y {
//                 Some(y) => match x.get(0) {
//                     Some(first) => dispatch_to_atoms_fix_y::<Add, _>(&y, first)?,
//                     None => return Ok(y.clone()),
//                 }
//                 None => match x.get(0) {
//                     Some(first) => Val::new(first.to_val()),
//                     None => return err!("Error: fold with no input"),
//                 }
//             };
//             for c in x {
//                 seed = dispatch_to_atoms_fix_y::<Add, _>(seed.as_val(), c)?;
//             }
//             Ok(seed)
//         }
//     }
// }
