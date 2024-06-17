use crate::ops::*;
use crate::val::*;
use crate::util::err;

type Res<A> = Result<A, String>;

// These should match Val::type_name
pub trait NamedType { const TYPE_NAME: &'static str; }
impl NamedType for u8 { const TYPE_NAME: &'static str = &"char"; }
impl NamedType for i64 { const TYPE_NAME: &'static str = &"int"; }
impl NamedType for f64 { const TYPE_NAME: &'static str = &"float"; }
impl NamedType for Func { const TYPE_NAME: &'static str = &"function"; }

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

enum Add {}
impl AtomOp2 for Add {}
impl_op2!(
    Add,
    (x: &u8, y: &i64) -> Res<u8> { Ok((*x as i64 + *y) as u8) }
    (x: &u8, y: &f64) -> Res<u8> {
        if y.trunc() == *y {
            Ok((*x as i64 + *y as i64) as u8)
        } else {
            Err(domain_error::<u8, f64>(&"(An int-convertible float would've worked)"))
        }
    }
    (x: &i64, y: &u8)  -> Res<u8>  { Self::op(y, x) }
    (x: &i64, y: &i64) -> Res<i64> { Ok(x + y) }
    (x: &i64, y: &f64) -> Res<f64> { Ok(*x as f64 + *y) }
    (x: &f64, y: &u8)  -> Res<u8>  { Self::op(y, x) }
    (x: &f64, y: &i64) -> Res<f64> { Self::op(y, x) }
    (x: &f64, y: &f64) -> Res<f64> { Ok(x + y) }

    (u8, u8) | (Func, Func) | (u8, Func) | (Func, u8) |
    (f64, Func) | (Func, f64) | (i64, Func) | (Func, i64) -> domain_error()
);

pub fn add(x: RcVal, y: RcVal) -> Res<RcVal> {
    // match (x.as_val(), y.as_val()) {
    //     (Val::Int(x), Val::Int(y)) => return Ok(RcVal::new(Val::Int(*x + *y))),
    //     _ => {}
    // }
    // dbg!(RcVal::strong_count(&x));
    // if let Some(Val::I64s(x_mut)) = RcVal::get_mut(&mut x) {
    //     println!("Unique x");
    //     match y {
    //         Val::Int(y) => {
    //             println!("Special path ints+int!");
    //             for i in 0..x_mut.len() { x_mut[i] += y }
    //             return Ok(x)
    //         }
    //         Val::I64s(ys) if ys.len() == x_mut.len() => {
    //             println!("Special path ints+ints!");
    //             for i in 0..x_mut.len() { x_mut[i] += ys[i] }
    //             return Ok(x)
    //         }
    //         _ => {}
    //     }
    // }
    dispatch_to_atoms_rc::<Add>(x, y)
}

pub fn subtract(x: &Val, y: &Val) -> Res<RcVal> {
    enum Sub {}
    impl AtomOp2 for Sub {}

    impl_op2!(
        Sub,
        (x: &u8, y: &u8) -> Res<i64> { Ok(*x as i64 - *y as i64) }
        (x: &u8, y: &i64) -> Res<u8> { Ok((*x as i64 - *y) as u8) }
        (x: &u8, y: &f64) -> Res<u8> {
            if y.trunc() == *y {
                Ok((*x as i64 - *y as i64) as u8)
            } else {
                Err(domain_error::<u8, f64>(&"(An int-convertible float would've worked)"))
            }
        }
        (x: &i64, y: &i64) -> Res<i64> { Ok(x - y) }
        (x: &i64, y: &f64) -> Res<f64> { Ok(*x as f64 - *y) }
        (x: &f64, y: &i64) -> Res<f64> { Ok(*x - *y as f64) }
        (x: &f64, y: &f64) -> Res<f64> { Ok(x - y) }

        (i64, u8) | (f64, u8) | (Func, Func) | (u8, Func) | (Func, u8) |
        (f64, Func) | (Func, f64) | (i64, Func) | (Func, i64) -> domain_error()
    );

    dispatch_to_atoms::<Sub>(x, y)
}

pub fn multiply(x: &Val, y: &Val) -> Res<RcVal> {
    enum Mul {}
    impl AtomOp2 for Mul {}

    impl_op2!(
        Mul,
        (x: &i64, y: &i64) -> Res<i64> { Ok(x * y) }
        (x: &i64, y: &f64) -> Res<f64> { Ok(*x as f64 * *y) }
        (x: &f64, y: &i64) -> Res<f64> { Ok(*x * *y as f64) }
        (x: &f64, y: &f64) -> Res<f64> { Ok(x * y) }

        (u8, u8) | (u8, i64) | (i64, u8) | (u8, f64) | (f64, u8) |
        (Func, Func) | (u8, Func) | (Func, u8) | (f64, Func) |
        (Func, f64) | (i64, Func) | (Func, i64) -> domain_error()
    );

    dispatch_to_atoms::<Mul>(x, y)
}

pub fn divide(x: &Val, y: &Val) -> Res<RcVal> {
    enum Div {}
    impl AtomOp2 for Div {}

    impl_op2!(
        Div,
        (x: &i64, y: &i64) -> Res<f64> { Ok(*x as f64 / *y as f64) }
        (x: &i64, y: &f64) -> Res<f64> { Ok(*x as f64 / *y) }
        (x: &f64, y: &i64) -> Res<f64> { Ok(*x / *y as f64) }
        (x: &f64, y: &f64) -> Res<f64> { Ok(x / y) }

        (u8, u8) | (u8, i64) | (i64, u8) | (u8, f64) | (f64, u8) |
        (Func, Func) | (u8, Func) | (Func, u8) | (f64, Func) |
        (Func, f64) | (i64, Func) | (Func, i64) -> domain_error()
    );

    dispatch_to_atoms::<Div>(x, y)
}

pub fn int_divide(x: &Val, y: &Val) -> Res<RcVal> {
    enum IntDiv {}
    impl AtomOp2 for IntDiv {}

    impl_op2!(
        IntDiv,
        (x: &i64, y: &i64) -> Res<i64> { Ok(x.div_euclid(*y)) }
        (x: &i64, y: &f64) -> Res<i64> { Ok(x.div_euclid(*y as i64)) }
        (x: &f64, y: &i64) -> Res<i64> { Ok((x.floor() as i64).div_euclid(*y)) }
        (x: &f64, y: &f64) -> Res<i64> { Ok(x.div_euclid(*y).floor() as i64) }

        (u8, u8) | (u8, i64) | (i64, u8) | (u8, f64) | (f64, u8) |
        (Func, Func) | (u8, Func) | (Func, u8) | (f64, Func) |
        (Func, f64) | (i64, Func) | (Func, i64) -> domain_error()
    );

    dispatch_to_atoms::<IntDiv>(x, y)
}

pub fn int_mod(x: &Val, y: &Val) -> Res<RcVal> {
    enum Mod {}
    impl AtomOp2 for Mod {}

    impl_op2!(
        Mod,
        (x: &i64, y: &i64) -> Res<i64> { Ok(x.rem_euclid(*y)) }
        (x: &i64, y: &f64) -> Res<i64> { Ok(x.rem_euclid(*y as i64)) }
        (x: &f64, y: &i64) -> Res<i64> { Ok((x.floor() as i64).rem_euclid(*y)) }
        (x: &f64, y: &f64) -> Res<i64> { Ok(x.rem_euclid(*y).floor() as i64) }

        (u8, u8) | (u8, i64) | (i64, u8) | (u8, f64) | (f64, u8) |
        (Func, Func) | (u8, Func) | (Func, u8) | (f64, Func) |
        (Func, f64) | (i64, Func) | (Func, i64) -> domain_error()
    );

    dispatch_to_atoms::<Mod>(x, y)
}

pub fn pow(x: &Val, y: &Val) -> Res<RcVal> {
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
                        return Val::F64s(floats);
                    }
                }
            }
            Val::I64s(ints)
        }
    }

    impl_op2!(
        Pow,
        (x: &i64, y: &i64) -> Res<Either<i64, f64>> {
            Ok(if *y >= 0 {
                Either::Left(x.pow(*y as u32)) 
            } else {
                Either::Right((*x as f64).powi(*y as i32))
            })
        }
               
        (x: &i64, y: &f64) -> Res<f64> { Ok((*x as f64).powf(*y)) }
        (x: &f64, y: &i64) -> Res<f64> { Ok(x.powi(*y as i32)) }
        (x: &f64, y: &f64) -> Res<f64> { Ok(x.powf(*y)) }

        (u8, u8) | (u8, i64) | (i64, u8) | (u8, f64) | (f64, u8) |
        (Func, Func) | (u8, Func) | (Func, u8) | (f64, Func) |
        (Func, f64) | (i64, Func) | (Func, i64) -> domain_error()
    );

    dispatch_to_atoms::<Pow>(x, y)
}

pub fn sum(x: RcVal, y: Option<RcVal>) -> Res<RcVal> {
    match x.as_val() {
        atom!() => return match y {
            Some(y) => add(y, x),
            None => Ok(x),
        },
        Val::I64s(x) => {
            let x_sum = x.iter().sum();
            match y {
                None => Ok(RcVal::new(Val::Int(x_sum))),
                Some(y) => match RcVal::try_unwrap(y) {
                    Err(y) => dispatch_to_atoms_fix_y::<Add, _>(y.as_val(), &x_sum),
                    Ok(y) => dispatch_to_atoms_take_x_fix_y::<Add, _>(y, &x_sum),
                }
            }
        }
        Val::F64s(x) => {
            let x_sum = x.iter().sum();
            match y {
                None => Ok(RcVal::new(Val::Float(x_sum))),
                Some(y) => match RcVal::try_unwrap(y) {
                    Err(y) => dispatch_to_atoms_fix_y::<Add, _>(y.as_val(), &x_sum),
                    Ok(y) => dispatch_to_atoms_take_x_fix_y::<Add, _>(y, &x_sum),
                }
            }
        }
        Val::Vals(x) => {
            let (mut seed, start) = match y {
                Some(y) => (y, 0),
                None => match x.get(0) {
                    Some(first) => (first.clone(), 1),
                    None => return err!("Error: fold with no input"),
                }
            };
            for x_elem in &x[start..] {
                seed = dispatch_to_atoms_rc::<Add>(seed, x_elem.clone())?;
            }
            Ok(seed)
        }
        Val::U8s(x) => {
            let mut seed = match y {
                Some(y) => match x.get(0) {
                    Some(first) => dispatch_to_atoms_fix_y::<Add, _>(&y, first)?,
                    None => return Ok(y.clone()),
                }
                None => match x.get(0) {
                    Some(first) => RcVal::new(first.to_val()),
                    None => return err!("Error: fold with no input"),
                }
            };
            for c in x {
                seed = dispatch_to_atoms_fix_y::<Add, _>(seed.as_val(), c)?;
            }
            Ok(seed)
        }
    }
}
