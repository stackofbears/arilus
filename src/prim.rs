use crate::ops::*;
use crate::val::*;

type Res<A> = Result<A, String>;

pub fn add(x: &Val, y: &Val) -> Res<RcVal> {
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

    dispatch_to_atoms::<Add>(x, y)
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
