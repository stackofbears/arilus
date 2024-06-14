use crate::ops::*;
use crate::val::*;

type Res<A> = Result<A, String>;

pub fn add(x: &Val, y: &Val) -> Res<RcVal> {
    enum Add {}
    impl AtomOp for Add {}

    impl_op!(
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
