use std::rc::Rc;

use crate::ops::{self, IsVal, Op2Val, AtomOp2, dispatch_to_atoms};
use crate::val::*;
use crate::util::{cold, err, cold_err, float_as_int};

type Res<A> = Result<A, String>;

pub fn add<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    use Val::*;
    match (x.to_val_ref(), y.to_val_ref()) {
        (Int(x), Char(y)) => ops::Add::op2val(*x, *y),
        (Char(x), Int(y)) => ops::Add::op2val(*x, *y),
        (Int(x), Int(y)) => ops::Add::op2val(*x, *y),
        (Int(x), Float(y)) => ops::Add::op2val(*x, *y),
        (Float(x), Int(y)) => ops::Add::op2val(*x, *y),
        (Float(x), Float(y)) => ops::Add::op2val(*x, *y),
        _ => dispatch_to_atoms(AtomOp2::Add, x, y),
    }
}

pub fn multiply<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    use Val::*;
    match (x.to_val_ref(), y.to_val_ref()) {
        (Int(x), Int(y)) => ops::Mul::op2val(*x, *y),
        (Int(x), Float(y)) => ops::Mul::op2val(*x, *y),
        (Float(x), Int(y)) => ops::Mul::op2val(*x, *y),
        (Float(x), Float(y)) => ops::Mul::op2val(*x, *y),
        _ => dispatch_to_atoms(AtomOp2::Mul, x, y)
    }
}

pub fn subtract<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    use Val::*;
    match (x.to_val_ref(), y.to_val_ref()) {
        (Char(x), Int(y)) => ops::Sub::op2val(*x, *y),
        (Int(x), Int(y)) => ops::Sub::op2val(*x, *y),
        (Int(x), Float(y)) => ops::Sub::op2val(*x, *y),
        (Float(x), Int(y)) => ops::Sub::op2val(*x, *y),
        (Float(x), Float(y)) => ops::Sub::op2val(*x, *y),
        _ => dispatch_to_atoms(AtomOp2::Sub, x, y),
    }
}

pub fn divide<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    use Val::*;
    match (x.to_val_ref(), y.to_val_ref()) {
        (Int(x), Int(y)) => ops::Div::op2val(*x, *y),
        (Int(x), Float(y)) => ops::Div::op2val(*x, *y),
        (Float(x), Int(y)) => ops::Div::op2val(*x, *y),
        (Float(x), Float(y)) => ops::Div::op2val(*x, *y),
        _ => dispatch_to_atoms(AtomOp2::Div, x, y),
    }
}

pub fn int_divide<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    use Val::*;
    match (x.to_val_ref(), y.to_val_ref()) {
        (Int(x), Int(y)) => ops::IntDiv::op2val(*x, *y),
        (Int(x), Float(y)) => ops::IntDiv::op2val(*x, *y),
        (Float(x), Int(y)) => ops::IntDiv::op2val(*x, *y),
        (Float(x), Float(y)) => ops::IntDiv::op2val(*x, *y),
        _ => dispatch_to_atoms(AtomOp2::IntDiv, x, y),
    }
}

pub fn int_mod<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    use Val::*;
    match (x.to_val_ref(), y.to_val_ref()) {
        (Int(x), Int(y)) => ops::Mod::op2val(*x, *y),
        (Int(x), Float(y)) => ops::Mod::op2val(*x, *y),
        (Float(x), Int(y)) => ops::Mod::op2val(*x, *y),
        (Float(x), Float(y)) => ops::Mod::op2val(*x, *y),
        _ => dispatch_to_atoms(AtomOp2::Mod, x, y),
    }
}

pub fn pow<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    use Val::*;
    match (x.to_val_ref(), y.to_val_ref()) {
        (Int(x), Int(y)) => ops::Pow::op2val(*x, *y),
        (Int(x), Float(y)) => ops::Pow::op2val(*x, *y),
        (Float(x), Int(y)) => ops::Pow::op2val(*x, *y),
        (Float(x), Float(y)) => ops::Pow::op2val(*x, *y),
        _ => dispatch_to_atoms(AtomOp2::Pow, x, y),
    }
}

pub fn sum(x: Val, y: Option<Val>) -> Res<Val> {
    match x {
        atom!() => return match y {
            Some(y) => add(y, x),
            None => Ok(x),
        },
        Val::I64s(x) => {
            let x_sum = x.iter().sum();
            match y {
                None => Ok(Val::Int(x_sum)),
                Some(y) => y.dispatch(ops::XAtom { flip: true, op: AtomOp2::Add, x: x_sum }),
            }
        }
        Val::F64s(x) => {
            let x_sum = x.iter().sum();
            match y {
                None => Ok(Val::Float(x_sum)),
                Some(y) => y.dispatch(ops::XAtom { flip: true, op: AtomOp2::Add, x: x_sum }),
            }
        }
        Val::Vals(x) => {
            match y {
                Some(y) => match Rc::try_unwrap(x) {
                    Ok(x) => x.into_iter().try_fold(y, |acc, x| add(acc, x)),
                    Err(x) => x.as_slice().iter().try_fold(y, |acc, x| add(acc, x)),
                }
                None => {
                    let reduced = match Rc::try_unwrap(x) {
                        Ok(x) => x.into_iter().try_reduce(|acc, x| add(acc, x)),
                        Err(x) => {
                            if let Some(seed) = x.as_slice().first() {
                                x.as_slice()[1..].iter().try_fold(seed.clone(), |acc, x| add(acc, x)).map(Some)
                            } else {
                                Ok(None)
                            }
                        }
                    }?;

                    if let Some(ret) = reduced {
                        Ok(ret)
                    } else {
                        cold_err!("Error: fold with no input")
                    }
                }
            }
        }
        Val::U8s(x) => {
            match y {
                Some(y) => match Rc::try_unwrap(x) {
                    Ok(x) => x.into_iter().try_fold(y, |acc, x| add(acc, Val::Char(x))),
                    Err(x) => x.as_slice().iter().try_fold(y, |acc, x| add(acc, Val::Char(*x))),
                }
                None => {
                    let reduced = match Rc::try_unwrap(x) {
                        Ok(x) => x.into_iter().map(Val::Char).try_reduce(|acc, x| add(acc, x)),
                        Err(x) => {
                            if let Some(seed) = x.as_slice().first() {
                                x.as_slice()[1..].iter().copied()
                                    .map(Val::Char)
                                    .try_fold(Val::Char(*seed), |acc, x| add(acc, x))
                                    .map(Some)
                            } else {
                                Ok(None)
                            }
                        }
                    }?;

                    if let Some(ret) = reduced {
                        Ok(ret)
                    } else {
                        cold_err!("Error: fold with no input")
                    }
                }
            }
        }
    }
}

pub fn group_indices(x: Val) -> Res<Val> {
    match x {
        Val::Float(f) => match float_as_int(f) {
            Some(i) => group_indices(Val::Int(i)),
            _ => cold_err!("domain\nExpected non-negative integers, got {f}"),
        }
        Val::Int(i) => {
            if i < 0 { return Ok(Val::I64s(Rc::new(vec![]))) }
            let mut v = Vec::with_capacity(i as usize + 1);
            v.resize(i as usize, Val::I64s(Rc::new(vec![])));
            v.push(Val::I64s(Rc::new(vec![0])));
            Ok(Val::Vals(Rc::new(v)))
        }
        Val::I64s(is) => {
            let result_len = match is.iter().max() {
                None => 0,
                Some(&max) => if max < 0 { 0 } else { (max + 1) as usize },
            };
            let mut vs = vec![vec![]; result_len];
            for (xi, ri) in is.iter().enumerate() {
                if *ri >= 0 { vs[*ri as usize].push(xi as i64) }
            }
            let result = vs.into_iter().map(|v| Val::I64s(Rc::new(v))).collect();
            Ok(Val::Vals(Rc::new(result)))
        }
        Val::F64s(fs) => {
            let is = fs.iter().map(|f| match float_as_int(*f) {
                Some(i) => Ok(i),
                _ => cold_err!("domain\nExpected non-negative integers, got {f}"),
            }).collect::<Res<_>>()?;
            group_indices(Val::I64s(Rc::new(is)))
        }
        Val::Vals(x) => Ok(match Rc::try_unwrap(x) {
            Ok(x) => Val::Vals(Rc::new(x.into_iter().map(group_indices).collect::<Res<_>>()?)),
            Err(x) => Val::Vals(Rc::new(x.iter().cloned().map(group_indices).collect::<Res<_>>()?)),
        }),
        _ => cold_err!("domain\nExpected non-negative integers, got {x:?}"),
    }
}

pub fn has(x: Val, y: Val) -> bool {
    find(&x, &y) != x.len().unwrap_or(1) as i64
}

// Attempts to find the whole of y as an element of x.
// TODO flip argument order?
pub fn find(x: &Val, y: &Val) -> i64 {
    use Val::*;
    match (x, y) {
        (atom!(), _) => if x == y { 0 } else { 1 },
        (U8s(xs), Char(c)) => index_of(&**xs, c),
        (I64s(xs), Int(i)) => index_of(&**xs, i),
        (I64s(xs), Float(f)) =>
            float_as_int(*f).map(|i| index_of(&**xs, &i)).unwrap_or(xs.len() as i64),
        (F64s(xs), Float(f)) => index_of(&**xs, f),
        (F64s(xs), Int(i)) => index_of(&**xs, &(*i as f64)),
        (Vals(xs), _) => index_of(xs.iter().map(|rc_val| rc_val.as_val()), y),
        _ => x.len().unwrap_or(1) as i64,
    }
}

fn index_of<'a, A: 'a + PartialEq, I: IntoIterator<Item=&'a A, IntoIter: ExactSizeIterator>>(x: I, y: &A) -> i64 {
    let mut iter = x.into_iter();
    let len = iter.len();
    iter.position(|x| x == y).unwrap_or(len) as i64
}
