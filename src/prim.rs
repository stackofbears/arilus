use std::rc::Rc;

use crate::ops::{self, IsVal, Op2Val, AtomOp2, dispatch_to_atoms};
use crate::val::*;
use crate::util::{cold, err, cold_err};

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
