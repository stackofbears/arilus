use crate::ops::*;
use crate::val::*;

type Res<A> = Result<A, String>;

pub fn add<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    dispatch_to_atoms(AtomOp2::Add, x, y)
}

pub fn multiply<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    dispatch_to_atoms(AtomOp2::Mul, x, y)
}

pub fn subtract<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    dispatch_to_atoms(AtomOp2::Sub, x, y)
}

pub fn divide<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    dispatch_to_atoms(AtomOp2::Div, x, y)
}

pub fn int_divide<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    dispatch_to_atoms(AtomOp2::IntDiv, x, y)
}

pub fn int_mod<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    dispatch_to_atoms(AtomOp2::Mod, x, y)
}

pub fn pow<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    dispatch_to_atoms(AtomOp2::Pow, x, y)
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
