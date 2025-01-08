use std::cmp;
use std::rc::Rc;

use crate::ops::{self, IsVal, ToVal, Op2Val, AtomOp2, dispatch_to_atoms};
use crate::val::*;
use crate::util::{cold, err, cold_err, float_as_int, offset_by, Empty, Res};

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

pub fn subtract<X: IsVal + std::fmt::Debug, Y: IsVal + std::fmt::Debug>(x: X, y: Y) -> Res<Val> {
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

pub fn or<X: IsVal, Y: IsVal>(x: X, y: Y) -> Res<Val> {
    use Val::*;
    match (x.to_val_ref(), y.to_val_ref()) {
        (Int(x), Int(y)) => ops::Or::op2val(*x, *y),
        (Int(x), Float(y)) => ops::Or::op2val(*x, *y),
        (Float(x), Int(y)) => ops::Or::op2val(*x, *y),
        (Float(x), Float(y)) => ops::Or::op2val(*x, *y),
        _ => dispatch_to_atoms(AtomOp2::Or, x, y),
    }
}

pub fn not(x: Val) -> Res<Val> {
    use Val::*;
    Ok(match x {
        Int(x) => Int(1 - x),
        Float(x) => Float(1.0 - x),
        I64s(x) => I64s(Rc::new(match Rc::try_unwrap(x) {
            Ok(x) => x.into_iter().map(|x| 1 - x).collect(),
            Err(x) => x.iter().map(|x| 1 - x).collect(),
        })),
        F64s(x) => F64s(Rc::new(match Rc::try_unwrap(x) {
            Ok(x) => x.into_iter().map(|x| 1.0 - x).collect(),
            Err(x) => x.iter().map(|x| 1.0 - x).collect(),
        })),
        Vals(x) => Vals(Rc::new(match Rc::try_unwrap(x) {
            Ok(x) => x.into_iter().map(not).collect::<Res<_>>()?,
            Err(x) => x.iter().cloned().map(not).collect::<Res<_>>()?,
        })),
        x => return cold_err!("domain\nExpected a numeric value, got {}", x.type_name()),
    })
}

pub fn negate(x: Val) -> Res<Val> {
    use Val::*;
    Ok(match x {
        Int(x) => Int(-x),
        Float(x) => Float(-x),
        I64s(x) => match Rc::try_unwrap(x) {
            Ok(x) => I64s(Rc::new(x.into_iter().map(|x| -x).collect())),
            Err(x) => I64s(Rc::new(x.as_slice().iter().map(|x| -*x).collect())),
        }
        F64s(x) => match Rc::try_unwrap(x) {
            Ok(x) => F64s(Rc::new(x.into_iter().map(|x| -x).collect())),
            Err(x) => F64s(Rc::new(x.as_slice().iter().map(|x| -*x).collect())),
        }
        Vals(x) => match Rc::try_unwrap(x) {
            Ok(x) => Vals(Rc::new(x.into_iter().map(negate).collect::<Result<_, _>>()?)),
            Err(x) => Vals(Rc::new(x.as_slice().iter().cloned().map(negate).collect::<Result<_, _>>()?)),
        }
        x => return cold_err!("domain\nExpected a numeric value, got {}", x.type_name()),
    })
}

pub fn abs(x: Val) -> Res<Val> {
    use Val::*;
    Ok(match x {
        Int(x) => Int(x.abs()),
        Float(x) => Float(x.abs()),
        I64s(x) => match Rc::try_unwrap(x) {
            Ok(x) => I64s(Rc::new(x.into_iter().map(|x| x.abs()).collect())),
            Err(x) => I64s(Rc::new(x.as_slice().iter().map(|x| x.abs()).collect())),
        }
        F64s(x) => match Rc::try_unwrap(x) {
            Ok(x) => F64s(Rc::new(x.into_iter().map(|x| x.abs()).collect())),
            Err(x) => F64s(Rc::new(x.as_slice().iter().map(|x| x.abs()).collect())),
        }
        Vals(x) => match Rc::try_unwrap(x) {
            Ok(x) => Vals(Rc::new(x.into_iter().map(abs).collect::<Result<_, _>>()?)),
            Err(x) => Vals(Rc::new(x.as_slice().iter().cloned().map(abs).collect::<Result<_, _>>()?)),
        }
        x => return cold_err!("domain\nExpected a numeric value, got {}", x.type_name()),
    })
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
        (atom!(), _) => if x == y { 0 } else { 1 }
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

pub fn windows(x: &Val, y: &Val) -> Res<Val> {
    let mut vals = vec![];

    let x_len = x.len().unwrap_or(1);
    if x_len == 0 { return Ok(vals.to_val()); }

    let single_aux = &mut [0];
    let mut aux: Vec<i64>  = vec![];
    let lengths = prep_lengths_for_windows_or_chunks(y, single_aux, &mut aux)?;

    vals.reserve(lengths.len());
    let mut i = 0;

    // TODO this is way better with slices
    for len in lengths.iter().copied().map(|len| len as usize).cycle() {
        vals.push(slice_val(x, i, len));
        if i + len >= x_len { break }
        i += 1;
    }
    Ok(vals.to_val())
}

pub fn chunks(x: &Val, y: &Val) -> Res<Val> {
    let single_aux = &mut [0];
    let mut aux: Vec<i64>  = vec![];
    let lengths = prep_lengths_for_windows_or_chunks(y, single_aux, &mut aux)?;

    // TODO this is way better with slices
    let mut vals = vec![];
    let x_len = x.len().unwrap_or(1);
    let mut i = 0;
    for len in lengths.iter().copied().map(|len| len as usize).cycle() {
        if i >= x_len { break }
        vals.push(slice_val(x, i, len));
        i += len;
    }
    Ok(vals.to_val())
}

// May need auxiliary storage if `y` has a length of 1 or consists of floats that need conversion.
fn prep_lengths_for_windows_or_chunks<'a>(
    y: &'a Val, one_aux: &'a mut [i64; 1], many_aux: &'a mut Vec<i64>
) -> Res<&'a [i64]> {
    use Val::*;

    fn convert_float(f: f64) -> Res<i64> {
        float_as_int(f).ok_or_else(
            || cold(format!("domain\nExpected integer, got {f}\n(an int-convertible float would've worked)"))
        )
    }

    let lengths = match y {
        Int(y) => {
            one_aux[0] = *y;
            one_aux
        }
        I64s(y) => y.as_slice(),
        Float(y) => {
            one_aux[0] = convert_float(*y)?;
            one_aux
        }
        F64s(y) => {
            if y.as_slice().len() == 1 {
                one_aux[0] = convert_float(y.as_slice()[0])?;
                one_aux
            } else {
                *many_aux = y.as_slice().iter().copied().map(convert_float).collect::<Res<_>>()?;
                &many_aux[..]
            }
        }
        _ => return cold_err!("domain\nExpected integer, got {y:?}"),
    };
    
    if lengths.is_empty() {
        return cold_err!("length\nExpected non-empty y");
    }

    Ok(lengths)
}

fn slice_val(x: &Val, start: usize, count: usize) -> Val {
    fn slice_to_iter<'a, A: Clone>(xs: &'a [A], start: usize, count: usize) -> impl Iterator<Item=A> + 'a {
        xs[start .. cmp::min(xs.len(), start + count)].iter().cloned()
    }
    fn slice<A: Clone>(xs: &[A], start: usize, count: usize) -> Rc<Vec<A>> {
        Rc::new(xs[start .. cmp::min(xs.len(), start + count)].to_vec())
    }
    match x {
        atom!() => x.clone(),
        Val::U8s(cs) => Val::U8s(slice(cs.as_slice(), start, count)),
        Val::I64s(is) => Val::I64s(slice(is.as_slice(), start, count)),
        Val::F64s(fs) => Val::F64s(slice(fs.as_slice(), start, count)),
        Val::Vals(vals) => collect_list(slice_to_iter(vals.as_slice(), start, count).map(Ok::<Val, Empty>)).unwrap(),
    }
}

pub fn remove(x: Val, y: &Val) -> Val {
    collect_list(iter_val(x).filter(|x| x != y).map(Ok::<Val, Empty>)).unwrap()
}
    
pub fn parse_int(x: &Val) -> Res<Val> {
    use Val::*;
    match x {
        &Char(c) => if c >= '0' as u8 && c <= '9' as u8 {
            Ok(Int((c - '0' as u8) as i64))
        } else {
            cold_err!("failed parse; {c:?} isn't a valid integer")
        }
        U8s(cs) => {
            let s = std::str::from_utf8(cs.as_ref()).map_err(|e| e.to_string())?;
            match s.parse::<i64>() {
                Ok(i) => Ok(Int(i)),
                Err(e) => cold_err!("failed parse; {s:?} isn't a valid integer ({e})"),
            }
        }
        Vals(vals) => collect_list(vals.iter().map(parse_int)),
        _ => cold_err!("failed parse; expected string, got {}", x.type_name()),
    }
}

pub fn parse_float(x: &Val) -> Res<Val> {
    use Val::*;
    match x {
        &Char(c) => if c >= '0' as u8 && c <= '9' as u8 {
            Ok(Float((c - '0' as u8) as f64))
        } else {
            cold_err!("failed parse; {c:?} isn't a valid float")
        }
        U8s(cs) => {
            let s = std::str::from_utf8(cs.as_ref()).map_err(|e| e.to_string())?;
            match s.parse::<f64>() {
                Ok(i) => Ok(Float(i)),
                Err(e) => cold_err!("failed parse; {s:?} isn't a valid float ({e})"),
            }
        }
        Vals(vals) => collect_list(vals.iter().map(parse_float)),
        _ => cold_err!("failed parse; expected string, got {}", x.type_name()),
    }
}

pub fn amend(x: Val, i: Val, new_val: Val) -> Res<Val> {
    #[cold]
    fn oob(i: i64, len: usize) -> String {
        format!("index out of bounds\nRequested index {i}, but length is {len}")
    }
    fn to_index(i: i64, len: usize) -> usize {
        if i >= 0 { i as usize } else { offset_by(len, i) }  // TODO overflow?
    }
    fn update_in_place<A>(slice: &mut [A], i: i64, new: A) -> Res<()> {
        match slice.get_mut(to_index(i, slice.len())) {
            Some(item) => { *item = new; Ok(()) }
            None => Err(oob(i, slice.len())),
        }
    }
    fn amend_atom(i: i64, new_val: Val) -> Res<Val> {
        let len = 1;
        if to_index(i, len) == 0 { Ok(new_val) } else { Err(oob(i, len)) }
    }

    fn amend_rc<A: Clone>(mut rc: Rc<Vec<A>>, i: i64, new: A) -> Res<Val>
    where Rc<Vec<A>>: ToVal, Vec<A>: ToVal {
        match Rc::get_mut(&mut rc) {
            Some(vec_ref) => {
                update_in_place(vec_ref, i, new)?;
                Ok(rc.to_val())
            }
            None => {
                let mut vec: Vec<A> = rc.to_vec();
                update_in_place(vec.as_mut_slice(), i, new)?;
                Ok(vec.to_val())
            }
        }
    }

    use Val::*;
    match (x, zip_vals(i, new_val)) {
        // Err means both i and new_val are atoms.
        (U8s(rc), Err((Int(i), Char(new)))) => amend_rc(rc, i, new),
        (I64s(rc), Err((Int(i), Int(new)))) => amend_rc(rc, i, new),
        (F64s(rc), Err((Int(i), Float(new)))) => amend_rc(rc, i, new),

        // TODO: we may benefit from squashing here if `val` replaces the only non-uniform item of
        // x, as in 1 "a" 3 (@ 1)<-2
        (Vals(rc), Err((Int(i), val))) => amend_rc(rc, i, val),

        (x, Err((Int(i), val))) => match x.to_vals() {
            Err(_atom) => amend_atom(i, val),
            Ok(mut vec) => {
                // x has a uniform type that `val` doesn't match.
                update_in_place(&mut vec, i, val)?;
                Ok(vec.to_val())
            }
        }

        (_, Err((non_int, _))) => cold_err!("invalid index\nExpected integer, got {non_int:?}"),

        // TODO avoid repeated matches on x.
        (mut x, Ok(zipped)) => {
            for (i, new_val) in zipped? {
                x = amend(x, i, new_val)?;
            }
            Ok(x)
        }
    }
}
