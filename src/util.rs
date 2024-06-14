macro_rules! irrefutable {
    ($e:expr, $p:pat => $body:expr) => {
        match $e { $p => $body, _ => unreachable!() }
    };
}
pub(crate) use irrefutable;

macro_rules! err {
    ($($arg:tt)*) => {
        Err(format!($($arg)*))
    };
}
pub(crate) use err;

pub fn float_as_int(f: f64) -> Option<i64> {
    let trunc = f.trunc();
    if trunc == f { Some(trunc as i64) } else { None }
}
