#[derive(Debug)]
pub enum Empty {}

macro_rules! irrefutable {
    ($e:expr, $p:pat => $body:expr) => {
        match $e { $p => $body, _ => unreachable!() }
    };
}
pub(crate) use irrefutable;

#[macro_export]
macro_rules! err {
    ($($arg:tt)*) => {
        Err(format!($($arg)*))
    };
}
pub(crate) use err;

macro_rules! cold_err {
    ($($arg:tt)*) => { cold(err!($($arg)*)) };
}
pub(crate) use cold_err;

pub fn float_as_int(f: f64) -> Option<i64> {
    let trunc = f.trunc();
    if trunc == f { Some(trunc as i64) } else { None }
}

#[inline]
#[cold]
pub fn cold<A>(a: A) -> A { a }

#[inline]
#[allow(dead_code)]
pub fn likely(b: bool) -> bool {
    if !b { cold(()) }
    b
}

#[inline]
#[allow(dead_code)]
pub fn unlikely(b: bool) -> bool {
    if b { cold(()) }
    b
}

#[cold]
pub fn length_mismatch_error(xlen: usize, ylen: usize) -> Result<(), String> {
    // TODO include name/position of verb
    cold_err!("length mismatch: {xlen} vs {ylen}")
}

#[inline]
pub fn match_lengths(xlen: usize, ylen: usize) -> Result<(), String> {
    if xlen != ylen { length_mismatch_error(xlen, ylen) }
    else { Ok(()) }
}
