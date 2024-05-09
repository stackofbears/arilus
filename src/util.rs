macro_rules! irrefutable {
    ($e:expr, $p:pat => $body:expr) => {
        match $e { $p => $body, _ => unreachable!() }
    };
}
pub(crate) use irrefutable;
