extern crate criterion;

use criterion::*;

use ltr_rs::compile_string;
use ltr_rs::Mem;

fn mem_with_code(text: &str) -> Mem {
    let code = crate::compile_string(text).unwrap();
    let mut mem = Mem::new();
    mem.code = code;
    mem
}

fn bench_code(c: &mut Criterion, name: &str, text: &str) {
    let mut mem = mem_with_code(text);
    c.bench_function(name, |b| b.iter(|| {
        black_box(mem.execute_from_toplevel(0)).unwrap();
    }));
}

macro_rules! bench {
    ($name:ident, $code:literal) => {
        fn $name(c: &mut Criterion) {
            bench_code(c, stringify!($name), $code);
        }
    }
}

bench!(naive_triangular,
r#"
    Tri: {if(x=1; 1; x-1 Rec + x)}
    1e3 Tri
"#);

bench!(naive_triangular_tail,
r#"
    Tri: {if(x=1; y; x-1 Rec (y + x))} 1
    1e3 Tri
"#);

criterion_group!(basic_recursion, naive_triangular, naive_triangular_tail);

bench!(int_sum,
r#"
    1e5/ \+
"#);

bench!(sum_2d,
r#"
    [100/] # 1e4 \+
"#);

criterion_group!(linear_ops, int_sum, sum_2d);

criterion_main!(basic_recursion, linear_ops);
