// Microbenchmarks to help eyeball the effects of interpreter changes.

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

// TODO
//  benches!(
//     basic_recursion {
//         naive_triangular {r#"
//           Tri: {if(x=1; 1; x-1 Rec + x)}
//           1e3 Tri
//         "#}
//         naive_triangular_tail {r#"
//           Tri: {if(x=1; y; x-1 Rec (y + x))} 1
//           1e3 Tri
//         "#}
//     }
//     linear_ops {
//     }
//     misc {
//     }
// );

criterion_main!(basic_recursion, linear_ops, misc);

criterion_group!(basic_recursion, naive_triangular, naive_triangular_cases, naive_triangular_tail, naive_triangular_cases_tail);

bench!(naive_triangular,
r#"
    Tri: {if(x=1; 1; x-1 Rec + x)}
    1e3 Tri
"#);

bench!(naive_triangular_cases,
r#"
    Tri: {[1] 1 : [x] x-1 Rec + x}
    1e3 Tri
"#);

bench!(naive_triangular_tail,
r#"
    Tri: {if(x=1; y; x-1 Rec (y+x))}
    1e3 Tri 1
"#);

bench!(naive_triangular_cases_tail,
r#"
    Tri: {[x] x Rec 1 : [1;y] y : [x;y] x-1 Rec (y + x)}
    1e3 Tri
"#);

criterion_group!(linear_ops, int_sum, transpose, transpose_copies);

bench!(int_sum, r"1e5/ \+");

bench!(transpose,
r#"
    Transpose: {x@0#/ `{[i]x`@i}}
    a: 1e3#100 '/
    a Transpose
"#);

bench!(transpose_copies,
r#"
    Transpose: {x@0#/ `{[i]x`@i}}
    \ Rows share the same underlying array (reduced memory pressure)
    a: [1e3/]#100
    a Transpose
"#);


criterion_group!(misc, spectral_norm, spectral_a_explicit, spectral_a_tacit, binary_trees, sum_ravel, sum_rows_cols, sum_cols_rows, split_group_indices, split_runs);

bench!(split_group_indices,
r#"
    Split: {x = y \:+ GroupIndices ~@ x}
    a:1e3/ % 8
    a Split 0
"#);

bench!(split_runs,
r#"
    Split: { x Runs[#:] (x=!y) ->x#:(x'#>0) }
    a:1e3/ % 8
    a Split 0
"#);

// Based on
// https://sschakraborty.github.io/benchmark/spectralnorm-csharpcore-1.html
//
// Expected answer for n=100: 1.2742199912349306
bench!(spectral_norm,
r#"
    A: { x+y*(x+y+1)/2+x+1 ~/ 1.0 }
    Approximate: {[n]
      i: n/
      MultiplyAv: { i'(P A i * x \+) }
      MultiplyAtv: { i'(P ~A i * x \+) }
      MultiplyAtAv: MultiplyAv MultiplyAtv

      v: 19/ \(P MultiplyAtAv) (1#n)
      vBv: v MultiplyAtAv * v \+
      vv: v * v \+
      vBv/vv ^ 0.5
    }

    100 Approximate
"#);

bench!(spectral_a_explicit,
r#"
    A: { x+y*(x+y+1)/2+x+1 ~/ 1.0 }
    1000/ 'A (1000/)
"#);

bench!(spectral_a_tacit,
r#"
    A: +*(++1)/2+P+1~/1.0
    1000/ 'A (1000/)
"#);

// Based on
// https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/binarytrees-gcc-3.html
bench!(binary_trees,
r#"
    zeros: 0 0

    \ Node: 2-item array: each item is 0 or another node
    CreateTree: {[0] zeros : [depth] [depth-1 Rec; depth-1 Rec]}

    ComputeChecksum: {[0 0] 1 : [x] x 'rec \+ 1 }

    Main: {[minNodes; maxNodes]
      stretchTree: maxNodes+1 CreateTree
      longLivedTree: minNodes CreateTree

      maxNodes-minNodes+1/+minNodes'{
        iterations: 2 ^ x
        iterations/+1 '(CreateTree ComputeChecksum) \+
      }
    }

    2 Main 4
"#);

bench!(sum_ravel, r"[1e3/]#100 , \+");

bench!(sum_rows_cols, r"[1e3/]#100 \+ \+");

bench!(sum_cols_rows, r"[1e3/]#100 '\+ \+");
