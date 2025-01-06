// Microbenchmarks to help eyeball the effects of interpreter changes.

extern crate criterion;

use criterion::*;
use arilus::{compile::Compiler, vm::Mem};

macro_rules! bench {
    ($name:ident, $prep:literal, $code:literal) => {
        fn $name(c: &mut Criterion) {
            let mut compiler = Compiler::new();
            compiler.compile_string($prep).unwrap();
            let code_start = compiler.code.len();

            let mut mem = Mem::new();
            std::mem::swap(&mut mem.code, &mut compiler.code);
            mem.execute_from_toplevel(0).unwrap();

            std::mem::swap(&mut mem.code, &mut compiler.code);
            compiler.compile_string($code).unwrap();
            
            std::mem::swap(&mut mem.code, &mut compiler.code);
            c.bench_function(stringify!($name), |b| b.iter(
                || black_box(mem.execute_from_toplevel(code_start)).unwrap()
            ));
        }
    };

    ($name:ident, $code:literal) => {
        bench!($name, "", $code);
    };
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
"#,
r#"
    1e3 Tri
"#);

bench!(naive_triangular_cases,
r#"
    Tri: {[1] 1 : [x] x-1 Rec + x}
"#,
r#"
    1e3 Tri
"#);

bench!(naive_triangular_tail,
r#"
    Tri: {if(x=1; y; x-1 Rec (y+x))}
"#,
r#"
    1e3 Tri 1
"#);

bench!(naive_triangular_cases_tail,
r#"
    Tri: {[x] x Rec 1 : [1;y] y : [x;y] x-1 Rec (y + x)}
"#,
r#"
    1e3 Tri
"#);

criterion_group!(linear_ops, int_sum, transpose, transpose_copies);

bench!(int_sum, r"1e5/ \+");

bench!(transpose,
r#"
    Transpose: {x@0#/ `{[i]x`@i}}
    a: 1e3#100 '/
"#,
r#"
    a Transpose
"#);

bench!(transpose_copies,
r#"
    Transpose: {x@0#/ `{[i]x`@i}}
    \ Rows share the same underlying array (reduced memory pressure)
    a: [1e3/]#100
"#,
r#"
    a Transpose
"#);


criterion_group!(misc, spectral_norm, spectral_a_explicit, spectral_a_tacit, binary_trees, sum_ravel, sum_rows_cols, sum_cols_rows, split_group_indices, split_runs, times_if, times_match, times_fold);

bench!(split_group_indices,
r#"
    Split: {
      grouped: x[x=y \:+ @]
      grouped#1 , (grouped$1`$1) -: []
    }
    a: 1e3/ % 8
"#,
r#"
    a Split 0
"#);

bench!(split_runs,
r#"
    Split: { x Runs[#:] (x`==y!) -: [] }
    a:1e3/ % 8
"#,
r#"
    a Split 0
"#);

bench!(times_if,
r#"
  Times: {[n;f]
    Loop: { if(y=0; x; x F Rec (y-1)) }
    {x Loop n}
  }
"#,
r#"
  0 Times[1e4; {x+1}]
"#);

bench!(times_match,
r#"
  Times: {[n;f]
    Loop: {[x; 0] x : [x; y] x F Rec (y-1) }
    {x Loop n}
  }
"#,
r#"
  0 Times[1e4; {x+1}]
"#);

bench!(times_fold,
r#"
  Times: {[n;f]
    {c0#n \{[x;_] x F} x}
  }
"#,
r#"
  0 Times[1e4; {x+1}]
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
"#,
r#"
    100 Approximate
"#);

bench!(spectral_a_explicit,
r#"
    A: { x+y*(x+y+1)/2+x+1 ~/ 1.0 }
    x: 1000/
"#,
r#"
    x 'A x
"#);

bench!(spectral_a_tacit,
r#"
    A: +*(++1)/2+P+1~/1.0
    x: 1000/
"#,
r#"
    x 'A x
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
"#,
r#"
    2 Main 4
"#);

bench!(sum_ravel, r"[1e3/]#100 , \+");

bench!(sum_rows_cols, r"[1e3/]#100 \+ \+");

bench!(sum_cols_rows, r"[1e3/]#100 '\+ \+");
