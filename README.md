This is a hobby programming language I'm using to experiment with implementing
array programming concepts. It's inspired by languages in the APL lineage,
primarily J, K, and BQN. It's dynamically typed and generally encourages working
with immutable data. Writing working programs is possible, though it requires
knowledge beyond what you'll learn here.

**Note:** This project isn't yet meant to be accessible to anyone who isn't me!
It's insufficiently documented, it lacks tests, and the language itself isn't
very user-friendly to boot. If you want to figure out what's going on, sadly,
you'll have to read the source code.

TODO: include a full example program

# How to try it

Download the code, enter the root directory, and `cargo run --release` to open
the repl, or `cargo run --release -- filepath` to run code from file
`filepath`. The release profile has no dependencies beyond the Rust standard
library. I'm currently using rustc 1.82.0-nightly; this project uses the nightly
features `iterator_try_reduce` and `min_specialization`.

# About

This implementation consists of a compiler, bytecode instruction set, and
virtual machine all written in Rust. In comparison to other array languages,
this one

- uses the ASCII character set (for now?)
- uses a list-based array model (for now?)
- executes left-to-right instead of right-to-left (forever)
- can be parsed before execution (no comment)

## Features

- Operations on scalars (e.g. arithmetic) automatically scale up to arrays,
  pairing up elements.
- Full lexical scoping with closures.
- Guaranteed tail call elimination.
- Pattern matching in assignment and function parameter lists, including array
  destructuring and view patterns.
- Limited support for tacit programming.

## Example

Here's an example of a simple REPL session that calculates parenthesis nesting
depth. `\` followed by whitespace introduces a comment until the line ends. User
input appears after a two-space prompt. Everything else is the displayed result
of an expression, but note that assignments don't display their results.

      str: "(a(b c) (fg (h)) i)j"
      depth: str `~? "()" ~@ 1 _1 0 \:+  \ Mark each char with its depth
      ->depth + (str = ")")              \ Bump the depth of each right paren
      [str; depth Show ,]                \ Display chars over their depths
    ["(a(b c) (fg (h)) i)j"
     "11222221222233321110"]
      str[depth @]                       \ Group chars by depth (increasing)
    ["j"
     "(a  i)"
     "(b c)(fg )"
     "(h)"]

If you're not familiar with array languages, code like that might come as a
shock. The symbols!

## Codebase overview

- `src/`
    - [`main.rs`](src/main.rs) - entrypoint
    - [`lex.rs`](src/lex.rs) - tokenizer
    - [`parse.rs`](src/parse.rs) - parser
    - [`compile.rs`](src/compile.rs) - bytecode compiler
    - [`bytecode.rs`](src/bytecode.rs) - bytecode instruction set
    - [`val.rs`](src/val.rs) - runtime value representation
    - [`vm.rs`](src/vm.rs) - bytecode interpreter
    - [`prim.rs`](src/prim.rs) - some primitives
    - [`ops.rs`](src/ops.rs) - generic code for atomic operations
    - [`util.rs`](src/util.rs) - macros & general utilities
- [`benches/benchmark.rs`](benches/benchmark.rs) - silly little microbenchmarks
  (don't take these too seriously)
- [`emacs/arilus-mode.el`](emacs/arilus-mode.el) - simple emacs mode for syntax
  highlighting
- [`stdlib.arl`](stdlib.arl) - limited utility to support programming in the
  language
