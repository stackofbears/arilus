use crate::lex::*;

// The machine has
//   - An value stack: values - data and functions - are pushed onto the stack and popped to call functions
//   - A function call stack (frames contain locals and a pointer to the closure environment)
//
// TODO repr(c) and store parameters inline (i.e. [i32], decode instructions to
// enum, read parameters afterward)
//
// TODO optimize PushSubject1+PopToSubject2 -> LoadSubject2
// TODO instead of subject2, maybe we just have call1 and call2 instructions (less shuffling around). Can we do all this with just one stack? [..., x, v, y] -> v(x, y)
#[derive(Debug, Clone, Copy)]
pub enum Instr {
    Nop,
    Halt { exit_status: i32 },  // Terminate the virtual machine and return `exit_status` to the OS.
    MakeClosure { num_closure_vars: usize },  // Immediately followed by MakeFunc and the function's body, then `num_closure_vars` PushVar instructions which form the closure environment.  Leaves the function object in subject1. The closure environment comes after the function body so we can compile functions in one pass, since we don't know how many closure vars there are until after compiling a function.
    MakeFunc { num_instructions: usize },  // Followed by the function's body (num_instructions instructions).
    AllocLocals { num_locals: usize },  // Allocates space for `num_locals` locals on the current stack frame.
    Return,  // Discards the current stack frame and returns control to the instruction after the Call.
    PushLiteralInteger(i64),
    PushLiteralFloat(f64),
    PushVar { src: Var },  // Inside MakeClosure: Var to include in the closure environment. Otherwise: pushes src's value onto stack.
    PushPrimVerb { prim: PrimVerb },  // Pushes `prim` onto stack.
    Call1,  // Let [x, f] be the top two stack values (f on top). Pops both, calls f with x as an argument, and pushes the result of the call.
    Call2,  // Let [x, f, y] be the top three values of the stack (y on top). Pops all three, calls f with x and y as its left and right arguments, and pushes the result.
    Pop,  // TODO currenltly we compile multi-statment expressions into (E1; Pop; E2; Pop; ...; EN) - can we instead do (E1; E2; ...; Pop(N-1); EN)? Pro - fewer pops; con - hold onto vals longer than necessary, may make a reference non-unique when it can be
    StoreTo { dst: Var }, // Copies the top stack value into dst.
    CallPrimVerb1 { prim: PrimVerb },  // Pops the top stack value, calls `prim` on it, and pushes the result
    CallPrimVerb2 { prim: PrimVerb },  // Let [x, y] be the top two stack values (y on top). Pops both, calls `prim` with x and y as its left and right arguments, and pushes the result.
    CallPrimAdverb { prim: PrimAdverb },  // Let [f] be the top stack value. Pops `f`, Calls `prim` on it, and pushes the result.
    MakeString { num_bytes: usize }, // Followed by ceil(num_bytes/8) LiteralBytes.
    LiteralBytes { bytes: [u8; 8] }, // Following MakeString, forms the contents of the string. Outside, this is a char literal. TODO as a char literal, this is currently only ascii, and the first byte is the character; the rest are 0

    // Pops the top `num_elems` stack elements and collects them into an array, which is then pushed.
    CollectToArray { num_elems: usize },
}
               
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place { Local, ClosureEnv }
#[derive(Debug, Clone, Copy)]
pub struct Var { pub place: Place, pub slot: usize }
               
               // Array literals: [1,2,3]:
               //   PushEmptyArray        (sub1: [])
               //
               //   PushPrimVerb ,        (sub1: [])      (verb: ,)
               //   PushEmptyArray        (sub1: [] [])   (verb: ,)
               //   PushLiteralInteger 1  (sub1: [] [] 1) (verb: ,)
               //   PopToSubject2         (sub1: [] [])   (verb: ,) (sub2: 1)
               //   Call                  (sub1: [] [1])
               //   PushPrimVerb ,        (sub1: [] [1])  (verb: ,)
               //   PopToSubject2         (sub1: [])      (verb: ,) (sub2: [1])
               //   Call                  (sub1: [1])
               //
               //   PushPrimVerb ,        (sub1: [1])      (verb: ,)
               //   PushEmptyArray        (sub1: [1] [])   (verb: ,)
               //   PushLiteralInteger 1  (sub1: [1] [] 2) (verb: ,)
               //   PopToSubject2         (sub1: [1] [])   (verb: ,) (sub2: 2)
               //   Call                  (sub1: [1] [2])
               //   PushPrimVerb ,        (sub1: [1] [2])  (verb: ,)
               //   PopToSubject2         (sub1: [1])      (verb: ,) (sub2: [2])
               //   Call                  (sub1: [1,2])
               //
               //   PushPrimVerb ,        (sub1: [1,2])      (verb: ,)
               //   PushEmptyArray        (sub1: [1,2] [])   (verb: ,)
               //   PushLiteralInteger 1  (sub1: [1,2] [] 3) (verb: ,)
               //   PopToSubject2         (sub1: [1,2] [])   (verb: ,) (sub2: 3)
               //   Call                  (sub1: [1,2] [3])
               //   PushPrimVerb ,        (sub1: [1,2] [3])  (verb: ,)
               //   PopToSubject2         (sub1: [1,2])      (verb: ,) (sub2: [3])
               //   Call                  (sub1: [1,2,3])

               // vs having an AppendItem prim
               //   PushEmptyArray        (sub1: [])
               //
               //   PushPrimVerb AI       (sub1: [])      (verb: AI)
               //   PushLiteralInteger 1  (sub1: [] 1)    (verb: AI)
               //   PopToSubject2         (sub1: [])      (verb: AI) (sub2: 1)
               //   Call                  (sub1: [1])
               //
               //   PushPrimVerb AI       (sub1: [1])      (verb: AI)
               //   PushLiteralInteger 2  (sub1: [1] 2)    (verb: AI)
               //   PopToSubject2         (sub1: [1])      (verb: AI) (sub2: 2)
               //   Call                  (sub1: [1,2])

               //   PushPrimVerb AI       (sub1: [1,2])      (verb: AI)
               //   PushLiteralInteger 2  (sub1: [1,2] 3)    (verb: AI)
               //   PopToSubject2         (sub1: [1,2])      (verb: AI) (sub2: 3)
               //   Call                  (sub1: [1,2,3])

               // vs having special array literal instructions
               //   MakeArray 3
               //   PushLiteralInteger 1  
               //   Pop                   (sub1: [1])
               //   PushLiteralInteger 2
               //   Pop                   (sub1: [1,2])
               //   PushLiteralInteger 3
               //   Pop                   (sub1: [1,2,3])

// TODO function has number of locals in code or in Val? Probably in code since
// it can't change. Principle: put it in code if you can

/* Compilation of functions
MakeClosure (closure env len)
(closure env len * Var) -> copy vars to env
EnterFunc (num locals)
*/
