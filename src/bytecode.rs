use crate::lex::*;

// The machine has
//   - Three registers: subject1, verb, and subject2 (subject1 and verb are both stacks - is subject2 too? (verb formation))
//   - A function call stack (frames contain locals and a pointer to the closure environment)
//   - A heap
//
// TODO repr(c) and store parameters inline (i.e. [i32], decode instructions to
// enum, read parameters afterward)
//
// TODO optimize PushSubject1+PopToSubject2 -> LoadSubject2
// TODO instead of subject2, maybe we just have call1 and call2 instructions (less shuffling around)
#[derive(Debug)]
pub enum Instr {
    Nop,
    Halt { exit_status: u8 },
    MakeClosure { num_closure_vars: usize },  // Immediately followed by MakeFunc and the function's body, then `num_closure_vars` PushVar instructions which form the closure environment.  Leaves the function object in subject1. The closure environment comes after the function body so we can compile functions in one pass, since we don't know how many closure vars there are until after compiling a function.
    MakeFunc { num_instructions: usize },  // Followed by the function's body (num_instructions instructions).
    AllocLocals { num_locals: usize },  // Allocates space for `num_locals` locals on the current stack frame.
    Return,  // Discards the current stack frame and returns control to the instruction after the Call.
    PushLiteralInteger(i64),   // Copies value to subject1, pushing subject1's previous contents (if any)
    PushVar { src: Var },  // Inside MakeClosure: Var to include in the closure environment. Otherwise: copies src to subject1, pushing subject1's previous contents (if any)
    PushPrimVerb { prim: PrimVerb },  // Copies src to verb, pushing verb's previous contents (if any)
    PushVerb { src: Var },            // Copies src to verb, pushing verb's previous contents (if any)  (TODO instead provide start instr?)
    Call,  // Calls verb on subject1 and, if present, subject2. Discards subject1, verb, and subject2.
    Pop,  //  Reinstates the previous subject1 (/verb? and discards subject2?)
    PopVerb,  //  Reinstates the previous verb
    PopToSubject2,  // Copies subject1 to subject2 and reinstates the previous subject1
    PopToVerb,      // Copies subject1 to verb and reinstates the previous subject1
    StoreVerbTo { dst: Var }, // Copies verb to dst
    StoreTo { dst: Var },     // Copies subject1 to dst
    CallPrimVerb { prim: PrimVerb },  // Calls prim on subject1 and, if present, subject2; places the result in subject1
    CallPrimAdverb { prim: PrimAdverb },  // Operates on verb
    MoveVerbToSubject1,
    MakeString { num_bytes: usize }, // Followed by num_bytes/8 LiteralBytes.
    LiteralBytes { bytes: [u8; 8] }, // Following MakeString, forms the contents of the string. Outside, this is a char literal
    
    // Collects the top `num_elems` on the subject1 stack into an array, which is pushed to subject1.
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
