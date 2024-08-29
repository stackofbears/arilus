use std::fmt;

use crate::lex::*;

// The machine has
//   - An value stack: values - data and functions - are pushed onto the stack and popped to call functions
//   - A function call stack (frames contain locals and a pointer to the closure environment)
//
// When you add an instruction that accesses a Var, update the following functions in compile.rs:
//   - `accessed`
//   - `decrement_locals`
//   - `mark_last_local_uses`
//
// When you add an instruction that jumps or terminates a scope (like
// MakeClosure), update `mark_last_local_uses`.
//
// TODO repr(c) and store parameters inline (i.e. [i32], decode instructions to
// enum, read parameters afterward)
#[derive(Debug, Clone, Copy)]
pub enum Instr {
    Nop,
    LoadModule { code_index: usize },  // code_index points right after module_start
    ModuleStart { num_instructions: usize },  // Followed by the module's instructions (num_instructions instructions, including ModuleEnd).
    ModuleEnd,
    Dup,  // Duplicates the top stack element.
    MakeClosure { num_closure_vars: usize },  // Immediately follows the Return instruction of a MakeFunc. Followed by `num_closure_vars` PushVar instructions which form the closure environment. Pops the top value of the stack, which must be an explicit function, and pushes a function with the closure environment formed by those PushVar instructions. The closure environment comes after the function body so we can compile functions in one pass, since we don't know how many closure vars there are until after compiling a function.
    MakeClosureFromStack { num_closure_vars: usize },  // Like MakeClosure, but takes `num_closure_vars` from the stack instead of being followed by PushVar instructions.
    MakeFunc { num_instructions: usize },  // Followed by the function's body (num_instructions instructions, including Return).
    Header { next_case_offset: i64 },  // Begin executing a function header. If it fails before running HeaderPassed, jump to (ip + `next_case_offset`), where `ip` points to the instruction after this Header instruction.
    HeaderPassed,  // Indicates that the function call's arguments conform to the current header. Splat failures after this will be errors instead of resulting in falling back to the next function case.
    Return,  // Discards the current stack frame and returns control to the instruction after the Call.
    JumpRelative { offset: i64 },  // Add `offset` to ip.
    JumpRelativeUnless { offset: i64 },  // Pops the top of the stack, then adds `offset` to `ip`, unless the popped value is falsy (its first atom is 0).
    PushLiteralInteger(i64),
    PushLiteralFloat(f64),
    PushVar { src: Var },  // Inside MakeClosure: Var to include in the closure environment. Otherwise: pushes src's value onto stack.
    PushVarLastUse { src: Var },  // Like PushVar, but this is the last possible use of this Var, so it can be moved out of the locals or closure environment. If this Var was the unique reference to its underlying array, then the array's storage may be reused.
    TuckVar { src: Var },  // Like PushVar, but places `src` below the top stack element instead of on the top.
    TuckVarLastUse { src: Var },  // Like PushVarLastUse, but places `src` below the top stack element instead of on the top.
    PushPrimFunc { prim: PrimFunc },  // Pushes `prim` onto stack.

    // All call instructions leave a stack marker if they're calling an explicit function, which is eventually popped by ArgCheckEq or HeaderPassed
    Call1,  // Let [x, f] be the top two stack values (f on top). Pops both, calls f with x as an argument, and pushes the result of the call.
    Call2,  // Let [x, f, y] be the top three values of the stack (y on top). Pops all three, calls f with x and y as its left and right arguments, and pushes the result.
    CallMarked,  // Let [f, x1, x2, .., xN] be the top values of the stack, with f..xN marked and xN on top. Reverses all of them and pops f, so the new stack is [xN, .. x2, x1] and calls f.
    CallOnArgs { var: Var },  // Calls `var` on the marked arguments which are on the stack in reverse order.
    MarkStack,  // Create a marker at the current stack position. Stack elements at or above that marker are considered "marked".
    CopyArgs,  // Within an explicit function, pushes a new set of the provided arguments.
    Pop,  // TODO currenltly we compile multi-statment expressions into (E1; Pop; E2; Pop; ...; EN) - can we instead do (E1; E2; ...; Pop(N-1); EN)? Pro - fewer pops; con - hold onto vals longer than necessary, may make a reference non-unique when it can be
    StoreTo { dst: Var }, // Copies the top stack value into dst.
    CallPrimFunc1 { prim: PrimFunc },  // Pops the top stack value, calls `prim` on it, and pushes the result. `prim` must not be Verb(PrimVerb::Rec).
    CallPrimFunc2 { prim: PrimFunc },  // Let [x, y] be the top two stack values (y on top). Pops both, calls `prim` with x and y as its left and right arguments, and pushes the result. `prim` must not be Verb(PrimVerb::Rec).
    CallPrimAdverb { prim: PrimAdverb },  // Let [f] be the top stack value. Pops `f`, Calls `prim` on it, and pushes the result.
    MakeString { num_bytes: usize }, // Followed by ceil(num_bytes/8) LiteralBytes.
    LiteralBytes { bytes: [u8; 8] }, // Following MakeString, forms the contents of the string. Outside, this is a char literal. TODO as a char literal, this is currently only ascii, and the first byte is the character; the rest are 0

    // Consumes the current stack marker; pops the marked elements and collects
    // them into an array, which is then pushed.
    CollectMarkedToArray,

    // Pops the marked elements, except for the `suffix_count` of them closest
    // to the marker. If `keep`, collects them into an array in reverse order
    // and pushes the array. Note that this doesn't pop the stack marker.
    CollectArgs { suffix_count: u32, keep: bool },

    // Signals an error if the top element isn't an array. Pushes that array's
    // elements (so that the array's last element ends up on top).
    //
    // TODO maybe pops the top of the stack? Popping is good for argument
    // lists/nested patterns, but for forward assignment we want to keep the
    // topmost val there.
    Splat,

    // Like Splat, but pushes the array's elements in reverse order (so the
    // first element ends up on top). Signals an error if the top element isn't an
    // array of exactly `count` elements.
    SplatReverse { count: usize },

    // Like SplatReverse, but
    //   - Expects an array with at least `prefix_count` + `suffix_count` elements.
    //   - Leaves the stack with
    //       1) `prefix_count` elements on top, reversed;
    //       2) if `keep_splice` is true, the elements between the prefix and the
    //          suffix, *not reversed*;
    //       3) `suffix_count` elements, reversed.
    //
    // This allows us to support patterns like [first; ..; last] without
    // necessarily traversing the whole array.
    SplatReverseWithSplice { prefix_count: u32, suffix_count: u32, keep_splice: bool },

    // Check that the function has been provided `count` args. If this fails,
    // throw an error; if it succeeds, the current marker is popped and all its
    // marked stack values are popped.
    ArgCheckEq { count: usize },

    // Like ArgCheckEq, but checks that the function has been provided *at
    // least* `count` args and leaves the current stack marker where it is.
    ArgCheckGe { count: usize },

    // Pop the top element of the stack: if its first atom is 0, throw an error
    // or move to the next header.
    Assert,
}
               
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place { Local, ClosureEnv }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Var { pub place: Place, pub slot: usize }

impl fmt::Display for Instr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Instr::PushVar { src } => write!(f, "PushVar({src})"),
            Instr::PushVarLastUse { src } => write!(f, "PushVarLastUse({src})"),
            Instr::TuckVar { src } => write!(f, "TuckVar({src})"),
            Instr::TuckVarLastUse { src } => write!(f, "TuckVarLastUse({src})"),
            Instr::CallOnArgs { var } => write!(f, "CallOnArgs({var})"),
            Instr::StoreTo { dst } => write!(f, "StoreTo({dst})"),
            Instr::CallPrimFunc1 { prim } => write!(f, "CallPrimFunc1({prim})"),
            Instr::CallPrimFunc2 { prim } => write!(f, "CallPrimFunc2({prim})"),
            Instr::CallPrimAdverb { prim } => write!(f, "CallPrimAdverb({prim})"),
            Instr::LiteralBytes { bytes } => {
                let as_str = std::str::from_utf8(bytes).map_err(|_| fmt::Error)?;
                write!(f, "LiteralBytes({as_str:?})")
            }
            _ => <Instr as fmt::Debug>::fmt(self, f),
        }
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.place {
            Place::Local => write!(f, "Local({})", self.slot),
            Place::ClosureEnv => write!(f, "ClosureEnv({})", self.slot),
        }
    }
}

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimFunc {
    // Built-in verbs, possibly with monadic and dyadic cases.
    Verb(PrimVerb),
    
    // Derived functions recognized from patterns in the code (e.g. a certain
    // application of an adverb).
    Sum,
    
    // Rec stands for the innermost explicit function.
    Rec,

    // Primitive identifier functions with fixed arity.
    // Monadic
    Ints,
    Rev,
    Where,
    Nub,
    Identity,
    Asc,
    Desc,
    Sort,
    SortDesc,
    Inits,
    Tails,
    Not,
    Ravel,
    Floor, Ceil,
    Length,
    Exit,
    Show,
    Print,
    GetLine,
    ReadFile,
    Rand,
    Type,
    PrintBytecode,

    // Dyadic
    Take,
    Drop,
    Rot,
    Find,
    FindAll,
    FindSubseq,
    In,
    Copy,
    IdentityLeft,
    IdentityRight,
    ShiftLeft,
    ShiftRight,
    And,
    Or,
    Equal,
    NotEqual,
    Match,
    NotMatch,
    GreaterThan,
    GreaterThanEqual,
    LessThan,
    LessThanEqual,
    Index,
    Pick,
    Append,
    Cons,
    Snoc,
    Cut,
    Add, Sub, Neg, Abs, Mul, Div, IntDiv, Mod, Pow, Log,
    Min, Max,
    Windows, Chunks,
    GroupBy,
    GroupIndices,
    SendToIndex,

    // Invisible primitives. This is actually controlled by their absence from
    // make_primitive_identifier_map in parsing.

    // Currently used only for repl display, but this could be exposed once it
    // prints in valid syntax (instead of Rust Debug).
    DebugPrint,
}

impl std::fmt::Display for PrimFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            PrimFunc::Verb(prim_verb) => prim_verb.fmt(f),
            PrimFunc::Sum => f.write_str("\\+"),
            _ => fmt::Debug::fmt(self, f),
        }
    }
}
