use std::fmt;

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
    LoadModule { code_index: usize },  // code_index points right after module_start
    ModuleStart { num_instructions: usize },  // Followed by the module's instructions (num_instructions instructions, including ModuleEnd).
    ModuleEnd,
    Dup,  // Duplicates the top stack element.
    MakeClosure { num_closure_vars: usize },  // Immediately follows the Return instruction of a MakeFunc. Followed by `num_closure_vars` PushVar instructions which form the closure environment. Pops the top value of the stack, which must be an explicit function, and pushes a function with the closure environment formed by those PushVar instructions. The closure environment comes after the function body so we can compile functions in one pass, since we don't know how many closure vars there are until after compiling a function.
    MakeFunc { num_instructions: usize },  // Followed by the function's body (num_instructions instructions, including Return).
    Return,  // Discards the current stack frame and returns control to the instruction after the Call.
    JumpRelative { offset: i64 },  // Add `offset` to ip.
    JumpRelativeUnless { offset: i64 },  // Pops the top of the stack, then adds `offset` to `ip`, unless the popped value is falsy (its first atom is 0).
    PushLiteralInteger(i64),
    PushLiteralFloat(f64),
    PushVar { src: Var },  // Inside MakeClosure: Var to include in the closure environment. Otherwise: pushes src's value onto stack.
    PushVarLastUse { src: Var },  // Like PushVar, but this is the last possible use of this Var, so it can be moved out of the locals or closure environment. If this Var was the unique reference to its underlying array, then the array's storage may be reused.
    PushPrimFunc { prim: PrimFunc },  // Pushes `prim` onto stack.

    // TODO [..]F[..] ?
    Call1,  // Let [x, f] be the top two stack values (f on top). Pops both, calls f with x as an argument, and pushes the result of the call.
    Call2,  // Let [x, f, y] be the top three values of the stack (y on top). Pops all three, calls f with x and y as its left and right arguments, and pushes the result.
    CallN { num_args: usize }, // Let [f, x1, x2, .., xN] be the top num_args+1 values of the stack (xN on top). Pops all of them, calls f

    Pop,  // TODO currenltly we compile multi-statment expressions into (E1; Pop; E2; Pop; ...; EN) - can we instead do (E1; E2; ...; Pop(N-1); EN)? Pro - fewer pops; con - hold onto vals longer than necessary, may make a reference non-unique when it can be
    StoreTo { dst: Var }, // Copies the top stack value into dst.
    CallPrimFunc1 { prim: PrimFunc },  // Pops the top stack value, calls `prim` on it, and pushes the result. `prim` must not be Verb(PrimVerb::Rec).
    CallPrimFunc2 { prim: PrimFunc },  // Let [x, y] be the top two stack values (y on top). Pops both, calls `prim` with x and y as its left and right arguments, and pushes the result. `prim` must not be Verb(PrimVerb::Rec).
    CallPrimAdverb { prim: PrimAdverb },  // Let [f] be the top stack value. Pops `f`, Calls `prim` on it, and pushes the result.
    MakeString { num_bytes: usize }, // Followed by ceil(num_bytes/8) LiteralBytes.
    LiteralBytes { bytes: [u8; 8] }, // Following MakeString, forms the contents of the string. Outside, this is a char literal. TODO as a char literal, this is currently only ascii, and the first byte is the character; the rest are 0

    // Pops the top `num_elems` stack elements and collects them into an array, which is then pushed.
    CollectToArray { num_elems: usize },

    // Signals an error if the top element isn't an array of exactly `count`
    // elements. Otherwise, pushes that array's elements in reverse order (so
    // that the array's first element ends up on the top).
    //
    // TODO maybe pops the top of the stack? Popping is good for argument
    // lists/nested patterns, but for forward assignment we want to keep the
    // topmost val there.
    Splat { count: usize },

    // Pops the top two elements of the stack (D on top, M second), which must
    // be functions, and pushes a new verb where the monadic case is M and the
    // dyadic case is D.
    CollectVerbAlternatives,

    // Pops the top two elements of the stack (G on top, F second), both functions,
    // and pushes a new function equivalent to {x F G} : {x F y G}.
    MakeAtopFunc,

    // Pops the top two elements of the stack (n on top, F second), where F is a
    // function and n is a noun, and pushes a new function equivalent to {x F n}.
    MakeBoundFunc,

    // Pops the top three elements of the stack (G on top, H second, F third),
    // which must be functions, and pushes a new function equivalent to 
    // {x F H (x G)} : {x F y H (x G y)}
    MakeForkFunc,
}
               
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place { Local, ClosureEnv }

#[derive(Debug, Clone, Copy)]
pub struct Var { pub place: Place, pub slot: usize }

impl fmt::Display for Instr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Instr::PushVar { src } => write!(f, "PushVar({src})"),
            Instr::PushVarLastUse { src } => write!(f, "PushVarLastUse({src})"),
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
