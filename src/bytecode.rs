use std::fmt;

use crate::lex::*;

// The machine has two stacks:
//   - A value stack: values are pushed and popped to evaluate expressions.
//   - A function call stack: frames contain locals and a pointer to the closure environment
//
// "The stack" refers to the value stack. When a comment supposes the top of the stack is [a, b, c],
// the values are sorted from deepest to shallowest, so "c" is on top of the stack.
//
// "ip" denotes the instruction pointer - the index of an instruction in the code. When an
// instruction is executing, `ip` already points to the next instruction, so jump offsets are
// actually relative to the instruction *after* the jump.
//
// TODO explain how calls + headers work
//
// When you add an instruction that accesses a Var, update the following functions in compile.rs:
//   - `accessed`
//   - `decrement_locals`
//   - `mark_last_local_uses`
//
// TODO repr(c) and store parameters inline (i.e. [i32], decode instruction to enum, read parameters
// afterward)
#[derive(Debug, Clone, Copy)]
pub enum Instr {
    // Proceed to the next instruction. Generally speaking, Nops are a waste of space, but it can be
    // convenient for a compiler to emit them so it doesn't have to move everything over to remove
    // an instruction (which would involve adjusting offsets); it can just replace it with a Nop.
    Nop,

    // Followed by the module's instructions (`num_instructions` instructions, including ModuleEnd).
    ModuleStart { num_instructions: usize },

    // `code_index` points right after the ModuleStart of the module to load
    LoadModule { code_index: usize },

    // Returns to the instruction after LoadModule.
    ModuleEnd,

    // Followed by the function's body (num_instructions instructions, including Return).
    MakeFunc { num_instructions: usize },

    // Immediately follows the Return instruction of a MakeFunc. Followed by `num_closure_vars`
    // PushVar (or PushVarLastUse) instructions, which form the closure environment. Pops the top
    // value of the stack, which must be an explicit function, and pushes a function with the
    // closure environment formed by those PushVar instructions. The closure environment comes after
    // the function body so we can compile functions in one pass, since we don't know how many
    // closure vars there are until after compiling a function.
    //
    // After executing, `ip` points at the instruction after the last PushVar.
    MakeClosure { num_closure_vars: usize },

    // Like MakeClosure, but takes `num_closure_vars` values from the stack instead of being
    // followed by PushVar instructions.
    MakeClosureFromStack { num_closure_vars: usize },

    // Begin executing a function header. If it fails before running HeaderPassed, jump to (ip +
    // `next_case_offset`), where `ip` points to the instruction after this Header instruction.
    Header { next_case_offset: i64 },

    // Indicates that the function call's arguments conform to the current header. Splat failures
    // after this will be errors instead of resulting in falling back to the next function case.
    HeaderPassed,

    // Discards the current stack frame and returns control to the instruction after the Call.
    Return,

    // Add `offset` to ip.
    JumpRelative { offset: i64 },

    // Pops the top of the stack, then adds `offset` to `ip`, unless the popped value is falsy (its
    // first atom is 0).
    JumpRelativeUnless { offset: i64 },

    PushLiteralInteger(i64),

    PushLiteralFloat(f64),

    Pop,

    // Duplicates the top stack element.
    Dup,

    // Inside MakeClosure: Var to include in the closure environment. Otherwise: pushes src's value
    // onto stack.
    PushVar { src: Var },

    // Like PushVar, but this is the last possible use of this Var, so it can be moved out of the
    // locals or closure environment. If this Var was the unique reference to its underlying array,
    // then the array's storage may be reused.
    PushVarLastUse { src: Var },

    // Like PushVar, but places `src` below the top stack element instead of on the top.
    TuckVar { src: Var },

    // Like PushVarLastUse, but places `src` below the top stack element instead of on the top.
    TuckVarLastUse { src: Var },

    // Pops the top stack value and stores it in `dst`.
    StoreTo { dst: Var },

    PushPrimFunc { prim: PrimFunc },
    
    // Create a marker at the current stack position. Stack elements at or above that marker are
    // considered "marked". This is an ad-hoc way for instructions to address regions of the stack,
    // but the markers can be pretty confusing to manage in the vm code.
    MarkStack,

    /*
     * All call instructions leave a stack marker if they're calling an explicit function, which is
     * eventually popped by ArgCheckEq or HeaderPassed.
     */

    // Let [x, f] be the top of the stack. Pops both, calls f with x as an argument, and pushes the
    // result of the call.
    Call1,

    // Let [x, f, y] be the top of the stack. Pops all three, calls f with x and y as its left and
    // right arguments, and pushes the result.
    Call2,

    // Let [f, x1, x2, .., xN] be the top of the stack, with f..xN marked. Reverses all of themand
    // pops f, so the new top of stack is [xN, .. x2, x1], and then calls f.
    CallMarked,

    // Calls `var` on the marked arguments, which are on the stack in reverse order.
    CallOnArgs { var: Var },

    // Within an explicit function, pushes a new set of the provided arguments.
    CopyArgs,

    // Pops the marked elements, except for the `suffix_count` of them closest to the marker. If
    // `keep`, collects them into an array in reverse order and pushes the array. Note that this
    // doesn't pop the stack marker.
    CollectArgs { suffix_count: u32, keep: bool },

    // Pops the top stack value, calls `prim` on it, and pushes the result. `prim` must not be
    // Verb(PrimVerb::Rec).
    CallPrimFunc1 { prim: PrimFunc },

    // Let [x, y] be the top of the stack. Pops both, calls `prim` with x and y as its left and
    // right arguments, and pushes the result. `prim` must not be Verb(PrimVerb::Rec).
    CallPrimFunc2 { prim: PrimFunc },

    // Let [f] be the top of the stack. Pops `f`, Calls `prim` on it, and pushes the result.
    CallPrimAdverb { prim: PrimAdverb },

    // Followed by ceil(num_bytes/8) LiteralBytes.
    MakeString { num_bytes: usize },

    // Following MakeString, forms the contents of the string. Outside, this is a char literal. TODO
    // as a char literal, this is currently only ascii, and the first byte is the character; the
    // rest are 0
    LiteralBytes { bytes: [u8; 8] },

    // Consumes the current stack marker; pops the marked elements and collects them into an array,
    // which is then pushed.
    CollectMarkedToArray,

    // Signals an error if the top element isn't an array. Pushes that array's elements (so that the
    // array's last element ends up on top).
    Splat,

    // Like Splat, but pushes the array's elements in reverse order (so the first element ends up on
    // top). Signals an error if the top element isn't an array of exactly `count` elements.
    SplatReverse { count: usize },

    // Like SplatReverse, but
    //   - Expects an array with at least `prefix_count` + `suffix_count` elements.
    //   - Leaves the stack with (in order from top to bottom)
    //       1) `prefix_count` elements on top, reversed;
    //       2) if `keep_splice` is true, the elements between the prefix and the suffix, *not
    //          reversed*;
    //       3) `suffix_count` elements, reversed.
    //
    // This allows us to support patterns like [first; ..; last] without necessarily traversing the
    // whole array.
    SplatReverseWithSplice { prefix_count: u32, suffix_count: u32, keep_splice: bool },

    // Checks that the function has been provided `count` args. If this fails, throw an error; if it
    // succeeds, the current marker is popped and all its marked stack values are popped.
    ArgCheckEq { count: usize },

    // Like ArgCheckEq, but checks that the function has been provided *at least* `count` args and
    // leaves the current stack marker where it is.
    ArgCheckGe { count: usize },

    // Pop the top element of the stack: if its first atom is 0, throw an error or move to the next
    // header.
    Assert,
}
               
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place { Local, ClosureEnv }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Var { pub place: Place, pub slot: usize }

impl Var {
    pub fn is_local(&self) -> bool {
        self.place == Place::Local
    }
}

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

    // "Adverbs"
    Runs,  // {|f| {|x| :|x;y|}

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
    Has,
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

impl PrimFunc {
    pub fn from_verb(verb: PrimVerb, monad: bool) -> Self {
        use PrimFunc::*;
        let fail = || {
            todo!("{} `{verb}'", if monad { "monadic" } else { "dyadic" })
        };
        match verb {
            PrimVerb::At => if monad { GroupIndices } else { Index }
            PrimVerb::Comma => if monad { Ravel } else { Append }
            PrimVerb::CommaColon => if monad { fail() } else { Windows }
            PrimVerb::DotColon => if monad { fail() }  else { Chunks }
            PrimVerb::Plus => if monad { fail() } else { Add }
            PrimVerb::Minus => if monad { Neg } else { Sub }
            PrimVerb::Asterisk => if monad { fail() } else { Mul }
            PrimVerb::Slash => if monad { Ints } else { Div }
            PrimVerb::DoubleSlash => if monad { fail() } else { IntDiv }
            PrimVerb::Percent => if monad { fail() } else { Mod }
            PrimVerb::Equals => if monad { fail() } else { Equal }
            PrimVerb::EqualBang => if monad { fail() } else { NotEqual }
            PrimVerb::DoubleEquals => if monad { fail() } else { Match }
            PrimVerb::Hash => if monad { Length } else { Take }
            PrimVerb::HashColon => if monad { fail() } else { Copy }
            PrimVerb::Caret => if monad { Inits } else { Pow }
            PrimVerb::Pipe => if monad { Rev } else { Or }
            PrimVerb::Bang => if monad { Not } else { fail() }
            PrimVerb::Dollar => if monad { Tails } else { Drop }
            PrimVerb::LessThan => if monad { Sort } else { LessThan }
            PrimVerb::LessThanColon => if monad { Asc } else { Min }
            PrimVerb::LessThanEquals => if monad { fail()  } else { LessThanEqual }
            PrimVerb::GreaterThan => if monad { SortDesc } else { GreaterThan }
            PrimVerb::GreaterThanColon => if monad { Desc } else { Max }
            PrimVerb::GreaterThanEquals => if monad { fail() } else { GreaterThanEqual }
            PrimVerb::Question => if monad { Where } else { Find }
            PrimVerb::QuestionColon => if monad { fail() } else { FindAll }
            PrimVerb::P => if monad { Identity } else { IdentityLeft }
            PrimVerb::Q => if monad { Identity } else { IdentityRight }
            PrimVerb::DoublePipe => fail(),
            PrimVerb::DoubleAmpersand => fail(),
            PrimVerb::Ampersand => fail(),
        }
    }
}
