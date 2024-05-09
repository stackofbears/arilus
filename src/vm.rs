// TODO there are definitely some places here where we don't track the refcounts
// correctly
//
// TODO make refcounts atomic; read
// https://doc.rust-lang.org/nomicon/atomics.html, links at
// https://stackoverflow.com/questions/30407121/which-stdsyncatomicordering-to-use

use std::{
    borrow::Cow,
    mem::{self, ManuallyDrop},
};


use crate::ptr_range::*;
use crate::bytecode::*;
use crate::lex::*;
use crate::util::*;

// Maybe we go back to moving lists to the heap when they're sliced (and in
// general moving vals to the heap when they're referenced).
// Does this work for e.g. x '# 2
//   Would want to increment rc when creating slices
//   Could re-traverse after whole operation and inc rcs?
//    But how would we know which refs were there before and which are new?
//    Could dec beforehand w/o deletion and then inc all after
//
// Preventing cycles: including element at equal or greater depth triggers copy?

// alt formulation: two heaps, one for most vals & one for array vecs
//
// alt: Slice becomes raw heap slot (for rc)+ptr to start+len. we just agree that vector elements can't be moved (or mutated?) through a non-unique reference

/*
problem: some vals need to go on heap for refs



o1: put everything on heap, just use heap indices as values
  pro
o2: move val to heap when ref is taken
  [1;2;3]
o3: arrays (ie dyn sized objects) always on heap, others off or on (default off, values move on when ref taken)
actually, maybe o2 since arrays aren't *actually* expensive to move
the difference is that capturing lists by value would always elicit a move to heap, unlike slices/ints/refs, so why not put them there in the first place?

[1;2;3]->a  \ a is Val:Ints(Vec)
a#2  \ Val:Ints moves to heap w/ rc 2, a becomes Val:Slice(0,len,slot), result is Val:Slice(0,2,slot)

a: retaking reference to array x

[1;2;3]->a
a#2->b
c:ref a

a:points to array
b:slice of a
c: 


question:
a:1
F:{a:a+1}
G:F
[]F  \ a=2
[]G  \ ? Does G share F's environment? If not, then you can't mutate closure envs by eg sending them to functions

f{[]X}

maybe closure envs are refs by default

*/

// TODO rcval = Rc<Val>?

// TODO Ref
// TODO write Clone instance manually and panic on List
#[derive(Debug, Clone)]
enum Val {
    Int(i64),  // TODO TwoInts, ThreeInts
    Char(u8),

    Slice(DataHandle),

    // TODO switch this out for the non-token type (some primitives won't have
    // token representations)
    PrimFunc(PrimVerb),

    // TODO decide if closures should refer to a shared environment or be value
    // types (copying copies environment. Currently, we hold a reference to the env.
    //   a:1; F:{a:a+1}; G:F
    //   []F  \ 2
    //   []G  \ Currently this returns 3. Should G have its own copy of the env, making this 2?
    ExplicitFunc {
        // Always points to List::Vals.
        closure_env_heap_slot: usize,
        // The function's first instruction (ALWAYS points after MakeFunc, so
        // you can get the function's instruction count with
        // code[func.code_index-1]
        code_index: usize,
    },
    
    AdverbDerivedFunc {
        adverb: PrimAdverb,
        operand_heap_slot: usize,
    },

    // Other Vals are easily copied, but lists are special! When they first
    // become referents of another Val, they must move to the heap and replaced
    // with a full slice. This happens even when they're selected by index from
    // another list, e.g.
    // 
    //   a:[[1;2;3]]; a@0  \ Executing a@0 changes a's first item into a Val::Slice.
    //
    // There's no semantic difference between a List and a Slice; there's just a
    // little extra bookkeeping and indirection that go into taking care of a
    // Slice. Maybe we'll eventually apply some reasoning - like lifetime
    // analysis stuff or idiom recognition - so expressions like a@0@0 don't
    // necessitate a move of a@0 onto the heap. The same stuff could be used to
    // avoid refcount operations.
    //
    // TODO unnest list enums?
    List(ManuallyDrop<List>),
}

impl Val {
    fn len(&self) -> Option<usize> {
        use Val::*;
        match self {
            Slice(DataHandle { len, .. }) => Some(*len),
            List(list) => Some(list.len()),
            Int(_) | Char(_) | PrimFunc(_) | ExplicitFunc {..} | AdverbDerivedFunc {..} => None,
        }
    }
    
    fn length_mismatch(&self, other: &Val) -> Option<(usize, usize)> {
        let len1 = self.len()?;
        let len2 = other.len()?;
        if len1 == len2 { None } else { Some((len1, len2)) }
    }
}

#[derive(Debug, Clone)]
struct DataHandle {
    heap_slot: usize,
    start_offset: usize,
    len: usize,
}

// Assertion
const _: [(); mem::align_of::<Val>()] = [(); mem::align_of::<usize>()];

// TODO Refs
#[derive(Debug, Clone)]
enum List {
    I64s(Vec<i64>),
    U8s(Vec<u8>),
    Vals(Vec<Val>),
}    

impl List {
    fn len(&self) -> usize {
        match self {
            List::I64s(v) => v.len(),
            List::U8s(v) => v.len(),
            List::Vals(v) => v.len(),
        }
    }
}

impl FromIterator<i64> for List {
    fn from_iter<T>(iter: T) -> Self where T: IntoIterator<Item = i64> {
        List::I64s(iter.into_iter().collect())
    }
}
impl FromIterator<u8> for List {
    fn from_iter<T>(iter: T) -> Self where T: IntoIterator<Item = u8> {
        List::U8s(iter.into_iter().collect())
    }
}
impl FromIterator<Val> for List {
    fn from_iter<T>(iter: T) -> Self where T: IntoIterator<Item = Val> {
        List::Vals(iter.into_iter().collect())
    }
}

enum Slice {
    I64s(PtrRange<i64>),
    U8s(PtrRange<u8>),
    Vals(PtrRange<Val>),
}

#[derive(Debug)]
struct RcVal {
    rc: usize,  // TODO make atomic
    val: Val,
}

impl RcVal {
    fn new_null() -> Self {
        Self { rc: 0, val: Val::Int(0) }  // Arbitrarily choose Int
    }
}

#[derive(Debug)]
struct StackFrame {
    // Heap slot. TODO replace with NonNull<Val>?
    closure_env_heap_slot: usize,

    // Index into locals_stack. Local slots are offsets from this index.
    locals_start: usize,

    // Code index after Call; or, if usize::MAX, return from the execute
    // loop. TODO replace with Option<NonZeroUsize>?
    ret_addr: usize,
}

pub struct Mem<'a> {
    code: &'a [Instr],

    subject1: Vec<Val>,
    verb: Vec<Val>,
    subject2: Option<Val>,
    
    // Stack of local scopes
    locals_stack: Vec<Val>,
    stack_frames: Vec<StackFrame>,

    heap: Vec<RcVal>,
    free_heap_slots: Vec<usize>,  // TODO scan heap for rc==0 or store bitvec aside instead?
}

impl<'a> Mem<'a> {
    pub fn new(initial_heap_slots: usize) -> Self {
        Self {
            code: &[],
            subject1: vec![],
            verb: vec![],
            subject2: None,
            locals_stack: vec![],  // TODO stdlib?
            stack_frames: vec![StackFrame {
                closure_env_heap_slot: usize::MAX,
                locals_start: 0,
                ret_addr: usize::MAX,
            }],
            heap: {
                let mut heap = vec![];
                heap.resize_with(initial_heap_slots, RcVal::new_null);
                heap
            },
            free_heap_slots: (0..initial_heap_slots).rev().collect(),
        }
    }

    pub fn set_code(&mut self, code: &'a [Instr]) {
        self.code = code;
    }

    fn store(&mut self, dst: Var, val: Val) {
        for heap_slot in self.get_heap_slot(&val) { self.increment_rc(heap_slot); }
        let frame = self.stack_frames.last().unwrap();
        match dst.place {
            Place::Local => {
                let absolute_slot = frame.locals_start + dst.slot;
                if absolute_slot >= self.locals_stack.len() {
                    self.locals_stack.resize(absolute_slot + 1, Val::Int(0));  // TODO is Int(0) right?
                }
                let old_val = mem::replace(&mut self.locals_stack[absolute_slot], val);
                self.drop_val(old_val);
            }
            Place::ClosureEnv => match &mut self.heap[frame.closure_env_heap_slot].val {
                Val::List(list) => match &mut **list {
                    List::Vals(vals) => {
                        let old_val = mem::replace(&mut vals[dst.slot], val);
                        self.drop_val(old_val);
                    }
                    // TODO consolidate closure envs of all-matching types to
                    // e.g. Int64s instead of Vals.
                    _ => unreachable!(),
                }
                // TODO if a closure env consists of one value, it's fine not to
                // use a whole list.
                _ => unreachable!(),  
            }
        }
    }

    fn get_heap_slot(&self, val: &Val) -> Option<usize> {
        match val {
            Val::ExplicitFunc { closure_env_heap_slot: heap_slot, .. } |
            Val::AdverbDerivedFunc { operand_heap_slot: heap_slot, .. } |
            Val::Slice(DataHandle { heap_slot, .. }) => 
                if *heap_slot != usize::MAX { Some(*heap_slot) } else { None }
            // TODO List should return the heap slots of all its elements, but
            // adjusting their refcounts would require allocating or holding
            // onto a ref to the list
            Val::Int(_) | Val::Char(_) | Val::PrimFunc(_) | Val::List(_) => None,  
        }
    }

    fn increment_rc(&mut self, heap_slot: usize) {
        if heap_slot == usize::MAX { return; }
        self.heap[heap_slot].rc += 1;
    }
    
    fn decrement_rc(&mut self, heap_slot: usize) {
        if heap_slot == usize::MAX { return; }
        let RcVal { rc, val } = &mut self.heap[heap_slot];
        *rc -= 1;
        if *rc == 0 {
            let old_val = mem::replace(val, Val::Int(0));
            self.drop_val(old_val);
            self.free_heap_slots.push(heap_slot);
        }
    }

    fn put_on_heap(&mut self, data: RcVal) -> usize {
        match self.free_heap_slots.pop() {
            Some(slot) => {
                self.heap[slot] = data;
                slot
            }
            None => {
                self.heap.push(data);
                self.heap.len() - 1
            }
        }
    }

    fn drop_val(&mut self, val: Val) {
        match val {
            Val::Slice(DataHandle { heap_slot, .. }) => self.decrement_rc(heap_slot),
            Val::ExplicitFunc { closure_env_heap_slot, .. } => self.decrement_rc(closure_env_heap_slot),
            Val::AdverbDerivedFunc { operand_heap_slot, .. } => self.decrement_rc(operand_heap_slot),
            Val::List(list) => if let List::Vals(vals) = ManuallyDrop::into_inner(list) {
                for val in vals { self.drop_val(val); }
            },
            Val::Int(_) | Val::Char(_) | Val::PrimFunc(_) => (),
        }
    }

    fn drop_subject1(&mut self) {
        if let Some(val) = self.subject1.pop() {
            self.drop_val(val);
        }
    }

    fn drop_verb(&mut self) {
        if let Some(val) = self.verb.pop() {
            self.drop_val(val);
        }
    }

    fn drop_subject2(&mut self) {
        if let Some(val) = self.subject2.take() {
            self.drop_val(val);
        }
    }

    pub fn execute(&mut self, mut ip: usize) -> Result<u8, String> {
        use Instr::*;
        while ip < self.code.len() {
            match self.code[ip] {
                Nop => ip += 1,
                Halt { exit_status } => return Ok(exit_status),
                MakeClosure { num_closure_vars } => {
                    ip += 1;
                    let num_instructions = match &self.code[ip] {
                        MakeFunc { num_instructions } => { ip += 1; num_instructions }
                        bad => panic!("Malformed code at ip {ip}: expected MakeFunc after MakeClosure, but found {bad:?}"),
                    };

                    let code_index = ip;  // First instruction of function body

                    ip += num_instructions;
                    let closure_env_heap_slot = if num_closure_vars == 0 {
                        usize::MAX
                    } else {
                        let mut closure_data = Vec::with_capacity(num_closure_vars);
                        for _ in 0..num_closure_vars {
                            if let PushVar { src } = self.code[ip] {
                                ip += 1;
                                // TODO use e.g. List::I64s if closure vals are all ints
                                let val = self.copy_from_var(src);
                                closure_data.push(val);
                            } else {
                                panic!("Malformed code at ip {ip}: expected PushVar after MakeClosure, but found {:?}", self.code[ip]);
                            }
                        }
                        self.put_on_heap(RcVal { rc: 1, val: Val::List(ManuallyDrop::new(List::Vals(closure_data))) })
                    };
                    self.verb.push(Val::ExplicitFunc { closure_env_heap_slot, code_index });
                }
                MakeFunc { num_instructions } => {
                    self.subject1.push(Val::ExplicitFunc {
                        closure_env_heap_slot: usize::MAX,
                        code_index: ip + 1,
                    });
                    ip += num_instructions;
                }
                AllocLocals { num_locals } => {
                    ip += 1;
                    // TODO is 0 the right thing to fill with here?
                    self.locals_stack.resize(self.locals_stack.len() + num_locals, Val::Int(0));
                }
                Return => {
                    let frame = self.stack_frames.pop().unwrap();

                    // We hate this! But drop_val doesn't access locals, so it's fine.
                    // TODO make drop_val + decrement_rc methods on heap
                    let mut locals = mem::replace(&mut self.locals_stack, vec![]);
                    for local in locals.drain(frame.locals_start..) {
                        self.drop_val(local);
                    }
                    self.locals_stack = locals;

                    self.drop_verb();

                    if frame.ret_addr == usize::MAX { return Ok(0); }
                    else { ip = frame.ret_addr; }
                }
                PushLiteralInteger(value) => {
                    ip += 1;
                    self.subject1.push(Val::Int(value));
                }
                PushVar { src } => {
                    ip += 1;
                    let val = self.copy_from_var(src);
                    self.subject1.push(val);
                }
                PushPrimVerb { prim } => {
                    ip += 1;
                    self.verb.push(Val::PrimFunc(prim));
                }
                PushVerb { src } => {
                    ip += 1;
                    let val = self.copy_from_var(src);
                    self.verb.push(val);
                }
                // TODO call1/call2
                Call => match self.verb.last().unwrap() {
                    &Val::PrimFunc(prim) => {
                        ip += 1;
                        self.call_prim_verb(prim)?;
                    }

                    &Val::AdverbDerivedFunc { adverb, operand_heap_slot } => {
                        ip += 1;
                        self.call_adverb_derived_func(adverb, operand_heap_slot)?;
                    }

                    &Val::ExplicitFunc { closure_env_heap_slot, code_index } => {
                        self.stack_frames.push(StackFrame {
                            closure_env_heap_slot,
                            locals_start: self.locals_stack.len(),
                            ret_addr: ip + 1,
                        });

                        let x = self.subject1.pop().unwrap();
                        self.locals_stack.push(x);
                        self.locals_stack.push(
                            self.subject2.take().unwrap_or(Val::Int(0))  // TODO monads won't even mention y, but locals slot 1 is always y
                        );
                        ip = code_index;
                    }

                    constant@(&Val::Int(_) | &Val::Char(_) | &Val::Slice(_)) => {
                        ip += 1;
                        let ret = constant.clone();
                        self.drop_subject1();
                        self.drop_subject2();
                        self.drop_verb();
                        self.subject1.push(ret);
                    }

                    // This was on the verb stack, so it can't be List
                    Val::List(_) => unreachable!(),
                },
                CallPrimVerb { prim } => {
                    ip += 1;
                    self.call_prim_verb(prim)?;
                }
                Pop => {
                    ip += 1;
                    self.drop_subject1();
                    
                    // TODO find out if we discard verb here
                    if self.subject2.is_some() {
                        todo!("Pop should discard subject2")
                    }
                }
                PopVerb => {
                    ip += 1;
                    self.drop_verb();
                } 
                PopToSubject2 => {
                    ip += 1;
                    self.subject2 = self.subject1.pop();
                }
                PopToVerb => {
                    ip += 1;
                    self.verb.push(self.subject1.pop().unwrap());
                }
                MoveVerbToSubject1 => {
                    ip += 1;
                    self.subject1.push(self.verb.pop().unwrap());
                }
                StoreTo { dst } => {
                    ip += 1;
                    self.store(dst, self.subject1.last().unwrap().clone());
                }
                StoreVerbTo { dst } => {
                    ip += 1;
                    self.store(dst, self.verb.last().unwrap().clone());
                }
                CallPrimAdverb { prim: adverb } => {
                    ip += 1;
                    let verb = self.verb.pop().unwrap();
                    let operand_heap_slot = self.put_on_heap(RcVal { rc: 1, val: verb });
                    self.verb.push(Val::AdverbDerivedFunc { adverb, operand_heap_slot });
                }
                MakeString { num_bytes } => todo!("string literals"),
                LiteralBytes { bytes } => todo!("char literals"),                
                CollectToArray { num_elems } => {
                    ip += 1;
                    let mut all_ints = true;
                    let mut all_chars = true;
                    for elem in &self.subject1[(self.subject1.len() - num_elems)..] {
                        all_ints &= matches!(elem, Val::Int(_));
                        all_chars &= matches!(elem, Val::Char(_));
                    }

                    let list = if all_ints {
                        let mut vec = Vec::with_capacity(num_elems);
                        for elem in &self.subject1[(self.subject1.len() - num_elems)..] {
                            vec.push(if let Val::Int(int) = elem { *int } else { unreachable!() });
                        }
                        List::I64s(vec)
                    } else if all_chars {
                        let mut vec = Vec::with_capacity(num_elems);
                        for elem in &self.subject1[(self.subject1.len() - num_elems)..] {
                            vec.push(if let Val::Char(ch) = elem { *ch } else { unreachable!() });
                        }
                        List::U8s(vec)
                    } else {
                        let mut vec = Vec::with_capacity(num_elems);
                        for elem in &self.subject1[(self.subject1.len() - num_elems)..] {
                            vec.push(elem.clone());
                        }
                        List::Vals(vec)
                    };
                    self.subject1.truncate(self.subject1.len() - num_elems);
                    let heap_slot = self.put_on_heap(RcVal { rc: 1, val: Val::List(ManuallyDrop::new(list)) });
                    self.subject1.push(Val::Slice(DataHandle { heap_slot, start_offset: 0, len: num_elems}));
                }
            }
        }
        Ok(0)
    }

    fn call_adverb_derived_func(&mut self, adverb: PrimAdverb, operand_heap_slot: usize) -> Result<(), String> {
        let x = self.subject1.pop().unwrap();  // TODO drop val? or auto-dropped because local?
        let y = self.subject2.take();
        let ret = self.call_prim_adverb(adverb, operand_heap_slot, x, y)?;
        self.subject1.push(ret);
        Ok(())
    }

    fn call_val_from_heap_slot(&mut self, heap_slot: usize, x: Val, y: Option<Val>) -> Result<Val, String> {
        let result = match &self.heap[heap_slot].val {
            val@&Val::ExplicitFunc { closure_env_heap_slot, code_index } => {
                self.verb.push(val.clone());
                self.increment_rc(closure_env_heap_slot);
                self.stack_frames.push(StackFrame {
                    closure_env_heap_slot,
                    locals_start: self.locals_stack.len(),
                    ret_addr: usize::MAX,
                });
                self.locals_stack.push(x);
                self.locals_stack.push(y.unwrap_or(Val::Int(0)));
                self.execute(code_index)?;  // TODO throwing away exit status?
                self.subject1.pop().unwrap()
            }
            val@&Val::AdverbDerivedFunc {
                adverb: inner_adverb,
                operand_heap_slot: inner_operand_heap_slot
            } => {
                self.verb.push(val.clone());  // TODO necessary?
                self.increment_rc(inner_operand_heap_slot);
                self.call_prim_adverb(inner_adverb, inner_operand_heap_slot, x, y)?
            }
            &Val::PrimFunc(prim) => if let Some(y) = y {
                self.call_prim_dyad(prim, x, y)?
            } else {
                self.call_prim_monad(prim, x)?
            },
            &Val::Int(int) => Val::Int(int),
            &Val::Char(ch) => Val::Char(ch),
            Val::Slice(handle) => {
                let handle = handle.clone();
                self.increment_rc(handle.heap_slot);
                Val::Slice(handle)
            }
            Val::List(list) => {
                let len = list.len();
                self.increment_rc(heap_slot);
                Val::Slice(DataHandle { heap_slot, start_offset: 0, len })
            }
        };
        Ok(result)
    }

    fn call_prim_adverb(&mut self,
                        adverb: PrimAdverb,
                        operand_heap_slot: usize,
                        x: Val,
                        y: Option<Val>) -> Result<Val, String> {
        todo!()
        // use PrimAdverb::*;
        // let result = match adverb {
        //     Dot => self.call_val_from_heap_slot(operand_heap_slot, x, y)?,
        //     SingleQuote => match x {
        //         Val::Slice(handle) => match y {
        //             Some(Val::Slice(y_handle)) => match self.get_slice(&handle) {
        //                 Slice::I64s(i64s) => self.slice_of(
        //                     i64s.map(|int| self.call_val_from_heap_slot(operand_heap_slot, Val::Int(unsafe{*int}))?)
        //                         .collect::<Result<_, _>>()?
        //                 ),
        //                 Slice::U8s(u8s) => self.slice_of(
        //                     u8s.map(|ch| self.call_val_from_heap_slot_monad(operand_heap_slot, Val::Char(unsafe{*ch}))?)
        //                         .collect::<Result<_, _>>()?
        //                 ),
        //                 Slice::Vals(vals) => self.slice_of(
        //                     vals.map(|val| self.call_val_from_heap_slot_monad(operand_heap_slot, unsafe{&*val}.clone()))?
        //                         .collect::<Result<_, _>>()?
        //                 ),
        //             }
        //         }
        //         Val::List(_) => unreachable!(),
        //         _ => self.call_val_from_heap_slot_monad(operand_heap_slot, x)?,
        //     },
        //     Tilde => {
        //         let y = x.clone();
        //         if let Some(heap_slot) = self.get_heap_slot(&y) {
        //             self.increment_rc(heap_slot);
        //         }
        //         self.call_val_from_heap_slot_dyad(operand_heap_slot, x, y)?
        //     }
        // };
        // Ok(result)
    }

    // Never returns Val::List
    fn copy_from_heap(&mut self, heap_slot: usize) -> Val {
        match &self.heap[heap_slot].val {
            &Val::Int(int) => Val::Int(int),
            &Val::Char(ch) => Val::Char(ch),
            Val::Slice(handle) => {
                let handle = handle.clone();
                self.increment_rc(handle.heap_slot);
                Val::Slice(handle)
            }
            Val::List(list) => {
                let len = list.len();
                self.increment_rc(heap_slot);
                Val::Slice(DataHandle { heap_slot, start_offset: 0, len })
            }
            &Val::PrimFunc(prim) => Val::PrimFunc(prim),
            &Val::ExplicitFunc { closure_env_heap_slot, code_index } => {
                self.increment_rc(closure_env_heap_slot);
                Val::ExplicitFunc { closure_env_heap_slot, code_index }
            }
            &Val::AdverbDerivedFunc { adverb, operand_heap_slot } => {
                self.increment_rc(operand_heap_slot);
                Val::AdverbDerivedFunc { adverb, operand_heap_slot }
            }
        }
    }

    fn copy_from_var(&mut self, var: Var) -> Val {
        // TODO do all local/closure vars point to non-lists (instead to slices)?
        let frame = self.stack_frames.last().unwrap();
        let val = match var.place {
            Place::Local => self.locals_stack[frame.locals_start + var.slot].clone(),
            Place::ClosureEnv => match &self.heap[frame.closure_env_heap_slot].val {
                Val::List(list) => match &**list {
                    List::Vals(vals) => vals[var.slot].clone(),
                    // TODO consolidate closure envs of all-matching types to
                    // e.g. Int64s instead of Vals.
                    _ => unreachable!(),
                }
                // TODO if a closure env consists of one value, it's fine not to
                // use a whole list.
                _ => unreachable!(),  
            }
        };

        if let Some(heap_slot) = self.get_heap_slot(&val) {
            self.increment_rc(heap_slot);
        }
        val        
    }

    fn call_prim_adverb_dyad(&mut self, adverb: PrimAdverb, operand_heap_slot: usize, x: Val, y: Val) -> Result<Val, String> {
        todo!()
    }

    fn call_prim_verb(&mut self, prim: PrimVerb) -> Result<(), String> {
        let x = self.subject1.pop().unwrap();  // TODO drop val?
        let ret = if let Some(y) = self.subject2.take() {  // TODO drop val?
            self.call_prim_dyad(prim, x, y)?
        } else {
            self.call_prim_monad(prim, x)?
        };
        self.subject1.push(ret);
        Ok(())
    }

    fn call_prim_monad(&mut self, v: PrimVerb, x: Val) -> Result<Val, String> {
        use PrimVerb::*;
        match v {
            Print => { self.print_val(&x)?; Ok(x) }
            _ => todo!("{x:?} {v:?}")
        }
    }

    fn call_prim_dyad(&mut self, v: PrimVerb, x: Val, y: Val) -> Result<Val, String> {
        use PrimVerb::*;
        match v {
            Plus => self.add_vals(&x, &y),  // TODO take x,y by ref above?
            Snoc => todo!(),
            _ => todo!("{x:?} {v:?} {y:?}"),
        }
    }
    
    // TODO output formatting
    fn print_val(&self, x: &Val) -> Result<(), String> {
        use std::ops::Bound;
        fn print_slice(mem: &Mem, list: &List, bounds: (Bound<usize>, Bound<usize>)) -> Result<(), String> {
            match list {
                List::I64s(ints) => println!("{:?}", &ints[bounds]),
                List::U8s(chars) => match std::str::from_utf8(&chars[bounds]) {  // TODO unicode
                    Ok(s) => println!("{s}"),
                    Err(err) => return Err(err.to_string()),
                }
                List::Vals(vals) => {
                    println!("[");
                    for val in &vals[bounds] {
                        mem.print_val(val)?;
                    }
                    println!("]");
                }
            }
            Ok(())
        }
        
        match x {
            Val::Int(i) => println!("{i}"),
            Val::Char(c) => println!("{c}"),
            Val::PrimFunc(prim) => println!("{prim}"),
            Val::Slice(DataHandle { heap_slot, start_offset, len }) => match &self.heap[*heap_slot].val {
                Val::List(list) => print_slice(self, list, (Bound::Included(*start_offset), Bound::Excluded(*start_offset + *len)))?,
                _ => unreachable!(),
            },
            Val::List(list) => print_slice(self, &*list, (Bound::Unbounded, Bound::Unbounded))?,
            Val::AdverbDerivedFunc { adverb, operand_heap_slot } => {
                todo!("implement adverb-derived verb printing")
            }
            Val::ExplicitFunc { closure_env_heap_slot, code_index } => {
                todo!("implement explicit func printing")  // map code index -> tokens?
            }
        }
        
        Ok(())
    }

    // CAREFUL: As long as the returned Slice might be accessed, do not
    // invalidate its pointers (e.g. by resizing the vec in `handle`'s heap
    // slot).
    fn get_slice(&self, handle: &DataHandle) -> Slice {
        let range = handle.start_offset .. handle.start_offset + handle.len;
        match &self.heap[handle.heap_slot].val {
            Val::List(list) => match &**list {
                List::I64s(vec) => Slice::I64s(PtrRange::from_slice(&vec[range])),
                List::U8s(vec) => Slice::U8s(PtrRange::from_slice(&vec[range])),
                List::Vals(vec) => Slice::Vals(PtrRange::from_slice(&vec[range])),
            }
            _ => unreachable!(),  // Slices should only point to Lists
        }
    }
    
    fn slice_of(&mut self, list: List) -> Val {
        let len = list.len();
        let heap_slot = self.put_on_heap(RcVal {
            rc: 1,
            val: Val::List(ManuallyDrop::new(list)),
        });
        Val::Slice(DataHandle { heap_slot, start_offset: 0, len })
    }

    // `handle` must point to a List::Vals.
    fn irrefutable_vals(&self, handle: &DataHandle) -> &[Val] {
        irrefutable!(
            &self.heap[handle.heap_slot].val,
            Val::List(list) => irrefutable!(&**list, List::Vals(vals) => vals)
        )
    }

    fn add_vals(&self, x: &Val, y: &Val) -> Result<Val, String> {
        use Val::*;
        use crate::vm::List;
        let result = match (x, y) {
            (Int(i), Int(j)) => Int(i + j),
            (Int(i), Char(c)) | (Char(c), Int(i)) => Char((*c as i64 + i) as u8),
            (Slice(handle), Int(i)) | (Int(i), Slice(handle)) => {
                let list_result = irrefutable!(
                    &self.heap[handle.heap_slot].val,
                    List(list) => match &**list {
                        List::I64s(ints) => List::I64s(ints.iter().map(|int| *int + i).collect()),
                        List::U8s(chars) => List::U8s(chars.iter().map(|ch| (*ch as i64 + *i) as u8).collect()),
                        List::Vals(vals) => List::Vals(
                            vals.iter().map(|val| self.add_vals(val, &Int(*i))).collect::<Result<_, _>>()?
                        ),
                    }
                );
                List(ManuallyDrop::new(list_result))
            }
            (Slice(handle), Char(ch)) | (Char(ch), Slice(handle)) => {
                let list_result = irrefutable!(
                    &self.heap[handle.heap_slot].val,
                    List(list) => match &**list {
                        List::I64s(ints) => List::U8s(ints.iter().map(|i| (*ch as i64 + *i) as u8).collect()),
                        List::U8s(_) => return Err(format!("Can't add char to char")),
                        List::Vals(vals) => List::Vals(
                            vals.iter().map(|val| self.add_vals(val, &Char(*ch))).collect::<Result<_, _>>()?
                        ),
                    }
                );
                List(ManuallyDrop::new(list_result))
            }
            (Slice(xh), Slice(yh)) => {
                match_length(xh.len, yh.len)?;
                let lists = irrefutable!((&self.heap[xh.heap_slot].val, &self.heap[yh.heap_slot].val),
                                         (List(x), List(y)) => (&**x, &**y));
                let list_result = match lists {
                    (List::I64s(xints), List::I64s(yints)) => List::I64s(
                        xints.iter().zip(yints.iter()).map(|(i, j)| *i + *j).collect()
                    ),
                    (List::U8s(chars), List::I64s(ints)) |
                    (List::I64s(ints), List::U8s(chars)) => List::U8s(
                        chars.iter().zip(ints.iter()).map(|(ch, i)| (*ch as i64 + i) as u8).collect()
                    ),
                    (List::Vals(vals), List::I64s(ints)) |
                    (List::I64s(ints), List::Vals(vals)) => List::Vals(
                        vals.iter().zip(ints.iter())
                            .map(|(val, int)| self.add_vals(val, &Val::Int(*int)))
                            .collect::<Result<_, _>>()?
                    ),
                    (List::Vals(vals), List::U8s(chars)) |
                    (List::U8s(chars), List::Vals(vals)) => List::Vals(
                        vals.iter().zip(chars.iter())
                            .map(|(val, ch)| self.add_vals(val, &Val::Char(*ch)))
                            .collect::<Result<_, _>>()?
                    ),
                    (List::Vals(vals1), List::Vals(vals2)) => List::Vals(
                        vals1.iter().zip(vals2.iter())
                            .map(|(val1, val2)| self.add_vals(val1, val2))
                            .collect::<Result<_, _>>()?
                    ),
                    _ => return Err(format!("Can't add {x:?} and {y:?}")),
                };
                List(ManuallyDrop::new(list_result))
            }
            _ => return Err(format!("Can't add {x:?} and {y:?}")),
        };

        Ok(result)
    }

    // Slices and lists index normally, and atoms cycle.
    // fn get_nth_item(&self, x: &Val, i: usize) -> Option<&Val> {
    //     use Val::*;
    //     match x {
    //         Int(_) |
    //         Char(_) |
    //         PrimFunc(_) |
    //         ExplicitFunc {..} |
    //         AdverbDerivedFunc {..} => Some(x),
    //         List(list) => index_list_as_val(list.deref, i),
    //         Slice(handle) => irrefutable!(&self.heap[handle.heap_slot],
    //                                       List(list) => index_list_as_val(list.deref(), handle.start_offset + i)),
    //     }
    // }
}

// TODO
// fn as_vals<'a>(list: &'a List) -> impl Iterator<Item=Cow<'a, Val>> {
//     match list {
//         List::I64s(x) => x.iter().map(|i| Cow::Owned(Val::Int(*i))),
//         List::U8s(x) => x.iter().map(|ch| Cow::Owned(Val::Char(*ch))),
//         List::Vals(x) => x.iter().map(|val| Cow::Borrowed(val)),
//     }
// }

// fn index_list_as_val(list: &List, i: usize) -> Option<&Val> {
//     match list {
//         List::I64s(ints) => ints.get(i).copied().map(Val::Int),
//         List::U8s(chars) => Val::Char(chars.get(i)?),
//         List::Vals(vals) => vals.get(i),
//     }        
// }


fn match_length(xlen: usize, ylen: usize) -> Result<(), String> {
    if xlen == ylen { return Ok(()); }
    // TODO include name/position of verb
    Err(format!("Length mismatch: {xlen} vs {ylen}"))
}
