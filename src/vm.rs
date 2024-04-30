use crate::ptr_range::*;
use crate::bytecode::*;
use crate::lex::*;

enum Datatype { Int, Char }

/*
problem: some vals need to go on heap for refs



o1: put everything on heap, just use heap indices as values
  pro
o2: move val to heap when ref is taken
  [1;2;3]
o3: arrays (ie dyn sized objects) always on heap, others off or on (default off, values move on when ref taken)
actually, maybe o2 since arrays aren't *actually* expensive to move
[1;2;3]->a  \ a is Val:Ints(Vec)
a#2  \ Val:Ints moves to heap w/ rc 2, a becomes Val:Slice(0,len,slot), result is Val:Slice(0,2,slot)

a: retaking reference to array x

[1;2;3]->a
a#2->b
c:ref a

a:points to array
b:slice of a
c: 

*/

// TODO Ref
#[derive(Debug, Clone)]
enum Val {
    Int(i64),  // TODO TwoInts, ThreeInts
    Char(u8),

    List(DataHandle),

    // Item type: Val (closure env)
    ExplicitFunc {
        // We always use the whole closure environment, so no offset
        closure_env_heap_slot: usize,
        // The function's first instruction (points after MakeFunc)
        code_index: usize,
    },

    // TODO switch this out for the non-token type (some primitives won't have
    // token representations)
    PrimFunc(PrimVerb),

    // Item type: Val (always exactly one)
    // TODO store primitive/explicit operands inline
    AdverbDerivedFunc {
        adverb: PrimAdverb,
        operand_heap_slot: usize,
    },
}

#[derive(Debug, Clone)]
struct DataHandle {
    heap_slot: usize,
    start_offset: usize,
    len: usize,
}

// Assertion
const _: [(); std::mem::align_of::<Val>()] = [(); std::mem::align_of::<usize>()];

// TODO Refs
#[derive(Debug)]
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
struct Data {
    // TODO make atomic
    rc: usize,
    list: List,
}

impl Data {
    fn new_null() -> Self {
        Self { rc: 0, list: List::I64s(vec![]) }  // Arbitrarily choose I64s
    }
}

#[derive(Debug)]
struct StackFrame {
    closure_env_heap_slot: usize,  // Heap slot. TODO replace with NonNull<Val>?
    locals_start: usize,  // Index into locals_stack. Local slots are offsets from this index.
    ret_addr: usize,  // Code index after Call
}

pub struct Mem {
    code: Vec<Instr>,

    subject1: Vec<Val>,
    verb: Vec<Val>,
    subject2: Option<Val>,
    
    // Stack of local scopes
    locals_stack: Vec<Val>,
    stack_frames: Vec<StackFrame>,

    heap: Vec<Data>,
    free_heap_slots: Vec<usize>,  // TODO scan heap for rc==0 or store bitvec aside instead?
}

impl Mem {
    pub fn new(initial_heap_slots: usize) -> Self {
        Self {
            code: vec![],
            subject1: vec![],
            verb: vec![],
            subject2: None,
            locals_stack: vec![],  // TODO stdlib?
            stack_frames: vec![StackFrame {
                closure_env_heap_slot: usize::MAX,
                locals_start: 0,
                ret_addr: usize::MAX
            }],
            heap: {
                let mut heap = vec![];
                heap.resize_with(initial_heap_slots, Data::new_null);
                heap
            },
            free_heap_slots: (0..initial_heap_slots).rev().collect(),
        }
    }

    fn load(&self, src: Var) -> &Val {
        let frame = self.stack_frames.last().unwrap();
        match src.place {
            Place::Local => &self.locals_stack[frame.locals_start + src.slot],
            Place::ClosureEnv => match &self.heap[frame.closure_env_heap_slot].list {
                List::Vals(vec) => &vec[src.slot],
                _ => unreachable!(),
            }
        }
    }

    fn store(&mut self, dst: Var, val: Val) {
        for heap_slot in self.get_heap_slot(&val) { self.increment_rc(heap_slot); }
        let frame = self.stack_frames.last().unwrap();
        match dst.place {
            Place::Local => {
                let absolute_slot = frame.locals_start + dst.slot;
                if absolute_slot >= self.locals_stack.len() {
                    self.locals_stack.resize(absolute_slot + 1, Val::Int(0)); // TODO is Int(0) right?
                }
                self.locals_stack[absolute_slot] = val;
            }
            Place::ClosureEnv => match &mut self.heap[frame.closure_env_heap_slot].list {
                List::Vals(vec) => vec[dst.slot] = val,
                _ => unreachable!(),
            }
        }
    }

    fn get_heap_slot(&self, val: &Val) -> Option<usize> {
        match val {
            Val::List(DataHandle { heap_slot, .. }) |
            Val::ExplicitFunc { closure_env_heap_slot: heap_slot, .. } |
            Val::AdverbDerivedFunc { operand_heap_slot: heap_slot, .. } => Some(*heap_slot),
            Val::Int(_) | Val::Char(_) | Val::PrimFunc(_) => None,
        }
    }

    fn increment_rc(&mut self, heap_slot: usize) {
        if heap_slot == usize::MAX { return; }
        self.heap[heap_slot].rc += 1;
    }
    
    fn decrement_rc(&mut self, heap_slot: usize) {
        if heap_slot == usize::MAX { return; }
        let data = &mut self.heap[heap_slot];
        data.rc -= 1;
        if data.rc == 0 {
            data.list = List::I64s(vec![]);  // Drop old vec
            self.free_heap_slots.push(heap_slot);
        }
    }

    fn move_to_heap(&mut self, data: Data) -> usize {
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

    fn discard_subject1(&mut self) {
        for val in self.subject1.pop() {
            for heap_slot in self.get_heap_slot(&val) {
                self.decrement_rc(heap_slot);
            }
        }
    }

    fn discard_verb(&mut self) {
        for val in self.verb.pop() {
            for heap_slot in self.get_heap_slot(&val) {
                self.decrement_rc(heap_slot);
            }
        }
    }

    fn discard_subject2(&mut self) {
        for val in self.subject2.take() {
            for heap_slot in self.get_heap_slot(&val) {
                self.decrement_rc(heap_slot);
            }
        }
    }

    pub fn execute(&mut self, code: &[Instr]) -> Result<u8, String> {
        use Instr::*;

        let mut ip: usize = 0;
        while ip < code.len() {
            dbg!(&code[ip], &self.stack_frames, &self.locals_stack);
            println!();
            match code[ip] {
                Nop => ip += 1,
                Halt { exit_status } => return Ok(exit_status),
                // Creates a closure object consisting of the following `num_closure_vars` PushVar instructions. Followed by EnterFunc and the function's body.
                MakeClosure { num_closure_vars } => {
                    ip += 1;
                    let num_instructions = match &code[ip] {
                        MakeFunc { num_instructions } => { ip += 1; num_instructions }
                        bad => panic!("Malformed code at ip {ip}: expected MakeFunc after MakeClosure, but found {bad:?}"),
                    };

                    let code_index = ip;  // First instruction of function body

                    ip += num_instructions;
                    dbg!(&"After instrs", &code[ip]);
                    let closure_env_heap_slot = if num_closure_vars == 0 {
                        usize::MAX
                    } else {
                        let mut closure_data = Vec::with_capacity(num_closure_vars);
                        for _ in 0..num_closure_vars {
                            if let PushVar { src } = code[ip] {
                                ip += 1;
                                let val = self.load(src).clone();
                                if let Some(heap_slot) = self.get_heap_slot(&val) {
                                    self.increment_rc(heap_slot);
                                }
                                closure_data.push(val);
                            } else {
                                panic!("Malformed code at ip {ip}: expected PushVar after MakeClosure, but found {:?}", code[ip]);
                            }
                        }
                        self.move_to_heap(Data { rc: 1, list: List::Vals(closure_data) })
                    };
                    dbg!(closure_env_heap_slot);
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
                    for local_slot in frame.locals_start .. self.locals_stack.len() {
                        let local_val = &self.locals_stack[local_slot];
                        for heap_slot in self.get_heap_slot(local_val) {
                            self.decrement_rc(heap_slot);
                        }
                    }
                    self.locals_stack.truncate(frame.locals_start);  // TODO necessary?
                    ip = frame.ret_addr;
                }
                PushLiteralInteger(value) => {
                    ip += 1;
                    self.subject1.push(Val::Int(value));
                }
                PushVar { src } => {
                    ip += 1;
                    let val = self.load(src).clone();
                    for heap_slot in self.get_heap_slot(&val) {
                        self.increment_rc(heap_slot);
                    }
                    self.subject1.push(val);
                }
                PushPrimVerb { prim } => {
                    ip += 1;
                    self.verb.push(Val::PrimFunc(prim));
                }
                PushVerb { src } => {
                    ip += 1;
                    let val = self.load(src).clone();
                    for heap_slot in self.get_heap_slot(&val) {
                        self.increment_rc(heap_slot);
                    }
                    self.verb.push(val);
                }
                // TODO call1/call2
                Call => match self.verb.pop().unwrap() {
                    Val::ExplicitFunc { closure_env_heap_slot, code_index } => {
                        self.stack_frames.push(StackFrame {
                            closure_env_heap_slot,
                            locals_start: self.locals_stack.len(),
                            ret_addr: ip + 1,
                        });

                        let x = self.subject1.pop().unwrap();
                        self.locals_stack.push(x);
                        self.locals_stack.push(
                            self.subject2.take().unwrap_or(Val::Int(0))  // TODO
                        );
                        ip = code_index;
                    }

                    Val::PrimFunc(prim) => {
                        ip += 1;
                        self.call_prim_verb(prim)?;
                    }
                    _ => todo!("call adverb derived func")
                }
                CallPrimVerb { prim } => {
                    ip += 1;
                    self.call_prim_verb(prim)?;
                }
                Pop => {
                    ip += 1;
                    self.discard_subject1();
                    
                    // TODO find out if we discard verb here
                    if self.subject2.is_some() {
                        todo!("Pop should discard subject2")
                    }
                }
                PopVerb => {
                    ip += 1;
                    self.discard_verb();
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
                    match adverb {
                        _ => todo!("implement adverbs"),
                    }
                }
                MakeString { num_bytes } => todo!("string literals"),
                LiteralBytes { bytes } => todo!("char literals"),                
                MakeArray { num_elems } => {
                    let heap_slot = self.move_to_heap(Data {
                        rc: 1,
                        list: List::I64s(Vec::with_capacity(num_elems)),
                    });
                    self.subject1.push(Val::List(DataHandle { heap_slot, start_offset: 0, len: 0}));
                }
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
                    
                    let heap_slot = self.move_to_heap(Data { rc: 1, list });
                    self.subject1.push(Val::List(DataHandle { heap_slot, start_offset: 0, len: num_elems}));
                }
            }
        }
        Ok(0)
    }

    fn call_prim_verb(&mut self, prim: PrimVerb) -> Result<(), String> {
        let x = self.subject1.pop().unwrap();
        let ret = if let Some(y) = self.subject2.take() {
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
            Print => self.print_val(x),
            _ => todo!("{x:?} {v:?}")
        }
    }

    fn call_prim_dyad(&mut self, v: PrimVerb, x: Val, y: Val) -> Result<Val, String> {
        use PrimVerb::*;
        match v {
            Plus => self.add_val_val(x, y),
            Snoc => todo!(),
            _ => todo!("{x:?} {v:?} {y:?}"),
        }
    }
    
    // TODO output formatting
    fn print_val(&self, x: Val) -> Result<Val, String> {
        match x {
            Val::Int(i) => println!("{i}"),
            Val::Char(c) => println!("{c}"),
            Val::PrimFunc(prim) => println!("{prim}"),
            Val::List(DataHandle { heap_slot, start_offset, len }) => {
                let slice = start_offset..(start_offset+len);
                match &self.heap[heap_slot].list {
                    List::I64s(ints) => println!("{:?}", &ints[slice]),
                    List::U8s(chars) => match std::str::from_utf8(&chars[slice]) {  // TODO unicode
                        Ok(s) => println!("{s}"),
                        Err(err) => return Err(err.to_string()),
                    }
                    List::Vals(vals) => {
                        println!("[");
                        for val in &vals[slice] {
                            self.print_val(val.clone())?;
                        }
                        println!("]");
                    }
                }
            }
            Val::AdverbDerivedFunc { adverb, operand_heap_slot } => {
                todo!("implement adverb-derived verb printing")
            }
            Val::ExplicitFunc { closure_env_heap_slot, code_index } => {
                todo!("implement explicit func printing")  // map code index -> tokens?
            }
        }
        
        Ok(x)
    }

    fn add_val_val(&mut self, x: Val, y: Val) -> Result<Val, String> {
        match x {
            Val::Int(x) => self.add_int_val(x, y),
            Val::Char(x) => self.add_char_val(x, y),
            Val::List(x_handle) => match self.get_slice(&x_handle) {
                Slice::I64s(x) => self.add_ints_val(x, y),
                Slice::U8s(x) => self.add_chars_val(x, y),
                Slice::Vals(x) => self.add_vals_val(x, y),
            }
            _ => todo!("add {:?}, {:?}", x, y),
        }
    }

    fn add_ints_val(&mut self, x: PtrRange<i64>, y: Val) -> Result<Val, String> {
        match y {
            Val::Int(y) => Ok(self.add_int_ints(y, x)),
            Val::List(y_handle) => match self.get_slice(&y_handle) {
                Slice::I64s(y) => self.add_ints_ints(x, y),
                Slice::U8s(y) => self.add_ints_chars(x, y),
                Slice::Vals(y) => self.add_ints_vals(x, y),
            }
            _ => todo!("add {:?}, {:?}", x, y),
        }
    }

    fn add_chars_val(&mut self, x: PtrRange<u8>, y: Val) -> Result<Val, String> {
        match y {
            Val::Int(y) => Ok(self.add_int_chars(y, x)),
            Val::List(y_handle) => match self.get_slice(&y_handle) {
                Slice::I64s(y) => self.add_ints_chars(y, x),
                Slice::U8s(y) => Err(format!("Can't add chars to chars")),
                Slice::Vals(y) => self.add_chars_vals(x, y),
            }
            _ => todo!("add {:?}, {:?}", x, y),
        }
    }

    fn add_vals_val(&mut self, x: PtrRange<Val>, y: Val) -> Result<Val, String> {
        match y {
            Val::Int(y) => self.add_int_vals(y, x),
            Val::List(y_handle) => match self.get_slice(&y_handle) {
                Slice::I64s(y) => self.add_ints_vals(y, x),
                Slice::U8s(y) => self.add_chars_vals(y, x),
                Slice::Vals(y) => self.add_vals_vals(x, y),
            }
            _ => todo!("add {:?}, {:?}", x, y),
        }
    }

    fn add_int_val(&mut self, x: i64, y: Val) -> Result<Val, String> {
        match y {
            Val::Int(y) => Ok(Val::Int(x + y)),
            Val::Char(y) => Ok(Val::Char(x as u8 + y)),  // TODO overflow
            Val::List(y_handle) => match self.get_slice(&y_handle) {
                Slice::I64s(y) => Ok(self.add_int_ints(x, y)),
                Slice::U8s(y) => Ok(self.add_int_chars(x, y)),
                Slice::Vals(y) => self.add_int_vals(x, y),
            }
            _ => todo!("add {:?}, {:?}", x, y),
        }
    }

    fn add_char_val(&mut self, x: u8, y: Val) -> Result<Val, String> {
        match y {
            Val::Int(y) => Ok(Val::Char(x + y as u8)),
            Val::Char(y) => Err(format!("Can't add char to char")),
            Val::List(y_handle) => match self.get_slice(&y_handle) {
                Slice::I64s(y) => Ok(self.add_char_ints(x, y)),
                Slice::U8s(y) => Err(format!("Can't add char to char")),
                Slice::Vals(y) => self.add_char_vals(x, y),
            }
            _ => todo!("add {:?}, {:?}", x, y),
        }
    }

    fn add_int_ints(&mut self, x: i64, y: PtrRange<i64>) -> Val {
        self.list_val(y.map(|item| unsafe{*item} + x).collect())
    }

    fn add_int_chars(&mut self, x: i64, y: PtrRange<u8>) -> Val {
        self.list_val(y.map(|item| unsafe{*item} + x as u8).collect())
    }

    fn add_int_vals(&mut self, x: i64, y: PtrRange<Val>) -> Result<Val, String> {
        let list = y.map(|val| self.add_int_val(x, unsafe{&*val}.clone())).collect::<Result<_, _>>()?;
        Ok(self.list_val(list))
    }

    fn add_char_ints(&mut self, x: u8, y: PtrRange<i64>) -> Val {
        self.list_val(y.map(|y| y as u8 + x).collect())
    }

    fn add_char_vals(&mut self, x: u8, y: PtrRange<Val>) -> Result<Val, String> {
        let list = y.map(|val| self.add_char_val(x, unsafe{&*val}.clone())).collect::<Result<_, _>>()?;
        Ok(self.list_val(list))
    }

    fn add_ints_ints(&mut self, x: PtrRange<i64>, y: PtrRange<i64>) -> Result<Val, String> {
        match_length(x.len(), y.len())?;
        let list = x.zip(y).map(|(x, y)| unsafe{*x} + unsafe{*y}).collect();
        Ok(self.list_val(list))
    }

    fn add_ints_chars(&mut self, x: PtrRange<i64>, y: PtrRange<u8>) -> Result<Val, String> {
        match_length(x.len(), y.len())?;
        let list = x.zip(y).map(|(x, y)| unsafe{*x} as u8 + unsafe{*y}).collect();
        Ok(self.list_val(list))    
    }

    fn add_ints_vals(&mut self, x: PtrRange<i64>, y: PtrRange<Val>) -> Result<Val, String> {
        match_length(x.len(), y.len())?;
        let list = x.zip(y)
            .map(|(x, val)| self.add_int_val(unsafe{*x}, unsafe{&*val}.clone()))
            .collect::<Result<_, _>>()?;
        Ok(self.list_val(list))
    }

    fn add_chars_vals(&mut self, x: PtrRange<u8>, y: PtrRange<Val>) -> Result<Val, String> {
        match_length(x.len(), y.len())?;
        let list = x.zip(y)
            .map(|(x, val)| self.add_char_val(unsafe{*x}, unsafe{&*val}.clone()))
            .collect::<Result<_, _>>()?;
        Ok(self.list_val(list))
    }

    fn add_vals_vals(&mut self, x: PtrRange<Val>, y: PtrRange<Val>) -> Result<Val, String> {
        match_length(x.len(), y.len())?;
        let list = x.zip(y)
            .map(|(x_val, y_val)| self.add_val_val(unsafe{&*x_val}.clone(), unsafe{&*y_val}.clone()))
            .collect::<Result<List, String>>()?;
        Ok(self.list_val(list))
    }

    // CAREFUL: As long as the returned Slice might be accessed, do not
    // invalidate its pointers (e.g. by resizing the vec in `handle`'s heap
    // slot).
    fn get_slice(&self, handle: &DataHandle) -> Slice {
      let range = handle.start_offset .. handle.start_offset + handle.len;
      match &self.heap[handle.heap_slot].list {
          List::I64s(vec) => Slice::I64s(PtrRange::from_slice(&vec[range])),
          List::U8s(vec) => Slice::U8s(PtrRange::from_slice(&vec[range])),
          List::Vals(vec) => Slice::Vals(PtrRange::from_slice(&vec[range])),
      }
    }
    
    fn list_val(&mut self, list: List) -> Val {
        let len = list.len();
        let heap_slot = self.move_to_heap(Data { rc: 1, list });
        Val::List(DataHandle { heap_slot, start_offset: 0, len })
    }
}

fn match_length(xlen: usize, ylen: usize) -> Result<(), String> {
    if xlen == ylen { return Ok(()); }
    // TODO include name/position of verb
    Err(format!("Length mismatch: {xlen} vs {ylen}"))
}
