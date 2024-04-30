use std::alloc::{
    alloc,
    dealloc,
    Layout,
};
use std::mem;
use std::ops::Drop;

// Bytes aligned to usize. Must be deallocated manually.
pub struct AlignedBytes {
    size: usize,
    bytes: *mut u8,
}

impl AlignedBytes {
    pub fn null() -> Self {
        Self { size: 0, bytes: std::ptr::null_mut() }
    }

    pub fn alloc<Elem>(num_elems: usize) -> Self {
        if num_elems == 0 || mem::size_of::<Elem>() == 0 {
            return AlignedBytes::null();
        }
        
        // TODO document safety
        let layout = unsafe {
            Layout::from_size_align_unchecked(num_elems * mem::size_of::<Elem>(),
                                              mem::align_of::<usize>())
        };
        let bytes = alloc(layout);
        std::alloc::handle_alloc_error(layout);
        Self { size: layout.size(), bytes }
    }

    // TODO is this actually unsafe?
    pub unsafe fn dealloc(&mut self) {
        // TODO have null() just set size=0 and dangling aligned pointer, remove check for null
        if self.size == 0 || self.bytes.is_null() { return; }

        let layout = unsafe {
            Layout::from_size_align_unchecked(self.size,
                                              mem::align_of::<usize>())
        };
        dealloc(self.bytes, layout);
        self.size = 0;
        self.bytes = std::ptr::null_mut();
    }

    // Safety: To dereference the returned pointer,
    //   - `Elem` must match the `Elem` passed to `alloc`.
    //   - `index` must be in 0..`num_elems` for the `num_elems` passed to `new`.
    //   - `index * size_of<Elem>()` must not exceed `isize::MAX`
    #[inline]
    pub fn get_mut<Elem>(&mut self, index: usize) -> *mut Elem {
        (self.bytes as *mut Elem).add(index)
    }

    #[inline]
    pub fn read<Elem>(&self, index: usize) -> Elem {
        *(self.bytes as *const Elem).add(index)
    }

    // TODO is it okay to return a reference here?
    #[inline]
    pub fn read_val<'a>(&'a self, index: usize) -> &'a Val {
        (self.bytes as *const Val).add(index) as &Val
    }

    #[inline]
    pub fn write<Elem>(&mut self, index: usize, val: Elem) {
        *self.get_mut::<Elem>(index) = val;
    }

    pub fn offset<Elem>(&self, offset: usize) -> AlignedBytes {
        AlignedBytes { size: self.size - offset * mem::size_of::<Elem>(),
                       bytes: (self.bytes as *mut Elem).add(offset) }
    }

    pub fn typed<'a, Elem>(&'a mut self) -> TypedBytes<'a, Elem> {
        TypedBytes
    }
}
