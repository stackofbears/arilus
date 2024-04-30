#[derive(Debug, Clone)]  // This Debug impl isn't very interesting
pub struct PtrRange<T>(std::ops::Range<*const T>);

impl<T> PtrRange<T> {
    pub fn from_slice(slice: &[T]) -> Self {
        Self(slice.as_ptr_range())
    }

    #[inline]
    pub fn start(&self) -> *const T { self.0.start }

    // Returns a pointer one byte past the end of the last valid element.
    #[inline]
    pub fn end(&self) -> *const T { self.0.end }

    // PtrRange 
    pub fn len(&self) -> usize {
        // SAFETY: These pointers are guaranteed to come from a slice.
        // TODO: sub_ptr would be better here, but it's only in nightly right now.
        return unsafe { self.0.end.offset_from(self.0.start) } as usize
    }
}

impl<T> Iterator for PtrRange<T> where {
    type Item = *const T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.start < self.0.end {
            let ptr = self.0.start;
            // SAFETY: start < end, which is guaranteed one byte past a valid
            // object.
            self.0.start = unsafe { self.start().add(1) };
            Some(ptr)
        } else {
            None
        }
    }
}

