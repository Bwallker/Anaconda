use super::bytecode::{AnacondaValue, StackFrame, Type};
use crate::util::FastMap;
use std::{
    alloc::Layout,
    hash::{Hash, Hasher},
    ptr::NonNull, intrinsics::transmute,
};
#[derive(Copy, Clone)]
struct GarbageCollectorItem {
    drop_func: unsafe fn(*mut u8),
    size: usize,
    allignment: usize,
    value: *mut u8,
}

impl GarbageCollectorItem {
    fn new<T>(value: *mut T) -> Self {
        let value = value as *mut u8;
        let size = std::mem::size_of::<T>();
        let allignment = std::mem::align_of::<T>();
        let drop_func = unsafe{transmute::<unsafe fn(*mut T), unsafe fn(*mut u8)>(std::ptr::drop_in_place::<T>)};
        Self {
            drop_func,
            size,
            value,
            allignment,
        }
    }
}

impl Hash for GarbageCollectorItem {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl PartialEq for GarbageCollectorItem {
    fn eq(&self, other: &GarbageCollectorItem) -> bool {
        self.value == other.value
    }
}

impl Eq for GarbageCollectorItem {}
pub(crate) struct GarbageCollector {
    allocated_objects: FastMap<GarbageCollectorItem, bool>,
}

impl GarbageCollector {
    pub(crate) fn new() -> Self {
        Self {
            allocated_objects: FastMap::default(),
        }
    }

    pub(crate) fn insert<T>(&mut self, value: *mut T, is_used: bool) {
        self.allocated_objects
            .insert(GarbageCollectorItem::new(value), is_used);
    }

    pub(crate) fn collect_garbage<'a>(
        &'_ mut self,
        stack: &Vec<AnacondaValue<'a>>,
        stack_frames: &Vec<StackFrame<'a>>,
    ) {
        for is_current_used in self.allocated_objects.values_mut() {
            *is_current_used = false;
        }
        for stack_frame in stack_frames {
            if let StackFrame::Scope(s) = stack_frame {
                for var in s.variables.values() {
                    if let AnacondaValue::Type(t) = var {
                        self.register_type(t)
                    }
                }
            }
        }
        for var in stack {
            if let AnacondaValue::Type(t) = var {
                self.register_type(t)
            }
        }

        let unused_objects: Vec<GarbageCollectorItem> = self
            .allocated_objects
            .iter()
            .filter_map(|(k, is_used)| if !is_used { Some(*k) } else { None })
            .collect();

        for unused_object in unused_objects {
            self.allocated_objects.remove(&unused_object);
            unsafe {
                let GarbageCollectorItem {
                    drop_func,
                    value,
                    size,
                    allignment,
                } = unused_object;
                drop_func(value);
                std::alloc::dealloc(value, Layout::from_size_align_unchecked(size, allignment));
            }
        }
    }

    fn register_type(&mut self, t: &GcValue<Type<'_>>) {
        if !self
            .allocated_objects
            .get(&GarbageCollectorItem::new(t.value.as_ptr()))
            .unwrap()
        {
            self.insert(t.value.as_ptr(), true);
        } else {
            return;
        }
        t.with(|v| {
            v.fields.values().for_each(|x| {
                if let AnacondaValue::Type(t) = x {
                    self.register_type(t);
                }
            });
        })
    }
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub(crate) struct GcValue<T: Sized> {
    // _not_thread_safe_marker: PhantomData<()>,
    value: NonNull<T>,
}

impl<T: Sized> GcValue<T> {
    pub(crate) fn new(value: T, gc: &mut GarbageCollector) -> Self {
        let layout = std::alloc::Layout::new::<T>();
        let val: GcValue<T> = GcValue {
            value: unsafe { NonNull::new_unchecked(std::alloc::alloc(layout) as *mut T) },
            //_not_thread_safe_marker: PhantomData,
        };
        unsafe {
            val.value.as_ptr().write(value);
        }
        let item = GarbageCollectorItem::new(val.value.as_ptr());
        gc.allocated_objects.insert(item, false);
        val
    }
    /// This function takes a closure as an argument and applies that closure to the value this GcValue wraps.
    ///
    ///
    /// The closure this function receives should propably not save the reference this method gives it as pointer, and especially not as a reference because the value might be garbage collected at any time unexpectedly.
    pub(crate) fn with<F: FnOnce(&T) -> R, R>(&self, func: F) -> R {
        func(unsafe { &*self.value.as_ptr() })
    }
    ///This function takes a closure as an argumnent and applies that closure to the value this GcValue wraps.
    ///
    /// The closure this function receives should propably not save the reference this method gives it as pointer, and especially not as a reference because the value might be garbage collected at any time unexpectedly.
    pub(crate) fn with_mut<F: FnOnce(&mut T) -> R, R>(&mut self, func: F) -> R {
        func(unsafe { &mut *self.value.as_ptr() })
    }
}
