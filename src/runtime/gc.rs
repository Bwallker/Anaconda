use super::bytecode::{AnacondaValue, StackFrame, Type};
use crate::util::FastMap;
use std::{
    alloc::Layout,
    hash::{Hash, Hasher},
    intrinsics::transmute,
    ptr::NonNull,
};
#[derive(Copy, Clone)]
struct GarbageCollectorItem {
    /// A pointer to the drop implementation for the type stored with in this GC Item.
    drop_func: unsafe fn(*mut u8),
    /// Size of the type stored within
    size: usize,
    // Alignment of the type stored within.
    alignment: usize,
    /// Pointer to the value owned by this GC value.
    value: *mut u8,
}

impl GarbageCollectorItem {
    /// Creates a new GC Item from a pointer to a value of type T.
    ///
    /// SAFETY: The pointer must point to a valid instance of T, that has not been dropped, and that has been allocated on the heap by the default allocator.
    unsafe fn new<T>(value: *mut T) -> Self {
        let value = value as *mut u8;
        let size = std::mem::size_of::<T>();
        let alignment = std::mem::align_of::<T>();
        // SAFETY: Spooky transmute. Miri says it's fine ¯\_(ツ)_/¯
        // SAFETY: Seriously though, I thinks it's fine, because fn(*mut u8) has the exact same function signature as fn(*mut T) in assembly.
        let drop_func =
            transmute::<unsafe fn(*mut T), unsafe fn(*mut u8)>(std::ptr::drop_in_place::<T>);
        Self {
            drop_func,
            size,
            value,
            alignment,
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
    /// Register the value with the GC.
    ///
    /// SAFETY: Value must point to a valid instance of T, that has not been dropped, and that has been allocated on the heap by the default allocator.
    unsafe fn insert<T>(&mut self, value: *mut T, is_used: bool) {
        self.allocated_objects
            .insert(GarbageCollectorItem::new(value), is_used);
    }
    /// Drop and free all orphaned objects registered with this garbage collector instanced.
    /// Takes the current state of the stack and the stack frames from the virtual machine.
    /// SAFETY: The stack and stack frames given to this function must be from the same virtual machine that owns this garbage collector instance.
    /// SAFETY: The caller must also take care to ensure they do not have any references or pointers pointing to any objects that may be dropped.
    pub(crate) unsafe fn collect_garbage<'a>(
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
                        self.handle_type(t)
                    }
                }
            }
        }
        for var in stack {
            if let AnacondaValue::Type(t) = var {
                self.handle_type(t)
            }
        }

        let unused_objects: Vec<GarbageCollectorItem> = self
            .allocated_objects
            .iter()
            .filter_map(|(k, is_used)| if !is_used { Some(*k) } else { None })
            .collect();

        for unused_object in unused_objects {
            self.allocated_objects.remove(&unused_object);

            let GarbageCollectorItem {
                drop_func,
                value,
                size,
                alignment,
            } = unused_object;
            // SAFETY: Here we are calling the drop implementation for the type stored inside the GC Item.
            // SAFETY: In reality drop_func takes a *mut T and value is *mut T, but the type system thinks they are u8 pointers.
            // SAFETY: This is borderline UB, but miri says it's fine?
            drop_func(value);
            std::alloc::dealloc(value, Layout::from_size_align_unchecked(size, alignment));
        }
    }
    /// Handle garbage collection for this type instance. Mark all its children and itself as used.
    /// The current implementation panics if t is not owned by this garbage collector. Future implementations may change this to something else. Calling this function with values not owned by this garbage collector in the future may result in UB.
    fn handle_type(&mut self, t: &GcValue<Type<'_>>) {
        // Return early if we have already handled this type instance. Otherwise indicate that it is currently being used.
        unsafe {
            // SAFETY: Creating a GarbageCollectorItem from a GcValue is safe because the value with a GcValue is always owned by this garbage collector and it points to a valid heap allocated non dropped instance.
            if !self
                .allocated_objects
                .get(&GarbageCollectorItem::new(t.value.as_ptr()))
                .unwrap()
            {
                // SAFETY: Pointer with GcValue is definitely valid.
                self.insert(t.value.as_ptr(), true);
            } else {
                return;
            }
        }

        // Handle the children of this type instance.
        t.with(|v| {
            v.fields.values().for_each(|x| {
                if let AnacondaValue::Type(t) = x {
                    self.handle_type(t);
                }
            });
        })
    }
}
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
/// Struct representing a garbage collected value. Users ought to be careful when using this struct, as the value within may be unexpectedly garbage collected if it does not exist on the stack of the virtual machine, or in a variable inside of the virtual machine.
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
        let item = unsafe {
            // SAFETY val.value points to a valid heap allocation so we can write into it.
            val.value.as_ptr().write(value);
            // SAFETY: val.value.as_ptr() definitely points to a valid heap allocation.
            GarbageCollectorItem::new(val.value.as_ptr())
        };
        gc.allocated_objects.insert(item, false);
        val
    }
    /// This function takes a closure as an argument and applies that closure to the value this GcValue wraps.
    ///
    ///
    /// The closure this function receives should probably not save the reference this method gives it as pointer, and especially not as a reference because the value might be garbage collected at any time unexpectedly.
    pub(crate) fn with<F: FnOnce(&T) -> R, R>(&self, func: F) -> R {
        func(unsafe { &*self.value.as_ptr() })
    }
    ///This function takes a closure as an argument and applies that closure to the value this GcValue wraps.
    ///
    /// The closure this function receives should probably not save the reference this method gives it as pointer, and especially not as a reference because the value might be garbage collected at any time unexpectedly.
    pub(crate) fn with_mut<F: FnOnce(&mut T) -> R, R>(&mut self, func: F) -> R {
        func(unsafe { &mut *self.value.as_ptr() })
    }
}
