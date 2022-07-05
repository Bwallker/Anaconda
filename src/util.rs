use std::{iter::Peekable, collections::{HashMap, HashSet}, hash::BuildHasherDefault};

use twox_hash::XxHash64;


pub(crate) struct RemoveLast<I: Iterator<Item = T>, T> {
    iter: Peekable<I>,
}

impl<I: Iterator<Item = T>, T> Iterator for RemoveLast<I, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.iter.next();
        if self.iter.peek().is_none() {
            None
        } else {
            next
        }
    }
}

pub(crate) trait CharAtIndex {
    fn char_at_index(&self, index: usize) -> Option<char>;
}

impl<T: AsRef<str>> CharAtIndex for T {
    /// Returns the first char in the string.
    fn char_at_index(&self, index: usize) -> Option<char> {
        let this = self.as_ref();
        this.get(index..).and_then(|x| x.chars().next())
    }
}

pub(crate) trait RemoveLastTrait<I: Iterator<Item = T>, T> {
    fn remove_last(self) -> RemoveLast<I, T>;
}

impl<I: Iterator<Item = T>, T> RemoveLastTrait<I, T> for I {
    fn remove_last(self) -> RemoveLast<I, T> {
        RemoveLast {
            iter: self.peekable(),
        }
    }
}

pub(crate) type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<XxHash64>>;
pub(crate) type FastSet<T> = HashSet<T, BuildHasherDefault<XxHash64>>;