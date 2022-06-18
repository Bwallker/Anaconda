use std::{collections::HashMap, fmt::Debug, hash::BuildHasherDefault};

use super::{Statement, ast::GenerateBytecode};
use ibig::IBig;
use std::hash::Hash;
use twox_hash::XxHash64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Program<'a> {
    pub(crate) statements: Vec<Statement<'a>>,
    pub(crate) big_int_literals: ValueStore<IBig>,
    pub(crate) string_literals: ValueStore<&'a str>,
}
#[derive(PartialEq, Eq, Debug, Clone)]
pub(crate) struct ValueStore<T: PartialEq + Eq + Hash + Debug + Clone> {
    data: HashMap<usize, T, BuildHasherDefault<XxHash64>>,
    reverse_data: HashMap<T, usize, BuildHasherDefault<XxHash64>>,
    index: usize,
}

impl<T: PartialEq + Eq + Hash + Debug + Clone> ValueStore<T> {
    pub(crate) fn new() -> Self {
        Self {
            data: HashMap::default(),
            reverse_data: HashMap::default(),
            index: 0,
        }
    }

    pub(crate) fn register_value(&mut self, val: T) -> usize {
        if let Some(v) = self.reverse_data.get(&val) {
            return *v;
        }
        self.data.insert(self.index, val.clone());
        self.reverse_data.insert(val, self.index);
        let ret = self.index;
        self.index += 1;
        ret
    }
}

fn generate_bytecode(statements: Vec<Statement>) -> Vec<u8> {
    let mut res = Vec::new();
    for mut statement in statements {
        statement.gen_bytecode(&mut res);
    }
    res
}
