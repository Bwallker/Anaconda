use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::{Debug, Display},
    hash::BuildHasherDefault,
};
const USIZE_BYTES: usize = (usize::BITS / 8) as usize;
use super::{
    ast::{GenerateBytecode, Identifier},
    Statement,
};
use ibig::IBig;
use std::hash::Hash;
use twox_hash::XxHash64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Program<'a> {
    pub(crate) statements: Vec<Statement<'a>>,
    pub(crate) big_int_literals: ValueStore<IBig>,
    pub(crate) string_literals: ValueStore<&'a str>,
    pub(crate) identifier_literals: ValueStore<&'a str>,
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
#[repr(transparent)]
#[derive(Debug)]
pub(crate) struct Bytecode {
    pub(crate) instructions: Vec<u8>,
}

impl Bytecode {
    pub(crate) fn push(&mut self, instruction: OpCodes) {
        println!("Pushing instruction {instruction} into bytecode vec");
        self.instructions.push(instruction as u8)
    }

    pub(crate) fn push_usize(&mut self, num: usize) {
        println!("Pushing usize {num} into bytecode vec");

        self.instructions.extend_from_slice(&num.to_le_bytes())
    }

    pub(crate) fn read_usize(&self, index: usize) -> usize {
        usize::from_le_bytes(
            self.instructions
                .get(index..index + USIZE_BYTES)
                .and_then(|x| x.try_into().ok())
                .unwrap(),
        )
    }

    pub(crate) fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }
}

pub(crate) fn generate_bytecode(statements: Vec<Statement>) -> Bytecode {
    let mut res = Bytecode::new();
    for mut statement in statements {
        statement.gen_bytecode(&mut res);
    }
    res
}
use num_enum::TryFromPrimitive;
// BYTECODE INSTRUCTIONS:
#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
pub(crate) enum OpCodes {
    LoadSmallIntLiteral = 0,
    LoadBigIntLiteral,
    LoadStringLiteral,
    LoadIdentifierLiteral,
    LoadVar,
    CallFunction,
}

impl Display for OpCodes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}

struct Scope<'a> {
    variables: HashMap<&'a str, AnacondaValue<'a>, BuildHasherDefault<XxHash64>>,
}

impl<'a> Scope<'a> {
    fn new() -> Self {
        Self {
            variables: HashMap::default(),
        }
    }
}
struct BytecodeInterpreter<'a> {
    pub(crate) index: usize,
    pub(crate) bytecode: Bytecode,
    pub(crate) big_int_literals: ValueStore<IBig>,
    pub(crate) string_literals: ValueStore<&'a str>,
    pub(crate) identifier_literals: ValueStore<&'a str>,
    pub(crate) stack: Vec<AnacondaValue<'a>>,
    pub(crate) global_scope: Scope<'a>,
}

impl<'a> BytecodeInterpreter<'a> {
    fn new(program: Program<'a>, bytecode: Bytecode) -> Self {
        let mut this = Self {
            big_int_literals: program.big_int_literals,
            string_literals: program.string_literals,
            identifier_literals: program.identifier_literals,
            index: 0,
            bytecode,
            stack: vec![],
            global_scope: Scope::new(),
        };
        this.add_builtins();
        this
    }

    fn add_builtins(&mut self) {
        //self.global_scope.variables.insert("print")
    }

    fn interpret_next_instruction(&mut self) {
        match self.bytecode.instructions[self.index].try_into().unwrap() {
            OpCodes::LoadSmallIntLiteral => {
                self.index += 1;
                let value = self.bytecode.read_usize(self.index);
                self.index += USIZE_BYTES;
                self.stack.push(AnacondaValue::Int(IBig::from(value)));
            }
            OpCodes::LoadBigIntLiteral => {
                self.index += 1;
                let index = self.bytecode.read_usize(self.index);
                self.index += USIZE_BYTES;
                let value = self.big_int_literals.data.get(&index).unwrap().clone();
                self.stack.push(AnacondaValue::Int(value));
            }
            OpCodes::LoadIdentifierLiteral => {
                self.index += 1;
                let index = self.bytecode.read_usize(self.index);
                self.index += USIZE_BYTES;
                let value = self.identifier_literals.data.get(&index).unwrap();
                self.stack.push(AnacondaValue::Identifier(value));
            }
            OpCodes::LoadStringLiteral => {
                self.index += 1;
                let index = self.bytecode.read_usize(self.index);
                self.index += USIZE_BYTES;
                let value = self.string_literals.data.get(&index).unwrap();
                self.stack.push(AnacondaValue::String(Cow::Borrowed(value)));
            }
            OpCodes::CallFunction => {
                self.index += 1;
                let args_len = self.bytecode.read_usize(self.index);
                self.index += USIZE_BYTES;
                match self.stack.pop().unwrap() {
                    AnacondaValue::Identifier(i) => {
                        
                    },
                    AnacondaValue::Function(f) => {

                    }
                }
            }
            _ => todo!(),
        }
    }
}

struct Function<'a> {
    /// Index at which the function starts in the bytecode.
    start_index: usize,
    params: Vec<&'a str>,
    extra_args: bool,
}

enum AnacondaValue<'a> {
    Int(IBig),
    String(Cow<'a, str>),
    Function(Function<'a>),
    Identifier(&'a str),
}
