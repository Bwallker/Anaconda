use core::panic;
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::{Debug, Display},
    hash::BuildHasherDefault,
};
const USIZE_BYTES: usize = (usize::BITS / 8) as usize;
use super::{ast::GenerateBytecode, Statement};
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
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Bytecode {
    pub(crate) instructions: Vec<u8>,
}

impl Bytecode {
    pub(crate) fn push(&mut self, instruction: OpCodes) {
        println!("Pushed instruction {instruction:#?} into bytecode");
        self.instructions.push(instruction as u8)
    }

    pub(crate) fn push_usize(&mut self, num: usize) {
        println!("Pushed usize {num:#?} into bytecode");

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
    LoadNameOfIdentifierFromIndex,
    LoadVariableValueFromIndex,
    LoadNothing,
    Print,
    CallFunction,
    Return,
    Pop,
    PrintStack,
    Break,
    StartOfFunctionDefinition,
    EndOfFunctionDefinition,
    SetVariableFromIndex,
}

impl Display for OpCodes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Scope<'a> {
    variables: HashMap<usize, AnacondaValue<'a>, BuildHasherDefault<XxHash64>>,
}

impl<'a> Scope<'a> {
    fn new() -> Self {
        Self {
            variables: HashMap::default(),
        }
    }
}
pub(crate) struct BytecodeInterpreter<'a> {
    pub(crate) program_counter: usize,
    pub(crate) bytecode: Bytecode,
    pub(crate) big_int_literals: ValueStore<IBig>,
    pub(crate) string_literals: ValueStore<&'a str>,
    pub(crate) identifier_literals: ValueStore<&'a str>,
    pub(crate) stack: Vec<AnacondaValue<'a>>,
    pub(crate) scopes: Vec<Scope<'a>>,
}

impl<'a> BytecodeInterpreter<'a> {
    pub(crate) fn new(program: Program<'a>, bytecode: Bytecode) -> Self {
        let mut this = Self {
            big_int_literals: program.big_int_literals,
            string_literals: program.string_literals,
            identifier_literals: program.identifier_literals,
            program_counter: 0,
            bytecode,
            stack: vec![],
            scopes: vec![Scope::new()],
        };
        this.add_builtins();
        this
    }

    fn add_builtins(&mut self) {
        // + 1 because the function proper begins after the start function def opcode.
        let print_start_index = self.bytecode.instructions.len() + 1;
        let idx = self.identifier_literals.register_value("value");
        self.bytecode.push(OpCodes::StartOfFunctionDefinition);
        self.bytecode.push(OpCodes::LoadVariableValueFromIndex);
        self.bytecode.push_usize(idx);
        self.bytecode.push(OpCodes::Print);
        self.bytecode.push(OpCodes::LoadNothing);
        self.bytecode.push(OpCodes::Return);
        self.bytecode.push(OpCodes::EndOfFunctionDefinition);

        let print = Function {
            extra_args: false,
            params: vec![idx],
            start_index: print_start_index,
        };
        let print_idx = self.identifier_literals.register_value("print");
        self.scopes[0]
            .variables
            .insert(print_idx, AnacondaValue::Function(print));

        //self.global_scope.variables.insert("print")
    }

    fn interpret_next_instruction(&mut self) {
        let opcode = self.current_opcode();
        match opcode {
            OpCodes::LoadSmallIntLiteral => {
                self.program_counter += 1;
                let value = self.bytecode.read_usize(self.program_counter);
                self.program_counter += USIZE_BYTES;
                self.stack.push(AnacondaValue::Int(IBig::from(value)));
            }
            OpCodes::LoadBigIntLiteral => {
                self.program_counter += 1;
                let index = self.bytecode.read_usize(self.program_counter);
                self.program_counter += USIZE_BYTES;
                let value = self.big_int_literals.data.get(&index).unwrap().clone();
                self.stack.push(AnacondaValue::Int(value));
            }
            OpCodes::LoadNameOfIdentifierFromIndex => {
                self.program_counter += 1;
                let index = self.bytecode.read_usize(self.program_counter);
                self.program_counter += USIZE_BYTES;
                let name = self.identifier_literals.data.get(&index).unwrap();
                self.stack.push(AnacondaValue::Identifier(name));
            }

            OpCodes::LoadVariableValueFromIndex => {
                self.program_counter += 1;
                let index = self.bytecode.read_usize(self.program_counter);
                self.program_counter += USIZE_BYTES;
                self.stack
                    .push(self.find_var_by_index(index).unwrap().clone());
            },
            OpCodes::SetVariableFromIndex => {
                self.program_counter += 1;
                let index = self.bytecode.read_usize(self.program_counter);
                self.program_counter += USIZE_BYTES;
                
            },
            OpCodes::LoadStringLiteral => {
                self.program_counter += 1;
                let index = self.bytecode.read_usize(self.program_counter);
                self.program_counter += USIZE_BYTES;
                let value = self.string_literals.data.get(&index).unwrap();
                self.stack.push(AnacondaValue::String(Cow::Borrowed(value)));
            }
            OpCodes::CallFunction => {
                self.program_counter += 1;
                let args_len = self.bytecode.read_usize(self.program_counter);
                self.program_counter += USIZE_BYTES;
                let f = match self.stack.pop().unwrap() {
                    AnacondaValue::Identifier(i) => self.identifier_to_function(i).clone(),
                    AnacondaValue::Function(f) => f,
                    invalid => panic!("{invalid:#?} is not a valid function"),
                };
                self.scopes.push(Scope::new());
                let idx_of_last = self.scopes.len() - 1;
                let newest_scope = &mut self.scopes[idx_of_last];
                for i in (0..args_len).rev() {
                    let idx_of_var = f.params[i];
                    newest_scope
                        .variables
                        .insert(idx_of_var, self.stack.pop().unwrap());
                }
                self.stack
                    .push(AnacondaValue::Int(IBig::from(self.program_counter)));
                self.program_counter = f.start_index;
            }
            OpCodes::Pop => {
                self.stack.pop();
                self.program_counter += 1;
            }
            OpCodes::Return => {
                self.scopes.pop();
                let return_value = self.stack.pop().unwrap();
                let return_address = self.stack.pop().unwrap();
                let return_address = match return_address {
                    AnacondaValue::Int(i) => i,
                    _ => {
                        unreachable!()
                    }
                };
                self.stack.push(return_value);
                let u: usize = return_address.try_into().unwrap();

                self.program_counter = u;
            }
            OpCodes::Print => {
                let to_print = self.stack.pop().unwrap();
                println!("{to_print}");
                self.program_counter += 1;
            }
            OpCodes::LoadNothing => {
                self.stack.push(AnacondaValue::Nothing);
                self.program_counter += 1;
            }
            OpCodes::StartOfFunctionDefinition => {
                let mut start_vs_end = 1isize;
                while start_vs_end > 0 {
                    self.program_counter += self.current_opcode_len();
                    match self.current_opcode() {
                        OpCodes::StartOfFunctionDefinition => start_vs_end += 1,
                        OpCodes::EndOfFunctionDefinition => start_vs_end -= 1,
                        _ => (),
                    }
                }
                self.program_counter += 1;
            },
            
            _ => todo!(),
        }
    }

    fn current_opcode(&self) -> OpCodes {
        self.bytecode.instructions[self.program_counter]
            .try_into()
            .unwrap()
    }
    /// Returns how many bytes in the bytecode the current opcode and its arguments take up.
    fn current_opcode_len(&self) -> usize {
        let curr = self.current_opcode();
        match curr {
            OpCodes::Break => 1,
            OpCodes::CallFunction => 1 + USIZE_BYTES,
            OpCodes::LoadBigIntLiteral => 1 + USIZE_BYTES,
            OpCodes::LoadSmallIntLiteral => 1 + USIZE_BYTES,
            OpCodes::LoadNameOfIdentifierFromIndex => 1 + USIZE_BYTES,
            OpCodes::LoadVariableValueFromIndex => 1 + USIZE_BYTES,
            OpCodes::Pop => 1,
            OpCodes::Print => 1,
            OpCodes::PrintStack => 1,
            OpCodes::Return => 1,
            OpCodes::StartOfFunctionDefinition => 1,
            OpCodes::EndOfFunctionDefinition => 1,
            OpCodes::LoadStringLiteral => 1 + USIZE_BYTES,
            OpCodes::LoadNothing => 1,
            OpCodes::SetVariableFromIndex => 1 + USIZE_BYTES,
        }
    }

    pub(crate) fn interpret_bytecode(&mut self) {
        while self.program_counter < self.bytecode.instructions.len() {
            self.interpret_next_instruction();
        }
    }

    fn identifier_to_function<'b>(&'b mut self, identifier: &'a str) -> &'b Function {
        match self.find_var_by_identifier(identifier) {
            Some(v) => match v {
                AnacondaValue::Function(f) => f,
                _ => panic!("Expected Function, found {v:#?}"),
            },
            None => {
                panic!("{identifier} is not the name of a known variable.")
            }
        }
    }

    fn find_var_by_index<'b>(&'b self, idx: usize) -> Option<&'b AnacondaValue<'a>> {
        for i in (0..self.scopes.len()).rev() {
            let scope = &self.scopes[i];
            if let Some(v) = scope.variables.get(&idx) {
                return Some(v);
            }
        }
        None
    }

    fn find_var_by_identifier<'b>(&'b self, ident: &'a str) -> Option<&'b AnacondaValue<'a>> {
        self.find_var_by_index(*self.identifier_literals.reverse_data.get(ident).unwrap())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Function {
    params: Vec<usize>,
    extra_args: bool,
    start_index: usize,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum AnacondaValue<'a> {
    Int(IBig),
    String(Cow<'a, str>),
    Function(Function),
    Identifier(&'a str),
    Nothing,
}

impl<'a> Display for AnacondaValue<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnacondaValue::Nothing => write!(f, "Nothing"),
            AnacondaValue::String(s) => write!(f, "\"{s}\""),
            AnacondaValue::Int(i) => write!(f, "{i}"),
            AnacondaValue::Identifier(i) => write!(f, "{i}"),
            AnacondaValue::Function(fun) => write!(f, "{:#?}", fun),
        }
    }
}
