use core::panic;
use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::HashMap,
    fmt::{Debug, Display},
    hash::BuildHasherDefault,
};
const USIZE_BYTES: usize = (usize::BITS / 8) as usize;

use super::{ast::GenerateBytecode, Statement};
use ibig::{ibig, IBig};
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
#[derive(Debug, TryFromPrimitive, PartialEq, Eq)]
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
    Assign,
    AddAndAssign,
    SubAndAssign,
    MultiplyAndAssign,
    DivideAndAssign,
    ModuloAndAssign,
    BitwiseAndAndAssign,
    BitwiseOrAndAssign,
    BitwiseXorAndAssign,
    BitshiftLeftAndAssign,
    BitshiftRightAndAssign,
    BooleanAnd,
    BooleanOr,
    BooleanNot,
    Equals,
    NotEquals,
    GreaterThan,
    GreaterThanEquals,
    LessThan,
    LessThanEquals,

    UnaryPlus,
    UnaryMinus,
    BitwiseNot,

    Add,
    Sub,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,

    Multiply,
    Divide,
    Modulo,
    BitshiftLeft,
    BitshiftRight,
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
                    .push(self.get_var_by_index(index).unwrap().clone());
            }
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
            }
            OpCodes::Assign
            | OpCodes::AddAndAssign
            | OpCodes::SubAndAssign
            | OpCodes::BitshiftLeftAndAssign
            | OpCodes::BitshiftRightAndAssign
            | OpCodes::BitwiseAndAndAssign
            | OpCodes::BitwiseOrAndAssign
            | OpCodes::BitwiseXorAndAssign
            | OpCodes::DivideAndAssign
            | OpCodes::MultiplyAndAssign
            | OpCodes::ModuloAndAssign => {
                macro_rules! assign_op {
                    ($e: tt) => {
                        {
                            self.program_counter += 1;
                            let from_stack_var = self.stack.pop().unwrap();
                            let idx = self.bytecode.read_usize(self.program_counter);
                            self.program_counter += USIZE_BYTES;
                            let var = self.get_var_by_index_mut(idx).unwrap();
                            match (var, from_stack_var) {
                                (AnacondaValue::Int(val), AnacondaValue::Int(from_stack)) => {
                                    if opcode == OpCodes::BitshiftLeftAndAssign {
                                        *val *= ibig!(2).pow((from_stack % (IBig::from(usize::MAX) + 1usize)).try_into().unwrap())
                                    }
                                    else if opcode == OpCodes::BitshiftRightAndAssign {
                                        *val /= ibig!(2).pow((from_stack % (IBig::from(usize::MAX) + 1usize)).try_into().unwrap())

                                    } else {
                                        let _ = *val $e from_stack;

                                    }
                                },
                                (AnacondaValue::String(val), AnacondaValue::String(from_stack)) => {
                                    if opcode == OpCodes::AddAndAssign {
                                        val.to_mut().push_str(&from_stack);
                                    } else if opcode == OpCodes::Assign {
                                        *val = from_stack;
                                    }
                                    else {
                                        panic!("Cannot perform operation {} on string.", stringify!($e))
                                    }
                                },
                                (incorrect_1, incorrect_2) => {
                                    panic!("Cannot perform operation {} on {incorrect_1} and {incorrect_2}", stringify!($e))
                                }
                            }

                        }
                }
                }

                match opcode {
                    OpCodes::Assign => assign_op!(=),
                    OpCodes::AddAndAssign => assign_op!(+=),
                    OpCodes::SubAndAssign => assign_op!(-=),
                    OpCodes::MultiplyAndAssign => assign_op!(*=),
                    OpCodes::DivideAndAssign => assign_op!(/=),
                    OpCodes::ModuloAndAssign => assign_op!(%=),
                    /* These two don't matter because we handle them as special cases because IBig does not support <<= or >>= */
                    OpCodes::BitshiftLeftAndAssign => assign_op!(!=),
                    OpCodes::BitshiftRightAndAssign => assign_op!(==),
                    /* END COMMENT */
                    OpCodes::BitwiseAndAndAssign => assign_op!(&=),
                    OpCodes::BitwiseOrAndAssign => assign_op!(|=),
                    OpCodes::BitwiseXorAndAssign => assign_op!(^=),
                    _ => unreachable!(),
                }
            }

            OpCodes::BooleanAnd => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                if !first.as_bool() {
                    self.stack.pop();
                    self.stack.push(first);
                }
            }
            OpCodes::BooleanOr => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                if first.as_bool() {
                    self.stack.pop();
                    self.stack.push(first);
                }
            }
            OpCodes::BooleanNot => {
                let val = self.stack.pop().unwrap();
                let inverse = !val.as_bool();
                self.stack.push(AnacondaValue::Bool(inverse));
            }
            OpCodes::PrintStack => println!("{:#?}", self.stack),
            OpCodes::EndOfFunctionDefinition => {
                panic!("Hit EndOfFunctionDefinition instruction. This should never happen.")
            }
            OpCodes::Break => todo!(),

            OpCodes::Equals => {
                self.program_counter += 1;

                let first = self.stack.pop();
                let second = self.stack.pop();
                self.stack.push(AnacondaValue::Bool(first == second));
            }
            OpCodes::NotEquals => {
                self.program_counter += 1;

                let first = self.stack.pop();
                let second = self.stack.pop();
                self.stack.push(AnacondaValue::Bool(first != second));
            }
            OpCodes::GreaterThan => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                self.stack.push(AnacondaValue::Bool(match (first, second) {
                    (AnacondaValue::Int(i1), AnacondaValue::Int(i2)) => i1 > i2,
                    (v1, v2) => {
                        panic!("Gannot perform > operation on {v1} and {v2}")
                    }
                }))
            }
            OpCodes::GreaterThanEquals => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                self.stack.push(AnacondaValue::Bool(match (first, second) {
                    (AnacondaValue::Int(i1), AnacondaValue::Int(i2)) => i1 >= i2,
                    (v1, v2) => {
                        panic!("Gannot perform >= operation on {v1} and {v2}")
                    }
                }))
            }
            OpCodes::LessThan => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                self.stack.push(AnacondaValue::Bool(match (first, second) {
                    (AnacondaValue::Int(i1), AnacondaValue::Int(i2)) => i1 < i2,
                    (v1, v2) => {
                        panic!("Gannot perform < operation on {v1} and {v2}")
                    }
                }))
            }
            OpCodes::LessThanEquals => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                self.stack.push(AnacondaValue::Bool(match (first, second) {
                    (AnacondaValue::Int(i1), AnacondaValue::Int(i2)) => i1 <= i2,
                    (v1, v2) => {
                        panic!("Gannot perform <= operation on {v1} and {v2}")
                    }
                }))
            }

            OpCodes::UnaryPlus => {
                self.program_counter += 1;
                let val = self.stack.last_mut().unwrap();
                if let AnacondaValue::Int(_) = val {
                } else {
                    panic!("Cannot perform unary plus operation on {val}.")
                }
            }
            OpCodes::UnaryMinus => {
                self.program_counter += 1;
                let val = self.stack.last_mut().unwrap();
                if let AnacondaValue::Int(i) = val {
                    *i *= ibig!(-1);
                } else {
                    panic!("Cannot perform unary minus operation on {val}.")
                }
            }
            OpCodes::BitwiseNot => {
                self.program_counter += 1;
                let val = self.stack.last_mut().unwrap();
                if let AnacondaValue::Int(i) = val {
                    *i *= ibig!(-1);
                    *i -= ibig!(1);
                } else {
                    panic!("Cannot perform unary minus operation on {val}.")
                }
            }

            OpCodes::BitwiseAnd => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                match (first, second) {
                    (AnacondaValue::Int(mut i1), AnacondaValue::Int(i2)) => {
                        i1 &= i2;
                        self.stack.push(AnacondaValue::Int(i1));
                    }
                    (v1, v2) => {
                        panic!("Cannot perform & operation on {v1} and {v2}")
                    }
                }
            }
            OpCodes::BitwiseOr => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                match (first, second) {
                    (AnacondaValue::Int(mut i1), AnacondaValue::Int(i2)) => {
                        i1 |= i2;
                        self.stack.push(AnacondaValue::Int(i1));
                    }
                    (v1, v2) => {
                        panic!("Cannot perform | operation on {v1} and {v2}")
                    }
                }
            }
            OpCodes::BitwiseXor => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                match (first, second) {
                    (AnacondaValue::Int(mut i1), AnacondaValue::Int(i2)) => {
                        i1 ^= i2;
                        self.stack.push(AnacondaValue::Int(i1));
                    }
                    (v1, v2) => {
                        panic!("Cannot perform ^ operation on {v1} and {v2}")
                    }
                }
            }
            OpCodes::Add => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                match (first, second) {
                    (AnacondaValue::Int(mut i1), AnacondaValue::Int(i2)) => {
                        i1 += i2;
                        self.stack.push(AnacondaValue::Int(i1));
                    }
                    (AnacondaValue::String(mut s1), AnacondaValue::String(s2)) => {
                        s1.to_mut().push_str(&s2);
                        self.stack.push(AnacondaValue::String(s1));
                    }
                    (v1, v2) => {
                        panic!("Cannot perform + operation on {v1} and {v2}")
                    }
                }
            }
            OpCodes::Sub => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                match (first, second) {
                    (AnacondaValue::Int(mut i1), AnacondaValue::Int(i2)) => {
                        i1 -= i2;
                        self.stack.push(AnacondaValue::Int(i1));
                    }
                    (v1, v2) => {
                        panic!("Cannot perform - operation on {v1} and {v2}")
                    }
                }
            }
            OpCodes::Multiply => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                match (first, second) {
                    (AnacondaValue::Int(mut i1), AnacondaValue::Int(i2)) => {
                        i1 *= i2;
                        self.stack.push(AnacondaValue::Int(i1));
                    }
                    (AnacondaValue::String(s), AnacondaValue::Int(i))
                    | (AnacondaValue::Int(i), AnacondaValue::String(s)) => {
                        match i.cmp(&ibig!(0)) {
                            Ordering::Equal => {
                                self.stack.push(AnacondaValue::String(Cow::Borrowed("")));
                            }
                            Ordering::Less => {
                                panic!("Cannot multipy a string by a value less than 0");
                            }
                            Ordering::Greater => {
                                let s = s.as_ref().repeat(i.try_into().unwrap_or(usize::MAX));
                                self.stack.push(AnacondaValue::String(Cow::Owned(s)));
                            }
                        }
                    }
                    (v1, v2) => {
                        panic!("Cannot perform * operation on {v1} and {v2}")
                    }
                }
            }
            OpCodes::Divide => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                match (first, second) {
                    (AnacondaValue::Int(mut i1), AnacondaValue::Int(i2)) => {
                        if i2 == ibig!(0) {
                            panic!("Cannot divide by 0!");
                        }
                        i1 /= i2;
                        self.stack.push(AnacondaValue::Int(i1));
                    }
                    (v1, v2) => {
                        panic!("Cannot perform / operation on {v1} and {v2}")
                    }
                }
            }
            OpCodes::Modulo => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                match (first, second) {
                    (AnacondaValue::Int(mut i1), AnacondaValue::Int(i2)) => {
                        i1 %= i2;
                        self.stack.push(AnacondaValue::Int(i1));
                    }
                    (v1, v2) => {
                        panic!("Cannot perform % operation on {v1} and {v2}")
                    }
                }
            }
            OpCodes::BitshiftLeft => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                match (first, second) {
                    (AnacondaValue::Int(mut i1), AnacondaValue::Int(i2)) => {
                        i1 *= ibig!(2)
                            .pow((i2 % (IBig::from(usize::MAX) + 1usize)).try_into().unwrap());
                        self.stack.push(AnacondaValue::Int(i1));
                    }
                    (v1, v2) => {
                        panic!("Cannot perform << operation on {v1} and {v2}")
                    }
                }
            }
            OpCodes::BitshiftRight => {
                self.program_counter += 1;
                let first = self.stack.pop().unwrap();
                let second = self.stack.pop().unwrap();
                match (first, second) {
                    (AnacondaValue::Int(mut i1), AnacondaValue::Int(i2)) => {
                        i1 /= ibig!(2)
                            .pow((i2 % (IBig::from(usize::MAX) + 1usize)).try_into().unwrap());

                        self.stack.push(AnacondaValue::Int(i1));
                    }
                    (v1, v2) => {
                        panic!("Cannot perform >> operation on {v1} and {v2}")
                    }
                }
            }
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
            OpCodes::Assign => 1 + USIZE_BYTES,
            OpCodes::AddAndAssign => 1 + USIZE_BYTES,
            OpCodes::SubAndAssign => 1 + USIZE_BYTES,
            OpCodes::MultiplyAndAssign => 1 + USIZE_BYTES,
            OpCodes::DivideAndAssign => 1 + USIZE_BYTES,
            OpCodes::ModuloAndAssign => 1 + USIZE_BYTES,

            OpCodes::BitwiseAndAndAssign => 1 + USIZE_BYTES,
            OpCodes::BitwiseOrAndAssign => 1 + USIZE_BYTES,
            OpCodes::BitwiseXorAndAssign => 1 + USIZE_BYTES,

            OpCodes::BitshiftLeftAndAssign => 1 + USIZE_BYTES,
            OpCodes::BitshiftRightAndAssign => 1 + USIZE_BYTES,

            OpCodes::BooleanAnd => 1,
            OpCodes::BooleanOr => 1,
            OpCodes::BooleanNot => 1,

            OpCodes::Equals => 1,
            OpCodes::NotEquals => 1,
            OpCodes::LessThan => 1,
            OpCodes::LessThanEquals => 1,
            OpCodes::GreaterThan => 1,
            OpCodes::GreaterThanEquals => 1,

            OpCodes::UnaryMinus => 1,
            OpCodes::UnaryPlus => 1,
            OpCodes::BitwiseNot => 1,

            OpCodes::BitwiseOr => 1,
            OpCodes::BitwiseXor => 1,
            OpCodes::BitwiseAnd => 1,

            OpCodes::Add => 1,
            OpCodes::Sub => 1,

            OpCodes::Modulo => 1,
            OpCodes::Multiply => 1,
            OpCodes::Divide => 1,
            OpCodes::BitshiftLeft => 1,
            OpCodes::BitshiftRight => 1,
        }
    }

    pub(crate) fn interpret_bytecode(&mut self) {
        while self.program_counter < self.bytecode.instructions.len() {
            self.interpret_next_instruction();
        }
    }

    fn identifier_to_function<'b>(&'b mut self, identifier: &'a str) -> &'b Function {
        match self.get_var_by_identifier(identifier) {
            Some(v) => match v {
                AnacondaValue::Function(f) => f,
                _ => panic!("Expected Function, found {v:#?}"),
            },
            None => {
                panic!("{identifier} is not the name of a known variable.")
            }
        }
    }

    fn get_var_by_index<'b>(&'b self, idx: usize) -> Option<&'b AnacondaValue<'a>> {
        for i in (0..self.scopes.len()).rev() {
            let scope = &self.scopes[i];
            if let Some(v) = scope.variables.get(&idx) {
                return Some(v);
            }
        }
        None
    }

    fn get_var_by_index_mut<'b>(&'b mut self, idx: usize) -> Option<&'b mut AnacondaValue<'a>> {
        let this = self as *mut Self;
        let len = self.scopes.len();
        for i in (0..len).rev() {
            let scope = unsafe { &mut (*this).scopes[i] as *mut Scope };
            if let Some(v) = unsafe { (*scope).variables.get_mut(&idx) } {
                return Some(v);
            }
        }
        None
    }

    fn get_var_by_identifier<'b>(&'b self, ident: &'a str) -> Option<&'b AnacondaValue<'a>> {
        self.get_var_by_index(*self.identifier_literals.reverse_data.get(ident).unwrap())
    }

    fn get_var_by_identifier_mut<'b>(
        &'b mut self,
        ident: &'a str,
    ) -> Option<&'b mut AnacondaValue<'a>> {
        self.get_var_by_index_mut(*self.identifier_literals.reverse_data.get(ident).unwrap())
    }

    fn set_var_by_index<'b>(&'b mut self, index: usize, val: AnacondaValue<'a>) {
        for i in (0..self.scopes.len()).rev() {
            let scope = &mut self.scopes[i];
            if scope.variables.get(&index).is_some() {
                scope.variables.insert(index, val);
                return;
            }
        }
        let len = self.scopes.len();
        self.scopes[len - 1].variables.insert(index, val);
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
    Bool(bool),
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
            AnacondaValue::Bool(b) => write!(f, "{b}"),
        }
    }
}

impl<'a> AnacondaValue<'a> {
    fn as_bool(&self) -> bool {
        match self {
            AnacondaValue::Int(i) => *i != ibig!(0),
            AnacondaValue::Nothing => false,
            AnacondaValue::Function(_) => true,
            AnacondaValue::Identifier(i) => !i.is_empty(),
            AnacondaValue::String(s) => !s.is_empty(),
            AnacondaValue::Bool(b) => *b,
        }
    }
}
