use core::panic;
use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::HashMap,
    fmt::{Debug, Display},
};
pub(crate) const USIZE_BYTES: usize = (usize::BITS / 8) as usize;
use std::mem::replace;

use crate::{parser::ast::Block, util::FastMap};
use ibig::{ibig, IBig};
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Program<'a> {
    pub(crate) base_block: Option<Block<'a>>,
    pub(crate) big_int_literals: ValueStore<IBig>,
    pub(crate) string_literals: ValueStore<&'a str>,
    pub(crate) identifier_literals: ValueStore<&'a str>,
    pub(crate) function_definitions: ValueStore<GcValue<Function>>,
}
#[derive(PartialEq, Eq, Debug, Clone)]
pub(crate) struct ValueStore<T: PartialEq + Eq + Hash + Debug + Clone> {
    pub(crate) data: FastMap<usize, T>,
    pub(crate) reverse_data: FastMap<T, usize>,
    pub(crate) index: usize,
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
impl Display for Bytecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut pc = 0;
        while pc < self.instructions.len() {
            let curr_opcode = self.instructions[pc];
            let curr_opcode = OpCodes::try_from(curr_opcode).unwrap();
            if curr_opcode.len() == 1 {
                writeln!(f, "{pc}: {:#?}", curr_opcode)?;
                pc += 1;
            } else {
                writeln!(f, "{pc}: {:#?} {}", curr_opcode, self.read_usize(pc + 1))?;
                pc += USIZE_BYTES + 1;
            }
        }
        Ok(())
    }
}
impl Bytecode {
    pub(crate) fn push_opcode(&mut self, instruction: OpCodes) {
        //println!("Pushed instruction {instruction:#?} into bytecode");
        self.instructions.push(instruction as u8)
    }

    pub(crate) fn push_usize(&mut self, num: usize) {
        //println!("Pushed usize {num:#?} into bytecode");

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

    pub(crate) fn set_usize(&mut self, index: usize, val: usize) {
        let mut i = 0;
        let mut bytes = val.to_le_bytes().into_iter();
        while i < USIZE_BYTES {
            self.instructions[index + i] = bytes.next().unwrap();
            i += 1;
        }
    }

    pub(crate) fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }
}

use num_enum::TryFromPrimitive;

use super::gc::{GarbageCollector, GcValue};
// BYTECODE INSTRUCTIONS:
#[repr(u8)]
#[derive(Copy, Clone, Debug, TryFromPrimitive, PartialEq, Eq)]
pub(crate) enum OpCodes {
    LoadSmallIntLiteral = 0,
    LoadBigIntLiteral,
    LoadStringLiteral,
    LoadVariableValueFromIndex,
    LoadPoison,
    LoadFalse,
    LoadTrue,

    ToString,

    Exponent,

    Print,
    Println,

    CallFunction,

    CallMethod,

    Return,
    Pop,
    PrintStack,
    Continue,
    Break,
    BreakIfFalse,
    StartOfFunctionDefinition,
    EndOfFunctionDefinition,
    BeginBlock,
    EndBlock,

    LoadTemp,
    StoreTemp,

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
    ExponentAndAssign,

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

    StartOfLoop,

    IfTrueGoto,
    IfFalseGoto,
    Goto,
}

impl OpCodes {
    const fn len(self) -> usize {
        match self {
            OpCodes::ToString => 1,

            OpCodes::StartOfLoop => 1 + USIZE_BYTES,

            OpCodes::Break => 1,
            OpCodes::BreakIfFalse => 1,
            OpCodes::Continue => 1,

            OpCodes::LoadTemp => 1,
            OpCodes::StoreTemp => 1,

            OpCodes::CallFunction => 1 + USIZE_BYTES,
            OpCodes::CallMethod => 1 + USIZE_BYTES,

            OpCodes::LoadBigIntLiteral => 1 + USIZE_BYTES,
            OpCodes::LoadSmallIntLiteral => 1 + USIZE_BYTES,
            OpCodes::LoadVariableValueFromIndex => 1 + USIZE_BYTES,
            OpCodes::Pop => 1,
            OpCodes::Print => 1,
            OpCodes::Println => 1,
            OpCodes::PrintStack => 1,
            OpCodes::Return => 1,
            OpCodes::StartOfFunctionDefinition => 1 + USIZE_BYTES,
            OpCodes::EndOfFunctionDefinition => 1,
            OpCodes::BeginBlock => 1,
            OpCodes::EndBlock => 1,
            OpCodes::LoadStringLiteral => 1 + USIZE_BYTES,
            OpCodes::LoadPoison => 1,
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

            OpCodes::ExponentAndAssign => 1 + USIZE_BYTES,

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
            OpCodes::Exponent => 1,

            OpCodes::LoadFalse => 1,
            OpCodes::LoadTrue => 1,

            OpCodes::Goto => 1 + USIZE_BYTES,
            OpCodes::IfTrueGoto => 1 + USIZE_BYTES,
            OpCodes::IfFalseGoto => 1 + USIZE_BYTES,
        }
    }
}
impl Display for OpCodes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self)
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Scope<'a> {
    pub(crate) variables: FastMap<usize, AnacondaValue<'a>>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StackFrame<'a> {
    Scope(Scope<'a>),
    Function(usize),
    Loop(LoopAddresses),
}

impl<'a> StackFrame<'a> {
    fn as_scope(&self) -> &Scope<'a> {
        match self {
            StackFrame::Scope(scope) => scope,
            _ => panic!("Not a scope"),
        }
    }
}

impl<'a> Scope<'a> {
    fn new() -> Self {
        Self {
            variables: HashMap::default(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoopAddresses {
    pub(crate) address_of_start: usize,
    pub(crate) address_of_end: usize,
}
pub(crate) struct BytecodeInterpreter<'a> {
    pub(crate) program_counter: usize,
    pub(crate) bytecode: Bytecode,
    pub(crate) big_int_literals: ValueStore<IBig>,
    pub(crate) string_literals: ValueStore<&'a str>,
    pub(crate) identifier_literals: ValueStore<&'a str>,
    pub(crate) function_definitions: ValueStore<GcValue<Function>>,
    pub(crate) stack: Vec<AnacondaValue<'a>>,
    pub(crate) stack_frames: Vec<StackFrame<'a>>,
    pub(crate) temp: AnacondaValue<'a>,
    pub(crate) gc: GarbageCollector,
}

impl Drop for BytecodeInterpreter<'_> {
    fn drop(&mut self) {
        // SAFETY: Calling collect when the interpreter is dropped is safe because all outstanding references and pointers to things owned by the GC should be gone at this point, and the GC will go away after this function so we need to clean up all the memory that has not been collected yet.
        unsafe {
            self.gc.collect_garbage(&vec![], &vec![]);
        }
    }
}

impl<'a> BytecodeInterpreter<'a> {
    pub(crate) fn new(program: Program<'a>, bytecode: Bytecode, gc: GarbageCollector) -> Box<Self> {
        let mut this = Box::new(Self {
            big_int_literals: program.big_int_literals,
            string_literals: program.string_literals,
            identifier_literals: program.identifier_literals,
            function_definitions: program.function_definitions,
            program_counter: 0,
            bytecode,
            stack: Vec::with_capacity(100),
            stack_frames: {
                let mut stack_frames = Vec::with_capacity(20);
                stack_frames.push(StackFrame::Scope(Scope::new()));
                stack_frames
            },
            temp: AnacondaValue::Poison,
            gc,
        });
        this.register_builtins();
        this.register_types();
        this
    }

    fn register_meta(&mut self) {
        let meta_start_index = self.bytecode.instructions.len() + 1 + USIZE_BYTES;
        let idx = self.identifier_literals.register_value("value");
        let meta = Function {
            extra_args: false,
            params: vec![idx],
            start_index: meta_start_index,
        };
        let meta = GcValue::new(meta, &mut self.gc);
        let def_idx = self.function_definitions.register_value(meta.clone());
        self.bytecode
            .push_opcode(OpCodes::StartOfFunctionDefinition);
        self.bytecode.push_usize(def_idx);
        self.bytecode.push_opcode(OpCodes::LoadTrue);

        self.bytecode.push_opcode(OpCodes::Return);
        self.bytecode.push_opcode(OpCodes::EndOfFunctionDefinition);
        self.bytecode.push_opcode(OpCodes::Pop);

        let meta_idx = self.identifier_literals.register_value("meta");
        match self.stack_frames[0] {
            StackFrame::Scope(ref mut s) => {
                s.variables.insert(meta_idx, AnacondaValue::Function(meta));
            }
            _ => unreachable!(),
        }
    }

    fn register_str(&mut self) {
        let str_start_index = self.bytecode.instructions.len() + 1 + USIZE_BYTES;
        let idx = self.identifier_literals.register_value("value");
        let str_ = Function {
            extra_args: false,
            params: vec![idx],
            start_index: str_start_index,
        };
        let str_ = GcValue::new(str_, &mut self.gc);
        let def_idx = self.function_definitions.register_value(str_.clone());
        self.bytecode
            .push_opcode(OpCodes::StartOfFunctionDefinition);
        self.bytecode.push_usize(def_idx);

        self.bytecode
            .push_opcode(OpCodes::LoadVariableValueFromIndex);
        self.bytecode.push_usize(idx);

        self.bytecode.push_opcode(OpCodes::ToString);

        self.bytecode.push_opcode(OpCodes::Return);
        self.bytecode.push_opcode(OpCodes::EndOfFunctionDefinition);
        self.bytecode.push_opcode(OpCodes::Pop);
        let str_idx = self.identifier_literals.register_value("str");
        match self.stack_frames[0] {
            StackFrame::Scope(ref mut s) => {
                s.variables.insert(str_idx, AnacondaValue::Function(str_));
            }
            _ => unreachable!(),
        }
    }
    fn register_builtins(&mut self) {
        macro_rules! register_print_fn {
            ($print_opcode: expr, $func_name: expr) => {
                // + 5 because the function proper begins after the start function def opcode.
                let print_start_index = self.bytecode.instructions.len() + 1 + USIZE_BYTES;
                let idx = self.identifier_literals.register_value("value");
                let print = Function {
                    extra_args: false,
                    params: vec![idx],
                    start_index: print_start_index,
                };
                let print = GcValue::new(print, &mut self.gc);
                let def_idx = self.function_definitions.register_value(print.clone());
                self.bytecode
                    .push_opcode(OpCodes::StartOfFunctionDefinition);
                self.bytecode.push_usize(def_idx);
                self.bytecode
                    .push_opcode(OpCodes::LoadVariableValueFromIndex);
                self.bytecode.push_usize(idx);
                self.bytecode.push_opcode($print_opcode);
                self.bytecode.push_opcode(OpCodes::LoadPoison);
                self.bytecode.push_opcode(OpCodes::Return);
                self.bytecode.push_opcode(OpCodes::EndOfFunctionDefinition);
                self.bytecode.push_opcode(OpCodes::Pop);

                let print_idx = self.identifier_literals.register_value($func_name);
                match self.stack_frames[0] {
                    StackFrame::Scope(ref mut s) => {
                        s.variables
                            .insert(print_idx, AnacondaValue::Function(print));
                    }
                    _ => unreachable!(),
                }
            };
        }
        register_print_fn!(OpCodes::Print, "print");
        register_print_fn!(OpCodes::Println, "println");
        self.register_meta();
        self.register_str();
    }

    fn register_types(&mut self) {
        let base_fields = {
            let mut fields = HashMap::default();
            let id = self.identifier_literals.reverse_data.get("str").unwrap();
            fields.insert(
                *id,
                self.stack_frames[0]
                    .as_scope()
                    .variables
                    .get(id)
                    .unwrap()
                    .clone(),
            );

            fields
        };
        let int = Type {
            name: "integer",
            fields: base_fields.clone(),
        };
        let str_ = Type {
            name: "string",
            fields: base_fields.clone(),
        };
        let bool_ = Type {
            name: "boolean",
            fields: base_fields.clone(),
        };
        let function = Type {
            name: "function",
            fields: base_fields.clone(),
        };

        match self.stack_frames[0] {
            StackFrame::Scope(ref mut s) => {
                s.variables.insert(
                    self.identifier_literals.register_value("integer"),
                    AnacondaValue::Type(GcValue::new(int, &mut self.gc)),
                );
                s.variables.insert(
                    self.identifier_literals.register_value("string"),
                    AnacondaValue::Type(GcValue::new(str_, &mut self.gc)),
                );
                s.variables.insert(
                    self.identifier_literals.register_value("boolean"),
                    AnacondaValue::Type(GcValue::new(bool_, &mut self.gc)),
                );
                s.variables.insert(
                    self.identifier_literals.register_value("function"),
                    AnacondaValue::Type(GcValue::new(function, &mut self.gc)),
                );
            }
            _ => unreachable!(),
        }
    }

    fn break_from_loop(&mut self) {
        while let Some(StackFrame::Scope(_)) = self.stack_frames.last() {
            self.stack_frames.pop().unwrap();
        }
        match self.stack_frames.pop().unwrap() {
            StackFrame::Loop(l) => {
                self.program_counter = l.address_of_end;
            }
            _ => panic!("Break instruction was not inside a loop!"),
        }
    }

    fn interpret_next_instruction(&mut self) {
        let opcode = self.current_opcode();
        //println!("{opcode:#?}");
        //println!("{:#?}", self.stack);
        match opcode {
            OpCodes::LoadTemp => {
                self.stack
                    .push(std::mem::replace(&mut self.temp, AnacondaValue::Poison));
                self.program_counter += 1;
            }
            OpCodes::StoreTemp => {
                self.temp = self.stack.pop().unwrap();
                self.program_counter += 1;
            }
            OpCodes::ToString => {
                let val = self.stack.pop().unwrap();
                self.stack
                    .push(AnacondaValue::String(Cow::Owned(val.to_string())));
                self.program_counter += 1;
            }
            OpCodes::StartOfLoop => {
                self.program_counter += 1;
                let address_of_end = self.bytecode.read_usize(self.program_counter);
                self.program_counter += USIZE_BYTES;
                let address_of_start = self.program_counter;
                self.stack_frames.push(StackFrame::Loop(LoopAddresses {
                    address_of_start,
                    address_of_end,
                }));
            }
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

            OpCodes::LoadVariableValueFromIndex => {
                self.program_counter += 1;
                let index = self.bytecode.read_usize(self.program_counter);
                self.program_counter += USIZE_BYTES;
                //println!("{:#?}", self.identifier_literals.data.get(&index));
                //let id = self.identifier_literals.data.get(&index);
                let val = self.get_var_by_index(index).unwrap().clone();
                //println!("{id:#?} = {val:#?}");
                self.stack.push(val);
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
                self.stack_frames
                    .push(StackFrame::Function(self.program_counter));
                self.stack_frames.push(StackFrame::Scope(Scope::new()));
                let last_elem_idx = self.stack.len() - 1;
                let f = match replace(
                    &mut self.stack[last_elem_idx - args_len],
                    AnacondaValue::Poison,
                ) {
                    AnacondaValue::Function(f) => f,
                    _ => {
                        println!("{:#?}", self.stack);

                        panic!("Tried to call a non-function!")
                    }
                };
                f.with(|f| {
                    let idx_of_last = self.stack_frames.len() - 1;
                    match self.stack_frames[idx_of_last] {
                        StackFrame::Scope(ref mut s) => {
                            for i in (0..args_len).rev() {
                                let idx_of_var = f.params[i];
                                s.variables.insert(idx_of_var, self.stack.pop().unwrap());
                            }
                        }
                        _ => unreachable!(),
                    }

                    self.stack.pop(); // pop the function

                    self.program_counter = f.start_index;
                });
            }
            OpCodes::CallMethod => {
                todo!()
            }
            OpCodes::Pop => {
                self.stack.pop();
                self.program_counter += 1;
            }
            OpCodes::Return => {
                while let StackFrame::Scope(_) | StackFrame::Loop(_) =
                    self.stack_frames.last().unwrap()
                {
                    self.stack_frames.pop();
                }
                match self.stack_frames.pop().unwrap() {
                    StackFrame::Function(address) => {
                        self.program_counter = address;
                    }
                    _ => unreachable!(),
                }
            }
            OpCodes::Print => {
                let to_print = self.stack.pop().unwrap();
                print!("{to_print}");
                self.program_counter += 1;
            }
            OpCodes::Println => {
                let to_print = self.stack.pop().unwrap();
                println!("{to_print}");
                self.program_counter += 1;
            }
            OpCodes::LoadPoison => {
                self.stack.push(AnacondaValue::Poison);
                self.program_counter += 1;
            }
            OpCodes::LoadTrue => {
                self.stack.push(AnacondaValue::Bool(true));
                self.program_counter += 1;
            }
            OpCodes::LoadFalse => {
                self.stack.push(AnacondaValue::Bool(false));
                self.program_counter += 1;
            }
            OpCodes::StartOfFunctionDefinition => {
                self.program_counter += 1;
                let idx = self.bytecode.read_usize(self.program_counter);
                self.program_counter += USIZE_BYTES;
                self.stack.push(AnacondaValue::Function(
                    self.function_definitions.data.get(&idx).unwrap().clone(),
                ));

                // Skip through the rest of the function.
                let mut start_vs_end = 1isize;
                while start_vs_end > 0 {
                    match self.current_opcode() {
                        OpCodes::StartOfFunctionDefinition => start_vs_end += 1,
                        OpCodes::EndOfFunctionDefinition => start_vs_end -= 1,
                        _ => (),
                    };
                    self.program_counter += self.current_opcode_len();
                }
            }

            OpCodes::BeginBlock => {
                self.stack_frames.push(StackFrame::Scope(Scope::new()));
                self.program_counter += 1;
            }
            OpCodes::EndBlock => {
                let v = self.stack_frames.pop().unwrap();
                if !matches!(v, StackFrame::Scope(_)) {
                    panic!("Reached end of block but the last stackframe was not a block! This should never happen! Here's what we got instead: {:#?}", v);
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
            | OpCodes::ModuloAndAssign
            | OpCodes::ExponentAndAssign => {
                macro_rules! assign_op {
                    ($e: tt) => {
                        {
                            self.program_counter += 1;
                            let from_stack_var = self.stack.pop().unwrap();
                            let idx = self.bytecode.read_usize(self.program_counter);
                            self.program_counter += USIZE_BYTES;
                            let var = self.get_var_by_index_mut(idx);

                            match (var, from_stack_var) {
                                (AnacondaValue::Int(val), AnacondaValue::Int(from_stack)) => {
                                    if opcode == OpCodes::BitshiftLeftAndAssign {
                                        *val *= ibig!(2).pow((from_stack % (IBig::from(usize::MAX) + 1usize)).try_into().unwrap())
                                    } else if opcode == OpCodes::BitshiftRightAndAssign {
                                        *val /= ibig!(2).pow((from_stack % (IBig::from(usize::MAX) + 1usize)).try_into().unwrap())
                                    } else if opcode == OpCodes::ExponentAndAssign {
                                        *val = val.pow(from_stack.try_into().unwrap())
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
                                (_, AnacondaValue::Poison) => {
                                    panic!("Cannot assign variable {} to poison!", self.identifier_literals.data.get(&idx).unwrap())
                                }
                                (other_1, other_2) => {
                                    if opcode == OpCodes::Assign {
                                        *other_1 = other_2;
                                    } else {
                                        panic!("Cannot perform operation {} on {other_1} and {other_2}", stringify!($e))

                                    }
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
                    /* These three don't matter because we handle them as special cases because IBig does not support <<=, >>= or **= */
                    OpCodes::BitshiftLeftAndAssign => assign_op!(!=),
                    OpCodes::BitshiftRightAndAssign => assign_op!(==),
                    OpCodes::ExponentAndAssign => assign_op!(>=),
                    /* END COMMENT */
                    OpCodes::BitwiseAndAndAssign => assign_op!(&=),
                    OpCodes::BitwiseOrAndAssign => assign_op!(|=),
                    OpCodes::BitwiseXorAndAssign => assign_op!(^=),
                    _ => unreachable!(),
                }
            }

            OpCodes::BooleanAnd => {
                self.program_counter += 1;
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
                if !first.as_bool() {
                    self.stack.push(first);
                } else {
                    self.stack.push(second);
                }
            }
            OpCodes::BooleanOr => {
                self.program_counter += 1;
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
                if first.as_bool() {
                    self.stack.push(first);
                } else {
                    self.stack.push(second);
                }
            }
            OpCodes::BooleanNot => {
                self.program_counter += 1;
                let val = self.stack.pop().unwrap();
                let inverse = !val.as_bool();
                self.stack.push(AnacondaValue::Bool(inverse));
            }
            OpCodes::PrintStack => {
                self.program_counter += 1;
                println!("{:#?}", self.stack)
            }
            OpCodes::EndOfFunctionDefinition => {
                panic!("Hit EndOfFunctionDefinition instruction. This should never happen.")
            }
            OpCodes::Break => {
                self.break_from_loop();
            }
            OpCodes::BreakIfFalse => {
                self.program_counter += 1;
                let val = self.stack.pop().unwrap();
                if !val.as_bool() {
                    self.break_from_loop();
                }
            }
            OpCodes::Continue => {
                while let Some(StackFrame::Scope(_)) = self.stack_frames.last() {
                    self.stack_frames.pop().unwrap();
                }
                match self.stack_frames.last().unwrap() {
                    StackFrame::Loop(l) => {
                        self.program_counter = l.address_of_start;
                    }
                    _ => panic!("Continue instruction was not inside a loop!"),
                }
            }

            OpCodes::Equals => {
                self.program_counter += 1;
                let second = self.stack.pop();
                let first = self.stack.pop();
                self.stack.push(AnacondaValue::Bool(first == second));
            }
            OpCodes::NotEquals => {
                self.program_counter += 1;

                let second = self.stack.pop();
                let first = self.stack.pop();
                self.stack.push(AnacondaValue::Bool(first != second));
            }
            OpCodes::GreaterThan => {
                self.program_counter += 1;
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
                self.stack.push(AnacondaValue::Bool(match (first, second) {
                    (AnacondaValue::Int(i1), AnacondaValue::Int(i2)) => i1 > i2,
                    (AnacondaValue::Bool(b1), AnacondaValue::Bool(b2)) => b1 & !b2,
                    (v1, v2) => {
                        panic!("Cannot perform > operation on {v1} and {v2}")
                    }
                }))
            }
            OpCodes::GreaterThanEquals => {
                self.program_counter += 1;
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
                self.stack.push(AnacondaValue::Bool(match (first, second) {
                    (AnacondaValue::Int(i1), AnacondaValue::Int(i2)) => i1 >= i2,
                    (AnacondaValue::Bool(b1), AnacondaValue::Bool(b2)) => b1 >= b2,
                    (v1, v2) => {
                        panic!("Cannot perform >= operation on {v1} and {v2}")
                    }
                }))
            }
            OpCodes::LessThan => {
                self.program_counter += 1;
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
                self.stack.push(AnacondaValue::Bool(match (first, second) {
                    (AnacondaValue::Int(i1), AnacondaValue::Int(i2)) => i1 < i2,
                    (AnacondaValue::Bool(b1), AnacondaValue::Bool(b2)) => !b1 & b2,
                    (v1, v2) => {
                        panic!("Cannot perform < operation on {v1} and {v2}")
                    }
                }))
            }
            OpCodes::LessThanEquals => {
                self.program_counter += 1;
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
                self.stack.push(AnacondaValue::Bool(match (first, second) {
                    (AnacondaValue::Int(i1), AnacondaValue::Int(i2)) => i1 <= i2,
                    (AnacondaValue::Bool(b1), AnacondaValue::Bool(b2)) => b1 <= b2,

                    (v1, v2) => {
                        panic!("Cannot perform <= operation on {v1} and {v2}")
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
                    panic!("Cannot perform unary bitwise not operation on {val}.")
                }
            }

            OpCodes::BitwiseAnd => {
                self.program_counter += 1;
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
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
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
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
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
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
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
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
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
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
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
                match (first, second) {
                    (AnacondaValue::Int(mut i1), AnacondaValue::Int(i2)) => {
                        i1 *= i2;
                        self.stack.push(AnacondaValue::Int(i1));
                    }
                    (AnacondaValue::String(s), AnacondaValue::Int(i))
                    | (AnacondaValue::Int(i), AnacondaValue::String(s)) => match i.cmp(&ibig!(0)) {
                        Ordering::Equal => {
                            self.stack.push(AnacondaValue::String(Cow::Borrowed("")));
                        }
                        Ordering::Less => {
                            panic!("Cannot multiply a string by a value less than 0");
                        }
                        Ordering::Greater => {
                            let s = s.as_ref().repeat(i.try_into().unwrap_or(usize::MAX));
                            self.stack.push(AnacondaValue::String(Cow::Owned(s)));
                        }
                    },
                    (v1, v2) => {
                        panic!("Cannot perform * operation on {v1} and {v2}")
                    }
                }
            }
            OpCodes::Divide => {
                self.program_counter += 1;
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
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
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
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
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
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
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
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
            OpCodes::Exponent => {
                self.program_counter += 1;
                let second = self.stack.pop().unwrap();
                let first = self.stack.pop().unwrap();
                match (first, second) {
                    (AnacondaValue::Int(mut i1), AnacondaValue::Int(i2)) => {
                        i1 = i1.pow(i2.try_into().unwrap());
                        self.stack.push(AnacondaValue::Int(i1));
                    }
                    (v1, v2) => {
                        panic!("Cannot perform ** operation on {v1} and {v2}")
                    }
                }
            }
            OpCodes::Goto => {
                let addr = self.bytecode.read_usize(self.program_counter + 1);
                self.program_counter = addr;
            }
            OpCodes::IfTrueGoto => {
                let val = self.stack.pop().unwrap();
                if val.as_bool() {
                    let addr = self.bytecode.read_usize(self.program_counter + 1);
                    self.program_counter = addr;
                } else {
                    self.program_counter += 1 + USIZE_BYTES;
                }
            }
            OpCodes::IfFalseGoto => {
                let val = self.stack.pop().unwrap();
                if !val.as_bool() {
                    let addr = self.bytecode.read_usize(self.program_counter + 1);
                    self.program_counter = addr;
                } else {
                    self.program_counter += 1 + USIZE_BYTES;
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
        self.current_opcode().len()
    }

    pub(crate) fn interpret_bytecode(&mut self) {
        while self.program_counter < self.bytecode.instructions.len() {
            self.interpret_next_instruction();
        }
        println!("{:#?}", self.stack_frames);
        println!("{:#?}", self.stack);
        // SAFETY: Calling collect_garbage is safe because we do not hold any pointer or references to anything owned by the GC.
        unsafe {
            self.gc.collect_garbage(&self.stack, &self.stack_frames);

        }
    }

    fn get_var_by_index<'b>(&'b self, idx: usize) -> Option<&'b AnacondaValue<'a>> {
        for i in (0..self.stack_frames.len()).rev() {
            let scope = &self.stack_frames[i];
            if let StackFrame::Scope(scope) = scope {
                if let Some(v) = scope.variables.get(&idx) {
                    return Some(v);
                }
            }
        }
        None
    }

    fn get_var_by_index_mut<'b>(&'b mut self, idx: usize) -> &'b mut AnacondaValue<'a> {
        let this = self as *mut Self;
        let len = self.stack_frames.len();
        for i in (0..len).rev() {
            let scope = unsafe { &mut (*this).stack_frames[i] as *mut StackFrame };
            if let StackFrame::Scope(scope) = unsafe { &mut (*scope) } {
                if let Some(v) = scope.variables.get_mut(&idx) {
                    return v;
                }
            }
        }
        for sf in self.stack_frames.iter_mut().rev() {
            if let StackFrame::Scope(scope) = sf {
                scope.variables.insert(idx, AnacondaValue::Poison);
                return scope.variables.get_mut(&idx).unwrap();
            }
        }
        unreachable!()
    }

    fn get_type_from_name(&self, name: &str) -> GcValue<Type<'a>> {
        self.stack_frames[0]
            .as_scope()
            .variables
            .get(self.identifier_literals.reverse_data.get(name).unwrap())
            .unwrap()
            .clone()
            .as_type()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Function {
    pub(crate) params: Vec<usize>,
    pub(crate) extra_args: bool,
    pub(crate) start_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Type<'a> {
    pub(crate) name: &'a str,
    pub(crate) fields: FastMap<usize, AnacondaValue<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AnacondaValue<'a> {
    Int(IBig),
    String(Cow<'a, str>),
    Function(GcValue<Function>),
    Bool(bool),
    Type(GcValue<Type<'a>>),
    Poison,
}

impl<'a> Display for AnacondaValue<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnacondaValue::Poison => panic!("Cannot display poison!"),
            AnacondaValue::String(s) => write!(f, "{s}"),
            AnacondaValue::Int(i) => write!(f, "{i}"),
            AnacondaValue::Function(fun_container) => {
                fun_container.with(|fun| write!(f, "{:#?}", fun))
            }
            AnacondaValue::Bool(b) => write!(f, "{b}"),
            AnacondaValue::Type(t) => t.with(|v| write!(f, "type({})", v.name)),
        }
    }
}

impl<'a> AnacondaValue<'a> {
    fn as_bool(&self) -> bool {
        match self {
            AnacondaValue::Int(i) => *i != ibig!(0),
            AnacondaValue::Poison => panic!("Cannot cast poison to a bool!"),
            AnacondaValue::Function(_) => true,
            AnacondaValue::String(s) => !s.is_empty(),
            AnacondaValue::Bool(b) => *b,
            AnacondaValue::Type(_) => true,
        }
    }

    fn as_type(&self) -> GcValue<Type<'a>> {
        match self {
            AnacondaValue::Type(t) => t.clone(),
            _ => panic!("{:#?} is not a type.", self),
        }
    }

    fn get_type(&self, interpreter: &BytecodeInterpreter<'a>) -> GcValue<Type<'a>> {
        match self {
            AnacondaValue::Type(t) => t.clone(),
            AnacondaValue::Int(_) => interpreter.get_type_from_name("integer"),
            AnacondaValue::String(_) => interpreter.get_type_from_name("string"),
            AnacondaValue::Bool(_) => interpreter.get_type_from_name("boolean"),
            AnacondaValue::Poison => panic!("Cannot get type of poison!"),
            AnacondaValue::Function(_) => interpreter.get_type_from_name("function"),
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{runtime::gc::{GarbageCollector, GcValue}};

    #[test]
    fn test_ub() -> Result<(), Box<dyn std::error::Error>> {
        let mut gc = GarbageCollector::new();
        let _val = GcValue::new("ABC".to_string(), &mut gc);
        unsafe {
            // SAFETY: We hold no outstanding references to anything owned by the GC
            // SAFETY: And calling collect_garbage with an empty stack and stack_frames is always safe under these circumstances.
            gc.collect_garbage(&vec![], &vec![]);

        }
        
        Ok(())
    }
}
