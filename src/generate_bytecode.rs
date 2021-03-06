pub(crate) mod bytecode_generator;
mod set_has_return_value;
use crate::{parser::ast::Ast, runtime::gc::GarbageCollector};
use crate::runtime::bytecode::{Bytecode, OpCodes};
use self::bytecode_generator::GenerateBytecode;

pub(crate) fn generate_bytecode(ast: &mut Ast<'_>, gc: &mut GarbageCollector) -> Bytecode {
    let mut res = Bytecode::new();
    let block = ast.program.base_block.take().unwrap();
    block.gen_bytecode(&mut res, ast, gc);
    // EndBlock opcode to offset the global scope we add when initializing our VM.
    res.push_opcode(OpCodes::EndBlock);
    //println!("{:#?}", res.instructions.len());
    res
}