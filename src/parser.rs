use crate::lexer::lex::{Lexer, Token};
mod ast;
use ast::Ast;
use color_eyre::eyre::eyre;
use color_eyre::Result;

use self::ast::Statement;
use self::bytecode::Program;
pub(crate) mod bytecode;
pub(crate) fn parse<'a>(program: &'a str) -> Result<Program<'a>> {
    let mut lexer = Lexer::new(program);
    let tokens: Vec<Token<'a>> = lexer.collect_tokens().map_err(|e| eyre!(e.to_string()))?;
    Ast::new(tokens).parse().map_err(|s| eyre!(s.to_string()))
}
