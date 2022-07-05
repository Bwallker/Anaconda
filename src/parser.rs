use crate::lexer::lex::{Lexer, Token};
pub(crate) mod ast;
use ast::Ast;
use color_eyre::eyre::eyre;
use color_eyre::Result;

pub(crate) fn parse<'a>(program: &'a str) -> Result<Ast<'a>> {
    let mut lexer = Lexer::new(program);
    let tokens: Vec<Token<'a>> = lexer.collect_tokens().map_err(|e| eyre!(e.to_string()))?;
    Ast::new(tokens, program).parse().map_err(|s| eyre!(s.to_string()))
}
