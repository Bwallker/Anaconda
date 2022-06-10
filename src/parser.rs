use crate::lexer::lex::Lexer;

use color_eyre::Result;
pub fn parse(program: &str) -> Result<()> {
    let _ast = Lexer::new(program).collect_tokens();
    Ok(())
}
