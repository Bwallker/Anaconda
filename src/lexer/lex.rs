use counted_array::counted_array;
use lazy_static::lazy_static;
use pretty_assertions::assert_eq;
use std::fmt::{Display, Formatter};
use std::iter::Peekable;
struct RemoveLast<I: Iterator<Item = T>, T> {
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

trait RemoveLastTrait<I: Iterator<Item = T>, T> {
    fn remove_last(self) -> RemoveLast<I, T>;
}

impl<I: Iterator<Item = T>, T> RemoveLastTrait<I, T> for I {
    fn remove_last(self) -> RemoveLast<I, T> {
        RemoveLast {
            iter: self.peekable(),
        }
    }
}
macro_rules! hex_pattern {
    () => {
        b'_' | b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F'
    }
}

macro_rules! bin_pattern {
    () => {
        b'_' | b'0' | b'1'
    };
}

macro_rules! dec_pattern {
    () => {
        b'_' | b'0'..=b'9'
    };
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum TokenType {
    Int,
    Decimal,
    String,
    LineComment,
    BlockComment,
    Identifier,
    LParen,
    RParen,
    Newline,
    Eoi,
    Plus,
    PlusAssign,
    Minus,
    MinusAssign,
    Multiply,
    MultiplyAssign,
    Divide,
    DivideAssign,
    Assign,
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessThanEquals,
    GreaterThanEquals,
    Whitespace,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub struct Position {
    line_number: LineNumber,
    column_number: ColumnNumber,
    index: usize,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub struct Token {
    start: Position,
    len: usize,
    end: Position,
    token_type: TokenType,
}

type LineNumber = usize;
type ColumnNumber = usize;

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct LexerError<'a> {
    line_number: LineNumber,
    column_number: ColumnNumber,
    index: usize,
    error_message: String,
    input: &'a str,
}

impl<'a> LexerError<'a> {
    fn from(
        line_number: LineNumber,
        column_number: ColumnNumber,
        index: usize,
        error_message: String,
        input: &'a str,
    ) -> LexerError<'a> {
        LexerError {
            line_number,
            column_number,
            error_message,
            input,
            index,
        }
    }
    fn find_line(&self) -> &str {
        let mut current_line = 0;
        let mut start_index: LineNumber = 0;
        let mut chars = self.input.chars();
        while current_line < self.line_number {
            let c = chars.next().unwrap();
            if c == '\n' {
                current_line += 1;
            }
            start_index += 1;
        }
        let mut end_index = start_index + 1;
        while chars.next().is_some() && chars.next().unwrap() != '\n' {
            end_index += 1;
        }

        &self.input[start_index..=end_index + 1]
    }

    fn place_caret(&self) -> String {
        let mut i = 0;
        let mut res = String::new();

        while i < self.column_number {
            res.push(' ');
            i += 1;
        }
        res.push('^');
        i += self.index - self.column_number;
        while i < self.input.len() && &self.input[i..=i] != "\n" {
            res.push(' ');
            i += 1;
        }
        res
    }
}

impl Display for LexerError<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}\n{}\n{}:{} {}",
            self.find_line(),
            self.place_caret(),
            self.column_number,
            self.line_number,
            self.error_message
        )
    }
}

type LexerResult<'a, T> = Result<T, LexerError<'a>>;

trait Parser: Send + Sync {
    fn parse<'a>(&self, lexer: &'a mut Lexer) -> LexerResult<'a, Token>;
}

trait Boxed {
    fn boxed(self) -> Box<Self>;
}

impl<T> Boxed for T
where
    T: Into<Box<T>>,
{
    fn boxed(self) -> Box<Self> {
        Box::from(self)
    }
}
struct TagParser {
    tag: &'static str,
    token_type: TokenType,
}

impl Parser for TagParser {
    fn parse<'a>(&self, lexer: &'a mut Lexer) -> LexerResult<'a, Token> {
        let slice = &lexer.input[lexer.index..lexer.index + self.tag.len()];
        if slice != self.tag {
            return Err(lexer.create_lex_error(format!("Expected {}", self.tag)));
        }
        Ok(lexer.create_token(self.token_type, self.tag.len()))
    }
}

struct HexIntParser;

impl Parser for HexIntParser {
    fn parse<'a>(&self, lexer: &'a mut Lexer) -> LexerResult<'a, Token> {
        let mut index = lexer.index;
        if index == lexer.input.len() - 1 {
            return Err(
                lexer.create_lex_error("Not enough input left for a hexadecimal integer".into())
            );
        }
        if &lexer.input[index..=index + 1] != "0x" && &lexer.input[index..=index + 1] != "0X" {
            return Err(
                lexer.create_lex_error("Hexadecimal integers must start with 0x or 0X".into())
            );
        }
        index += 2;
        if index >= lexer.input.len() {
            return Err(lexer
                .create_lex_error("Reached EOI while trying to parse hexadecimal integer".into()));
        }
        if !matches!(lexer.input.as_bytes()[index], hex_pattern!())
            || lexer.input.as_bytes()[index] == b'_'
        {
            return Err(
                lexer.create_lex_error("Hexadecimal integers must start a valid digit".into())
            );
        }
        loop {
            let next = lexer.input.as_bytes()[index];
            match next {
                hex_pattern!() => index += 1,
                _ => break,
            }
            if index >= lexer.input.len() {
                break;
            }
        }
        Ok(lexer.create_token(TokenType::Int, index - lexer.index))
    }
}

struct BinIntParser;

impl Parser for BinIntParser {
    fn parse<'a>(&self, lexer: &'a mut Lexer) -> LexerResult<'a, Token> {
        let mut index = lexer.index;
        if index == lexer.input.len() - 1 {
            return Err(lexer.create_lex_error("Not enough input left for a binary integer".into()));
        }
        if &lexer.input[index..=index + 1] != "0b" && &lexer.input[index..=index + 1] != "0B" {
            return Err(lexer.create_lex_error("Binary integers must start with 0b or 0B".into()));
        }
        index += 2;
        if index >= lexer.input.len() {
            return Err(
                lexer.create_lex_error("Reached EOI while trying to parse binary integer".into())
            );
        }
        if !matches!(lexer.input.as_bytes()[index], bin_pattern!())
            || lexer.input.as_bytes()[index] == b'_'
        {
            return Err(
                lexer.create_lex_error("Binary integers must start with a valid digit".into())
            );
        }
        loop {
            let next = lexer.input.as_bytes()[index];
            match next {
                bin_pattern!() => index += 1,
                _ => break,
            }
            if index >= lexer.input.len() {
                break;
            }
        }
        Ok(lexer.create_token(TokenType::Int, index - lexer.index))
    }
}

struct DecIntParser;

impl Parser for DecIntParser {
    fn parse<'a>(&self, lexer: &'a mut Lexer) -> LexerResult<'a, Token> {
        let mut index = lexer.index;
        if index >= lexer.input.len() {
            return Err(
                lexer.create_lex_error("Reached EOI while trying to parse decimal integer".into())
            );
        }
        if !matches!(lexer.input.as_bytes()[index], dec_pattern!())
            || lexer.input.as_bytes()[index] == b'_'
        {
            return Err(
                lexer.create_lex_error("Decimal integers must start with a valid digit".into())
            );
        }
        loop {
            let next = lexer.input.as_bytes()[index];
            match next {
                dec_pattern!() => index += 1,
                _ => break,
            }
            if index >= lexer.input.len() {
                break;
            }
        }
        Ok(lexer.create_token(TokenType::Int, index - lexer.index))
    }
}

struct EOIParser;
impl Parser for EOIParser {
    fn parse<'a>(&self, lexer: &'a mut Lexer) -> LexerResult<'a, Token> {
        if lexer.index == lexer.input.len() {
            return Ok(lexer.create_token(TokenType::Eoi, 0));
        }
        return Err(lexer.create_lex_error("EOI is not yet reached.".into()));
    }
}

struct WhitespaceParser;
impl Parser for WhitespaceParser {
    fn parse<'a>(&self, lexer: &'a mut Lexer) -> LexerResult<'a, Token> {
        let mut len = 0;
        if !matches!(
            &lexer.input[lexer.index + len..=lexer.index + len],
            " " | "\t",
        ) {
            return Err(
                lexer.create_lex_error("Whitespace token did not contain any whitespace".into())
            );
        }
        while matches!(
            &lexer.input[lexer.index + len..=lexer.index + len],
            " " | "\t",
        ) {
            len += 1;
        }
        Ok(lexer.create_token(TokenType::Whitespace, len))
    }
}
fn tag(tag: &'static str, token_type: TokenType) -> TagParser {
    TagParser { tag, token_type }
}
#[derive(Debug)]
pub struct Lexer<'a> {
    line_number: LineNumber,
    column_number: ColumnNumber,
    input: &'a str,
    index: usize,
}

impl<'a> Lexer<'a> {
    pub(crate) fn new(input: &'a str) -> Self {
        Lexer {
            line_number: 0,
            column_number: 0,
            input,
            index: 0,
        }
    }

    pub(crate) fn collect_tokens(&mut self) -> LexerResult<Vec<Token>> {
        let this = self as *mut Self;
        let mut res = vec![];
        loop {
            // SAFETY: Hack to get around the borrow checker not being able to reason about loops. self only gets permanently borrowed when the error is returned.
            let next = unsafe { (*this).yield_token()? };
            res.push(next);
            if next.token_type == TokenType::Eoi {
                return Ok(res);
            }
        }
    }

    fn create_token(&mut self, token_type: TokenType, len: usize) -> Token {
        println!("Called create_token with {token_type:?}");
        Token {
            token_type,
            len,
            start: Position {
                index: self.index,
                column_number: self.column_number,
                line_number: self.line_number,
            },
            end: self.calculate_offset(len),
        }
    }

    fn calculate_offset(&mut self, offset: usize) -> Position {
        let mut line_number = self.line_number;
        let mut column_number = self.column_number;
        let mut index = self.index;
        println!("{index}, {offset}, {}", self.input.len());
        for i in self.index..self.index + offset {
            index += 1;
            let c = &self.input[i..=i];
            if c == "\n" {
                column_number = 0;
                line_number += 1;
            } else {
                column_number += 1;
            }
        }
        self.line_number = line_number;
        self.column_number = column_number;
        self.index = index;
        Position {
            index,
            line_number,
            column_number,
        }
    }

    fn yield_token(&mut self) -> LexerResult<Token> {
        counted_array!(lazy_static PARSERS: [Box<dyn Parser>; _] = [EOIParser.boxed(),
            HexIntParser.boxed(),
            BinIntParser.boxed(),
            DecIntParser.boxed(),
            tag("!=", TokenType::NotEquals).boxed(),
            tag("==", TokenType::Equals).boxed(),
            tag(">=", TokenType::GreaterThanEquals).boxed(),
            tag("<=", TokenType::LessThanEquals).boxed(),
            tag(">", TokenType::GreaterThan).boxed(),
            tag("<", TokenType::LessThan).boxed(),
            tag("(", TokenType::LParen).boxed(),
            tag(")", TokenType::RParen).boxed(),
            tag("/=", TokenType::DivideAssign).boxed(),
            tag("*=", TokenType::MultiplyAssign).boxed(),
            tag("+=", TokenType::PlusAssign).boxed(),
            tag("-=", TokenType::MinusAssign).boxed(),
            tag("/", TokenType::Divide).boxed(),
            tag("*", TokenType::Multiply).boxed(),
            tag("+", TokenType::Plus).boxed(),
            tag("-", TokenType::Minus).boxed(),
            tag("=", TokenType::Assign).boxed(),
            WhitespaceParser.boxed()]);
        if self.index > self.input.len() {
            self.index = self.input.len() - 1;
            return Err(self.create_lex_error("Lexer has ran past the end of the input".into()));
        }
        for (_i, parser) in PARSERS.iter().remove_last().enumerate() {
            let res = parser.parse(self);
            match res {
                Err(_) => continue,
                Ok(v) => return Ok(v),
            }
        }
        let res = PARSERS[PARSERS.len() - 1].parse(self);
        return match res {
            Err(_) => Err(self.create_lex_error("Unknown token!".into())),
            Ok(v) => Ok(v),
        };
    }
    fn create_lex_error(&self, message: String) -> LexerError {
        LexerError::from(
            self.line_number,
            self.column_number,
            self.index,
            message,
            self.input,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::lexer::lex::{Position, Token, TokenType};

    use super::Lexer;
    use pretty_assertions::assert_eq;
    #[test]
    fn test_lexer_1() {
        pretty_assertions::assert_eq!(
            Lexer::new("2 != 3").collect_tokens().unwrap(),
            vec![
                Token {
                    token_type: TokenType::Int,
                    start: Position {
                        column_number: 0,
                        line_number: 0,
                        index: 0,
                    },
                    len: 1,
                    end: Position {
                        column_number: 1,
                        line_number: 0,
                        index: 1,
                    },
                },
                Token {
                    token_type: TokenType::Whitespace,
                    start: Position {
                        column_number: 1,
                        line_number: 0,
                        index: 1,
                    },
                    len: 1,
                    end: Position {
                        column_number: 2,
                        line_number: 0,
                        index: 2,
                    },
                },
                Token {
                    token_type: TokenType::NotEquals,
                    start: Position {
                        column_number: 2,
                        line_number: 0,
                        index: 2,
                    },
                    len: 2,
                    end: Position {
                        column_number: 4,
                        line_number: 0,
                        index: 4,
                    },
                },
                Token {
                    token_type: TokenType::Whitespace,
                    start: Position {
                        column_number: 4,
                        line_number: 0,
                        index: 4,
                    },
                    len: 1,
                    end: Position {
                        column_number: 5,
                        line_number: 0,
                        index: 5,
                    },
                },
                Token {
                    token_type: TokenType::Int,
                    start: Position {
                        column_number: 5,
                        line_number: 0,
                        index: 5,
                    },
                    len: 1,
                    end: Position {
                        column_number: 6,
                        line_number: 0,
                        index: 6,
                    },
                },
                Token {
                    end: Position { line_number: 0, column_number: 6, index: 6 },
                    start: Position { line_number: 0, column_number: 6, index: 6 },
                    len: 0,
                    token_type: TokenType::Eoi,
                }
                //Token::Int("2"),
                //Token::Whitespace(" "),
                //Token::NotEquals,
                //Token::Whitespace(" "),
                //Token::Int("3"),
            ]
        )
    }
}
