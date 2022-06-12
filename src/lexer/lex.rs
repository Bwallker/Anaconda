use counted_array::counted_array;
use lazy_static::lazy_static;
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

macro_rules! whitespace_pattern {
    () => {
        b' ' | b'\t'
    };
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum TokenType {
    Eoi,
    Comment(CommentTokenType),
    Operator(OperatorTokenType),
    Literal(LiteralTokenType),
    Whitespace,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum LiteralTokenType {
    Int,
    Decimal,
    String,
    Identifier,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum CommentTokenType {
    Line,
    Block,
    DocsLine,
    DocsBlock,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum OperatorTokenType {
    // Arithmetic.
    Plus,
    PlusAssign,
    Minus,
    MinusAssign,
    Multiply,
    MultiplyAssign,
    Divide,
    DivideAssign,
    BitShiftLeft,
    BitShiftLeftAssign,
    BitShiftRight,
    BitShiftRightAssign,
    BitwiseAnd,
    BitwiseAndAssign,
    BitwiseOr,
    BitwiseOrAssign,
    BitwiseXor,
    BitwiseXorAssign,
    BitwiseNot,
    BitwiseNotAssign,
    // Boolean.
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessThanEquals,
    GreaterThanEquals,
    // Structural.
    Comma,
    LParen,
    RParen,
    LSquare,
    RSquare,
    Assign,
    Terminator,

}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub struct Position {
    line_number: LineNumber,
    column_number: ColumnNumber,
    index: usize,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub struct Token<'a> {
    contents: &'a str,
    start: Position,
    len: usize,
    end: Position,
    token_type: TokenType,
}

type LineNumber = usize;
type ColumnNumber = usize;

#[derive(Eq, PartialEq, Debug, Clone, Copy)]

enum LexerErrorType {
    /// For potential tokens that are clearly malformed versions of valid tokens. For instance 0x without a digit.
    Malformed,
    /// For potential tokens that are completely invalid for the given parser.
    Unknown,
}
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct LexerError<'a> {
    line_number: LineNumber,
    column_number: ColumnNumber,
    index: usize,
    error_message: String,
    input: &'a str,
    error_type: LexerErrorType,
    len: usize,
}

impl<'a> LexerError<'a> {
    fn from(
        line_number: LineNumber,
        column_number: ColumnNumber,
        index: usize,
        error_message: String,
        input: &'a str,
        error_type: LexerErrorType,
        len: usize,
    ) -> LexerError<'a> {
        LexerError {
            line_number,
            column_number,
            error_message,
            input,
            index,
            error_type,
            len,
        }
    }
    fn find_line(&self) -> String {
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
        let mut end_index = start_index.wrapping_sub(1);
        let mut next = chars.next();
        while next.is_some() && next.unwrap() != '\n' {
            next = chars.next();
            end_index = end_index.wrapping_add(1);
        }
        let mut res = String::from(&self.input[start_index..=end_index]);
        if end_index + 1 < self.input.len() {
            res.push_str(&self.input[end_index + 1..=end_index + 1]);
        } else {
            res.push(' ');
        }
        res
    }

    fn place_caret(&self) -> String {
        let mut i = 0;
        let mut res = String::new();

        while i < self.column_number {
            res.push(' ');
            i += 1;
        }
        for _ in 0..self.len {
            res.push('^');

            i += 1;
        }
        i += self.index - self.column_number;
        while i < self.input.len() && !matches!(&self.input[i..=i], "\n" | "\t" | "\r" | " ") {
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
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>>;
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
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>> {
        let end = lexer.index + self.tag.len();
        if end > lexer.input.len() {
            return Err(lexer.create_parser_lex_error());
        }
        let slice = &lexer.input[lexer.index..end];
        if slice != self.tag {
            return Err(lexer.create_parser_lex_error());
        }
        Ok(lexer.create_token(self.token_type, self.tag.len()))
    }
}

struct HexIntParser;

impl Parser for HexIntParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>>  {
        let mut index = lexer.index;
        if index == lexer.input.len() - 1 {
            return Err(lexer.create_parser_lex_error());
        }
        if &lexer.input[index..=index + 1] != "0x" && &lexer.input[index..=index + 1] != "0X" {
            return Err(lexer.create_parser_lex_error());
        }
        index += 2;
        if index >= lexer.input.len() {
            return Err(lexer.create_malformed_lex_error_with_len(
                "Reached EOI while trying to parse hexadecimal integer".into(),
                3,
            ));
        }
        if !matches!(lexer.input.as_bytes()[index], hex_pattern!())
            || lexer.input.as_bytes()[index] == b'_'
        {
            return Err(lexer.create_malformed_lex_error_with_len(
                "Hexadecimal integers must start with a valid digit".into(),
                3,
            ));
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
        Ok(lexer.create_token(TokenType::Literal(LiteralTokenType::Int), index - lexer.index))
    }
}

struct BinIntParser;

impl Parser for BinIntParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>>  {
        let mut index = lexer.index;
        if index == lexer.input.len() - 1 {
            return Err(lexer.create_parser_lex_error());
        }
        if &lexer.input[index..=index + 1] != "0b" && &lexer.input[index..=index + 1] != "0B" {
            return Err(lexer.create_parser_lex_error());
        }
        index += 2;
        if index >= lexer.input.len() {
            return Err(lexer.create_malformed_lex_error_with_len(
                "Reached EOI while trying to parse binary integer".into(),
                3,
            ));
        }
        if !matches!(lexer.input.as_bytes()[index], bin_pattern!())
            || lexer.input.as_bytes()[index] == b'_'
        {
            return Err(lexer.create_malformed_lex_error_with_len(
                "Binary integers must start with a valid digit".into(),
                3,
            ));
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
        Ok(lexer.create_token(TokenType::Literal(LiteralTokenType::Int), index - lexer.index))
    }
}

struct DecIntParser;

impl Parser for DecIntParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>>  {
        let mut index = lexer.index;

        if index >= lexer.input.len() {
            return Err(lexer.create_parser_lex_error());
        }
        if !matches!(lexer.input.as_bytes()[index], dec_pattern!())
            || lexer.input.as_bytes()[index] == b'_'
        {
            return Err(lexer.create_parser_lex_error());
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
        Ok(lexer.create_token(TokenType::Literal(LiteralTokenType::Int), index - lexer.index))
    }
}

struct EOIParser;
impl Parser for EOIParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>>  {
        if lexer.index == lexer.input.len() {
            return Ok(lexer.create_token(TokenType::Eoi, 0));
        }
        return Err(lexer.create_parser_lex_error());
    }
}

struct WhitespaceParser;
impl Parser for WhitespaceParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>>  {
        let mut len = 0;
        if !matches!(
            lexer.input.as_bytes()[lexer.index + len],
            whitespace_pattern!(),
        ) {
            return Err(lexer.create_parser_lex_error());
        }
        while matches!(
            lexer.input.as_bytes()[lexer.index + len],
            whitespace_pattern!(),
        ) {
            len += 1;
            if lexer.index + len >= lexer.input.len() {
                break;
            }
        }
        Ok(lexer.create_token(TokenType::Whitespace, len))
    }
}

struct LineCommentParser {
    prefix: &'static str,
    return_type: CommentTokenType,
}

impl Parser for LineCommentParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>>  {
        if lexer.index + self.prefix.len() > lexer.input.len() {
            // Not enough input left for our prefix to fit in -> cannot be a line comment.
            return Err(lexer.create_parser_lex_error());
        }
        if &lexer.input[lexer.index..lexer.index+self.prefix.len()] != self.prefix {
            return Err(lexer.create_parser_lex_error());
        }
        let mut index = self.prefix.len() + lexer.index;

        while index < lexer.input.len() {
            index += 1;
            if lexer.input.as_bytes()[index] == b'\n' {
                index += 1;
                break;
            }
        }

        Ok(lexer.create_token(TokenType::Comment(self.return_type), index - lexer.index))
    }
}
fn tag(tag: &'static str, token_type: TokenType) -> TagParser {
    TagParser { tag, token_type }
}
#[derive(Debug, Copy, Clone)]
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
            let next = self.yield_token()?;
            res.push(next);
            if next.token_type == TokenType::Eoi {
                return Ok(res);
            }
        }
    }

    fn create_token(&mut self, token_type: TokenType, len: usize) -> Token<'a> {
        let index = self.index;
        let column_number = self.column_number;
        let line_number = self.line_number;
        let end = self.calculate_offset(len);
        let contents = if end.index < self.input.len() {
            &self.input[index..=end.index]
        } else {
            ""
        };
        Token {
            token_type,
            len,
            start: Position {
                index,
                column_number,
                line_number,
            },
            end,
            contents,
        }
    }

    fn calculate_offset(&mut self, offset: usize) -> Position {
        let mut line_number = self.line_number;
        let mut column_number = self.column_number;
        let mut index = self.index;
        for i in self.index..(self.index + offset - 1) {
            index += 1;
            let c = &self.input[i..=i];
            if c == "\n" {
                column_number = 0;
                line_number += 1;
            } else {
                column_number += 1;
            }
        }
        let last = &self.input[self.index + offset - 1..=self.index + offset - 1];

        self.line_number = line_number;
        self.column_number = column_number;
        self.index = index + 1;
        if last == "\n" {
            self.column_number = 0;
            self.line_number += 1;
        } else {
            self.column_number += 1;
        }
        Position {
            index,
            line_number,
            column_number,
        }
    }
    fn yield_token(&mut self) -> LexerResult<'a, Token<'a>> {
        use OperatorTokenType::*;
        use TokenType::*;
        
        counted_array!(lazy_static PARSERS: [Box<dyn Parser>; _] = [EOIParser.boxed(),
            HexIntParser.boxed(),
            BinIntParser.boxed(),
            DecIntParser.boxed(),
            LineCommentParser {prefix: "///", return_type: CommentTokenType::DocsLine}.boxed(),
            LineCommentParser {prefix: "//", return_type: CommentTokenType::Line}.boxed(),
            tag(";", Operator(Terminator)).boxed(),
            tag("\r\n", Operator(Terminator)).boxed(),
            tag("\r", Operator(Terminator)).boxed(),
            tag("\n", Operator(Terminator)).boxed(),
            tag("<<=", Operator(BitShiftLeftAssign)).boxed(),
            tag("<<", Operator(BitShiftLeft)).boxed(),
            tag(">>=", Operator(BitShiftRightAssign)).boxed(),
            tag(">>", Operator(BitShiftRight)).boxed(),
            tag("&=", Operator(BitwiseAndAssign)).boxed(),
            tag("&", Operator(BitwiseAnd)).boxed(),
            tag("|=", Operator(BitwiseOrAssign)).boxed(),
            tag("|", Operator(BitwiseOr)).boxed(),
            tag("~=", Operator(BitwiseNotAssign)).boxed(),
            tag("~", Operator(BitwiseNot)).boxed(),
            tag("^=", Operator(BitwiseXorAssign)).boxed(),
            tag("^", Operator(BitwiseXor)).boxed(),

            tag("!=", Operator(NotEquals)).boxed(),
            tag("==", Operator(Equals)).boxed(),
            tag(">=", Operator(GreaterThanEquals)).boxed(),
            tag("<=", Operator(LessThanEquals)).boxed(),
            tag(">", Operator(GreaterThan)).boxed(),
            tag("<", Operator(LessThan)).boxed(),
            tag("(", Operator(LParen)).boxed(),
            tag(")", Operator(RParen)).boxed(),

            tag("/=", Operator(DivideAssign)).boxed(),
            tag("*=", Operator(MultiplyAssign)).boxed(),
            tag("+=", Operator(PlusAssign)).boxed(),
            tag("-=", Operator(MinusAssign)).boxed(),
            tag("/", Operator(Divide)).boxed(),
            tag("*", Operator(Multiply)).boxed(),
            tag("+", Operator(Plus)).boxed(),
            tag("-", Operator(Minus)).boxed(),
            tag("=", Operator(Assign)).boxed(),
            tag(",", Operator(Comma)).boxed(),

            tag("[", Operator(LSquare)).boxed(),
            tag("]", Operator(RSquare)).boxed(),
            
            WhitespaceParser.boxed()]);
        if self.index > self.input.len() {
            self.index = self.input.len() - 1;
            return Err(
                self.create_unknown_lex_error("Lexer has ran past the end of the input".into())
            );
        }
        for (_i, parser) in PARSERS.iter().remove_last().enumerate() {
            let res = parser.parse(self);

            match res {
                Err(e) => match e.error_type {
                    LexerErrorType::Malformed => return Err(e),
                    LexerErrorType::Unknown => continue,
                },
                Ok(v) => return Ok(v),
            }
        }
        let res = PARSERS[PARSERS.len() - 1].parse(self);
        return match res {
            Err(_) => Err(
                self.create_unknown_lex_error_with_len(
                    format!(
                        "Unknown Token starting with '{}'!",
                        &self.input[self.index..=self.index]
                    ),
                    self.distance_to_whitespace(),
                )
            ),
            Ok(v) => Ok(v),
        };
    }

    fn distance_to_whitespace(&self) -> usize {
        let mut index = self.index;
        while index < self.input.len()
            && !matches!(self.input.as_bytes()[index], whitespace_pattern!())
        {
            index += 1;
        }

        index - self.index
    }
    fn create_parser_lex_error(&self) -> LexerError<'a> {
        self.create_lex_error_base(String::new(), LexerErrorType::Unknown, 1)
    }

    fn create_unknown_lex_error(&self, message: String) -> LexerError<'a> {
        self.create_lex_error_base(message, LexerErrorType::Unknown, 1)
    }

    fn create_unknown_lex_error_with_len(&self, message: String, len: usize) -> LexerError<'a> {
        self.create_lex_error_base(message, LexerErrorType::Unknown, len)
    }

    fn create_lex_error_base(
        &self,
        message: String,
        error_type: LexerErrorType,
        len: usize,
    ) -> LexerError<'a> {
        LexerError::from(
            self.line_number,
            self.column_number,
            self.index,
            message,
            self.input,
            error_type,
            len,
        )
    }

    fn create_malformed_lex_error_with_len(&self, message: String, len: usize) -> LexerError<'a> {
        self.create_lex_error_base(message, LexerErrorType::Malformed, len)
    }
}

#[cfg(test)]
mod tests {
    use crate::lexer::lex::{LexerError, OperatorTokenType, Position, Token, TokenType, LiteralTokenType};

    use super::Lexer;
    #[test]
    fn test_lexer_1() {
        pretty_assertions::assert_eq!(
            Lexer::new("2 != 3").collect_tokens().unwrap(),
            vec![
                Token {
                    token_type: TokenType::Literal(LiteralTokenType::Int),
                    start: Position {
                        column_number: 0,
                        line_number: 0,
                        index: 0,
                    },
                    len: 1,
                    end: Position {
                        column_number: 0,
                        line_number: 0,
                        index: 0,
                    },
                    contents: "2",
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
                        column_number: 1,
                        line_number: 0,
                        index: 1,
                    },
                    contents: " ",
                },
                Token {
                    token_type: TokenType::Operator(OperatorTokenType::NotEquals),
                    start: Position {
                        column_number: 2,
                        line_number: 0,
                        index: 2,
                    },
                    len: 2,
                    end: Position {
                        column_number: 3,
                        line_number: 0,
                        index: 3,
                    },
                    contents: "!=",
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
                        column_number: 4,
                        line_number: 0,
                        index: 4,
                    },
                    contents: " ",
                },
                Token {
                    token_type: TokenType::Literal(LiteralTokenType::Int),
                    start: Position {
                        column_number: 5,
                        line_number: 0,
                        index: 5,
                    },
                    len: 1,
                    end: Position {
                        column_number: 5,
                        line_number: 0,
                        index: 5,
                    },
                    contents: "3",
                },
                Token {
                    token_type: TokenType::Eoi,
                    start: Position { line_number: 0, column_number: 6, index: 6 },
                    end: Position { line_number: 0, column_number: 6, index: 6 },
                    len: 0,
                    contents: "",
                }
                //Token::Int("2"),
                //Token::Whitespace(" "),
                //Token::NotEquals,
                //Token::Whitespace(" "),
                //Token::Int("3"),
            ]
        )
    }
    #[test]
    fn test_lexer_2() {
        pretty_assertions::assert_eq!(
            Lexer::new("0x").collect_tokens(),
            Err(LexerError {
                error_message: "Reached EOI while trying to parse hexadecimal integer".into(),
                index: 0,
                line_number: 0,
                column_number: 0,
                input: "0x",
                error_type: crate::lexer::lex::LexerErrorType::Malformed,
                len: 3,
            })
        );
    }

    #[test]
    fn test_lexer_3() {
        pretty_assertions::assert_eq!(
            Lexer::new("0xh").collect_tokens(),
            Err(LexerError {
                error_message: "Hexadecimal integers must start with a valid digit".into(),
                index: 0,
                line_number: 0,
                column_number: 0,
                input: "0xh",
                error_type: crate::lexer::lex::LexerErrorType::Malformed,
                len: 3,
            })
        );
    }
    #[test]
    fn test_lexer_4() {
        pretty_assertions::assert_eq!(
            Lexer::new("1\n2\n3").collect_tokens().unwrap(),
            vec![
                Token {
                    token_type: TokenType::Literal(LiteralTokenType::Int),
                    start: Position {
                        line_number: 0,
                        column_number: 0,
                        index: 0,
                    },
                    len: 1,
                    end: Position {
                        line_number: 0,
                        column_number: 0,
                        index: 0,
                    },
                    contents: "1",
                },
                Token {
                    token_type: TokenType::Whitespace,
                    start: Position {
                        line_number: 0,
                        column_number: 1,
                        index: 1,
                    },
                    len: 1,
                    end: Position {
                        line_number: 0,
                        column_number: 1,
                        index: 1,
                    },
                    contents: "\n",
                },
                Token {
                    token_type: TokenType::Literal(LiteralTokenType::Int),
                    start: Position {
                        line_number: 1,
                        column_number: 0,
                        index: 2,
                    },
                    len: 1,
                    end: Position {
                        line_number: 1,
                        column_number: 0,
                        index: 2,
                    },
                    contents: "2",
                },
                Token {
                    token_type: TokenType::Whitespace,
                    start: Position {
                        line_number: 1,
                        column_number: 1,
                        index: 3,
                    },
                    len: 1,
                    end: Position {
                        line_number: 1,
                        column_number: 1,
                        index: 3,
                    },
                    contents: "\n",
                },
                Token {
                    token_type: TokenType::Literal(LiteralTokenType::Int),
                    start: Position {
                        line_number: 2,
                        column_number: 0,
                        index: 4,
                    },
                    len: 1,
                    end: Position {
                        line_number: 2,
                        column_number: 0,
                        index: 4,
                    },
                    contents: "3",
                },
                Token {
                    token_type: TokenType::Eoi,
                    start: Position {
                        line_number: 2,
                        column_number: 1,
                        index: 5
                    },
                    end: Position {
                        line_number: 2,
                        column_number: 1,
                        index: 5
                    },
                    len: 0,
                    contents: "",
                }
            ]
        );
    }
}
