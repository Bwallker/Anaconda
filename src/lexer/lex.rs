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

trait CharAtIndex {
    fn char_at_index(&self, index: usize) -> Option<char>;
}

impl<T: AsRef<str>> CharAtIndex for T {
    /// Returns the first char in the string.
    fn char_at_index(&self, index: usize) -> Option<char> {
        let this = self.as_ref();
        this.get(index..).and_then(|x| x.chars().next())
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

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum LexerError<'a> {
    /// The next token is obviously syntactically invalid.
    Incorrect(Vec<LexerErrorContents<'a>>),
    /// The next token is of the wrong form for this parser, and may still be valid for another parser.
    WrongForm,
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct LexerErrorContents<'a> {
    line_number: LineNumber,
    column_number: ColumnNumber,
    index: usize,
    error_message: String,
    input: &'a str,
    len: usize,
}

impl<'a> LexerErrorContents<'a> {
    fn find_line(&self) -> &str {
        let mut current_line = 0usize;
        let mut start_index = 0;
        while current_line < self.line_number {
            if next_is_newline(self.input, &mut start_index, true) {
                current_line += 1;
            } else {
                start_index += 1;
            }
        }
        let mut end_index = start_index;
        loop {
            if end_index >= self.input.len() {
                break;
            }
            if self.input.get(end_index..=end_index) == Some("\n") {
                break;
            }
            if self.input.get(end_index..=end_index + 1) == Some("\r\n") {
                break;
            }
            if self.input.get(end_index..=end_index) == Some("\r") {
                break;
            }
            end_index += 1;
        }
        if end_index >= self.input.len() {
            end_index = self.input.len() - 1;
        }
        while self.input.as_bytes().get(end_index) == Some(&b'\r')
            || self.input.as_bytes().get(end_index) == Some(&b'\n')
        {
            end_index -= 1;
        }
        &self.input[start_index..=end_index]
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
        res
    }
    fn with_contents(
        line_number: LineNumber,
        column_number: ColumnNumber,
        index: usize,
        error_message: String,
        input: &'a str,
        len: usize,
    ) -> Self {
        LexerErrorContents {
            line_number,
            column_number,
            error_message,
            input,
            index,
            len,
        }
    }
}
impl<'a> LexerError<'a> {
    fn with_contents(
        line_number: LineNumber,
        column_number: ColumnNumber,
        index: usize,
        error_message: String,
        input: &'a str,
        len: usize,
    ) -> LexerError<'a> {
        LexerError::Incorrect(vec![LexerErrorContents {
            line_number,
            column_number,
            error_message,
            input,
            index,
            len,
        }])
    }

    /// line_number, column_number, index, error_message, input, len
    fn create_errors<const N: usize>(
        args: [(LineNumber, ColumnNumber, usize, String, &'a str, usize); N],
    ) -> LexerError<'a> {
        LexerError::Incorrect(Vec::from(
            args.map(|x| LexerErrorContents::with_contents(x.0, x.1, x.2, x.3, x.4, x.5)),
        ))
    }
}
impl Display for LexerError<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LexerError::Incorrect(errors) => {
                for e in errors {
                    write!(
                        f,
                        "{}\n{}\n------------\n{}:{} {}\n\n",
                        e.find_line(),
                        e.place_caret(),
                        e.line_number + 1,
                        e.column_number + 1,
                        e.error_message,
                    )?
                }
                Ok(())
            }
            LexerError::WrongForm => unreachable!(),
        }
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
        let slice = lexer.input.get(lexer.index..end);
        if slice != Some(self.tag) {
            return Err(LexerError::WrongForm);
        }
        Ok(lexer.create_token(self.token_type, self.tag.len()))
    }
}

fn tag(tag: &'static str, token_type: TokenType) -> TagParser {
    TagParser { tag, token_type }
}

struct StringParser;

impl Parser for StringParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>> {
        let mut index = lexer.index;
        if lexer.input.as_bytes().get(index) != Some(&b'\'') {
            return Err(LexerError::WrongForm);
        }
        index += 1;
        let mut backslashes_in_a_row = 0usize;
        loop {
            let next = lexer.input.as_bytes().get(index);
            if next == None {
                return Err(lexer.create_lex_error_with_len("This string never ends.".into(), 1));
            }
            let next = *next.unwrap();
            if backslashes_in_a_row % 2 == 0 && next == b'\'' {
                index += 1;
                break;
            }
            if next == b'\\' {
                backslashes_in_a_row += 1;
            } else {
                backslashes_in_a_row = 0;
            }
            index += 1;
        }

        Ok(lexer.create_token(
            TokenType::Literal(LiteralTokenType::String),
            index - lexer.index,
        ))
    }
}

struct IdentifierParser;

impl Parser for IdentifierParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>> {
        let mut index = lexer.index;

        let first = lexer.input.as_bytes().get(index);
        if first == None {
            return Err(LexerError::WrongForm);
        }
        let first = *first.unwrap() as char;
        if first.is_ascii() && !first.is_ascii_alphabetic() && first != '_' {
            return Err(lexer.create_lex_error(format!(
                "Variable names and function names must start with an ASCII alphabetic character or a unicode character or an underscore. '{}' is not that.",
                first
            )));
        }

        index += 1;

        loop {
            let next = lexer.input.char_at_index(index);
            if next == None {
                break;
            }
            let next = next.unwrap();
            if next.is_ascii() && !next.is_ascii_alphanumeric() && next != '_' {
                break;
            }
            index += next.len_utf8();
        }

        Ok(lexer.create_token(
            TokenType::Literal(LiteralTokenType::Identifier),
            index - lexer.index,
        ))
    }
}
struct HexIntParser;

impl Parser for HexIntParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>> {
        let mut index = lexer.index;
        if !matches!(lexer.input.get(index..=index + 1), Some("0x") | Some("0X")) {
            return Err(LexerError::WrongForm);
        }
        index += 2;
        if index >= lexer.input.len() {
            return Err(lexer.create_lex_error_with_len(
                "The prefix for a hexadecimal integer must be followed by a valid digit.".into(),
                3,
            ));
        }
        if !matches!(lexer.input.as_bytes()[index], hex_pattern!())
            || lexer.input.as_bytes()[index] == b'_'
        {
            return Err(lexer.create_lex_error_with_len(
                "The prefix for a hexadecimal integer must be followed by a valid digit.".into(),
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
        Ok(lexer.create_token(
            TokenType::Literal(LiteralTokenType::Int),
            index - lexer.index,
        ))
    }
}

struct BinIntParser;

impl Parser for BinIntParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>> {
        let mut index = lexer.index;
        if !matches!(lexer.input.get(index..=index + 1), Some("0b") | Some("0B")) {
            return Err(LexerError::WrongForm);
        }
        index += 2;
        if index >= lexer.input.len() {
            return Err(lexer.create_lex_error_with_len(
                "The prefix for a binary integer must be followed by a valid digit.".into(),
                3,
            ));
        }
        if !matches!(lexer.input.as_bytes()[index], bin_pattern!())
            || lexer.input.as_bytes()[index] == b'_'
        {
            return Err(lexer.create_lex_error_with_len(
                "The prefix for a binary integer must be followed by a valid digit.".into(),
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
        Ok(lexer.create_token(
            TokenType::Literal(LiteralTokenType::Int),
            index - lexer.index,
        ))
    }
}

struct DecIntParser;

impl Parser for DecIntParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>> {
        let mut index = lexer.index;

        if index >= lexer.input.len() {
            return Err(LexerError::WrongForm);
        }
        if !matches!(lexer.input.as_bytes()[index], dec_pattern!())
            || lexer.input.as_bytes()[index] == b'_'
        {
            return Err(LexerError::WrongForm);
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
        Ok(lexer.create_token(
            TokenType::Literal(LiteralTokenType::Int),
            index - lexer.index,
        ))
    }
}

struct EOIParser;
impl Parser for EOIParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>> {
        if lexer.index == lexer.input.len() {
            return Ok(lexer.create_token(TokenType::Eoi, 0));
        }
        Err(LexerError::WrongForm)
    }
}

struct WhitespaceParser;
impl Parser for WhitespaceParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>> {
        let mut len = 0;
        if !matches!(
            lexer.input.as_bytes()[lexer.index + len],
            whitespace_pattern!(),
        ) {
            return Err(LexerError::WrongForm);
        }
        if lexer.first_whitespace == None {
            lexer.first_whitespace = Some(FirstWhitespace {
                position: Position {
                    line_number: lexer.line_number,
                    column_number: lexer.column_number,
                    index: lexer.index,
                },
                whitespace_type: if lexer.input.as_bytes()[lexer.index + len] == b'\t' {
                    WhitespaceType::Tab
                } else {
                    WhitespaceType::Space
                },
            })
        }

        let first_whitespace = lexer.first_whitespace.unwrap();
        while matches!(
            lexer.input.as_bytes()[lexer.index + len],
            whitespace_pattern!(),
        ) {
            if first_whitespace.whitespace_type as u8 != lexer.input.as_bytes()[lexer.index + len] {
                return Err(LexerError::create_errors([
                    (
                        first_whitespace.position.line_number,
                        first_whitespace.position.column_number,
                        first_whitespace.position.index,
                        format!(
                            "This first whitespace character is a {0} character, which implies that the rest of the file uses {0} as its whitespace character.",
                            first_whitespace.whitespace_type
                        ),
                        lexer.input,
                        1,
                    ),
                    (
                        lexer.line_number,
                        lexer.column_number + len,
                        lexer.index + len,
                        format!(
                            "But here is a {} character, which is not the same.",
                            if lexer.input.as_bytes()[lexer.index + len] == b'\t' {
                                "tab"
                            } else {
                                "space"
                            }
                        ),
                        lexer.input,
                        1,
                    ),
                ]));
            }

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
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>> {
        if lexer
            .input
            .get(lexer.index..lexer.index + self.prefix.len())
            != Some(self.prefix)
        {
            return Err(LexerError::WrongForm);
        }
        let mut index = self.prefix.len() + lexer.index;

        while index < lexer.input.len() {
            if index < lexer.input.len() - 1 && &lexer.input[index..=index + 1] == "\r\n" {
                index += 2;
                break;
            }
            if lexer.input.as_bytes()[index] == b'\n' || lexer.input.as_bytes()[index] == b'\r' {
                index += 1;
                break;
            }
            index += 1;
        }

        Ok(lexer.create_token(TokenType::Comment(self.return_type), index - lexer.index))
    }
}

struct BlockCommentParser {
    prefix: &'static str,
    return_type: CommentTokenType,
}

impl Parser for BlockCommentParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>> {
        if lexer.index + self.prefix.len() > lexer.input.len() {
            // Not enough input left for our prefix to fit in -> cannot be a block comment.
            return Err(LexerError::WrongForm);
        }
        if lexer
            .input
            .get(lexer.index..lexer.index + self.prefix.len())
            != Some(self.prefix)
        {
            return Err(LexerError::WrongForm);
        }
        let mut index = self.prefix.len() + lexer.index;

        while index < lexer.input.len() {
            if lexer.input.get(index..=index + 1) == Some("*/") {
                index += 2;
                break;
            }
            index += 1;
        }

        Ok(lexer.create_token(TokenType::Comment(self.return_type), index - lexer.index))
    }
}

fn next_is_newline(input: &str, index: &mut usize, increment_index: bool) -> bool {
    if input.get(*index..=*index + 1) == Some("\r\n") {
        if increment_index {
            *index += 2;
        }
        return true;
    }

    if input.char_at_index(*index) == Some('\r') {
        if increment_index {
            *index += 1;
        }
        return true;
    }

    if input.char_at_index(*index) == Some('\n') {
        if increment_index {
            *index += 1;
        }
        return true;
    }
    false
}
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
#[repr(u8)]
enum WhitespaceType {
    Space = b' ',
    Tab = b'\t',
}

impl Display for WhitespaceType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            WhitespaceType::Space => write!(f, "space"),
            WhitespaceType::Tab => write!(f, "tab"),
        }
    }
}
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
struct FirstWhitespace {
    whitespace_type: WhitespaceType,
    position: Position,
}

#[derive(Debug, Copy, Clone)]
pub struct Lexer<'a> {
    line_number: LineNumber,
    column_number: ColumnNumber,
    input: &'a str,
    index: usize,
    first_whitespace: Option<FirstWhitespace>,
}

impl<'a> Lexer<'a> {
    pub(crate) fn new(input: &'a str) -> Self {
        Lexer {
            line_number: 0,
            column_number: 0,
            input,
            index: 0,
            first_whitespace: None,
        }
    }

    pub(crate) fn collect_tokens(&mut self) -> LexerResult<Vec<Token>> {
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
        let mut column_number = self.column_number;
        let mut line_number = self.line_number;
        let mut index = self.index;
        match offset {
            0 => (),
            1 => {
                let c = self.input.as_bytes()[index];
                if c == b'\r' || c == b'\n' {
                    self.column_number = 0;
                    self.line_number += 1;
                } else {
                    self.column_number += 1;
                }
                self.index += 1;
            }
            _ => {
                while index < self.index + offset - 2 {
                    if next_is_newline(self.input, &mut index, true) {
                        column_number = 0;
                        line_number += 1;
                    } else {
                        column_number += 1;
                        let n = self.input.char_at_index(index).unwrap().len_utf8();
                        index += n;
                    }
                }

                // Special case if there is only one byte left.
                if index == self.index + offset - 1 {
                    self.index = index + 1;
                    self.line_number = line_number;
                    self.column_number = column_number;
                    let c = self.input.as_bytes()[index];
                    if c == b'\r' || c == b'\n' {
                        self.line_number += 1;
                        self.column_number = 0;
                    } else {
                        self.column_number += 1;
                    }
                    return Position {
                        index,
                        line_number,
                        column_number,
                    };
                }
                self.line_number = line_number;
                self.column_number = column_number;
                self.index = index;
                // Handle last two bytes.
                self.index += 2;
                let s = &self.input.as_bytes()[index..=index + 1];
                index += 1;
                if s == b"\r\n" {
                    self.column_number = 0;
                    self.line_number += 1;
                    column_number += 1;
                } else {
                    if s[0] == b'\r' || s[0] == b'\n' {
                        line_number += 1;
                        column_number = 0;
                    } else {
                        column_number += 1;
                    }
                    for &b in s.iter() {
                        if b == b'\n' || b == b'\r' {
                            self.line_number += 1;
                            self.column_number = 0;
                        } else {
                            self.column_number += 1;
                        }
                    }
                }
            }
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

        counted_array!(lazy_static PARSERS: [Box<dyn Parser>; _] = [
        EOIParser.boxed(),
        StringParser.boxed(),

        HexIntParser.boxed(),
        BinIntParser.boxed(),
        DecIntParser.boxed(),


        LineCommentParser {prefix: "///", return_type: CommentTokenType::DocsLine}.boxed(),
        LineCommentParser {prefix: "//", return_type: CommentTokenType::Line}.boxed(),

        BlockCommentParser {prefix: "/**", return_type: CommentTokenType::DocsBlock}.boxed(),
        BlockCommentParser {prefix : "/*", return_type: CommentTokenType::Block}.boxed(),

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
        tag("[", Operator(LSquare)).boxed(),
        tag("]", Operator(RSquare)).boxed(),
        tag(",", Operator(Comma)).boxed(),

        tag("/=", Operator(DivideAssign)).boxed(),
        tag("*=", Operator(MultiplyAssign)).boxed(),
        tag("+=", Operator(PlusAssign)).boxed(),
        tag("-=", Operator(MinusAssign)).boxed(),
        tag("/", Operator(Divide)).boxed(),
        tag("*", Operator(Multiply)).boxed(),
        tag("+", Operator(Plus)).boxed(),
        tag("-", Operator(Minus)).boxed(),
        tag("=", Operator(Assign)).boxed(),



        WhitespaceParser.boxed(),
        IdentifierParser.boxed(),
        ]);
        if self.index > self.input.len() {
            self.index = self.input.len() - 1;
            return Err(self.create_lex_error("Lexer has ran past the end of the input".into()));
        }
        for (_i, parser) in PARSERS.iter().remove_last().enumerate() {
            let res = parser.parse(self);

            match res {
                Err(e) => match e {
                    LexerError::Incorrect(_) => return Err(e),
                    LexerError::WrongForm => continue,
                },
                Ok(v) => return Ok(v),
            }
        }
        let res = PARSERS[PARSERS.len() - 1].parse(self);
        return match res {
            Err(e) => match e {
                LexerError::WrongForm => Err(self.create_lex_error_with_len(
                    format!(
                        "Unknown Token starting with '{}'!",
                        &self.input[self.index..=self.index]
                    ),
                    1,
                )),
                LexerError::Incorrect(_) => Err(e),
            },
            Ok(v) => Ok(v),
        };
    }

    fn create_lex_error(&self, message: String) -> LexerError<'a> {
        self.create_lex_error_with_len(message, 1)
    }

    fn create_lex_error_with_len(&self, message: String, len: usize) -> LexerError<'a> {
        LexerError::with_contents(
            self.line_number,
            self.column_number,
            self.index,
            message,
            self.input,
            len,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::CommentTokenType::*;
    use super::LiteralTokenType::*;
    use super::OperatorTokenType::*;
    use super::*;
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
            Err(LexerError::with_contents(
                0,
                0,
                0,
                "The prefix for a hexadecimal integer must be followed by a valid digit.".into(),
                "0x",
                3,
            ))
        );
    }

    #[test]
    fn test_lexer_3() {
        pretty_assertions::assert_eq!(
            Lexer::new("0xh").collect_tokens(),
            Err(LexerError::with_contents(
                0,
                0,
                0,
                "The prefix for a hexadecimal integer must be followed by a valid digit.".into(),
                "0xh",
                3,
            ))
        );
    }
    #[test]
    fn test_lexer_4() {
        pretty_assertions::assert_eq!(
            Lexer::new("1\r\n2\r\n3").collect_tokens().unwrap(),
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
                    token_type: TokenType::Operator(Terminator),
                    start: Position {
                        line_number: 0,
                        column_number: 1,
                        index: 1,
                    },
                    len: 2,
                    end: Position {
                        line_number: 0,
                        column_number: 2,
                        index: 2,
                    },
                    contents: "\r\n",
                },
                Token {
                    token_type: TokenType::Literal(Int),
                    start: Position {
                        line_number: 1,
                        column_number: 0,
                        index: 3,
                    },
                    len: 1,
                    end: Position {
                        line_number: 1,
                        column_number: 0,
                        index: 3,
                    },
                    contents: "2",
                },
                Token {
                    token_type: TokenType::Operator(Terminator),
                    start: Position {
                        line_number: 1,
                        column_number: 1,
                        index: 4,
                    },
                    len: 2,
                    end: Position {
                        line_number: 1,
                        column_number: 2,
                        index: 5,
                    },
                    contents: "\r\n",
                },
                Token {
                    token_type: TokenType::Literal(LiteralTokenType::Int),
                    start: Position {
                        line_number: 2,
                        column_number: 0,
                        index: 6,
                    },
                    len: 1,
                    end: Position {
                        line_number: 2,
                        column_number: 0,
                        index: 6,
                    },
                    contents: "3",
                },
                Token {
                    token_type: TokenType::Eoi,
                    start: Position {
                        line_number: 2,
                        column_number: 1,
                        index: 7
                    },
                    end: Position {
                        line_number: 2,
                        column_number: 1,
                        index: 7
                    },
                    len: 0,
                    contents: "",
                }
            ]
        );
    }

    #[test]
    fn test_lexer_5() {
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
                    token_type: TokenType::Operator(Terminator),
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
                    token_type: TokenType::Literal(Int),
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
                    token_type: TokenType::Operator(Terminator),
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

    #[test]
    fn test_lexer_6() {
        pretty_assertions::assert_eq!(
            Lexer::new("/*\r\n\r\n\r\nI am comment!\r\n*/123/*I am second comment!\r\n*/;;")
                .collect_tokens()
                .unwrap(),
            vec![
                Token {
                    token_type: TokenType::Comment(Block),
                    start: Position {
                        line_number: 0,
                        column_number: 0,
                        index: 0,
                    },
                    len: 25,
                    end: Position {
                        line_number: 4,
                        column_number: 1,
                        index: 24,
                    },
                    contents: "/*\r\n\r\n\r\nI am comment!\r\n*/",
                },
                Token {
                    token_type: TokenType::Literal(Int),
                    start: Position {
                        line_number: 4,
                        column_number: 2,
                        index: 25,
                    },
                    len: 3,
                    end: Position {
                        line_number: 4,
                        column_number: 4,
                        index: 27,
                    },
                    contents: "123",
                },
                Token {
                    token_type: TokenType::Comment(Block),
                    start: Position {
                        line_number: 4,
                        column_number: 5,
                        index: 28,
                    },
                    len: 26,
                    end: Position {
                        line_number: 5,
                        column_number: 1,
                        index: 53,
                    },
                    contents: "/*I am second comment!\r\n*/",
                },
                Token {
                    token_type: TokenType::Operator(Terminator),
                    start: Position {
                        line_number: 5,
                        column_number: 2,
                        index: 54,
                    },
                    len: 1,
                    end: Position {
                        line_number: 5,
                        column_number: 2,
                        index: 54,
                    },
                    contents: ";",
                },
                Token {
                    token_type: TokenType::Operator(Terminator),
                    start: Position {
                        line_number: 5,
                        column_number: 3,
                        index: 55,
                    },
                    len: 1,
                    end: Position {
                        line_number: 5,
                        column_number: 3,
                        index: 55,
                    },
                    contents: ";",
                },
                Token {
                    token_type: TokenType::Eoi,
                    start: Position {
                        line_number: 5,
                        column_number: 4,
                        index: 56
                    },
                    end: Position {
                        line_number: 5,
                        column_number: 4,
                        index: 56
                    },
                    len: 0,
                    contents: "",
                }
            ]
        );
    }
    #[test]
    fn test_lexer_7() {
        pretty_assertions::assert_eq!(
            Lexer::new("'äääää'").collect_tokens().unwrap(),
            vec![
                Token {
                    token_type: TokenType::Literal(String),
                    start: Position {
                        line_number: 0,
                        column_number: 0,
                        index: 0
                    },
                    end: Position {
                        line_number: 0,
                        column_number: 6,
                        index: 11
                    },
                    len: 12,
                    contents: "'äääää'"
                },
                Token {
                    token_type: TokenType::Eoi,
                    start: Position {
                        line_number: 0,
                        column_number: 7,
                        index: 12
                    },
                    end: Position {
                        line_number: 0,
                        column_number: 7,
                        index: 12
                    },
                    len: 0,
                    contents: "",
                }
            ]
        )
    }
    #[test]
    fn test_lexer_8() {
        pretty_assertions::assert_eq!(
            Lexer::new("\t ").collect_tokens().unwrap_err(),
            LexerError::create_errors([(
                0,
                0,
                0,
                "This first whitespace character is a tab character, which implies that the rest of the file uses tab as its whitespace character.".into(),
                "\t ",
                1
            ),
            (
                0,
                1,
                1,
                "But here is a space character, which is not the same.".into(),
                "\t ",
                1
            )])
        )
    }
}
