#![allow(unused_macros, unused_imports)]

use counted_array::counted_array;
use lazy_static::lazy_static;
use std::fmt::{Display, Formatter};
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
pub(crate) enum TokenType {
    Eoi,
    Comment(CommentTokenType),
    Operator(OperatorTokenType),
    Literal(LiteralTokenType),
    Whitespace,
    Keyword(KeywordTokenType),
}

macro_rules! eoi {
    () => {
        crate::lexer::lex::TokenType::Eoi
    };
}

pub(crate) use eoi;

macro_rules! white_space {
    () => {
        crate::lexer::lex::TokenType::Whitespace
    };
}

pub(crate) use white_space;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum LiteralTokenType {
    Int,
    Decimal,
    String,
    Identifier,
}

macro_rules! identifier {
    () => {
        crate::lexer::lex::TokenType::Literal(crate::lexer::lex::LiteralTokenType::Identifier)
    };
}

pub(crate) use identifier;

macro_rules! int {
    () => {
        crate::lexer::lex::TokenType::Literal(crate::lexer::lex::LiteralTokenType::Int)
    };
}
pub(crate) use int;

macro_rules! string {
    () => {
        crate::lexer::lex::TokenType::Literal(crate::lexer::lex::LiteralTokenType::String)
    };
}

pub(crate) use string;

macro_rules! decimal {
    () => {
        crate::lexer::lex::TokenType::Literal(crate::lexer::lex::LiteralTokenType::Decimal)
    };
}

pub(crate) use decimal;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum CommentTokenType {
    Line,
    Block,
    DocsLine,
    DocsBlock,
}

macro_rules! comment_tt {
    () => {
        crate::lexer::lex::TokenType::Comment(_)
    };
}

pub(crate) use comment_tt;

macro_rules! line_comment_tt {
    () => {
        crate::lexer::lex::TokenType::Comment(crate::lexer::lex::CommentTokenType::Line)
            | crate::lexer::lex::TokenType::Comment(crate::lexer::lex::CommentTokenType::DocsLine)
    };
}

pub(crate) use line_comment_tt;

macro_rules! block_comment_tt {
    () => {
        crate::lexer::lex::TokenType::Comment(crate::lexer::lex::CommentTokenType::Block)
            | crate::lexer::lex::TokenType::Comment(crate::lexer::lex::CommentTokenType::DocsBlock)
    };
}

pub(crate) use block_comment_tt;

macro_rules! normal_line_comment {
    () => {
        crate::lexer::lex::TokenType::Comment(crate::lexer::lex::CommentTokenType::Line)
    };
}

pub(crate) use normal_line_comment;

macro_rules! normal_block_comment {
    () => {
        crate::lexer::lex::TokenType::Comment(crate::lexer::lex::CommentTokenType::Block)
    };
}

pub(crate) use normal_block_comment;

macro_rules! docs_line_comment {
    () => {
        crate::lexer::lex::TokenType::Comment(crate::lexer::lex::CommentTokenType::DocsLine)
    };
}

pub(crate) use docs_line_comment;

macro_rules! docs_block_comment {
    () => {
        crate::lexer::lex::TokenType::Comment(crate::lexer::lex::CommentTokenType::DocsBlock)
    };
}

pub(crate) use docs_block_comment;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum KeywordTokenType {
    BooleanComparison(BooleanComparisonKeywordTokenType),
    If(IfKeywordTokenType),
    Not(NotKeywordTokenType),
    Statement(StatementKeywordTokenType),
    FunctionDefinition(FunctionDefinitionKeywordTokenType),
    Loop(LoopKeywordTokenType),
    Value(ValueKeywordTokenType),
}

macro_rules! keyword_tt {
    () => {
        crate::lexer::lex::TokenType::Keyword(_)
    };
    ($i: ident) => {
        crate::lexer::lex::TokenType::Keyword($i)
    };
}

pub(crate) use keyword_tt;
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum BooleanComparisonKeywordTokenType {
    And,
    Or,
}

macro_rules! bool_comp_kw_tt {
    () => {
        crate::lexer::lex::TokenType::Keyword(
            crate::lexer::lex::KeywordTokenType::BooleanComparison(_),
        )
    };
    ($i: ident) => {
        crate::lexer::lex::TokenType::Keyword(
            crate::lexer::lex::KeywordTokenType::BooleanComparison($i),
        )
    };
}

pub(crate) use bool_comp_kw_tt;

macro_rules! and {
    () => {
        crate::lexer::lex::TokenType::Keyword(
            crate::lexer::lex::KeywordTokenType::BooleanComparison(
                crate::lexer::lex::BooleanComparisonKeywordTokenType::And,
            ),
        )
    };
}

pub(crate) use and;

macro_rules! or {
    () => {
        crate::lexer::lex::TokenType::Keyword(
            crate::lexer::lex::KeywordTokenType::BooleanComparison(
                crate::lexer::lex::BooleanComparisonKeywordTokenType::Or,
            ),
        )
    };
}

pub(crate) use or;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum IfKeywordTokenType {
    If,
    Elif,
    Else,
}
macro_rules! if_kw_tt {
    () => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::If(_))
    };
    ($i: ident) => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::If($i))
    };
}
pub(crate) use if_kw_tt;

macro_rules! if_ {
    () => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::If(
            crate::lexer::lex::IfKeywordTokenType::If,
        ))
    };
}

pub(crate) use if_;

macro_rules! elif {
    () => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::If(
            crate::lexer::lex::IfKeywordTokenType::Elif,
        ))
    };
}

pub(crate) use elif;

macro_rules! else_ {
    () => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::If(
            crate::lexer::lex::IfKeywordTokenType::Else,
        ))
    };
}

pub(crate) use else_;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum NotKeywordTokenType {
    Not,
}

macro_rules! not {
    () => {
        TokenType::Keyword(KeywordTokenType::Not(NotKeywordTokenType::Not))
    };
}

pub(crate) use not;
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum StatementKeywordTokenType {
    Continue,
    Break,
    Return,
}

macro_rules! statement_kw_tt {
    () => {
        crate::lexer::lex::KeywordTokenType::Statement(_)
    };
    ($i: ident) => {
        crate::lexer::lex::KeywordTokenType::Statement($i)
    };
}
pub(crate) use statement_kw_tt;

macro_rules! continue_ {
    () => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::Statement(
            crate::lexer::lex::StatementKeywordTokenType::Continue,
        ))
    };
}

pub(crate) use continue_;

macro_rules! break_ {
    () => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::Statement(
            crate::lexer::lex::StatementKeywordTokenType::Break,
        ))
    };
}

pub(crate) use break_;

macro_rules! return_ {
    () => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::Statement(
            crate::lexer::lex::StatementKeywordTokenType::Return,
        ))
    };
}

pub(crate) use return_;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum FunctionDefinitionKeywordTokenType {
    Fun,
}

macro_rules! fun {
    () => {
        crate::lexer::lex::TokenType::Keyword(
            crate::lexer::lex::KeywordTokenType::FunctionDefinition(
                crate::lexer::lex::FunctionDefinitionKeywordTokenType::Fun,
            ),
        )
    };
}

pub(crate) use fun;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum LoopKeywordTokenType {
    Loop,
    While,
}

macro_rules! loop_kw_tt {
    () => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::Loop(_))
    };
    ($i: ident) => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::Loop($i))
    };
}

pub(crate) use loop_kw_tt;

macro_rules! loop_ {
    () => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::Loop(
            crate::lexer::lex::LoopKeywordTokenType::Loop,
        ))
    };
}

pub(crate) use loop_;

macro_rules! while_ {
    () => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::Loop(
            crate::lexer::lex::LoopKeywordTokenType::While,
        ))
    };
}

pub(crate) use while_;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum ValueKeywordTokenType {
    False,
    True,
}

macro_rules! false_ {
    () => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::Value(
            crate::lexer::lex::ValueKeywordTokenType::False,
        ))
    };
}

pub(crate) use false_;

macro_rules! true_ {
    () => {
        crate::lexer::lex::TokenType::Keyword(crate::lexer::lex::KeywordTokenType::Value(
            crate::lexer::lex::ValueKeywordTokenType::True,
        ))
    };
}

pub(crate) use true_;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum OperatorTokenType {
    Assignment(AssignmentOperatorTokenType),
    Arithmetic(ArithmeticOperatorTokenType),
    Comparison(ComparisonOperatorTokenType),
    Term(TermOperatorTokenType),
    Exponent,
    // Structural.
    Comma,
    LParen,
    RParen,
    LSquare,
    RSquare,
    Terminator,
    Colon,
    Dot,
}
impl TryFrom<TokenType> for OperatorTokenType {
    type Error = ();

    fn try_from(value: TokenType) -> Result<Self, Self::Error> {
        match value {
            TokenType::Operator(o) => Ok(o),
            _ => Err(()),
        }
    }
}

macro_rules! operator_tt {
    () => {
        crate::lexer::lex::TokenType::Operator(_)
    };
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]

pub(crate) enum AssignmentOperatorTokenType {
    Assign,
    PlusAssign,
    MinusAssign,
    StarAssign,
    SlashAssign,
    ProcentAssign,
    BitshiftLeftAssign,
    BitshiftRightAssign,
    BitwiseAndAssign,
    BitwiseOrAssign,
    BitwiseXorAssign,

    ExponentAssign,
}

impl TryFrom<OperatorTokenType> for AssignmentOperatorTokenType {
    type Error = ();
    fn try_from(value: OperatorTokenType) -> Result<Self, Self::Error> {
        match value {
            OperatorTokenType::Assignment(assignment_operator_token_type) => {
                Ok(assignment_operator_token_type)
            }
            _ => Err(()),
        }
    }
}

macro_rules! assignment_operator_tt {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(_))
    };
    ($i: ident) => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment($i))
    };
}

pub(crate) use assignment_operator_tt;

macro_rules! assign {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(
            crate::lexer::lex::AssignmentOperatorTokenType::Assign,
        ))
    };
}

pub(crate) use assign;

macro_rules! plus_assign {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(
            crate::lexer::lex::AssignmentOperatorTokenType::PlusAssign,
        ))
    };
}

pub(crate) use plus_assign;

macro_rules! minus_assign {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(
            crate::lexer::lex::AssignmentOperatorTokenType::MinusAssign,
        ))
    };
}

pub(crate) use minus_assign;

macro_rules! star_assign {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(
            crate::lexer::lex::AssignmentOperatorTokenType::StarAssign,
        ))
    };
}

pub(crate) use star_assign;

macro_rules! slash_assign {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(
            crate::lexer::lex::AssignmentOperatorTokenType::SlashAssign,
        ))
    };
}

pub(crate) use slash_assign;

macro_rules! procent_assign {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(
            crate::lexer::lex::AssignmentOperatorTokenType::ProcentAssign,
        ))
    };
}

pub(crate) use procent_assign;

macro_rules! bitshift_left_assign {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(
            crate::lexer::lex::AssignmentOperatorTokenType::BitshiftLeftAssign,
        ))
    };
}

pub(crate) use bitshift_left_assign;

macro_rules! bitshift_right_assign {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(
            crate::lexer::lex::AssignmentOperatorTokenType::BitshiftRightAssign,
        ))
    };
}

pub(crate) use bitshift_right_assign;

macro_rules! bitwise_and_assign {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(
            crate::lexer::lex::AssignmentOperatorTokenType::BitwiseAndAssign,
        ))
    };
}

pub(crate) use bitwise_and_assign;

macro_rules! bitwise_or_assign {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(
            crate::lexer::lex::AssignmentOperatorTokenType::BitwiseOrAssign,
        ))
    };
}

pub(crate) use bitwise_or_assign;

macro_rules! bitwise_xor_assign {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(
            crate::lexer::lex::AssignmentOperatorTokenType::BitwiseXorAssign,
        ))
    };
}

pub(crate) use bitwise_xor_assign;

macro_rules! exponent_assign {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Assignment(
            crate::lexer::lex::AssignmentOperatorTokenType::ExponentAssign,
        ))
    };
}

pub(crate) use exponent_assign;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum ArithmeticOperatorTokenType {
    Unary(UnaryOperatorTokenType),
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
}

macro_rules! arithmetic_operator_tt {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Arithmetic(_))
    };
    ($i: ident) => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Arithmetic($i))
    };
}

pub(crate) use arithmetic_operator_tt;

macro_rules! bitwise_xor {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Arithmetic(
            crate::lexer::lex::ArithmeticOperatorTokenType::BitwiseXor,
        ))
    };
}

pub(crate) use bitwise_xor;

macro_rules! bitwise_or {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Arithmetic(
            crate::lexer::lex::ArithmeticOperatorTokenType::BitwiseOr,
        ))
    };
}

pub(crate) use bitwise_or;

macro_rules! bitwise_and {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Arithmetic(
            crate::lexer::lex::ArithmeticOperatorTokenType::BitwiseAnd,
        ))
    };
}

pub(crate) use bitwise_and;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]

pub(crate) enum UnaryOperatorTokenType {
    Plus,
    Minus,
    BitwiseNot,
}

macro_rules! unary_operator_tt {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Arithmetic(
            crate::lexer::lex::ArithmeticOperatorTokenType::Unary(_),
        ))
    };
    ($i: ident) => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Arithmetic(
            crate::lexer::lex::ArithmeticOperatorTokenType::Unary($i),
        ))
    };
}

pub(crate) use unary_operator_tt;

macro_rules! plus {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Arithmetic(
            crate::lexer::lex::ArithmeticOperatorTokenType::Unary(
                crate::lexer::lex::UnaryOperatorTokenType::Plus,
            ),
        ))
    };
}

pub(crate) use plus;

macro_rules! minus {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Arithmetic(
            crate::lexer::lex::ArithmeticOperatorTokenType::Unary(
                crate::lexer::lex::UnaryOperatorTokenType::Minus,
            ),
        ))
    };
}

pub(crate) use minus;

macro_rules! bitwise_not {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Arithmetic(
            crate::lexer::lex::ArithmeticOperatorTokenType::Unary(
                crate::lexer::lex::UnaryOperatorTokenType::BitwiseNot,
            ),
        ))
    };
}

pub(crate) use bitwise_not;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum TermOperatorTokenType {
    Star,
    Slash,
    Percent,
    BitshiftLeft,
    BitshiftRight,
}

macro_rules! term_operator_tt {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Term(_))
    };
    ($i: ident) => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Term($i))
    };
}

pub(crate) use term_operator_tt;

macro_rules! star {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Term(
            crate::lexer::lex::TermOperatorTokenType::Star,
        ))
    };
}

pub(crate) use star;

macro_rules! slash {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Term(
            crate::lexer::lex::TermOperatorTokenType::Slash,
        ))
    };
}

pub(crate) use slash;

macro_rules! percent {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Term(
            crate::lexer::lex::TermOperatorTokenType::Percent,
        ))
    };
}

pub(crate) use percent;

macro_rules! bitshift_left {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Term(
            crate::lexer::lex::TermOperatorTokenType::BitshiftLeft,
        ))
    };
}

pub(crate) use bitshift_left;

macro_rules! bitshift_right {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Term(
            crate::lexer::lex::TermOperatorTokenType::BitshiftRight,
        ))
    };
}

pub(crate) use bitshift_right;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) enum ComparisonOperatorTokenType {
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessThanEquals,
    GreaterThanEquals,
}

macro_rules! comparison_operator_tt {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Comparison(_))
    };
    ($i: ident) => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Comparison($i))
    };
}

pub(crate) use comparison_operator_tt;

macro_rules! equals {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Comparison(
            crate::lexer::lex::ComparisonOperatorTokenType::Equals,
        ))
    };
}

pub(crate) use equals;

macro_rules! not_equals {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Comparison(
            crate::lexer::lex::ComparisonOperatorTokenType::NotEquals,
        ))
    };
}

pub(crate) use not_equals;

macro_rules! less_than {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Comparison(
            crate::lexer::lex::ComparisonOperatorTokenType::LessThan,
        ))
    };
}

pub(crate) use less_than;

macro_rules! greater_than {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Comparison(
            crate::lexer::lex::ComparisonOperatorTokenType::GreaterThan,
        ))
    };
}

pub(crate) use greater_than;

macro_rules! less_than_equals {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Comparison(
            crate::lexer::lex::ComparisonOperatorTokenType::LessThanEquals,
        ))
    };
}

pub(crate) use less_than_equals;

macro_rules! greater_than_equals {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Comparison(
            crate::lexer::lex::ComparisonOperatorTokenType::GreaterThanEquals,
        ))
    };
}

pub(crate) use greater_than_equals;

macro_rules! exponent {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Exponent)
    };
}

pub(crate) use exponent;

macro_rules! l_paren {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::LParen)
    };
}

pub(crate) use l_paren;

macro_rules! r_paren {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::RParen)
    };
}

pub(crate) use r_paren;

macro_rules! r_square {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::RSquare)
    };
}

pub(crate) use r_square;

macro_rules! l_square {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::LSquare)
    };
}

pub(crate) use l_square;

macro_rules! terminator {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Terminator)
    };
}

pub(crate) use terminator;

macro_rules! comma {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Comma)
    };
}

pub(crate) use comma;

macro_rules! colon {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Colon)
    };
}

pub(crate) use colon;

macro_rules! dot {
    () => {
        crate::lexer::lex::TokenType::Operator(crate::lexer::lex::OperatorTokenType::Dot)
    };
}

pub(crate) use dot;

use crate::util::{CharAtIndex, RemoveLastTrait};

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) struct Position {
    pub(crate) line_number: LineNumber,
    pub(crate) column_number: ColumnNumber,
    pub(crate) index: usize,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub(crate) struct Token<'a> {
    pub(crate) contents: &'a str,
    pub(crate) start: Position,
    pub(crate) len: usize,
    pub(crate) end: Position,
    pub(crate) token_type: TokenType,
}

type LineNumber = usize;
type ColumnNumber = usize;

#[derive(Eq, PartialEq, Debug, Clone)]
pub(crate) enum LexerError<'a> {
    /// The next token is obviously syntactically invalid.
    Incorrect(Vec<LexerErrorContents<'a>>),
    /// The next token is of the wrong form for this parser, and may still be valid for another parser.
    WrongForm,
}

impl<'a> std::error::Error for LexerError<'a> {}

#[derive(Eq, PartialEq, Debug, Clone)]
pub(crate) struct LexerErrorContents<'a> {
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

        Ok(lexer.create_token(string!(), index - lexer.index))
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

        Ok(lexer.create_token(identifier!(), index - lexer.index))
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
        Ok(lexer.create_token(int!(), index - lexer.index))
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
        Ok(lexer.create_token(int!(), index - lexer.index))
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
        Ok(lexer.create_token(int!(), index - lexer.index))
    }
}

struct EOIParser;
impl Parser for EOIParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>> {
        if lexer.index == lexer.input.len() {
            return Ok(lexer.create_token(eoi!(), 0));
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
        Ok(lexer.create_token(white_space!(), len))
    }
}

struct LineCommentParser {
    prefix: &'static str,
    return_type: TokenType,
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
                break;
            }
            if lexer.input.as_bytes()[index] == b'\n' || lexer.input.as_bytes()[index] == b'\r' {
                break;
            }
            index += 1;
        }

        Ok(lexer.create_token(self.return_type, index - lexer.index))
    }
}

struct BlockCommentParser {
    prefix: &'static str,
    return_type: TokenType,
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

        Ok(lexer.create_token(self.return_type, index - lexer.index))
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

struct KeywordParser {
    tag: &'static str,
    keyword_type: TokenType,
}

impl Parser for KeywordParser {
    fn parse<'a>(&self, lexer: &mut Lexer<'a>) -> LexerResult<'a, Token<'a>> {
        let (index, column_number, line_number) =
            (lexer.index, lexer.column_number, lexer.line_number);
        let mut res = IdentifierParser.parse(lexer)?;
        if res.contents == self.tag {
            res.token_type = self.keyword_type;
            Ok(res)
        } else {
            lexer.index = index;
            lexer.column_number = column_number;
            lexer.line_number = line_number;
            Err(LexerError::WrongForm)
        }
    }
}

fn keyword(tag: &'static str, keyword_type: TokenType) -> KeywordParser {
    KeywordParser { tag, keyword_type }
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
pub(crate) struct Lexer<'a> {
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

    pub(crate) fn collect_tokens(&mut self) -> LexerResult<Vec<Token<'a>>> {
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

    pub(crate) fn yield_token(&mut self) -> LexerResult<'a, Token<'a>> {
        counted_array!(lazy_static PARSERS: [Box<dyn Parser>; _] = [
        EOIParser.boxed(),
        StringParser.boxed(),

        HexIntParser.boxed(),
        BinIntParser.boxed(),
        DecIntParser.boxed(),


        LineCommentParser {prefix: "///", return_type: docs_line_comment!()}.boxed(),
        LineCommentParser {prefix: "//", return_type: normal_line_comment!()}.boxed(),
        tag("/***/", docs_block_comment!()).boxed(),
        tag("/**/", normal_block_comment!()).boxed(),
        BlockCommentParser {prefix: "/**", return_type: docs_block_comment!()}.boxed(),
        BlockCommentParser {prefix : "/*", return_type: normal_block_comment!()}.boxed(),

        tag(";", terminator!()).boxed(),
        tag("\r\n", terminator!()).boxed(),
        tag("\r", terminator!()).boxed(),
        tag("\n", terminator!()).boxed(),

        tag("**=", exponent_assign!()).boxed(),

        tag("/=", slash_assign!()).boxed(),
        tag("*=", star_assign!()).boxed(),
        tag("%=", procent_assign!()).boxed(),
        tag("<<=", bitshift_left_assign!()).boxed(),
        tag(">>=", bitshift_right_assign!()).boxed(),

        tag("!=", not_equals!()).boxed(),
        tag("==", equals!()).boxed(),
        tag(">=", greater_than_equals!()).boxed(),
        tag("<=", less_than_equals!()).boxed(),
        tag(">", greater_than!()).boxed(),
        tag("<", less_than!()).boxed(),

        tag("+=", plus_assign!()).boxed(),
        tag("-=", minus_assign!()).boxed(),
        tag("&=", bitwise_and_assign!()).boxed(),
        tag("|=", bitwise_or_assign!()).boxed(),
        tag("^=", bitwise_xor_assign!()).boxed(),
        tag("=", assign!()).boxed(),

            tag("**", exponent!()).boxed(),

        tag("<<", bitshift_left!()).boxed(),
        tag(">>", bitshift_right!()).boxed(),
        tag("/", slash!()).boxed(),
        tag("*", star!()).boxed(),
        tag("%", percent!()).boxed(),

        tag("&", bitwise_and!()).boxed(),
        tag("|", bitwise_or!()).boxed(),
        tag("~", bitwise_not!()).boxed(),
        tag("^", bitwise_xor!()).boxed(),



        tag("(", l_paren!()).boxed(),
        tag(")", r_paren!()).boxed(),
        tag("[", l_square!()).boxed(),
        tag("]", r_square!()).boxed(),
        tag(",", comma!()).boxed(),


        tag("+", plus!()).boxed(),
        tag("-", minus!()).boxed(),

        tag(":", colon!()).boxed(),
        tag(".", dot!()).boxed(),


        WhitespaceParser.boxed(),

        keyword("and", and!()).boxed(),
        keyword("or", or!()).boxed(),
        keyword("fun", fun!()).boxed(),
        keyword("if", if_!()).boxed(),
        keyword("elif", elif!()).boxed(),
        keyword("else", else_!()).boxed(),
        keyword("not", not!()).boxed(),
        keyword("loop", loop_!()).boxed(),
        keyword("while", while_!()).boxed(),
        keyword("break", break_!()).boxed(),
        keyword("continue", continue_!()).boxed(),
        keyword("return", return_!()).boxed(),

        keyword("true", true_!()).boxed(),
        keyword("false", false_!()).boxed(),


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
    use super::*;
    #[test]
    fn test_lexer_1() {
        pretty_assertions::assert_eq!(
            Lexer::new("2 != 3").collect_tokens().unwrap(),
            vec![
                Token {
                    token_type: int!(),
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
                    token_type: white_space!(),
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
                    token_type: not_equals!(),
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
                    token_type: white_space!(),
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
                    token_type: int!(),
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
                    token_type: eoi!(),
                    start: Position {
                        line_number: 0,
                        column_number: 6,
                        index: 6
                    },
                    end: Position {
                        line_number: 0,
                        column_number: 6,
                        index: 6
                    },
                    len: 0,
                    contents: "",
                } //Token::Int("2"),
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
                    token_type: int!(),
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
                    token_type: terminator!(),
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
                    token_type: int!(),
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
                    token_type: terminator!(),
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
                    token_type: int!(),
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
                    token_type: eoi!(),
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
                    token_type: int!(),
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
                    token_type: terminator!(),
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
                    token_type: int!(),
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
                    token_type: terminator!(),
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
                    token_type: int!(),
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
                    token_type: eoi!(),
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
                    token_type: normal_block_comment!(),
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
                    token_type: int!(),
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
                    token_type: normal_block_comment!(),
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
                    token_type: terminator!(),
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
                    token_type: terminator!(),
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
                    token_type: eoi!(),
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
                    token_type: string!(),
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
                    contents: "'äääää'",
                },
                Token {
                    token_type: eoi!(),
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
    #[test]
    fn test_lexer_9() {
        pretty_assertions::assert_eq!(
            Lexer::new("\t\ta\nb\n\tc\n\t\td").collect_tokens().unwrap(),
            vec![
                Token {
                    token_type: white_space!(),
                    start: Position {
                        line_number: 0,
                        column_number: 0,
                        index: 0,
                    },
                    end: Position {
                        line_number: 0,
                        column_number: 1,
                        index: 1,
                    },
                    len: 2,
                    contents: "\t\t",
                },
                Token {
                    token_type: identifier!(),
                    start: Position {
                        line_number: 0,
                        column_number: 2,
                        index: 2,
                    },
                    end: Position {
                        line_number: 0,
                        column_number: 2,
                        index: 2,
                    },
                    len: 1,
                    contents: "a",
                },
                Token {
                    token_type: terminator!(),
                    start: Position {
                        line_number: 0,
                        column_number: 3,
                        index: 3
                    },
                    end: Position {
                        line_number: 0,
                        column_number: 3,
                        index: 3
                    },
                    len: 1,
                    contents: "\n",
                },
                Token {
                    token_type: identifier!(),
                    start: Position {
                        line_number: 1,
                        column_number: 0,
                        index: 4
                    },
                    end: Position {
                        line_number: 1,
                        column_number: 0,
                        index: 4
                    },
                    len: 1,
                    contents: "b",
                },
                Token {
                    token_type: terminator!(),
                    start: Position {
                        line_number: 1,
                        column_number: 1,
                        index: 5
                    },
                    end: Position {
                        line_number: 1,
                        column_number: 1,
                        index: 5
                    },
                    len: 1,
                    contents: "\n",
                },
                Token {
                    token_type: white_space!(),
                    start: Position {
                        line_number: 2,
                        column_number: 0,
                        index: 6
                    },
                    end: Position {
                        line_number: 2,
                        column_number: 0,
                        index: 6
                    },
                    len: 1,
                    contents: "\t",
                },
                Token {
                    token_type: identifier!(),
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
                    len: 1,
                    contents: "c",
                },
                Token {
                    token_type: terminator!(),
                    start: Position {
                        line_number: 2,
                        column_number: 2,
                        index: 8
                    },
                    end: Position {
                        line_number: 2,
                        column_number: 2,
                        index: 8
                    },
                    len: 1,
                    contents: "\n",
                },
                Token {
                    token_type: white_space!(),
                    start: Position {
                        line_number: 3,
                        column_number: 0,
                        index: 9,
                    },
                    end: Position {
                        line_number: 3,
                        column_number: 1,
                        index: 10,
                    },
                    len: 2,
                    contents: "\t\t",
                },
                Token {
                    token_type: identifier!(),
                    start: Position {
                        line_number: 3,
                        column_number: 2,
                        index: 11
                    },
                    end: Position {
                        line_number: 3,
                        column_number: 2,
                        index: 11
                    },
                    len: 1,
                    contents: "d",
                },
                Token {
                    token_type: eoi!(),
                    start: Position {
                        line_number: 3,
                        column_number: 3,
                        index: 12,
                    },
                    end: Position {
                        line_number: 3,
                        column_number: 3,
                        index: 12
                    },
                    len: 0,
                    contents: "",
                }
            ]
        )
    }

    #[test]
    fn test_lexer_line_comments() {
        pretty_assertions::assert_eq!(
            Lexer::new("// comment\n// comment\n// comment\n")
                .collect_tokens()
                .unwrap(),
            vec![
                Token {
                    token_type: normal_line_comment!(),
                    start: Position {
                        line_number: 0,
                        column_number: 0,
                        index: 0,
                    },
                    end: Position {
                        line_number: 0,
                        column_number: 9,
                        index: 9,
                    },
                    contents: "// comment",
                    len: 10,
                },
                Token {
                    token_type: terminator!(),
                    start: Position {
                        line_number: 0,
                        column_number: 10,
                        index: 10,
                    },
                    end: Position {
                        line_number: 0,
                        column_number: 10,
                        index: 10,
                    },
                    len: 1,
                    contents: "\n",
                },
                Token {
                    token_type: normal_line_comment!(),
                    start: Position {
                        line_number: 1,
                        column_number: 0,
                        index: 11,
                    },
                    end: Position {
                        line_number: 1,
                        column_number: 9,
                        index: 20,
                    },
                    contents: "// comment",
                    len: 10,
                },
                Token {
                    token_type: terminator!(),
                    start: Position {
                        line_number: 1,
                        column_number: 10,
                        index: 21,
                    },
                    end: Position {
                        line_number: 1,
                        column_number: 10,
                        index: 21,
                    },
                    len: 1,
                    contents: "\n",
                },
                Token {
                    token_type: normal_line_comment!(),
                    start: Position {
                        line_number: 2,
                        column_number: 0,
                        index: 22,
                    },
                    end: Position {
                        line_number: 2,
                        column_number: 9,
                        index: 31,
                    },
                    contents: "// comment",
                    len: 10,
                },
                Token {
                    token_type: terminator!(),
                    start: Position {
                        line_number: 2,
                        column_number: 10,
                        index: 32,
                    },
                    end: Position {
                        line_number: 2,
                        column_number: 10,
                        index: 32,
                    },
                    len: 1,
                    contents: "\n",
                },
                Token {
                    token_type: eoi!(),
                    start: Position {
                        line_number: 3,
                        column_number: 0,
                        index: 33,
                    },
                    end: Position {
                        line_number: 3,
                        column_number: 0,
                        index: 33,
                    },
                    len: 0,
                    contents: "",
                }
            ]
        )
    }
}
