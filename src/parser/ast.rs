// Grammar:
// expression = (literal | identifier | (expression operator expression))
use super::generate_bytecode::{Program, ValueStore};
use crate::lexer::lex::{
    terminator, white_space, ArithmeticOperatorTokenType, AssignmentOperatorTokenType,
    BooleanComparisonKeywordTokenType, CommaOperatorTokenType, ComparisonOperatorTokenType,
    KeywordTokenType, LParenOperatorTokenType, LiteralTokenType, NotKeywordTokenType,
    OperatorTokenType::{self, *},
    RParenOperatorTokenType, TermOperatorTokenType, Token, TokenType, UnaryOperatorTokenType,
};
use ibig::{ibig, IBig};
use std::fmt::{Debug, Display};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct NodeMetaData<'a> {
    pub(crate) contents: &'a str,
    pub(crate) index: usize,
    pub(crate) len: usize,
    pub(crate) line_number: usize,
    pub(crate) column_number: usize,
}

pub(crate) trait GenerateBytecode {
    fn gen_bytecode(&mut self, bytecode: &mut Vec<u8>);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Statement<'a> {
    pub(crate) meta: NodeMetaData<'a>,
    pub(crate) statement_type: StatementType<'a>,
}

impl<'a> GenerateBytecode for Statement<'a> {
    fn gen_bytecode(&mut self, bytecode: &mut Vec<u8>) {
        match self.statement_type {
            super::ast::StatementType::Expr(ref mut e) => e.gen_bytecode(bytecode),
            _ => unreachable!(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StatementType<'a> {
    Return,
    Break,
    Assignment(Identifier<'a>, AssignmentOperatorTokenType, Expression<'a>),
    Expr(Expression<'a>),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct Identifier<'a> {
    pub(crate) id: &'a str,
    pub(crate) meta: NodeMetaData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Expression<'a> {
    pub(crate) main_expression: ComparisonExpression<'a>,
    pub(crate) sub_expressions: Vec<SubExpression<'a>>,
    pub(crate) meta: NodeMetaData<'a>,
}

impl<'a> GenerateBytecode for Expression<'a> {
    fn gen_bytecode(&mut self, bytecode: &mut Vec<u8>) {
        match self.main_expression.comparison_type {
            super::ast::ComparisonExpressionType::ComparisonChain(ref mut chain) => {
                chain.gen_bytecode(bytecode)
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubExpression<'a> {
    pub(crate) keyword: BooleanComparisonKeywordTokenType,
    pub(crate) expression: ComparisonExpression<'a>,
    pub(crate) meta: NodeMetaData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComparisonChainExpression<'a> {
    pub(crate) main_expression: ArithmeticExpression<'a>,
    pub(crate) sub_expressions: Vec<SubComparisonExpression<'a>>,
    pub(crate) meta: NodeMetaData<'a>,
}

impl<'a> GenerateBytecode for ComparisonChainExpression<'a> {
    fn gen_bytecode(&mut self, bytecode: &mut Vec<u8>) {
        match self
            .main_expression
            .main_expression
            .main_expression
            .factor_type
        {
            super::ast::FactorType::Call(ref mut call) => {
                if call.params.is_some() {
                } else {
                }
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubComparisonExpression<'a> {
    pub(crate) operator: ComparisonOperatorTokenType,
    pub(crate) expression: ArithmeticExpression<'a>,
    pub(crate) meta: NodeMetaData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComparisonExpression<'a> {
    pub(crate) meta: NodeMetaData<'a>,
    pub(crate) comparison_type: ComparisonExpressionType<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]

pub(crate) enum ComparisonExpressionType<'a> {
    Not(NotKeywordTokenType, Box<ComparisonExpression<'a>>),
    ComparisonChain(ComparisonChainExpression<'a>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArithmeticExpression<'a> {
    pub(crate) meta: NodeMetaData<'a>,
    pub(crate) main_expression: Term<'a>,
    pub(crate) sub_expressions: Vec<SubArithmeticExpression<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubArithmeticExpression<'a> {
    pub(crate) operator: ArithmeticOperatorTokenType,
    pub(crate) expression: Term<'a>,
    pub(crate) meta: NodeMetaData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Term<'a> {
    pub(crate) meta: NodeMetaData<'a>,
    pub(crate) main_expression: Factor<'a>,
    pub(crate) sub_expressions: Vec<SubTerm<'a>>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubTerm<'a> {
    pub(crate) meta: NodeMetaData<'a>,
    pub(crate) operator: TermOperatorTokenType,
    pub(crate) expression: Factor<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Factor<'a> {
    pub(crate) factor_type: FactorType<'a>,
    pub(crate) meta: NodeMetaData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FactorType<'a> {
    UnaryFactor(UnaryOperatorTokenType, Box<Factor<'a>>),
    Call(Call<'a>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Call<'a> {
    pub(crate) atom: Atom<'a>,
    pub(crate) params: Option<Vec<Expression<'a>>>,
    pub(crate) meta: NodeMetaData<'a>,
}
/*
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Params<'a> {
    lparen: LParenOperatorTokenType,
    args: Vec<Param<'a>>,
    rparen: RParenOperatorTokenType,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct Param<'a> {
    name: Identifier<'a>,
}
*/
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Atom<'a> {
    pub(crate) meta: NodeMetaData<'a>,
    pub(crate) atom_type: AtomType<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Int {
    Big(usize),
    Small(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AtomType<'a> {
    Int(Int),
    Identifier(&'a str),
    String(usize),
    ExpressionWithParentheses(
        LParenOperatorTokenType,
        Box<Expression<'a>>,
        RParenOperatorTokenType,
    ),
    List(ListExpression),
    Dict(DictExpression),
    If(IfExpression),
    For(ForExpression),
    While(WhileExpression),
    FuncDef(FunctionDefinitionExpression),
    Loop(LoopExpression),
}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct ListExpression;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct DictExpression;
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct IfExpression;
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct ForExpression;
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct WhileExpression;
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct FunctionDefinitionExpression;
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct LoopExpression;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParserError;

impl Display for ParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ParserError")
    }
}
impl std::error::Error for ParserError {}
pub(crate) type ParserResult<T> = Result<T, ParserError>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Ast<'a> {
    index: usize,
    tokens: Vec<Token<'a>>,
    current_token: Token<'a>,
    program: Program<'a>,
}

impl<'a> Ast<'a> {
    pub fn new(tokens: Vec<Token<'a>>) -> Self {
        Ast {
            index: 0,
            current_token: tokens[0],
            tokens,
            program: Program {
                statements: Vec::new(),
                big_int_literals: ValueStore::new(),
                string_literals: ValueStore::new(),
            },
        }
    }

    fn step_over_whitespace(&mut self) {
        if self.current_token.token_type == white_space() {
            self.next_token();
        }
    }

    pub fn parse(mut self) -> ParserResult<Program<'a>> {
        self.expect_statements()?;
        Ok(self.program)
    }

    fn expect_statements(&mut self) -> ParserResult<()> {
        let first_statement = self.expect_statement()?;
        self.program.statements.push(first_statement);
        while self.has_tokens_left() {
            if self.current_token.token_type != terminator() {
                break;
            }
            self.next_token();

            let next_statement = self.expect_statement()?;
            self.program.statements.push(next_statement);
        }
        Ok(())
    }

    fn expect_statement(&mut self) -> ParserResult<Statement<'a>> {
        self.step_over_whitespace();
        let expression = self.expect_expression()?;
        Ok(Statement {
            meta: expression.meta,
            statement_type: StatementType::Expr(expression),
        })
    }

    fn expect_expression(&mut self) -> ParserResult<Expression<'a>> {
        self.step_over_whitespace();

        let comp_expr = self.expect_comparison_expression()?;
        Ok(Expression {
            meta: comp_expr.meta,
            sub_expressions: vec![],
            main_expression: comp_expr,
        })
    }

    fn expect_comparison_expression(&mut self) -> ParserResult<ComparisonExpression<'a>> {
        self.step_over_whitespace();

        let (comparison_type, meta) = if self.current_token.token_type
            == TokenType::Keyword(KeywordTokenType::Not(NotKeywordTokenType::Not))
        {
            self.next_token();
            let inner = self.expect_comparison_expression()?;
            let meta = inner.meta;
            (
                ComparisonExpressionType::Not(NotKeywordTokenType::Not, Box::new(inner)),
                meta,
            )
        } else {
            let arith = self.expect_arithmetic_expression()?;
            let meta = arith.meta;

            (
                ComparisonExpressionType::ComparisonChain(ComparisonChainExpression {
                    meta: arith.meta,
                    main_expression: arith,
                    sub_expressions: vec![],
                }),
                meta,
            )
        };
        Ok(ComparisonExpression {
            comparison_type,
            meta,
        })
    }

    fn expect_arithmetic_expression(&mut self) -> ParserResult<ArithmeticExpression<'a>> {
        self.step_over_whitespace();

        let term = self.expect_term()?;
        Ok(ArithmeticExpression {
            meta: term.meta,
            main_expression: term,
            sub_expressions: vec![],
        })
    }

    fn expect_term(&mut self) -> ParserResult<Term<'a>> {
        self.step_over_whitespace();

        let factor = self.expect_factor()?;
        let meta = factor.meta;
        Ok(Term {
            main_expression: factor,
            sub_expressions: vec![],
            meta,
        })
    }

    fn expect_factor(&mut self) -> ParserResult<Factor<'a>> {
        self.step_over_whitespace();

        let call = self.expect_call()?;
        let meta = call.meta;
        Ok(Factor {
            factor_type: FactorType::Call(call),
            meta,
        })
    }

    fn expect_call(&mut self) -> ParserResult<Call<'a>> {
        self.step_over_whitespace();
        let atom = self.expect_atom()?;

        self.step_over_whitespace();
        let mut params: Option<Vec<Expression<'a>>> = None;
        if self.current_token.token_type
            == TokenType::Operator(OperatorTokenType::LParen(LParenOperatorTokenType::LParen))
        {
            self.next_token();
            params = Some(vec![]);
            let expr = self.expect_expression()?;
            params.as_mut().unwrap().push(expr);
            self.step_over_whitespace();
            while self.current_token.token_type
                == TokenType::Operator(OperatorTokenType::Comma(CommaOperatorTokenType::Comma))
            {
                let next_expr = self.expect_expression()?;
                params.as_mut().unwrap().push(next_expr);
                self.step_over_whitespace();
            }
            if self.current_token.token_type
                != TokenType::Operator(OperatorTokenType::RParen(RParenOperatorTokenType::RParen))
            {
                return Err(ParserError);
            }
            self.next_token();
        }
        Ok(Call {
            meta: atom.meta,
            atom,
            params,
        })
    }

    fn expect_atom(&mut self) -> ParserResult<Atom<'a>> {
        self.step_over_whitespace();

        let atom_type = match self.current_token.token_type {
            TokenType::Literal(LiteralTokenType::Int) => {
                let value = parse_int(self.current_token.contents);
                if value > IBig::from(usize::MAX) {
                    let idx = self.program.big_int_literals.register_value(value.clone());
                    AtomType::Int(Int::Big(idx))
                } else {
                    AtomType::Int(Int::Small(value.try_into().unwrap()))
                }
            }
            TokenType::Literal(LiteralTokenType::String) => {
                let value = self
                    .current_token
                    .contents
                    .get(1..self.current_token.contents.len() - 1)
                    .unwrap_or("");
                let idx = self.program.string_literals.register_value(value);
                AtomType::String(idx)
            }
            TokenType::Identifier => AtomType::Identifier(self.current_token.contents),
            _ => unreachable!(),
        };
        let meta = NodeMetaData {
            index: self.current_token.start.index,
            column_number: self.current_token.start.column_number,
            line_number: self.current_token.start.line_number,
            len: self.current_token.len,
            contents: self.current_token.contents,
        };
        self.next_token();
        Ok(Atom { atom_type, meta })
    }

    fn next_token(&mut self) {
        self.index += 1;
        self.current_token = self.tokens[self.index];
    }

    fn has_tokens_left(&self) -> bool {
        self.index < self.tokens.len()
    }
}

fn parse_int(contents: &str) -> IBig {
    let mut res = ibig!(0);
    if matches!(contents.get(0..=1), Some("0b") | Some("0B")) {
        let mut bytes = contents.bytes();
        bytes.next();
        bytes.next();
        for b in contents.bytes() {
            match b {
                b'_' => (),
                _ => {
                    res *= 2;
                    res += b - 48;
                }
            };
        }
    } else if matches!(contents.get(0..=1), Some("0x") | Some("0X")) {
        let mut bytes = contents.bytes();
        bytes.next();
        bytes.next();
        for b in contents.bytes() {
            match b {
                b'0'..=b'9' => {
                    res *= 16;
                    res += b - 48;
                }
                b'a'..=b'f' => {
                    res *= 16;
                    res += b - 87;
                }
                b'A'..=b'F' => {
                    res *= 16;
                    res += b - 55;
                }
                // b'_'
                _ => (),
            };
        }
    } else {
        for b in contents.bytes() {
            match b {
                b'_' => (),
                _ => {
                    res *= 10;
                    res += b - 48;
                }
            };
        }
    }

    res
}
