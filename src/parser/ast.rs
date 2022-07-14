use crate::lexer::lex::{
    access_modifier_kw_tt, and, assignment_operator_tt, bitshift_left, bitshift_right, bitwise_and,
    bitwise_or, bitwise_xor, break_, class, comma, comment_tt, continue_, docs_block_comment, elif,
    else_, eoi, equals, exponent, false_, fun, greater_than, greater_than_equals, identifier, if_,
    int, l_paren, less_than, less_than_equals, loop_, minus, normal_block_comment, not, not_equals,
    or, percent, plus, r_paren, return_, slash, star, static_, string, sub_type_of, terminator,
    true_, unary_operator_tt, while_, white_space, AccessModifierTokenType,
    ArithmeticOperatorTokenType, AssignmentOperatorTokenType, BooleanComparisonKeywordTokenType,
    ComparisonOperatorTokenType, KeywordTokenType, LexerError, LexerErrorContents,
    TermOperatorTokenType, Token, TokenType, UnaryOperatorTokenType,
};
use crate::runtime::bytecode::{Program, ValueStore};
use ibig::{ibig, IBig};
use std::fmt::{Debug, Display};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParserErrorContents<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) message: String,
    pub(crate) input: &'a str,
    pub(crate) offending_tokens: Vec<Token<'a>>,
}
impl<'a> Display for ParserError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongForm => unreachable!(),
            Self::IncorrectSyntax(e) => {
                let v: Vec<_> = e
                    .iter()
                    .map(|x| LexerErrorContents {
                        line_number: x.position.start.line_number,
                        column_number: x.position.start.column_number,
                        error_message: x.message.clone(),
                        index: x.offending_tokens[0].start.index,
                        input: x.input,
                        len: x.offending_tokens[x.offending_tokens.len() - 1].end.index
                            - x.offending_tokens[0].start.index
                            + 1,
                    })
                    .collect();
                let l = LexerError::Incorrect(v);
                writeln!(f, "{l}")?;
                Ok(())
            }
        }
    }
}

impl std::error::Error for ParserError<'_> {}

pub(crate) type ParserResult<'a, T> = Result<T, ParserError<'a>>;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct NodePositionData<'a> {
    pub(crate) contents: &'a str,
    pub(crate) index: usize,
    pub(crate) len: usize,
    pub(crate) start: Coordinate,
    pub(crate) end: Coordinate,
}
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct Coordinate {
    pub(crate) index: usize,
    pub(crate) line_number: usize,
    pub(crate) column_number: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Block<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) children: Vec<BlockChild<'a>>,
    pub(crate) indentation_level: usize,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BlockChild<'a> {
    Statement(Box<Statement<'a>>),
    Block(Block<'a>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Statement<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) statement_type: StatementType<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StatementType<'a> {
    Return(Option<Expression<'a>>),
    Continue,
    Break,
    Expr(Expression<'a>),
    LoopStatement(LoopStatement<'a>),
    WhileStatement(WhileStatement<'a>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoopStatement<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) body: Block<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhileStatement<'a> {
    pub(crate) condition: Expression<'a>,
    pub(crate) body: Block<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Expression<'a> {
    pub(crate) main_expression: ComparisonExpression<'a>,
    pub(crate) sub_expressions: Vec<SubExpression<'a>>,
    pub(crate) position: NodePositionData<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubExpression<'a> {
    pub(crate) keyword: BooleanComparisonKeywordTokenType,
    pub(crate) expression: ComparisonExpression<'a>,
    pub(crate) position: NodePositionData<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComparisonExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) comparison_type: ComparisonExpressionType<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ComparisonExpressionType<'a> {
    Not(Box<ComparisonExpression<'a>>),
    ComparisonChain(Box<ComparisonChainExpression<'a>>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComparisonChainExpression<'a> {
    pub(crate) main_expression: ArithmeticExpression<'a>,
    pub(crate) sub_expressions: Vec<SubComparisonExpression<'a>>,
    pub(crate) position: NodePositionData<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubComparisonExpression<'a> {
    pub(crate) operator: ComparisonOperatorTokenType,
    pub(crate) expression: ArithmeticExpression<'a>,
    pub(crate) position: NodePositionData<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArithmeticExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) main_expression: TermExpression<'a>,
    pub(crate) sub_expressions: Vec<SubArithmeticExpression<'a>>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubArithmeticExpression<'a> {
    pub(crate) operator: ArithmeticOperatorTokenType,
    pub(crate) expression: TermExpression<'a>,
    pub(crate) position: NodePositionData<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TermExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) main_expression: FactorExpression<'a>,
    pub(crate) sub_expressions: Vec<SubTermExpression<'a>>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubTermExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) operator: TermOperatorTokenType,
    pub(crate) expression: FactorExpression<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FactorExpression<'a> {
    pub(crate) factor_type: FactorExpressionType<'a>,
    pub(crate) position: NodePositionData<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FactorExpressionType<'a> {
    UnaryFactor(UnaryOperatorTokenType, Box<FactorExpression<'a>>),
    Exponent(Box<ExponentExpression<'a>>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExponentExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) main_expression: CallExpression<'a>,
    pub(crate) sub_expressions: Vec<SubExponentExpression<'a>>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubExponentExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) expression: CallExpression<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CallExpression<'a> {
    pub(crate) atom: AtomicExpression<'a>,
    /// The params for this function calls and all the ones immediately following it.
    /// For instance, consider this code: `foo(1, 2, 3)(4, 5, 6)`. Here we are calling two functions in a row. We parse this into our vec of vecs of params as [[1, 2, 3], [4, 5, 6]].
    pub(crate) func_call_params: Option<Vec<Vec<Expression<'a>>>>,
    pub(crate) position: NodePositionData<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AtomicExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) atom_type: AtomicExpressionType<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AtomicExpressionType<'a> {
    Int(Int),
    Identifier(usize),
    String(usize),
    Bool(bool),
    Assignment(AssignmentExpression<'a>),
    ExpressionWithParentheses(Box<Expression<'a>>),
    IfExpression(Box<IfExpression<'a>>),
    List(ListExpression),
    Dict(DictExpression),
    For(ForExpression),
    FuncDef(FunctionDefinitionExpression<'a>),
    ClassDef(ClassDefinitionExpression<'a>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Int {
    Big(usize),
    Small(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AssignmentExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) assignment_type: AssignmentExpressionType<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]

pub(crate) enum AssignmentExpressionType<'a> {
    ClassMember(ClassMember<'a>),
    Ident(usize, AssignmentOperatorTokenType, Expression<'a>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IfExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) condition: Expression<'a>,
    pub(crate) then_block: Block<'a>,
    pub(crate) elif_expressions: Vec<ElifExpression<'a>>,
    pub(crate) else_expression: Option<ElseExpression<'a>>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ElifExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) condition: Expression<'a>,
    pub(crate) then_block: Block<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ElseExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) else_block: Block<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct ListExpression;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct DictExpression;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct ForExpression;
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct WhileExpression;
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FunctionDefinitionExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) args: Vec<&'a str>,
    pub(crate) body: Block<'a>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClassDefinitionExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) body: ClassBody<'a>,
    pub(crate) super_types: Vec<usize>,
    pub(crate) ctx: Context,
} 

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClassMember<'a> {
    pub(crate) is_static: bool,
    pub(crate) access_modifier: AccessModifierTokenType,
    pub(crate) name: usize,
    pub(crate) value: Option<Expression<'a>>,
    pub(crate) operator: Option<AssignmentOperatorTokenType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ClassBody<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) members: Vec<AssignmentExpression<'a>>,
    pub(crate) ctx: Context,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParserError<'a> {
    WrongForm,
    IncorrectSyntax(Vec<ParserErrorContents<'a>>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Ast<'a> {
    pub(crate) index: usize,
    pub(crate) tokens: Vec<Token<'a>>,
    pub(crate) program: Program<'a>,
    pub(crate) input: &'a str,
}

impl<'a> Ast<'a> {
    pub fn new(tokens: Vec<Token<'a>>, input: &'a str) -> Self {
        Ast {
            index: 0,
            program: Program {
                base_block: None,
                function_definitions: ValueStore::new(),
                big_int_literals: ValueStore::new(),
                string_literals: ValueStore::new(),
                identifier_literals: ValueStore::new(),
            },
            tokens,
            input,
        }
    }
    fn current_token(&self) -> Token<'a> {
        self.tokens[self.index]
    }

    fn current_column_number(&self) -> usize {
        self.current_token().start.column_number
    }

    fn current_token_type(&self) -> TokenType {
        self.current_token().token_type
    }
    fn create_standard_position(&mut self, first_token_index: usize) -> NodePositionData<'a> {
        self.create_position(first_token_index, self.index - first_token_index + 1)
    }
    fn create_position(
        &mut self,
        first_token_index: usize,
        mut number_of_tokens_used: usize,
    ) -> NodePositionData<'a> {
        if self.current_token_type() == eoi!() {
            number_of_tokens_used -= 1
        }
        let first_token = self.tokens[first_token_index];
        let last_token = self.tokens[first_token_index + number_of_tokens_used - 1];

        NodePositionData {
            index: first_token_index,
            len: number_of_tokens_used,
            contents: &self.input[first_token.start.index..=last_token.end.index],
            start: Coordinate {
                index: first_token.start.index,
                column_number: first_token.start.column_number,
                line_number: first_token.start.line_number,
            },
            end: Coordinate {
                index: last_token.end.index,
                column_number: last_token.end.column_number,
                line_number: last_token.end.line_number,
            },
        }
    }

    fn create_parse_error_with_message(
        &mut self,
        first_token_index: usize,
        number_of_tokens_used: usize,
        message: String,
    ) -> ParserError<'a> {
        ParserError::IncorrectSyntax(vec![ParserErrorContents {
            message,
            position: self.create_position(first_token_index, number_of_tokens_used),
            input: self.input,
            offending_tokens: self.tokens
                [first_token_index..first_token_index + number_of_tokens_used]
                .to_vec(),
        }])
    }

    fn step_over_whitespace_and_block_comments(&mut self) {
        while matches!(
            self.current_token_type(),
            white_space!() | docs_block_comment!() | normal_block_comment!()
        ) {
            self.index += 1;
        }
    }

    fn step_over_whitespace_and_comments(&mut self) {
        while matches!(self.current_token_type(), white_space!() | comment_tt!()) {
            self.index += 1;
        }
    }

    fn step_over_whitespace_and_comments_and_terminators(&mut self) {
        while matches!(
            self.current_token_type(),
            white_space!() | comment_tt!() | terminator!()
        ) {
            self.index += 1;
        }
    }

    pub fn parse(mut self) -> ParserResult<'a, Self> {
        let base_block = Block::expect(&mut self, Context {
            is_class_definition: false,
            has_return_value: false,
        })?;

        self.program.base_block = Some(base_block);
        Ok(self)
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
        for b in bytes {
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
        for b in bytes {
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

trait ExpectSelf<'a>: Sized {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self>;
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Context {
    pub(crate) has_return_value: bool,
    pub(crate) is_class_definition: bool,

}
impl<'a> ExpectSelf<'a> for Block<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        let first_token_index = ast.index;
        ast.step_over_whitespace_and_block_comments();
        let indentation_level_for_this_block = ast.current_column_number();
        let mut children = Vec::new();
        let mut last_valid_index = first_token_index;
        loop {
            ast.step_over_whitespace_and_comments_and_terminators();
            if ast.current_token().token_type == eoi!() {
                break;
            }
            let cmp = ast
                .current_column_number()
                .cmp(&indentation_level_for_this_block);
            match cmp {
                std::cmp::Ordering::Less => break,

                std::cmp::Ordering::Equal => {
                    let mut new_ctx = ctx;
                    new_ctx.has_return_value = false;
                    let statement = match Statement::expect(ast, new_ctx) {
                        Ok(statement) => statement,
                        Err(error) => match error {
                            ParserError::WrongForm => break,

                            _ => return Err(error),
                        },
                    };
                    children.push(BlockChild::Statement(Box::new(statement)));
                }
                std::cmp::Ordering::Greater => {
                    let mut new_ctx = ctx;
                    new_ctx.has_return_value = false;
                    let block = match Block::expect(ast, new_ctx) {
                        Ok(block) => block,
                        Err(error) => match error {
                            ParserError::WrongForm => break,

                            _ => return Err(error),
                        },
                    };
                    children.push(BlockChild::Block(block));
                }
            }
            last_valid_index = ast.index;
            ast.index += 1;
        }
        ast.index = last_valid_index;
        if ast.current_token_type() == eoi!() {
            ast.index -= 1;
        }
        if ast.index < first_token_index {
            ast.index = first_token_index;
            return Err(ParserError::WrongForm);
        }
        let number_of_tokens_used = ast.index - first_token_index + 1;

        let mut b = Block {
            position: ast.create_position(first_token_index, number_of_tokens_used),
            indentation_level: indentation_level_for_this_block,
            children,
            ctx,
        };
        match b.children.last_mut() {
            None => (),
            Some(ref mut v) => match v {
                BlockChild::Block(b) => {
                    b.ctx.has_return_value = ctx.has_return_value
                }
                BlockChild::Statement(s) => {
                    if b.ctx.has_return_value {
                        s.ctx.has_return_value = true;
                        match s.statement_type {
                            StatementType::Expr(_) => (),
                            _ => {
                                return Err(ast.create_parse_error_with_message(
                                    s.position.index,
                                    s.position.len,
                                    "Blocks that are expressions must end with an expression."
                                        .into(),
                                ))
                            }
                        }
                    }
                }
            },
        }

        Ok(b)
    }
}

impl<'a> ExpectSelf<'a> for Statement<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        let indent_at_start = ast.current_column_number();
        ast.step_over_whitespace_and_comments_and_terminators();
        let index_before_statement = ast.index;
        if ast.current_column_number() < indent_at_start {
            ast.index = index_before_statement;
            return Err(ParserError::WrongForm);
        }

        match ast.current_token_type() {
            break_!() => {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_comments();
                match ast.current_token_type() {
                    terminator!() => {
                        return Ok(Statement {
                            statement_type: StatementType::Break,
                            position: ast.create_standard_position(first_token_index),
                            ctx,
                        });
                    }
                    _ => {
                        return Err(ast.create_parse_error_with_message(
                            ast.index,
                            1,
                            "Expected a terminator after break keyword.".to_string(),
                        ))
                    }
                }
            }
            continue_!() => {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_comments();
                match ast.current_token_type() {
                    terminator!() => {
                        return Ok(Statement {
                            statement_type: StatementType::Continue,
                            position: ast.create_standard_position(first_token_index),
                            ctx,
                        });
                    }
                    _ => {
                        return Err(ast.create_parse_error_with_message(
                            ast.index,
                            1,
                            "Expected a terminator after continue keyword.".to_string(),
                        ))
                    }
                }
            }
            return_!() => {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_comments();
                let mut new_ctx = ctx;
                new_ctx.has_return_value = true;
                match ast.current_token_type() {
                    terminator!() => {
                        return Ok(Statement {
                            statement_type: StatementType::Return(None),
                            position: ast.create_standard_position(first_token_index),
                            ctx,
                        });
                    }
                    _ => match Expression::expect(ast, new_ctx) {
                        Ok(expression) => {
                            return Ok(Statement {
                                statement_type: StatementType::Return(Some(expression)),
                                position: ast.create_standard_position(first_token_index),
                                ctx,
                            });
                        }
                        Err(error) => match error {
                            ParserError::WrongForm => {
                                return Err(ast.create_parse_error_with_message(
                                    ast.index,
                                    1,
                                    "Expected a terminator or an expression after return keyword."
                                        .to_string(),
                                ))
                            }
                            _ => Err(error),
                        },
                    },
                }
            }
            loop_!() => {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_comments();
                if ast.current_token().token_type != terminator!() {
                    return Err(ast.create_parse_error_with_message(
                        ast.index,
                        1,
                        "Expected a terminator after 'loop' keyword".into(),
                    ));
                }
                ast.index += 1;
                ast.step_over_whitespace_and_comments_and_terminators();
                let mut new_ctx = ctx;
                new_ctx.has_return_value = false;
                let body = match Block::expect(ast, new_ctx) {
                    Ok(v) => v,
                    Err(e) => match e {
                        ParserError::WrongForm => {
                            return Err(ast.create_parse_error_with_message(
                                first_token_index,
                                ast.index - first_token_index + 1,
                                "Expected an indented block after 'loop' keyword".into(),
                            ))
                        }
                        _ => return Err(e),
                    },
                };
                let pos = ast.create_standard_position(first_token_index);
                return Ok(Statement {
                    statement_type: StatementType::LoopStatement(LoopStatement {
                        position: pos,
                        body,
                        ctx,
                    }),
                    position: pos,
                    ctx,
                });
            }
            while_!() => {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_block_comments();
                let mut new_ctx = ctx;
                new_ctx.has_return_value = true;
                let condition = match Expression::expect(ast, new_ctx) {
                    Ok(v) => v,
                    Err(e) => match e {
                        ParserError::WrongForm => {
                            return Err(ast.create_parse_error_with_message(
                                first_token_index,
                                ast.index - first_token_index + 1,
                                "Expected an expression after 'while' keyword".into(),
                            ))
                        }
                        _ => return Err(e),
                    },
                };
                ast.index += 1;
                ast.step_over_whitespace_and_comments();
                if ast.current_token().token_type != terminator!() {
                    return Err(ast.create_parse_error_with_message(
                        ast.index,
                        1,
                        "Expected a terminator after 'while' condition".into(),
                    ));
                }
                ast.index += 1;
                ast.step_over_whitespace_and_comments_and_terminators();
                let mut new_ctx = ctx;
                new_ctx.has_return_value = false;
                let body = match Block::expect(ast, new_ctx) {
                    Ok(v) => v,
                    Err(e) => match e {
                        ParserError::WrongForm => {
                            return Err(ast.create_parse_error_with_message(
                                first_token_index,
                                ast.index - first_token_index + 1,
                                "Expected an indented block after 'while' condition".into(),
                            ))
                        }
                        _ => return Err(e),
                    },
                };
                let statement_type =
                    StatementType::WhileStatement(WhileStatement { condition, body, ctx, });
                return Ok(Statement {
                    statement_type,
                    position: ast.create_standard_position(first_token_index),
                    ctx,
                });
            }
            _ => {
                ast.index = index_before_statement;

                let expression = Expression::expect(ast, ctx)?;
                return Ok(Statement {
                    position: expression.position,
                    statement_type: StatementType::Expr(expression),
                    ctx,
                });
            }
        }
    }
}

impl<'a> ExpectSelf<'a> for Expression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let comp_expr = ComparisonExpression::expect(ast, ctx)?;
        ast.index += 1;
        let mut sub_expressions = vec![];
        if ast.current_token_type() == eoi!() {
            ast.index -= 1;

            return Ok(Expression {
                position: ast.create_standard_position(first_token_index),
                main_expression: comp_expr,
                sub_expressions,
                ctx,
            });
        }
        loop {
            let sub_expr = match SubExpression::expect(ast, ctx) {
                Ok(v) => v,
                Err(e) => match e {
                    ParserError::WrongForm => {
                        break;
                    }
                    _ => return Err(e),
                },
            };
            ast.index += 1;
            sub_expressions.push(sub_expr);
            if ast.current_token_type() == eoi!() {
                ast.index -= 1;

                return Ok(Expression {
                    position: ast.create_standard_position(first_token_index),
                    main_expression: comp_expr,
                    sub_expressions,
                    ctx,
                });
            }
        }
        ast.index -= 1;

        Ok(Expression {
            position: ast.create_standard_position(first_token_index),
            sub_expressions,
            main_expression: comp_expr,
            ctx,
        })
    }
}

impl<'a> ExpectSelf<'a> for SubExpression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let keyword = match ast.current_token().token_type {
            and!() => BooleanComparisonKeywordTokenType::And,
            or!() => BooleanComparisonKeywordTokenType::Or,
            _ => return Err(ParserError::WrongForm),
        };
        ast.index += 1;
        ast.step_over_whitespace_and_block_comments();

        let expr = match ComparisonExpression::expect(ast, ctx) {
            Ok(v) => v,
            Err(e) => match e {
                ParserError::WrongForm => return Err(ast.create_parse_error_with_message(ast.index, 1, "the 'and' and 'or' keywords must be followed by a valid comparison expression.".into())),
                _ => return Err(e),
            }
        };

        Ok(SubExpression {
            keyword,
            expression: expr,
            position: ast.create_standard_position(first_token_index),
            ctx,
        })
    }
}

impl<'a> ExpectSelf<'a> for ComparisonExpression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let comparison_type = if ast.current_token().token_type == not!() {
            ast.index += 1;

            let inner =
                match ComparisonExpression::expect(ast, ctx) {
                    Ok(v) => v,
                    Err(e) => match e {
                        ParserError::WrongForm => return Err(ast.create_parse_error_with_message(
                            ast.index,
                            1,
                            "the 'not' keyword must be followed by a valid comparison expression."
                                .into(),
                        )),
                        _ => return Err(e),
                    },
                };
            ComparisonExpressionType::Not(Box::new(inner))
        } else {
            let first_token_index = ast.index;
            let arith = ArithmeticExpression::expect(ast, ctx)?;
            ast.index += 1;
            let mut sub_exprs = vec![];
            if ast.current_token_type() == eoi!() {
                ast.index -= 1;
                let position = ast.create_standard_position(first_token_index);
                return Ok(ComparisonExpression {
                    position,
                    comparison_type: ComparisonExpressionType::ComparisonChain(Box::new(
                        ComparisonChainExpression {
                            main_expression: arith,
                            sub_expressions: sub_exprs,
                            position,
                            ctx,
                        },
                    )),
                    ctx,
                });
            }
            loop {
                let sub_expr = match SubComparisonExpression::expect(ast, ctx) {
                    Ok(v) => v,
                    Err(e) => match e {
                        ParserError::WrongForm => {
                            ast.index -= 1;
                            break;
                        }
                        _ => return Err(e),
                    },
                };
                ast.index += 1;
                sub_exprs.push(sub_expr);
                if ast.current_token_type() == eoi!() {
                    ast.index -= 1;
                    let position = ast.create_standard_position(first_token_index);
                    return Ok(ComparisonExpression {
                        ctx,
                        position,
                        comparison_type: ComparisonExpressionType::ComparisonChain(Box::new(
                            ComparisonChainExpression {
                                main_expression: arith,
                                sub_expressions: sub_exprs,
                                position,
                                ctx,
                            },
                        )),
                    });
                }
            }

            ComparisonExpressionType::ComparisonChain(Box::new(ComparisonChainExpression {
                position: ast.create_standard_position(first_token_index),
                main_expression: arith,
                sub_expressions: sub_exprs,
                ctx,
            }))
        };

        Ok(ComparisonExpression {
            comparison_type,
            position: ast.create_standard_position(first_token_index),
            ctx,
        })
    }
}

impl<'a> ExpectSelf<'a> for SubComparisonExpression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let operator = match ast.current_token().token_type {
            equals!() => ComparisonOperatorTokenType::Equals,
            not_equals!() => ComparisonOperatorTokenType::NotEquals,
            less_than!() => ComparisonOperatorTokenType::LessThan,
            greater_than!() => ComparisonOperatorTokenType::GreaterThan,
            less_than_equals!() => ComparisonOperatorTokenType::LessThanEquals,
            greater_than_equals!() => ComparisonOperatorTokenType::GreaterThanEquals,
            _ => return Err(ParserError::WrongForm),
        };
        ast.index += 1;
        ast.step_over_whitespace_and_block_comments();
        let expr = match ArithmeticExpression::expect(ast, ctx) {
            Ok(v) => v,
            Err(e) => match e {
                ParserError::WrongForm => return Err(ast.create_parse_error_with_message(ast.index, 1, "the '==', '!=', '<', '>', '<=', and '>=' operators must be followed by a valid arithmetic expression.".into())),
                _ => return Err(e),
            }
        };

        Ok(SubComparisonExpression {
            operator,
            expression: expr,
            position: ast.create_standard_position(first_token_index),
            ctx,
        })
    }
}

impl<'a> ExpectSelf<'a> for ArithmeticExpression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let term = TermExpression::expect(ast, ctx)?;
        ast.index += 1;
        let mut sub_exprs = vec![];
        if ast.current_token_type() == eoi!() {
            ast.index -= 1;
            let position = ast.create_standard_position(first_token_index);

            return Ok(ArithmeticExpression {
                position,
                main_expression: term,
                sub_expressions: sub_exprs,
                ctx,
            });
        }
        loop {
            let sub_expr = match SubArithmeticExpression::expect(ast, ctx) {
                Ok(v) => v,
                Err(e) => match e {
                    ParserError::WrongForm => {
                        ast.index -= 1;
                        break;
                    }
                    _ => return Err(e),
                },
            };
            ast.index += 1;
            sub_exprs.push(sub_expr);
            if ast.current_token_type() == eoi!() {
                ast.index -= 1;
                let position = ast.create_standard_position(first_token_index);

                return Ok(ArithmeticExpression {
                    position,
                    main_expression: term,
                    sub_expressions: sub_exprs,
                    ctx,
                });
            }
        }

        Ok(ArithmeticExpression {
            position: ast.create_standard_position(first_token_index),
            main_expression: term,
            sub_expressions: sub_exprs,
            ctx,
        })
    }
}

impl<'a> ExpectSelf<'a> for SubArithmeticExpression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let operator = match ast.current_token().token_type {
            plus!() => ArithmeticOperatorTokenType::Unary(UnaryOperatorTokenType::Plus),
            minus!() => ArithmeticOperatorTokenType::Unary(UnaryOperatorTokenType::Minus),
            bitwise_and!() => ArithmeticOperatorTokenType::BitwiseAnd,
            bitwise_or!() => ArithmeticOperatorTokenType::BitwiseOr,
            bitwise_xor!() => ArithmeticOperatorTokenType::BitwiseXor,
            _ => return Err(ParserError::WrongForm),
        };
        ast.index += 1;
        ast.step_over_whitespace_and_block_comments();

        let expr = match TermExpression::expect(ast, ctx) {
            Ok(v) => v,
            Err(e) => match e {
                ParserError::WrongForm => return Err(ast.create_parse_error_with_message(ast.index, 1, "the '+', '-', '&', '|', and '^' operators must be followed by a valid term expression.".into())),
                _ => return Err(e),
            }
        };

        Ok(SubArithmeticExpression {
            operator,
            expression: expr,
            position: ast.create_standard_position(first_token_index),
            ctx,
        })
    }
}

impl<'a> ExpectSelf<'a> for TermExpression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let factor = FactorExpression::expect(ast, ctx)?;
        ast.index += 1;
        let mut sub_exprs = vec![];
        if ast.current_token_type() == eoi!() {
            ast.index -= 1;
            let position = ast.create_standard_position(first_token_index);

            return Ok(TermExpression {
                position,
                main_expression: factor,
                sub_expressions: sub_exprs,
                ctx,
            });
        }
        loop {
            let sub_expr = match SubTermExpression::expect(ast, ctx) {
                Ok(v) => v,
                Err(e) => match e {
                    ParserError::WrongForm => {
                        ast.index -= 1;
                        break;
                    }
                    _ => return Err(e),
                },
            };
            ast.index += 1;
            sub_exprs.push(sub_expr);
            if ast.current_token_type() == eoi!() {
                ast.index -= 1;
                let position = ast.create_standard_position(first_token_index);

                return Ok(TermExpression {
                    position,
                    main_expression: factor,
                    sub_expressions: sub_exprs,
                    ctx,
                });
            }
        }

        Ok(TermExpression {
            main_expression: factor,
            sub_expressions: sub_exprs,
            position: ast.create_standard_position(first_token_index),
            ctx,
        })
    }
}

impl<'a> ExpectSelf<'a> for SubTermExpression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let operator = match ast.current_token().token_type {
            star!() => TermOperatorTokenType::Star,
            slash!() => TermOperatorTokenType::Slash,
            percent!() => TermOperatorTokenType::Percent,
            bitshift_left!() => TermOperatorTokenType::BitshiftLeft,
            bitshift_right!() => TermOperatorTokenType::BitshiftRight,
            _ => return Err(ParserError::WrongForm),
        };
        ast.index += 1;
        ast.step_over_whitespace_and_block_comments();
        let expr = match FactorExpression::expect(ast, ctx) {
            Ok(v) => v,
            Err(e) => match e {
                ParserError::WrongForm => return Err(ast.create_parse_error_with_message(ast.index, 1, "the '*', '/', '%', '<<', and '>>' operators must be followed by a valid factor expression.".into())),
                _ => return Err(e),
            }
        };

        Ok(SubTermExpression {
            operator,
            expression: expr,
            position: ast.create_standard_position(first_token_index),
            ctx,
        })
    }
}

impl<'a> ExpectSelf<'a> for FactorExpression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        match ast.current_token().token_type {
            unary_operator_tt!(v) => {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_block_comments();
                let expr = match FactorExpression::expect(ast, ctx) {
                    Ok(v) => v,
                    Err(e) => match e {
                        ParserError::WrongForm => return Err(ast.create_parse_error_with_message(ast.index, 1, "the unary operators '~', '+' and '-' must be followed by a valid factor expression.".into())),
                        _ => return Err(e),
                    }
                };

                Ok(FactorExpression {
                    factor_type: FactorExpressionType::UnaryFactor(v, Box::new(expr)),
                    position: ast.create_standard_position(first_token_index),
                    ctx,
                })
            }
            _ => {
                let exponent = ExponentExpression::expect(ast, ctx)?;

                Ok(FactorExpression {
                    factor_type: FactorExpressionType::Exponent(Box::new(exponent)),
                    position: ast.create_standard_position(first_token_index),
                    ctx,
                })
            }
        }
    }
}

impl<'a> ExpectSelf<'a> for ExponentExpression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let call = CallExpression::expect(ast, ctx)?;
        ast.index += 1;
        let mut sub_exprs = vec![];
        if ast.current_token_type() == eoi!() {
            ast.index -= 1;
            let position = ast.create_standard_position(first_token_index);

            return Ok(ExponentExpression {
                position,
                main_expression: call,
                sub_expressions: sub_exprs,
                ctx,
            });
        }
        loop {
            let sub_expr = match SubExponentExpression::expect(ast, ctx) {
                Ok(v) => v,
                Err(e) => match e {
                    ParserError::WrongForm => {
                        ast.index -= 1;
                        break;
                    }
                    _ => return Err(e),
                },
            };
            ast.index += 1;
            sub_exprs.push(sub_expr);
            if ast.current_token_type() == eoi!() {
                ast.index -= 1;
                let position = ast.create_standard_position(first_token_index);

                return Ok(ExponentExpression {
                    position,
                    main_expression: call,
                    sub_expressions: sub_exprs,
                    ctx,
                });
            }
        }
        Ok(ExponentExpression {
            main_expression: call,
            sub_expressions: sub_exprs,
            position: ast.create_standard_position(first_token_index),
            ctx,
        })
    }
}

impl<'a> ExpectSelf<'a> for SubExponentExpression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        if ast.current_token_type() != exponent!() {
            return Err(ParserError::WrongForm);
        }
        ast.index += 1;
        ast.step_over_whitespace_and_block_comments();
        let expr =
            match CallExpression::expect(ast, ctx) {
                Ok(v) => v,
                Err(e) => match e {
                    ParserError::WrongForm => return Err(ast.create_parse_error_with_message(
                        ast.index,
                        1,
                        "the exponent operators '**' must be followed by a valid call expression."
                            .into(),
                    )),
                    _ => return Err(e),
                },
            };

        Ok(SubExponentExpression {
            expression: expr,
            position: ast.create_standard_position(first_token_index),
            ctx,
        })
    }
}

impl<'a> ExpectSelf<'a> for CallExpression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let mut atom = AtomicExpression::expect(ast, ctx)?;
        let index_after_atom = ast.index;
        ast.index += 1;
        ast.step_over_whitespace_and_block_comments();
        let mut params = None;
        if ast.current_token_type() == eoi!() {
            ast.index = index_after_atom;

            return Ok(CallExpression {
                atom,
                func_call_params: params,
                position: ast.create_standard_position(first_token_index),
                ctx,
            });
        }
        if ast.current_token().token_type == l_paren!() {
            params = Some(vec![]);
        } else {
            ast.index = index_after_atom;
        }
        while ast.current_token().token_type == l_paren!() {
            ast.index += 1;
            let mut args = vec![];
            let mut new_ctx = ctx;
            new_ctx.has_return_value = true;
            match Expression::expect(ast, new_ctx) {
                Ok(v) => {
                    args.push(v);
                    ast.index += 1;
                    ast.step_over_whitespace_and_block_comments();
                    while ast.current_token().token_type == comma!() {
                        let mut new_ctx = ctx;
                        new_ctx.has_return_value = true;
                        let next_expr = match Expression::expect(ast, new_ctx) {
                            Ok(v) => v,
                            Err(e) => match e {
                                ParserError::WrongForm => {
                                    return Err(ast.create_parse_error_with_message(
                                        ast.index,
                                        1,
                                        "Invalid expression in the middle of function arguments."
                                            .into(),
                                    ))
                                }
                                _ => return Err(e),
                            },
                        };
                        args.push(next_expr);
                        ast.index += 1;
                        ast.step_over_whitespace_and_block_comments();
                        if ast.current_token_type() == eoi!() {
                            return Err(ast.create_parse_error_with_message(
                                ast.index - 1,
                                1,
                                "Reached EOI while trying to parse function arguments".into(),
                            ));
                        }
                    }
                }
                Err(e) => match e {
                    ParserError::WrongForm => (),
                    _ => return Err(e),
                },
            };
            ast.step_over_whitespace_and_block_comments();
            if ast.current_token().token_type != r_paren!() {
                return Err(ast.create_parse_error_with_message(
                    ast.index,
                    1,
                    format!(
                        "Expected a ')' after parameters to function call. Found {:?} instead.",
                        ast.current_token_type()
                    ),
                ));
            }
            if let Some(ref mut v) = params {
                v.push(args);
            }
            let index_after_params = ast.index;
            ast.index += 1;
            ast.step_over_whitespace_and_block_comments();
            if ast.current_token_type() != l_paren!() {
                ast.index = index_after_params;
                break;
            }
        }
        if params.is_some() {
            atom.ctx.has_return_value = true;
        }
        let position = ast.create_standard_position(first_token_index);
        Ok(CallExpression {
            position,
            atom,
            func_call_params: params,
            ctx,
        })
    }
}

impl<'a> ExpectSelf<'a> for AtomicExpression<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let atom_type = match ast.current_token_type() {
            int!() => {
                let value = parse_int(ast.current_token().contents);
                if value > IBig::from(usize::MAX) {
                    let idx = ast.program.big_int_literals.register_value(value);
                    AtomicExpressionType::Int(Int::Big(idx))
                } else {
                    AtomicExpressionType::Int(Int::Small(value.try_into().unwrap()))
                }
            }
            string!() => {
                let value = ast
                    .current_token()
                    .contents
                    .get(1..ast.current_token().contents.len() - 1)
                    .unwrap_or("");
                let idx = ast.program.string_literals.register_value(value);
                AtomicExpressionType::String(idx)
            }
            identifier!() => match AssignmentExpression::expect(ast, ctx) {
                Ok(v) => AtomicExpressionType::Assignment(v),
                Err(e) => match e {
                    ParserError::WrongForm => {
                        let idx = ast
                            .program
                            .identifier_literals
                            .register_value(ast.current_token().contents);

                        AtomicExpressionType::Identifier(idx)
                    }
                    _ => return Err(e),
                },
            },
            true_!() | false_!() => {
                AtomicExpressionType::Bool(ast.current_token().token_type == true_!())
            }
            l_paren!() => {
                let first_token_index = ast.index;
                ast.index += 1;
                let expr = match Expression::expect(ast, ctx) {
                    Ok(v) => v,
                    Err(e) => match e {
                        ParserError::WrongForm => {
                            return Err(ast.create_parse_error_with_message(
                                first_token_index,
                                ast.index - first_token_index + 1,
                                "Expected an expression inside of these parentheses".to_string(),
                            ))
                        }
                        _ => return Err(e),
                    },
                };
                ast.index += 1;
                if ast.current_token().token_type != r_paren!() {
                    return Err(ast.create_parse_error_with_message(
                        first_token_index,
                        ast.index - first_token_index + 1,
                        "Expected ')' after '(' and expression".into(),
                    ));
                }
                AtomicExpressionType::ExpressionWithParentheses(Box::new(expr))
            }
            fun!() => {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_block_comments();
                if ast.current_token().token_type != l_paren!() {
                    return Err(ast.create_parse_error_with_message(
                        first_token_index,
                        ast.index - first_token_index + 1,
                        "Expected '(' after 'fun' keyword".into(),
                    ));
                }
                ast.index += 1;
                ast.step_over_whitespace_and_block_comments();
                let mut args = vec![];
                while ast.current_token().token_type == identifier!() {
                    let ident = ast.current_token().contents;
                    ast.program.identifier_literals.register_value(ident);
                    args.push(ident);
                    ast.index += 1;
                    ast.step_over_whitespace_and_block_comments();
                    if ast.current_token().token_type != comma!() {
                        ast.step_over_whitespace_and_block_comments();
                        break;
                    }
                    ast.index += 1;
                    ast.step_over_whitespace_and_block_comments();
                }
                ast.step_over_whitespace_and_block_comments();
                if ast.current_token().token_type != r_paren!() {
                    return Err(ast.create_parse_error_with_message(
                        ast.index,
                        1,
                        "Expected ')' after function arguments in function definition.".into(),
                    ));
                }
                ast.index += 1;
                ast.step_over_whitespace_and_comments();
                if ast.current_token().token_type != terminator!() {
                    return Err(ast.create_parse_error_with_message(
                        ast.index,
                        1,
                        "Expected a terminator after function definition.".into(),
                    ));
                }
                ast.index += 1;
                ast.step_over_whitespace_and_comments_and_terminators();
                let mut new_ctx = ctx;
                new_ctx.has_return_value = false;
                let body = match Block::expect(ast, new_ctx) {
                    Ok(v) => v,
                    Err(e) => match e {
                        ParserError::WrongForm => {
                            return Err(ast.create_parse_error_with_message(
                                first_token_index,
                                ast.index - first_token_index + 1,
                                "Expected an indented block after function definition.".into(),
                            ))
                        }
                        _ => return Err(e),
                    },
                };
                AtomicExpressionType::FuncDef(FunctionDefinitionExpression {
                    args,
                    body,
                    position: ast.create_standard_position(first_token_index),
                    ctx,
                })
            }
            if_!() => {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_block_comments();
                let mut new_ctx = ctx;
                new_ctx.has_return_value = true;
                let condition = match Expression::expect(ast, new_ctx) {
                    Ok(v) => v,
                    Err(e) => match e {
                        ParserError::WrongForm => {
                            return Err(ast.create_parse_error_with_message(
                                first_token_index,
                                ast.index - first_token_index + 1,
                                "Expected an expression after 'if' keyword".into(),
                            ))
                        }
                        _ => return Err(e),
                    },
                };
                ast.index += 1;
                ast.step_over_whitespace_and_comments();
                if ast.current_token().token_type != terminator!() {
                    return Err(ast.create_parse_error_with_message(
                        ast.index,
                        1,
                        "Expected a terminator after 'if' condition".into(),
                    ));
                }
                ast.index += 1;
                ast.step_over_whitespace_and_comments_and_terminators();
                let then_block = match Block::expect(ast, ctx) {
                    Ok(v) => v,
                    Err(e) => match e {
                        ParserError::WrongForm => {
                            return Err(ast.create_parse_error_with_message(
                                first_token_index,
                                ast.index - first_token_index + 1,
                                "Expected an indented block after 'if' condition".into(),
                            ))
                        }
                        _ => return Err(e),
                    },
                };
                let mut last_valid_index = ast.index;
                ast.index += 1;
                let mut elif_expressions = vec![];
                ast.step_over_whitespace_and_comments_and_terminators();
                while ast.current_token_type() == elif!() {
                    let first_token_index = ast.index;
                    ast.index += 1;
                    ast.step_over_whitespace_and_block_comments();
                    let mut new_ctx = ctx;
                    new_ctx.has_return_value = true;
                    let condition = match Expression::expect(ast, new_ctx) {
                        Ok(v) => v,
                        Err(e) => match e {
                            ParserError::WrongForm => {
                                return Err(ast.create_parse_error_with_message(
                                    first_token_index,
                                    ast.index - first_token_index + 1,
                                    "Expected an expression after 'elif' keyword".into(),
                                ))
                            }
                            _ => return Err(e),
                        },
                    };
                    ast.index += 1;
                    ast.step_over_whitespace_and_comments();
                    if ast.current_token().token_type != terminator!() {
                        return Err(ast.create_parse_error_with_message(
                            ast.index,
                            1,
                            "Expected a terminator after 'elif' condition".into(),
                        ));
                    }
                    ast.index += 1;
                    ast.step_over_whitespace_and_comments_and_terminators();
                    let then_block = match Block::expect(ast, ctx) {
                        Ok(v) => v,
                        Err(e) => match e {
                            ParserError::WrongForm => {
                                return Err(ast.create_parse_error_with_message(
                                    first_token_index,
                                    ast.index - first_token_index + 1,
                                    "Expected an indented block after 'elif' condition".into(),
                                ))
                            }
                            _ => return Err(e),
                        },
                    };
                    elif_expressions.push(ElifExpression {
                        position: ast.create_standard_position(first_token_index),
                        then_block,
                        condition,
                        ctx,
                    });
                    last_valid_index = ast.index;
                    ast.index += 1;
                    ast.step_over_whitespace_and_comments_and_terminators();
                }
                let else_expression = if ast.current_token_type() == else_!() {
                    let first_token_index = ast.index;
                    ast.index += 1;
                    ast.step_over_whitespace_and_block_comments();
                    let else_block = match Block::expect(ast, ctx) {
                        Ok(v) => v,
                        Err(e) => match e {
                            ParserError::WrongForm => {
                                return Err(ast.create_parse_error_with_message(
                                    first_token_index,
                                    ast.index - first_token_index + 1,
                                    "Expected an indented block after 'else' keyword".into(),
                                ))
                            }
                            _ => return Err(e),
                        },
                    };
                    Some(ElseExpression {
                        position: ast.create_standard_position(first_token_index),
                        else_block,
                        ctx,
                    })
                } else {
                    ast.index = last_valid_index;
                    None
                };
                if ctx.has_return_value && else_expression.is_none() {
                    return Err(ast.create_parse_error_with_message(
                        ast.index,
                        1,
                        "Expected an 'else' block after 'if' block with a return value".into(),
                    ));
                }
                let if_expression = IfExpression {
                    condition,
                    then_block,
                    elif_expressions,
                    else_expression,
                    position: ast.create_standard_position(first_token_index),
                    ctx,
                };
                AtomicExpressionType::IfExpression(Box::new(if_expression))
            }
            class!() => {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_block_comments();
                let mut super_types = vec![];
                if ast.current_token_type() == sub_type_of!() {
                    let idx_of_sub_type_of_keyword = ast.index;
                    ast.index += 1;
                    ast.step_over_whitespace_and_block_comments();
                    if ast.current_token_type() != identifier!() {
                        return Err(ast.create_parse_error_with_message(
                            idx_of_sub_type_of_keyword,
                            1,
                            "Expected the name of one or more supertypes after 'sub_type_of' keyword".into(),
                        ));
                    }
                    while ast.current_token_type() == identifier!() {
                        let id = ast
                            .program
                            .identifier_literals
                            .register_value(ast.current_token().contents);
                        super_types.push(id);
                        ast.index += 1;
                        ast.step_over_whitespace_and_block_comments();
                        if ast.current_token_type() == comma!() {
                            ast.index += 1;
                            ast.step_over_whitespace_and_block_comments();
                        }
                    }
                }
                if ast.current_token_type() != terminator!() {
                    return Err(ast.create_parse_error_with_message(
                        ast.index,
                        1,
                        "Expected a terminator after 'class' declaration".into(),
                    ));
                }
                
                let mut new_ctx = ctx;
                new_ctx.is_class_definition = true;

                let body = match ClassBody::expect(ast, new_ctx) {
                    Ok(v) => v,
                    Err(e) => match e {
                        ParserError::WrongForm => {
                            return Err(ast.create_parse_error_with_message(
                                first_token_index,
                                ast.index - first_token_index + 1,
                                "Expected a class body after 'class' declaration".into(),
                            ))
                        }
                        _ => return Err(e),
                    },
                    
                };

                let class_expression = ClassDefinitionExpression {
                    position: ast.create_standard_position(first_token_index),
                    super_types,
                    body,
                    ctx,
                };
                AtomicExpressionType::ClassDef(class_expression)
            }
            terminator!() | r_paren!() => return Err(ParserError::WrongForm),
            x => {
                println!("{x:#?}");
                println!("{}", ast.index);
                for (i, t) in ast.tokens.iter().enumerate() {
                    println!("{i}: {}", t.contents);
                }
                //println!("{:#?}", ast.tokens[ast.index - 1].token_type);
                unreachable!()
            }
        };
        let position = ast.create_standard_position(first_token_index);

        let ret = AtomicExpression {
            atom_type,
            position,
            ctx,
        };
        Ok(ret)
    }
}

impl<'a> ExpectSelf<'a> for AssignmentExpression<'a> {
    fn expect(
        ast: &mut Ast<'a>,
        ctx: Context,
    ) -> ParserResult<'a, Self> {
        let first_token_index = ast.index;
        if let static_!() = ast.current_token_type() {
            ast.index += 1;
            ast.step_over_whitespace_and_block_comments();
            if let access_modifier_kw_tt!() = ast.current_token_type() {
                return Err(ast.create_parse_error_with_message(
                    ast.index,
                    1,
                    "The access modifier must come before the static keyword.".into(),
                ));
            }
        }
        let access_modifier_idx = ast.index;
        let access_modifier =
            if let access_modifier_kw_tt!(access_modifier_tt) = ast.current_token_type() {
                ast.index += 1;
                ast.step_over_whitespace_and_block_comments();
                Some(access_modifier_tt)
            } else {
                None
            };
        let is_class_member = ctx.is_class_definition;
        if !is_class_member && access_modifier.is_some() {
            return Err(ast.create_parse_error_with_message(
                first_token_index,
                1,
                "Accessibility modifiers are not allowed outside of class bodies.".into(),
            ));
        }
        let static_idx = ast.index;
        let is_static = if let static_!() = ast.current_token_type() {
            if !is_class_member {
                return Err(ast.create_parse_error_with_message(
                    ast.index,
                    1,
                    "The static keyword can only be used in class members.".into(),
                ));
            }
            ast.index += 1;
            ast.step_over_whitespace_and_block_comments();
            true
        } else {
            false
        };
        let ident_id;
        let mut operator = None;
        let expr = if let identifier!() = ast.current_token_type() {
            ident_id = ast
                .program
                .identifier_literals
                .register_value(ast.current_token().contents);
            ast.index += 1;
            ast.step_over_whitespace_and_block_comments();
            match ast.current_token_type() {
                assignment_operator_tt!(assign_tok) => {
                    operator = Some(assign_tok);
                    ast.index += 1;
                    ast.step_over_whitespace_and_block_comments();
                    let mut new_ctx = ctx;
                    new_ctx.has_return_value = true;
                    match Expression::expect(ast, new_ctx) {
                        Ok(expression) => Some(expression),
                        Err(error) => match error {
                            ParserError::WrongForm => {
                                if is_class_member {
                                    None
                                } else {
                                    return Err(ast.create_parse_error_with_message(
                                        ast.index,
                                        1,
                                        "Expected an expression after assignment operator."
                                            .to_string(),
                                    ));
                                }
                            }
                            _ => return Err(error),
                        },
                    }
                }

                _ => {
                    if is_class_member {
                        None
                    } else {
                        ast.index = first_token_index;
                        return Err(ParserError::WrongForm);
                    }
                }
            }
        } else {
            if is_static {
                return Err(ast.create_parse_error_with_message(
                    static_idx,
                    1,
                    "The static keyword must be followed by an identifier.".into(),
                ));
            }
            if access_modifier.is_some() {
                return Err(ast.create_parse_error_with_message(
                    access_modifier_idx,
                    1,
                    "An access modifier must be followed by an identifier or the static keyword."
                        .into(),
                ));
            }
            ast.index = first_token_index;
            return Err(ParserError::WrongForm);
        };
        let access_modifier = access_modifier.unwrap_or(AccessModifierTokenType::Prot);
        let position = ast.create_standard_position(first_token_index);
        Ok(AssignmentExpression {
            position,
            ctx,
            assignment_type: {
                if is_class_member {
                    AssignmentExpressionType::ClassMember(ClassMember {
                        access_modifier,
                        is_static,
                        name: ident_id,
                        operator,
                        value: expr,
                    })
                } else {
                    AssignmentExpressionType::Ident(ident_id, operator.unwrap(), expr.unwrap())
                }
            },
        })
    }
}


impl<'a> ExpectSelf<'a> for ClassBody<'a> {
    fn expect(ast: &mut Ast<'a>, ctx: Context) -> ParserResult<'a, Self> {
        unimplemented!()
    }
}