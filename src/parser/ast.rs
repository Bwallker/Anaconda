use crate::lexer::lex::{
    and, assignment_operator_tt, bitshift_left, bitshift_right, bitwise_and, bitwise_or,
    bitwise_xor, break_, comma, comment_tt, continue_, docs_block_comment, elif, else_, eoi,
    equals, exponent, false_, fun, greater_than, greater_than_equals, identifier, if_, int,
    l_paren, less_than, less_than_equals, loop_, minus, normal_block_comment, not, not_equals, or,
    percent, plus, r_paren, return_, slash, star, string, terminator, true_, unary_operator_tt,
    while_, white_space, ArithmeticOperatorTokenType, AssignmentOperatorTokenType,
    BooleanComparisonKeywordTokenType, ComparisonOperatorTokenType, KeywordTokenType,
    NotKeywordTokenType, TermOperatorTokenType, Token, TokenType, UnaryOperatorTokenType,
};
use crate::runtime::bytecode::{Program, ValueStore};
use ibig::{ibig, IBig};
use std::fmt::{Debug, Display};

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
    pub(crate) last_statement_is_expression: bool,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StatementType<'a> {
    Return(Option<Expression<'a>>),
    Continue,
    Break,
    Assignment(usize, AssignmentOperatorTokenType, Expression<'a>),
    Expr(Expression<'a>),
    IfStatement(Box<IfNode<'a>>),
    LoopStatement(LoopStatement<'a>),
    WhileStatement(WhileStatement<'a>),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IfNode<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) condition: Expression<'a>,
    pub(crate) then_block: Block<'a>,
    pub(crate) elif_nodes: Vec<ElifNode<'a>>,
    pub(crate) else_nodes: Option<ElseNode<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ElifNode<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) condition: Expression<'a>,
    pub(crate) then_block: Block<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ElseNode<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) else_block: Block<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoopStatement<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) body: Block<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WhileStatement<'a> {
    pub(crate) condition: Expression<'a>,
    pub(crate) body: Block<'a>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct Identifier<'a> {
    pub(crate) id: &'a str,
    pub(crate) position: NodePositionData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Expression<'a> {
    pub(crate) main_expression: ComparisonExpression<'a>,
    pub(crate) sub_expressions: Vec<SubExpression<'a>>,
    pub(crate) position: NodePositionData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubExpression<'a> {
    pub(crate) keyword: BooleanComparisonKeywordTokenType,
    pub(crate) expression: ComparisonExpression<'a>,
    pub(crate) position: NodePositionData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComparisonExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) comparison_type: ComparisonExpressionType<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ComparisonExpressionType<'a> {
    Not(NotKeywordTokenType, Box<ComparisonExpression<'a>>),
    ComparisonChain(Box<ComparisonChainExpression<'a>>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComparisonChainExpression<'a> {
    pub(crate) main_expression: ArithmeticExpression<'a>,
    pub(crate) sub_expressions: Vec<SubComparisonExpression<'a>>,
    pub(crate) position: NodePositionData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubComparisonExpression<'a> {
    pub(crate) operator: ComparisonOperatorTokenType,
    pub(crate) expression: ArithmeticExpression<'a>,
    pub(crate) position: NodePositionData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArithmeticExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) main_expression: TermExpression<'a>,
    pub(crate) sub_expressions: Vec<SubArithmeticExpression<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubArithmeticExpression<'a> {
    pub(crate) operator: ArithmeticOperatorTokenType,
    pub(crate) expression: TermExpression<'a>,
    pub(crate) position: NodePositionData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TermExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) main_expression: FactorExpression<'a>,
    pub(crate) sub_expressions: Vec<SubTermExpression<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubTermExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) operator: TermOperatorTokenType,
    pub(crate) expression: FactorExpression<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FactorExpression<'a> {
    pub(crate) factor_type: FactorExpressionType<'a>,
    pub(crate) position: NodePositionData<'a>,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubExponentExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) expression: CallExpression<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CallExpression<'a> {
    pub(crate) atom: AtomicExpression<'a>,
    /// The params for this function calls and all the ones immediately following it.
    /// For instance, consider this code: `foo(1, 2, 3)(4, 5, 6)`. Here we are calling two functions in a row. We parse this into our vec of vecs of params as [[1, 2, 3], [4, 5, 6]].
    pub(crate) func_call_params: Option<Vec<Vec<Expression<'a>>>>,
    pub(crate) position: NodePositionData<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AtomicExpression<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) atom_type: AtomicExpressionType<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Int {
    Big(usize),
    Small(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AtomicExpressionType<'a> {
    Int(Int),
    Identifier(usize),
    String(usize),
    Bool(bool),
    ExpressionWithParentheses(Box<Expression<'a>>),
    IfExpression(Box<IfNode<'a>>),
    List(ListExpression),
    Dict(DictExpression),
    For(ForExpression),
    FuncDef(FunctionDefinitionExpression<'a>),
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
    pub(crate) args: Vec<Identifier<'a>>,
    pub(crate) body: Block<'a>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParserError<'a> {
    WrongForm,
    IncorrectSyntax(Vec<ParserErrorContents<'a>>),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParserErrorContents<'a> {
    pub(crate) position: NodePositionData<'a>,
    pub(crate) message: String,
}
impl<'a> Display for ParserError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongForm => unreachable!(),
            Self::IncorrectSyntax(e) => {
                write!(f, "{e:#?}")
            }
        }
    }
}

impl std::error::Error for ParserError<'_> {}

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
            len: last_token.end.index - first_token.start.index,
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
        let base_block = Block::expect(&mut self)?;

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

pub(crate) type ParserResult<'a, T> = Result<T, ParserError<'a>>;

trait ExpectSelf<'a, T = ()>: Sized {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self>;
    #[allow(unused_variables)]
    fn expect_with_extra_args(ast: &mut Ast<'a>, extra_args: T) -> ParserResult<'a, Self> {
        Self::expect(ast)
    }
}

impl<'a> ExpectSelf<'a, bool> for Block<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
        Self::expect_with_extra_args(ast, false)
    }
    fn expect_with_extra_args(
        ast: &mut Ast<'a>,
        last_statement_is_expression: bool,
    ) -> ParserResult<'a, Self> {
        let first_token_index = ast.index;
        ast.step_over_whitespace_and_block_comments();
        let indentation_level_for_this_block = ast.current_column_number();
        let mut children = Vec::new();
        loop {
            ast.step_over_whitespace_and_comments_and_terminators();
            if ast.current_token().token_type == eoi!() {
                break;
            }
            let cmp = ast
                .current_column_number()
                .cmp(&indentation_level_for_this_block);
            match cmp {
                std::cmp::Ordering::Less => {
                    ast.index -= 1;
                    break;
                }
                std::cmp::Ordering::Equal => {
                    let statement = match Statement::expect(ast) {
                        Ok(statement) => statement,
                        Err(error) => match error {
                            ParserError::WrongForm => {
                                ast.index -= 1;
                                break;
                            }
                            _ => return Err(error),
                        },
                    };
                    children.push(BlockChild::Statement(Box::new(statement)));
                }
                std::cmp::Ordering::Greater => {
                    let block = match Block::expect_with_extra_args(ast, false) {
                        Ok(block) => block,
                        Err(error) => match error {
                            ParserError::WrongForm => {
                                ast.index -= 1;
                                break;
                            }

                            _ => return Err(error),
                        },
                    };
                    children.push(BlockChild::Block(block));
                }
            }
            ast.index = ast.index.wrapping_add(1);
        }
        if ast.current_token_type() == eoi!() {
            ast.index -= 1;
        }
        if ast.index < first_token_index {
            ast.index = first_token_index;
            return Err(ParserError::WrongForm);
        }
        let number_of_tokens_used = ast.index - first_token_index + 1;

        let b = Block {
            position: ast.create_position(first_token_index, number_of_tokens_used),
            indentation_level: indentation_level_for_this_block,
            children,
            last_statement_is_expression,
        };
        match b.children.last() {
            None => (),
            Some(ref v) => {
                match v {
                    &BlockChild::Block(_) => (),
                    BlockChild::Statement(s) => {
                        if b.last_statement_is_expression {
                            match s.statement_type {
                                StatementType::Expr(_) => (),
                                _ => return Err(ast.create_parse_error_with_message(
                                    s.position.index,
                                    s.position.len,
                                    "Blocks that return expressions must end with an expression."
                                        .into(),
                                )),
                            }
                        }
                    }
                }
            }
        }

        Ok(b)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IfNodeType {
    Expression,
    Statement,
}

fn expect_if_node<'a>(ast: &mut Ast<'a>, type_: IfNodeType) -> ParserResult<'a, IfNode<'a>> {
    let last_statement_is_expression = type_ == IfNodeType::Expression;
    match ast.current_token_type() {
        if_!() => {
            let first_token_index = ast.index;
            ast.index += 1;
            ast.step_over_whitespace_and_block_comments();
            let condition = match Expression::expect(ast) {
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
            let then_block = match Block::expect_with_extra_args(ast, last_statement_is_expression)
            {
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
                let condition = match Expression::expect(ast) {
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
                let then_block =
                    match Block::expect_with_extra_args(ast, last_statement_is_expression) {
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
                elif_expressions.push(ElifNode {
                    position: ast
                        .create_position(first_token_index, ast.index - first_token_index + 1),
                    then_block,
                    condition,
                });
                last_valid_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_comments_and_terminators();
            }
            let else_expression = if ast.current_token_type() == else_!() {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_block_comments();
                let else_block =
                    match Block::expect_with_extra_args(ast, last_statement_is_expression) {
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
                Some(ElseNode {
                    position: ast
                        .create_position(first_token_index, ast.index - first_token_index + 1),
                    else_block,
                })
            } else {
                ast.index = last_valid_index;
                if last_statement_is_expression {
                    return Err(ast.create_parse_error_with_message(
                        ast.index,
                        1,
                        "If expressions must include an else case.".into(),
                    ));
                }
                None
            };
            Ok(IfNode {
                condition,
                then_block,
                elif_nodes: elif_expressions,
                else_nodes: else_expression,
                position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
            })
        }
        _ => Err(ParserError::WrongForm),
    }
}

impl<'a> ExpectSelf<'a, bool> for Statement<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
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
                            position: ast.create_position(
                                first_token_index,
                                ast.index - first_token_index + 1,
                            ),
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
                            position: ast.create_position(
                                first_token_index,
                                ast.index - first_token_index + 1,
                            ),
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
                match ast.current_token_type() {
                    terminator!() => {
                        return Ok(Statement {
                            statement_type: StatementType::Return(None),
                            position: ast.create_position(
                                first_token_index,
                                ast.index - first_token_index + 1,
                            ),
                        });
                    }
                    _ => match Expression::expect(ast) {
                        Ok(expression) => {
                            return Ok(Statement {
                                statement_type: StatementType::Return(Some(expression)),
                                position: ast.create_position(
                                    first_token_index,
                                    ast.index - first_token_index + 1,
                                ),
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
            if_!() => match expect_if_node(ast, IfNodeType::Statement) {
                Ok(i) => Ok(Statement {
                    statement_type: StatementType::IfStatement(Box::new(i)),
                    position: ast.create_position(
                        index_before_statement,
                        ast.index - index_before_statement + 1,
                    ),
                }),
                Err(e) => Err(e),
            },
            loop_!() => {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_comments();
                if ast.current_token().token_type != terminator!() {
                    return Err(ast.create_parse_error_with_message(
                        ast.index,
                        1,
                        "Expected a terminator after 'loop' keyowrd".into(),
                    ));
                }
                ast.index += 1;
                ast.step_over_whitespace_and_comments_and_terminators();
                let body = match Block::expect(ast) {
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
                let pos = ast.create_position(first_token_index, ast.index - first_token_index + 1);
                return Ok(Statement {
                    statement_type: StatementType::LoopStatement(LoopStatement {
                        position: pos,
                        body,
                    }),
                    position: pos,
                });
            }
            while_!() => {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_block_comments();
                let condition = match Expression::expect(ast) {
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
                let body = match Block::expect(ast) {
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
                    StatementType::WhileStatement(WhileStatement { condition, body });
                return Ok(Statement {
                    statement_type,
                    position: ast
                        .create_position(first_token_index, ast.index - first_token_index + 1),
                });
            }
            identifier!() => {
                let first_token_index = ast.index;
                let ident_id = ast
                    .program
                    .identifier_literals
                    .register_value(ast.current_token().contents);
                ast.index += 1;
                ast.step_over_whitespace_and_block_comments();
                match ast.current_token_type() {
                    assignment_operator_tt!(assign_tok) => {
                        ast.index += 1;
                        ast.step_over_whitespace_and_block_comments();
                        match Expression::expect(ast) {
                                Ok(expression) => {

                                    return Ok(Statement {
                                        statement_type: StatementType::Assignment(
                                            ident_id,
                                            assign_tok,
                                            expression,
                                        ),
                                        position: ast.create_position(first_token_index, ast.index - first_token_index + 1)
                                    });
                                }
                                Err(error) => match error {
                                    ParserError::WrongForm => {
                                        return Err(ast.create_parse_error_with_message(
                                            ast.index,
                                            1,
                                            "Expected a terminator or an expression after assignment operator.".to_string()
                                        ))
                                    },
                                    _ => Err(error),
                                },
                        }
                    }
                    _ => {
                        ast.index = index_before_statement;
                        let expression = match Expression::expect(ast) {
                            Ok(v) => v,
                            Err(e) => match e {
                                ParserError::WrongForm => {
                                    ast.index = first_token_index;
                                    return Err(e);
                                }
                                _ => return Err(e),
                            },
                        };

                        return Ok(Statement {
                            position: expression.position,
                            statement_type: StatementType::Expr(expression),
                        });
                    }
                }
            }
            _ => {
                ast.index = index_before_statement;
                let expression = Expression::expect(ast)?;
                return Ok(Statement {
                    position: expression.position,
                    statement_type: StatementType::Expr(expression),
                });
            }
        }
    }
}

impl<'a> ExpectSelf<'a> for Expression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let comp_expr = ComparisonExpression::expect(ast)?;
        ast.index += 1;
        let mut sub_expressions = vec![];
        if ast.current_token_type() == eoi!() {
            ast.index -= 1;

            return Ok(Expression {
                position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
                main_expression: comp_expr,
                sub_expressions,
            });
        }
        loop {
            let sub_expr = match SubExpression::expect(ast) {
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
                    position: ast
                        .create_position(first_token_index, ast.index - first_token_index + 1),
                    main_expression: comp_expr,
                    sub_expressions,
                });
            }
        }
        ast.index -= 1;

        Ok(Expression {
            position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
            sub_expressions,
            main_expression: comp_expr,
        })
    }
}

impl<'a> ExpectSelf<'a> for SubExpression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let keyword = match ast.current_token().token_type {
            and!() => BooleanComparisonKeywordTokenType::And,
            or!() => BooleanComparisonKeywordTokenType::Or,
            _ => return Err(ParserError::WrongForm),
        };
        ast.index += 1;
        ast.step_over_whitespace_and_block_comments();

        let expr = match ComparisonExpression::expect(ast) {
            Ok(v) => v,
            Err(e) => match e {
                ParserError::WrongForm => return Err(ast.create_parse_error_with_message(ast.index, 1, "the 'and' and 'or' keywords must be followed by a valid comparison expression.".into())),
                _ => return Err(e),
            }
        };

        Ok(SubExpression {
            keyword,
            expression: expr,
            position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
        })
    }
}

impl<'a> ExpectSelf<'a> for ComparisonExpression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let comparison_type = if ast.current_token().token_type == not!() {
            ast.index += 1;

            let inner =
                match ComparisonExpression::expect(ast) {
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
            ComparisonExpressionType::Not(NotKeywordTokenType::Not, Box::new(inner))
        } else {
            let first_token_index = ast.index;
            let arith = ArithmeticExpression::expect(ast)?;
            ast.index += 1;
            let mut sub_exprs = vec![];
            if ast.current_token_type() == eoi!() {
                ast.index -= 1;
                let position =
                    ast.create_position(first_token_index, ast.index - first_token_index + 1);
                return Ok(ComparisonExpression {
                    position,
                    comparison_type: ComparisonExpressionType::ComparisonChain(Box::new(
                        ComparisonChainExpression {
                            main_expression: arith,
                            sub_expressions: sub_exprs,
                            position,
                        },
                    )),
                });
            }
            loop {
                let sub_expr = match SubComparisonExpression::expect(ast) {
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
                    let position =
                        ast.create_position(first_token_index, ast.index - first_token_index + 1);
                    return Ok(ComparisonExpression {
                        position,
                        comparison_type: ComparisonExpressionType::ComparisonChain(Box::new(
                            ComparisonChainExpression {
                                main_expression: arith,
                                sub_expressions: sub_exprs,
                                position,
                            },
                        )),
                    });
                }
            }

            ComparisonExpressionType::ComparisonChain(Box::new(ComparisonChainExpression {
                position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
                main_expression: arith,
                sub_expressions: sub_exprs,
            }))
        };

        Ok(ComparisonExpression {
            comparison_type,
            position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
        })
    }
}

impl<'a> ExpectSelf<'a> for SubComparisonExpression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
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
        let expr = match ArithmeticExpression::expect(ast) {
            Ok(v) => v,
            Err(e) => match e {
                ParserError::WrongForm => return Err(ast.create_parse_error_with_message(ast.index, 1, "the '==', '!=', '<', '>', '<=', and '>=' operators must be followed by a valid arithmetic expression.".into())),
                _ => return Err(e),
            }
        };

        Ok(SubComparisonExpression {
            operator,
            expression: expr,
            position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
        })
    }
}

impl<'a> ExpectSelf<'a> for ArithmeticExpression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let term = TermExpression::expect(ast)?;
        ast.index += 1;
        let mut sub_exprs = vec![];
        if ast.current_token_type() == eoi!() {
            ast.index -= 1;
            let position =
                ast.create_position(first_token_index, ast.index - first_token_index + 1);

            return Ok(ArithmeticExpression {
                position,
                main_expression: term,
                sub_expressions: sub_exprs,
            });
        }
        loop {
            let sub_expr = match SubArithmeticExpression::expect(ast) {
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
                let position =
                    ast.create_position(first_token_index, ast.index - first_token_index + 1);

                return Ok(ArithmeticExpression {
                    position,
                    main_expression: term,
                    sub_expressions: sub_exprs,
                });
            }
        }

        Ok(ArithmeticExpression {
            position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
            main_expression: term,
            sub_expressions: sub_exprs,
        })
    }
}

impl<'a> ExpectSelf<'a> for SubArithmeticExpression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
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

        let expr = match TermExpression::expect(ast) {
            Ok(v) => v,
            Err(e) => match e {
                ParserError::WrongForm => return Err(ast.create_parse_error_with_message(ast.index, 1, "the '+', '-', '&', '|', and '^' operators must be followed by a valid term expression.".into())),
                _ => return Err(e),
            }
        };

        Ok(SubArithmeticExpression {
            operator,
            expression: expr,
            position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
        })
    }
}

impl<'a> ExpectSelf<'a> for TermExpression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let factor = FactorExpression::expect(ast)?;
        ast.index += 1;
        let mut sub_exprs = vec![];
        if ast.current_token_type() == eoi!() {
            ast.index -= 1;
            let position =
                ast.create_position(first_token_index, ast.index - first_token_index + 1);

            return Ok(TermExpression {
                position,
                main_expression: factor,
                sub_expressions: sub_exprs,
            });
        }
        loop {
            let sub_expr = match SubTermExpression::expect(ast) {
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
                let position =
                    ast.create_position(first_token_index, ast.index - first_token_index + 1);

                return Ok(TermExpression {
                    position,
                    main_expression: factor,
                    sub_expressions: sub_exprs,
                });
            }
        }

        Ok(TermExpression {
            main_expression: factor,
            sub_expressions: sub_exprs,
            position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
        })
    }
}

impl<'a> ExpectSelf<'a> for SubTermExpression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
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
        let expr = match FactorExpression::expect(ast) {
            Ok(v) => v,
            Err(e) => match e {
                ParserError::WrongForm => return Err(ast.create_parse_error_with_message(ast.index, 1, "the '*', '/', '%', '<<', and '>>' operators must be followed by a valid factor expression.".into())),
                _ => return Err(e),
            }
        };

        Ok(SubTermExpression {
            operator,
            expression: expr,
            position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
        })
    }
}

impl<'a> ExpectSelf<'a> for FactorExpression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        match ast.current_token().token_type {
            unary_operator_tt!(v) => {
                let first_token_index = ast.index;
                ast.index += 1;
                ast.step_over_whitespace_and_block_comments();
                let expr = match FactorExpression::expect(ast) {
                    Ok(v) => v,
                    Err(e) => match e {
                        ParserError::WrongForm => return Err(ast.create_parse_error_with_message(ast.index, 1, "the unary operators '~', '+' and '-' must be followed by a valid factor expression.".into())),
                        _ => return Err(e),
                    }
                };

                Ok(FactorExpression {
                    factor_type: FactorExpressionType::UnaryFactor(v, Box::new(expr)),
                    position: ast
                        .create_position(first_token_index, ast.index - first_token_index + 1),
                })
            }
            _ => {
                let exponent = ExponentExpression::expect(ast)?;

                Ok(FactorExpression {
                    factor_type: FactorExpressionType::Exponent(Box::new(exponent)),
                    position: ast
                        .create_position(first_token_index, ast.index - first_token_index + 1),
                })
            }
        }
    }
}

impl<'a> ExpectSelf<'a> for ExponentExpression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let call = CallExpression::expect(ast)?;
        ast.index += 1;
        let mut sub_exprs = vec![];
        if ast.current_token_type() == eoi!() {
            ast.index -= 1;
            let position =
                ast.create_position(first_token_index, ast.index - first_token_index + 1);

            return Ok(ExponentExpression {
                position,
                main_expression: call,
                sub_expressions: sub_exprs,
            });
        }
        loop {
            let sub_expr = match SubExponentExpression::expect(ast) {
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
                let position =
                    ast.create_position(first_token_index, ast.index - first_token_index + 1);

                return Ok(ExponentExpression {
                    position,
                    main_expression: call,
                    sub_expressions: sub_exprs,
                });
            }
        }
        Ok(ExponentExpression {
            main_expression: call,
            sub_expressions: sub_exprs,
            position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
        })
    }
}

impl<'a> ExpectSelf<'a> for SubExponentExpression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        if ast.current_token_type() != exponent!() {
            return Err(ParserError::WrongForm);
        }
        ast.index += 1;
        ast.step_over_whitespace_and_block_comments();
        let expr =
            match CallExpression::expect(ast) {
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
            position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
        })
    }
}

impl<'a> ExpectSelf<'a> for CallExpression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let atom = AtomicExpression::expect(ast)?;
        let index_after_atom = ast.index;
        ast.index += 1;
        ast.step_over_whitespace_and_block_comments();
        let mut params = None;
        if ast.current_token_type() == eoi!() {
            ast.index = index_after_atom;

            return Ok(CallExpression {
                atom,
                func_call_params: params,
                position: ast.create_position(first_token_index, ast.index - first_token_index + 1),
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
            match Expression::expect(ast) {
                Ok(v) => {
                    args.push(v);
                    ast.index += 1;
                    ast.step_over_whitespace_and_block_comments();
                    while ast.current_token().token_type == comma!() {
                        let next_expr = match Expression::expect(ast) {
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
        let position = ast.create_position(first_token_index, ast.index - first_token_index + 1);
        Ok(CallExpression {
            position,
            atom,
            func_call_params: params,
        })
    }
}

impl<'a> ExpectSelf<'a> for AtomicExpression<'a> {
    fn expect(ast: &mut Ast<'a>) -> ParserResult<'a, Self> {
        ast.step_over_whitespace_and_block_comments();
        let first_token_index = ast.index;
        let atom_type = match ast.current_token().token_type {
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
            identifier!() => {
                let idx = ast
                    .program
                    .identifier_literals
                    .register_value(ast.current_token().contents);

                AtomicExpressionType::Identifier(idx)
            }
            true_!() | false_!() => {
                AtomicExpressionType::Bool(ast.current_token().token_type == true_!())
            }
            l_paren!() => {
                let first_token_index = ast.index;
                ast.index += 1;
                let expr = match Expression::expect(ast) {
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
                    let ident = Identifier {
                        id: ast.current_token().contents,
                        position: ast.create_position(ast.index, 1),
                    };
                    ast.program.identifier_literals.register_value(ident.id);
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
                let body = match Block::expect(ast) {
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
                AtomicExpressionType::FuncDef(FunctionDefinitionExpression { args, body })
            }
            if_!() => {
                let if_node = expect_if_node(ast, IfNodeType::Expression)?;
                AtomicExpressionType::IfExpression(Box::new(if_node))
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
        let position = ast.create_position(first_token_index, ast.index - first_token_index + 1);

        let ret = AtomicExpression {
            atom_type,
            position,
        };
        Ok(ret)
    }
}
