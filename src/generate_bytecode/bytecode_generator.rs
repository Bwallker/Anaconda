use crate::runtime::{
    bytecode::{Bytecode, Function, OpCodes, USIZE_BYTES},
    gc::{GarbageCollector, GcValue},
};
use crate::{
    lexer::lex::{
        ArithmeticOperatorTokenType, AssignmentOperatorTokenType,
        BooleanComparisonKeywordTokenType, ComparisonOperatorTokenType, TermOperatorTokenType,
        UnaryOperatorTokenType,
    },
    parser::ast::{
        ArithmeticExpression, Ast, AtomicExpression, AtomicExpressionType, Block, BlockChild,
        CallExpression, ComparisonChainExpression, ComparisonExpression, ComparisonExpressionType,
        ExponentExpression, Expression, FactorExpression, FactorExpressionType, Int, Statement,
        StatementType, SubArithmeticExpression, SubComparisonExpression, SubExponentExpression,
        SubExpression, SubTermExpression, TermExpression,
    },
    util::RemoveLastTrait,
};
pub(crate) trait GenerateBytecode {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector);
}

impl<'a> GenerateBytecode for Block<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        bytecode.push_opcode(OpCodes::BeginBlock);
        let idx_before_children = ast.index;
        for child in self.children.iter().remove_last() {
            child.gen_bytecode(bytecode, ast, gc);
        }
        match self.children.last() {
            None => (),
            Some(v) => match v {
                BlockChild::Block(b) => b.gen_bytecode(bytecode, ast, gc),
                BlockChild::Statement(s) => {
                    if self.last_statement_is_expression {
                        match s.statement_type {
                            StatementType::Expr(ref e) => e.gen_bytecode(bytecode, ast, gc),
                            _ => unreachable!(),
                        }
                    } else {
                        s.gen_bytecode(bytecode, ast, gc)
                    }
                }
            },
        }
        let idx_after_children = ast.index;
        // If our children generated no instructions, we don't need to create a block.
        if idx_before_children == idx_after_children {
            bytecode.instructions.pop();
        } else {
            bytecode.push_opcode(OpCodes::EndBlock);
        }
    }
}

impl<'a> GenerateBytecode for BlockChild<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        match self {
            BlockChild::Statement(statement) => statement.gen_bytecode(bytecode, ast, gc),
            BlockChild::Block(block) => block.gen_bytecode(bytecode, ast, gc),
        }
    }
}

impl<'a> GenerateBytecode for Statement<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        self.statement_type.gen_bytecode(bytecode, ast, gc);
    }
}

impl<'a> GenerateBytecode for StatementType<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        match self {
            Self::Expr(e) => {
                e.gen_bytecode(bytecode, ast, gc);
            }
            Self::Return(ret_val) => {
                if let Some(v) = ret_val {
                    v.gen_bytecode(bytecode, ast, gc);
                } else {
                    bytecode.push_opcode(OpCodes::LoadPoison);
                }
                bytecode.push_opcode(OpCodes::EndBlock);
                bytecode.push_opcode(OpCodes::Return);
            }
            Self::Break => {
                bytecode.push_opcode(OpCodes::Break);
            }
            Self::Continue => {
                bytecode.push_opcode(OpCodes::Continue);
            }
            Self::Assignment(i, a, e) => {
                let operator_opcode = match a {
                    AssignmentOperatorTokenType::Assign => OpCodes::Assign,
                    AssignmentOperatorTokenType::BitshiftLeftAssign => {
                        OpCodes::BitshiftLeftAndAssign
                    }

                    AssignmentOperatorTokenType::BitshiftRightAssign => {
                        OpCodes::BitshiftRightAndAssign
                    }

                    AssignmentOperatorTokenType::BitwiseAndAssign => OpCodes::BitwiseAndAndAssign,
                    AssignmentOperatorTokenType::BitwiseOrAssign => OpCodes::BitwiseOrAndAssign,
                    AssignmentOperatorTokenType::BitwiseXorAssign => OpCodes::BitwiseXorAndAssign,
                    AssignmentOperatorTokenType::MinusAssign => OpCodes::SubAndAssign,
                    AssignmentOperatorTokenType::PlusAssign => OpCodes::AddAndAssign,
                    AssignmentOperatorTokenType::StarAssign => OpCodes::MultiplyAndAssign,
                    AssignmentOperatorTokenType::SlashAssign => OpCodes::DivideAndAssign,
                    AssignmentOperatorTokenType::PercentAssign => OpCodes::ModuloAndAssign,
                    AssignmentOperatorTokenType::ExponentAssign => OpCodes::ExponentAndAssign,
                };
                e.gen_bytecode(bytecode, ast, gc);
                bytecode.push_opcode(operator_opcode);
                bytecode.push_usize(*i);
            }
            StatementType::LoopStatement(l) => {
                bytecode.push_opcode(OpCodes::StartOfLoop);
                let addr_of_loop_start = bytecode.instructions.len();
                bytecode.push_usize(0);
                l.body.gen_bytecode(bytecode, ast, gc);
                bytecode.push_opcode(OpCodes::Continue);
                let end_addr = bytecode.instructions.len();
                bytecode.set_usize(addr_of_loop_start, end_addr);
            }
            StatementType::WhileStatement(w) => {
                bytecode.push_opcode(OpCodes::StartOfLoop);
                let addr_of_loop_start = bytecode.instructions.len();
                bytecode.push_usize(0);
                w.condition.gen_bytecode(bytecode, ast, gc);
                bytecode.push_opcode(OpCodes::BreakIfFalse);
                w.body.gen_bytecode(bytecode, ast, gc);
                bytecode.push_opcode(OpCodes::Continue);
                let end_addr = bytecode.instructions.len();
                bytecode.set_usize(addr_of_loop_start, end_addr);
            }
        }
    }
}

impl<'a> GenerateBytecode for Expression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        self.main_expression.gen_bytecode(bytecode, ast, gc);
        for sub_expr in self.sub_expressions.iter() {
            sub_expr.gen_bytecode(bytecode, ast, gc)
        }
    }
}

impl<'a> GenerateBytecode for SubExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        self.expression.gen_bytecode(bytecode, ast, gc);
        if self.has_return_value {
            let opcode = match self.keyword {
                BooleanComparisonKeywordTokenType::And => OpCodes::BooleanAnd,
                BooleanComparisonKeywordTokenType::Or => OpCodes::BooleanOr,
            };
            bytecode.push_opcode(opcode);
        }
    }
}

impl<'a> GenerateBytecode for ComparisonExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        match self.comparison_type {
            ComparisonExpressionType::ComparisonChain(ref chain) => {
                chain.gen_bytecode(bytecode, ast, gc)
            }
            ComparisonExpressionType::Not(ref e) => {
                e.gen_bytecode(bytecode, ast, gc);
                if self.has_return_value {
                    bytecode.push_opcode(OpCodes::BooleanNot);
                }
            }
        }
    }
}

impl<'a> GenerateBytecode for ComparisonChainExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        self.main_expression.gen_bytecode(bytecode, ast, gc);
        for sub_expr in self.sub_expressions.iter() {
            sub_expr.gen_bytecode(bytecode, ast, gc)
        }
    }
}

impl<'a> GenerateBytecode for SubComparisonExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        self.expression.gen_bytecode(bytecode, ast, gc);
        if self.has_return_value {
            let comparator = match self.operator {
                ComparisonOperatorTokenType::Equals => OpCodes::Equals,
                ComparisonOperatorTokenType::NotEquals => OpCodes::NotEquals,
                ComparisonOperatorTokenType::LessThan => OpCodes::LessThan,
                ComparisonOperatorTokenType::GreaterThan => OpCodes::GreaterThan,
                ComparisonOperatorTokenType::LessThanEquals => OpCodes::LessThanEquals,
                ComparisonOperatorTokenType::GreaterThanEquals => OpCodes::GreaterThanEquals,
            };
            bytecode.push_opcode(comparator);
        }
    }
}

impl<'a> GenerateBytecode for ArithmeticExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        self.main_expression.gen_bytecode(bytecode, ast, gc);
        for sub_expr in self.sub_expressions.iter() {
            sub_expr.gen_bytecode(bytecode, ast, gc)
        }
    }
}

impl<'a> GenerateBytecode for SubArithmeticExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        self.expression.gen_bytecode(bytecode, ast, gc);
        if self.has_return_value {
            let arith_op = match self.operator {
                ArithmeticOperatorTokenType::Unary(u) => match u {
                    UnaryOperatorTokenType::Minus => OpCodes::Sub,
                    UnaryOperatorTokenType::Plus => OpCodes::Add,
                    UnaryOperatorTokenType::BitwiseNot => unreachable!(),
                },
                ArithmeticOperatorTokenType::BitwiseAnd => OpCodes::BitwiseAnd,
                ArithmeticOperatorTokenType::BitwiseOr => OpCodes::BitwiseOr,
                ArithmeticOperatorTokenType::BitwiseXor => OpCodes::BitwiseXor,
            };
            bytecode.push_opcode(arith_op);
        }
    }
}

impl<'a> GenerateBytecode for TermExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        self.main_expression.gen_bytecode(bytecode, ast, gc);
        for sub_expr in self.sub_expressions.iter() {
            sub_expr.gen_bytecode(bytecode, ast, gc)
        }
    }
}

impl<'a> GenerateBytecode for SubTermExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        self.expression.gen_bytecode(bytecode, ast, gc);
        if self.has_return_value {
            bytecode.push_opcode(match self.operator {
                TermOperatorTokenType::BitshiftLeft => OpCodes::BitshiftLeft,
                TermOperatorTokenType::BitshiftRight => OpCodes::BitshiftRight,
                TermOperatorTokenType::Slash => OpCodes::Divide,
                TermOperatorTokenType::Star => OpCodes::Multiply,
                TermOperatorTokenType::Percent => OpCodes::Modulo,
            })
        }
    }
}

impl<'a> GenerateBytecode for FactorExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        match self.factor_type {
            FactorExpressionType::Exponent(ref exponent) => {
                exponent.gen_bytecode(bytecode, ast, gc);
            }
            FactorExpressionType::UnaryFactor(u, ref f) => {
                f.gen_bytecode(bytecode, ast, gc);
                if self.has_return_value {
                    bytecode.push_opcode(match u {
                        UnaryOperatorTokenType::BitwiseNot => OpCodes::BitwiseNot,
                        UnaryOperatorTokenType::Plus => OpCodes::UnaryPlus,
                        UnaryOperatorTokenType::Minus => OpCodes::UnaryMinus,
                    })
                }
            }
        }
    }
}

impl<'a> GenerateBytecode for ExponentExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        self.main_expression.gen_bytecode(bytecode, ast, gc);
        for sub_expr in self.sub_expressions.iter() {
            sub_expr.gen_bytecode(bytecode, ast, gc)
        }
    }
}

impl<'a> GenerateBytecode for SubExponentExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        self.expression.gen_bytecode(bytecode, ast, gc);
        if self.has_return_value {
            bytecode.push_opcode(OpCodes::Exponent)
        }
    }
}

impl<'a> GenerateBytecode for CallExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        if let Some(ref func_calls_params) = self.func_call_params {
            self.atom.gen_bytecode(bytecode, ast, gc);
            for params in func_calls_params.iter() {
                for param in params.iter() {
                    param.gen_bytecode(bytecode, ast, gc)
                }
                bytecode.push_opcode(OpCodes::CallFunction);
                bytecode.push_usize(params.len());
            }
            if !self.has_return_value {
                bytecode.push_opcode(OpCodes::Pop);
            }
        } else {
            self.atom.gen_bytecode(bytecode, ast, gc);
        }
    }
}

impl<'a> GenerateBytecode for AtomicExpression<'a> {
    fn gen_bytecode(&self, bytecode: &mut Bytecode, ast: &mut Ast<'_>, gc: &mut GarbageCollector) {
        match self.atom_type {
            AtomicExpressionType::Int(ref i) => match i {
                Int::Small(value) => {
                    if self.has_return_value {
                        bytecode.push_opcode(OpCodes::LoadSmallIntLiteral);
                        bytecode.push_usize(*value);
                    }
                }
                Int::Big(index) => {
                    if self.has_return_value {
                        bytecode.push_opcode(OpCodes::LoadBigIntLiteral);
                        bytecode.push_usize(*index);
                    }
                }
            },
            AtomicExpressionType::String(ref index) => {
                if self.has_return_value {
                    bytecode.push_opcode(OpCodes::LoadStringLiteral);
                    bytecode.push_usize(*index);
                }
            }
            AtomicExpressionType::Identifier(ref index) => {
                if self.has_return_value {
                    bytecode.push_opcode(OpCodes::LoadVariableValueFromIndex);
                    bytecode.push_usize(*index);
                }
            }
            AtomicExpressionType::Bool(b) => {
                if self.has_return_value {
                    if b {
                        bytecode.push_opcode(OpCodes::LoadTrue)
                    } else {
                        bytecode.push_opcode(OpCodes::LoadFalse);
                    }
                }
            }
            AtomicExpressionType::ExpressionWithParentheses(ref e) => {
                e.gen_bytecode(bytecode, ast, gc)
            }
            AtomicExpressionType::FuncDef(ref f) => {
                let function = Function {
                    params: f
                        .args
                        .iter()
                        .map(|arg| {
                            *ast.program
                                .identifier_literals
                                .reverse_data
                                .get(arg)
                                .unwrap()
                        })
                        .collect(),
                    extra_args: false,
                    start_index: bytecode.instructions.len() + USIZE_BYTES + 1,
                };
                let idx = ast
                    .program
                    .function_definitions
                    .register_value(GcValue::new(function, gc));
                bytecode.push_opcode(OpCodes::StartOfFunctionDefinition);
                bytecode.push_usize(idx);
                f.body.gen_bytecode(bytecode, ast, gc);
                bytecode.push_opcode(OpCodes::LoadPoison);
                bytecode.push_opcode(OpCodes::Return);
                bytecode.push_opcode(OpCodes::EndOfFunctionDefinition);
            }
            AtomicExpressionType::IfExpression(ref if_node) => {
                if_node.condition.gen_bytecode(bytecode, ast, gc);

                bytecode.push_opcode(OpCodes::IfFalseGoto);
                let addr_of_if_goto = bytecode.instructions.len();
                bytecode.push_usize(0);
                if_node.then_block.gen_bytecode(bytecode, ast, gc);

                bytecode.push_opcode(OpCodes::Goto);
                let addr_of_body_goto = bytecode.instructions.len();
                bytecode.push_usize(0);

                let first_elif_addr = bytecode.instructions.len();
                let mut elif_exprs_end_addresses = Vec::with_capacity(if_node.elif_nodes.len());
                for elif_expr in if_node.elif_nodes.iter() {
                    elif_expr.condition.gen_bytecode(bytecode, ast, gc);
                    bytecode.push_opcode(OpCodes::IfFalseGoto);
                    let addr_of_elif_goto = bytecode.instructions.len();
                    bytecode.push_usize(0);
                    elif_expr.then_block.gen_bytecode(bytecode, ast, gc);
                    bytecode.push_opcode(OpCodes::Goto);
                    let addr_of_body_goto = bytecode.instructions.len();
                    bytecode.push_usize(0);
                    elif_exprs_end_addresses.push(addr_of_body_goto);
                    bytecode.set_usize(addr_of_elif_goto, bytecode.instructions.len());
                }
                let addr_of_else_goto = bytecode.instructions.len();
                if let Some(ref v) = if_node.else_nodes {
                    v.else_block.gen_bytecode(bytecode, ast, gc);
                }
                let first_thing_after = bytecode.instructions.len();
                bytecode.set_usize(
                    addr_of_if_goto,
                    if if_node.elif_nodes.is_empty() {
                        if if_node.else_nodes.is_none() {
                            first_thing_after
                        } else {
                            addr_of_else_goto
                        }
                    } else {
                        first_elif_addr
                    },
                );

                for addr in elif_exprs_end_addresses {
                    bytecode.set_usize(addr, first_thing_after);
                }
                bytecode.set_usize(addr_of_body_goto, first_thing_after);
            }

            ref unknown => {
                println!("{:#?}", unknown);
                todo!()
            }
        }
    }
}
