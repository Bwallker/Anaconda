use super::ast::{
    Ast, AtomicExpressionType, Block, BlockChild, ComparisonExpressionType, Expression,
    FactorExpressionType, IfNode, ParserResult, Statement, StatementType,
};

pub(crate) fn validate_ast<'a>(ast: &mut Ast<'a>) -> ParserResult<'a, ()> {
    if let Some(ref mut v) = ast.program.base_block {
        v.validate()?
    }
    Ok(())
}

trait ValidateSelf<'a> {
    fn validate(&mut self) -> ParserResult<'a, ()>;
}

impl<'a> ValidateSelf<'a> for Block<'a> {
    fn validate(&mut self) -> ParserResult<'a, ()> {
        /*for _child in &mut self.children.iter() {

        }*/

        Ok(())
    }
}

fn get_if_statement<'expr, 'a>(expr: &'expr mut Expression<'a>) -> Option<&'expr mut IfNode<'a>> {
    match expr.main_expression.comparison_type {
        ComparisonExpressionType::ComparisonChain(ref mut c) => match c
            .main_expression
            .main_expression
            .main_expression
            .factor_type
        {
            FactorExpressionType::Exponent(ref mut e) => match e.main_expression.atom.atom_type {
                AtomicExpressionType::IfExpression(ref mut i) => Some(i),
                AtomicExpressionType::ExpressionWithParentheses(ref mut e) => get_if_statement(e),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}
fn get_block_if_expressions<'a, 'block>(
    block: &'block mut Block<'a>,
    res: &mut Vec<&'block mut IfNode<'a>>,
) {
    for child in block.children.iter_mut() {
        match child {
            BlockChild::Block(ref mut b) => get_block_if_expressions(b, res),
            BlockChild::Statement(ref mut s) => get_if_expressions_of_statement(s, res),
        }
    }
}
fn get_if_expressions_of_statement<'a, 'statement>(
    statement: &'statement mut Statement<'a>,
    res: &mut Vec<&'statement mut IfNode<'a>>,
) {
    match statement.statement_type {
        StatementType::Assignment(ref _ident, ref _op, ref mut expr) => {
            get_if_expressions_of_expr(expr, res)
        }

        StatementType::Return(ref mut expr) => {
            expr.as_mut().map(|e| get_if_expressions_of_expr(e, res)).unwrap_or(())
        }

        StatementType::LoopStatement(ref mut l) => get_block_if_expressions(&mut l.body, res),

        StatementType::WhileStatement(ref mut w) => {
            get_if_expressions_of_expr(&mut w.condition, res);
            get_block_if_expressions(&mut w.body, res);
        }
        _ => (),
    }
}
fn get_if_expressions<'a, 'block>(block: &'block mut Block<'a>) -> Vec<&'block mut IfNode<'a>> {
    let mut res = vec![];
    get_block_if_expressions(block, &mut res);
    res
}

fn get_if_expressions_of_expr<'a, 'expr>(
    expr: &'expr mut Expression<'a>,
    so_far: &mut Vec<&'expr mut IfNode<'a>>,
) {
    for sub_expr in expr.sub_expressions.iter() {
        sub_expr.
    }
}
