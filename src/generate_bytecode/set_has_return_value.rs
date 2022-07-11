use crate::parser::ast::{
    ArithmeticExpression, AtomicExpression, AtomicExpressionType, Block, BlockChild,
    CallExpression, ComparisonChainExpression, ComparisonExpression, ComparisonExpressionType,
    ExponentExpression, Expression, FactorExpression, FactorExpressionType, Statement,
    StatementType, SubArithmeticExpression, SubComparisonExpression, SubExponentExpression,
    SubExpression, SubTermExpression, TermExpression,
};

pub(crate) trait SetHasReturnValue {
    fn set_has_return_value(&mut self, has_return_value: bool);
}

impl SetHasReturnValue for Block<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.last_statement_is_expression = has_return_value;
        for child in self.children.iter_mut() {
            match child {
                BlockChild::Statement(s) => s.set_has_return_value(has_return_value),
                BlockChild::Block(b) => b.set_has_return_value(has_return_value),
            }
        }
    }
}

impl SetHasReturnValue for Statement<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        match self.statement_type {
            StatementType::LoopStatement(ref mut l) => {
                l.body.set_has_return_value(has_return_value)
            }
            StatementType::WhileStatement(ref mut w) => {
                w.body.set_has_return_value(has_return_value)
            }
            StatementType::Expr(ref mut e) => e.set_has_return_value(has_return_value),
            _ => (),
        }
    }
}

impl SetHasReturnValue for Expression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        self.main_expression.set_has_return_value(has_return_value);
        for sub_expression in self.sub_expressions.iter_mut() {
            sub_expression.set_has_return_value(has_return_value);
        }
    }
}

impl SetHasReturnValue for SubExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        self.expression.set_has_return_value(has_return_value);
    }
}

impl SetHasReturnValue for ComparisonExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        match self.comparison_type {
            ComparisonExpressionType::Not(ref mut e) => e.set_has_return_value(has_return_value),
            ComparisonExpressionType::ComparisonChain(ref mut c) => {
                c.set_has_return_value(has_return_value)
            }
        }
    }
}

impl SetHasReturnValue for ComparisonChainExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        self.main_expression.set_has_return_value(has_return_value);
        for comparison in self.sub_expressions.iter_mut() {
            comparison.set_has_return_value(has_return_value);
        }
    }
}

impl SetHasReturnValue for SubComparisonExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        self.expression.set_has_return_value(has_return_value);
    }
}

impl SetHasReturnValue for ArithmeticExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        self.main_expression.set_has_return_value(has_return_value);
        for sub_expression in self.sub_expressions.iter_mut() {
            sub_expression.set_has_return_value(has_return_value);
        }
    }
}

impl SetHasReturnValue for SubArithmeticExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        self.expression.set_has_return_value(has_return_value);
    }
}

impl SetHasReturnValue for TermExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        self.main_expression.set_has_return_value(has_return_value);
        for sub_expression in self.sub_expressions.iter_mut() {
            sub_expression.set_has_return_value(has_return_value);
        }
    }
}

impl SetHasReturnValue for SubTermExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        self.expression.set_has_return_value(has_return_value);
    }
}

impl SetHasReturnValue for FactorExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        match self.factor_type {
            FactorExpressionType::Exponent(ref mut e) => e.set_has_return_value(has_return_value),
            FactorExpressionType::UnaryFactor(_u, ref mut e) => {
                e.set_has_return_value(has_return_value)
            }
        }
    }
}

impl SetHasReturnValue for ExponentExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        self.main_expression.set_has_return_value(has_return_value);
        for sub_expression in self.sub_expressions.iter_mut() {
            sub_expression.set_has_return_value(has_return_value);
        }
    }
}

impl SetHasReturnValue for SubExponentExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        self.expression.set_has_return_value(has_return_value);
    }
}

impl SetHasReturnValue for CallExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        if self.func_call_params.is_none() {
            self.atom.set_has_return_value(has_return_value)
        }
    }
}

impl SetHasReturnValue for AtomicExpression<'_> {
    fn set_has_return_value(&mut self, has_return_value: bool) {
        self.has_return_value = has_return_value;
        if let AtomicExpressionType::IfExpression(ref mut if_expression) = self.atom_type {
            if_expression.then_block.set_has_return_value(has_return_value);
            for elif_expression in if_expression.elif_expressions.iter_mut() {
                elif_expression.then_block.set_has_return_value(has_return_value);
            }
            if let Some(ref mut v) = if_expression.else_expression {
                v.else_block.set_has_return_value(has_return_value);
            }
        }
    }
}
