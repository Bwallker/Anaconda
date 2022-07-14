use crate::parser::ast::{
    ArithmeticExpression, AtomicExpression, AtomicExpressionType, Block, BlockChild,
    CallExpression, ComparisonChainExpression, ComparisonExpression, ComparisonExpressionType,
    Context, ExponentExpression, Expression, FactorExpression, FactorExpressionType, Statement,
    StatementType, SubArithmeticExpression, SubComparisonExpression, SubExponentExpression,
    SubExpression, SubTermExpression, TermExpression,
};

pub(crate) trait SetContext {
    fn set_ctx(&mut self, ctx: Context);
}

impl SetContext for Block<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        for child in self.children.iter_mut() {
            match child {
                BlockChild::Statement(s) => s.set_ctx(ctx),
                BlockChild::Block(b) => b.set_ctx(ctx),
            }
        }
    }
}

impl SetContext for Statement<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        match self.statement_type {
            StatementType::LoopStatement(ref mut l) => l.body.set_ctx(ctx),
            StatementType::WhileStatement(ref mut w) => w.body.set_ctx(ctx),
            StatementType::Expr(ref mut e) => e.set_ctx(ctx),
            _ => (),
        }
    }
}

impl SetContext for Expression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        self.main_expression.set_ctx(ctx);
        for sub_expression in self.sub_expressions.iter_mut() {
            sub_expression.set_ctx(ctx);
        }
    }
}

impl SetContext for SubExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        self.expression.set_ctx(ctx);
    }
}

impl SetContext for ComparisonExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        match self.comparison_type {
            ComparisonExpressionType::Not(ref mut e) => e.set_ctx(ctx),
            ComparisonExpressionType::ComparisonChain(ref mut c) => c.set_ctx(ctx),
        }
    }
}

impl SetContext for ComparisonChainExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        self.main_expression.set_ctx(ctx);
        for comparison in self.sub_expressions.iter_mut() {
            comparison.set_ctx(ctx);
        }
    }
}

impl SetContext for SubComparisonExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        self.expression.set_ctx(ctx);
    }
}

impl SetContext for ArithmeticExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        self.main_expression.set_ctx(ctx);
        for sub_expression in self.sub_expressions.iter_mut() {
            sub_expression.set_ctx(ctx);
        }
    }
}

impl SetContext for SubArithmeticExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        self.expression.set_ctx(ctx);
    }
}

impl SetContext for TermExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        self.main_expression.set_ctx(ctx);
        for sub_expression in self.sub_expressions.iter_mut() {
            sub_expression.set_ctx(ctx);
        }
    }
}

impl SetContext for SubTermExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        self.expression.set_ctx(ctx);
    }
}

impl SetContext for FactorExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        match self.factor_type {
            FactorExpressionType::Exponent(ref mut e) => e.set_ctx(ctx),
            FactorExpressionType::UnaryFactor(_u, ref mut e) => e.set_ctx(ctx),
        }
    }
}

impl SetContext for ExponentExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        self.main_expression.set_ctx(ctx);
        for sub_expression in self.sub_expressions.iter_mut() {
            sub_expression.set_ctx(ctx);
        }
    }
}

impl SetContext for SubExponentExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        self.expression.set_ctx(ctx);
    }
}

impl SetContext for CallExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        if self.func_call_params.is_none() {
            self.atom.set_ctx(ctx)
        }
    }
}

impl SetContext for AtomicExpression<'_> {
    fn set_ctx(&mut self, ctx: Context) {
        self.ctx = ctx;
        if let AtomicExpressionType::IfExpression(ref mut if_expression) = self.atom_type {
            if_expression.then_block.set_ctx(ctx);
            for elif_expression in if_expression.elif_expressions.iter_mut() {
                elif_expression.then_block.set_ctx(ctx);
            }
            if let Some(ref mut v) = if_expression.else_expression {
                v.else_block.set_ctx(ctx);
            }
        }
    }
}
