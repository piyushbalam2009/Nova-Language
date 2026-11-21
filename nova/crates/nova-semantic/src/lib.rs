//! Nova Semantic Analyzer & Static Type Checker.

use nova_lexer::Span;
use nova_parser::{BinOp, Expr, Mutability, Program, Stmt, TypeAnn, UnaryOp};
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Float,
    Bool,
    Char,
    String,
    Void,
    None,
    Any,
    Struct(String),
    Array(Box<Type>),
    Tensor(Box<Type>),
    Fn(Vec<Type>, Box<Type>),
}

impl Type {
    fn is_numeric(&self) -> bool {
        matches!(self, Type::Int | Type::Float | Type::Any)
    }

    fn is_tensor(&self) -> bool {
        matches!(self, Type::Tensor(_))
    }

    fn compatible_with(&self, actual: &Type) -> bool {
        if *self == Type::Any || *actual == Type::Any || *self == *actual {
            return true;
        }
        // Int widens cleanly to Float when a Float parameter/binding expects numeric precision.
        if *self == Type::Float && *actual == Type::Int {
            return true;
        }
        match (self, actual) {
            (Type::Array(e1), Type::Array(e2)) => e1.compatible_with(e2),
            (Type::Tensor(e1), Type::Tensor(e2)) => e1.compatible_with(e2),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticError {
    pub message: String,
    pub span: Span,
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "type error at {}:{}: {}",
            self.span.line, self.span.column, self.message
        )
    }
}

#[derive(Debug, Clone)]
struct Binding {
    ty: Type,
    mutable: bool,
    min_args: Option<usize>,
    max_args: Option<usize>,
}

pub fn check_program(program: &Program) -> Result<(), Vec<SemanticError>> {
    let mut checker = TypeChecker::new();
    checker.check(program);
    if checker.errors.is_empty() {
        Ok(())
    } else {
        Err(checker.errors)
    }
}

struct TypeChecker {
    scopes: Vec<HashMap<String, Binding>>,
    structs: HashSet<String>,
    errors: Vec<SemanticError>,
    current_ret: Option<Type>,
    loop_depth: usize,
}

impl TypeChecker {
    fn new() -> Self {
        let mut globals = HashMap::new();
        let mut reg = |name: &str, params: Vec<Type>, ret: Type, min: usize, max: Option<usize>| {
            globals.insert(
                name.to_string(),
                Binding {
                    ty: Type::Fn(params, Box::new(ret)),
                    mutable: false,
                    min_args: Some(min),
                    max_args: max,
                },
            );
        };

        let tensor_f = Type::Tensor(Box::new(Type::Float));

        // Core I/O & conversion stdlib
        reg("print", vec![Type::Any], Type::Void, 0, None);
        reg("str", vec![Type::Any], Type::String, 1, Some(1));
        reg("int", vec![Type::Any], Type::Int, 1, Some(1));
        reg("float", vec![Type::Any], Type::Float, 1, Some(1));
        reg("len", vec![Type::Any], Type::Int, 1, Some(1));
        reg("type_of", vec![Type::Any], Type::String, 1, Some(1));

        // Math stdlib
        reg("abs", vec![Type::Float], Type::Float, 1, Some(1));
        reg("sqrt", vec![Type::Float], Type::Float, 1, Some(1));
        reg("pow", vec![Type::Float, Type::Float], Type::Float, 2, Some(2));
        reg("min", vec![Type::Float, Type::Float], Type::Float, 2, Some(2));
        reg("max", vec![Type::Float, Type::Float], Type::Float, 2, Some(2));
        reg("floor", vec![Type::Float], Type::Int, 1, Some(1));
        reg("ceil", vec![Type::Float], Type::Int, 1, Some(1));
        reg("sin", vec![Type::Float], Type::Float, 1, Some(1));
        reg("cos", vec![Type::Float], Type::Float, 1, Some(1));
        reg("exp", vec![Type::Float], Type::Float, 1, Some(1));
        reg("log", vec![Type::Float], Type::Float, 1, Some(1));

        // String & array stdlib
        reg("upper", vec![Type::String], Type::String, 1, Some(1));
        reg("lower", vec![Type::String], Type::String, 1, Some(1));
        reg("trim", vec![Type::String], Type::String, 1, Some(1));
        reg("contains", vec![Type::String, Type::String], Type::Bool, 2, Some(2));
        reg("range", vec![Type::Int, Type::Int], Type::Array(Box::new(Type::Int)), 2, Some(2));
        reg("sum", vec![Type::Array(Box::new(Type::Any))], Type::Float, 1, Some(1));

        // Native AI Engine builtins (tensors, autograd, layers, optimizers, training)
        reg("tensor", vec![Type::Any], tensor_f.clone(), 1, Some(1));
        reg("zeros", vec![Type::Any], tensor_f.clone(), 1, Some(1));
        reg("ones", vec![Type::Any], tensor_f.clone(), 1, Some(1));
        reg("full", vec![Type::Any, Type::Float], tensor_f.clone(), 2, Some(2));
        reg("randn", vec![Type::Any, Type::Int], tensor_f.clone(), 1, Some(2));
        reg("matmul", vec![tensor_f.clone(), tensor_f.clone()], tensor_f.clone(), 2, Some(2));
        reg("dot", vec![tensor_f.clone(), tensor_f.clone()], Type::Float, 2, Some(2));
        reg("transpose", vec![tensor_f.clone()], tensor_f.clone(), 1, Some(1));
        reg("reshape", vec![tensor_f.clone(), Type::Any], tensor_f.clone(), 2, Some(2));
        reg("relu", vec![tensor_f.clone()], tensor_f.clone(), 1, Some(1));
        reg("sigmoid", vec![tensor_f.clone()], tensor_f.clone(), 1, Some(1));
        reg("tanh", vec![tensor_f.clone()], tensor_f.clone(), 1, Some(1));
        reg("softmax", vec![tensor_f.clone()], tensor_f.clone(), 1, Some(1));
        reg("mse_loss", vec![tensor_f.clone(), tensor_f.clone()], Type::Float, 2, Some(2));
        reg("bce_loss", vec![tensor_f.clone(), tensor_f.clone()], Type::Float, 2, Some(2));
        reg("grad_var", vec![Type::Any], Type::Any, 1, Some(1));
        reg("grad_add", vec![Type::Any, Type::Any], Type::Any, 2, Some(2));
        reg("grad_sub", vec![Type::Any, Type::Any], Type::Any, 2, Some(2));
        reg("grad_mul", vec![Type::Any, Type::Any], Type::Any, 2, Some(2));
        reg("grad_matmul", vec![Type::Any, Type::Any], Type::Any, 2, Some(2));
        reg("grad_relu", vec![Type::Any], Type::Any, 1, Some(1));
        reg("grad_sigmoid", vec![Type::Any], Type::Any, 1, Some(1));
        reg("grad_tanh", vec![Type::Any], Type::Any, 1, Some(1));
        reg("grad_mse", vec![Type::Any, Type::Any], Type::Any, 2, Some(2));
        reg("backward", vec![Type::Any], Type::Void, 1, Some(1));
        reg("grad_of", vec![Type::Any], tensor_f.clone(), 1, Some(1));
        reg("value_of", vec![Type::Any], tensor_f.clone(), 1, Some(1));
        reg("zero_grad", vec![Type::Any], Type::Void, 1, Some(1));
        reg("model_new", vec![], Type::Any, 0, Some(0));
        reg("model_add_linear", vec![Type::Any, Type::Int, Type::Int, Type::Int], Type::Any, 3, Some(4));
        reg("model_add_activation", vec![Type::Any, Type::String], Type::Any, 2, Some(2));
        reg("model_forward", vec![Type::Any, Type::Any], tensor_f, 2, Some(2));
        reg("model_param_count", vec![Type::Any], Type::Int, 1, Some(1));
        reg("optimizer_sgd", vec![Type::Float, Type::Float], Type::Any, 1, Some(2));
        reg("optimizer_adam", vec![Type::Float], Type::Any, 1, Some(1));
        reg("train_step", vec![Type::Any, Type::Any, Type::Any, Type::Any, Type::String], Type::Float, 4, Some(5));
        reg("train_model", vec![Type::Any, Type::Any, Type::Any, Type::Any, Type::Int, Type::String], Type::Any, 5, Some(6));
        reg("train_xor", vec![Type::Int, Type::Float], Type::Any, 2, Some(2));
        reg("autograd_demo", vec![Type::Float, Type::Float], Type::Any, 2, Some(2));

        Self {
            scopes: vec![globals],
            structs: HashSet::new(),
            errors: Vec::new(),
            current_ret: None,
            loop_depth: 0,
        }
    }

    fn resolve_ann(&self, ann: &TypeAnn) -> Type {
        match ann {
            TypeAnn::Simple(s) => match s.as_str() {
                "Int" => Type::Int,
                "Float" => Type::Float,
                "Bool" => Type::Bool,
                "Char" => Type::Char,
                "String" => Type::String,
                "Void" => Type::Void,
                "None" => Type::None,
                "Tensor" => Type::Tensor(Box::new(Type::Float)),
                other if self.structs.contains(other) => Type::Struct(other.to_string()),
                _ => Type::Any,
            },
            TypeAnn::Generic(name, args) => {
                let inner = args
                    .first()
                    .map(|a| self.resolve_ann(a))
                    .unwrap_or(Type::Any);
                match name.as_str() {
                    "Array" => Type::Array(Box::new(inner)),
                    "Tensor" => Type::Tensor(Box::new(inner)),
                    _ => Type::Any,
                }
            }
        }
    }

    fn lookup(&self, name: &str) -> Option<&Binding> {
        for scope in self.scopes.iter().rev() {
            if let Some(b) = scope.get(name) {
                return Some(b);
            }
        }
        None
    }

    fn block_definitely_returns(stmts: &[Stmt]) -> bool {
        for stmt in stmts {
            match stmt {
                Stmt::Return { .. } => return true,
                Stmt::If {
                    then_body,
                    else_body: Some(eb),
                    ..
                } => {
                    if Self::block_definitely_returns(then_body)
                        && Self::block_definitely_returns(eb)
                    {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    fn check(&mut self, program: &Program) {
        // Pass 1: Register struct types
        for item in &program.items {
            if let Stmt::Struct { name, span, .. } = item {
                if !self.structs.insert(name.clone()) {
                    self.errors.push(SemanticError {
                        message: format!("Duplicate struct declaration '{name}'"),
                        span: *span,
                    });
                }
            }
        }

        // Pass 2: Register user function signatures (allowing shadowing of builtins, rejecting duplicates)
        let mut user_fns: HashSet<String> = HashSet::new();
        for item in &program.items {
            if let Stmt::Fn {
                name,
                params,
                ret_type,
                span,
                ..
            } = item
            {
                if !user_fns.insert(name.clone()) {
                    self.errors.push(SemanticError {
                        message: format!("Duplicate function declaration '{name}'"),
                        span: *span,
                    });
                    continue;
                }
                let p_types: Vec<Type> = params.iter().map(|(_, t)| self.resolve_ann(t)).collect();
                let arity = p_types.len();
                let r_type = ret_type
                    .as_ref()
                    .map(|t| self.resolve_ann(t))
                    .unwrap_or(Type::Void);
                if let Some(global) = self.scopes.first_mut() {
                    global.insert(
                        name.clone(),
                        Binding {
                            ty: Type::Fn(p_types, Box::new(r_type)),
                            mutable: false,
                            min_args: Some(arity),
                            max_args: Some(arity),
                        },
                    );
                }
            }
        }

        // Pass 3: Check top-level bindings and function bodies
        for item in &program.items {
            self.check_stmt(item);
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Use { .. } | Stmt::Struct { .. } => {}
            Stmt::Fn {
                name,
                params,
                ret_type,
                body,
                span,
            } => {
                let mut fn_scope = HashMap::new();
                for (p_name, ann) in params {
                    if fn_scope.contains_key(p_name) {
                        self.errors.push(SemanticError {
                            message: format!(
                                "Duplicate parameter '{p_name}' in function '{name}'"
                            ),
                            span: *span,
                        });
                    }
                    fn_scope.insert(
                        p_name.clone(),
                        Binding {
                            ty: self.resolve_ann(ann),
                            mutable: false,
                            min_args: None,
                            max_args: None,
                        },
                    );
                }
                self.scopes.push(fn_scope);
                let expected_ret = ret_type
                    .as_ref()
                    .map(|t| self.resolve_ann(t))
                    .unwrap_or(Type::Void);
                let prev_ret = self.current_ret.replace(expected_ret.clone());
                let prev_loop_depth = self.loop_depth;
                self.loop_depth = 0;

                for s in body {
                    self.check_stmt(s);
                }

                if expected_ret != Type::Void
                    && expected_ret != Type::Any
                    && expected_ret != Type::None
                    && !Self::block_definitely_returns(body)
                {
                    self.errors.push(SemanticError {
                        message: format!(
                            "Function '{name}' with return type {expected_ret:?} does not return a value on all code paths"
                        ),
                        span: *span,
                    });
                }

                self.loop_depth = prev_loop_depth;
                self.current_ret = prev_ret;
                self.scopes.pop();
            }
            Stmt::VarDecl {
                mutability,
                name,
                ty,
                init,
                span,
            } => {
                let init_ty = self.infer_expr(init);
                let final_ty = if let Some(ann) = ty {
                    let expected = self.resolve_ann(ann);
                    if !expected.compatible_with(&init_ty) {
                        self.errors.push(SemanticError {
                            message: format!(
                                "Type mismatch: cannot assign {init_ty:?} to '{name}' of type {expected:?}"
                            ),
                            span: *span,
                        });
                    }
                    expected
                } else {
                    init_ty
                };
                if let Some(scope) = self.scopes.last_mut() {
                    scope.insert(
                        name.clone(),
                        Binding {
                            ty: final_ty,
                            mutable: matches!(mutability, Mutability::Mut),
                            min_args: None,
                            max_args: None,
                        },
                    );
                }
            }
            Stmt::Assign { name, value, span } => {
                let val_ty = self.infer_expr(value);
                match self.lookup(name).cloned() {
                    None => self.errors.push(SemanticError {
                        message: format!("Undefined variable '{name}'"),
                        span: *span,
                    }),
                    Some(b) if !b.mutable => self.errors.push(SemanticError {
                        message: format!("Cannot assign to immutable binding '{name}'"),
                        span: *span,
                    }),
                    Some(b) if !b.ty.compatible_with(&val_ty) => {
                        self.errors.push(SemanticError {
                            message: format!(
                                "Type mismatch: cannot assign {val_ty:?} to '{name}' of type {:?}",
                                b.ty
                            ),
                            span: *span,
                        });
                    }
                    _ => {}
                }
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
                span,
            } => {
                let c_ty = self.infer_expr(cond);
                if c_ty != Type::Bool && c_ty != Type::Any {
                    self.errors.push(SemanticError {
                        message: format!("Condition must be Bool, found {c_ty:?}"),
                        span: *span,
                    });
                }
                self.scopes.push(HashMap::new());
                for s in then_body {
                    self.check_stmt(s);
                }
                self.scopes.pop();
                if let Some(eb) = else_body {
                    self.scopes.push(HashMap::new());
                    for s in eb {
                        self.check_stmt(s);
                    }
                    self.scopes.pop();
                }
            }
            Stmt::While { cond, body, span } => {
                let c_ty = self.infer_expr(cond);
                if c_ty != Type::Bool && c_ty != Type::Any {
                    self.errors.push(SemanticError {
                        message: format!("While condition must be Bool, found {c_ty:?}"),
                        span: *span,
                    });
                }
                self.scopes.push(HashMap::new());
                self.loop_depth += 1;
                for s in body {
                    self.check_stmt(s);
                }
                self.loop_depth -= 1;
                self.scopes.pop();
            }
            Stmt::Break(span) => {
                if self.loop_depth == 0 {
                    self.errors.push(SemanticError {
                        message: "'break' statement can only be used inside a loop".to_string(),
                        span: *span,
                    });
                }
            }
            Stmt::Continue(span) => {
                if self.loop_depth == 0 {
                    self.errors.push(SemanticError {
                        message: "'continue' statement can only be used inside a loop".to_string(),
                        span: *span,
                    });
                }
            }
            Stmt::Return { value, span } => {
                let actual = value
                    .as_ref()
                    .map(|e| self.infer_expr(e))
                    .unwrap_or(Type::Void);
                if let Some(expected) = &self.current_ret {
                    if !expected.compatible_with(&actual) {
                        self.errors.push(SemanticError {
                            message: format!(
                                "Return type mismatch: expected {expected:?}, found {actual:?}"
                            ),
                            span: *span,
                        });
                    }
                } else {
                    self.errors.push(SemanticError {
                        message: "'return' statement outside of function".to_string(),
                        span: *span,
                    });
                }
            }
            Stmt::Expr(e) => {
                self.infer_expr(e);
            }
        }
    }

    fn infer_expr(&mut self, expr: &Expr) -> Type {
        match expr {
            Expr::Int(_, _) => Type::Int,
            Expr::Float(_, _) => Type::Float,
            Expr::Str(_, _) => Type::String,
            Expr::Char(_, _) => Type::Char,
            Expr::Bool(_, _) => Type::Bool,
            Expr::None(_) => Type::None,
            Expr::Ident(name, span) => match self.lookup(name) {
                Some(b) => b.ty.clone(),
                None => {
                    self.errors.push(SemanticError {
                        message: format!("Undefined variable '{name}'"),
                        span: *span,
                    });
                    Type::Any
                }
            },
            Expr::Unary { op, expr, span } => {
                let inner = self.infer_expr(expr);
                match op {
                    UnaryOp::Neg => {
                        if inner.is_numeric() || inner.is_tensor() {
                            inner
                        } else {
                            self.errors.push(SemanticError {
                                message: format!("Cannot apply unary '-' to {inner:?}"),
                                span: *span,
                            });
                            Type::Any
                        }
                    }
                    UnaryOp::Not => {
                        if inner == Type::Bool || inner == Type::Any {
                            Type::Bool
                        } else {
                            self.errors.push(SemanticError {
                                message: format!("Cannot apply unary '!' to {inner:?}"),
                                span: *span,
                            });
                            Type::Bool
                        }
                    }
                }
            }
            Expr::Binary {
                op,
                left,
                right,
                span,
            } => {
                let l = self.infer_expr(left);
                let r = self.infer_expr(right);
                match op {
                    BinOp::Add => {
                        if l == Type::String && r == Type::String {
                            Type::String
                        } else if l == Type::Int && r == Type::Int {
                            Type::Int
                        } else if l.is_numeric() && r.is_numeric() {
                            if l == Type::Any || r == Type::Any {
                                Type::Any
                            } else {
                                Type::Float
                            }
                        } else if (l.is_tensor() && (r.is_tensor() || r.is_numeric()))
                            || (r.is_tensor() && l.is_numeric())
                        {
                            Type::Tensor(Box::new(Type::Float))
                        } else if l == Type::Any || r == Type::Any {
                            Type::Any
                        } else {
                            self.errors.push(SemanticError {
                                message: format!("Cannot apply '+' to {l:?} and {r:?}"),
                                span: *span,
                            });
                            Type::Any
                        }
                    }
                    BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        if l == Type::Int && r == Type::Int {
                            Type::Int
                        } else if l.is_numeric() && r.is_numeric() {
                            if l == Type::Any || r == Type::Any {
                                Type::Any
                            } else {
                                Type::Float
                            }
                        } else if (l.is_tensor() && (r.is_tensor() || r.is_numeric()))
                            || (r.is_tensor() && l.is_numeric())
                        {
                            Type::Tensor(Box::new(Type::Float))
                        } else {
                            self.errors.push(SemanticError {
                                message: format!("Cannot apply {op:?} to {l:?} and {r:?}"),
                                span: *span,
                            });
                            Type::Any
                        }
                    }
                    BinOp::Mod => {
                        if l == Type::Int && r == Type::Int {
                            Type::Int
                        } else if l.is_numeric() && r.is_numeric() {
                            Type::Float
                        } else {
                            self.errors.push(SemanticError {
                                message: format!("Cannot apply '%' to {l:?} and {r:?}"),
                                span: *span,
                            });
                            Type::Any
                        }
                    }
                    BinOp::MatMul => {
                        if (l.is_tensor() || l == Type::Any) && (r.is_tensor() || r == Type::Any) {
                            Type::Tensor(Box::new(Type::Float))
                        } else {
                            self.errors.push(SemanticError {
                                message: format!(
                                    "Matrix multiplication '@' requires Tensor operands, found {l:?} and {r:?}"
                                ),
                                span: *span,
                            });
                            Type::Tensor(Box::new(Type::Float))
                        }
                    }
                    BinOp::Eq | BinOp::NotEq => Type::Bool,
                    BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq => {
                        let valid = (l.is_numeric() && r.is_numeric())
                            || (l == Type::String && r == Type::String)
                            || l == Type::Any
                            || r == Type::Any;
                        if !valid {
                            self.errors.push(SemanticError {
                                message: format!(
                                    "Cannot compare {l:?} and {r:?} with ordering operator {op:?}"
                                ),
                                span: *span,
                            });
                        }
                        Type::Bool
                    }
                    BinOp::And | BinOp::Or => {
                        if (l != Type::Bool && l != Type::Any) || (r != Type::Bool && r != Type::Any)
                        {
                            self.errors.push(SemanticError {
                                message: format!(
                                    "Logical operator {op:?} requires Bool operands, found {l:?} and {r:?}"
                                ),
                                span: *span,
                            });
                        }
                        Type::Bool
                    }
                }
            }
            Expr::Call { callee, args, span } => {
                let arg_types: Vec<Type> = args.iter().map(|a| self.infer_expr(a)).collect();
                match self.lookup(callee).cloned() {
                    Some(Binding {
                        ty: Type::Fn(params, ret),
                        min_args,
                        max_args,
                        ..
                    }) => {
                        let min_a = min_args.unwrap_or(params.len());
                        let max_a = max_args;
                        if arg_types.len() < min_a || max_a.is_some_and(|m| arg_types.len() > m) {
                            let expected_desc = match max_a {
                                Some(m) if m == min_a => format!("{min_a}"),
                                Some(m) => format!("{min_a}..={m}"),
                                None => format!("at least {min_a}"),
                            };
                            self.errors.push(SemanticError {
                                message: format!(
                                    "Function '{callee}' expects {expected_desc} argument(s), got {}",
                                    arg_types.len()
                                ),
                                span: *span,
                            });
                        } else {
                            for (idx, (param_ty, arg_ty)) in
                                params.iter().zip(arg_types.iter()).enumerate()
                            {
                                if !param_ty.compatible_with(arg_ty) {
                                    self.errors.push(SemanticError {
                                        message: format!(
                                            "Argument {} of '{callee}' expects {param_ty:?}, found {arg_ty:?}",
                                            idx + 1
                                        ),
                                        span: *span,
                                    });
                                }
                            }
                        }
                        *ret
                    }
                    _ => {
                        self.errors.push(SemanticError {
                            message: format!("Undefined function '{callee}'"),
                            span: *span,
                        });
                        Type::Any
                    }
                }
            }
            Expr::Array(els, _) => {
                let mut elem_ty = Type::Any;
                for (i, e) in els.iter().enumerate() {
                    let t = self.infer_expr(e);
                    if i == 0 {
                        elem_ty = t;
                    } else if elem_ty != t {
                        elem_ty = Type::Any;
                    }
                }
                Type::Array(Box::new(elem_ty))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nova_lexer::tokenize;
    use nova_parser::Parser;

    #[test]
    fn rejects_invalid_type_annotation() {
        let src = "fn main() {\n    let age: Int = \"Hello\"\n}";
        let tokens = tokenize(src).unwrap();
        let prog = Parser::new(&tokens).parse_program().unwrap();
        assert!(check_program(&prog).is_err());
    }

    #[test]
    fn rejects_immutable_reassignment() {
        let src = "fn main() {\n    let x = 1\n    x = 2\n}";
        let tokens = tokenize(src).unwrap();
        let prog = Parser::new(&tokens).parse_program().unwrap();
        assert!(check_program(&prog).is_err());
    }

    #[test]
    fn rejects_wrong_argument_type_and_missing_return() {
        let src = r#"
fn add(a: Int, b: Int) -> Int {
    let x = a + b
}

fn main() {
    let r = add("bad", 2)
}
"#;
        let tokens = tokenize(src).unwrap();
        let prog = Parser::new(&tokens).parse_program().unwrap();
        let errs = check_program(&prog).unwrap_err();
        assert!(errs.iter().any(|e| e.message.contains("does not return")));
        assert!(errs.iter().any(|e| e.message.contains("Argument 1")));
    }

    #[test]
    fn rejects_break_outside_loop_and_invalid_binary_operands() {
        let src = r#"
fn main() {
    let x = "a" - 1
    break
}
"#;
        let tokens = tokenize(src).unwrap();
        let prog = Parser::new(&tokens).parse_program().unwrap();
        let errs = check_program(&prog).unwrap_err();
        assert!(errs.iter().any(|e| e.message.contains("Cannot apply")));
        assert!(errs.iter().any(|e| e.message.contains("inside a loop")));
    }

    #[test]
    fn accepts_ai_builtins_and_tensor_operations() {
        let src = r#"
fn main() {
    let a = tensor([[1.0, 2.0], [3.0, 4.0]])
    let b = tensor([[2.0, 0.0], [1.0, 2.0]])
    let c = (a @ b) + 1.0
    print(c)
}
"#;
        let tokens = tokenize(src).unwrap();
        let prog = Parser::new(&tokens).parse_program().unwrap();
        assert!(check_program(&prog).is_ok());
    }
}
