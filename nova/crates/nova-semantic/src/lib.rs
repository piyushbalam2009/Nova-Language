//! Nova Semantic Analyzer & Static Type Checker.

use nova_lexer::Span;
use nova_parser::{BinOp, Expr, Mutability, Program, Stmt, TypeAnn};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Float,
    Bool,
    String,
    Void,
    None,
    Any,
    Array(Box<Type>),
    Tensor(Box<Type>),
    Fn(Vec<Type>, Box<Type>),
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
    errors: Vec<SemanticError>,
    current_ret: Option<Type>,
}

impl TypeChecker {
    fn new() -> Self {
        let mut globals = HashMap::new();
        globals.insert(
            "print".to_string(),
            Binding {
                ty: Type::Fn(vec![Type::Any], Box::new(Type::Void)),
                mutable: false,
            },
        );
        globals.insert(
            "str".to_string(),
            Binding {
                ty: Type::Fn(vec![Type::Any], Box::new(Type::String)),
                mutable: false,
            },
        );
        Self {
            scopes: vec![globals],
            errors: Vec::new(),
            current_ret: None,
        }
    }

    fn resolve_ann(ann: &TypeAnn) -> Type {
        match ann {
            TypeAnn::Simple(s) => match s.as_str() {
                "Int" => Type::Int,
                "Float" => Type::Float,
                "Bool" => Type::Bool,
                "String" => Type::String,
                "Void" => Type::Void,
                "None" => Type::None,
                _ => Type::Any,
            },
            TypeAnn::Generic(name, args) => {
                let inner = args.first().map(Self::resolve_ann).unwrap_or(Type::Any);
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

    fn check(&mut self, program: &Program) {
        // Register function signatures first
        for item in &program.items {
            if let Stmt::Fn {
                name,
                params,
                ret_type,
                ..
            } = item
            {
                let p_types = params.iter().map(|(_, t)| Self::resolve_ann(t)).collect();
                let r_type = ret_type.as_ref().map(Self::resolve_ann).unwrap_or(Type::Void);
                if let Some(global) = self.scopes.first_mut() {
                    global.insert(
                        name.clone(),
                        Binding {
                            ty: Type::Fn(p_types, Box::new(r_type)),
                            mutable: false,
                        },
                    );
                }
            }
        }
        for item in &program.items {
            self.check_stmt(item);
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Use { .. } | Stmt::Struct { .. } => {}
            Stmt::Fn {
                params,
                ret_type,
                body,
                ..
            } => {
                let mut fn_scope = HashMap::new();
                for (name, ann) in params {
                    fn_scope.insert(
                        name.clone(),
                        Binding {
                            ty: Self::resolve_ann(ann),
                            mutable: false,
                        },
                    );
                }
                self.scopes.push(fn_scope);
                let prev = self.current_ret.replace(
                    ret_type.as_ref().map(Self::resolve_ann).unwrap_or(Type::Void),
                );
                for s in body {
                    self.check_stmt(s);
                }
                self.current_ret = prev;
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
                    let expected = Self::resolve_ann(ann);
                    if expected != Type::Any && init_ty != Type::Any && expected != init_ty {
                        self.errors.push(SemanticError {
                            message: format!(
                                "Type mismatch: cannot assign {:?} to '{}' of type {:?}",
                                init_ty, name, expected
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
                    Some(b) if b.ty != Type::Any && val_ty != Type::Any && b.ty != val_ty => {
                        self.errors.push(SemanticError {
                            message: format!(
                                "Type mismatch: cannot assign {:?} to '{name}' of type {:?}",
                                val_ty, b.ty
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
                        message: format!("Condition must be Bool, found {:?}", c_ty),
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
                        message: format!("While condition must be Bool, found {:?}", c_ty),
                        span: *span,
                    });
                }
                self.scopes.push(HashMap::new());
                for s in body {
                    self.check_stmt(s);
                }
                self.scopes.pop();
            }
            Stmt::Return { value, span } => {
                let actual = value
                    .as_ref()
                    .map(|e| self.infer_expr(e))
                    .unwrap_or(Type::Void);
                if let Some(expected) = &self.current_ret {
                    if *expected != Type::Any && actual != Type::Any && *expected != actual {
                        self.errors.push(SemanticError {
                            message: format!(
                                "Return type mismatch: expected {:?}, found {:?}",
                                expected, actual
                            ),
                            span: *span,
                        });
                    }
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
                        } else if (l == Type::Float || l == Type::Int)
                            && (r == Type::Float || r == Type::Int)
                        {
                            Type::Float
                        } else if l == Type::Any || r == Type::Any {
                            Type::Any
                        } else {
                            self.errors.push(SemanticError {
                                message: format!("Cannot apply '+' to {:?} and {:?}", l, r),
                                span: *span,
                            });
                            Type::Any
                        }
                    }
                    BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        if l == Type::Int && r == Type::Int {
                            Type::Int
                        } else {
                            Type::Float
                        }
                    }
                    BinOp::MatMul => Type::Tensor(Box::new(Type::Float)),
                    BinOp::Eq
                    | BinOp::NotEq
                    | BinOp::Lt
                    | BinOp::LtEq
                    | BinOp::Gt
                    | BinOp::GtEq
                    | BinOp::And
                    | BinOp::Or => Type::Bool,
                }
            }
            Expr::Call { callee, args, span } => {
                let arg_types: Vec<Type> = args.iter().map(|a| self.infer_expr(a)).collect();
                match self.lookup(callee).cloned() {
                    Some(Binding {
                        ty: Type::Fn(params, ret),
                        ..
                    }) => {
                        if callee != "print" && params.len() != arg_types.len() {
                            self.errors.push(SemanticError {
                                message: format!(
                                    "Function '{callee}' expects {} arguments, got {}",
                                    params.len(),
                                    arg_types.len()
                                ),
                                span: *span,
                            });
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
                let elem = els.first().map(|e| self.infer_expr(e)).unwrap_or(Type::Any);
                Type::Array(Box::new(elem))
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
}
