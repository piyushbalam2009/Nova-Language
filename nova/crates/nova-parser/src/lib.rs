//! Nova Pratt Parser & Abstract Syntax Tree.

use nova_lexer::{Span, Token, TokenKind};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "parse error at {}:{}: {}",
            self.span.line, self.span.column, self.message
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnn {
    Simple(String),
    Generic(String, Vec<TypeAnn>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Mutability {
    Let,
    Mut,
    Const,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Use { path: Vec<String>, span: Span },
    Fn {
        name: String,
        params: Vec<(String, TypeAnn)>,
        ret_type: Option<TypeAnn>,
        body: Vec<Stmt>,
        span: Span,
    },
    Struct {
        name: String,
        fields: Vec<(String, TypeAnn)>,
        span: Span,
    },
    VarDecl {
        mutability: Mutability,
        name: String,
        ty: Option<TypeAnn>,
        init: Expr,
        span: Span,
    },
    Assign { name: String, value: Expr, span: Span },
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Option<Vec<Stmt>>,
        span: Span,
    },
    While { cond: Expr, body: Vec<Stmt>, span: Span },
    Break(Span),
    Continue(Span),
    Return { value: Option<Expr>, span: Span },
    Expr(Expr),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    MatMul,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Int(i64, Span),
    Float(f64, Span),
    Str(String, Span),
    Char(char, Span),
    Bool(bool, Span),
    None(Span),
    Ident(String, Span),
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
        span: Span,
    },
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    Call {
        callee: String,
        args: Vec<Expr>,
        span: Span,
    },
    Array(Vec<Expr>, Span),
}

pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    errors: Vec<ParseError>,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens,
            pos: 0,
            errors: Vec::new(),
        }
    }

    pub fn parse_program(mut self) -> Result<Program, Vec<ParseError>> {
        let mut items = Vec::new();
        self.skip_newlines();
        while !self.is_eof() {
            if let Some(stmt) = self.parse_stmt() {
                items.push(stmt);
            } else {
                self.advance();
            }
            self.skip_newlines();
        }
        if self.errors.is_empty() {
            Ok(Program { items })
        } else {
            Err(self.errors)
        }
    }

    fn peek(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .unwrap_or_else(|| &self.tokens[self.tokens.len() - 1])
    }

    fn advance(&mut self) -> &Token {
        let idx = self.pos;
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        &self.tokens[idx]
    }

    fn is_eof(&self) -> bool {
        matches!(self.peek().kind, TokenKind::Eof)
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek().kind, TokenKind::Newline | TokenKind::Semicolon) {
            self.advance();
        }
    }

    fn expect(&mut self, expected: TokenKind, label: &str) -> Option<Span> {
        if self.peek().kind == expected {
            Some(self.advance().span)
        } else {
            let tok = self.peek().clone();
            self.errors.push(ParseError {
                message: format!("Expected {label}, found {:?}", tok.kind),
                span: tok.span,
            });
            None
        }
    }

    fn parse_stmt(&mut self) -> Option<Stmt> {
        self.skip_newlines();
        match self.peek().kind.clone() {
            TokenKind::Use => self.parse_use(),
            TokenKind::Fn => self.parse_fn(),
            TokenKind::Struct => self.parse_struct(),
            TokenKind::Let => self.parse_var(Mutability::Let),
            TokenKind::Mut => self.parse_var(Mutability::Mut),
            TokenKind::Const => self.parse_var(Mutability::Const),
            TokenKind::If => self.parse_if(),
            TokenKind::While => self.parse_while(),
            TokenKind::Break => {
                let span = self.advance().span;
                Some(Stmt::Break(span))
            }
            TokenKind::Continue => {
                let span = self.advance().span;
                Some(Stmt::Continue(span))
            }
            TokenKind::Return => self.parse_return(),
            TokenKind::Eof => None,
            _ => {
                let expr = self.parse_expr(0)?;
                let compound_op = match self.peek().kind {
                    TokenKind::Eq => Some(None),
                    TokenKind::PlusEq => Some(Some(BinOp::Add)),
                    TokenKind::MinusEq => Some(Some(BinOp::Sub)),
                    TokenKind::StarEq => Some(Some(BinOp::Mul)),
                    TokenKind::SlashEq => Some(Some(BinOp::Div)),
                    _ => None,
                };
                if let Some(op_opt) = compound_op {
                    let eq_span = self.advance().span;
                    if let Expr::Ident(name, id_span) = expr {
                        let rhs = self.parse_expr(0)?;
                        let value = match op_opt {
                            None => rhs,
                            Some(op) => Expr::Binary {
                                op,
                                left: Box::new(Expr::Ident(name.clone(), id_span)),
                                right: Box::new(rhs),
                                span: eq_span,
                            },
                        };
                        return Some(Stmt::Assign {
                            name,
                            value,
                            span: eq_span,
                        });
                    }
                    self.errors.push(ParseError {
                        message: "Invalid assignment target".to_string(),
                        span: eq_span,
                    });
                    return None;
                }
                Some(Stmt::Expr(expr))
            }
        }
    }

    fn parse_use(&mut self) -> Option<Stmt> {
        let span = self.advance().span;
        let mut path = Vec::new();
        if let TokenKind::Ident(seg) = self.peek().kind.clone() {
            self.advance();
            path.push(seg);
            while matches!(self.peek().kind, TokenKind::Dot) {
                self.advance();
                if let TokenKind::Ident(next) = self.peek().kind.clone() {
                    self.advance();
                    path.push(next);
                } else {
                    let tok = self.peek().clone();
                    self.errors.push(ParseError {
                        message: format!("Expected module identifier after '.', found {:?}", tok.kind),
                        span: tok.span,
                    });
                    break;
                }
            }
        } else {
            let tok = self.peek().clone();
            self.errors.push(ParseError {
                message: format!("Expected module path after 'use', found {:?}", tok.kind),
                span: tok.span,
            });
        }
        Some(Stmt::Use { path, span })
    }

    fn parse_struct(&mut self) -> Option<Stmt> {
        let span = self.advance().span;
        let name = match self.peek().kind.clone() {
            TokenKind::Ident(n) => {
                self.advance();
                n
            }
            _ => {
                let tok = self.peek().clone();
                self.errors.push(ParseError {
                    message: format!("Expected struct name, found {:?}", tok.kind),
                    span: tok.span,
                });
                return None;
            }
        };
        self.skip_newlines();
        if self.expect(TokenKind::LBrace, "'{'").is_none() {
            return None;
        }
        let mut fields = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek().kind, TokenKind::RBrace | TokenKind::Eof) {
            if let TokenKind::Ident(field_name) = self.peek().kind.clone() {
                self.advance();
                self.expect(TokenKind::Colon, "':'");
                let ty = self.parse_type();
                fields.push((field_name, ty));
                if matches!(self.peek().kind, TokenKind::Comma) {
                    self.advance();
                }
            } else {
                let tok = self.peek().clone();
                self.errors.push(ParseError {
                    message: format!("Expected field name in struct '{name}', found {:?}", tok.kind),
                    span: tok.span,
                });
                self.advance();
            }
            self.skip_newlines();
        }
        self.expect(TokenKind::RBrace, "'}'");
        Some(Stmt::Struct { name, fields, span })
    }

    fn parse_type(&mut self) -> TypeAnn {
        if let TokenKind::Ident(name) = self.peek().kind.clone() {
            self.advance();
            if matches!(self.peek().kind, TokenKind::Lt) {
                self.advance();
                let mut args = Vec::new();
                while !matches!(self.peek().kind, TokenKind::Gt | TokenKind::Eof) {
                    let before = self.pos;
                    args.push(self.parse_type());
                    if matches!(self.peek().kind, TokenKind::Comma) {
                        self.advance();
                    } else if self.pos == before {
                        // Guard against infinite loop on malformed generic arguments
                        self.advance();
                        break;
                    } else if !matches!(self.peek().kind, TokenKind::Gt) {
                        break;
                    }
                }
                self.expect(TokenKind::Gt, "'>'");
                return TypeAnn::Generic(name, args);
            }
            TypeAnn::Simple(name)
        } else {
            let tok = self.peek().clone();
            self.errors.push(ParseError {
                message: format!("Expected type name, found {:?}", tok.kind),
                span: tok.span,
            });
            TypeAnn::Simple("Any".to_string())
        }
    }

    fn parse_block(&mut self) -> Vec<Stmt> {
        self.skip_newlines();
        if self.expect(TokenKind::LBrace, "'{'").is_none() {
            return Vec::new();
        }
        let mut body = Vec::new();
        self.skip_newlines();
        while !matches!(self.peek().kind, TokenKind::RBrace | TokenKind::Eof) {
            if let Some(s) = self.parse_stmt() {
                body.push(s);
            } else {
                self.advance();
            }
            self.skip_newlines();
        }
        self.expect(TokenKind::RBrace, "'}'");
        body
    }

    fn parse_fn(&mut self) -> Option<Stmt> {
        let span = self.advance().span;
        let name = match self.peek().kind.clone() {
            TokenKind::Ident(n) => {
                self.advance();
                n
            }
            _ => {
                self.errors.push(ParseError {
                    message: "Expected function name".to_string(),
                    span: self.peek().span,
                });
                return None;
            }
        };
        self.expect(TokenKind::LParen, "'('");
        let mut params = Vec::new();
        while !matches!(self.peek().kind, TokenKind::RParen | TokenKind::Eof) {
            if let TokenKind::Ident(p_name) = self.peek().kind.clone() {
                self.advance();
                self.expect(TokenKind::Colon, "':'");
                let ty = self.parse_type();
                params.push((p_name, ty));
            } else {
                let tok = self.peek().clone();
                self.errors.push(ParseError {
                    message: format!("Expected parameter name, found {:?}", tok.kind),
                    span: tok.span,
                });
                self.advance();
                break;
            }
            if matches!(self.peek().kind, TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(TokenKind::RParen, "')'");
        let ret_type = if matches!(self.peek().kind, TokenKind::Arrow) {
            self.advance();
            Some(self.parse_type())
        } else {
            None
        };
        let body = self.parse_block();
        Some(Stmt::Fn {
            name,
            params,
            ret_type,
            body,
            span,
        })
    }

    fn parse_var(&mut self, mutability: Mutability) -> Option<Stmt> {
        let span = self.advance().span;
        let name = match self.peek().kind.clone() {
            TokenKind::Ident(n) => {
                self.advance();
                n
            }
            _ => {
                self.errors.push(ParseError {
                    message: "Expected variable name".to_string(),
                    span: self.peek().span,
                });
                return None;
            }
        };
        let ty = if matches!(self.peek().kind, TokenKind::Colon) {
            self.advance();
            Some(self.parse_type())
        } else {
            None
        };
        self.expect(TokenKind::Eq, "'='");
        let init = self.parse_expr(0)?;
        Some(Stmt::VarDecl {
            mutability,
            name,
            ty,
            init,
            span,
        })
    }

    fn parse_if(&mut self) -> Option<Stmt> {
        let span = self.advance().span;
        let cond = self.parse_expr(0)?;
        let then_body = self.parse_block();
        let else_body = if matches!(self.peek().kind, TokenKind::Else) {
            self.advance();
            if matches!(self.peek().kind, TokenKind::If) {
                let nested = self.parse_if()?;
                Some(vec![nested])
            } else {
                Some(self.parse_block())
            }
        } else {
            None
        };
        Some(Stmt::If {
            cond,
            then_body,
            else_body,
            span,
        })
    }

    fn parse_while(&mut self) -> Option<Stmt> {
        let span = self.advance().span;
        let cond = self.parse_expr(0)?;
        let body = self.parse_block();
        Some(Stmt::While { cond, body, span })
    }

    fn parse_return(&mut self) -> Option<Stmt> {
        let span = self.advance().span;
        if matches!(
            self.peek().kind,
            TokenKind::Newline | TokenKind::RBrace | TokenKind::Eof
        ) {
            return Some(Stmt::Return { value: None, span });
        }
        let value = self.parse_expr(0);
        Some(Stmt::Return { value, span })
    }

    fn infix_bp(kind: &TokenKind) -> Option<(u8, u8, BinOp)> {
        match kind {
            TokenKind::OrOr => Some((10, 11, BinOp::Or)),
            TokenKind::AndAnd => Some((20, 21, BinOp::And)),
            TokenKind::EqEq => Some((30, 31, BinOp::Eq)),
            TokenKind::NotEq => Some((30, 31, BinOp::NotEq)),
            TokenKind::Lt => Some((40, 41, BinOp::Lt)),
            TokenKind::LtEq => Some((40, 41, BinOp::LtEq)),
            TokenKind::Gt => Some((40, 41, BinOp::Gt)),
            TokenKind::GtEq => Some((40, 41, BinOp::GtEq)),
            TokenKind::Plus => Some((50, 51, BinOp::Add)),
            TokenKind::Minus => Some((50, 51, BinOp::Sub)),
            TokenKind::Star => Some((60, 61, BinOp::Mul)),
            TokenKind::Slash => Some((60, 61, BinOp::Div)),
            TokenKind::Percent => Some((60, 61, BinOp::Mod)),
            TokenKind::At => Some((60, 61, BinOp::MatMul)),
            _ => None,
        }
    }

    fn parse_expr(&mut self, min_bp: u8) -> Option<Expr> {
        let mut lhs = self.parse_primary()?;
        while let Some((l_bp, r_bp, op)) = Self::infix_bp(&self.peek().kind) {
            if l_bp < min_bp {
                break;
            }
            let op_span = self.advance().span;
            let rhs = self.parse_expr(r_bp)?;
            lhs = Expr::Binary {
                op,
                left: Box::new(lhs),
                right: Box::new(rhs),
                span: op_span,
            };
        }
        Some(lhs)
    }

    fn parse_primary(&mut self) -> Option<Expr> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::Int(v) => {
                self.advance();
                Some(Expr::Int(v, tok.span))
            }
            TokenKind::Float(v) => {
                self.advance();
                Some(Expr::Float(v, tok.span))
            }
            TokenKind::String(s) => {
                self.advance();
                Some(Expr::Str(s, tok.span))
            }
            TokenKind::Char(c) => {
                self.advance();
                Some(Expr::Char(c, tok.span))
            }
            TokenKind::True => {
                self.advance();
                Some(Expr::Bool(true, tok.span))
            }
            TokenKind::False => {
                self.advance();
                Some(Expr::Bool(false, tok.span))
            }
            TokenKind::None => {
                self.advance();
                Some(Expr::None(tok.span))
            }
            TokenKind::Minus => {
                self.advance();
                let inner = self.parse_expr(70)?;
                Some(Expr::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(inner),
                    span: tok.span,
                })
            }
            TokenKind::Bang => {
                self.advance();
                let inner = self.parse_expr(70)?;
                Some(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(inner),
                    span: tok.span,
                })
            }
            TokenKind::Ident(name) => {
                self.advance();
                if matches!(self.peek().kind, TokenKind::LParen) {
                    self.advance();
                    let mut args = Vec::new();
                    while !matches!(self.peek().kind, TokenKind::RParen | TokenKind::Eof) {
                        if let Some(a) = self.parse_expr(0) {
                            args.push(a);
                        }
                        if matches!(self.peek().kind, TokenKind::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.expect(TokenKind::RParen, "')'");
                    Some(Expr::Call {
                        callee: name,
                        args,
                        span: tok.span,
                    })
                } else {
                    Some(Expr::Ident(name, tok.span))
                }
            }
            TokenKind::LParen => {
                self.advance();
                let inner = self.parse_expr(0)?;
                self.expect(TokenKind::RParen, "')'");
                Some(inner)
            }
            TokenKind::LBracket => {
                self.advance();
                let mut els = Vec::new();
                while !matches!(self.peek().kind, TokenKind::RBracket | TokenKind::Eof) {
                    if let Some(e) = self.parse_expr(0) {
                        els.push(e);
                    }
                    if matches!(self.peek().kind, TokenKind::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                self.expect(TokenKind::RBracket, "']'");
                Some(Expr::Array(els, tok.span))
            }
            _ => {
                self.errors.push(ParseError {
                    message: format!("Unexpected token {:?} in expression", tok.kind),
                    span: tok.span,
                });
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nova_lexer::tokenize;

    #[test]
    fn parses_hello_world_program() {
        let src = "fn main() {\n    let name = \"Nova\"\n    print(\"Hello, \" + name)\n}";
        let tokens = tokenize(src).expect("lex failed");
        let prog = Parser::new(&tokens).parse_program().expect("parse failed");
        assert_eq!(prog.items.len(), 1);
    }

    #[test]
    fn parses_unary_else_if_struct_break_continue_and_compound_assign() {
        let src = r#"
struct Point {
    x: Float,
    y: Float
}

fn main() {
    mut n = -5
    n += 10
    if !false && n > 0 {
        while n > 0 {
            n -= 1
            if n == 2 {
                continue
            }
            if n == 1 {
                break
            }
        }
    } else if n == 0 {
        print("zero")
    } else {
        print("neg")
    }
}
"#;
        let tokens = tokenize(src).expect("lex failed");
        let prog = Parser::new(&tokens).parse_program().expect("parse failed");
        assert_eq!(prog.items.len(), 2);
        assert!(matches!(prog.items[0], Stmt::Struct { ref name, .. } if name == "Point"));
    }

    #[test]
    fn terminates_cleanly_on_malformed_generic_type() {
        let src = "fn bad(x: Array<123>) {}";
        let tokens = tokenize(src).expect("lex failed");
        let res = Parser::new(&tokens).parse_program();
        assert!(res.is_err());
    }
}
