//! Nova Lexer: converts `.nova` source text into spanned tokens with source diagnostics.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "lex error at {}:{}: {}",
            self.span.line, self.span.column, self.message
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Ident(String),
    Int(i64),
    Float(f64),
    String(String),
    Char(char),

    // Keywords
    Fn,
    Let,
    Mut,
    Const,
    If,
    Else,
    For,
    In,
    While,
    Return,
    Break,
    Continue,
    Struct,
    Use,
    True,
    False,
    None,
    Ok,
    Error,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    At,
    EqEq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    AndAnd,
    OrOr,
    Bang,
    Eq,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,

    // Punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Semicolon,
    Dot,
    DotDot,
    Arrow,
    Newline,
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

pub fn tokenize(source: &str) -> Result<Vec<Token>, Vec<LexError>> {
    let chars: Vec<(usize, char)> = source.char_indices().collect();
    let mut i = 0;
    let mut line = 1usize;
    let mut col = 1usize;
    let mut tokens = Vec::new();
    let mut errors = Vec::new();

    let peek = |idx: usize| -> Option<(usize, char)> { chars.get(idx).copied() };

    while let Some((byte_idx, ch)) = peek(i) {
        if ch == ' ' || ch == '\t' || ch == '\r' {
            i += 1;
            col += 1;
            continue;
        }

        if ch == '\n' {
            tokens.push(Token {
                kind: TokenKind::Newline,
                span: Span {
                    start: byte_idx,
                    end: byte_idx + 1,
                    line,
                    column: col,
                },
            });
            i += 1;
            line += 1;
            col = 1;
            continue;
        }

        // Line comments: // or #
        if (ch == '/' && peek(i + 1).map(|(_, c)| c) == Some('/')) || ch == '#' {
            while let Some((_, c)) = peek(i) {
                if c == '\n' {
                    break;
                }
                i += 1;
                col += 1;
            }
            continue;
        }

        let start_col = col;
        let start_line = line;

        // String literal
        if ch == '"' {
            i += 1;
            col += 1;
            let mut value = String::new();
            let mut closed = false;
            while let Some((_, c)) = peek(i) {
                if c == '"' {
                    i += 1;
                    col += 1;
                    closed = true;
                    break;
                }
                if c == '\n' {
                    break;
                }
                if c == '\\' {
                    i += 1;
                    col += 1;
                    if let Some((_, esc)) = peek(i) {
                        i += 1;
                        col += 1;
                        match esc {
                            'n' => value.push('\n'),
                            't' => value.push('\t'),
                            'r' => value.push('\r'),
                            '"' => value.push('"'),
                            '\\' => value.push('\\'),
                            other => errors.push(LexError {
                                message: format!("Unknown escape sequence '\\{other}'"),
                                span: Span {
                                    start: byte_idx,
                                    end: byte_idx + 2,
                                    line: start_line,
                                    column: col,
                                },
                            }),
                        }
                    }
                } else {
                    value.push(c);
                    i += 1;
                    col += 1;
                }
            }
            let end_byte = peek(i).map(|(b, _)| b).unwrap_or(source.len());
            let span = Span {
                start: byte_idx,
                end: end_byte,
                line: start_line,
                column: start_col,
            };
            if closed {
                tokens.push(Token {
                    kind: TokenKind::String(value),
                    span,
                });
            } else {
                errors.push(LexError {
                    message: "Unterminated string literal".to_string(),
                    span,
                });
            }
            continue;
        }

        // Numbers
        if ch.is_ascii_digit() {
            let mut text = String::new();
            let mut is_float = false;
            while let Some((_, c)) = peek(i) {
                if c.is_ascii_digit() {
                    text.push(c);
                    i += 1;
                    col += 1;
                } else if c == '_' {
                    i += 1;
                    col += 1;
                } else {
                    break;
                }
            }
            if peek(i).map(|(_, c)| c) == Some('.')
                && peek(i + 1).map(|(_, c)| c.is_ascii_digit()).unwrap_or(false)
            {
                is_float = true;
                text.push('.');
                i += 1;
                col += 1;
                while let Some((_, c)) = peek(i) {
                    if c.is_ascii_digit() {
                        text.push(c);
                        i += 1;
                        col += 1;
                    } else {
                        break;
                    }
                }
            }
            let end_byte = peek(i).map(|(b, _)| b).unwrap_or(source.len());
            let span = Span {
                start: byte_idx,
                end: end_byte,
                line: start_line,
                column: start_col,
            };
            if is_float {
                match text.parse::<f64>() {
                    Ok(v) => tokens.push(Token {
                        kind: TokenKind::Float(v),
                        span,
                    }),
                    Err(_) => errors.push(LexError {
                        message: format!("Invalid float literal '{text}'"),
                        span,
                    }),
                }
            } else {
                match text.parse::<i64>() {
                    Ok(v) => tokens.push(Token {
                        kind: TokenKind::Int(v),
                        span,
                    }),
                    Err(_) => errors.push(LexError {
                        message: format!("Invalid integer literal '{text}'"),
                        span,
                    }),
                }
            }
            continue;
        }

        // Identifiers & Keywords
        if ch.is_ascii_alphabetic() || ch == '_' {
            let mut ident = String::new();
            while let Some((_, c)) = peek(i) {
                if c.is_ascii_alphanumeric() || c == '_' {
                    ident.push(c);
                    i += 1;
                    col += 1;
                } else {
                    break;
                }
            }
            let end_byte = peek(i).map(|(b, _)| b).unwrap_or(source.len());
            let span = Span {
                start: byte_idx,
                end: end_byte,
                line: start_line,
                column: start_col,
            };
            let kind = match ident.as_str() {
                "fn" => TokenKind::Fn,
                "let" => TokenKind::Let,
                "mut" => TokenKind::Mut,
                "const" => TokenKind::Const,
                "if" => TokenKind::If,
                "else" => TokenKind::Else,
                "for" => TokenKind::For,
                "in" => TokenKind::In,
                "while" => TokenKind::While,
                "return" => TokenKind::Return,
                "break" => TokenKind::Break,
                "continue" => TokenKind::Continue,
                "struct" => TokenKind::Struct,
                "use" => TokenKind::Use,
                "true" => TokenKind::True,
                "false" => TokenKind::False,
                "None" => TokenKind::None,
                "Ok" => TokenKind::Ok,
                "Error" => TokenKind::Error,
                _ => TokenKind::Ident(ident),
            };
            tokens.push(Token { kind, span });
            continue;
        }

        // Two-char operators
        let next_ch = peek(i + 1).map(|(_, c)| c);
        let two_tok = match (ch, next_ch) {
            ('=', Some('=')) => Some(TokenKind::EqEq),
            ('!', Some('=')) => Some(TokenKind::NotEq),
            ('<', Some('=')) => Some(TokenKind::LtEq),
            ('>', Some('=')) => Some(TokenKind::GtEq),
            ('&', Some('&')) => Some(TokenKind::AndAnd),
            ('|', Some('|')) => Some(TokenKind::OrOr),
            ('+', Some('=')) => Some(TokenKind::PlusEq),
            ('-', Some('=')) => Some(TokenKind::MinusEq),
            ('*', Some('=')) => Some(TokenKind::StarEq),
            ('/', Some('=')) => Some(TokenKind::SlashEq),
            ('.', Some('.')) => Some(TokenKind::DotDot),
            ('-', Some('>')) => Some(TokenKind::Arrow),
            _ => None,
        };
        if let Some(kind) = two_tok {
            i += 2;
            col += 2;
            tokens.push(Token {
                kind,
                span: Span {
                    start: byte_idx,
                    end: byte_idx + 2,
                    line: start_line,
                    column: start_col,
                },
            });
            continue;
        }

        // Single-char operators & punctuation
        let one_tok = match ch {
            '+' => Some(TokenKind::Plus),
            '-' => Some(TokenKind::Minus),
            '*' => Some(TokenKind::Star),
            '/' => Some(TokenKind::Slash),
            '%' => Some(TokenKind::Percent),
            '@' => Some(TokenKind::At),
            '<' => Some(TokenKind::Lt),
            '>' => Some(TokenKind::Gt),
            '!' => Some(TokenKind::Bang),
            '=' => Some(TokenKind::Eq),
            '(' => Some(TokenKind::LParen),
            ')' => Some(TokenKind::RParen),
            '{' => Some(TokenKind::LBrace),
            '}' => Some(TokenKind::RBrace),
            '[' => Some(TokenKind::LBracket),
            ']' => Some(TokenKind::RBracket),
            ',' => Some(TokenKind::Comma),
            ':' => Some(TokenKind::Colon),
            ';' => Some(TokenKind::Semicolon),
            '.' => Some(TokenKind::Dot),
            _ => None,
        };
        if let Some(kind) = one_tok {
            i += 1;
            col += 1;
            tokens.push(Token {
                kind,
                span: Span {
                    start: byte_idx,
                    end: byte_idx + ch.len_utf8(),
                    line: start_line,
                    column: start_col,
                },
            });
            continue;
        }

        i += 1;
        col += 1;
        errors.push(LexError {
            message: format!("Unexpected character '{ch}'"),
            span: Span {
                start: byte_idx,
                end: byte_idx + ch.len_utf8(),
                line: start_line,
                column: start_col,
            },
        });
    }

    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span {
            start: source.len(),
            end: source.len(),
            line,
            column: col,
        },
    });

    if errors.is_empty() {
        Ok(tokens)
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_hello_world() {
        let src = "fn main() {\n    let name = \"Nova\"\n    print(\"Hello, \" + name)\n}";
        let toks = tokenize(src).expect("should lex cleanly");
        assert!(matches!(toks[0].kind, TokenKind::Fn));
        assert!(matches!(toks[1].kind, TokenKind::Ident(ref s) if s == "main"));
    }

    #[test]
    fn reports_unterminated_string() {
        let res = tokenize("let s = \"unterminated");
        assert!(res.is_err());
    }
}
