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

        // Block comments: /* ... */
        if ch == '/' && peek(i + 1).map(|(_, c)| c) == Some('*') {
            i += 2;
            col += 2;
            let mut closed = false;
            while let Some((_, c)) = peek(i) {
                if c == '*' && peek(i + 1).map(|(_, n)| n) == Some('/') {
                    i += 2;
                    col += 2;
                    closed = true;
                    break;
                }
                if c == '\n' {
                    line += 1;
                    col = 1;
                } else {
                    col += 1;
                }
                i += 1;
            }
            if !closed {
                errors.push(LexError {
                    message: "Unterminated block comment".to_string(),
                    span: Span {
                        start: byte_idx,
                        end: source.len(),
                        line: start_line,
                        column: start_col,
                    },
                });
            }
            continue;
        }

        // Char literal
        if ch == '\'' {
            i += 1;
            col += 1;
            let char_val = match peek(i) {
                Some((_, '\\')) => {
                    let esc_start = peek(i).map(|(b, _)| b).unwrap_or(byte_idx);
                    let esc_col = col;
                    i += 1;
                    col += 1;
                    match peek(i) {
                        Some((eb, esc)) => {
                            i += 1;
                            col += 1;
                            match esc {
                                'n' => Some('\n'),
                                't' => Some('\t'),
                                'r' => Some('\r'),
                                '0' => Some('\0'),
                                '\'' => Some('\''),
                                '"' => Some('"'),
                                '\\' => Some('\\'),
                                other => {
                                    errors.push(LexError {
                                        message: format!("Unknown escape sequence '\\{other}'"),
                                        span: Span {
                                            start: esc_start,
                                            end: eb + other.len_utf8(),
                                            line: start_line,
                                            column: esc_col,
                                        },
                                    });
                                    None
                                }
                            }
                        }
                        None => None,
                    }
                }
                Some((_, '\'')) | Some((_, '\n')) | None => None,
                Some((_, c)) => {
                    i += 1;
                    col += 1;
                    Some(c)
                }
            };
            let closed = if peek(i).map(|(_, c)| c) == Some('\'') {
                i += 1;
                col += 1;
                true
            } else {
                // Recover to closing quote or newline
                while let Some((_, c)) = peek(i) {
                    if c == '\'' {
                        i += 1;
                        col += 1;
                        break;
                    }
                    if c == '\n' {
                        break;
                    }
                    i += 1;
                    col += 1;
                }
                false
            };
            let end_byte = peek(i).map(|(b, _)| b).unwrap_or(source.len());
            let span = Span {
                start: byte_idx,
                end: end_byte,
                line: start_line,
                column: start_col,
            };
            match (char_val, closed) {
                (Some(c), true) => tokens.push(Token {
                    kind: TokenKind::Char(c),
                    span,
                }),
                _ => errors.push(LexError {
                    message: "Invalid or unterminated character literal".to_string(),
                    span,
                }),
            }
            continue;
        }

        // String literal
        if ch == '"' {
            i += 1;
            col += 1;
            let mut value = String::new();
            let mut closed = false;
            while let Some((cur_byte, c)) = peek(i) {
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
                    let esc_col = col;
                    i += 1;
                    col += 1;
                    if let Some((esc_byte, esc)) = peek(i) {
                        i += 1;
                        col += 1;
                        match esc {
                            'n' => value.push('\n'),
                            't' => value.push('\t'),
                            'r' => value.push('\r'),
                            '0' => value.push('\0'),
                            '\'' => value.push('\''),
                            '"' => value.push('"'),
                            '\\' => value.push('\\'),
                            other => errors.push(LexError {
                                message: format!("Unknown escape sequence '\\{other}'"),
                                span: Span {
                                    start: cur_byte,
                                    end: esc_byte + other.len_utf8(),
                                    line: start_line,
                                    column: esc_col,
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

        // Numbers (decimal, hex 0x, binary 0b, float, scientific notation)
        if ch.is_ascii_digit() {
            if ch == '0' && matches!(peek(i + 1).map(|(_, c)| c), Some('x' | 'X' | 'b' | 'B')) {
                let prefix = peek(i + 1).map(|(_, c)| c).unwrap_or('x');
                let radix = if prefix == 'x' || prefix == 'X' { 16 } else { 2 };
                i += 2;
                col += 2;
                let mut digits = String::new();
                while let Some((_, c)) = peek(i) {
                    if c == '_' {
                        i += 1;
                        col += 1;
                    } else if (radix == 16 && c.is_ascii_hexdigit())
                        || (radix == 2 && (c == '0' || c == '1'))
                    {
                        digits.push(c);
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
                match i64::from_str_radix(&digits, radix) {
                    Ok(v) => tokens.push(Token {
                        kind: TokenKind::Int(v),
                        span,
                    }),
                    Err(_) => errors.push(LexError {
                        message: format!("Invalid base-{radix} integer literal"),
                        span,
                    }),
                }
                continue;
            }

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
                    } else if c == '_' {
                        i += 1;
                        col += 1;
                    } else {
                        break;
                    }
                }
            }
            // Scientific notation exponent: e.g. 1e-5, 2.5E+3
            if matches!(peek(i).map(|(_, c)| c), Some('e' | 'E')) {
                let next1 = peek(i + 1).map(|(_, c)| c);
                let next2 = peek(i + 2).map(|(_, c)| c);
                let has_exp_digits = matches!(next1, Some(c) if c.is_ascii_digit())
                    || (matches!(next1, Some('+' | '-'))
                        && matches!(next2, Some(c) if c.is_ascii_digit()));
                if has_exp_digits {
                    is_float = true;
                    text.push('e');
                    i += 1;
                    col += 1;
                    if let Some((_, sign @ ('+' | '-'))) = peek(i) {
                        text.push(sign);
                        i += 1;
                        col += 1;
                    }
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

    #[test]
    fn tokenizes_char_block_comments_and_numeric_formats() {
        let src = "/* multi\nline */ let c = 'x'\nlet h = 0xff\nlet b = 0b1010\nlet f = 1.5e-2";
        let toks = tokenize(src).expect("should lex cleanly");
        assert!(toks.iter().any(|t| t.kind == TokenKind::Char('x')));
        assert!(toks.iter().any(|t| t.kind == TokenKind::Int(255)));
        assert!(toks.iter().any(|t| t.kind == TokenKind::Int(10)));
        assert!(toks.iter().any(|t| matches!(t.kind, TokenKind::Float(v) if (v - 0.015).abs() < 1e-12)));
    }

    #[test]
    fn reports_accurate_escape_span_and_unterminated_block_comment() {
        let err = tokenize("let s = \"abc\\q\"").unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0].span.start, 12);
        assert_eq!(err[0].span.column, 13);

        let block_err = tokenize("/* unterminated").unwrap_err();
        assert_eq!(block_err.len(), 1);
        assert!(block_err[0].message.contains("block comment"));
    }
}
