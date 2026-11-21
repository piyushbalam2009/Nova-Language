import { Diagnostic, Span } from './diagnostics';

export enum TokenKind {
  // Literals & identifiers
  Ident = 'Ident',
  Int = 'Int',
  Float = 'Float',
  String = 'String',
  Char = 'Char',

  // Keywords
  Fn = 'Fn',
  Let = 'Let',
  Mut = 'Mut',
  Const = 'Const',
  If = 'If',
  Else = 'Else',
  For = 'For',
  In = 'In',
  While = 'While',
  Return = 'Return',
  Break = 'Break',
  Continue = 'Continue',
  Struct = 'Struct',
  Use = 'Use',
  True = 'True',
  False = 'False',
  None = 'None',
  Ok = 'Ok',
  Error = 'Error',

  // Arithmetic & tensor operators
  Plus = 'Plus',
  Minus = 'Minus',
  Star = 'Star',
  Slash = 'Slash',
  Percent = 'Percent',
  At = 'At', // @ matrix multiplication

  // Comparison
  EqEq = 'EqEq',
  NotEq = 'NotEq',
  Lt = 'Lt',
  LtEq = 'LtEq',
  Gt = 'Gt',
  GtEq = 'GtEq',

  // Logical
  AndAnd = 'AndAnd',
  OrOr = 'OrOr',
  Bang = 'Bang',

  // Assignment
  Eq = 'Eq',
  PlusEq = 'PlusEq',
  MinusEq = 'MinusEq',
  StarEq = 'StarEq',
  SlashEq = 'SlashEq',

  // Punctuation & delimiters
  LParen = 'LParen',
  RParen = 'RParen',
  LBrace = 'LBrace',
  RBrace = 'RBrace',
  LBracket = 'LBracket',
  RBracket = 'RBracket',
  Comma = 'Comma',
  Colon = 'Colon',
  Semicolon = 'Semicolon',
  Dot = 'Dot',
  DotDot = 'DotDot',
  Arrow = 'Arrow',

  Newline = 'Newline',
  Eof = 'Eof',
}

export interface Token {
  kind: TokenKind;
  lexeme: string;
  span: Span;
}

const KEYWORDS: Record<string, TokenKind> = {
  fn: TokenKind.Fn,
  let: TokenKind.Let,
  mut: TokenKind.Mut,
  const: TokenKind.Const,
  if: TokenKind.If,
  else: TokenKind.Else,
  for: TokenKind.For,
  in: TokenKind.In,
  while: TokenKind.While,
  return: TokenKind.Return,
  break: TokenKind.Break,
  continue: TokenKind.Continue,
  struct: TokenKind.Struct,
  use: TokenKind.Use,
  true: TokenKind.True,
  false: TokenKind.False,
  None: TokenKind.None,
  Ok: TokenKind.Ok,
  Error: TokenKind.Error,
};

export interface LexResult {
  tokens: Token[];
  diagnostics: Diagnostic[];
}

export function lex(source: string): LexResult {
  const tokens: Token[] = [];
  const diagnostics: Diagnostic[] = [];

  let i = 0;
  let line = 1;
  let col = 1;
  const n = source.length;

  const peek = (offset = 0): string => (i + offset < n ? source[i + offset] : '\0');

  const advance = (): string => {
    const ch = source[i++];
    if (ch === '\n') {
      line++;
      col = 1;
    } else {
      col++;
    }
    return ch;
  };

  const isDigit = (c: string): boolean => c >= '0' && c <= '9';
  const isIdentStart = (c: string): boolean =>
    (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c === '_';
  const isIdentPart = (c: string): boolean => isIdentStart(c) || isDigit(c);

  while (i < n) {
    const ch = peek();

    // Horizontal whitespace
    if (ch === ' ' || ch === '\t' || ch === '\r') {
      advance();
      continue;
    }

    // Newline
    if (ch === '\n') {
      const start = i;
      const startLine = line;
      const startCol = col;
      advance();
      tokens.push({
        kind: TokenKind.Newline,
        lexeme: '\n',
        span: { start, end: i, line: startLine, column: startCol },
      });
      continue;
    }

    // Line comments: // or #
    if ((ch === '/' && peek(1) === '/') || ch === '#') {
      while (i < n && peek() !== '\n') {
        advance();
      }
      continue;
    }

    // Block comments: /* ... */
    if (ch === '/' && peek(1) === '*') {
      const start = i;
      const startLine = line;
      const startCol = col;
      advance();
      advance();
      let closed = false;
      while (i < n) {
        if (peek() === '*' && peek(1) === '/') {
          advance();
          advance();
          closed = true;
          break;
        }
        advance();
      }
      if (!closed) {
        diagnostics.push({
          stage: 'lexer',
          severity: 'error',
          code: 'E0001',
          message: 'Unterminated block comment',
          span: { start, end: i, line: startLine, column: startCol },
          hint: 'Close the block comment with */',
        });
      }
      continue;
    }

    const start = i;
    const startLine = line;
    const startCol = col;

    // String literal
    if (ch === '"') {
      advance(); // consume opening "
      let value = '';
      let terminated = false;
      while (i < n) {
        const c = peek();
        if (c === '"') {
          advance();
          terminated = true;
          break;
        }
        if (c === '\n') {
          break;
        }
        if (c === '\\') {
          advance();
          if (i >= n) break;
          const esc = advance();
          switch (esc) {
            case 'n': value += '\n'; break;
            case 't': value += '\t'; break;
            case 'r': value += '\r'; break;
            case '0': value += '\0'; break;
            case '"': value += '"'; break;
            case "'": value += "'"; break;
            case '\\': value += '\\'; break;
            default:
              diagnostics.push({
                stage: 'lexer',
                severity: 'error',
                code: 'E0002',
                message: `Unknown escape sequence '\\${esc}'`,
                span: { start: i - 2, end: i, line, column: Math.max(1, col - 2) },
              });
              value += esc;
          }
        } else {
          value += advance();
        }
      }
      if (!terminated) {
        diagnostics.push({
          stage: 'lexer',
          severity: 'error',
          code: 'E0003',
          message: 'Unterminated string literal',
          span: { start, end: i, line: startLine, column: startCol },
          hint: 'Add a closing double quote "',
        });
      } else {
        tokens.push({
          kind: TokenKind.String,
          lexeme: value,
          span: { start, end: i, line: startLine, column: startCol },
        });
      }
      continue;
    }

    // Char literal
    if (ch === "'") {
      advance();
      let charVal = '';
      if (i < n && peek() === '\\') {
        advance();
        const esc = i < n ? advance() : '';
        if (esc === 'n') charVal = '\n';
        else if (esc === 't') charVal = '\t';
        else if (esc === 'r') charVal = '\r';
        else if (esc === '0') charVal = '\0';
        else if (esc === "'") charVal = "'";
        else if (esc === '\\') charVal = '\\';
        else charVal = esc;
      } else if (i < n && peek() !== "'" && peek() !== '\n') {
        charVal = advance();
      }
      if (peek() !== "'") {
        diagnostics.push({
          stage: 'lexer',
          severity: 'error',
          code: 'E0004',
          message: 'Malformed or unterminated character literal',
          span: { start, end: i, line: startLine, column: startCol },
          hint: "Character literals must contain a single character, e.g. 'a'",
        });
        while (i < n && peek() !== "'" && peek() !== '\n') advance();
        if (peek() === "'") advance();
      } else {
        advance();
        if (charVal.length === 0) {
          diagnostics.push({
            stage: 'lexer',
            severity: 'error',
            code: 'E0004',
            message: 'Empty character literal',
            span: { start, end: i, line: startLine, column: startCol },
          });
        } else {
          tokens.push({
            kind: TokenKind.Char,
            lexeme: charVal,
            span: { start, end: i, line: startLine, column: startCol },
          });
        }
      }
      continue;
    }

    // Number literal (Hex, Binary, Int, or Float)
    if (isDigit(ch)) {
      // Hexadecimal: 0x...
      if (ch === '0' && (peek(1) === 'x' || peek(1) === 'X')) {
        advance();
        advance();
        let hexDigits = '';
        const isHex = (c: string) => isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        while (i < n && (isHex(peek()) || peek() === '_')) {
          const d = advance();
          if (d !== '_') hexDigits += d;
        }
        if (hexDigits.length === 0 || isIdentPart(peek())) {
          while (i < n && isIdentPart(peek())) advance();
          diagnostics.push({
            stage: 'lexer',
            severity: 'error',
            code: 'E0006',
            message: `Invalid hexadecimal literal '0x${hexDigits}'`,
            span: { start, end: i, line: startLine, column: startCol },
          });
          continue;
        }
        tokens.push({
          kind: TokenKind.Int,
          lexeme: String(Number.parseInt(hexDigits, 16)),
          span: { start, end: i, line: startLine, column: startCol },
        });
        continue;
      }

      // Binary: 0b...
      if (ch === '0' && (peek(1) === 'b' || peek(1) === 'B')) {
        advance();
        advance();
        let binDigits = '';
        while (i < n && (peek() === '0' || peek() === '1' || peek() === '_')) {
          const d = advance();
          if (d !== '_') binDigits += d;
        }
        if (binDigits.length === 0 || isIdentPart(peek())) {
          while (i < n && isIdentPart(peek())) advance();
          diagnostics.push({
            stage: 'lexer',
            severity: 'error',
            code: 'E0006',
            message: `Invalid binary literal '0b${binDigits}'`,
            span: { start, end: i, line: startLine, column: startCol },
          });
          continue;
        }
        tokens.push({
          kind: TokenKind.Int,
          lexeme: String(Number.parseInt(binDigits, 2)),
          span: { start, end: i, line: startLine, column: startCol },
        });
        continue;
      }

      let text = '';
      let isFloat = false;
      while (i < n && (isDigit(peek()) || peek() === '_')) {
        const d = advance();
        if (d !== '_') text += d;
      }
      if (peek() === '.' && peek(1) !== '.' && isDigit(peek(1))) {
        isFloat = true;
        text += advance(); // '.'
        while (i < n && (isDigit(peek()) || peek() === '_')) {
          const d = advance();
          if (d !== '_') text += d;
        }
      }
      // Scientific notation e.g. 1e-4 or 2.5e3
      if (peek() === 'e' || peek() === 'E') {
        isFloat = true;
        text += advance();
        if (peek() === '+' || peek() === '-') {
          text += advance();
        }
        if (!isDigit(peek())) {
          diagnostics.push({
            stage: 'lexer',
            severity: 'error',
            code: 'E0005',
            message: `Invalid floating-point exponent in '${text}'`,
            span: { start, end: i, line: startLine, column: startCol },
          });
          continue;
        }
        while (i < n && isDigit(peek())) {
          text += advance();
        }
      }
      if (isIdentStart(peek())) {
        while (i < n && isIdentPart(peek())) text += advance();
        diagnostics.push({
          stage: 'lexer',
          severity: 'error',
          code: 'E0006',
          message: `Invalid numeric literal '${text}'`,
          span: { start, end: i, line: startLine, column: startCol },
        });
        continue;
      }
      tokens.push({
        kind: isFloat ? TokenKind.Float : TokenKind.Int,
        lexeme: text,
        span: { start, end: i, line: startLine, column: startCol },
      });
      continue;
    }

    // Identifiers and keywords
    if (isIdentStart(ch)) {
      let ident = '';
      while (i < n && isIdentPart(peek())) {
        ident += advance();
      }
      const kind = KEYWORDS[ident] ?? TokenKind.Ident;
      tokens.push({
        kind,
        lexeme: ident,
        span: { start, end: i, line: startLine, column: startCol },
      });
      continue;
    }

    // Two-character operators
    const two = ch + peek(1);
    const twoCharMap: Record<string, TokenKind> = {
      '==': TokenKind.EqEq,
      '!=': TokenKind.NotEq,
      '<=': TokenKind.LtEq,
      '>=': TokenKind.GtEq,
      '&&': TokenKind.AndAnd,
      '||': TokenKind.OrOr,
      '+=': TokenKind.PlusEq,
      '-=': TokenKind.MinusEq,
      '*=': TokenKind.StarEq,
      '/=': TokenKind.SlashEq,
      '..': TokenKind.DotDot,
      '->': TokenKind.Arrow,
    };

    if (twoCharMap[two]) {
      advance();
      advance();
      tokens.push({
        kind: twoCharMap[two],
        lexeme: two,
        span: { start, end: i, line: startLine, column: startCol },
      });
      continue;
    }

    // Single-character operators and punctuation
    const oneCharMap: Record<string, TokenKind> = {
      '+': TokenKind.Plus,
      '-': TokenKind.Minus,
      '*': TokenKind.Star,
      '/': TokenKind.Slash,
      '%': TokenKind.Percent,
      '@': TokenKind.At,
      '<': TokenKind.Lt,
      '>': TokenKind.Gt,
      '!': TokenKind.Bang,
      '=': TokenKind.Eq,
      '(': TokenKind.LParen,
      ')': TokenKind.RParen,
      '{': TokenKind.LBrace,
      '}': TokenKind.RBrace,
      '[': TokenKind.LBracket,
      ']': TokenKind.RBracket,
      ',': TokenKind.Comma,
      ':': TokenKind.Colon,
      ';': TokenKind.Semicolon,
      '.': TokenKind.Dot,
    };

    if (oneCharMap[ch]) {
      advance();
      tokens.push({
        kind: oneCharMap[ch],
        lexeme: ch,
        span: { start, end: i, line: startLine, column: startCol },
      });
      continue;
    }

    // Unknown character
    advance();
    diagnostics.push({
      stage: 'lexer',
      severity: 'error',
      code: 'E0007',
      message: `Unexpected character '${ch}'`,
      span: { start, end: i, line: startLine, column: startCol },
    });
  }

  tokens.push({
    kind: TokenKind.Eof,
    lexeme: '',
    span: { start: i, end: i, line, column: col },
  });

  return { tokens, diagnostics };
}
