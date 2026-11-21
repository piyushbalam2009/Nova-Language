import {
  AssignOp,
  BinaryOp,
  Expr,
  FnDecl,
  ForStmt,
  IfStmt,
  MapEntry,
  Param,
  Program,
  Stmt,
  StructDecl,
  StructFieldDecl,
  StructLiteralField,
  TypeAnnotation,
  UseStmt,
  VarDecl,
  WhileStmt,
} from './ast';
import { Diagnostic, Span } from './diagnostics';
import { Token, TokenKind } from './lexer';

export interface ParseResult {
  program: Program;
  diagnostics: Diagnostic[];
}

// Pratt binding powers (left, right)
function infixBindingPower(kind: TokenKind): [number, number] | null {
  switch (kind) {
    case TokenKind.OrOr: return [10, 11];
    case TokenKind.AndAnd: return [20, 21];
    case TokenKind.EqEq:
    case TokenKind.NotEq: return [30, 31];
    case TokenKind.Lt:
    case TokenKind.LtEq:
    case TokenKind.Gt:
    case TokenKind.GtEq: return [40, 41];
    case TokenKind.DotDot: return [45, 46];
    case TokenKind.Plus:
    case TokenKind.Minus: return [50, 51];
    case TokenKind.Star:
    case TokenKind.Slash:
    case TokenKind.Percent:
    case TokenKind.At: return [60, 61];
    default: return null;
  }
}

function tokenToBinaryOp(kind: TokenKind): BinaryOp {
  switch (kind) {
    case TokenKind.Plus: return '+';
    case TokenKind.Minus: return '-';
    case TokenKind.Star: return '*';
    case TokenKind.Slash: return '/';
    case TokenKind.Percent: return '%';
    case TokenKind.At: return '@';
    case TokenKind.EqEq: return '==';
    case TokenKind.NotEq: return '!=';
    case TokenKind.Lt: return '<';
    case TokenKind.LtEq: return '<=';
    case TokenKind.Gt: return '>';
    case TokenKind.GtEq: return '>=';
    case TokenKind.AndAnd: return '&&';
    case TokenKind.OrOr: return '||';
    default:
      throw new Error(`Unexpected binary operator token: ${kind}`);
  }
}

export function parse(tokens: Token[]): ParseResult {
  const diagnostics: Diagnostic[] = [];
  let pos = 0;

  const peek = (offset = 0): Token => {
    const idx = Math.min(pos + offset, tokens.length - 1);
    return tokens[idx];
  };

  const isAtEnd = (): boolean => peek().kind === TokenKind.Eof;

  const advance = (): Token => {
    const tok = peek();
    if (!isAtEnd()) pos++;
    return tok;
  };

  const skipNewlines = (): void => {
    while (peek().kind === TokenKind.Newline || peek().kind === TokenKind.Semicolon) {
      advance();
    }
  };

  const skipOnlyNewlines = (): void => {
    while (peek().kind === TokenKind.Newline) {
      advance();
    }
  };

  const error = (code: string, message: string, span: Span, hint?: string): void => {
    diagnostics.push({ stage: 'parser', severity: 'error', code, message, span, hint });
  };

  const expect = (kind: TokenKind, expectedDesc: string): Token | null => {
    if (peek().kind === kind) {
      return advance();
    }
    const found = peek();
    error(
      'E0101',
      `Expected ${expectedDesc}, found ${found.kind === TokenKind.Eof ? 'end of file' : `'${found.lexeme || found.kind}'`}`,
      found.span
    );
    return null;
  };

  const mergeSpan = (a: Span, b: Span): Span => ({
    start: a.start,
    end: b.end,
    line: a.line,
    column: a.column,
  });

  const synchronize = (): void => {
    while (!isAtEnd()) {
      if (peek().kind === TokenKind.Newline || peek().kind === TokenKind.Semicolon) {
        advance();
        return;
      }
      if (peek().kind === TokenKind.RBrace) {
        return;
      }
      const next = peek().kind;
      if (
        next === TokenKind.Fn ||
        next === TokenKind.Struct ||
        next === TokenKind.Let ||
        next === TokenKind.Mut ||
        next === TokenKind.Const ||
        next === TokenKind.If ||
        next === TokenKind.While ||
        next === TokenKind.For ||
        next === TokenKind.Return ||
        next === TokenKind.Use
      ) {
        return;
      }
      advance();
    }
  };

  const parseTypeAnnotation = (): TypeAnnotation => {
    const nameTok = peek();
    if (nameTok.kind !== TokenKind.Ident && nameTok.kind !== TokenKind.None) {
      error('E0102', `Expected type name, found '${nameTok.lexeme || nameTok.kind}'`, nameTok.span);
      return { kind: 'Simple', name: 'Unknown', span: nameTok.span };
    }
    advance();
    if (peek().kind === TokenKind.Lt) {
      advance(); // '<'
      const args: TypeAnnotation[] = [];
      if (peek().kind !== TokenKind.Gt) {
        args.push(parseTypeAnnotation());
        while (peek().kind === TokenKind.Comma) {
          advance();
          args.push(parseTypeAnnotation());
        }
      }
      const endTok = expect(TokenKind.Gt, "'>' to close generic type arguments") ?? nameTok;
      return {
        kind: 'Generic',
        name: nameTok.lexeme,
        args,
        span: mergeSpan(nameTok.span, endTok.span),
      };
    }
    return { kind: 'Simple', name: nameTok.lexeme, span: nameTok.span };
  };

  const parseBlock = (): Stmt[] => {
    skipOnlyNewlines();
    if (!expect(TokenKind.LBrace, "'{' to start block")) {
      return [];
    }
    const stmts: Stmt[] = [];
    skipNewlines();
    while (!isAtEnd() && peek().kind !== TokenKind.RBrace) {
      const stmt = parseStmt();
      if (stmt) stmts.push(stmt);
      skipNewlines();
    }
    expect(TokenKind.RBrace, "'}' to close block");
    return stmts;
  };

  const parseUseStmt = (): UseStmt => {
    const kw = advance(); // use
    const path: string[] = [];
    const first = expect(TokenKind.Ident, 'module identifier');
    let endSpan = kw.span;
    if (first) {
      path.push(first.lexeme);
      endSpan = first.span;
      while (peek().kind === TokenKind.Dot) {
        advance();
        const seg = expect(TokenKind.Ident, 'submodule identifier');
        if (seg) {
          path.push(seg.lexeme);
          endSpan = seg.span;
        } else {
          break;
        }
      }
    }
    return { kind: 'UseStmt', path, span: mergeSpan(kw.span, endSpan) };
  };

  const parseFnDecl = (): FnDecl => {
    const kw = advance(); // fn
    const nameTok = expect(TokenKind.Ident, 'function name');
    const name = nameTok?.lexeme ?? '<anonymous>';
    expect(TokenKind.LParen, "'(' after function name");

    const params: Param[] = [];
    skipOnlyNewlines();
    if (peek().kind !== TokenKind.RParen && !isAtEnd()) {
      while (true) {
        skipOnlyNewlines();
        const pName = expect(TokenKind.Ident, 'parameter name');
        if (!pName) break;
        expect(TokenKind.Colon, "':' after parameter name");
        const pType = parseTypeAnnotation();
        params.push({
          name: pName.lexeme,
          typeAnnotation: pType,
          span: mergeSpan(pName.span, pType.span),
        });
        skipOnlyNewlines();
        if (peek().kind === TokenKind.Comma) {
          advance();
          skipOnlyNewlines();
          if (peek().kind === TokenKind.RParen) break;
        } else {
          break;
        }
      }
    }
    expect(TokenKind.RParen, "')' after parameters");

    let returnType: TypeAnnotation | undefined;
    if (peek().kind === TokenKind.Arrow) {
      advance();
      returnType = parseTypeAnnotation();
    }

    const body = parseBlock();
    const endSpan = body.length > 0 ? body[body.length - 1].span : (returnType?.span ?? kw.span);
    return {
      kind: 'FnDecl',
      name,
      params,
      returnType,
      body,
      span: mergeSpan(kw.span, endSpan),
    };
  };

  const parseStructDecl = (): StructDecl => {
    const kw = advance(); // struct
    const nameTok = expect(TokenKind.Ident, 'struct name');
    const name = nameTok?.lexeme ?? '<anonymous>';
    skipOnlyNewlines();
    expect(TokenKind.LBrace, "'{' after struct name");
    const fields: StructFieldDecl[] = [];
    skipNewlines();
    while (!isAtEnd() && peek().kind !== TokenKind.RBrace) {
      const fName = expect(TokenKind.Ident, 'struct field name');
      if (!fName) {
        synchronize();
        skipNewlines();
        continue;
      }
      expect(TokenKind.Colon, "':' after struct field name");
      const fType = parseTypeAnnotation();
      fields.push({
        name: fName.lexeme,
        typeAnnotation: fType,
        span: mergeSpan(fName.span, fType.span),
      });
      if (peek().kind === TokenKind.Comma) advance();
      skipNewlines();
    }
    const closeTok = expect(TokenKind.RBrace, "'}' to close struct") ?? kw;
    return {
      kind: 'StructDecl',
      name,
      fields,
      span: mergeSpan(kw.span, closeTok.span),
    };
  };

  const parseVarDecl = (): VarDecl => {
    const kw = advance();
    const mutability = kw.kind === TokenKind.Mut ? 'mut' : kw.kind === TokenKind.Const ? 'const' : 'let';
    const nameTok = expect(TokenKind.Ident, 'variable name');
    const name = nameTok?.lexeme ?? '_';
    let typeAnnotation: TypeAnnotation | undefined;
    if (peek().kind === TokenKind.Colon) {
      advance();
      typeAnnotation = parseTypeAnnotation();
    }
    expect(TokenKind.Eq, "'=' in variable declaration");
    const initializer = parseExpr(0);
    return {
      kind: 'VarDecl',
      mutability,
      name,
      typeAnnotation,
      initializer,
      span: mergeSpan(kw.span, initializer.span),
    };
  };

  const parseIfStmt = (): IfStmt => {
    const kw = advance(); // if
    const condition = parseExpr(0, true);
    const thenBranch = parseBlock();
    let elseBranch: Stmt[] | IfStmt | undefined;
    skipOnlyNewlines();
    if (peek().kind === TokenKind.Else) {
      advance();
      skipOnlyNewlines();
      if (peek().kind === TokenKind.If) {
        elseBranch = parseIfStmt();
      } else {
        elseBranch = parseBlock();
      }
    }
    return {
      kind: 'IfStmt',
      condition,
      thenBranch,
      elseBranch,
      span: mergeSpan(kw.span, condition.span),
    };
  };

  const parseWhileStmt = (): WhileStmt => {
    const kw = advance(); // while
    const condition = parseExpr(0, true);
    const body = parseBlock();
    return {
      kind: 'WhileStmt',
      condition,
      body,
      span: mergeSpan(kw.span, condition.span),
    };
  };

  const parseForStmt = (): ForStmt => {
    const kw = advance(); // for
    const varTok = expect(TokenKind.Ident, 'loop variable name');
    expect(TokenKind.In, "'in' keyword in for loop");
    const iterable = parseExpr(0, true);
    const body = parseBlock();
    return {
      kind: 'ForStmt',
      variable: varTok?.lexeme ?? '_',
      iterable,
      body,
      span: mergeSpan(kw.span, iterable.span),
    };
  };

  const parseReturnStmt = (): Stmt => {
    const kw = advance(); // return
    if (
      peek().kind === TokenKind.Newline ||
      peek().kind === TokenKind.Semicolon ||
      peek().kind === TokenKind.RBrace ||
      peek().kind === TokenKind.Eof
    ) {
      return { kind: 'ReturnStmt', span: kw.span };
    }
    const value = parseExpr(0);
    return { kind: 'ReturnStmt', value, span: mergeSpan(kw.span, value.span) };
  };

  const parseStmt = (): Stmt | null => {
    skipNewlines();
    const tok = peek();
    switch (tok.kind) {
      case TokenKind.Use: return parseUseStmt();
      case TokenKind.Fn: return parseFnDecl();
      case TokenKind.Struct: return parseStructDecl();
      case TokenKind.Let:
      case TokenKind.Mut:
      case TokenKind.Const: return parseVarDecl();
      case TokenKind.If: return parseIfStmt();
      case TokenKind.While: return parseWhileStmt();
      case TokenKind.For: return parseForStmt();
      case TokenKind.Return: return parseReturnStmt();
      case TokenKind.Break: {
        advance();
        return { kind: 'BreakStmt', span: tok.span };
      }
      case TokenKind.Continue: {
        advance();
        return { kind: 'ContinueStmt', span: tok.span };
      }
      case TokenKind.Eof:
        return null;
      default: {
        const expr = parseExpr(0);
        const assignKinds: Partial<Record<TokenKind, AssignOp>> = {
          [TokenKind.Eq]: '=',
          [TokenKind.PlusEq]: '+=',
          [TokenKind.MinusEq]: '-=',
          [TokenKind.StarEq]: '*=',
          [TokenKind.SlashEq]: '/=',
        };
        const assignOp = assignKinds[peek().kind];
        if (assignOp) {
          advance();
          const val = parseExpr(0);
          return {
            kind: 'AssignStmt',
            target: expr,
            op: assignOp,
            value: val,
            span: mergeSpan(expr.span, val.span),
          };
        }
        return { kind: 'ExprStmt', expr, span: expr.span };
      }
    }
  };

  // Pratt expression parser. `noStructLiteral` avoids ambiguity in `if cond { ... }`.
  const parseExpr = (minBp: number, noStructLiteral = false): Expr => {
    let lhs = parsePrefix(noStructLiteral);

    // Postfix loop: calls `(...)`, field access `.foo`, indexing `[i]`
    const parsePostfix = (base: Expr): Expr => {
      let expr = base;
      while (true) {
        if (peek().kind === TokenKind.LParen) {
          advance();
          const args: Expr[] = [];
          skipOnlyNewlines();
          if (peek().kind !== TokenKind.RParen && !isAtEnd()) {
            args.push(parseExpr(0));
            skipOnlyNewlines();
            while (peek().kind === TokenKind.Comma) {
              advance();
              skipOnlyNewlines();
              if (peek().kind === TokenKind.RParen) break;
              args.push(parseExpr(0));
              skipOnlyNewlines();
            }
          }
          const close = expect(TokenKind.RParen, "')' after arguments") ?? peek();
          expr = { kind: 'CallExpr', callee: expr, args, span: mergeSpan(expr.span, close.span) };
        } else if (peek().kind === TokenKind.Dot) {
          advance();
          const fieldTok = expect(TokenKind.Ident, 'field or method name');
          const fieldName = fieldTok?.lexeme ?? '';
          expr = {
            kind: 'FieldExpr',
            object: expr,
            field: fieldName,
            span: mergeSpan(expr.span, fieldTok?.span ?? expr.span),
          };
        } else if (peek().kind === TokenKind.LBracket) {
          advance();
          const index = parseExpr(0);
          const close = expect(TokenKind.RBracket, "']' after index") ?? peek();
          expr = { kind: 'IndexExpr', object: expr, index, span: mergeSpan(expr.span, close.span) };
        } else {
          break;
        }
      }
      return expr;
    };

    lhs = parsePostfix(lhs);

    while (true) {
      const opTok = peek();
      const bp = infixBindingPower(opTok.kind);
      if (!bp) break;
      const [lBp, rBp] = bp;
      if (lBp < minBp) break;

      advance();
      const rhs = parseExpr(rBp, noStructLiteral);
      if (opTok.kind === TokenKind.DotDot) {
        lhs = {
          kind: 'RangeExpr',
          start: lhs,
          end: rhs,
          span: mergeSpan(lhs.span, rhs.span),
        };
      } else {
        lhs = {
          kind: 'BinaryExpr',
          op: tokenToBinaryOp(opTok.kind),
          left: lhs,
          right: rhs,
          span: mergeSpan(lhs.span, rhs.span),
        };
      }
    }

    return lhs;
  };

  const parsePrefix = (noStructLiteral: boolean): Expr => {
    const tok = peek();
    switch (tok.kind) {
      case TokenKind.Int: {
        advance();
        return { kind: 'IntLiteral', value: Number.parseInt(tok.lexeme, 10), span: tok.span };
      }
      case TokenKind.Float: {
        advance();
        return { kind: 'FloatLiteral', value: Number.parseFloat(tok.lexeme), span: tok.span };
      }
      case TokenKind.String: {
        advance();
        return { kind: 'StringLiteral', value: tok.lexeme, span: tok.span };
      }
      case TokenKind.Char: {
        advance();
        return { kind: 'CharLiteral', value: tok.lexeme, span: tok.span };
      }
      case TokenKind.True: {
        advance();
        return { kind: 'BoolLiteral', value: true, span: tok.span };
      }
      case TokenKind.False: {
        advance();
        return { kind: 'BoolLiteral', value: false, span: tok.span };
      }
      case TokenKind.None: {
        advance();
        return { kind: 'NoneLiteral', span: tok.span };
      }
      case TokenKind.Ok:
      case TokenKind.Error: {
        const variant = tok.kind === TokenKind.Ok ? 'Ok' : 'Error';
        advance();
        expect(TokenKind.LParen, `'(' after ${variant}`);
        const value = parseExpr(0);
        const close = expect(TokenKind.RParen, `')' after ${variant} payload`) ?? tok;
        return { kind: 'ResultExpr', variant, value, span: mergeSpan(tok.span, close.span) };
      }
      case TokenKind.Minus:
      case TokenKind.Bang: {
        advance();
        const operand = parseExpr(70, noStructLiteral);
        return {
          kind: 'UnaryExpr',
          op: tok.kind === TokenKind.Minus ? '-' : '!',
          operand,
          span: mergeSpan(tok.span, operand.span),
        };
      }
      case TokenKind.LParen: {
        advance();
        const inner = parseExpr(0);
        expect(TokenKind.RParen, "')' to close grouped expression");
        return inner;
      }
      case TokenKind.LBracket: {
        advance();
        const elements: Expr[] = [];
        skipOnlyNewlines();
        if (peek().kind !== TokenKind.RBracket && !isAtEnd()) {
          elements.push(parseExpr(0));
          skipOnlyNewlines();
          while (peek().kind === TokenKind.Comma) {
            advance();
            skipOnlyNewlines();
            if (peek().kind === TokenKind.RBracket) break;
            elements.push(parseExpr(0));
            skipOnlyNewlines();
          }
        }
        const close = expect(TokenKind.RBracket, "']' to close array literal") ?? tok;
        return { kind: 'ArrayLiteral', elements, span: mergeSpan(tok.span, close.span) };
      }
      case TokenKind.LBrace: {
        // Map literal: { "key": value, ... }
        advance();
        const entries: MapEntry[] = [];
        skipOnlyNewlines();
        if (peek().kind !== TokenKind.RBrace && !isAtEnd()) {
          while (true) {
            skipOnlyNewlines();
            if (peek().kind === TokenKind.RBrace) break;
            const key = parseExpr(0);
            expect(TokenKind.Colon, "':' in map entry");
            const value = parseExpr(0);
            entries.push({ key, value });
            skipOnlyNewlines();
            if (peek().kind === TokenKind.Comma) {
              advance();
              skipOnlyNewlines();
            } else {
              break;
            }
          }
        }
        const close = expect(TokenKind.RBrace, "'}' to close map literal") ?? tok;
        return { kind: 'MapLiteral', entries, span: mergeSpan(tok.span, close.span) };
      }
      case TokenKind.Ident: {
        advance();
        // Check for struct literal: CapitalizedIdent { field: value, ... }
        const isCapitalized = tok.lexeme.length > 0 && tok.lexeme[0] >= 'A' && tok.lexeme[0] <= 'Z';
        if (!noStructLiteral && isCapitalized && peek().kind === TokenKind.LBrace) {
          advance(); // '{'
          const fields: StructLiteralField[] = [];
          skipOnlyNewlines();
          while (!isAtEnd() && peek().kind !== TokenKind.RBrace) {
            const fName = expect(TokenKind.Ident, 'struct literal field name');
            if (!fName) {
              synchronize();
              break;
            }
            expect(TokenKind.Colon, "':' after field name in struct literal");
            const fVal = parseExpr(0);
            fields.push({
              name: fName.lexeme,
              value: fVal,
              span: mergeSpan(fName.span, fVal.span),
            });
            skipOnlyNewlines();
            if (peek().kind === TokenKind.Comma) {
              advance();
              skipOnlyNewlines();
            }
          }
          const close = expect(TokenKind.RBrace, "'}' to close struct literal") ?? tok;
          return {
            kind: 'StructLiteral',
            name: tok.lexeme,
            fields,
            span: mergeSpan(tok.span, close.span),
          };
        }
        return { kind: 'IdentExpr', name: tok.lexeme, span: tok.span };
      }
      default: {
        error(
          'E0103',
          `Unexpected token '${tok.lexeme || tok.kind}' while parsing expression`,
          tok.span
        );
        if (!isAtEnd()) advance();
        return { kind: 'NoneLiteral', span: tok.span };
      }
    }
  };

  const items: Stmt[] = [];
  skipNewlines();
  while (!isAtEnd()) {
    const startPos = pos;
    const stmt = parseStmt();
    if (stmt) {
      items.push(stmt);
    }
    if (pos === startPos && !isAtEnd()) {
      synchronize();
    }
    skipNewlines();
  }

  const startSpan = tokens[0]?.span ?? { start: 0, end: 0, line: 1, column: 1 };
  const endSpan = tokens[tokens.length - 1]?.span ?? startSpan;

  return {
    program: {
      kind: 'Program',
      items,
      span: mergeSpan(startSpan, endSpan),
    },
    diagnostics,
  };
}
