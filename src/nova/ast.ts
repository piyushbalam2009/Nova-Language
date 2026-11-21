import { Span } from './diagnostics';

export type TypeAnnotation =
  | { kind: 'Simple'; name: string; span: Span }
  | { kind: 'Generic'; name: string; args: TypeAnnotation[]; span: Span };

export interface Program {
  kind: 'Program';
  items: Stmt[];
  span: Span;
}

export type Stmt =
  | UseStmt
  | FnDecl
  | StructDecl
  | VarDecl
  | AssignStmt
  | IfStmt
  | WhileStmt
  | ForStmt
  | ReturnStmt
  | BreakStmt
  | ContinueStmt
  | ExprStmt;

export interface UseStmt {
  kind: 'UseStmt';
  path: string[];
  span: Span;
}

export interface Param {
  name: string;
  typeAnnotation: TypeAnnotation;
  span: Span;
}

export interface FnDecl {
  kind: 'FnDecl';
  name: string;
  params: Param[];
  returnType?: TypeAnnotation;
  body: Stmt[];
  span: Span;
}

export interface StructFieldDecl {
  name: string;
  typeAnnotation: TypeAnnotation;
  span: Span;
}

export interface StructDecl {
  kind: 'StructDecl';
  name: string;
  fields: StructFieldDecl[];
  span: Span;
}

export interface VarDecl {
  kind: 'VarDecl';
  mutability: 'let' | 'mut' | 'const';
  name: string;
  typeAnnotation?: TypeAnnotation;
  initializer: Expr;
  span: Span;
}

export type AssignOp = '=' | '+=' | '-=' | '*=' | '/=';

export interface AssignStmt {
  kind: 'AssignStmt';
  target: Expr; // IdentExpr, FieldExpr, or IndexExpr
  op: AssignOp;
  value: Expr;
  span: Span;
}

export interface IfStmt {
  kind: 'IfStmt';
  condition: Expr;
  thenBranch: Stmt[];
  elseBranch?: Stmt[] | IfStmt;
  span: Span;
}

export interface WhileStmt {
  kind: 'WhileStmt';
  condition: Expr;
  body: Stmt[];
  span: Span;
}

export interface ForStmt {
  kind: 'ForStmt';
  variable: string;
  iterable: Expr; // RangeExpr or ArrayExpr
  body: Stmt[];
  span: Span;
}

export interface ReturnStmt {
  kind: 'ReturnStmt';
  value?: Expr;
  span: Span;
}

export interface BreakStmt {
  kind: 'BreakStmt';
  span: Span;
}

export interface ContinueStmt {
  kind: 'ContinueStmt';
  span: Span;
}

export interface ExprStmt {
  kind: 'ExprStmt';
  expr: Expr;
  span: Span;
}

export type Expr =
  | IntLiteral
  | FloatLiteral
  | StringLiteral
  | CharLiteral
  | BoolLiteral
  | NoneLiteral
  | IdentExpr
  | UnaryExpr
  | BinaryExpr
  | RangeExpr
  | CallExpr
  | FieldExpr
  | IndexExpr
  | ArrayLiteral
  | MapLiteral
  | StructLiteral
  | ResultExpr;

export interface IntLiteral {
  kind: 'IntLiteral';
  value: number;
  span: Span;
}

export interface FloatLiteral {
  kind: 'FloatLiteral';
  value: number;
  span: Span;
}

export interface StringLiteral {
  kind: 'StringLiteral';
  value: string;
  span: Span;
}

export interface CharLiteral {
  kind: 'CharLiteral';
  value: string;
  span: Span;
}

export interface BoolLiteral {
  kind: 'BoolLiteral';
  value: boolean;
  span: Span;
}

export interface NoneLiteral {
  kind: 'NoneLiteral';
  span: Span;
}

export interface IdentExpr {
  kind: 'IdentExpr';
  name: string;
  span: Span;
}

export interface UnaryExpr {
  kind: 'UnaryExpr';
  op: '-' | '!';
  operand: Expr;
  span: Span;
}

export type BinaryOp =
  | '+' | '-' | '*' | '/' | '%' | '@'
  | '==' | '!=' | '<' | '<=' | '>' | '>='
  | '&&' | '||';

export interface BinaryExpr {
  kind: 'BinaryExpr';
  op: BinaryOp;
  left: Expr;
  right: Expr;
  span: Span;
}

export interface RangeExpr {
  kind: 'RangeExpr';
  start: Expr;
  end: Expr;
  span: Span;
}

export interface CallExpr {
  kind: 'CallExpr';
  callee: Expr;
  args: Expr[];
  span: Span;
}

export interface FieldExpr {
  kind: 'FieldExpr';
  object: Expr;
  field: string;
  span: Span;
}

export interface IndexExpr {
  kind: 'IndexExpr';
  object: Expr;
  index: Expr;
  span: Span;
}

export interface ArrayLiteral {
  kind: 'ArrayLiteral';
  elements: Expr[];
  span: Span;
}

export interface MapEntry {
  key: Expr;
  value: Expr;
}

export interface MapLiteral {
  kind: 'MapLiteral';
  entries: MapEntry[];
  span: Span;
}

export interface StructLiteralField {
  name: string;
  value: Expr;
  span: Span;
}

export interface StructLiteral {
  kind: 'StructLiteral';
  name: string;
  fields: StructLiteralField[];
  span: Span;
}

export interface ResultExpr {
  kind: 'ResultExpr';
  variant: 'Ok' | 'Error';
  value: Expr;
  span: Span;
}

export function formatTypeAnnotation(t: TypeAnnotation): string {
  if (t.kind === 'Simple') return t.name;
  return `${t.name}<${t.args.map(formatTypeAnnotation).join(', ')}>`;
}
