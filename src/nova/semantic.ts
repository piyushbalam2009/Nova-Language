import {
  Expr,
  FnDecl,
  Program,
  Stmt,
  StructDecl,
  TypeAnnotation,
} from './ast';
import { Diagnostic, Span } from './diagnostics';

export type NovaType =
  | { tag: 'Int' }
  | { tag: 'Float' }
  | { tag: 'Bool' }
  | { tag: 'String' }
  | { tag: 'Char' }
  | { tag: 'Void' }
  | { tag: 'None' }
  | { tag: 'Any' }
  | { tag: 'Range' }
  | { tag: 'Array'; element: NovaType }
  | { tag: 'Map'; key: NovaType; value: NovaType }
  | { tag: 'Tensor'; element: NovaType }
  | { tag: 'Option'; inner: NovaType }
  | { tag: 'Result'; ok: NovaType; err: NovaType }
  | { tag: 'Struct'; name: string }
  | { tag: 'Module'; name: string }
  | { tag: 'Fn'; params: NovaType[]; ret: NovaType; variadic?: boolean };

export const T_INT: NovaType = { tag: 'Int' };
export const T_FLOAT: NovaType = { tag: 'Float' };
export const T_BOOL: NovaType = { tag: 'Bool' };
export const T_STRING: NovaType = { tag: 'String' };
export const T_CHAR: NovaType = { tag: 'Char' };
export const T_VOID: NovaType = { tag: 'Void' };
export const T_NONE: NovaType = { tag: 'None' };
export const T_ANY: NovaType = { tag: 'Any' };
export const T_RANGE: NovaType = { tag: 'Range' };
export const T_TENSOR_FLOAT: NovaType = { tag: 'Tensor', element: T_FLOAT };

export function formatType(t: NovaType): string {
  switch (t.tag) {
    case 'Array': return `Array<${formatType(t.element)}>`;
    case 'Map': return `Map<${formatType(t.key)}, ${formatType(t.value)}>`;
    case 'Tensor': return `Tensor<${formatType(t.element)}>`;
    case 'Option': return `Option<${formatType(t.inner)}>`;
    case 'Result': return `Result<${formatType(t.ok)}, ${formatType(t.err)}>`;
    case 'Struct': return t.name;
    case 'Module': return `module(${t.name})`;
    case 'Fn': return `fn(${t.params.map(formatType).join(', ')}) -> ${formatType(t.ret)}`;
    default: return t.tag;
  }
}

export function typesCompatible(expected: NovaType, actual: NovaType): boolean {
  if (expected.tag === 'Any' || actual.tag === 'Any') return true;
  if (expected.tag === 'Option' && actual.tag === 'None') return true;
  if (expected.tag !== actual.tag) return false;
  switch (expected.tag) {
    case 'Array':
      return typesCompatible(expected.element, (actual as { tag: 'Array'; element: NovaType }).element);
    case 'Map': {
      const a = actual as { tag: 'Map'; key: NovaType; value: NovaType };
      return typesCompatible(expected.key, a.key) && typesCompatible(expected.value, a.value);
    }
    case 'Tensor':
      return typesCompatible(expected.element, (actual as { tag: 'Tensor'; element: NovaType }).element);
    case 'Option':
      return typesCompatible(expected.inner, (actual as { tag: 'Option'; inner: NovaType }).inner);
    case 'Result': {
      const a = actual as { tag: 'Result'; ok: NovaType; err: NovaType };
      return typesCompatible(expected.ok, a.ok) && typesCompatible(expected.err, a.err);
    }
    case 'Struct':
      return expected.name === (actual as { tag: 'Struct'; name: string }).name;
    default:
      return true;
  }
}

export interface SymbolInfo {
  name: string;
  type: NovaType;
  mutability: 'let' | 'mut' | 'const' | 'fn' | 'param' | 'module';
  span?: Span;
}

export interface StructInfo {
  name: string;
  fields: Map<string, NovaType>;
  fieldOrder: string[];
  span: Span;
}

export interface SemanticResult {
  diagnostics: Diagnostic[];
  symbols: SymbolInfo[];
  structs: Map<string, StructInfo>;
  functions: Map<string, { params: NovaType[]; ret: NovaType; variadic?: boolean }>;
  imports: string[];
}

const KNOWN_MODULES = new Set([
  'nova.io',
  'nova.math',
  'nova.string',
  'nova.array',
  'nova.json',
  'nova.time',
  'nova.ai',
  'nova.nn',
]);

const MODULE_EXPORTS: Record<string, Record<string, { params: NovaType[]; ret: NovaType; variadic?: boolean }>> = {
  'nova.io': {
    print: { params: [T_ANY], ret: T_VOID, variadic: true },
  },
  'nova.math': {
    sqrt: { params: [T_FLOAT], ret: T_FLOAT },
    abs: { params: [T_FLOAT], ret: T_FLOAT },
    pow: { params: [T_FLOAT, T_FLOAT], ret: T_FLOAT },
    sin: { params: [T_FLOAT], ret: T_FLOAT },
    cos: { params: [T_FLOAT], ret: T_FLOAT },
    exp: { params: [T_FLOAT], ret: T_FLOAT },
    log: { params: [T_FLOAT], ret: T_FLOAT },
    floor: { params: [T_FLOAT], ret: T_INT },
    ceil: { params: [T_FLOAT], ret: T_INT },
    min: { params: [T_FLOAT, T_FLOAT], ret: T_FLOAT },
    max: { params: [T_FLOAT, T_FLOAT], ret: T_FLOAT },
  },
  'nova.string': {
    len: { params: [T_STRING], ret: T_INT },
    upper: { params: [T_STRING], ret: T_STRING },
    lower: { params: [T_STRING], ret: T_STRING },
    trim: { params: [T_STRING], ret: T_STRING },
    contains: { params: [T_STRING, T_STRING], ret: T_BOOL },
    split: { params: [T_STRING, T_STRING], ret: { tag: 'Array', element: T_STRING } },
  },
  'nova.array': {
    len: { params: [{ tag: 'Array', element: T_ANY }], ret: T_INT },
    push: { params: [{ tag: 'Array', element: T_ANY }, T_ANY], ret: T_VOID },
    sum: { params: [{ tag: 'Array', element: T_ANY }], ret: T_FLOAT },
    range: { params: [T_INT, T_INT], ret: { tag: 'Array', element: T_INT } },
  },
  'nova.json': {
    stringify: { params: [T_ANY], ret: T_STRING },
    parse: { params: [T_STRING], ret: T_ANY },
  },
  'nova.time': {
    now_ms: { params: [], ret: T_INT },
  },
  'nova.ai': {
    tensor: { params: [T_ANY], ret: T_TENSOR_FLOAT },
    zeros: { params: [{ tag: 'Array', element: T_INT }], ret: T_TENSOR_FLOAT },
    ones: { params: [{ tag: 'Array', element: T_INT }], ret: T_TENSOR_FLOAT },
    full: { params: [{ tag: 'Array', element: T_INT }, T_FLOAT], ret: T_TENSOR_FLOAT },
    randn: { params: [{ tag: 'Array', element: T_INT }, T_INT], ret: T_TENSOR_FLOAT },
    matmul: { params: [T_TENSOR_FLOAT, T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    dot: { params: [T_TENSOR_FLOAT, T_TENSOR_FLOAT], ret: T_FLOAT },
    transpose: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    reshape: { params: [T_TENSOR_FLOAT, { tag: 'Array', element: T_INT }], ret: T_TENSOR_FLOAT },
    relu: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    sigmoid: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    tanh: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    softmax: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    mse_loss: { params: [T_TENSOR_FLOAT, T_TENSOR_FLOAT], ret: T_FLOAT },
    bce_loss: { params: [T_TENSOR_FLOAT, T_TENSOR_FLOAT], ret: T_FLOAT },
    grad_var: { params: [T_ANY], ret: T_ANY },
    grad_add: { params: [T_ANY, T_ANY], ret: T_ANY },
    grad_sub: { params: [T_ANY, T_ANY], ret: T_ANY },
    grad_mul: { params: [T_ANY, T_ANY], ret: T_ANY },
    grad_matmul: { params: [T_ANY, T_ANY], ret: T_ANY },
    grad_relu: { params: [T_ANY], ret: T_ANY },
    grad_sigmoid: { params: [T_ANY], ret: T_ANY },
    grad_tanh: { params: [T_ANY], ret: T_ANY },
    grad_mse: { params: [T_ANY, T_ANY], ret: T_ANY },
    backward: { params: [T_ANY], ret: T_VOID },
    grad_of: { params: [T_ANY], ret: T_TENSOR_FLOAT },
    value_of: { params: [T_ANY], ret: T_TENSOR_FLOAT },
    zero_grad: { params: [T_ANY], ret: T_VOID },
    train_xor: { params: [T_INT, T_FLOAT], ret: T_ANY },
    autograd_demo: { params: [T_FLOAT, T_FLOAT], ret: T_ANY },
  },
  'nova.nn': {
    relu: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    sigmoid: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    tanh: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    softmax: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    mse_loss: { params: [T_TENSOR_FLOAT, T_TENSOR_FLOAT], ret: T_FLOAT },
    bce_loss: { params: [T_TENSOR_FLOAT, T_TENSOR_FLOAT], ret: T_FLOAT },
    model_new: { params: [], ret: T_ANY },
    model_add_linear: { params: [T_ANY, T_INT, T_INT, T_INT], ret: T_ANY },
    model_add_activation: { params: [T_ANY, T_STRING], ret: T_ANY },
    model_forward: { params: [T_ANY, T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    model_param_count: { params: [T_ANY], ret: T_INT },
    optimizer_sgd: { params: [T_FLOAT, T_FLOAT], ret: T_ANY },
    optimizer_adam: { params: [T_FLOAT], ret: T_ANY },
    train_step: { params: [T_ANY, T_ANY, T_TENSOR_FLOAT, T_TENSOR_FLOAT, T_STRING], ret: T_FLOAT },
    train_model: { params: [T_ANY, T_ANY, T_TENSOR_FLOAT, T_TENSOR_FLOAT, T_INT, T_STRING], ret: T_ANY },
    train_xor: { params: [T_INT, T_FLOAT], ret: T_ANY },
  },
};

class Scope {
  private entries = new Map<string, SymbolInfo>();
  constructor(public parent: Scope | null = null) {}

  define(sym: SymbolInfo): boolean {
    if (this.entries.has(sym.name)) return false;
    this.entries.set(sym.name, sym);
    return true;
  }

  override(sym: SymbolInfo): void {
    this.entries.set(sym.name, sym);
  }

  lookup(name: string): SymbolInfo | undefined {
    const local = this.entries.get(name);
    if (local) return local;
    return this.parent?.lookup(name);
  }
}

export function analyze(program: Program): SemanticResult {
  const diagnostics: Diagnostic[] = [];
  const structs = new Map<string, StructInfo>();
  const functions = new Map<string, { params: NovaType[]; ret: NovaType; variadic?: boolean }>();
  const recordedSymbols: SymbolInfo[] = [];
  const imports: string[] = [];

  const globalScope = new Scope(null);

  // Built-in functions available everywhere
  const builtins: Record<string, { params: NovaType[]; ret: NovaType; variadic?: boolean }> = {
    print: { params: [T_ANY], ret: T_VOID, variadic: true },
    str: { params: [T_ANY], ret: T_STRING },
    int: { params: [T_ANY], ret: T_INT },
    float: { params: [T_ANY], ret: T_FLOAT },
    len: { params: [T_ANY], ret: T_INT },
    type_of: { params: [T_ANY], ret: T_STRING },
    is_ok: { params: [T_ANY], ret: T_BOOL },
    is_error: { params: [T_ANY], ret: T_BOOL },
    unwrap: { params: [T_ANY], ret: T_ANY },
    sqrt: { params: [T_ANY], ret: T_FLOAT },
    abs: { params: [T_ANY], ret: T_FLOAT },
    tensor: { params: [T_ANY], ret: T_TENSOR_FLOAT },
    zeros: { params: [{ tag: 'Array', element: T_INT }], ret: T_TENSOR_FLOAT },
    ones: { params: [{ tag: 'Array', element: T_INT }], ret: T_TENSOR_FLOAT },
    full: { params: [{ tag: 'Array', element: T_INT }, T_FLOAT], ret: T_TENSOR_FLOAT },
    randn: { params: [{ tag: 'Array', element: T_INT }, T_INT], ret: T_TENSOR_FLOAT },
    matmul: { params: [T_TENSOR_FLOAT, T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    dot: { params: [T_TENSOR_FLOAT, T_TENSOR_FLOAT], ret: T_FLOAT },
    relu: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    sigmoid: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    tanh: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    softmax: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    transpose: { params: [T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    reshape: { params: [T_TENSOR_FLOAT, { tag: 'Array', element: T_INT }], ret: T_TENSOR_FLOAT },
    mse_loss: { params: [T_TENSOR_FLOAT, T_TENSOR_FLOAT], ret: T_FLOAT },
    bce_loss: { params: [T_TENSOR_FLOAT, T_TENSOR_FLOAT], ret: T_FLOAT },
    grad_var: { params: [T_ANY], ret: T_ANY },
    grad_add: { params: [T_ANY, T_ANY], ret: T_ANY },
    grad_sub: { params: [T_ANY, T_ANY], ret: T_ANY },
    grad_mul: { params: [T_ANY, T_ANY], ret: T_ANY },
    grad_matmul: { params: [T_ANY, T_ANY], ret: T_ANY },
    grad_relu: { params: [T_ANY], ret: T_ANY },
    grad_sigmoid: { params: [T_ANY], ret: T_ANY },
    grad_tanh: { params: [T_ANY], ret: T_ANY },
    grad_mse: { params: [T_ANY, T_ANY], ret: T_ANY },
    backward: { params: [T_ANY], ret: T_VOID },
    grad_of: { params: [T_ANY], ret: T_TENSOR_FLOAT },
    value_of: { params: [T_ANY], ret: T_TENSOR_FLOAT },
    zero_grad: { params: [T_ANY], ret: T_VOID },
    model_new: { params: [], ret: T_ANY },
    model_add_linear: { params: [T_ANY, T_INT, T_INT, T_INT], ret: T_ANY },
    model_add_activation: { params: [T_ANY, T_STRING], ret: T_ANY },
    model_forward: { params: [T_ANY, T_TENSOR_FLOAT], ret: T_TENSOR_FLOAT },
    model_param_count: { params: [T_ANY], ret: T_INT },
    optimizer_sgd: { params: [T_FLOAT, T_FLOAT], ret: T_ANY },
    optimizer_adam: { params: [T_FLOAT], ret: T_ANY },
    train_step: { params: [T_ANY, T_ANY, T_TENSOR_FLOAT, T_TENSOR_FLOAT, T_STRING], ret: T_FLOAT },
    train_model: { params: [T_ANY, T_ANY, T_TENSOR_FLOAT, T_TENSOR_FLOAT, T_INT, T_STRING], ret: T_ANY },
    train_xor: { params: [T_INT, T_FLOAT], ret: T_ANY },
    autograd_demo: { params: [T_FLOAT, T_FLOAT], ret: T_ANY },
  };

  for (const [name, sig] of Object.entries(builtins)) {
    functions.set(name, sig);
    globalScope.define({
      name,
      type: { tag: 'Fn', params: sig.params, ret: sig.ret, variadic: sig.variadic },
      mutability: 'fn',
    });
  }

  const error = (code: string, message: string, span: Span, hint?: string): void => {
    diagnostics.push({ stage: 'semantic', severity: 'error', code, message, span, hint });
  };

  const resolveTypeAnnotation = (ann: TypeAnnotation): NovaType => {
    if (ann.kind === 'Simple') {
      switch (ann.name) {
        case 'Int': return T_INT;
        case 'Float': return T_FLOAT;
        case 'Bool': return T_BOOL;
        case 'String': return T_STRING;
        case 'Char': return T_CHAR;
        case 'Void': return T_VOID;
        case 'None': return T_NONE;
        case 'Any': return T_ANY;
        default: {
          if (structs.has(ann.name)) {
            return { tag: 'Struct', name: ann.name };
          }
          error('E0201', `Unknown type '${ann.name}'`, ann.span);
          return T_ANY;
        }
      }
    }
    const args = ann.args.map(resolveTypeAnnotation);
    switch (ann.name) {
      case 'Array':
        return { tag: 'Array', element: args[0] ?? T_ANY };
      case 'Map':
        return { tag: 'Map', key: args[0] ?? T_ANY, value: args[1] ?? T_ANY };
      case 'Tensor':
        return { tag: 'Tensor', element: args[0] ?? T_FLOAT };
      case 'Option':
        return { tag: 'Option', inner: args[0] ?? T_ANY };
      case 'Result':
        return { tag: 'Result', ok: args[0] ?? T_ANY, err: args[1] ?? T_STRING };
      default:
        error('E0202', `Unknown generic type '${ann.name}'`, ann.span);
        return T_ANY;
    }
  };

  // Pass 1: Collect imports and struct declarations (names first for self/mutual refs)
  for (const item of program.items) {
    if (item.kind === 'UseStmt') {
      const modName = item.path.join('.');
      if (!KNOWN_MODULES.has(modName)) {
        error(
          'E0203',
          `Unknown module '${modName}'`,
          item.span,
          `Available modules: ${Array.from(KNOWN_MODULES).join(', ')}`
        );
      } else {
        imports.push(modName);
        const alias = item.path[item.path.length - 1];
        globalScope.define({ name: alias, type: { tag: 'Module', name: modName }, mutability: 'module', span: item.span });
        const exports = MODULE_EXPORTS[modName] ?? {};
        for (const [fnName, sig] of Object.entries(exports)) {
          if (!functions.has(fnName)) {
            functions.set(fnName, sig);
            globalScope.define({
              name: fnName,
              type: { tag: 'Fn', params: sig.params, ret: sig.ret, variadic: sig.variadic },
              mutability: 'fn',
            });
          }
        }
      }
    } else if (item.kind === 'StructDecl') {
      if (structs.has(item.name)) {
        error('E0204', `Duplicate struct declaration '${item.name}'`, item.span);
      } else {
        structs.set(item.name, {
          name: item.name,
          fields: new Map(),
          fieldOrder: [],
          span: item.span,
        });
      }
    }
  }

  // Pass 1b: Resolve struct field types
  for (const item of program.items) {
    if (item.kind === 'StructDecl') {
      const info = structs.get(item.name)!;
      for (const f of item.fields) {
        if (info.fields.has(f.name)) {
          error('E0205', `Duplicate field '${f.name}' in struct '${item.name}'`, f.span);
          continue;
        }
        info.fields.set(f.name, resolveTypeAnnotation(f.typeAnnotation));
        info.fieldOrder.push(f.name);
      }
    }
  }

  // Pass 2: Register all top-level function signatures so functions can call each other out of order
  for (const item of program.items) {
    if (item.kind === 'FnDecl') {
      const paramTypes = item.params.map((p) => resolveTypeAnnotation(p.typeAnnotation));
      const retType = item.returnType ? resolveTypeAnnotation(item.returnType) : T_VOID;
      const fnType: NovaType = { tag: 'Fn', params: paramTypes, ret: retType };
      const existing = globalScope.lookup(item.name);
      if (existing && existing.span) {
        error('E0206', `Function '${item.name}' is already defined`, item.span);
      } else {
        globalScope.override({ name: item.name, type: fnType, mutability: 'fn', span: item.span });
        functions.set(item.name, { params: paramTypes, ret: retType });
        recordedSymbols.push({ name: item.name, type: fnType, mutability: 'fn', span: item.span });
      }
    }
  }

  let currentReturnType: NovaType | null = null;
  let loopDepth = 0;

  const definitelyReturns = (stmts: Stmt[]): boolean => {
    for (const s of stmts) {
      if (s.kind === 'ReturnStmt') return true;
      if (s.kind === 'IfStmt' && s.elseBranch) {
        const thenRet = definitelyReturns(s.thenBranch);
        const elseRet = Array.isArray(s.elseBranch)
          ? definitelyReturns(s.elseBranch)
          : definitelyReturns([s.elseBranch]);
        if (thenRet && elseRet) return true;
      }
    }
    return false;
  };

  const checkStmt = (stmt: Stmt, scope: Scope): void => {
    switch (stmt.kind) {
      case 'UseStmt':
      case 'StructDecl':
        return;

      case 'FnDecl': {
        const fnScope = new Scope(scope);
        const sig = functions.get(stmt.name);
        const paramTypes = sig?.params ?? stmt.params.map((p) => resolveTypeAnnotation(p.typeAnnotation));
        const retType = sig?.ret ?? (stmt.returnType ? resolveTypeAnnotation(stmt.returnType) : T_VOID);

        for (let i = 0; i < stmt.params.length; i++) {
          const p = stmt.params[i];
          const ok = fnScope.define({
            name: p.name,
            type: paramTypes[i],
            mutability: 'param',
            span: p.span,
          });
          if (!ok) {
            error('E0207', `Duplicate parameter '${p.name}' in function '${stmt.name}'`, p.span);
          }
        }

        const prevRet = currentReturnType;
        currentReturnType = retType;
        for (const s of stmt.body) {
          checkStmt(s, fnScope);
        }
        currentReturnType = prevRet;

        if (retType.tag !== 'Void' && retType.tag !== 'Any' && !definitelyReturns(stmt.body)) {
          error(
            'E0228',
            `Function '${stmt.name}' declares return type '${formatType(retType)}' but does not return a value on all code paths`,
            stmt.span,
            `Add a 'return' statement returning '${formatType(retType)}'`
          );
        }
        return;
      }

      case 'VarDecl': {
        const initType = inferExpr(stmt.initializer, scope);
        let declaredType = initType;
        if (stmt.typeAnnotation) {
          const annotated = resolveTypeAnnotation(stmt.typeAnnotation);
          if (!typesCompatible(annotated, initType)) {
            error(
              'E0210',
              `Type mismatch: cannot assign '${formatType(initType)}' to '${stmt.name}' of type '${formatType(annotated)}'`,
              stmt.initializer.span,
              `Change the initializer to match '${formatType(annotated)}' or update the type annotation`
            );
          }
          declaredType = annotated;
        }
        const sym: SymbolInfo = {
          name: stmt.name,
          type: declaredType,
          mutability: stmt.mutability,
          span: stmt.span,
        };
        if (!scope.define(sym)) {
          error('E0211', `Variable '${stmt.name}' is already declared in this scope`, stmt.span);
        } else {
          recordedSymbols.push(sym);
        }
        return;
      }

      case 'AssignStmt': {
        const valType = inferExpr(stmt.value, scope);
        if (stmt.target.kind === 'IdentExpr') {
          const sym = scope.lookup(stmt.target.name);
          if (!sym) {
            error('E0212', `Undefined variable '${stmt.target.name}'`, stmt.target.span);
            return;
          }
          if (sym.mutability !== 'mut') {
            error(
              'E0213',
              `Cannot assign to immutable binding '${stmt.target.name}'`,
              stmt.span,
              `Declare '${stmt.target.name}' with 'mut' to allow reassignment`
            );
            return;
          }
          if (stmt.op !== '=') {
            const numeric = (t: NovaType) => t.tag === 'Int' || t.tag === 'Float' || t.tag === 'Any';
            if (!(numeric(sym.type) && numeric(valType))) {
              error(
                'E0214',
                `Compound operator '${stmt.op}' requires numeric operands, found '${formatType(sym.type)}' and '${formatType(valType)}'`,
                stmt.span
              );
            }
          } else if (!typesCompatible(sym.type, valType)) {
            error(
              'E0210',
              `Type mismatch: cannot assign '${formatType(valType)}' to '${stmt.target.name}' of type '${formatType(sym.type)}'`,
              stmt.value.span
            );
          }
        } else if (stmt.target.kind === 'FieldExpr' || stmt.target.kind === 'IndexExpr') {
          const targetType = inferExpr(stmt.target, scope);
          if (!typesCompatible(targetType, valType)) {
            error(
              'E0210',
              `Type mismatch in assignment: expected '${formatType(targetType)}', found '${formatType(valType)}'`,
              stmt.value.span
            );
          }
        } else {
          error('E0215', 'Invalid assignment target', stmt.target.span);
        }
        return;
      }

      case 'IfStmt': {
        const condType = inferExpr(stmt.condition, scope);
        if (condType.tag !== 'Bool' && condType.tag !== 'Any') {
          error(
            'E0216',
            `Condition in 'if' statement must be 'Bool', found '${formatType(condType)}'`,
            stmt.condition.span
          );
        }
        const thenScope = new Scope(scope);
        for (const s of stmt.thenBranch) checkStmt(s, thenScope);
        if (stmt.elseBranch) {
          if (Array.isArray(stmt.elseBranch)) {
            const elseScope = new Scope(scope);
            for (const s of stmt.elseBranch) checkStmt(s, elseScope);
          } else {
            checkStmt(stmt.elseBranch, scope);
          }
        }
        return;
      }

      case 'WhileStmt': {
        const condType = inferExpr(stmt.condition, scope);
        if (condType.tag !== 'Bool' && condType.tag !== 'Any') {
          error(
            'E0216',
            `Condition in 'while' loop must be 'Bool', found '${formatType(condType)}'`,
            stmt.condition.span
          );
        }
        const bodyScope = new Scope(scope);
        loopDepth++;
        for (const s of stmt.body) checkStmt(s, bodyScope);
        loopDepth--;
        return;
      }

      case 'ForStmt': {
        const iterType = inferExpr(stmt.iterable, scope);
        let elemType: NovaType = T_INT;
        if (iterType.tag === 'Range') {
          elemType = T_INT;
        } else if (iterType.tag === 'Array') {
          elemType = iterType.element;
        } else if (iterType.tag !== 'Any') {
          error(
            'E0217',
            `'for' loop requires a Range or Array, found '${formatType(iterType)}'`,
            stmt.iterable.span
          );
        }
        const bodyScope = new Scope(scope);
        bodyScope.define({ name: stmt.variable, type: elemType, mutability: 'let', span: stmt.span });
        loopDepth++;
        for (const s of stmt.body) checkStmt(s, bodyScope);
        loopDepth--;
        return;
      }

      case 'ReturnStmt': {
        if (currentReturnType === null) {
          error('E0218', "'return' statement outside of a function", stmt.span);
          return;
        }
        const actual = stmt.value ? inferExpr(stmt.value, scope) : T_VOID;
        if (!typesCompatible(currentReturnType, actual)) {
          error(
            'E0219',
            `Return type mismatch: expected '${formatType(currentReturnType)}', found '${formatType(actual)}'`,
            stmt.span
          );
        }
        return;
      }

      case 'BreakStmt':
      case 'ContinueStmt': {
        if (loopDepth === 0) {
          error('E0220', `'${stmt.kind === 'BreakStmt' ? 'break' : 'continue'}' used outside of a loop`, stmt.span);
        }
        return;
      }

      case 'ExprStmt': {
        inferExpr(stmt.expr, scope);
        return;
      }
    }
  };

  const inferExpr = (expr: Expr, scope: Scope): NovaType => {
    switch (expr.kind) {
      case 'IntLiteral': return T_INT;
      case 'FloatLiteral': return T_FLOAT;
      case 'StringLiteral': return T_STRING;
      case 'CharLiteral': return T_CHAR;
      case 'BoolLiteral': return T_BOOL;
      case 'NoneLiteral': return T_NONE;

      case 'IdentExpr': {
        const sym = scope.lookup(expr.name);
        if (!sym) {
          error('E0212', `Undefined variable '${expr.name}'`, expr.span);
          return T_ANY;
        }
        return sym.type;
      }

      case 'UnaryExpr': {
        const opT = inferExpr(expr.operand, scope);
        if (expr.op === '-') {
          if (opT.tag !== 'Int' && opT.tag !== 'Float' && opT.tag !== 'Tensor' && opT.tag !== 'Any') {
            error('E0221', `Unary '-' cannot be applied to '${formatType(opT)}'`, expr.span);
          }
          return opT;
        }
        if (opT.tag !== 'Bool' && opT.tag !== 'Any') {
          error('E0222', `Logical '!' requires 'Bool', found '${formatType(opT)}'`, expr.span);
        }
        return T_BOOL;
      }

      case 'RangeExpr': {
        const s = inferExpr(expr.start, scope);
        const e = inferExpr(expr.end, scope);
        if ((s.tag !== 'Int' && s.tag !== 'Any') || (e.tag !== 'Int' && e.tag !== 'Any')) {
          error('E0223', `Range bounds must be 'Int', found '${formatType(s)}..${formatType(e)}'`, expr.span);
        }
        return T_RANGE;
      }

      case 'BinaryExpr': {
        const l = inferExpr(expr.left, scope);
        const r = inferExpr(expr.right, scope);
        const op = expr.op;

        if (op === '@') {
          if ((l.tag === 'Tensor' || l.tag === 'Any') && (r.tag === 'Tensor' || r.tag === 'Any')) {
            return T_TENSOR_FLOAT;
          }
          error('E0224', `Matrix multiplication '@' requires Tensor operands, found '${formatType(l)}' and '${formatType(r)}'`, expr.span);
          return T_TENSOR_FLOAT;
        }

        if (op === '+') {
          if (l.tag === 'String' || r.tag === 'String') {
            if (l.tag === 'String' && r.tag === 'String') return T_STRING;
            if (l.tag === 'Any' || r.tag === 'Any') return T_STRING;
            error(
              'E0225',
              `Cannot concatenate '${formatType(l)}' and '${formatType(r)}' with '+'`,
              expr.span,
              'Use str(...) to convert non-string values to String'
            );
            return T_STRING;
          }
          if (l.tag === 'Tensor' && (r.tag === 'Tensor' || r.tag === 'Float' || r.tag === 'Int')) return T_TENSOR_FLOAT;
          if (l.tag === 'Int' && r.tag === 'Int') return T_INT;
          if ((l.tag === 'Float' || l.tag === 'Int') && (r.tag === 'Float' || r.tag === 'Int')) return T_FLOAT;
          if (l.tag === 'Any' || r.tag === 'Any') return l.tag === 'Any' ? r : l;
          error('E0225', `Operator '+' cannot be applied to '${formatType(l)}' and '${formatType(r)}'`, expr.span);
          return T_ANY;
        }

        if (op === '-' || op === '*' || op === '/' || op === '%') {
          if (l.tag === 'Tensor' && (r.tag === 'Tensor' || r.tag === 'Float' || r.tag === 'Int')) return T_TENSOR_FLOAT;
          if (l.tag === 'Int' && r.tag === 'Int') return T_INT;
          if ((l.tag === 'Float' || l.tag === 'Int') && (r.tag === 'Float' || r.tag === 'Int')) return T_FLOAT;
          if (l.tag === 'Any' || r.tag === 'Any') return l.tag === 'Any' ? r : l;
          error('E0225', `Operator '${op}' cannot be applied to '${formatType(l)}' and '${formatType(r)}'`, expr.span);
          return T_ANY;
        }

        if (op === '&&' || op === '||') {
          if ((l.tag !== 'Bool' && l.tag !== 'Any') || (r.tag !== 'Bool' && r.tag !== 'Any')) {
            error('E0226', `Logical operator '${op}' requires 'Bool' operands`, expr.span);
          }
          return T_BOOL;
        }

        if (op === '<' || op === '<=' || op === '>' || op === '>=') {
          const valid = (t: NovaType) => t.tag === 'Int' || t.tag === 'Float' || t.tag === 'String' || t.tag === 'Any';
          if (!valid(l) || !valid(r)) {
            error('E0227', `Cannot compare '${formatType(l)}' and '${formatType(r)}' with '${op}'`, expr.span);
          }
          return T_BOOL;
        }

        // == or !=
        return T_BOOL;
      }

      case 'CallExpr': {
        // Module-qualified call like math.sqrt(25.0) or ai.tensor(...)
        if (expr.callee.kind === 'FieldExpr') {
          const objType = inferExpr(expr.callee.object, scope);
          const argTypes = expr.args.map((a) => inferExpr(a, scope));
          if (objType.tag === 'Module') {
            const modFns = MODULE_EXPORTS[objType.name];
            const fnSig = modFns?.[expr.callee.field];
            if (!fnSig) {
              error('E0230', `Function '${expr.callee.field}' not found in module '${objType.name}'`, expr.callee.span);
              return T_ANY;
            }
            if (!fnSig.variadic && argTypes.length !== fnSig.params.length) {
              error(
                'E0231',
                `Function '${objType.name}.${expr.callee.field}' expects ${fnSig.params.length} argument(s), got ${argTypes.length}`,
                expr.span
              );
            }
            return fnSig.ret;
          }
          // Built-in value methods (e.g. tensor.shape(), array.len(), string.upper())
          if (objType.tag === 'Tensor') {
            if (expr.callee.field === 'shape') return { tag: 'Array', element: T_INT };
            if (expr.callee.field === 'argmax') return T_INT;
            if (
              expr.callee.field === 'sum' ||
              expr.callee.field === 'mean' ||
              expr.callee.field === 'max' ||
              expr.callee.field === 'min'
            ) {
              return T_FLOAT;
            }
            if (
              expr.callee.field === 'transpose' ||
              expr.callee.field === 'reshape' ||
              expr.callee.field === 'relu' ||
              expr.callee.field === 'sigmoid' ||
              expr.callee.field === 'tanh' ||
              expr.callee.field === 'softmax'
            ) {
              return T_TENSOR_FLOAT;
            }
          }
          if (objType.tag === 'Array') {
            if (expr.callee.field === 'len') return T_INT;
            if (expr.callee.field === 'push') return T_VOID;
          }
          if (objType.tag === 'String') {
            if (expr.callee.field === 'len') return T_INT;
            if (expr.callee.field === 'upper' || expr.callee.field === 'lower' || expr.callee.field === 'trim') return T_STRING;
          }
          return T_ANY;
        }

        if (expr.callee.kind === 'IdentExpr') {
          const fnName = expr.callee.name;
          const sym = scope.lookup(fnName);
          if (!sym || sym.type.tag !== 'Fn') {
            error('E0230', `Undefined function '${fnName}'`, expr.callee.span);
            expr.args.forEach((a) => inferExpr(a, scope));
            return T_ANY;
          }
          const fnType = sym.type;
          const argTypes = expr.args.map((a) => inferExpr(a, scope));
          if (!fnType.variadic && argTypes.length !== fnType.params.length) {
            error(
              'E0231',
              `Function '${fnName}' expects ${fnType.params.length} argument(s), got ${argTypes.length}`,
              expr.span
            );
          } else if (!fnType.variadic) {
            for (let i = 0; i < fnType.params.length; i++) {
              if (!typesCompatible(fnType.params[i], argTypes[i])) {
                error(
                  'E0232',
                  `Argument ${i + 1} of '${fnName}' expects '${formatType(fnType.params[i])}', found '${formatType(argTypes[i])}'`,
                  expr.args[i].span
                );
              }
            }
          }
          return fnType.ret;
        }

        expr.args.forEach((a) => inferExpr(a, scope));
        return T_ANY;
      }

      case 'FieldExpr': {
        const objType = inferExpr(expr.object, scope);
        if (objType.tag === 'Struct') {
          const sInfo = structs.get(objType.name);
          const fType = sInfo?.fields.get(expr.field);
          if (!fType) {
            error('E0233', `Struct '${objType.name}' has no field '${expr.field}'`, expr.span);
            return T_ANY;
          }
          return fType;
        }
        if (objType.tag === 'Module' || objType.tag === 'Any' || objType.tag === 'Tensor' || objType.tag === 'Array' || objType.tag === 'String') {
          return T_ANY;
        }
        error('E0234', `Cannot access field '${expr.field}' on type '${formatType(objType)}'`, expr.span);
        return T_ANY;
      }

      case 'IndexExpr': {
        const objType = inferExpr(expr.object, scope);
        const idxType = inferExpr(expr.index, scope);
        if (objType.tag === 'Array') {
          if (idxType.tag !== 'Int' && idxType.tag !== 'Any') {
            error('E0235', `Array index must be 'Int', found '${formatType(idxType)}'`, expr.index.span);
          }
          return objType.element;
        }
        if (objType.tag === 'Map') {
          return objType.value;
        }
        if (objType.tag === 'Tensor') {
          return T_FLOAT;
        }
        if (objType.tag === 'String') {
          return T_CHAR;
        }
        if (objType.tag === 'Any') return T_ANY;
        error('E0236', `Cannot index into type '${formatType(objType)}'`, expr.span);
        return T_ANY;
      }

      case 'ArrayLiteral': {
        if (expr.elements.length === 0) return { tag: 'Array', element: T_ANY };
        const firstType = inferExpr(expr.elements[0], scope);
        for (let i = 1; i < expr.elements.length; i++) {
          const elemType = inferExpr(expr.elements[i], scope);
          if (!typesCompatible(firstType, elemType)) {
            // Allow Int/Float mix in numeric arrays -> Float
            if (
              (firstType.tag === 'Int' && elemType.tag === 'Float') ||
              (firstType.tag === 'Float' && elemType.tag === 'Int')
            ) {
              continue;
            }
            error(
              'E0237',
              `Heterogeneous array element: expected '${formatType(firstType)}', found '${formatType(elemType)}'`,
              expr.elements[i].span
            );
          }
        }
        return { tag: 'Array', element: firstType };
      }

      case 'MapLiteral': {
        if (expr.entries.length === 0) return { tag: 'Map', key: T_ANY, value: T_ANY };
        const kType = inferExpr(expr.entries[0].key, scope);
        const vType = inferExpr(expr.entries[0].value, scope);
        for (let i = 1; i < expr.entries.length; i++) {
          inferExpr(expr.entries[i].key, scope);
          inferExpr(expr.entries[i].value, scope);
        }
        return { tag: 'Map', key: kType, value: vType };
      }

      case 'StructLiteral': {
        const sInfo = structs.get(expr.name);
        if (!sInfo) {
          error('E0238', `Undefined struct '${expr.name}'`, expr.span);
          expr.fields.forEach((f) => inferExpr(f.value, scope));
          return T_ANY;
        }
        const provided = new Set<string>();
        for (const f of expr.fields) {
          provided.add(f.name);
          const expected = sInfo.fields.get(f.name);
          const actual = inferExpr(f.value, scope);
          if (!expected) {
            error('E0233', `Struct '${expr.name}' has no field '${f.name}'`, f.span);
          } else if (!typesCompatible(expected, actual)) {
            error(
              'E0210',
              `Field '${f.name}' of struct '${expr.name}' expects '${formatType(expected)}', found '${formatType(actual)}'`,
              f.value.span
            );
          }
        }
        for (const reqField of sInfo.fieldOrder) {
          if (!provided.has(reqField)) {
            error('E0239', `Missing required field '${reqField}' in '${expr.name}' literal`, expr.span);
          }
        }
        return { tag: 'Struct', name: expr.name };
      }

      case 'ResultExpr': {
        const inner = inferExpr(expr.value, scope);
        if (expr.variant === 'Ok') {
          return { tag: 'Result', ok: inner, err: T_ANY };
        }
        return { tag: 'Result', ok: T_ANY, err: inner };
      }
    }
  };

  for (const item of program.items) {
    checkStmt(item, globalScope);
  }

  return {
    diagnostics,
    symbols: recordedSymbols,
    structs,
    functions,
    imports,
  };
}
