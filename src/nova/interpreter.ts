import {
  ActivationKind,
  AdamOptimizer,
  AutogradNode,
  bceLoss,
  GradVar,
  mseLoss,
  NovaOptimizer,
  NovaTensor,
  SequentialModel,
  SGDOptimizer,
  trainModel,
  trainStep,
  trainXorNetwork,
} from './ai';
import { Expr, FnDecl, Program, Stmt } from './ast';
import { Diagnostic, Span } from './diagnostics';
import { NovaHeap, NovaValue } from './gc';
import { lex } from './lexer';
import { parse } from './parser';
import { VMExecutionResult } from './vm';

class ReturnSignal {
  constructor(public readonly value: NovaValue) {}
}

class BreakSignal {}
class ContinueSignal {}

class RuntimeErrorWithSpan extends Error {
  constructor(
    message: string,
    public readonly span: Span
  ) {
    super(message);
  }
}

class Environment {
  private values = new Map<string, { value: NovaValue; mutable: boolean }>();

  constructor(public readonly parent: Environment | null = null) {}

  define(name: string, value: NovaValue, mutable: boolean): void {
    this.values.set(name, { value, mutable });
  }

  get(name: string): NovaValue | undefined {
    const local = this.values.get(name);
    if (local) return local.value;
    return this.parent?.get(name);
  }

  assign(name: string, value: NovaValue, span: Span): void {
    const local = this.values.get(name);
    if (local) {
      if (!local.mutable) {
        throw new RuntimeErrorWithSpan(`Cannot assign to immutable binding '${name}'`, span);
      }
      local.value = value;
      return;
    }
    if (this.parent) {
      this.parent.assign(name, value, span);
      return;
    }
    throw new RuntimeErrorWithSpan(`Undefined variable '${name}'`, span);
  }

  collectRoots(out: NovaValue[]): void {
    for (const entry of this.values.values()) {
      out.push(entry.value);
    }
    this.parent?.collectRoots(out);
  }
}

/**
 * Nova 0.1 AST Tree-Walking Interpreter.
 * Executes Nova programs directly from the AST independently of bytecode/LLVM,
 * following identical language semantics to the Nova VM and Native backend.
 * Also supports stateful REPL sessions via `evalReplChunk()`.
 */
export class NovaInterpreter {
  private heap = new NovaHeap();
  private globalEnv = new Environment(null);
  private functions = new Map<string, FnDecl>();
  private stdout: string[] = [];
  private steps = 0;
  private maxSteps = 250_000;
  private callDepth = 0;
  private maxCallDepth = 256;

  run(program: Program): VMExecutionResult {
    const startMs = performance.now();
    this.heap = new NovaHeap();
    this.globalEnv = new Environment(null);
    this.functions.clear();
    this.stdout = [];
    this.steps = 0;
    this.callDepth = 0;

    try {
      // Register functions first
      for (const item of program.items) {
        if (item.kind === 'FnDecl') {
          this.functions.set(item.name, item);
        }
      }

      // Execute top-level statements
      for (const item of program.items) {
        if (item.kind !== 'FnDecl' && item.kind !== 'StructDecl' && item.kind !== 'UseStmt') {
          this.execStmt(item, this.globalEnv);
        }
      }

      // Execute main() if defined
      const mainFn = this.functions.get('main');
      if (mainFn) {
        this.callFunction(mainFn, [], mainFn.span);
      }

      this.collectGarbage(this.globalEnv);

      return {
        stdout: this.stdout,
        exitCode: 0,
        instructionsExecuted: this.steps,
        durationMs: Number((performance.now() - startMs).toFixed(3)),
        gcStats: this.heap.getStats(),
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const span =
        err instanceof RuntimeErrorWithSpan ? err.span : { start: 0, end: 0, line: 1, column: 1 };
      return {
        stdout: this.stdout,
        exitCode: 1,
        runtimeError: {
          stage: 'vm',
          severity: 'error',
          code: 'E0501',
          message,
          span,
        },
        instructionsExecuted: this.steps,
        durationMs: Number((performance.now() - startMs).toFixed(3)),
        gcStats: this.heap.getStats(),
      };
    }
  }

  /**
   * Evaluates a single REPL line/chunk in the persistent global environment.
   */
  evalReplChunk(source: string): {
    ok: boolean;
    stdout: string[];
    lastValue?: string;
    diagnostics: Diagnostic[];
  } {
    const lexRes = lex(source);
    const parseRes = parse(lexRes.tokens);
    const diagnostics: Diagnostic[] = [...lexRes.diagnostics, ...parseRes.diagnostics];
    if (diagnostics.length > 0) {
      return { ok: false, stdout: [], diagnostics };
    }

    const prevStdoutLen = this.stdout.length;
    let lastVal: NovaValue | undefined;
    try {
      for (const item of parseRes.program.items) {
        if (item.kind === 'FnDecl') {
          this.functions.set(item.name, item);
        } else if (item.kind === 'ExprStmt') {
          lastVal = this.evalExpr(item.expr, this.globalEnv);
        } else if (item.kind !== 'StructDecl' && item.kind !== 'UseStmt') {
          this.execStmt(item, this.globalEnv);
        }
      }
      const newStdout = this.stdout.slice(prevStdoutLen);
      const formattedLast =
        lastVal && lastVal.kind !== 'None' ? this.heap.formatValue(lastVal) : undefined;
      return { ok: true, stdout: newStdout, lastValue: formattedLast, diagnostics: [] };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const span =
        err instanceof RuntimeErrorWithSpan ? err.span : { start: 0, end: 0, line: 1, column: 1 };
      return {
        ok: false,
        stdout: this.stdout.slice(prevStdoutLen),
        diagnostics: [{ stage: 'vm', severity: 'error', code: 'E0501', message, span }],
      };
    }
  }

  private tick(span: Span): void {
    if (++this.steps > this.maxSteps) {
      throw new RuntimeErrorWithSpan(
        `Execution limit exceeded (${this.maxSteps} AST steps). Check for infinite loops.`,
        span
      );
    }
  }

  private collectGarbage(env: Environment): void {
    const roots: NovaValue[] = [];
    env.collectRoots(roots);
    this.heap.collect(roots);
  }

  private execBlock(stmts: Stmt[], env: Environment): void {
    for (const s of stmts) {
      this.execStmt(s, env);
    }
  }

  private execStmt(stmt: Stmt, env: Environment): void {
    this.tick(stmt.span);
    switch (stmt.kind) {
      case 'UseStmt':
      case 'StructDecl':
        return;

      case 'FnDecl': {
        this.functions.set(stmt.name, stmt);
        return;
      }

      case 'VarDecl': {
        const val = this.evalExpr(stmt.initializer, env);
        env.define(stmt.name, val, stmt.mutability === 'mut');
        return;
      }

      case 'AssignStmt': {
        const compoundOpMap = { '+=': '+', '-=': '-', '*=': '*', '/=': '/' } as const;
        if (stmt.target.kind === 'IdentExpr') {
          const rhs = this.evalExpr(stmt.value, env);
          let nextVal = rhs;
          if (stmt.op !== '=') {
            const cur = env.get(stmt.target.name);
            if (!cur) {
              throw new RuntimeErrorWithSpan(`Undefined variable '${stmt.target.name}'`, stmt.target.span);
            }
            nextVal = this.applyBinary(compoundOpMap[stmt.op], cur, rhs, stmt.span);
          }
          env.assign(stmt.target.name, nextVal, stmt.span);
        } else if (stmt.target.kind === 'FieldExpr') {
          const obj = this.evalExpr(stmt.target.object, env);
          const rhs = this.evalExpr(stmt.value, env);
          const h = this.heap.deref(obj).payload;
          if (h.tag === 'Struct') {
            let nextVal = rhs;
            if (stmt.op !== '=') {
              const cur = h.fields.get(stmt.target.field);
              if (!cur) {
                throw new RuntimeErrorWithSpan(`Struct '${h.typeName}' has no field '${stmt.target.field}'`, stmt.span);
              }
              nextVal = this.applyBinary(compoundOpMap[stmt.op], cur, rhs, stmt.span);
            }
            h.fields.set(stmt.target.field, nextVal);
          } else if (h.tag === 'Map') {
            let nextVal = rhs;
            if (stmt.op !== '=') {
              const cur = h.entries.get(stmt.target.field) ?? { kind: 'None' };
              nextVal = this.applyBinary(compoundOpMap[stmt.op], cur, rhs, stmt.span);
            }
            h.entries.set(stmt.target.field, nextVal);
          } else {
            throw new RuntimeErrorWithSpan(`Cannot assign field '${stmt.target.field}' on ${h.tag}`, stmt.span);
          }
        } else if (stmt.target.kind === 'IndexExpr') {
          const obj = this.evalExpr(stmt.target.object, env);
          const idx = this.evalExpr(stmt.target.index, env);
          const rhs = this.evalExpr(stmt.value, env);
          const h = this.heap.deref(obj).payload;
          if (h.tag === 'Array') {
            const i = this.asNumber(idx, stmt.target.index.span);
            if (i < 0 || i >= h.elements.length) {
              throw new RuntimeErrorWithSpan(`Array index ${i} out of bounds`, stmt.span);
            }
            const nextVal =
              stmt.op === '=' ? rhs : this.applyBinary(compoundOpMap[stmt.op], h.elements[i], rhs, stmt.span);
            h.elements[i] = nextVal;
          } else if (h.tag === 'Map') {
            const key = this.heap.formatValue(idx);
            const cur = h.entries.get(key) ?? { kind: 'None' };
            const nextVal = stmt.op === '=' ? rhs : this.applyBinary(compoundOpMap[stmt.op], cur, rhs, stmt.span);
            h.entries.set(key, nextVal);
          } else {
            throw new RuntimeErrorWithSpan(`Cannot assign index on ${h.tag}`, stmt.span);
          }
        }
        return;
      }

      case 'IfStmt': {
        const cond = this.evalExpr(stmt.condition, env);
        if (this.asBool(cond)) {
          this.execBlock(stmt.thenBranch, new Environment(env));
        } else if (stmt.elseBranch) {
          if (Array.isArray(stmt.elseBranch)) {
            this.execBlock(stmt.elseBranch, new Environment(env));
          } else {
            this.execStmt(stmt.elseBranch, env);
          }
        }
        return;
      }

      case 'WhileStmt': {
        while (this.asBool(this.evalExpr(stmt.condition, env))) {
          this.tick(stmt.span);
          try {
            this.execBlock(stmt.body, new Environment(env));
          } catch (sig) {
            if (sig instanceof BreakSignal) break;
            if (sig instanceof ContinueSignal) continue;
            throw sig;
          }
        }
        return;
      }

      case 'ForStmt': {
        if (stmt.iterable.kind === 'RangeExpr') {
          const start = Math.trunc(this.asNumber(this.evalExpr(stmt.iterable.start, env), stmt.iterable.start.span));
          const end = Math.trunc(this.asNumber(this.evalExpr(stmt.iterable.end, env), stmt.iterable.end.span));
          for (let i = start; i < end; i++) {
            this.tick(stmt.span);
            const loopEnv = new Environment(env);
            loopEnv.define(stmt.variable, { kind: 'Int', value: i }, false);
            try {
              this.execBlock(stmt.body, loopEnv);
            } catch (sig) {
              if (sig instanceof BreakSignal) break;
              if (sig instanceof ContinueSignal) continue;
              throw sig;
            }
          }
        } else {
          const iterVal = this.evalExpr(stmt.iterable, env);
          const h = this.heap.deref(iterVal).payload;
          if (h.tag !== 'Array') {
            throw new RuntimeErrorWithSpan(`Cannot iterate over ${h.tag}`, stmt.iterable.span);
          }
          for (const el of h.elements) {
            this.tick(stmt.span);
            const loopEnv = new Environment(env);
            loopEnv.define(stmt.variable, el, false);
            try {
              this.execBlock(stmt.body, loopEnv);
            } catch (sig) {
              if (sig instanceof BreakSignal) break;
              if (sig instanceof ContinueSignal) continue;
              throw sig;
            }
          }
        }
        return;
      }

      case 'ReturnStmt': {
        const val = stmt.value ? this.evalExpr(stmt.value, env) : ({ kind: 'None' } as NovaValue);
        throw new ReturnSignal(val);
      }

      case 'BreakStmt':
        throw new BreakSignal();

      case 'ContinueStmt':
        throw new ContinueSignal();

      case 'ExprStmt': {
        this.evalExpr(stmt.expr, env);
        return;
      }
    }
  }

  private evalExpr(expr: Expr, env: Environment): NovaValue {
    this.tick(expr.span);
    switch (expr.kind) {
      case 'IntLiteral': return { kind: 'Int', value: expr.value };
      case 'FloatLiteral': return { kind: 'Float', value: expr.value };
      case 'BoolLiteral': return { kind: 'Bool', value: expr.value };
      case 'CharLiteral': return { kind: 'Char', value: expr.value };
      case 'NoneLiteral': return { kind: 'None' };
      case 'StringLiteral': return this.heap.alloc({ tag: 'String', value: expr.value });

      case 'IdentExpr': {
        const val = env.get(expr.name);
        if (!val) {
          throw new RuntimeErrorWithSpan(`Undefined variable '${expr.name}'`, expr.span);
        }
        return val;
      }

      case 'UnaryExpr': {
        const operand = this.evalExpr(expr.operand, env);
        if (expr.op === '!') {
          return { kind: 'Bool', value: !this.asBool(operand) };
        }
        const t = this.asTensor(operand);
        if (t) return this.heap.alloc({ tag: 'Tensor', tensor: t.mul(-1) });
        if (operand.kind === 'Int') return { kind: 'Int', value: -operand.value };
        return { kind: 'Float', value: -this.asNumber(operand, expr.span) };
      }

      case 'BinaryExpr': {
        // True short-circuit evaluation for && and ||
        if (expr.op === '&&') {
          const l = this.evalExpr(expr.left, env);
          if (!this.asBool(l)) return { kind: 'Bool', value: false };
          return { kind: 'Bool', value: this.asBool(this.evalExpr(expr.right, env)) };
        }
        if (expr.op === '||') {
          const l = this.evalExpr(expr.left, env);
          if (this.asBool(l)) return { kind: 'Bool', value: true };
          return { kind: 'Bool', value: this.asBool(this.evalExpr(expr.right, env)) };
        }
        const left = this.evalExpr(expr.left, env);
        const right = this.evalExpr(expr.right, env);
        return this.applyBinary(expr.op, left, right, expr.span);
      }

      case 'RangeExpr': {
        const s = Math.trunc(this.asNumber(this.evalExpr(expr.start, env), expr.start.span));
        const e = Math.trunc(this.asNumber(this.evalExpr(expr.end, env), expr.end.span));
        const els: NovaValue[] = [];
        for (let i = s; i < e; i++) els.push({ kind: 'Int', value: i });
        return this.heap.alloc({ tag: 'Array', elements: els });
      }

      case 'CallExpr': {
        if (expr.callee.kind === 'FieldExpr') {
          if (expr.callee.object.kind === 'IdentExpr' && env.get(expr.callee.object.name) === undefined) {
            const args = expr.args.map((a) => this.evalExpr(a, env));
            return this.invokeBuiltin(`${expr.callee.object.name}.${expr.callee.field}`, args, expr.span);
          }
          const obj = this.evalExpr(expr.callee.object, env);
          const args = expr.args.map((a) => this.evalExpr(a, env));
          return this.invokeMethod(obj, expr.callee.field, args, expr.span);
        }
        if (expr.callee.kind === 'IdentExpr') {
          const args = expr.args.map((a) => this.evalExpr(a, env));
          const userFn = this.functions.get(expr.callee.name);
          if (userFn) {
            return this.callFunction(userFn, args, expr.span);
          }
          return this.invokeBuiltin(expr.callee.name, args, expr.span);
        }
        throw new RuntimeErrorWithSpan('Invalid call target', expr.span);
      }

      case 'FieldExpr': {
        const obj = this.evalExpr(expr.object, env);
        const h = this.heap.deref(obj).payload;
        if (h.tag === 'Struct') {
          const v = h.fields.get(expr.field);
          if (!v) throw new RuntimeErrorWithSpan(`Struct '${h.typeName}' has no field '${expr.field}'`, expr.span);
          return v;
        }
        if (h.tag === 'Map') {
          return h.entries.get(expr.field) ?? { kind: 'None' };
        }
        throw new RuntimeErrorWithSpan(`Cannot access field '${expr.field}' on ${h.tag}`, expr.span);
      }

      case 'IndexExpr': {
        const obj = this.evalExpr(expr.object, env);
        const idx = this.evalExpr(expr.index, env);
        const h = this.heap.deref(obj).payload;
        if (h.tag === 'Array') {
          const i = this.asNumber(idx, expr.index.span);
          if (i < 0 || i >= h.elements.length) {
            throw new RuntimeErrorWithSpan(`Array index ${i} out of bounds (len=${h.elements.length})`, expr.span);
          }
          return h.elements[i];
        }
        if (h.tag === 'Map') {
          return h.entries.get(this.heap.formatValue(idx)) ?? { kind: 'None' };
        }
        if (h.tag === 'String') {
          const i = this.asNumber(idx, expr.index.span);
          const ch = h.value[i];
          if (ch === undefined) throw new RuntimeErrorWithSpan(`String index ${i} out of bounds`, expr.span);
          return { kind: 'Char', value: ch };
        }
        if (h.tag === 'Tensor') {
          const i = this.asNumber(idx, expr.index.span);
          if (i < 0 || i >= h.tensor.data.length) {
            throw new RuntimeErrorWithSpan(`Tensor index ${i} out of bounds`, expr.span);
          }
          return { kind: 'Float', value: h.tensor.data[i] };
        }
        throw new RuntimeErrorWithSpan(`Cannot index into ${h.tag}`, expr.span);
      }

      case 'ArrayLiteral':
        return this.heap.alloc({
          tag: 'Array',
          elements: expr.elements.map((e) => this.evalExpr(e, env)),
        });

      case 'MapLiteral': {
        const entries = new Map<string, NovaValue>();
        for (const e of expr.entries) {
          const k = this.heap.formatValue(this.evalExpr(e.key, env));
          const v = this.evalExpr(e.value, env);
          entries.set(k, v);
        }
        return this.heap.alloc({ tag: 'Map', entries });
      }

      case 'StructLiteral': {
        const fields = new Map<string, NovaValue>();
        for (const f of expr.fields) {
          fields.set(f.name, this.evalExpr(f.value, env));
        }
        return this.heap.alloc({ tag: 'Struct', typeName: expr.name, fields });
      }

      case 'ResultExpr':
        return this.heap.alloc({
          tag: 'Result',
          variant: expr.variant,
          value: this.evalExpr(expr.value, env),
        });
    }
  }

  private callFunction(fn: FnDecl, args: NovaValue[], span: Span): NovaValue {
    if (++this.callDepth > this.maxCallDepth) {
      throw new RuntimeErrorWithSpan(`Stack overflow: maximum recursion depth (${this.maxCallDepth}) exceeded in '${fn.name}'`, span);
    }
    const fnEnv = new Environment(this.globalEnv);
    for (let i = 0; i < fn.params.length; i++) {
      fnEnv.define(fn.params[i].name, args[i] ?? { kind: 'None' }, false);
    }
    try {
      this.execBlock(fn.body, fnEnv);
      return { kind: 'None' };
    } catch (sig) {
      if (sig instanceof ReturnSignal) {
        return sig.value;
      }
      throw sig;
    } finally {
      this.callDepth--;
    }
  }

  private applyBinary(op: string, l: NovaValue, r: NovaValue, span: Span): NovaValue {
    const lt = this.asTensor(l);
    const rt = this.asTensor(r);

    if (op === '@') {
      if (!lt || !rt) throw new RuntimeErrorWithSpan("Operator '@' requires two Tensors", span);
      return this.heap.alloc({ tag: 'Tensor', tensor: lt.matmul(rt) });
    }

    if (lt || rt) {
      if (op === '+') return this.heap.alloc({ tag: 'Tensor', tensor: lt ? lt.add(rt ?? this.asNumber(r, span)) : rt!.add(this.asNumber(l, span)) });
      if (op === '-') return this.heap.alloc({ tag: 'Tensor', tensor: lt ? lt.sub(rt ?? this.asNumber(r, span)) : new NovaTensor([1], [this.asNumber(l, span)]).sub(rt!) });
      if (op === '*') return this.heap.alloc({ tag: 'Tensor', tensor: lt ? lt.mul(rt ?? this.asNumber(r, span)) : rt!.mul(this.asNumber(l, span)) });
      if (op === '/') return this.heap.alloc({ tag: 'Tensor', tensor: lt ? lt.div(rt ?? this.asNumber(r, span)) : new NovaTensor([1], [this.asNumber(l, span)]).div(rt!) });
    }

    if (op === '+') {
      if (
        (l.kind === 'HeapRef' && this.heap.deref(l).payload.tag === 'String') ||
        (r.kind === 'HeapRef' && this.heap.deref(r).payload.tag === 'String')
      ) {
        return this.heap.alloc({ tag: 'String', value: this.heap.formatValue(l) + this.heap.formatValue(r) });
      }
      if (l.kind === 'Int' && r.kind === 'Int') return { kind: 'Int', value: l.value + r.value };
      return { kind: 'Float', value: this.asNumber(l, span) + this.asNumber(r, span) };
    }

    if (op === '-') {
      if (l.kind === 'Int' && r.kind === 'Int') return { kind: 'Int', value: l.value - r.value };
      return { kind: 'Float', value: this.asNumber(l, span) - this.asNumber(r, span) };
    }

    if (op === '*') {
      if (l.kind === 'Int' && r.kind === 'Int') return { kind: 'Int', value: l.value * r.value };
      return { kind: 'Float', value: this.asNumber(l, span) * this.asNumber(r, span) };
    }

    if (op === '/') {
      const rv = this.asNumber(r, span);
      if (rv === 0) throw new RuntimeErrorWithSpan('Runtime error: division by zero', span);
      if (l.kind === 'Int' && r.kind === 'Int') return { kind: 'Int', value: Math.trunc(l.value / rv) };
      return { kind: 'Float', value: this.asNumber(l, span) / rv };
    }

    if (op === '%') {
      const rv = this.asNumber(r, span);
      if (rv === 0) throw new RuntimeErrorWithSpan('Runtime error: modulo by zero', span);
      if (l.kind === 'Int' && r.kind === 'Int') return { kind: 'Int', value: l.value % rv };
      return { kind: 'Float', value: this.asNumber(l, span) % rv };
    }

    if (op === '==' || op === '!=') {
      const eq = this.valuesEqual(l, r);
      return { kind: 'Bool', value: op === '==' ? eq : !eq };
    }

    if (op === '<' || op === '<=' || op === '>' || op === '>=') {
      if (l.kind === 'HeapRef' && r.kind === 'HeapRef') {
        const ls = this.heap.formatValue(l);
        const rs = this.heap.formatValue(r);
        return {
          kind: 'Bool',
          value: op === '<' ? ls < rs : op === '<=' ? ls <= rs : op === '>' ? ls > rs : ls >= rs,
        };
      }
      const lv = this.asNumber(l, span);
      const rv = this.asNumber(r, span);
      return {
        kind: 'Bool',
        value: op === '<' ? lv < rv : op === '<=' ? lv <= rv : op === '>' ? lv > rv : lv >= rv,
      };
    }

    throw new RuntimeErrorWithSpan(`Unsupported binary operator '${op}'`, span);
  }

  private asNumber(v: NovaValue, span: Span): number {
    if (v.kind === 'Int' || v.kind === 'Float') return v.value;
    throw new RuntimeErrorWithSpan(`Expected numeric value, got ${v.kind}`, span);
  }

  private asBool(v: NovaValue): boolean {
    if (v.kind === 'Bool') return v.value;
    if (v.kind === 'Int') return v.value !== 0;
    if (v.kind === 'None') return false;
    return true;
  }

  private asTensor(v: NovaValue): NovaTensor | null {
    if (v.kind !== 'HeapRef') return null;
    const h = this.heap.deref(v).payload;
    return h.tag === 'Tensor' ? h.tensor : null;
  }

  private valuesEqual(a: NovaValue, b: NovaValue): boolean {
    if (a.kind === 'None' && b.kind === 'None') return true;
    if ((a.kind === 'Int' || a.kind === 'Float') && (b.kind === 'Int' || b.kind === 'Float')) {
      return a.value === b.value;
    }
    if (a.kind === 'Bool' && b.kind === 'Bool') return a.value === b.value;
    if (a.kind === 'Char' && b.kind === 'Char') return a.value === b.value;
    if (a.kind === 'HeapRef' && b.kind === 'HeapRef') {
      const ha = this.heap.deref(a).payload;
      const hb = this.heap.deref(b).payload;
      if (ha.tag === 'String' && hb.tag === 'String') return ha.value === hb.value;
      return a.id === b.id;
    }
    return false;
  }

  private toJS(v: NovaValue): unknown {
    switch (v.kind) {
      case 'Int':
      case 'Float':
      case 'Bool':
      case 'Char':
        return v.value;
      case 'None':
        return null;
      case 'HeapRef': {
        const p = this.heap.deref(v).payload;
        if (p.tag === 'String') return p.value;
        if (p.tag === 'Array') return p.elements.map((e) => this.toJS(e));
        if (p.tag === 'Map') {
          const out: Record<string, unknown> = {};
          for (const [k, val] of p.entries.entries()) out[k] = this.toJS(val);
          return out;
        }
        if (p.tag === 'Struct') {
          const out: Record<string, unknown> = {};
          for (const [k, val] of p.fields.entries()) out[k] = this.toJS(val);
          return out;
        }
        if (p.tag === 'Result') return { [p.variant]: this.toJS(p.value) };
        if (p.tag === 'Tensor') return p.tensor.toNested();
      }
    }
  }

  private fromJS(x: unknown): NovaValue {
    if (x === null || x === undefined) return { kind: 'None' };
    if (typeof x === 'boolean') return { kind: 'Bool', value: x };
    if (typeof x === 'number') {
      return Number.isInteger(x) ? { kind: 'Int', value: x } : { kind: 'Float', value: x };
    }
    if (typeof x === 'string') return this.heap.alloc({ tag: 'String', value: x });
    if (Array.isArray(x)) {
      return this.heap.alloc({ tag: 'Array', elements: x.map((item) => this.fromJS(item)) });
    }
    if (typeof x === 'object') {
      const entries = new Map<string, NovaValue>();
      for (const [k, val] of Object.entries(x as Record<string, unknown>)) {
        entries.set(k, this.fromJS(val));
      }
      return this.heap.alloc({ tag: 'Map', entries });
    }
    return { kind: 'None' };
  }

  private invokeMethod(obj: NovaValue, method: string, args: NovaValue[], span: Span): NovaValue {
    const h = this.heap.deref(obj).payload;
    if (h.tag === 'Tensor') {
      if (method === 'shape') return this.fromJS(h.tensor.shape);
      if (method === 'sum') return { kind: 'Float', value: h.tensor.sum() };
      if (method === 'mean') return { kind: 'Float', value: h.tensor.mean() };
      if (method === 'max') return { kind: 'Float', value: h.tensor.max() };
      if (method === 'min') return { kind: 'Float', value: h.tensor.min() };
      if (method === 'argmax') return { kind: 'Int', value: h.tensor.argmax() };
      if (method === 'transpose') return this.heap.alloc({ tag: 'Tensor', tensor: h.tensor.transpose() });
      if (method === 'reshape') {
        const shape = this.toJS(args[0]) as number[];
        return this.heap.alloc({ tag: 'Tensor', tensor: h.tensor.reshape(shape) });
      }
      if (method === 'relu') return this.heap.alloc({ tag: 'Tensor', tensor: h.tensor.relu() });
      if (method === 'sigmoid') return this.heap.alloc({ tag: 'Tensor', tensor: h.tensor.sigmoid() });
      if (method === 'tanh') return this.heap.alloc({ tag: 'Tensor', tensor: h.tensor.tanh() });
      if (method === 'softmax') return this.heap.alloc({ tag: 'Tensor', tensor: h.tensor.softmax() });
    }
    if (h.tag === 'Array') {
      if (method === 'len') return { kind: 'Int', value: h.elements.length };
      if (method === 'push') {
        h.elements.push(args[0] ?? { kind: 'None' });
        return { kind: 'None' };
      }
    }
    if (h.tag === 'String') {
      if (method === 'len') return { kind: 'Int', value: h.value.length };
      if (method === 'upper') return this.heap.alloc({ tag: 'String', value: h.value.toUpperCase() });
      if (method === 'lower') return this.heap.alloc({ tag: 'String', value: h.value.toLowerCase() });
      if (method === 'trim') return this.heap.alloc({ tag: 'String', value: h.value.trim() });
    }
    throw new RuntimeErrorWithSpan(`Unknown method '${method}' on ${h.tag}`, span);
  }

  private invokeBuiltin(rawName: string, args: NovaValue[], span: Span): NovaValue {
    const name = rawName.includes('.') ? rawName.split('.').pop()! : rawName;
    switch (name) {
      case 'print': {
        this.stdout.push(args.map((a) => this.heap.formatValue(a)).join(' '));
        return { kind: 'None' };
      }
      case 'str':
        return this.heap.alloc({ tag: 'String', value: this.heap.formatValue(args[0] ?? { kind: 'None' }) });
      case 'int': {
        const a = args[0];
        if (!a) return { kind: 'Int', value: 0 };
        if (a.kind === 'Int' || a.kind === 'Float') return { kind: 'Int', value: Math.trunc(a.value) };
        return { kind: 'Int', value: Number.parseInt(this.heap.formatValue(a), 10) || 0 };
      }
      case 'float': {
        const a = args[0];
        if (!a) return { kind: 'Float', value: 0 };
        if (a.kind === 'Int' || a.kind === 'Float') return { kind: 'Float', value: a.value };
        return { kind: 'Float', value: Number.parseFloat(this.heap.formatValue(a)) || 0 };
      }
      case 'len': {
        const a = args[0];
        if (!a || a.kind !== 'HeapRef') throw new RuntimeErrorWithSpan('len() requires String, Array, Map, or Tensor', span);
        const p = this.heap.deref(a).payload;
        if (p.tag === 'String') return { kind: 'Int', value: p.value.length };
        if (p.tag === 'Array') return { kind: 'Int', value: p.elements.length };
        if (p.tag === 'Map') return { kind: 'Int', value: p.entries.size };
        if (p.tag === 'Tensor') return { kind: 'Int', value: p.tensor.data.length };
        throw new RuntimeErrorWithSpan(`len() unsupported for ${p.tag}`, span);
      }
      case 'type_of': {
        const a = args[0];
        if (!a) return this.heap.alloc({ tag: 'String', value: 'None' });
        if (a.kind !== 'HeapRef') return this.heap.alloc({ tag: 'String', value: a.kind });
        const p = this.heap.deref(a).payload;
        return this.heap.alloc({ tag: 'String', value: p.tag === 'Struct' ? p.typeName : p.tag });
      }
      case 'is_ok': {
        const a = args[0];
        if (a?.kind === 'HeapRef') {
          const p = this.heap.deref(a).payload;
          return { kind: 'Bool', value: p.tag === 'Result' && p.variant === 'Ok' };
        }
        return { kind: 'Bool', value: false };
      }
      case 'is_error': {
        const a = args[0];
        if (a?.kind === 'HeapRef') {
          const p = this.heap.deref(a).payload;
          return { kind: 'Bool', value: p.tag === 'Result' && p.variant === 'Error' };
        }
        return { kind: 'Bool', value: false };
      }
      case 'unwrap': {
        const a = args[0];
        if (a?.kind === 'HeapRef') {
          const p = this.heap.deref(a).payload;
          if (p.tag === 'Result') {
            if (p.variant === 'Ok') return p.value;
            throw new RuntimeErrorWithSpan(`Unwrap called on Error(${this.heap.formatValue(p.value)})`, span);
          }
        }
        throw new RuntimeErrorWithSpan('unwrap() requires a Result<T, E> value', span);
      }
      case 'sqrt': return { kind: 'Float', value: Math.sqrt(this.asNumber(args[0], span)) };
      case 'abs': return { kind: 'Float', value: Math.abs(this.asNumber(args[0], span)) };
      case 'pow': return { kind: 'Float', value: Math.pow(this.asNumber(args[0], span), this.asNumber(args[1], span)) };
      case 'sin': return { kind: 'Float', value: Math.sin(this.asNumber(args[0], span)) };
      case 'cos': return { kind: 'Float', value: Math.cos(this.asNumber(args[0], span)) };
      case 'exp': return { kind: 'Float', value: Math.exp(this.asNumber(args[0], span)) };
      case 'log': return { kind: 'Float', value: Math.log(this.asNumber(args[0], span)) };
      case 'floor': return { kind: 'Int', value: Math.floor(this.asNumber(args[0], span)) };
      case 'ceil': return { kind: 'Int', value: Math.ceil(this.asNumber(args[0], span)) };
      case 'min': return { kind: 'Float', value: Math.min(this.asNumber(args[0], span), this.asNumber(args[1], span)) };
      case 'max': return { kind: 'Float', value: Math.max(this.asNumber(args[0], span), this.asNumber(args[1], span)) };
      case 'upper': return this.heap.alloc({ tag: 'String', value: this.heap.formatValue(args[0]).toUpperCase() });
      case 'lower': return this.heap.alloc({ tag: 'String', value: this.heap.formatValue(args[0]).toLowerCase() });
      case 'trim': return this.heap.alloc({ tag: 'String', value: this.heap.formatValue(args[0]).trim() });
      case 'contains': return { kind: 'Bool', value: this.heap.formatValue(args[0]).includes(this.heap.formatValue(args[1])) };
      case 'split': return this.fromJS(this.heap.formatValue(args[0]).split(this.heap.formatValue(args[1])));
      case 'range': {
        const s = Math.trunc(this.asNumber(args[0], span));
        const e = Math.trunc(this.asNumber(args[1], span));
        const els: NovaValue[] = [];
        for (let i = s; i < e; i++) els.push({ kind: 'Int', value: i });
        return this.heap.alloc({ tag: 'Array', elements: els });
      }
      case 'sum': {
        const arr = this.toJS(args[0]);
        if (!Array.isArray(arr)) throw new RuntimeErrorWithSpan('sum() requires an Array', span);
        return { kind: 'Float', value: arr.reduce((acc, v) => acc + Number(v), 0) };
      }
      case 'stringify': return this.heap.alloc({ tag: 'String', value: JSON.stringify(this.toJS(args[0])) });
      case 'parse': return this.fromJS(JSON.parse(this.heap.formatValue(args[0])));
      case 'now_ms': return { kind: 'Int', value: Date.now() };
      case 'tensor': return this.heap.alloc({ tag: 'Tensor', tensor: NovaTensor.fromNested(this.toJS(args[0])) });
      case 'zeros': return this.heap.alloc({ tag: 'Tensor', tensor: NovaTensor.zeros(this.toJS(args[0]) as number[]) });
      case 'ones': return this.heap.alloc({ tag: 'Tensor', tensor: NovaTensor.ones(this.toJS(args[0]) as number[]) });
      case 'full': {
        const shape = this.toJS(args[0]) as number[];
        const fillVal = this.asNumber(args[1], span);
        return this.heap.alloc({ tag: 'Tensor', tensor: NovaTensor.full(shape, fillVal) });
      }
      case 'randn': {
        const shape = this.toJS(args[0]) as number[];
        const seed = args[1] ? Math.trunc(this.asNumber(args[1], span)) : 42;
        return this.heap.alloc({ tag: 'Tensor', tensor: NovaTensor.randUniform(shape, seed, 1.0) });
      }
      case 'matmul': {
        const a = this.asTensor(args[0]);
        const b = this.asTensor(args[1]);
        if (!a || !b) throw new RuntimeErrorWithSpan('matmul() requires two Tensors', span);
        return this.heap.alloc({ tag: 'Tensor', tensor: a.matmul(b) });
      }
      case 'dot': {
        const a = this.asTensor(args[0]);
        const b = this.asTensor(args[1]);
        if (!a || !b) throw new RuntimeErrorWithSpan('dot() requires two Tensors', span);
        return { kind: 'Float', value: a.dot(b) };
      }
      case 'transpose': {
        const a = this.asTensor(args[0]);
        if (!a) throw new RuntimeErrorWithSpan('transpose() requires a Tensor', span);
        return this.heap.alloc({ tag: 'Tensor', tensor: a.transpose() });
      }
      case 'reshape': {
        const a = this.asTensor(args[0]);
        if (!a) throw new RuntimeErrorWithSpan('reshape() requires a Tensor', span);
        const shape = this.toJS(args[1]) as number[];
        return this.heap.alloc({ tag: 'Tensor', tensor: a.reshape(shape) });
      }
      case 'relu': {
        const a = this.asTensor(args[0]);
        if (!a) throw new RuntimeErrorWithSpan('relu() requires a Tensor', span);
        return this.heap.alloc({ tag: 'Tensor', tensor: a.relu() });
      }
      case 'sigmoid': {
        const a = this.asTensor(args[0]);
        if (!a) throw new RuntimeErrorWithSpan('sigmoid() requires a Tensor', span);
        return this.heap.alloc({ tag: 'Tensor', tensor: a.sigmoid() });
      }
      case 'tanh': {
        const a = this.asTensor(args[0]);
        if (!a) throw new RuntimeErrorWithSpan('tanh() requires a Tensor', span);
        return this.heap.alloc({ tag: 'Tensor', tensor: a.tanh() });
      }
      case 'softmax': {
        const a = this.asTensor(args[0]);
        if (!a) throw new RuntimeErrorWithSpan('softmax() requires a Tensor', span);
        return this.heap.alloc({ tag: 'Tensor', tensor: a.softmax() });
      }
      case 'mse_loss': {
        const a = this.asTensor(args[0]);
        const b = this.asTensor(args[1]);
        if (!a || !b) throw new RuntimeErrorWithSpan('mse_loss() requires two Tensors', span);
        return { kind: 'Float', value: mseLoss(a, b) };
      }
      case 'bce_loss': {
        const a = this.asTensor(args[0]);
        const b = this.asTensor(args[1]);
        if (!a || !b) throw new RuntimeErrorWithSpan('bce_loss() requires two Tensors', span);
        return { kind: 'Float', value: bceLoss(a, b) };
      }
      case 'grad_var': {
        const t = this.asTensor(args[0]) ?? NovaTensor.fromNested(this.toJS(args[0]));
        return this.heap.alloc({ tag: 'GradVar', gradVar: GradVar.leaf(t) });
      }
      case 'grad_add': {
        const a = this.asGradVar(args[0], span);
        const b = this.asGradVar(args[1], span);
        return this.heap.alloc({ tag: 'GradVar', gradVar: a.add(b) });
      }
      case 'grad_sub': {
        const a = this.asGradVar(args[0], span);
        const b = this.asGradVar(args[1], span);
        return this.heap.alloc({ tag: 'GradVar', gradVar: a.sub(b) });
      }
      case 'grad_mul': {
        const a = this.asGradVar(args[0], span);
        const b = this.asGradVar(args[1], span);
        return this.heap.alloc({ tag: 'GradVar', gradVar: a.mul(b) });
      }
      case 'grad_matmul': {
        const a = this.asGradVar(args[0], span);
        const b = this.asGradVar(args[1], span);
        return this.heap.alloc({ tag: 'GradVar', gradVar: a.matmul(b) });
      }
      case 'grad_relu':
        return this.heap.alloc({ tag: 'GradVar', gradVar: this.asGradVar(args[0], span).relu() });
      case 'grad_sigmoid':
        return this.heap.alloc({ tag: 'GradVar', gradVar: this.asGradVar(args[0], span).sigmoid() });
      case 'grad_tanh':
        return this.heap.alloc({ tag: 'GradVar', gradVar: this.asGradVar(args[0], span).tanh() });
      case 'grad_mse': {
        const pred = this.asGradVar(args[0], span);
        const target = this.asTensor(args[1]) ?? NovaTensor.fromNested(this.toJS(args[1]));
        return this.heap.alloc({ tag: 'GradVar', gradVar: pred.mseLoss(target) });
      }
      case 'backward': {
        this.asGradVar(args[0], span).backward();
        return { kind: 'None' };
      }
      case 'grad_of': {
        const gv = this.asGradVar(args[0], span);
        return this.heap.alloc({ tag: 'Tensor', tensor: new NovaTensor(gv.grad.shape, gv.grad.data) });
      }
      case 'value_of': {
        const gv = this.asGradVar(args[0], span);
        return this.heap.alloc({ tag: 'Tensor', tensor: new NovaTensor(gv.data.shape, gv.data.data) });
      }
      case 'zero_grad': {
        this.asGradVar(args[0], span).zeroGrad();
        return { kind: 'None' };
      }
      case 'model_new':
        return this.heap.alloc({ tag: 'Model', model: new SequentialModel() });
      case 'model_add_linear': {
        const m = this.asModel(args[0], span);
        const inF = Math.trunc(this.asNumber(args[1], span));
        const outF = Math.trunc(this.asNumber(args[2], span));
        const seed = args[3] ? Math.trunc(this.asNumber(args[3], span)) : 42;
        m.addLinear(inF, outF, seed);
        return args[0];
      }
      case 'model_add_activation': {
        const m = this.asModel(args[0], span);
        const act = this.heap.formatValue(args[1]).toLowerCase() as ActivationKind;
        if (act !== 'relu' && act !== 'sigmoid' && act !== 'tanh') {
          throw new RuntimeErrorWithSpan(`Unsupported activation '${act}'`, span);
        }
        m.addActivation(act);
        return args[0];
      }
      case 'model_forward': {
        const m = this.asModel(args[0], span);
        const x = this.asTensor(args[1]) ?? NovaTensor.fromNested(this.toJS(args[1]));
        return this.heap.alloc({ tag: 'Tensor', tensor: m.predict(x) });
      }
      case 'model_param_count':
        return { kind: 'Int', value: this.asModel(args[0], span).paramCount() };
      case 'optimizer_sgd': {
        const lr = this.asNumber(args[0], span);
        const momentum = args[1] ? this.asNumber(args[1], span) : 0.0;
        return this.heap.alloc({ tag: 'Optimizer', optimizer: new SGDOptimizer(lr, momentum) });
      }
      case 'optimizer_adam': {
        const lr = this.asNumber(args[0], span);
        return this.heap.alloc({ tag: 'Optimizer', optimizer: new AdamOptimizer(lr) });
      }
      case 'train_step': {
        const m = this.asModel(args[0], span);
        const opt = this.asOptimizer(args[1], span);
        const x = this.asTensor(args[2]) ?? NovaTensor.fromNested(this.toJS(args[2]));
        const y = this.asTensor(args[3]) ?? NovaTensor.fromNested(this.toJS(args[3]));
        const lossName = args[4] ? this.heap.formatValue(args[4]).toLowerCase() : 'mse';
        const lossVal = trainStep(m, opt, x, y, lossName === 'bce' ? 'bce' : 'mse');
        return { kind: 'Float', value: lossVal };
      }
      case 'train_model': {
        const m = this.asModel(args[0], span);
        const opt = this.asOptimizer(args[1], span);
        const x = this.asTensor(args[2]) ?? NovaTensor.fromNested(this.toJS(args[2]));
        const y = this.asTensor(args[3]) ?? NovaTensor.fromNested(this.toJS(args[3]));
        const epochs = Math.trunc(this.asNumber(args[4], span));
        const lossName = args[5] ? this.heap.formatValue(args[5]).toLowerCase() : 'mse';
        const report = trainModel(m, opt, x, y, epochs, lossName === 'bce' ? 'bce' : 'mse');
        return this.fromJS(report);
      }
      case 'autograd_demo': {
        const x = new AutogradNode(this.asNumber(args[0], span));
        const w = new AutogradNode(this.asNumber(args[1], span));
        const b = new AutogradNode(1.5);
        const y = x.mul(w).add(b).relu();
        y.backward();
        return this.fromJS({
          output: Number(y.data.toFixed(4)),
          dx: Number(x.grad.toFixed(4)),
          dw: Number(w.grad.toFixed(4)),
          db: Number(b.grad.toFixed(4)),
        });
      }
      case 'train_xor': {
        const summary = trainXorNetwork(Math.trunc(this.asNumber(args[0], span)), this.asNumber(args[1], span));
        return this.fromJS(summary);
      }
      default:
        throw new RuntimeErrorWithSpan(`Runtime error: unknown builtin function '${rawName}'`, span);
    }
  }

  private asGradVar(v: NovaValue | undefined, span: Span): GradVar {
    if (v?.kind === 'HeapRef') {
      const p = this.heap.deref(v).payload;
      if (p.tag === 'GradVar') return p.gradVar;
    }
    throw new RuntimeErrorWithSpan('Expected GradVar', span);
  }

  private asModel(v: NovaValue | undefined, span: Span): SequentialModel {
    if (v?.kind === 'HeapRef') {
      const p = this.heap.deref(v).payload;
      if (p.tag === 'Model') return p.model;
    }
    throw new RuntimeErrorWithSpan('Expected Sequential model', span);
  }

  private asOptimizer(v: NovaValue | undefined, span: Span): NovaOptimizer {
    if (v?.kind === 'HeapRef') {
      const p = this.heap.deref(v).payload;
      if (p.tag === 'Optimizer') return p.optimizer;
    }
    throw new RuntimeErrorWithSpan('Expected Optimizer', span);
  }
}

export function interpretProgram(program: Program): VMExecutionResult {
  const interpreter = new NovaInterpreter();
  return interpreter.run(program);
}
