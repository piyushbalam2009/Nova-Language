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
import { BytecodeChunk, CompiledFunction, decodeBytecode, OpCode } from './bytecode';
import { Diagnostic } from './diagnostics';
import { GCStats, NovaHeap, NovaValue } from './gc';
import { MAX_VERIFIED_STACK_DEPTH, verifyBytecode } from './verifier';

interface CallFrame {
  fn: CompiledFunction;
  ip: number;
  locals: NovaValue[];
}

export interface VMRunOptions {
  /** Run static bytecode verifier before execution (default: true). */
  verify?: boolean;
  maxInstructions?: number;
  maxCallDepth?: number;
  maxStackDepth?: number;
}

export interface VMExecutionResult {
  stdout: string[];
  exitCode: number;
  runtimeError?: Diagnostic;
  callStack?: string[];
  instructionsExecuted: number;
  durationMs: number;
  gcStats: GCStats;
}

export class NovaVM {
  private stack: NovaValue[] = [];
  private frames: CallFrame[] = [];
  private globals = new Map<string, NovaValue>();
  private heap = new NovaHeap();
  private stdout: string[] = [];
  private instructionsExecuted = 0;
  private maxInstructions = 250_000;
  private maxCallDepth = 256;
  private maxStackDepth = MAX_VERIFIED_STACK_DEPTH;

  /**
   * Decodes a binary `.nvbc` buffer, verifies its bytecode invariants, and executes it on the VM.
   */
  runBinary(bytes: Uint8Array, options: VMRunOptions = {}): VMExecutionResult {
    const startMs = performance.now();
    try {
      const chunk = decodeBytecode(bytes);
      return this.run(chunk, options);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return {
        stdout: [],
        exitCode: 1,
        runtimeError: {
          stage: 'bytecode',
          severity: 'error',
          code: 'E0401',
          message: `Failed to decode .nvbc bytecode: ${message}`,
          span: { start: 0, end: 0, line: 1, column: 1 },
        },
        instructionsExecuted: 0,
        durationMs: Number((performance.now() - startMs).toFixed(3)),
        gcStats: new NovaHeap().getStats(),
      };
    }
  }

  run(chunk: BytecodeChunk, options: VMRunOptions = {}): VMExecutionResult {
    const startMs = performance.now();
    this.stack = [];
    this.frames = [];
    this.globals.clear();
    this.heap = new NovaHeap();
    this.stdout = [];
    this.instructionsExecuted = 0;
    this.maxInstructions = options.maxInstructions ?? 250_000;
    this.maxCallDepth = options.maxCallDepth ?? 256;
    this.maxStackDepth = options.maxStackDepth ?? MAX_VERIFIED_STACK_DEPTH;

    if (options.verify !== false) {
      const verification = verifyBytecode(chunk);
      if (!verification.ok) {
        return {
          stdout: [],
          exitCode: 1,
          runtimeError: verification.diagnostics[0],
          instructionsExecuted: 0,
          durationMs: Number((performance.now() - startMs).toFixed(3)),
          gcStats: this.heap.getStats(),
        };
      }
    }

    try {
      // Execute top-level __init__ if present
      if (chunk.functionIndex['__init__'] !== undefined) {
        this.callUserFunction(chunk, '__init__', []);
        this.executeLoop(chunk);
        if (this.stack.length > 0) this.stack.pop();
      }

      // Execute main() if present
      if (chunk.functionIndex['main'] !== undefined) {
        this.callUserFunction(chunk, 'main', []);
        this.executeLoop(chunk);
      }

      // Trigger final mark-and-sweep cycle so GC stats reflect real cleanup
      this.collectGarbage();

      return {
        stdout: this.stdout,
        exitCode: 0,
        instructionsExecuted: this.instructionsExecuted,
        durationMs: Number((performance.now() - startMs).toFixed(3)),
        gcStats: this.heap.getStats(),
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const callStack = this.frames.map((f) => `${f.fn.name}@ip:${Math.max(0, f.ip - 1)}`);
      return {
        stdout: this.stdout,
        exitCode: 1,
        runtimeError: {
          stage: 'vm',
          severity: 'error',
          code: 'E0501',
          message,
          span: { start: 0, end: 0, line: 1, column: 1 },
        },
        callStack,
        instructionsExecuted: this.instructionsExecuted,
        durationMs: Number((performance.now() - startMs).toFixed(3)),
        gcStats: this.heap.getStats(),
      };
    }
  }

  private collectGarbage(): void {
    const roots: NovaValue[] = [...this.stack, ...this.globals.values()];
    for (const f of this.frames) {
      for (const l of f.locals) roots.push(l);
    }
    this.heap.collect(roots);
  }

  private maybeGC(): void {
    if (this.heap.shouldCollect()) {
      this.collectGarbage();
    }
  }

  private callUserFunction(chunk: BytecodeChunk, name: string, args: NovaValue[]): void {
    const fnIdx = chunk.functionIndex[name];
    if (fnIdx === undefined) {
      throw new Error(`Runtime error: undefined function '${name}'`);
    }
    if (this.frames.length >= this.maxCallDepth) {
      throw new Error(`Stack overflow: maximum recursion depth (${this.maxCallDepth}) exceeded in '${name}'`);
    }
    const fn = chunk.functions[fnIdx];
    const locals: NovaValue[] = Array.from({ length: fn.localCount }, () => ({ kind: 'None' }));
    for (let i = 0; i < Math.min(fn.arity, args.length); i++) {
      locals[i] = args[i];
    }
    this.frames.push({ fn, ip: 0, locals });
  }

  private push(v: NovaValue): void {
    if (this.stack.length >= this.maxStackDepth) {
      throw new Error(`VM operand stack overflow (limit ${this.maxStackDepth})`);
    }
    this.stack.push(v);
  }

  private pop(): NovaValue {
    const v = this.stack.pop();
    if (!v) throw new Error('VM operand stack underflow');
    return v;
  }

  private asNumber(v: NovaValue): number {
    if (v.kind === 'Int' || v.kind === 'Float') return v.value;
    throw new Error(`Expected numeric value, got ${v.kind}`);
  }

  private asBool(v: NovaValue): boolean {
    if (v.kind === 'Bool') return v.value;
    if (v.kind === 'Int') return v.value !== 0;
    if (v.kind === 'None') return false;
    return true;
  }

  private asString(v: NovaValue): string {
    return this.heap.formatValue(v);
  }

  private asTensor(v: NovaValue): NovaTensor | null {
    if (v.kind !== 'HeapRef') return null;
    const obj = this.heap.deref(v);
    return obj.payload.tag === 'Tensor' ? obj.payload.tensor : null;
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
        const obj = this.heap.deref(v);
        const p = obj.payload;
        switch (p.tag) {
          case 'String': return p.value;
          case 'Array': return p.elements.map((e) => this.toJS(e));
          case 'Map': {
            const out: Record<string, unknown> = {};
            for (const [k, val] of p.entries.entries()) out[k] = this.toJS(val);
            return out;
          }
          case 'Struct': {
            const out: Record<string, unknown> = {};
            for (const [k, val] of p.fields.entries()) out[k] = this.toJS(val);
            return out;
          }
          case 'Result':
            return { [p.variant]: this.toJS(p.value) };
          case 'Tensor':
            return p.tensor.toNested();
        }
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

  private executeLoop(chunk: BytecodeChunk): void {
    while (this.frames.length > 0) {
      const frame = this.frames[this.frames.length - 1];
      if (frame.ip >= frame.fn.code.length) {
        this.frames.pop();
        this.stack.push({ kind: 'None' });
        continue;
      }

      if (++this.instructionsExecuted > this.maxInstructions) {
        throw new Error(`Execution limit exceeded (${this.maxInstructions} bytecode instructions). Check for infinite loops.`);
      }
      if (this.stack.length > this.maxStackDepth) {
        throw new Error(`VM operand stack overflow (limit ${this.maxStackDepth})`);
      }

      const inst = frame.fn.code[frame.ip++];

      switch (inst.op) {
        case OpCode.PUSH_CONST: {
          const c = chunk.constants[Number(inst.arg)];
          switch (c.kind) {
            case 'Int': this.push({ kind: 'Int', value: c.value }); break;
            case 'Float': this.push({ kind: 'Float', value: c.value }); break;
            case 'Bool': this.push({ kind: 'Bool', value: c.value }); break;
            case 'Char': this.push({ kind: 'Char', value: c.value }); break;
            case 'None': this.push({ kind: 'None' }); break;
            case 'String':
              this.maybeGC();
              this.push(this.heap.alloc({ tag: 'String', value: c.value }));
              break;
          }
          break;
        }

        case OpCode.LOAD_LOCAL: {
          this.push(frame.locals[Number(inst.arg)] ?? { kind: 'None' });
          break;
        }

        case OpCode.STORE_LOCAL: {
          frame.locals[Number(inst.arg)] = this.pop();
          break;
        }

        case OpCode.LOAD_GLOBAL: {
          const name = String(inst.arg);
          const val = this.globals.get(name);
          if (!val) throw new Error(`Undefined global variable '${name}'`);
          this.stack.push(val);
          break;
        }

        case OpCode.STORE_GLOBAL: {
          this.globals.set(String(inst.arg), this.pop());
          break;
        }

        case OpCode.ADD: {
          const r = this.pop();
          const l = this.pop();
          const lt = this.asTensor(l);
          const rt = this.asTensor(r);
          if (lt || rt) {
            this.maybeGC();
            const res = lt ? lt.add(rt ?? this.asNumber(r)) : rt!.add(this.asNumber(l));
            this.stack.push(this.heap.alloc({ tag: 'Tensor', tensor: res }));
          } else if (
            (l.kind === 'HeapRef' && this.heap.deref(l).payload.tag === 'String') ||
            (r.kind === 'HeapRef' && this.heap.deref(r).payload.tag === 'String')
          ) {
            this.maybeGC();
            this.stack.push(this.heap.alloc({ tag: 'String', value: this.asString(l) + this.asString(r) }));
          } else if (l.kind === 'Int' && r.kind === 'Int') {
            this.stack.push({ kind: 'Int', value: l.value + r.value });
          } else {
            this.stack.push({ kind: 'Float', value: this.asNumber(l) + this.asNumber(r) });
          }
          break;
        }

        case OpCode.SUB: {
          const r = this.pop();
          const l = this.pop();
          const lt = this.asTensor(l);
          const rt = this.asTensor(r);
          if (lt || rt) {
            this.maybeGC();
            const res = lt ? lt.sub(rt ?? this.asNumber(r)) : new NovaTensor([1], [this.asNumber(l)]).sub(rt!);
            this.stack.push(this.heap.alloc({ tag: 'Tensor', tensor: res }));
          } else if (l.kind === 'Int' && r.kind === 'Int') {
            this.stack.push({ kind: 'Int', value: l.value - r.value });
          } else {
            this.stack.push({ kind: 'Float', value: this.asNumber(l) - this.asNumber(r) });
          }
          break;
        }

        case OpCode.MUL: {
          const r = this.pop();
          const l = this.pop();
          const lt = this.asTensor(l);
          const rt = this.asTensor(r);
          if (lt || rt) {
            this.maybeGC();
            const res = lt ? lt.mul(rt ?? this.asNumber(r)) : rt!.mul(this.asNumber(l));
            this.stack.push(this.heap.alloc({ tag: 'Tensor', tensor: res }));
          } else if (l.kind === 'Int' && r.kind === 'Int') {
            this.stack.push({ kind: 'Int', value: l.value * r.value });
          } else {
            this.stack.push({ kind: 'Float', value: this.asNumber(l) * this.asNumber(r) });
          }
          break;
        }

        case OpCode.DIV: {
          const r = this.pop();
          const l = this.pop();
          const lt = this.asTensor(l);
          const rt = this.asTensor(r);
          if (lt || rt) {
            this.maybeGC();
            const res = lt ? lt.div(rt ?? this.asNumber(r)) : new NovaTensor([1], [this.asNumber(l)]).div(rt!);
            this.stack.push(this.heap.alloc({ tag: 'Tensor', tensor: res }));
          } else {
            const rv = this.asNumber(r);
            if (rv === 0) throw new Error('Runtime error: division by zero');
            if (l.kind === 'Int' && r.kind === 'Int') {
              this.stack.push({ kind: 'Int', value: Math.trunc(l.value / rv) });
            } else {
              this.stack.push({ kind: 'Float', value: this.asNumber(l) / rv });
            }
          }
          break;
        }

        case OpCode.MOD: {
          const r = this.pop();
          const l = this.pop();
          const rv = this.asNumber(r);
          if (rv === 0) throw new Error('Runtime error: modulo by zero');
          if (l.kind === 'Int' && r.kind === 'Int') {
            this.stack.push({ kind: 'Int', value: l.value % rv });
          } else {
            this.stack.push({ kind: 'Float', value: this.asNumber(l) % rv });
          }
          break;
        }

        case OpCode.MATMUL: {
          const r = this.pop();
          const l = this.pop();
          const lt = this.asTensor(l);
          const rt = this.asTensor(r);
          if (!lt || !rt) throw new Error("Operator '@' requires two Tensor values");
          this.maybeGC();
          this.stack.push(this.heap.alloc({ tag: 'Tensor', tensor: lt.matmul(rt) }));
          break;
        }

        case OpCode.NEG: {
          const v = this.pop();
          const t = this.asTensor(v);
          if (t) {
            this.stack.push(this.heap.alloc({ tag: 'Tensor', tensor: t.mul(-1) }));
          } else if (v.kind === 'Int') {
            this.stack.push({ kind: 'Int', value: -v.value });
          } else {
            this.stack.push({ kind: 'Float', value: -this.asNumber(v) });
          }
          break;
        }

        case OpCode.NOT: {
          this.stack.push({ kind: 'Bool', value: !this.asBool(this.pop()) });
          break;
        }

        case OpCode.AND: {
          const r = this.pop();
          const l = this.pop();
          this.stack.push({ kind: 'Bool', value: this.asBool(l) && this.asBool(r) });
          break;
        }

        case OpCode.OR: {
          const r = this.pop();
          const l = this.pop();
          this.stack.push({ kind: 'Bool', value: this.asBool(l) || this.asBool(r) });
          break;
        }

        case OpCode.CMP_EQ:
        case OpCode.CMP_NEQ:
        case OpCode.CMP_LT:
        case OpCode.CMP_LTE:
        case OpCode.CMP_GT:
        case OpCode.CMP_GTE: {
          const r = this.pop();
          const l = this.pop();
          let res = false;
          if (inst.op === OpCode.CMP_EQ || inst.op === OpCode.CMP_NEQ) {
            const eq = this.valuesEqual(l, r);
            res = inst.op === OpCode.CMP_EQ ? eq : !eq;
          } else if (l.kind === 'HeapRef' && r.kind === 'HeapRef') {
            const ls = this.asString(l);
            const rs = this.asString(r);
            res = inst.op === OpCode.CMP_LT ? ls < rs : inst.op === OpCode.CMP_LTE ? ls <= rs : inst.op === OpCode.CMP_GT ? ls > rs : ls >= rs;
          } else {
            const lv = this.asNumber(l);
            const rv = this.asNumber(r);
            res = inst.op === OpCode.CMP_LT ? lv < rv : inst.op === OpCode.CMP_LTE ? lv <= rv : inst.op === OpCode.CMP_GT ? lv > rv : lv >= rv;
          }
          this.stack.push({ kind: 'Bool', value: res });
          break;
        }

        case OpCode.JUMP: {
          frame.ip = Number(inst.arg);
          break;
        }

        case OpCode.JUMP_IF_FALSE: {
          const cond = this.pop();
          if (!this.asBool(cond)) {
            frame.ip = Number(inst.arg);
          }
          break;
        }

        case OpCode.CALL: {
          const callee = String(inst.arg);
          const argc = Number(inst.arg2 ?? 0);
          const args: NovaValue[] = new Array(argc);
          for (let i = argc - 1; i >= 0; i--) args[i] = this.pop();

          if (chunk.functionIndex[callee] !== undefined) {
            this.callUserFunction(chunk, callee, args);
          } else {
            const retVal = this.invokeBuiltin(callee, args);
            this.stack.push(retVal);
          }
          break;
        }

        case OpCode.CALL_METHOD: {
          const method = String(inst.arg);
          const argc = Number(inst.arg2 ?? 0);
          const args: NovaValue[] = new Array(argc);
          for (let i = argc - 1; i >= 0; i--) args[i] = this.pop();
          const obj = this.pop();
          this.stack.push(this.invokeMethod(obj, method, args));
          break;
        }

        case OpCode.RETURN: {
          const retVal = this.pop();
          this.frames.pop();
          this.stack.push(retVal);
          break;
        }

        case OpCode.ALLOC_STRUCT: {
          const typeName = String(inst.arg);
          const fieldNames = inst.fieldNames ?? [];
          const values: NovaValue[] = new Array(fieldNames.length);
          for (let i = fieldNames.length - 1; i >= 0; i--) values[i] = this.pop();
          const fields = new Map<string, NovaValue>();
          fieldNames.forEach((name, idx) => fields.set(name, values[idx]));
          this.maybeGC();
          this.stack.push(this.heap.alloc({ tag: 'Struct', typeName, fields }));
          break;
        }

        case OpCode.LOAD_FIELD: {
          const obj = this.pop();
          const field = String(inst.arg);
          const heapObj = this.heap.deref(obj);
          if (heapObj.payload.tag === 'Struct') {
            const val = heapObj.payload.fields.get(field);
            if (!val) throw new Error(`Struct '${heapObj.payload.typeName}' has no field '${field}'`);
            this.stack.push(val);
          } else if (heapObj.payload.tag === 'Map') {
            this.stack.push(heapObj.payload.entries.get(field) ?? { kind: 'None' });
          } else {
            throw new Error(`Cannot access field '${field}' on ${heapObj.payload.tag}`);
          }
          break;
        }

        case OpCode.STORE_FIELD: {
          const val = this.pop();
          const obj = this.pop();
          const field = String(inst.arg);
          const heapObj = this.heap.deref(obj);
          if (heapObj.payload.tag === 'Struct') {
            heapObj.payload.fields.set(field, val);
          } else if (heapObj.payload.tag === 'Map') {
            heapObj.payload.entries.set(field, val);
          } else {
            throw new Error(`Cannot assign field '${field}' on ${heapObj.payload.tag}`);
          }
          break;
        }

        case OpCode.ALLOC_ARRAY: {
          const count = Number(inst.arg ?? 0);
          const elements: NovaValue[] = new Array(count);
          for (let i = count - 1; i >= 0; i--) elements[i] = this.pop();
          this.maybeGC();
          this.stack.push(this.heap.alloc({ tag: 'Array', elements }));
          break;
        }

        case OpCode.ALLOC_MAP: {
          const count = Number(inst.arg ?? 0);
          const pairs: Array<[NovaValue, NovaValue]> = new Array(count);
          for (let i = count - 1; i >= 0; i--) {
            const v = this.pop();
            const k = this.pop();
            pairs[i] = [k, v];
          }
          const entries = new Map<string, NovaValue>();
          for (const [k, v] of pairs) {
            entries.set(this.asString(k), v);
          }
          this.maybeGC();
          this.stack.push(this.heap.alloc({ tag: 'Map', entries }));
          break;
        }

        case OpCode.INDEX_GET: {
          const idx = this.pop();
          const obj = this.pop();
          const heapObj = this.heap.deref(obj);
          if (heapObj.payload.tag === 'Array') {
            const i = this.asNumber(idx);
            if (i < 0 || i >= heapObj.payload.elements.length) {
              throw new Error(`Array index ${i} out of bounds (len=${heapObj.payload.elements.length})`);
            }
            this.stack.push(heapObj.payload.elements[i]);
          } else if (heapObj.payload.tag === 'Map') {
            this.stack.push(heapObj.payload.entries.get(this.asString(idx)) ?? { kind: 'None' });
          } else if (heapObj.payload.tag === 'String') {
            const i = this.asNumber(idx);
            const ch = heapObj.payload.value[i];
            if (ch === undefined) throw new Error(`String index ${i} out of bounds`);
            this.stack.push({ kind: 'Char', value: ch });
          } else if (heapObj.payload.tag === 'Tensor') {
            const i = this.asNumber(idx);
            if (i < 0 || i >= heapObj.payload.tensor.data.length) {
              throw new Error(`Tensor flat index ${i} out of bounds`);
            }
            this.stack.push({ kind: 'Float', value: heapObj.payload.tensor.data[i] });
          } else {
            throw new Error(`Cannot index into ${heapObj.payload.tag}`);
          }
          break;
        }

        case OpCode.INDEX_SET: {
          const val = this.pop();
          const idx = this.pop();
          const obj = this.pop();
          const heapObj = this.heap.deref(obj);
          if (heapObj.payload.tag === 'Array') {
            const i = this.asNumber(idx);
            if (i < 0 || i >= heapObj.payload.elements.length) {
              throw new Error(`Array index ${i} out of bounds`);
            }
            heapObj.payload.elements[i] = val;
          } else if (heapObj.payload.tag === 'Map') {
            heapObj.payload.entries.set(this.asString(idx), val);
          } else {
            throw new Error(`Cannot assign index on ${heapObj.payload.tag}`);
          }
          break;
        }

        case OpCode.MAKE_OK: {
          const val = this.pop();
          this.maybeGC();
          this.stack.push(this.heap.alloc({ tag: 'Result', variant: 'Ok', value: val }));
          break;
        }

        case OpCode.MAKE_ERROR: {
          const val = this.pop();
          this.maybeGC();
          this.stack.push(this.heap.alloc({ tag: 'Result', variant: 'Error', value: val }));
          break;
        }

        case OpCode.POP: {
          this.pop();
          break;
        }
      }
    }
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

  private invokeMethod(obj: NovaValue, method: string, args: NovaValue[]): NovaValue {
    const h = this.heap.deref(obj).payload;
    if (h.tag === 'Tensor') {
      switch (method) {
        case 'shape': return this.fromJS(h.tensor.shape);
        case 'sum': return { kind: 'Float', value: h.tensor.sum() };
        case 'mean': return { kind: 'Float', value: h.tensor.mean() };
        case 'max': return { kind: 'Float', value: h.tensor.max() };
        case 'min': return { kind: 'Float', value: h.tensor.min() };
        case 'argmax': return { kind: 'Int', value: h.tensor.argmax() };
        case 'transpose': return this.heap.alloc({ tag: 'Tensor', tensor: h.tensor.transpose() });
        case 'reshape': {
          const shape = this.toJS(args[0]) as number[];
          return this.heap.alloc({ tag: 'Tensor', tensor: h.tensor.reshape(shape) });
        }
        case 'relu': return this.heap.alloc({ tag: 'Tensor', tensor: h.tensor.relu() });
        case 'sigmoid': return this.heap.alloc({ tag: 'Tensor', tensor: h.tensor.sigmoid() });
        case 'tanh': return this.heap.alloc({ tag: 'Tensor', tensor: h.tensor.tanh() });
        case 'softmax': return this.heap.alloc({ tag: 'Tensor', tensor: h.tensor.softmax() });
      }
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
    throw new Error(`Unknown method '${method}' on ${h.tag}`);
  }

  private invokeBuiltin(rawName: string, args: NovaValue[]): NovaValue {
    const name = rawName.includes('.') ? rawName.split('.').pop()! : rawName;
    switch (name) {
      case 'print': {
        const line = args.map((a) => this.heap.formatValue(a)).join(' ');
        this.stdout.push(line);
        return { kind: 'None' };
      }
      case 'str':
        return this.heap.alloc({ tag: 'String', value: this.heap.formatValue(args[0] ?? { kind: 'None' }) });
      case 'int': {
        const a = args[0];
        if (!a) return { kind: 'Int', value: 0 };
        if (a.kind === 'Int' || a.kind === 'Float') return { kind: 'Int', value: Math.trunc(a.value) };
        return { kind: 'Int', value: Number.parseInt(this.asString(a), 10) || 0 };
      }
      case 'float': {
        const a = args[0];
        if (!a) return { kind: 'Float', value: 0 };
        if (a.kind === 'Int' || a.kind === 'Float') return { kind: 'Float', value: a.value };
        return { kind: 'Float', value: Number.parseFloat(this.asString(a)) || 0 };
      }
      case 'len': {
        const a = args[0];
        if (!a || a.kind !== 'HeapRef') throw new Error('len() requires String, Array, Map, or Tensor');
        const p = this.heap.deref(a).payload;
        if (p.tag === 'String') return { kind: 'Int', value: p.value.length };
        if (p.tag === 'Array') return { kind: 'Int', value: p.elements.length };
        if (p.tag === 'Map') return { kind: 'Int', value: p.entries.size };
        if (p.tag === 'Tensor') return { kind: 'Int', value: p.tensor.data.length };
        throw new Error(`len() unsupported for ${p.tag}`);
      }
      case 'type_of': {
        const a = args[0];
        if (!a) return this.heap.alloc({ tag: 'String', value: 'None' });
        if (a.kind !== 'HeapRef') return this.heap.alloc({ tag: 'String', value: a.kind });
        const p = this.heap.deref(a).payload;
        const label = p.tag === 'Struct' ? p.typeName : p.tag;
        return this.heap.alloc({ tag: 'String', value: label });
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
            throw new Error(`Unwrap called on Error(${this.heap.formatValue(p.value)})`);
          }
        }
        throw new Error('unwrap() requires a Result<T, E> value');
      }
      // nova.math
      case 'sqrt': return { kind: 'Float', value: Math.sqrt(this.asNumber(args[0])) };
      case 'abs': return { kind: 'Float', value: Math.abs(this.asNumber(args[0])) };
      case 'pow': return { kind: 'Float', value: Math.pow(this.asNumber(args[0]), this.asNumber(args[1])) };
      case 'sin': return { kind: 'Float', value: Math.sin(this.asNumber(args[0])) };
      case 'cos': return { kind: 'Float', value: Math.cos(this.asNumber(args[0])) };
      case 'exp': return { kind: 'Float', value: Math.exp(this.asNumber(args[0])) };
      case 'log': return { kind: 'Float', value: Math.log(this.asNumber(args[0])) };
      case 'floor': return { kind: 'Int', value: Math.floor(this.asNumber(args[0])) };
      case 'ceil': return { kind: 'Int', value: Math.ceil(this.asNumber(args[0])) };
      case 'min': return { kind: 'Float', value: Math.min(this.asNumber(args[0]), this.asNumber(args[1])) };
      case 'max': return { kind: 'Float', value: Math.max(this.asNumber(args[0]), this.asNumber(args[1])) };

      // nova.string
      case 'upper': return this.heap.alloc({ tag: 'String', value: this.asString(args[0]).toUpperCase() });
      case 'lower': return this.heap.alloc({ tag: 'String', value: this.asString(args[0]).toLowerCase() });
      case 'trim': return this.heap.alloc({ tag: 'String', value: this.asString(args[0]).trim() });
      case 'contains': return { kind: 'Bool', value: this.asString(args[0]).includes(this.asString(args[1])) };
      case 'split': return this.fromJS(this.asString(args[0]).split(this.asString(args[1])));

      // nova.array
      case 'range': {
        const s = Math.trunc(this.asNumber(args[0]));
        const e = Math.trunc(this.asNumber(args[1]));
        const els: NovaValue[] = [];
        for (let i = s; i < e; i++) els.push({ kind: 'Int', value: i });
        return this.heap.alloc({ tag: 'Array', elements: els });
      }
      case 'sum': {
        const arr = this.toJS(args[0]);
        if (!Array.isArray(arr)) throw new Error('sum() requires an Array');
        const total = arr.reduce((acc, v) => acc + Number(v), 0);
        return { kind: 'Float', value: total };
      }

      // nova.json
      case 'stringify':
        return this.heap.alloc({ tag: 'String', value: JSON.stringify(this.toJS(args[0])) });
      case 'parse':
        return this.fromJS(JSON.parse(this.asString(args[0])));

      // nova.time
      case 'now_ms':
        return { kind: 'Int', value: Date.now() };

      // nova.ai & nova.nn
      case 'tensor': {
        const js = this.toJS(args[0]);
        const t = NovaTensor.fromNested(js);
        this.maybeGC();
        return this.heap.alloc({ tag: 'Tensor', tensor: t });
      }
      case 'zeros': {
        const shape = this.toJS(args[0]) as number[];
        return this.heap.alloc({ tag: 'Tensor', tensor: NovaTensor.zeros(shape) });
      }
      case 'ones': {
        const shape = this.toJS(args[0]) as number[];
        return this.heap.alloc({ tag: 'Tensor', tensor: NovaTensor.ones(shape) });
      }
      case 'full': {
        const shape = this.toJS(args[0]) as number[];
        const fillVal = this.asNumber(args[1]);
        return this.heap.alloc({ tag: 'Tensor', tensor: NovaTensor.full(shape, fillVal) });
      }
      case 'randn': {
        const shape = this.toJS(args[0]) as number[];
        const seed = args[1] ? Math.trunc(this.asNumber(args[1])) : 42;
        return this.heap.alloc({ tag: 'Tensor', tensor: NovaTensor.randUniform(shape, seed, 1.0) });
      }
      case 'matmul': {
        const a = this.asTensor(args[0]);
        const b = this.asTensor(args[1]);
        if (!a || !b) throw new Error('matmul() requires two Tensors');
        return this.heap.alloc({ tag: 'Tensor', tensor: a.matmul(b) });
      }
      case 'dot': {
        const a = this.asTensor(args[0]);
        const b = this.asTensor(args[1]);
        if (!a || !b) throw new Error('dot() requires two Tensors');
        return { kind: 'Float', value: a.dot(b) };
      }
      case 'transpose': {
        const a = this.asTensor(args[0]);
        if (!a) throw new Error('transpose() requires a Tensor');
        return this.heap.alloc({ tag: 'Tensor', tensor: a.transpose() });
      }
      case 'reshape': {
        const a = this.asTensor(args[0]);
        if (!a) throw new Error('reshape() requires a Tensor');
        const shape = this.toJS(args[1]) as number[];
        return this.heap.alloc({ tag: 'Tensor', tensor: a.reshape(shape) });
      }
      case 'relu': {
        const a = this.asTensor(args[0]);
        if (!a) throw new Error('relu() requires a Tensor');
        return this.heap.alloc({ tag: 'Tensor', tensor: a.relu() });
      }
      case 'sigmoid': {
        const a = this.asTensor(args[0]);
        if (!a) throw new Error('sigmoid() requires a Tensor');
        return this.heap.alloc({ tag: 'Tensor', tensor: a.sigmoid() });
      }
      case 'tanh': {
        const a = this.asTensor(args[0]);
        if (!a) throw new Error('tanh() requires a Tensor');
        return this.heap.alloc({ tag: 'Tensor', tensor: a.tanh() });
      }
      case 'softmax': {
        const a = this.asTensor(args[0]);
        if (!a) throw new Error('softmax() requires a Tensor');
        return this.heap.alloc({ tag: 'Tensor', tensor: a.softmax() });
      }
      case 'mse_loss': {
        const a = this.asTensor(args[0]);
        const b = this.asTensor(args[1]);
        if (!a || !b) throw new Error('mse_loss() requires two Tensors');
        return { kind: 'Float', value: mseLoss(a, b) };
      }
      case 'bce_loss': {
        const a = this.asTensor(args[0]);
        const b = this.asTensor(args[1]);
        if (!a || !b) throw new Error('bce_loss() requires two Tensors');
        return { kind: 'Float', value: bceLoss(a, b) };
      }
      case 'grad_var': {
        const t = this.asTensor(args[0]) ?? NovaTensor.fromNested(this.toJS(args[0]));
        return this.heap.alloc({ tag: 'GradVar', gradVar: GradVar.leaf(t) });
      }
      case 'grad_add':
        return this.heap.alloc({ tag: 'GradVar', gradVar: this.asGradVar(args[0]).add(this.asGradVar(args[1])) });
      case 'grad_sub':
        return this.heap.alloc({ tag: 'GradVar', gradVar: this.asGradVar(args[0]).sub(this.asGradVar(args[1])) });
      case 'grad_mul':
        return this.heap.alloc({ tag: 'GradVar', gradVar: this.asGradVar(args[0]).mul(this.asGradVar(args[1])) });
      case 'grad_matmul':
        return this.heap.alloc({ tag: 'GradVar', gradVar: this.asGradVar(args[0]).matmul(this.asGradVar(args[1])) });
      case 'grad_relu':
        return this.heap.alloc({ tag: 'GradVar', gradVar: this.asGradVar(args[0]).relu() });
      case 'grad_sigmoid':
        return this.heap.alloc({ tag: 'GradVar', gradVar: this.asGradVar(args[0]).sigmoid() });
      case 'grad_tanh':
        return this.heap.alloc({ tag: 'GradVar', gradVar: this.asGradVar(args[0]).tanh() });
      case 'grad_mse': {
        const pred = this.asGradVar(args[0]);
        const target = this.asTensor(args[1]) ?? NovaTensor.fromNested(this.toJS(args[1]));
        return this.heap.alloc({ tag: 'GradVar', gradVar: pred.mseLoss(target) });
      }
      case 'backward': {
        this.asGradVar(args[0]).backward();
        return { kind: 'None' };
      }
      case 'grad_of': {
        const gv = this.asGradVar(args[0]);
        return this.heap.alloc({ tag: 'Tensor', tensor: new NovaTensor(gv.grad.shape, gv.grad.data) });
      }
      case 'value_of': {
        const gv = this.asGradVar(args[0]);
        return this.heap.alloc({ tag: 'Tensor', tensor: new NovaTensor(gv.data.shape, gv.data.data) });
      }
      case 'zero_grad': {
        this.asGradVar(args[0]).zeroGrad();
        return { kind: 'None' };
      }
      case 'model_new':
        return this.heap.alloc({ tag: 'Model', model: new SequentialModel() });
      case 'model_add_linear': {
        const m = this.asModel(args[0]);
        const inF = Math.trunc(this.asNumber(args[1]));
        const outF = Math.trunc(this.asNumber(args[2]));
        const seed = args[3] ? Math.trunc(this.asNumber(args[3])) : 42;
        m.addLinear(inF, outF, seed);
        return args[0];
      }
      case 'model_add_activation': {
        const m = this.asModel(args[0]);
        const act = this.asString(args[1]).toLowerCase() as ActivationKind;
        if (act !== 'relu' && act !== 'sigmoid' && act !== 'tanh') {
          throw new Error(`Unsupported activation '${act}'`);
        }
        m.addActivation(act);
        return args[0];
      }
      case 'model_forward': {
        const m = this.asModel(args[0]);
        const x = this.asTensor(args[1]) ?? NovaTensor.fromNested(this.toJS(args[1]));
        return this.heap.alloc({ tag: 'Tensor', tensor: m.predict(x) });
      }
      case 'model_param_count':
        return { kind: 'Int', value: this.asModel(args[0]).paramCount() };
      case 'optimizer_sgd': {
        const lr = this.asNumber(args[0]);
        const momentum = args[1] ? this.asNumber(args[1]) : 0.0;
        return this.heap.alloc({ tag: 'Optimizer', optimizer: new SGDOptimizer(lr, momentum) });
      }
      case 'optimizer_adam': {
        const lr = this.asNumber(args[0]);
        return this.heap.alloc({ tag: 'Optimizer', optimizer: new AdamOptimizer(lr) });
      }
      case 'train_step': {
        const m = this.asModel(args[0]);
        const opt = this.asOptimizer(args[1]);
        const x = this.asTensor(args[2]) ?? NovaTensor.fromNested(this.toJS(args[2]));
        const y = this.asTensor(args[3]) ?? NovaTensor.fromNested(this.toJS(args[3]));
        const lossName = args[4] ? this.asString(args[4]).toLowerCase() : 'mse';
        const lossVal = trainStep(m, opt, x, y, lossName === 'bce' ? 'bce' : 'mse');
        return { kind: 'Float', value: lossVal };
      }
      case 'train_model': {
        const m = this.asModel(args[0]);
        const opt = this.asOptimizer(args[1]);
        const x = this.asTensor(args[2]) ?? NovaTensor.fromNested(this.toJS(args[2]));
        const y = this.asTensor(args[3]) ?? NovaTensor.fromNested(this.toJS(args[3]));
        const epochs = Math.trunc(this.asNumber(args[4]));
        const lossName = args[5] ? this.asString(args[5]).toLowerCase() : 'mse';
        const report = trainModel(m, opt, x, y, epochs, lossName === 'bce' ? 'bce' : 'mse');
        return this.fromJS(report);
      }
      case 'autograd_demo': {
        const xVal = this.asNumber(args[0]);
        const wVal = this.asNumber(args[1]);
        const x = new AutogradNode(xVal);
        const w = new AutogradNode(wVal);
        const b = new AutogradNode(1.5);
        // y = relu(x * w + b)
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
        const epochs = Math.trunc(this.asNumber(args[0]));
        const lr = this.asNumber(args[1]);
        const summary = trainXorNetwork(epochs, lr);
        return this.fromJS(summary);
      }
      default:
        throw new Error(`Runtime error: unknown builtin function '${rawName}'`);
    }
  }

  private asGradVar(v: NovaValue | undefined): GradVar {
    if (v?.kind === 'HeapRef') {
      const p = this.heap.deref(v).payload;
      if (p.tag === 'GradVar') return p.gradVar;
    }
    throw new Error('Expected GradVar');
  }

  private asModel(v: NovaValue | undefined): SequentialModel {
    if (v?.kind === 'HeapRef') {
      const p = this.heap.deref(v).payload;
      if (p.tag === 'Model') return p.model;
    }
    throw new Error('Expected Sequential model');
  }

  private asOptimizer(v: NovaValue | undefined): NovaOptimizer {
    if (v?.kind === 'HeapRef') {
      const p = this.heap.deref(v).payload;
      if (p.tag === 'Optimizer') return p.optimizer;
    }
    throw new Error('Expected Optimizer');
  }
}
