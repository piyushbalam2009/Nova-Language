import { Diagnostic } from './diagnostics';
import { IRCompareOp, IRConstValue, IRFunction, IRModule } from './ir';
import { SUPPORTED_BYTECODE_VERSION, verifyBytecode } from './verifier';

export enum OpCode {
  PUSH_CONST = 0x01,
  LOAD_LOCAL = 0x02,
  STORE_LOCAL = 0x03,
  LOAD_GLOBAL = 0x04,
  STORE_GLOBAL = 0x05,
  ADD = 0x10,
  SUB = 0x11,
  MUL = 0x12,
  DIV = 0x13,
  MOD = 0x14,
  MATMUL = 0x15,
  NEG = 0x16,
  NOT = 0x17,
  AND = 0x18,
  OR = 0x19,
  CMP_EQ = 0x20,
  CMP_NEQ = 0x21,
  CMP_LT = 0x22,
  CMP_LTE = 0x23,
  CMP_GT = 0x24,
  CMP_GTE = 0x25,
  JUMP = 0x30,
  JUMP_IF_FALSE = 0x31,
  CALL = 0x32,
  CALL_METHOD = 0x33,
  RETURN = 0x34,
  ALLOC_STRUCT = 0x40,
  LOAD_FIELD = 0x41,
  STORE_FIELD = 0x42,
  ALLOC_ARRAY = 0x43,
  ALLOC_MAP = 0x44,
  INDEX_GET = 0x45,
  INDEX_SET = 0x46,
  MAKE_OK = 0x47,
  MAKE_ERROR = 0x48,
  POP = 0x50,
}

export interface BytecodeInst {
  op: OpCode;
  arg?: number | string;
  arg2?: number | string;
  fieldNames?: string[];
}

export interface CompiledFunction {
  name: string;
  arity: number;
  localCount: number;
  localNames: string[];
  maxStackDepth?: number;
  code: BytecodeInst[];
}

export interface BytecodeChunk {
  magic: 'NOVA01';
  version: number;
  constants: IRConstValue[];
  functions: CompiledFunction[];
  functionIndex: Record<string, number>;
  entryPoint: string;
  byteSize: number;
}

const CMP_TO_OP: Record<IRCompareOp, OpCode> = {
  '==': OpCode.CMP_EQ,
  '!=': OpCode.CMP_NEQ,
  '<': OpCode.CMP_LT,
  '<=': OpCode.CMP_LTE,
  '>': OpCode.CMP_GT,
  '>=': OpCode.CMP_GTE,
};

export function compileToBytecode(mod: IRModule): BytecodeChunk {
  const constants: IRConstValue[] = [];
  const constKeyMap = new Map<string, number>();

  const addConst = (val: IRConstValue): number => {
    const key = JSON.stringify(val);
    const existing = constKeyMap.get(key);
    if (existing !== undefined) return existing;
    const idx = constants.length;
    constants.push(val);
    constKeyMap.set(key, idx);
    return idx;
  };

  const compileFunction = (fn: IRFunction): CompiledFunction => {
    const localSlots = new Map<string, number>();
    const localNames: string[] = [];

    const slotFor = (name: string): number => {
      let s = localSlots.get(name);
      if (s === undefined) {
        s = localNames.length;
        localNames.push(name);
        localSlots.set(name, s);
      }
      return s;
    };

    for (const p of fn.params) {
      slotFor(p);
    }

    const code: BytecodeInst[] = [];
    const blockOffsets = new Map<string, number>();
    const pendingJumps: Array<{ instIdx: number; targetLabel: string }> = [];

    for (const block of fn.blocks) {
      blockOffsets.set(block.label, code.length);
      for (const inst of block.instructions) {
        switch (inst.op) {
          case 'CONST': {
            const cIdx = addConst(inst.value);
            code.push({ op: OpCode.PUSH_CONST, arg: cIdx });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'LOAD': {
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.slot) });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'STORE': {
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.src) });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.slot) });
            break;
          }
          case 'LOAD_GLOBAL': {
            code.push({ op: OpCode.LOAD_GLOBAL, arg: inst.name });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'STORE_GLOBAL': {
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.src) });
            code.push({ op: OpCode.STORE_GLOBAL, arg: inst.name });
            break;
          }
          case 'ADD':
          case 'SUB':
          case 'MUL':
          case 'DIV':
          case 'MOD':
          case 'MATMUL':
          case 'AND':
          case 'OR': {
            const opMap: Record<string, OpCode> = {
              ADD: OpCode.ADD,
              SUB: OpCode.SUB,
              MUL: OpCode.MUL,
              DIV: OpCode.DIV,
              MOD: OpCode.MOD,
              MATMUL: OpCode.MATMUL,
              AND: OpCode.AND,
              OR: OpCode.OR,
            };
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.left) });
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.right) });
            code.push({ op: opMap[inst.op] });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'NEG':
          case 'NOT': {
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.src) });
            code.push({ op: inst.op === 'NEG' ? OpCode.NEG : OpCode.NOT });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'COMPARE': {
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.left) });
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.right) });
            code.push({ op: CMP_TO_OP[inst.cmp] });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'CALL': {
            for (const a of inst.args) {
              code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(a) });
            }
            code.push({ op: OpCode.CALL, arg: inst.callee, arg2: inst.args.length });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'CALL_METHOD': {
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.object) });
            for (const a of inst.args) {
              code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(a) });
            }
            code.push({ op: OpCode.CALL_METHOD, arg: inst.method, arg2: inst.args.length });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'ALLOC_STRUCT': {
            for (const f of inst.fields) {
              code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(f.src) });
            }
            code.push({
              op: OpCode.ALLOC_STRUCT,
              arg: inst.typeName,
              arg2: inst.fields.length,
              fieldNames: inst.fields.map((f) => f.name),
            });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'LOAD_FIELD': {
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.object) });
            code.push({ op: OpCode.LOAD_FIELD, arg: inst.field });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'STORE_FIELD': {
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.object) });
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.src) });
            code.push({ op: OpCode.STORE_FIELD, arg: inst.field });
            break;
          }
          case 'ALLOC_ARRAY': {
            for (const el of inst.elements) {
              code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(el) });
            }
            code.push({ op: OpCode.ALLOC_ARRAY, arg: inst.elements.length });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'ALLOC_MAP': {
            for (const e of inst.entries) {
              code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(e.key) });
              code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(e.value) });
            }
            code.push({ op: OpCode.ALLOC_MAP, arg: inst.entries.length });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'INDEX_GET': {
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.object) });
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.index) });
            code.push({ op: OpCode.INDEX_GET });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'INDEX_SET': {
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.object) });
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.index) });
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.src) });
            code.push({ op: OpCode.INDEX_SET });
            break;
          }
          case 'MAKE_RESULT': {
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.src) });
            code.push({ op: inst.variant === 'Ok' ? OpCode.MAKE_OK : OpCode.MAKE_ERROR });
            code.push({ op: OpCode.STORE_LOCAL, arg: slotFor(inst.dst) });
            break;
          }
          case 'JUMP': {
            const idx = code.length;
            code.push({ op: OpCode.JUMP, arg: 0 });
            pendingJumps.push({ instIdx: idx, targetLabel: inst.target });
            break;
          }
          case 'JUMP_IF_FALSE': {
            code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.cond) });
            const idx = code.length;
            code.push({ op: OpCode.JUMP_IF_FALSE, arg: 0 });
            pendingJumps.push({ instIdx: idx, targetLabel: inst.target });
            break;
          }
          case 'RETURN': {
            if (inst.src) {
              code.push({ op: OpCode.LOAD_LOCAL, arg: slotFor(inst.src) });
            } else {
              const noneIdx = addConst({ kind: 'None' });
              code.push({ op: OpCode.PUSH_CONST, arg: noneIdx });
            }
            code.push({ op: OpCode.RETURN });
            break;
          }
        }
      }
    }

    for (const j of pendingJumps) {
      const targetOffset = blockOffsets.get(j.targetLabel);
      // Use -1 sentinel if a block label failed to resolve so the verifier rejects it.
      code[j.instIdx].arg = targetOffset !== undefined ? targetOffset : -1;
    }

    return {
      name: fn.name,
      arity: fn.params.length,
      localCount: localNames.length,
      localNames,
      code,
    };
  };

  const functions = mod.functions.map(compileFunction);
  const functionIndex: Record<string, number> = {};
  functions.forEach((f, idx) => {
    functionIndex[f.name] = idx;
  });

  const entryPoint = functionIndex['main'] !== undefined ? 'main' : (functions[0]?.name ?? '');

  const chunk: BytecodeChunk = {
    magic: 'NOVA01',
    version: SUPPORTED_BYTECODE_VERSION,
    constants,
    functions,
    functionIndex,
    entryPoint,
    byteSize: 0,
  };

  chunk.byteSize = encodeBytecode(chunk).byteLength;
  return chunk;
}

export function validateBytecode(chunk: BytecodeChunk): Diagnostic[] {
  return verifyBytecode(chunk).diagnostics;
}

// ============================================================================
// Binary .nvbc Bytecode Format Encoder / Decoder
// ============================================================================

const TEXT_ENCODER = new TextEncoder();
const TEXT_DECODER = new TextDecoder();

const CONST_TAG = {
  Int: 0x01,
  Float: 0x02,
  Bool: 0x03,
  String: 0x04,
  Char: 0x05,
  None: 0x06,
} as const;

const ARG_TAG = {
  None: 0x00,
  Number: 0x01,
  String: 0x02,
} as const;

class BinaryWriter {
  private buf: number[] = [];

  writeU8(v: number): void {
    this.buf.push(v & 0xff);
  }

  writeU16(v: number): void {
    this.buf.push((v >>> 8) & 0xff, v & 0xff);
  }

  writeU32(v: number): void {
    this.buf.push((v >>> 24) & 0xff, (v >>> 16) & 0xff, (v >>> 8) & 0xff, v & 0xff);
  }

  writeI32(v: number): void {
    this.writeU32(v | 0);
  }

  writeF64(v: number): void {
    const ab = new ArrayBuffer(8);
    new DataView(ab).setFloat64(0, v, false);
    const bytes = new Uint8Array(ab);
    for (let i = 0; i < 8; i++) this.buf.push(bytes[i]);
  }

  writeRawBytes(bytes: Uint8Array): void {
    for (let i = 0; i < bytes.length; i++) this.buf.push(bytes[i]);
  }

  writeString(s: string): void {
    const encoded = TEXT_ENCODER.encode(s);
    this.writeU32(encoded.length);
    this.writeRawBytes(encoded);
  }

  writeArg(arg: number | string | undefined): void {
    if (arg === undefined) {
      this.writeU8(ARG_TAG.None);
    } else if (typeof arg === 'number') {
      this.writeU8(ARG_TAG.Number);
      this.writeI32(arg);
    } else {
      this.writeU8(ARG_TAG.String);
      this.writeString(arg);
    }
  }

  toUint8Array(): Uint8Array {
    return new Uint8Array(this.buf);
  }
}

class BinaryReader {
  private pos = 0;
  private view: DataView;

  constructor(private readonly bytes: Uint8Array) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  private ensure(n: number): void {
    if (this.pos + n > this.bytes.byteLength) {
      throw new Error(`Unexpected EOF decoding .nvbc bytecode at offset ${this.pos} (need ${n} bytes)`);
    }
  }

  readU8(): number {
    this.ensure(1);
    return this.bytes[this.pos++];
  }

  readU16(): number {
    this.ensure(2);
    const v = this.view.getUint16(this.pos, false);
    this.pos += 2;
    return v;
  }

  readU32(): number {
    this.ensure(4);
    const v = this.view.getUint32(this.pos, false);
    this.pos += 4;
    return v;
  }

  readI32(): number {
    this.ensure(4);
    const v = this.view.getInt32(this.pos, false);
    this.pos += 4;
    return v;
  }

  readF64(): number {
    this.ensure(8);
    const v = this.view.getFloat64(this.pos, false);
    this.pos += 8;
    return v;
  }

  readRawBytes(n: number): Uint8Array {
    this.ensure(n);
    const slice = this.bytes.subarray(this.pos, this.pos + n);
    this.pos += n;
    return slice;
  }

  readString(): string {
    const len = this.readU32();
    return TEXT_DECODER.decode(this.readRawBytes(len));
  }

  readArg(): number | string | undefined {
    const tag = this.readU8();
    if (tag === ARG_TAG.None) return undefined;
    if (tag === ARG_TAG.Number) return this.readI32();
    if (tag === ARG_TAG.String) return this.readString();
    throw new Error(`Invalid bytecode operand tag 0x${tag.toString(16)} at offset ${this.pos - 1}`);
  }
}

/**
 * Serializes a compiled Nova `BytecodeChunk` into the compact binary `.nvbc` wire format.
 */
export function encodeBytecode(chunk: BytecodeChunk): Uint8Array {
  const w = new BinaryWriter();

  // 6-byte magic header ("NOVA01") + 2-byte version
  const magicBytes = TEXT_ENCODER.encode(chunk.magic.padEnd(6, '\0').slice(0, 6));
  w.writeRawBytes(magicBytes);
  w.writeU16(chunk.version ?? SUPPORTED_BYTECODE_VERSION);

  // Entry point name
  w.writeString(chunk.entryPoint);

  // Constant pool
  w.writeU32(chunk.constants.length);
  for (const c of chunk.constants) {
    switch (c.kind) {
      case 'Int':
        w.writeU8(CONST_TAG.Int);
        w.writeF64(c.value);
        break;
      case 'Float':
        w.writeU8(CONST_TAG.Float);
        w.writeF64(c.value);
        break;
      case 'Bool':
        w.writeU8(CONST_TAG.Bool);
        w.writeU8(c.value ? 1 : 0);
        break;
      case 'String':
        w.writeU8(CONST_TAG.String);
        w.writeString(c.value);
        break;
      case 'Char':
        w.writeU8(CONST_TAG.Char);
        w.writeString(c.value);
        break;
      case 'None':
        w.writeU8(CONST_TAG.None);
        break;
    }
  }

  // Function table
  w.writeU32(chunk.functions.length);
  for (const fn of chunk.functions) {
    w.writeString(fn.name);
    w.writeU16(fn.arity);
    w.writeU16(fn.localCount);
    w.writeU16(fn.maxStackDepth ?? 0);

    w.writeU16(fn.localNames.length);
    for (const ln of fn.localNames) {
      w.writeString(ln);
    }

    w.writeU32(fn.code.length);
    for (const inst of fn.code) {
      w.writeU8(inst.op);
      w.writeArg(inst.arg);
      w.writeArg(inst.arg2);
      const fieldNames = inst.fieldNames ?? [];
      w.writeU16(fieldNames.length);
      for (const field of fieldNames) {
        w.writeString(field);
      }
    }
  }

  return w.toUint8Array();
}

/**
 * Deserializes a binary `.nvbc` buffer back into a verifiable `BytecodeChunk`.
 */
export function decodeBytecode(bytes: Uint8Array): BytecodeChunk {
  const r = new BinaryReader(bytes);
  const magic = TEXT_DECODER.decode(r.readRawBytes(6)) as 'NOVA01';
  const version = r.readU16();
  const entryPoint = r.readString();

  const constCount = r.readU32();
  const constants: IRConstValue[] = [];
  for (let i = 0; i < constCount; i++) {
    const tag = r.readU8();
    switch (tag) {
      case CONST_TAG.Int:
        constants.push({ kind: 'Int', value: r.readF64() });
        break;
      case CONST_TAG.Float:
        constants.push({ kind: 'Float', value: r.readF64() });
        break;
      case CONST_TAG.Bool:
        constants.push({ kind: 'Bool', value: r.readU8() !== 0 });
        break;
      case CONST_TAG.String:
        constants.push({ kind: 'String', value: r.readString() });
        break;
      case CONST_TAG.Char:
        constants.push({ kind: 'Char', value: r.readString() });
        break;
      case CONST_TAG.None:
        constants.push({ kind: 'None' });
        break;
      default:
        throw new Error(`Unknown constant tag 0x${tag.toString(16)} in .nvbc constant pool`);
    }
  }

  const fnCount = r.readU32();
  const functions: CompiledFunction[] = [];
  const functionIndex: Record<string, number> = {};

  for (let i = 0; i < fnCount; i++) {
    const name = r.readString();
    const arity = r.readU16();
    const localCount = r.readU16();
    const maxStackDepth = r.readU16();

    const localNameCount = r.readU16();
    const localNames: string[] = [];
    for (let j = 0; j < localNameCount; j++) {
      localNames.push(r.readString());
    }

    const instCount = r.readU32();
    const code: BytecodeInst[] = [];
    for (let k = 0; k < instCount; k++) {
      const op = r.readU8() as OpCode;
      const arg = r.readArg();
      const arg2 = r.readArg();
      const fieldCount = r.readU16();
      const inst: BytecodeInst = { op };
      if (arg !== undefined) inst.arg = arg;
      if (arg2 !== undefined) inst.arg2 = arg2;
      if (fieldCount > 0) {
        const fieldNames: string[] = [];
        for (let f = 0; f < fieldCount; f++) {
          fieldNames.push(r.readString());
        }
        inst.fieldNames = fieldNames;
      }
      code.push(inst);
    }

    functionIndex[name] = i;
    functions.push({ name, arity, localCount, localNames, maxStackDepth, code });
  }

  return {
    magic,
    version,
    constants,
    functions,
    functionIndex,
    entryPoint,
    byteSize: bytes.byteLength,
  };
}

export function formatBytecodeHex(bytes: Uint8Array, maxBytes = 128): string {
  const limit = Math.min(bytes.length, maxBytes);
  const lines: string[] = [];
  for (let offset = 0; offset < limit; offset += 16) {
    const slice = bytes.subarray(offset, Math.min(offset + 16, limit));
    const hex = Array.from(slice)
      .map((b) => b.toString(16).padStart(2, '0'))
      .join(' ');
    const ascii = Array.from(slice)
      .map((b) => (b >= 32 && b <= 126 ? String.fromCharCode(b) : '.'))
      .join('');
    lines.push(`${offset.toString(16).padStart(4, '0')}  ${hex.padEnd(47)}  |${ascii}|`);
  }
  if (bytes.length > limit) {
    lines.push(`... (${bytes.length - limit} more bytes)`);
  }
  return lines.join('\n');
}

const OPCODE_NAMES: Record<OpCode, string> = {
  [OpCode.PUSH_CONST]: 'PUSH_CONST',
  [OpCode.LOAD_LOCAL]: 'LOAD_LOCAL',
  [OpCode.STORE_LOCAL]: 'STORE_LOCAL',
  [OpCode.LOAD_GLOBAL]: 'LOAD_GLOBAL',
  [OpCode.STORE_GLOBAL]: 'STORE_GLOBAL',
  [OpCode.ADD]: 'ADD',
  [OpCode.SUB]: 'SUB',
  [OpCode.MUL]: 'MUL',
  [OpCode.DIV]: 'DIV',
  [OpCode.MOD]: 'MOD',
  [OpCode.MATMUL]: 'MATMUL',
  [OpCode.NEG]: 'NEG',
  [OpCode.NOT]: 'NOT',
  [OpCode.AND]: 'AND',
  [OpCode.OR]: 'OR',
  [OpCode.CMP_EQ]: 'CMP_EQ',
  [OpCode.CMP_NEQ]: 'CMP_NEQ',
  [OpCode.CMP_LT]: 'CMP_LT',
  [OpCode.CMP_LTE]: 'CMP_LTE',
  [OpCode.CMP_GT]: 'CMP_GT',
  [OpCode.CMP_GTE]: 'CMP_GTE',
  [OpCode.JUMP]: 'JUMP',
  [OpCode.JUMP_IF_FALSE]: 'JUMP_IF_FALSE',
  [OpCode.CALL]: 'CALL',
  [OpCode.CALL_METHOD]: 'CALL_METHOD',
  [OpCode.RETURN]: 'RETURN',
  [OpCode.ALLOC_STRUCT]: 'ALLOC_STRUCT',
  [OpCode.LOAD_FIELD]: 'LOAD_FIELD',
  [OpCode.STORE_FIELD]: 'STORE_FIELD',
  [OpCode.ALLOC_ARRAY]: 'ALLOC_ARRAY',
  [OpCode.ALLOC_MAP]: 'ALLOC_MAP',
  [OpCode.INDEX_GET]: 'INDEX_GET',
  [OpCode.INDEX_SET]: 'INDEX_SET',
  [OpCode.MAKE_OK]: 'MAKE_OK',
  [OpCode.MAKE_ERROR]: 'MAKE_ERROR',
  [OpCode.POP]: 'POP',
};

export function disassembleBytecode(chunk: BytecodeChunk): string {
  const lines: string[] = [];
  lines.push(`; Nova Bytecode Chunk [${chunk.magic}] (${chunk.byteSize} bytes)`);
  lines.push('; Constant Pool:');
  chunk.constants.forEach((c, i) => {
    const repr = c.kind === 'None' ? 'None' : c.kind === 'String' ? JSON.stringify(c.value) : String(c.value);
    lines.push(`  #${String(i).padStart(2, '0')}  ${c.kind.padEnd(6)}  ${repr}`);
  });
  lines.push('');

  for (const fn of chunk.functions) {
    lines.push(`.fn ${fn.name} (arity=${fn.arity}, locals=${fn.localCount})`);
    fn.code.forEach((inst, ip) => {
      const hexOp = `0x${inst.op.toString(16).padStart(2, '0')}`;
      const name = OPCODE_NAMES[inst.op] ?? `OP_${inst.op}`;
      const args: string[] = [];
      if (inst.arg !== undefined) {
        if (inst.op === OpCode.PUSH_CONST) {
          const c = chunk.constants[Number(inst.arg)];
          const repr = c ? (c.kind === 'None' ? 'None' : c.kind === 'String' ? JSON.stringify(c.value) : String(c.value)) : '?';
          args.push(`#${inst.arg} (${repr})`);
        } else if (inst.op === OpCode.LOAD_LOCAL || inst.op === OpCode.STORE_LOCAL) {
          args.push(`slot[${inst.arg}] (${fn.localNames[Number(inst.arg)] ?? '?'})`);
        } else if (inst.op === OpCode.JUMP || inst.op === OpCode.JUMP_IF_FALSE) {
          args.push(`-> ${String(inst.arg).padStart(4, '0')}`);
        } else {
          args.push(String(inst.arg));
        }
      }
      if (inst.arg2 !== undefined) {
        args.push(`argc=${inst.arg2}`);
      }
      lines.push(`  ${String(ip).padStart(4, '0')}  ${hexOp}  ${name.padEnd(15)} ${args.join(' ')}`.trimEnd());
    });
    lines.push('');
  }
  return lines.join('\n').trimEnd();
}
