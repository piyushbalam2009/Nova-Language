import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  BytecodeChunk,
  BytecodeInst,
  compileAndRun,
  compileToBytecode,
  decodeBytecode,
  DEFAULT_OPT_OPTIONS,
  disassembleBytecode,
  encodeBytecode,
  generateIR,
  lex,
  NovaVM,
  OpCode,
  parse,
  SUPPORTED_BYTECODE_VERSION,
  verifyBytecode,
} from './index';

const NO_OPT = {
  constantFolding: false,
  constantPropagation: false,
  deadCodeElimination: false,
};

/**
 * Helper that compiles and runs a Nova program across ALL four execution paths:
 * 1. Standalone AST Interpreter
 * 2. Nova VM (IR optimizations ON)
 * 3. Nova VM (IR optimizations OFF)
 * 4. Serialized binary `.nvbc` -> decoded -> verified -> executed on Nova VM
 * Asserts identical exitCode and stdout across all 4 engines!
 */
function assertFullExecutionEquivalence(source: string, label: string): string[] {
  const interp = compileAndRun(source, { filename: `${label}.nova`, engine: 'interpreter' });
  const vmOpt = compileAndRun(source, {
    filename: `${label}.nova`,
    engine: 'vm',
    optimizations: DEFAULT_OPT_OPTIONS,
  });
  const vmNoOpt = compileAndRun(source, {
    filename: `${label}.nova`,
    engine: 'vm',
    optimizations: NO_OPT,
  });

  assert.equal(interp.ok, true, `[${label}] Interpreter failed: ${interp.formattedDiagnostics.join('\n')}`);
  assert.equal(vmOpt.ok, true, `[${label}] VM (opt) failed: ${vmOpt.formattedDiagnostics.join('\n')}`);
  assert.equal(vmNoOpt.ok, true, `[${label}] VM (no-opt) failed: ${vmNoOpt.formattedDiagnostics.join('\n')}`);
  assert.equal(vmOpt.verification?.ok, true, `[${label}] Bytecode verification failed`);
  assert.equal(vmNoOpt.verification?.ok, true, `[${label}] Unoptimized bytecode verification failed`);
  assert.ok(vmOpt.bytecode, `[${label}] Missing bytecode chunk`);

  // Round-trip through binary .nvbc wire format
  const encoded = encodeBytecode(vmOpt.bytecode);
  assert.equal(encoded.byteLength, vmOpt.bytecode.byteSize);
  const vmBin = new NovaVM().runBinary(encoded, { verify: true });
  assert.equal(vmBin.exitCode, 0, `[${label}] .nvbc binary execution failed: ${vmBin.runtimeError?.message}`);

  const expected = interp.execution!.stdout;
  assert.deepEqual(vmOpt.execution!.stdout, expected, `[${label}] VM (opt) stdout diverged from Interpreter`);
  assert.deepEqual(vmNoOpt.execution!.stdout, expected, `[${label}] VM (no-opt) stdout diverged from Interpreter`);
  assert.deepEqual(vmBin.stdout, expected, `[${label}] .nvbc binary VM stdout diverged from Interpreter`);

  return expected;
}

// ============================================================================
// 1. Binary .nvbc Serialization & Round-Trip Fidelity
// ============================================================================

test('Bytecode Compiler: .nvbc binary encode/decode round-trip preserves exact disassembly and execution', () => {
  const src = `
struct Vec2 {
    x: Float
    y: Float
}

fn dot(a: Vec2, b: Vec2) -> Float {
    return a.x * b.x + a.y * b.y
}

fn main() {
    let v1 = Vec2 { x: 1.5, y: 2.0 }
    let v2 = Vec2 { x: 4.0, y: -1.0 }
    print("dot=" + str(dot(v1, v2)))
}
`;
  const res = compileAndRun(src, { execute: false });
  assert.equal(res.ok, true);
  assert.ok(res.bytecode);

  const bytes = encodeBytecode(res.bytecode);
  // First 6 bytes must be ASCII "NOVA01"
  assert.equal(new TextDecoder().decode(bytes.subarray(0, 6)), 'NOVA01');

  const decoded = decodeBytecode(bytes);
  const ver = verifyBytecode(decoded);
  assert.equal(ver.ok, true);
  assert.equal(disassembleBytecode(decoded), disassembleBytecode(res.bytecode));

  const execRes = new NovaVM().runBinary(bytes);
  assert.equal(execRes.exitCode, 0);
  assert.deepEqual(execRes.stdout, ['dot=4.0']);
});

// ============================================================================
// 2. Static Bytecode Verifier: Exhaustive Rejection of Corrupt/Unsafe Bytecode
// ============================================================================

function makeValidMinimalChunk(): BytecodeChunk {
  const lexRes = lex('fn main() {\n    let x = 1 + 2\n    print(x)\n}');
  const parseRes = parse(lexRes.tokens);
  const ir = generateIR(parseRes.program);
  return compileToBytecode(ir);
}

test('Bytecode Verifier: rejects invalid magic (E0401) and unsupported version', () => {
  const chunk = makeValidMinimalChunk();
  (chunk as unknown as { magic: string }).magic = 'BAD000';
  const v1 = verifyBytecode(chunk);
  assert.equal(v1.ok, false);
  assert.ok(v1.diagnostics.some((d) => d.code === 'E0401'));

  // VM must refuse to execute unverified chunk
  const vmRes = new NovaVM().run(chunk);
  assert.equal(vmRes.exitCode, 1);
  assert.equal(vmRes.runtimeError?.code, 'E0401');

  const chunk2 = makeValidMinimalChunk();
  chunk2.version = 99;
  const v2 = verifyBytecode(chunk2);
  assert.equal(v2.ok, false);
  assert.ok(v2.diagnostics.some((d) => d.code === 'E0401'));
});

test('Bytecode Verifier: rejects constant pool OOB (E0402), local slot OOB (E0403), jump OOB (E0404)', () => {
  // E0402: constant pool out of bounds
  const c1 = makeValidMinimalChunk();
  c1.functions[0].code[0] = { op: OpCode.PUSH_CONST, arg: 999 };
  assert.ok(verifyBytecode(c1).diagnostics.some((d) => d.code === 'E0402'));

  // E0403: local slot out of bounds
  const c2 = makeValidMinimalChunk();
  c2.functions[0].code.unshift({ op: OpCode.LOAD_LOCAL, arg: 42 }, { op: OpCode.POP });
  assert.ok(verifyBytecode(c2).diagnostics.some((d) => d.code === 'E0403'));

  // E0404: jump target out of bounds (including -1 sentinel and >= length)
  const c3 = makeValidMinimalChunk();
  c3.functions[0].code.unshift({ op: OpCode.JUMP, arg: c3.functions[0].code.length + 5 });
  assert.ok(verifyBytecode(c3).diagnostics.some((d) => d.code === 'E0404'));

  const c4 = makeValidMinimalChunk();
  c4.functions[0].code.unshift({ op: OpCode.JUMP, arg: -1 });
  assert.ok(verifyBytecode(c4).diagnostics.some((d) => d.code === 'E0404'));
});

test('Bytecode Verifier: rejects duplicate functions / bad entry (E0405), arity > locals (E0406), bad opcode (E0407)', () => {
  const c1 = makeValidMinimalChunk();
  c1.entryPoint = 'does_not_exist';
  assert.ok(verifyBytecode(c1).diagnostics.some((d) => d.code === 'E0405'));

  const c2 = makeValidMinimalChunk();
  c2.functions[0].arity = 5;
  c2.functions[0].localCount = 2;
  assert.ok(verifyBytecode(c2).diagnostics.some((d) => d.code === 'E0406'));

  const c3 = makeValidMinimalChunk();
  c3.functions[0].code[0] = { op: 0xfe as OpCode };
  assert.ok(verifyBytecode(c3).diagnostics.some((d) => d.code === 'E0407'));
});

test('Bytecode Verifier: rejects user CALL arity mismatch (E0408) and malformed ALLOC_STRUCT (E0409)', () => {
  const badCall: BytecodeChunk = {
    magic: 'NOVA01',
    version: SUPPORTED_BYTECODE_VERSION,
    constants: [{ kind: 'Int', value: 1 }, { kind: 'None' }],
    functions: [
      {
        name: 'callee',
        arity: 2,
        localCount: 2,
        localNames: ['a', 'b'],
        code: [{ op: OpCode.PUSH_CONST, arg: 1 }, { op: OpCode.RETURN }],
      },
      {
        name: 'main',
        arity: 0,
        localCount: 0,
        localNames: [],
        code: [
          { op: OpCode.PUSH_CONST, arg: 0 },
          { op: OpCode.CALL, arg: 'callee', arg2: 1 }, // expects 2 args!
          { op: OpCode.RETURN },
        ],
      },
    ],
    functionIndex: { callee: 0, main: 1 },
    entryPoint: 'main',
    byteSize: 0,
  };
  assert.ok(verifyBytecode(badCall).diagnostics.some((d) => d.code === 'E0408'));

  const badStruct: BytecodeChunk = {
    magic: 'NOVA01',
    version: SUPPORTED_BYTECODE_VERSION,
    constants: [{ kind: 'Int', value: 1 }],
    functions: [
      {
        name: 'main',
        arity: 0,
        localCount: 0,
        localNames: [],
        code: [
          { op: OpCode.PUSH_CONST, arg: 0 },
          { op: OpCode.ALLOC_STRUCT, arg: 'Point', arg2: 1, fieldNames: ['x', 'y'] },
          { op: OpCode.RETURN },
        ],
      },
    ],
    functionIndex: { main: 0 },
    entryPoint: 'main',
    byteSize: 0,
  };
  assert.ok(verifyBytecode(badStruct).diagnostics.some((d) => d.code === 'E0409'));
});

test('Bytecode Verifier: catches stack underflow (E0410), merge mismatch (E0412), and fallthrough without RETURN (E0413)', () => {
  // E0410: Stack underflow (ADD with empty stack)
  const underflow: BytecodeChunk = {
    magic: 'NOVA01',
    version: SUPPORTED_BYTECODE_VERSION,
    constants: [],
    functions: [
      {
        name: 'main',
        arity: 0,
        localCount: 0,
        localNames: [],
        code: [{ op: OpCode.ADD }, { op: OpCode.RETURN }],
      },
    ],
    functionIndex: { main: 0 },
    entryPoint: 'main',
    byteSize: 0,
  };
  const vUnder = verifyBytecode(underflow);
  assert.equal(vUnder.ok, false);
  assert.ok(vUnder.diagnostics.some((d) => d.code === 'E0410'));

  // E0412: Control-flow merge stack-height mismatch
  const mergeMismatch: BytecodeChunk = {
    magic: 'NOVA01',
    version: SUPPORTED_BYTECODE_VERSION,
    constants: [{ kind: 'Bool', value: true }, { kind: 'Int', value: 7 }],
    functions: [
      {
        name: 'main',
        arity: 0,
        localCount: 0,
        localNames: [],
        code: [
          { op: OpCode.PUSH_CONST, arg: 0 },
          { op: OpCode.JUMP_IF_FALSE, arg: 3 }, // depth 0 -> ip=3
          { op: OpCode.PUSH_CONST, arg: 1 }, // depth 1 -> ip=3
          { op: OpCode.PUSH_CONST, arg: 1 }, // merge point ip=3 sees depth 0 vs 1!
          { op: OpCode.RETURN },
        ],
      },
    ],
    functionIndex: { main: 0 },
    entryPoint: 'main',
    byteSize: 0,
  };
  const vMerge = verifyBytecode(mergeMismatch);
  assert.equal(vMerge.ok, false);
  assert.ok(vMerge.diagnostics.some((d) => d.code === 'E0412'));

  // E0413: Fallthrough without RETURN
  const noReturn: BytecodeChunk = {
    magic: 'NOVA01',
    version: SUPPORTED_BYTECODE_VERSION,
    constants: [{ kind: 'Int', value: 1 }],
    functions: [
      {
        name: 'main',
        arity: 0,
        localCount: 0,
        localNames: [],
        code: [{ op: OpCode.PUSH_CONST, arg: 0 }, { op: OpCode.POP }],
      },
    ],
    functionIndex: { main: 0 },
    entryPoint: 'main',
    byteSize: 0,
  };
  const vNoRet = verifyBytecode(noReturn);
  assert.equal(vNoRet.ok, false);
  assert.ok(vNoRet.diagnostics.some((d) => d.code === 'E0413'));

  // E0411: Operand stack overflow (> 1024)
  const overflowCode: BytecodeInst[] = Array.from({ length: 1030 }, () => ({ op: OpCode.PUSH_CONST, arg: 0 }));
  overflowCode.push({ op: OpCode.RETURN });
  const stackOverflow: BytecodeChunk = {
    magic: 'NOVA01',
    version: SUPPORTED_BYTECODE_VERSION,
    constants: [{ kind: 'Int', value: 1 }],
    functions: [
      {
        name: 'main',
        arity: 0,
        localCount: 0,
        localNames: [],
        code: overflowCode,
      },
    ],
    functionIndex: { main: 0 },
    entryPoint: 'main',
    byteSize: 0,
  };
  const vOver = verifyBytecode(stackOverflow);
  assert.equal(vOver.ok, false);
  assert.ok(vOver.diagnostics.some((d) => d.code === 'E0411'));
});

// ============================================================================
// 3. Execution-Equivalence Suite (Interpreter == VM Opt == VM No-Opt == .nvbc)
// ============================================================================

test('Execution Equivalence: all nova/examples/*.nova match across Interpreter, VM (opt/no-opt), and .nvbc binary', () => {
  const examplesDir = path.resolve(process.cwd(), 'nova/examples');
  const files = fs.readdirSync(examplesDir).filter((f) => f.endsWith('.nova')).sort();
  assert.ok(files.length >= 5);

  for (const file of files) {
    const src = fs.readFileSync(path.join(examplesDir, file), 'utf8');
    const out = assertFullExecutionEquivalence(src, file);
    assert.ok(out.length > 0, `${file} should produce output`);
  }
});

test('Execution Equivalence: mutual recursion, nested loops, break/continue, array iteration, and lexical shadowing', () => {
  const src = `
const BASE = 100

fn is_even(n: Int) -> Bool {
    if n == 0 {
        return true
    }
    return is_odd(n - 1)
}

fn is_odd(n: Int) -> Bool {
    if n == 0 {
        return false
    }
    return is_even(n - 1)
}

fn main() {
    let x = 10
    if x == 10 {
        let x = 25
        if true {
            let x = 50
            print("deep=" + str(x))
        }
        print("mid=" + str(x))
    }
    print("outer=" + str(x))

    mut total = 0
    for i in 0..10 {
        if i % 2 == 0 {
            continue
        }
        if i > 7 {
            break
        }
        total += i
    }
    print("odd_sum=" + str(total))

    let items = [3, 6, 9, 12]
    mut arr_acc = BASE
    for item in items {
        arr_acc += item
    }
    print("arr_acc=" + str(arr_acc))
    print("even_8=" + str(is_even(8)) + " odd_7=" + str(is_odd(7)) + " even_5=" + str(is_even(5)))
}
`;
  const out = assertFullExecutionEquivalence(src, 'control_flow_recursion');
  assert.deepEqual(out, [
    'deep=50',
    'mid=25',
    'outer=10',
    'odd_sum=16',
    'arr_acc=130',
    'even_8=true odd_7=true even_5=false',
  ]);
});

test('Execution Equivalence: short-circuit && and || preserve exact side-effect count', () => {
  const src = `
mut evals = 0

fn probe(tag: String, val: Bool) -> Bool {
    evals += 1
    print("probe:" + tag)
    return val
}

fn main() {
    let a = probe("L1", false) && probe("R1_SKIP", true)
    let b = probe("L2", true) || probe("R2_SKIP", false)
    let c = probe("L3", true) && probe("R3_RUN", true)
    let d = probe("L4", false) || probe("R4_RUN", false)
    print("res=" + str(a) + "," + str(b) + "," + str(c) + "," + str(d))
    print("evals=" + str(evals))
}
`;
  const out = assertFullExecutionEquivalence(src, 'short_circuit');
  assert.deepEqual(out, [
    'probe:L1',
    'probe:L2',
    'probe:L3',
    'probe:R3_RUN',
    'probe:L4',
    'probe:R4_RUN',
    'res=false,true,true,false',
    'evals=6',
  ]);
});

test('Execution Equivalence: compound assignment on structs, arrays, maps, plus Result<T,E> and stdlib', () => {
  const src = `
use nova.math
use nova.string
use nova.json

struct Counter {
    label: String
    value: Int
}

fn safe_div(a: Float, b: Float) -> Result<Float, String> {
    if b == 0.0 {
        return Error("zero")
    }
    return Ok(a / b)
}

fn main() {
    mut c = Counter { label: "hits", value: 10 }
    c.value += 15
    c.value *= 2
    c.value -= 5
    print(c.label + "=" + str(c.value))

    mut nums = [2, 4, 6]
    nums[0] += 8
    nums[1] *= 3
    nums[2] /= 2
    print("nums=" + str(nums[0]) + "," + str(nums[1]) + "," + str(nums[2]))

    mut scores = {"alice": 10, "bob": 20}
    scores["alice"] += 5
    print("alice=" + str(scores["alice"]))

    let r1 = safe_div(9.0, 2.0)
    let r2 = safe_div(9.0, 0.0)
    print("ok=" + str(is_ok(r1)) + " val=" + str(unwrap(r1)) + " err=" + str(is_error(r2)))
    print("math=" + str(floor(sqrt(81.0))) + " str=" + upper(trim("  nova  ")))
}
`;
  const out = assertFullExecutionEquivalence(src, 'compound_structs_stdlib');
  assert.deepEqual(out, [
    'hits=45',
    'nums=10,12,3',
    'alice=15',
    'ok=true val=4.5 err=true',
    'math=9 str=NOVA',
  ]);
});

test('Execution Equivalence: runtime errors (div-by-zero, OOB, unwrap Error, recursion overflow) fail identically in Interpreter and VM', () => {
  const errorPrograms = [
    {
      name: 'div_zero',
      src: 'fn main() {\n    mut z = 0\n    print(10 / z)\n}',
      pattern: /division by zero/i,
    },
    {
      name: 'mod_zero',
      src: 'fn main() {\n    mut z = 0\n    print(10 % z)\n}',
      pattern: /modulo by zero/i,
    },
    {
      name: 'array_oob',
      src: 'fn main() {\n    let a = [1, 2]\n    print(a[5])\n}',
      pattern: /out of bounds/i,
    },
    {
      name: 'unwrap_err',
      src: 'fn main() {\n    let e = Error("boom")\n    print(unwrap(e))\n}',
      pattern: /Unwrap called on Error/i,
    },
    {
      name: 'stack_overflow',
      src: 'fn inf(n: Int) -> Int {\n    return inf(n + 1)\n}\nfn main() {\n    print(inf(0))\n}',
      pattern: /Stack overflow/i,
    },
  ];

  for (const tc of errorPrograms) {
    const iRes = compileAndRun(tc.src, { engine: 'interpreter' });
    const vRes = compileAndRun(tc.src, { engine: 'vm' });
    assert.equal(iRes.execution?.exitCode, 1, `[${tc.name}] Interpreter should fail`);
    assert.equal(vRes.execution?.exitCode, 1, `[${tc.name}] VM should fail`);
    assert.match(iRes.execution?.runtimeError?.message ?? '', tc.pattern);
    assert.match(vRes.execution?.runtimeError?.message ?? '', tc.pattern);
  }
});

// ============================================================================
// 4. CLI Integration: nova verify, nova compile (.nvbc), and nova run (.nvbc)
// ============================================================================

test('CLI Integration: nova verify, nova compile to .nvbc, and nova run on .nvbc binary', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-bc-'));
  const nvbcPath = path.join(tmpDir, 'hello.nvbc');

  try {
    const verifyOut = execFileSync('nova', ['verify', 'hello.nova'], {
      cwd: process.cwd(),
      encoding: 'utf8',
    });
    assert.match(verifyOut, /VERIFIED: hello\.nova/);
    assert.match(verifyOut, /max stack \d+/);

    const compileOut = execFileSync('nova', ['compile', 'hello.nova', '-o', nvbcPath], {
      cwd: process.cwd(),
      encoding: 'utf8',
    });
    assert.match(compileOut, /Compiled & verified hello\.nova/);
    assert.equal(fs.existsSync(nvbcPath), true);

    const verifyBinOut = execFileSync('nova', ['verify', nvbcPath], {
      cwd: process.cwd(),
      encoding: 'utf8',
    });
    assert.match(verifyBinOut, /VERIFIED:/);

    const runBinOut = execFileSync('nova', ['run', nvbcPath], {
      cwd: process.cwd(),
      encoding: 'utf8',
    }).trim();
    assert.equal(runBinOut, 'Hello, Nova');

    // Corrupting the .nvbc magic header must cause nova verify & nova run to exit non-zero
    const corruptBytes = fs.readFileSync(nvbcPath);
    corruptBytes[0] = 0x58; // 'X' instead of 'N'
    const corruptPath = path.join(tmpDir, 'corrupt.nvbc');
    fs.writeFileSync(corruptPath, corruptBytes);

    let failedVerify = false;
    try {
      execFileSync('nova', ['verify', corruptPath], { cwd: process.cwd(), stdio: 'pipe' });
    } catch {
      failedVerify = true;
    }
    assert.equal(failedVerify, true, 'nova verify must reject corrupted .nvbc magic header');

    let failedRun = false;
    try {
      execFileSync('nova', ['run', corruptPath], { cwd: process.cwd(), stdio: 'pipe' });
    } catch {
      failedRun = true;
    }
    assert.equal(failedRun, true, 'nova run must refuse to execute corrupted .nvbc');
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});
