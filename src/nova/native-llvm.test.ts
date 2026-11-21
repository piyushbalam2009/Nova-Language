import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { compileAndRun, lowerToLLVMIR, validateLLVMIR } from './index';
import { buildNativeExecutable, detectNativeToolchain, executeNativeBinary } from './native-builder';

/**
 * Helper that verifies 3-way semantic equivalence across:
 *   1. Standalone AST Interpreter
 *   2. Nova Stack VM (via verified `.nvbc` bytecode)
 *   3. Native ELF Binary (compiled + linked against `libnovart.o`) in both Debug & Release profiles
 */
function assertThreeWayEquivalence(
  label: string,
  source: string,
  expectedStdout?: string[]
): {
  debugBytes: number;
  releaseBytes: number;
  stdout: string[];
} {
  const interp = compileAndRun(source, { engine: 'interpreter', execute: true });
  assert.equal(
    interp.ok,
    true,
    `[${label}] Interpreter failed: ${interp.formattedDiagnostics.join('\n')}`
  );
  assert.ok(interp.execution, `[${label}] Interpreter produced no execution result`);

  const vm = compileAndRun(source, { engine: 'vm', execute: true });
  assert.equal(vm.ok, true, `[${label}] VM failed: ${vm.formattedDiagnostics.join('\n')}`);
  assert.ok(vm.execution, `[${label}] VM produced no execution result`);
  assert.ok(vm.irOptimized, `[${label}] Missing Nova IR`);
  assert.ok(vm.llvmCodegen?.supported, `[${label}] Expected LLVM lowering to succeed`);
  assert.equal(
    vm.llvmCodegen?.validation?.ok,
    true,
    `[${label}] LLVM validation failed: ${vm.llvmCodegen?.validation?.errors.join('; ')}`
  );

  // Verify Interpreter == VM first
  assert.deepEqual(
    vm.execution.stdout,
    interp.execution.stdout,
    `[${label}] VM vs Interpreter stdout mismatch`
  );

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-3way-'));
  const debugBin = path.join(tmpDir, 'prog_debug');
  const releaseBin = path.join(tmpDir, 'prog_release');
  try {
    const dbgBuild = buildNativeExecutable(vm.irOptimized, {
      outputPath: debugBin,
      profile: 'debug',
      backend: 'auto',
      moduleName: label,
    });
    assert.equal(dbgBuild.ok, true, `[${label}] Debug native build failed: ${dbgBuild.reason}`);
    assert.equal(dbgBuild.supported, true);
    assert.ok((dbgBuild.programObjBytes ?? 0) > 0, `[${label}] Expected non-empty program.o`);
    assert.ok((dbgBuild.runtimeObjBytes ?? 0) > 0, `[${label}] Expected non-empty libnovart.o`);

    const relBuild = buildNativeExecutable(vm.irOptimized, {
      outputPath: releaseBin,
      profile: 'release',
      backend: 'auto',
      moduleName: label,
    });
    assert.equal(relBuild.ok, true, `[${label}] Release native build failed: ${relBuild.reason}`);
    assert.equal(relBuild.supported, true);

    const dbgExec = executeNativeBinary(debugBin);
    assert.equal(
      dbgExec.exitCode,
      0,
      `[${label}] Debug binary exited with ${dbgExec.exitCode}: ${dbgExec.stderr}`
    );
    assert.deepEqual(
      dbgExec.stdout,
      interp.execution.stdout,
      `[${label}] Native Debug vs Interpreter stdout mismatch`
    );

    const relExec = executeNativeBinary(releaseBin);
    assert.equal(
      relExec.exitCode,
      0,
      `[${label}] Release binary exited with ${relExec.exitCode}: ${relExec.stderr}`
    );
    assert.deepEqual(
      relExec.stdout,
      interp.execution.stdout,
      `[${label}] Native Release vs Interpreter stdout mismatch`
    );

    if (expectedStdout) {
      assert.deepEqual(relExec.stdout, expectedStdout, `[${label}] Unexpected stdout`);
    }

    return {
      debugBytes: dbgBuild.binaryBytes ?? 0,
      releaseBytes: relBuild.binaryBytes ?? 0,
      stdout: relExec.stdout,
    };
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

test('1. LLVM IR lowering & structural validation across structs, arrays, maps, globals, and loops', () => {
  const src = `
const SCALE = 3
struct Point {
    x: Int
    y: Int
}

fn sum_range(limit: Int) -> Int {
    mut total = 0
    for i in 0..limit {
        total += i
    }
    return total
}

fn main() {
    let p = Point { x: 4, y: 5 }
    let arr = [p.x, p.y, sum_range(5) * SCALE]
    let m = { "k": arr[2] }
    print(p, arr, m["k"])
}`;
  const res = compileAndRun(src, { execute: false });
  assert.equal(res.ok, true, res.formattedDiagnostics.join('\n'));
  assert.ok(res.irOptimized);

  const lowered = lowerToLLVMIR(res.irOptimized, { moduleName: 'struct_loop_test' });
  assert.equal(lowered.supported, true);
  assert.ok(lowered.llvmIR);
  assert.equal(
    lowered.validation?.ok,
    true,
    `LLVM IR validation errors: ${lowered.validation?.errors.join('\n')}`
  );
  assert.ok(lowered.llvmIR.includes('%struct.NVal = type { i32, i64, double, i1, ptr }'));
  assert.ok(lowered.llvmIR.includes('define internal %struct.NVal @nova_fn_sum_range('));
  assert.ok(lowered.llvmIR.includes('define internal %struct.NVal @nova_fn_main('));
  assert.ok(lowered.llvmIR.includes('define i32 @main()'));
  assert.ok(lowered.llvmIR.includes('@nv_alloc_struct'));
  assert.ok(lowered.llvmIR.includes('@nv_alloc_array'));
  assert.ok(lowered.llvmIR.includes('@nv_alloc_map'));
});

test('2. LLVM IR static verifier rejects malformed target triple, duplicate SSA, unterminated blocks, and bad branch targets', () => {
  const badIR = `
define i32 @main() {
entry:
  %r1 = call %struct.NVal @nv_int(i64 1)
  %r1 = call %struct.NVal @nv_int(i64 2)
  br label %does_not_exist
dangling:
  %r2 = call %struct.NVal @nv_none()
}
`;
  const report = validateLLVMIR(badIR);
  assert.equal(report.ok, false);
  assert.ok(
    report.errors.some((e) => e.includes('target triple')),
    `Expected missing target triple error, got: ${report.errors.join('; ')}`
  );
  assert.ok(
    report.errors.some((e) => e.includes('SSA violation')),
    `Expected SSA violation error, got: ${report.errors.join('; ')}`
  );
  assert.ok(
    report.errors.some((e) => e.includes('%does_not_exist')),
    `Expected undefined branch target error, got: ${report.errors.join('; ')}`
  );
  assert.ok(
    report.errors.some((e) => e.includes("block 'dangling'") && e.includes('terminator')),
    `Expected missing terminator error, got: ${report.errors.join('; ')}`
  );
});

test('3. Multi-stage linking produces real ELF binary and Release (-O3 + strip) shrinks binary vs Debug (-O0 -g)', () => {
  const toolchain = detectNativeToolchain();
  assert.equal(toolchain.gcc, true, 'Expected GCC toolchain to be available in container');

  const src = fs.readFileSync(path.resolve('nova/examples/structs_results.nova'), 'utf8');
  const compiled = compileAndRun(src, { execute: false });
  assert.equal(compiled.ok, true);
  assert.ok(compiled.irOptimized);

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-elf-'));
  const dbgBin = path.join(tmpDir, 'app_debug');
  const relBin = path.join(tmpDir, 'app_release');
  const llFile = path.join(tmpDir, 'app.ll');
  try {
    const dbg = buildNativeExecutable(compiled.irOptimized, {
      outputPath: dbgBin,
      profile: 'debug',
    });
    assert.equal(dbg.ok, true, dbg.reason);
    assert.equal(dbg.stripped, false);

    const rel = buildNativeExecutable(compiled.irOptimized, {
      outputPath: relBin,
      emitLlvmPath: llFile,
      profile: 'release',
    });
    assert.equal(rel.ok, true, rel.reason);
    assert.equal(fs.existsSync(llFile), true, 'Expected --emit-llvm file to be written');

    // Verify genuine ELF magic bytes (0x7f 'E' 'L' 'F')
    const header = fs.readFileSync(relBin).subarray(0, 4);
    assert.deepEqual(Array.from(header), [0x7f, 0x45, 0x4c, 0x46]);

    // Verify release stripping strictly reduces binary size relative to debug build
    if (toolchain.strip) {
      assert.equal(rel.stripped, true);
      assert.ok(
        (rel.binaryBytes ?? 0) < (rel.preStripBytes ?? 0),
        `Expected strip to reduce size (${rel.preStripBytes} -> ${rel.binaryBytes})`
      );
    }
    assert.ok(
      (rel.binaryBytes ?? 0) < (dbg.binaryBytes ?? 0),
      `Expected release binary (${rel.binaryBytes} B) to be smaller than debug binary (${dbg.binaryBytes} B)`
    );
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('4. 3-Way Equivalence across core examples: hello.nova, functions.nova, structs_results.nova, and IR optimizer', () => {
  for (const file of ['hello.nova', 'functions.nova', 'structs_results.nova']) {
    const src = fs.readFileSync(path.resolve('nova/examples', file), 'utf8');
    assertThreeWayEquivalence(file, src);
  }
  const optSrc = `
fn main() {
    let folded = 2 + 3 * 4 + (10 - 4)
    let dead = 999 * 100
    print("folded =", folded)
}`;
  assertThreeWayEquivalence('ir_optimizer_sample', optSrc, ['folded = 20']);
});

test('5. 3-Way Equivalence: recursion, mutual recursion, lexical shadowing, short-circuit &&/||, break/continue', () => {
  const src = `
mut side_effects = 0

fn bump(tag: Int) -> Bool {
    side_effects = side_effects + tag
    return true
}

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
    if true {
        let x = 99
        print("inner:", x)
    }
    print("outer:", x)

    let a = false && bump(100)
    let b = true || bump(200)
    let c = true && bump(5)
    let d = false || bump(7)
    print(a, b, c, d, side_effects)

    print(is_even(10), is_odd(10), is_even(7), is_odd(7))

    mut sum = 0
    for i in 0..20 {
        if i % 2 == 0 {
            continue
        }
        if i > 11 {
            break
        }
        sum += i
    }
    print("odd_sum:", sum)
}`;
  assertThreeWayEquivalence('control_flow_and_scoping', src, [
    'inner: 99',
    'outer: 10',
    'false true true true 12',
    'true false false true',
    'odd_sum: 36',
  ]);
});

test('6. 3-Way Equivalence: structs, arrays, maps, methods, Result<T, E>, and stdlib math/string', () => {
  const src = `
use nova.math
use nova.string

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
    c.value += 5
    c.value *= 2
    print(c, c.label, c.value)

    mut nums = [1, 2, 3]
    nums.push(4)
    nums[1] = 20
    mut total = 0
    for n in nums {
        total += n
    }
    print(nums, nums.len(), total)

    mut scores = { "alice": 90, "bob": 80 }
    scores["alice"] = 95
    scores["carol"] = 88
    print(scores["alice"], scores["bob"], scores["carol"])

    let ok_val = safe_div(9.0, 2.0)
    let err_val = safe_div(9.0, 0.0)
    print(ok_val, is_ok(ok_val), unwrap(ok_val))
    print(err_val, is_error(err_val))

    let cleaned = "  Nova Native  ".trim()
    print(cleaned.upper(), cleaned.lower(), contains(cleaned, "Native"))
    print(int(sqrt(81.0)), abs(-42.0), pow(2.0, 5.0), min(9.0, 4.0), max(9.0, 4.0))
    print(type_of(c), type_of(nums), type_of(ok_val))
}`;
  assertThreeWayEquivalence('heap_objects_and_stdlib', src, [
    'Counter { label: hits, value: 30 } hits 30',
    '[1, 20, 3, 4] 4 28',
    '95 80 88',
    'Ok(4.5) true 4.5',
    'Error(zero) true',
    'NOVA NATIVE nova native true',
    '9 42.0 32.0 4.0 9.0',
    'Counter Array Result',
  ]);
});

test('7. 3-Way Runtime Error Parity: division by zero, array OOB, and unwrap(Error) fail identically', () => {
  const errorCases: Array<{ name: string; source: string }> = [
    {
      name: 'div_by_zero',
      source: 'fn main() {\n    let a = 10\n    let b = 0\n    print(a / b)\n}',
    },
    {
      name: 'array_oob',
      source: 'fn main() {\n    let arr = [1, 2]\n    print(arr[5])\n}',
    },
    {
      name: 'unwrap_error',
      source: 'fn main() {\n    let r = Error("boom")\n    print(unwrap(r))\n}',
    },
  ];

  for (const tc of errorCases) {
    const interp = compileAndRun(tc.source, { engine: 'interpreter', execute: true });
    assert.equal(interp.ok, false, `[${tc.name}] Interpreter should fail`);

    const vm = compileAndRun(tc.source, { engine: 'vm', execute: true });
    assert.equal(vm.ok, false, `[${tc.name}] VM should fail`);
    assert.ok(vm.irOptimized);

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-err-'));
    const bin = path.join(tmpDir, 'err_prog');
    try {
      const built = buildNativeExecutable(vm.irOptimized, { outputPath: bin, profile: 'release' });
      assert.equal(built.ok, true, `[${tc.name}] Build failed: ${built.reason}`);
      const run = executeNativeBinary(bin);
      assert.notEqual(run.exitCode, 0, `[${tc.name}] Native binary must exit non-zero`);
      assert.match(run.stderr, /runtime error/i, `[${tc.name}] Expected panic on stderr, got: ${run.stderr}`);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  }
});

test('8. Live CLI: nova llvm, nova build --debug, and nova build --release --emit-llvm execute built binaries', () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-cli-native-'));
  const llOut = path.join(tmpDir, 'hello.ll');
  const dbgBin = path.join(tmpDir, 'hello_dbg');
  const relBin = path.join(tmpDir, 'structs_rel');
  const emittedLl = path.join(tmpDir, 'structs_rel.ll');

  try {
    // 1. nova llvm hello.nova -o hello.ll
    const llvmOut = execFileSync('nova', ['llvm', 'nova/examples/hello.nova', '-o', llOut], {
      encoding: 'utf8',
    });
    assert.ok(llvmOut.includes('Emitted verified LLVM IR'), `Unexpected output: ${llvmOut}`);
    assert.equal(fs.existsSync(llOut), true);
    const llText = fs.readFileSync(llOut, 'utf8');
    assert.ok(llText.includes('define i32 @main()'));

    // 2. nova build hello.nova --debug -o hello_dbg
    const dbgOut = execFileSync(
      'nova',
      ['build', 'nova/examples/hello.nova', '--debug', '-o', dbgBin],
      { encoding: 'utf8' }
    );
    assert.ok(dbgOut.includes('(debug,'), `Unexpected output: ${dbgOut}`);
    const runHello = execFileSync(dbgBin, { encoding: 'utf8' }).trim();
    assert.equal(runHello, 'Hello, Nova');

    // 3. nova build structs_results.nova --release --emit-llvm -o structs_rel
    const relOut = execFileSync(
      'nova',
      ['build', 'nova/examples/structs_results.nova', '--release', '--emit-llvm', '-o', relBin],
      { encoding: 'utf8' }
    );
    assert.ok(relOut.includes('(release,'), `Unexpected output: ${relOut}`);
    assert.equal(fs.existsSync(emittedLl), true);

    const nativeStdout = execFileSync(relBin, { encoding: 'utf8' }).replace(/\n$/, '').split('\n');
    const interpStdout = execFileSync(
      'nova',
      ['run', '--engine=interpreter', 'nova/examples/structs_results.nova'],
      { encoding: 'utf8' }
    )
      .replace(/\n$/, '')
      .split('\n');
    assert.deepEqual(nativeStdout, interpStdout);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});
