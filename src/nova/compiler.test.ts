import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  AutogradNode,
  compileAndRun,
  formatNovaSource,
  lex,
  NovaTensor,
  parse,
  TokenKind,
  trainXorNetwork,
} from './index';
import { buildNativeExecutable, executeNativeBinary } from './native-builder';

test('1. Lexer: tokenizes keywords, operators, spans, and rejects malformed input', () => {
  const { tokens, diagnostics } = lex('let a = 42 + 3.14\n// comment\nlet c = a @ b');
  assert.equal(diagnostics.length, 0);
  const kinds = tokens.map((t) => t.kind);
  assert.ok(kinds.includes(TokenKind.Let));
  assert.ok(kinds.includes(TokenKind.Int));
  assert.ok(kinds.includes(TokenKind.Float));
  assert.ok(kinds.includes(TokenKind.At));
  assert.equal(tokens[0].span.line, 1);
  assert.equal(tokens[0].span.column, 1);

  const bad = lex('let s = "unterminated\nlet x = $');
  assert.equal(bad.diagnostics.length, 2);
  assert.equal(bad.diagnostics[0].code, 'E0003');
  assert.equal(bad.diagnostics[1].code, 'E0007');
});

test('2. Parser: Pratt operator precedence and AST structure', () => {
  const { tokens } = lex('fn main() {\n  let x = 2 + 3 * 4\n}');
  const { program, diagnostics } = parse(tokens);
  assert.equal(diagnostics.length, 0);
  assert.equal(program.items.length, 1);
  const fn = program.items[0];
  assert.equal(fn.kind, 'FnDecl');
  if (fn.kind === 'FnDecl') {
    const decl = fn.body[0];
    assert.equal(decl.kind, 'VarDecl');
    if (decl.kind === 'VarDecl') {
      assert.equal(decl.initializer.kind, 'BinaryExpr');
      if (decl.initializer.kind === 'BinaryExpr') {
        assert.equal(decl.initializer.op, '+');
        assert.equal(decl.initializer.right.kind, 'BinaryExpr');
      }
    }
  }
});

test('3. Semantic Analyzer: catches type mismatches, immutability violations, undefined symbols, and arity errors', () => {
  const typeMismatch = compileAndRun('fn main() {\n  let age: Int = "Hello"\n}', { execute: false });
  assert.equal(typeMismatch.ok, false);
  assert.ok(typeMismatch.diagnostics.some((d) => d.code === 'E0210'));

  const immutableErr = compileAndRun('fn main() {\n  let x = 1\n  x = 2\n}', { execute: false });
  assert.equal(immutableErr.ok, false);
  assert.ok(immutableErr.diagnostics.some((d) => d.code === 'E0213'));

  const undefVar = compileAndRun('fn main() {\n  print(missing_var)\n}', { execute: false });
  assert.equal(undefVar.ok, false);
  assert.ok(undefVar.diagnostics.some((d) => d.code === 'E0212'));

  const wrongArity = compileAndRun('fn add(a: Int, b: Int) -> Int { return a + b }\nfn main() { add(1) }', {
    execute: false,
  });
  assert.equal(wrongArity.ok, false);
  assert.ok(wrongArity.diagnostics.some((d) => d.code === 'E0231'));

  const wrongReturn = compileAndRun('fn bad() -> Int { return "oops" }\nfn main() {}', { execute: false });
  assert.equal(wrongReturn.ok, false);
  assert.ok(wrongReturn.diagnostics.some((d) => d.code === 'E0219'));
});

test('4. First Execution Target: hello.nova prints Hello, Nova on Nova VM', () => {
  const src = `fn main() {\n    let name = "Nova"\n    print("Hello, " + name)\n}`;
  const res = compileAndRun(src);
  assert.equal(res.ok, true);
  assert.deepEqual(res.execution?.stdout, ['Hello, Nova']);
});

test('5. Nova IR Optimizer: folds constants and eliminates dead code', () => {
  const src = `fn main() {\n  let unused = 10 + 20\n  let val = 2 + 3 * 4\n  print(val)\n}`;
  const unopt = compileAndRun(src, {
    optimizations: { constantFolding: false, constantPropagation: false, deadCodeElimination: false },
  });
  const opt = compileAndRun(src, {
    optimizations: { constantFolding: true, constantPropagation: true, deadCodeElimination: true },
  });
  assert.equal(opt.ok, true);
  assert.deepEqual(opt.execution?.stdout, ['14']);
  assert.deepEqual(unopt.execution?.stdout, ['14']);
  assert.ok((opt.optimizationStats?.constantsFolded ?? 0) >= 2);
  assert.ok((opt.optimizationStats?.instructionsAfter ?? 999) < (opt.optimizationStats?.instructionsBefore ?? 0));
});

test('6. Functions, Recursion, Loops, Structs, and Result<T, E> on Nova VM', () => {
  const src = `
struct User {
    name: String
    age: Int
}

fn fib(n: Int) -> Int {
    if n <= 1 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}

fn divide(a: Float, b: Float) -> Result<Float, String> {
    if b == 0.0 {
        return Error("Division by zero")
    }
    return Ok(a / b)
}

fn main() {
    let u = User { name: "Alex", age: 22 }
    print(u.name + ":" + str(u.age))
    mut acc = 0
    for i in 0..7 {
        acc += fib(i)
    }
    print(acc)
    print(divide(10.0, 2.0))
    print(divide(10.0, 0.0))
}`;
  const res = compileAndRun(src);
  assert.equal(res.ok, true, res.formattedDiagnostics.join('\n'));
  assert.deepEqual(res.execution?.stdout, [
    'Alex:22',
    '20',
    'Ok(5.0)',
    'Error(Division by zero)',
  ]);
  assert.ok((res.execution?.gcStats.totalAllocated ?? 0) > 0);
  assert.ok((res.execution?.gcStats.totalFreed ?? 0) > 0);
});

test('7. Nova AI Engine: Tensor matmul @, Autograd backward, and XOR training', () => {
  const a = NovaTensor.fromNested([[1, 2], [3, 4]]);
  const b = NovaTensor.fromNested([[5, 6], [7, 8]]);
  const prod = a.matmul(b);
  assert.deepEqual(prod.shape, [2, 2]);
  assert.deepEqual(Array.from(prod.data), [19, 22, 43, 50]);

  // Reverse-mode autograd check: y = relu(x * w + 1.5), dy/dx = w = 2, dy/dw = x = 3
  const x = new AutogradNode(3);
  const w = new AutogradNode(2);
  const y = x.mul(w).add(new AutogradNode(1.5)).relu();
  y.backward();
  assert.equal(y.data, 7.5);
  assert.equal(x.grad, 2);
  assert.equal(w.grad, 3);

  // Neural network XOR training reduces loss significantly
  const summary = trainXorNetwork(600, 0.8);
  assert.ok(summary.finalLoss < summary.initialLoss);

  // Execute tensor program via Nova VM
  const vmRes = compileAndRun(`
use nova.ai
fn main() {
    let a = tensor([[1.0, 2.0], [3.0, 4.0]])
    let b = tensor([[5.0, 6.0], [7.0, 8.0]])
    let c = a @ b
    print(c)
}`);
  assert.equal(vmRes.ok, true);
  assert.equal(vmRes.execution?.stdout[0], 'Tensor([[19, 22], [43, 50]], shape=[2, 2])');
});

test('8. Formatter: produces canonical Nova source that compiles cleanly', () => {
  const messy = 'fn main(){let x=1+2\nprint(x)}';
  const fmt = formatNovaSource(messy);
  assert.equal(fmt.ok, true);
  assert.ok(fmt.formatted.includes('    let x = 1 + 2'));
  const run = compileAndRun(fmt.formatted);
  assert.equal(run.ok, true);
  assert.deepEqual(run.execution?.stdout, ['3']);
});

test('9. Execution Equivalence: Nova VM and Native GCC x86_64 binary produce identical output', () => {
  const src = `
fn fib(n: Int) -> Int {
    if n <= 1 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}

fn main() {
    let name = "Nova"
    print("Hello, " + name)
    mut sum = 0
    for i in 0..8 {
        sum += fib(i)
    }
    print(sum)
}`;
  const vmRes = compileAndRun(src);
  assert.equal(vmRes.ok, true);
  assert.equal(vmRes.nativeCodegen?.supported, true);
  assert.ok(vmRes.irOptimized);

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-equiv-'));
  const binFile = path.join(tmpDir, 'prog');
  try {
    const build = buildNativeExecutable(vmRes.irOptimized!, {
      outputPath: binFile,
      profile: 'release',
      backend: 'auto',
    });
    assert.equal(build.ok, true, build.reason);
    assert.equal(build.supported, true);
    const execRes = executeNativeBinary(binFile);
    assert.equal(execRes.exitCode, 0);
    assert.deepEqual(execRes.stdout, vmRes.execution?.stdout);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});
