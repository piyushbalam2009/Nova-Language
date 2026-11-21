import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  analyze,
  compileAndRun,
  lex,
  NovaInterpreter,
  parse,
  TokenKind,
} from './index';

// ============================================================================
// 1. LEXER COMPREHENSIVE TESTS
// ============================================================================

test('Nova 0.1 Lexer: hex, binary, underscored ints, scientific floats, chars, escapes, and comments', () => {
  const src = [
    '// line comment',
    '# hash comment',
    '/* multi-line',
    '   block comment */',
    'let h = 0xFF',
    'let b = 0b1010_0101',
    'let big = 1_000_000',
    'let sci = 2.5e-3',
    "let ch = 'z'",
    'let msg = "line1\\n\\t\\"ok\\""',
  ].join('\n');

  const res = lex(src);
  assert.equal(res.diagnostics.length, 0);
  const ints = res.tokens.filter((t) => t.kind === TokenKind.Int).map((t) => t.lexeme);
  assert.deepEqual(ints, ['255', '165', '1000000']);
  const floats = res.tokens.filter((t) => t.kind === TokenKind.Float).map((t) => t.lexeme);
  assert.deepEqual(floats, ['2.5e-3']);
  const chars = res.tokens.filter((t) => t.kind === TokenKind.Char).map((t) => t.lexeme);
  assert.deepEqual(chars, ['z']);
  const strs = res.tokens.filter((t) => t.kind === TokenKind.String).map((t) => t.lexeme);
  assert.deepEqual(strs, ['line1\n\t"ok"']);
});

test('Nova 0.1 Lexer: reports exact locations for malformed literals and unterminated block comments', () => {
  const badHex = lex('let x = 0xZZ');
  assert.equal(badHex.diagnostics.length, 1);
  assert.equal(badHex.diagnostics[0].code, 'E0006');
  assert.equal(badHex.diagnostics[0].span.line, 1);
  assert.equal(badHex.diagnostics[0].span.column, 9);

  const badComment = lex('let a = 1\n/* never closed');
  assert.equal(badComment.diagnostics.length, 1);
  assert.equal(badComment.diagnostics[0].code, 'E0001');
  assert.equal(badComment.diagnostics[0].span.line, 2);

  const badChar = lex("let c = ''");
  assert.equal(badChar.diagnostics.length, 1);
  assert.equal(badChar.diagnostics[0].code, 'E0004');

  const badExp = lex('let f = 1.2e+');
  assert.equal(badExp.diagnostics.length, 1);
  assert.equal(badExp.diagnostics[0].code, 'E0005');
});

// ============================================================================
// 2. PARSER & AST COMPREHENSIVE TESTS
// ============================================================================

test('Nova 0.1 Parser: parses unary ops, grouped precedence, else-if chains, loops, structs, maps, and Result', () => {
  const src = `
use nova.math

struct Point {
    x: Int
    y: Int
}

fn classify(n: Int) -> String {
    if n < 0 {
        return "neg"
    } else if n == 0 {
        return "zero"
    } else {
        return "pos"
    }
}

fn main() {
    let v = -(2 + 3) * 4
    let flag = !false && (1 < 2 || 3 > 4)
    let p = Point { x: 1, y: 2 }
    let m = { "a": 1, "b": 2 }
    let arr = [1, 2, 3]
    let ok_val = Ok(p.x + arr[1] + m["a"])
    mut i = 0
    while i < 5 {
        i += 1
        if i == 2 { continue }
        if i == 4 { break }
    }
    print(v, flag, classify(i), ok_val)
}`;
  const lexRes = lex(src);
  assert.equal(lexRes.diagnostics.length, 0);
  const parseRes = parse(lexRes.tokens);
  assert.equal(parseRes.diagnostics.length, 0);
  assert.equal(parseRes.program.items.length, 4);
});

test('Nova 0.1 Parser: reports clear syntax errors with line and column for invalid programs', () => {
  const missingBrace = parse(lex('fn main() {\n    let x = 1\n').tokens);
  assert.ok(missingBrace.diagnostics.length >= 1);
  assert.equal(missingBrace.diagnostics[0].stage, 'parser');

  const missingEq = parse(lex('fn main() {\n    let x 42\n}').tokens);
  assert.ok(missingEq.diagnostics.length >= 1);
  assert.equal(missingEq.diagnostics[0].span.line, 2);
});

// ============================================================================
// 3. SEMANTIC ANALYZER & TYPE CHECKER COMPREHENSIVE TESTS
// ============================================================================

test('Nova 0.1 Semantic: enforces definite returns, loop control flow, struct fields, and const immutability', () => {
  // Missing return on non-Void function
  const missingRet = compileAndRun(
    'fn compute(x: Int) -> Int {\n  if x > 0 {\n    return 1\n  }\n}\nfn main() {}',
    { execute: false }
  );
  assert.equal(missingRet.ok, false);
  assert.ok(missingRet.diagnostics.some((d) => d.code === 'E0228'));

  // Definite return on all branches passes
  const allBranchesRet = compileAndRun(
    'fn compute(x: Int) -> Int {\n  if x > 0 {\n    return 1\n  } else {\n    return 0\n  }\n}\nfn main() {}',
    { execute: false }
  );
  assert.equal(allBranchesRet.ok, true);

  // break outside loop
  const badBreak = compileAndRun('fn main() {\n  break\n}', { execute: false });
  assert.equal(badBreak.ok, false);
  assert.ok(badBreak.diagnostics.some((d) => d.code === 'E0220'));

  // const reassignment
  const badConst = compileAndRun('const LIMIT = 10\nfn main() {\n  LIMIT = 20\n}', { execute: false });
  assert.equal(badConst.ok, false);
  assert.ok(badConst.diagnostics.some((d) => d.code === 'E0213'));

  // Missing required struct field
  const badStruct = compileAndRun(
    'struct User { name: String, age: Int }\nfn main() {\n  let u = User { name: "Nova" }\n}',
    { execute: false }
  );
  assert.equal(badStruct.ok, false);
  assert.ok(badStruct.diagnostics.some((d) => d.code === 'E0239'));

  // Unknown module
  const badMod = compileAndRun('use nova.nonexistent\nfn main() {}', { execute: false });
  assert.equal(badMod.ok, false);
  assert.ok(badMod.diagnostics.some((d) => d.code === 'E0203'));

  // Duplicate variable in same scope rejected, but inner-block shadowing allowed
  const dupVar = analyze(parse(lex('fn main() {\n  let a = 1\n  let a = 2\n}').tokens).program);
  assert.ok(dupVar.diagnostics.some((d) => d.code === 'E0211'));
});

// ============================================================================
// 4. AST INTERPRETER & VM PARITY (SHADOWING, SHORT-CIRCUIT, LOOPS, STRUCTS)
// ============================================================================

test('Nova 0.1 Interpreter vs VM: identical semantics for lexical shadowing, short-circuit &&/||, break/continue, and structs', () => {
  const src = `
struct Counter {
    label: String
    value: Int
}

fn fact(n: Int) -> Int {
    if n <= 1 {
        return 1
    }
    return n * fact(n - 1)
}

fn main() {
    // 1. Lexical shadowing
    let x = 10
    if true {
        let x = 99
        print("inner:" + str(x))
    }
    print("outer:" + str(x))

    // 2. Short-circuit && and || prevent division by zero
    let denom = 0
    if denom != 0 && (10 / denom > 1) {
        print("should_not_run")
    } else {
        print("short_circuit_and_ok")
    }
    if denom == 0 || (10 / denom > 1) {
        print("short_circuit_or_ok")
    }

    // 3. Loops with continue and break
    mut acc = 0
    for i in 0..10 {
        if i % 2 == 0 { continue }
        if i > 7 { break }
        acc += i
    }
    print("loop_acc:" + str(acc))

    // 4. Struct mutation & recursion
    let c = Counter { label: "fact5", value: fact(5) }
    print(c.label + "=" + str(c.value))

    // 5. Hex & binary arithmetic
    print("bits:" + str(0x10 + 0b1010))
}`;

  const interpRes = compileAndRun(src, { engine: 'interpreter' });
  const vmRes = compileAndRun(src, { engine: 'vm' });

  assert.equal(interpRes.ok, true, interpRes.formattedDiagnostics.join('\n'));
  assert.equal(vmRes.ok, true, vmRes.formattedDiagnostics.join('\n'));

  const expected = [
    'inner:99',
    'outer:10',
    'short_circuit_and_ok',
    'short_circuit_or_ok',
    'loop_acc:16',
    'fact5=120',
    'bits:26',
  ];
  assert.deepEqual(interpRes.execution?.stdout, expected);
  assert.deepEqual(vmRes.execution?.stdout, expected);
});

test('Nova 0.1 Runtime Errors: division by zero and out-of-bounds index reported cleanly in both Interpreter and VM', () => {
  const divZero = 'fn main() {\n  mut d = 0\n  print(10 / d)\n}';
  const rInterp = compileAndRun(divZero, { engine: 'interpreter' });
  const rVm = compileAndRun(divZero, { engine: 'vm' });
  assert.equal(rInterp.ok, false);
  assert.equal(rVm.ok, false);
  assert.match(rInterp.diagnostics[0].message, /division by zero/i);
  assert.match(rVm.diagnostics[0].message, /division by zero/i);

  const oob = 'fn main() {\n  let a = [1, 2]\n  print(a[5])\n}';
  const oInterp = compileAndRun(oob, { engine: 'interpreter' });
  const oVm = compileAndRun(oob, { engine: 'vm' });
  assert.equal(oInterp.ok, false);
  assert.equal(oVm.ok, false);
  assert.match(oInterp.diagnostics[0].message, /out of bounds/i);
  assert.match(oVm.diagnostics[0].message, /out of bounds/i);
});

test('Nova 0.1 REPL: stateful NovaInterpreter persists bindings and functions across chunks', () => {
  const repl = new NovaInterpreter();
  assert.equal(repl.evalReplChunk('let x = 21').ok, true);
  assert.equal(repl.evalReplChunk('fn double(n: Int) -> Int { return n * 2 }').ok, true);
  const res = repl.evalReplChunk('double(x)');
  assert.equal(res.ok, true);
  assert.equal(res.lastValue, '42');
});

// ============================================================================
// 5. LIVE EXECUTABLE CLI TESTS (`nova run hello.nova`, `check`, `repl`, etc.)
// ============================================================================

test('Nova 0.1 CLI: nova run hello.nova, --engine=interpreter, check, tokens, ast, and error exit codes', () => {
  const outVm = execFileSync('nova', ['run', 'hello.nova'], { encoding: 'utf8' }).trim();
  assert.equal(outVm, 'Hello, Nova');

  const outInterp = execFileSync('nova', ['run', '--engine=interpreter', 'hello.nova'], {
    encoding: 'utf8',
  }).trim();
  assert.equal(outInterp, 'Hello, Nova');

  const outCheck = execFileSync('nova', ['check', 'hello.nova'], { encoding: 'utf8' }).trim();
  assert.match(outCheck, /^OK: hello\.nova/);

  const outTokens = execFileSync('nova', ['tokens', 'hello.nova'], { encoding: 'utf8' });
  assert.ok(outTokens.includes('Fn'));
  assert.ok(outTokens.includes('"Hello, "'));

  const outAst = execFileSync('nova', ['ast', 'hello.nova'], { encoding: 'utf8' });
  const parsedAst = JSON.parse(outAst);
  assert.equal(parsedAst.kind, 'Program');

  // Verify `nova check` exits non-zero on a type-invalid file
  const tmpBad = path.join(os.tmpdir(), `nova_bad_${Date.now()}.nova`);
  fs.writeFileSync(tmpBad, 'fn main() {\n  let age: Int = "Hello"\n}\n', 'utf8');
  try {
    const failed = spawnSync('nova', ['check', tmpBad], { encoding: 'utf8' });
    assert.notEqual(failed.status, 0);
    assert.match(failed.stderr, /E0210/);
  } finally {
    if (fs.existsSync(tmpBad)) fs.unlinkSync(tmpBad);
  }

  // Verify piped `nova repl`
  const replProc = spawnSync('nova', ['repl'], {
    input: 'let lang = "Nova"\nprint("Hi " + lang)\n:quit\n',
    encoding: 'utf8',
  });
  assert.equal(replProc.status, 0);
  assert.equal(replProc.stdout.trim(), 'Hi Nova');
});
