import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { compileAndRun } from './index';
import {
  buildNativeExecutable,
  executeNativeBinary,
  sanitizeModuleName,
} from './native-builder';

test('sanitizeModuleName strips traversal, quotes, and LLVM IR injection characters', () => {
  assert.equal(sanitizeModuleName('../../etc/passwd'), 'passwd');
  assert.equal(sanitizeModuleName('bad"\n;define i32 @pwn().nova'), 'bad___define_i32__pwn__.nova');
  assert.equal(sanitizeModuleName(''), 'nova_program');
  assert.equal(sanitizeModuleName('..'), 'nova_program');
  assert.ok(sanitizeModuleName('a'.repeat(200)).length <= 64);
});

test('Hello Nova example produces identical output across Interpreter, VM, and Native ELF', () => {
  const helloPath = path.resolve(process.cwd(), 'nova/examples/hello.nova');
  const source = fs.readFileSync(helloPath, 'utf8');

  const interp = compileAndRun(source, {
    filename: 'hello.nova',
    execute: true,
    engine: 'interpreter',
  });
  const vm = compileAndRun(source, {
    filename: 'hello.nova',
    execute: true,
    engine: 'vm',
  });

  assert.equal(interp.ok, true);
  assert.equal(vm.ok, true);
  assert.deepEqual(interp.execution?.stdout, ['Hello, Nova']);
  assert.deepEqual(vm.execution?.stdout, ['Hello, Nova']);

  assert.ok(vm.irOptimized, 'Expected optimized IR');
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-hello-test-'));
  const outBin = path.join(tmpDir, 'hello');
  try {
    const built = buildNativeExecutable(vm.irOptimized, {
      profile: 'release',
      backend: 'auto',
      outputPath: outBin,
      moduleName: 'hello.nova',
    });
    assert.equal(built.ok, true, built.reason);
    assert.ok(fs.existsSync(outBin), 'Expected compiled native executable on disk');

    const nativeRun = executeNativeBinary(outBin, 4000);
    assert.equal(nativeRun.exitCode, 0);
    assert.deepEqual(nativeRun.stdout, ['Hello, Nova']);
    assert.deepEqual(nativeRun.stdout, interp.execution?.stdout);
    assert.deepEqual(nativeRun.stdout, vm.execution?.stdout);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test('executeNativeBinary runs with sanitized environment (strips host secrets)', () => {
  const prevSecret = process.env.NOVA_TEST_SECRET_TOKEN;
  process.env.NOVA_TEST_SECRET_TOKEN = 'top-secret-value';
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-env-test-'));
  const probeScript = path.join(tmpDir, 'probe.sh');
  fs.writeFileSync(
    probeScript,
    '#!/bin/sh\nif [ -z "$NOVA_TEST_SECRET_TOKEN" ]; then echo "CLEAN"; else echo "LEAKED"; fi\n',
    { mode: 0o755 }
  );
  try {
    const res = executeNativeBinary(probeScript, 3000);
    assert.equal(res.exitCode, 0);
    assert.deepEqual(res.stdout, ['CLEAN']);
  } finally {
    if (prevSecret === undefined) {
      delete process.env.NOVA_TEST_SECRET_TOKEN;
    } else {
      process.env.NOVA_TEST_SECRET_TOKEN = prevSecret;
    }
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});
