#!/usr/bin/env tsx
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import {
  compileAndRun,
  decodeBytecode,
  encodeBytecode,
  ExecutionEngine,
  formatDiagnostic,
  formatNovaSource,
  lex,
  NovaInterpreter,
  NovaVM,
  parse,
  TokenKind,
  verifyBytecode,
} from '../src/nova/index';
import { buildNativeExecutable, NativeBackendPreference } from '../src/nova/native-builder';

const VERSION = '0.1.0';

function printHelp(): void {
  console.log(`Nova Programming Language CLI v${VERSION}

Usage:
  nova run <file.nova|file.nvbc> [--engine=vm|interpreter]
                                    Compile and execute a Nova program or .nvbc bytecode
  nova compile <file.nova> [-o out.nvbc]
                                    Compile and verify Nova source to binary .nvbc bytecode
  nova verify <file.nova|file.nvbc>
                                    Run static bytecode verifier (CFG & stack-depth check)
  nova llvm <file.nova> [-o out.ll] Lower Nova IR to verified LLVM IR (.ll)
  nova build <file.nova> [--debug|--release] [--backend=auto|gcc|llvm] [--emit-llvm] [-o out]
                                    Compile objects, link libnovart, and build native executable
  nova check <file.nova>            Lex, parse, and type-check without executing
  nova tokens <file.nova>           Print the lexer token stream with spans
  nova ast <file.nova>              Print the parsed Abstract Syntax Tree (JSON)
  nova ir <file.nova>               Emit optimized Nova Intermediate Representation
  nova bytecode <file.nova>         Disassemble compiled Nova VM bytecode
  nova fmt <file.nova>              Print canonically formatted Nova source
  nova repl                         Start an interactive Nova REPL session
  nova test                         Run the full Nova compiler, verifier, VM, and native test suite
  nova doctor                       Inspect installed host toolchains
  nova version                      Print Nova compiler version
`);
}

function readSource(filePath: string): string {
  if (!fs.existsSync(filePath)) {
    console.error(`error: file not found '${filePath}'`);
    process.exit(1);
  }
  return fs.readFileSync(filePath, 'utf8');
}

function parseEngineFlag(args: string[]): ExecutionEngine {
  const flag = args.find((a) => a.startsWith('--engine='));
  if (!flag) return 'vm';
  const val = flag.slice('--engine='.length);
  if (val === 'interpreter' || val === 'vm') return val;
  console.error(`error: invalid --engine value '${val}' (expected 'vm' or 'interpreter')`);
  process.exit(1);
}

function runRepl(): void {
  const interp = new NovaInterpreter();
  const isTTY = process.stdin.isTTY;
  if (isTTY) {
    console.log(`Nova ${VERSION} Interactive REPL (type :quit or Ctrl+D to exit)`);
  }
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: isTTY ? 'nova> ' : '',
  });
  if (isTTY) rl.prompt();

  rl.on('line', (rawLine) => {
    const line = rawLine.trim();
    if (line === ':quit' || line === ':exit') {
      rl.close();
      return;
    }
    if (line.length > 0) {
      const res = interp.evalReplChunk(line);
      if (!res.ok) {
        for (const d of res.diagnostics) {
          console.error(formatDiagnostic(d, line, '<repl>'));
        }
      } else {
        for (const out of res.stdout) console.log(out);
        if (res.lastValue !== undefined) console.log(`=> ${res.lastValue}`);
      }
    }
    if (isTTY) rl.prompt();
  });

  rl.on('close', () => {
    process.exit(0);
  });
}

function main(): void {
  const args = process.argv.slice(2);
  const cmd = args[0];

  if (!cmd || cmd === '--help' || cmd === '-h') {
    printHelp();
    process.exit(0);
  }

  if (cmd === 'version' || cmd === '--version' || cmd === '-V') {
    console.log(`nova ${VERSION}`);
    process.exit(0);
  }

  if (cmd === 'repl') {
    runRepl();
    return;
  }

  if (cmd === 'test') {
    const projectRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
    const testFiles = [
      path.join(projectRoot, 'src/nova/compiler.test.ts'),
      path.join(projectRoot, 'src/nova/nova01.test.ts'),
      path.join(projectRoot, 'src/nova/vm-equivalence.test.ts'),
      path.join(projectRoot, 'src/nova/native-llvm.test.ts'),
      path.join(projectRoot, 'src/nova/ai-engine.test.ts'),
      path.join(projectRoot, 'src/nova/security.test.ts'),
    ].filter((f) => fs.existsSync(f));
    execFileSync('npx', ['--yes', 'tsx', '--test', ...testFiles], {
      cwd: projectRoot,
      stdio: 'inherit',
    });
    process.exit(0);
  }

  if (cmd === 'doctor') {
    const checkBin = (bin: string): string => {
      try {
        return execFileSync(bin, ['--version'], { encoding: 'utf8' }).split('\n')[0].trim();
      } catch {
        return 'not installed';
      }
    };
    console.log(`Nova CLI:    ${VERSION}`);
    console.log(`Node.js:     ${process.version}`);
    console.log(`GCC:         ${checkBin('gcc')}`);
    console.log(`rustc:       ${checkBin('rustc')}`);
    console.log(`cargo:       ${checkBin('cargo')}`);
    console.log(`llvm-config: ${checkBin('llvm-config')}`);
    process.exit(0);
  }

  const positional: string[] = [];
  for (let i = 1; i < args.length; i++) {
    if (args[i] === '-o') {
      i++; // skip output path argument
      continue;
    }
    if (!args[i].startsWith('--')) positional.push(args[i]);
  }
  const file = positional[0];
  if (!file) {
    console.error(`error: missing source file for 'nova ${cmd}'`);
    printHelp();
    process.exit(1);
  }

  if (cmd === 'verify') {
    if (!fs.existsSync(file)) {
      console.error(`error: file not found '${file}'`);
      process.exit(1);
    }
    if (file.endsWith('.nvbc')) {
      try {
        const bytes = new Uint8Array(fs.readFileSync(file));
        const chunk = decodeBytecode(bytes);
        const verification = verifyBytecode(chunk);
        if (!verification.ok) {
          verification.diagnostics.forEach((d) => console.error(`error[${d.code}]: ${d.message}`));
          process.exit(1);
        }
        console.log(
          `VERIFIED: ${file} (${verification.functionsVerified} functions, ${verification.instructionsVerified} instructions, max stack ${verification.maxStackDepth}, ${bytes.byteLength} bytes .nvbc)`
        );
        process.exit(0);
      } catch (err) {
        console.error(`error: ${err instanceof Error ? err.message : String(err)}`);
        process.exit(1);
      }
    }
    const source = readSource(file);
    const res = compileAndRun(source, { filename: file, execute: false });
    if (!res.ok || !res.verification || !res.bytecode) {
      res.formattedDiagnostics.forEach((d) => console.error(d));
      process.exit(1);
    }
    console.log(
      `VERIFIED: ${file} (${res.verification.functionsVerified} functions, ${res.verification.instructionsVerified} instructions, max stack ${res.verification.maxStackDepth}, ${res.bytecode.byteSize} bytes .nvbc)`
    );
    process.exit(0);
  }

  if (cmd === 'compile') {
    const source = readSource(file);
    const res = compileAndRun(source, { filename: file, execute: false });
    if (!res.ok || !res.bytecode || !res.verification) {
      res.formattedDiagnostics.forEach((d) => console.error(d));
      process.exit(1);
    }
    const oIdx = args.indexOf('-o');
    const outPath =
      oIdx !== -1 && args[oIdx + 1]
        ? args[oIdx + 1]
        : `${path.basename(file, path.extname(file))}.nvbc`;
    const bytes = encodeBytecode(res.bytecode);
    fs.writeFileSync(outPath, bytes);
    console.log(
      `Compiled & verified ${file} -> ${outPath} (${bytes.byteLength} bytes, ${res.verification.instructionsVerified} insts, max stack ${res.verification.maxStackDepth})`
    );
    process.exit(0);
  }

  if (cmd === 'run' && file.endsWith('.nvbc')) {
    if (!fs.existsSync(file)) {
      console.error(`error: file not found '${file}'`);
      process.exit(1);
    }
    const bytes = new Uint8Array(fs.readFileSync(file));
    const vm = new NovaVM();
    const execRes = vm.runBinary(bytes, { verify: true });
    if (execRes.runtimeError) {
      console.error(`error[${execRes.runtimeError.code}]: ${execRes.runtimeError.message}`);
      process.exit(1);
    }
    for (const line of execRes.stdout) console.log(line);
    process.exit(execRes.exitCode);
  }

  const source = readSource(file);

  if (cmd === 'tokens') {
    const { tokens, diagnostics } = lex(source);
    if (diagnostics.length > 0) {
      diagnostics.forEach((d) => console.error(formatDiagnostic(d, source, file)));
      process.exit(1);
    }
    for (const tok of tokens) {
      if (tok.kind === TokenKind.Newline) continue;
      console.log(
        `${String(tok.span.line).padStart(3)}:${String(tok.span.column).padEnd(3)} ${tok.kind.padEnd(10)} ${tok.lexeme === '' ? 'EOF' : JSON.stringify(tok.lexeme)}`
      );
    }
    process.exit(0);
  }

  if (cmd === 'ast') {
    const lexRes = lex(source);
    const parseRes = parse(lexRes.tokens);
    const diags = [...lexRes.diagnostics, ...parseRes.diagnostics];
    if (diags.length > 0) {
      diags.forEach((d) => console.error(formatDiagnostic(d, source, file)));
      process.exit(1);
    }
    console.log(JSON.stringify(parseRes.program, null, 2));
    process.exit(0);
  }

  if (cmd === 'fmt') {
    const res = formatNovaSource(source);
    if (!res.ok) {
      res.diagnostics.forEach((d) => console.error(formatDiagnostic(d, source, file)));
      process.exit(1);
    }
    process.stdout.write(res.formatted);
    process.exit(0);
  }

  if (cmd === 'check') {
    const res = compileAndRun(source, { filename: file, execute: false });
    if (!res.ok) {
      res.formattedDiagnostics.forEach((d) => console.error(d));
      process.exit(1);
    }
    console.log(`OK: ${file} (${res.tokens.length} tokens, ${res.bytecode?.functions.length ?? 0} functions, 0 errors)`);
    process.exit(0);
  }

  if (cmd === 'ir') {
    const res = compileAndRun(source, { filename: file, execute: false });
    if (!res.ok) {
      res.formattedDiagnostics.forEach((d) => console.error(d));
      process.exit(1);
    }
    console.log(res.irText ?? '');
    process.exit(0);
  }

  if (cmd === 'bytecode') {
    const res = compileAndRun(source, { filename: file, execute: false });
    if (!res.ok) {
      res.formattedDiagnostics.forEach((d) => console.error(d));
      process.exit(1);
    }
    console.log(res.disassembly ?? '');
    process.exit(0);
  }

  if (cmd === 'run') {
    const engine = parseEngineFlag(args.slice(1));
    const res = compileAndRun(source, { filename: file, execute: true, engine });
    if (!res.ok) {
      res.formattedDiagnostics.forEach((d) => console.error(d));
      process.exit(1);
    }
    for (const line of res.execution?.stdout ?? []) {
      console.log(line);
    }
    process.exit(res.execution?.exitCode ?? 0);
  }

  if (cmd === 'llvm') {
    const res = compileAndRun(source, { filename: file, execute: false });
    if (!res.ok) {
      res.formattedDiagnostics.forEach((d) => console.error(d));
      process.exit(1);
    }
    if (!res.llvmCodegen?.supported || !res.llvmCodegen.llvmIR) {
      console.error(`error: ${res.llvmCodegen?.reason ?? 'LLVM IR lowering unsupported for this program'}`);
      process.exit(1);
    }
    const oIdx = args.indexOf('-o');
    if (oIdx !== -1 && args[oIdx + 1]) {
      fs.writeFileSync(args[oIdx + 1], res.llvmCodegen.llvmIR, 'utf8');
      console.log(
        `Emitted verified LLVM IR: ${args[oIdx + 1]} (${res.llvmCodegen.targetTriple}, ${res.llvmCodegen.functionCount} fns, ${res.llvmCodegen.blockCount} blocks)`
      );
    } else {
      console.log(res.llvmCodegen.llvmIR);
    }
    process.exit(0);
  }

  if (cmd === 'build') {
    const release = args.includes('--release');
    const emitLlvm = args.includes('--emit-llvm');
    const backendFlag = args.find((a) => a.startsWith('--backend='));
    const backend: NativeBackendPreference = backendFlag
      ? (backendFlag.slice('--backend='.length) as NativeBackendPreference)
      : 'auto';

    const res = compileAndRun(source, { filename: file, execute: false });
    if (!res.ok || !res.irOptimized) {
      res.formattedDiagnostics.forEach((d) => console.error(d));
      process.exit(1);
    }

    const baseName = path.basename(file, path.extname(file));
    const oIdx = args.indexOf('-o');
    const outBinary =
      oIdx !== -1 && args[oIdx + 1]
        ? path.resolve(process.cwd(), args[oIdx + 1])
        : path.resolve(process.cwd(), baseName);
    const emitLlvmPath = emitLlvm ? `${outBinary}.ll` : undefined;

    const buildRes = buildNativeExecutable(res.irOptimized, {
      profile: release ? 'release' : 'debug',
      backend,
      outputPath: outBinary,
      emitLlvmPath,
      moduleName: path.basename(file),
    });

    if (!buildRes.ok) {
      console.error(`error: ${buildRes.reason ?? 'Native build failed'}`);
      process.exit(1);
    }

    const stripInfo = buildRes.stripped
      ? `, stripped ${buildRes.preStripBytes} -> ${buildRes.binaryBytes} bytes`
      : `, ${buildRes.binaryBytes} bytes`;
    console.log(
      `Built native executable: ${buildRes.outputPath} (${buildRes.profile}, ${buildRes.backendUsed}, ${buildRes.targetTriple}${stripInfo})`
    );
    if (emitLlvmPath) {
      console.log(`Emitted LLVM IR: ${emitLlvmPath}`);
    }
    process.exit(0);
  }

  console.error(`error: unknown command '${cmd}'`);
  printHelp();
  process.exit(1);
}

main();
