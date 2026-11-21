import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { generateNativeC } from './codegen-c';
import { IRModule } from './ir';
import { defaultHostTargetTriple, LLVMValidationResult, lowerToLLVMIR } from './llvm';
import { NOVA_RUNTIME_C, NOVA_RUNTIME_H } from './runtime-c';

export type BuildProfile = 'debug' | 'release';
export type NativeBackendPreference = 'auto' | 'llvm' | 'gcc';

export interface NativeToolchainInfo {
  clang: boolean;
  llc: boolean;
  gcc: boolean;
  strip: boolean;
  targetTriple: string;
}

export interface NativeBuildOptions {
  profile?: BuildProfile;
  backend?: NativeBackendPreference;
  outputPath?: string;
  emitLlvmPath?: string;
  moduleName?: string;
}

export interface NativeBuildResult {
  ok: boolean;
  supported: boolean;
  reason?: string;
  profile: BuildProfile;
  backendUsed?: 'llvm-clang' | 'gcc-c99';
  targetTriple: string;
  outputPath?: string;
  emitLlvmPath?: string;
  llvmIR?: string;
  llvmValidation?: LLVMValidationResult;
  cSource?: string;
  programObjBytes?: number;
  runtimeObjBytes?: number;
  preStripBytes?: number;
  binaryBytes?: number;
  stripped?: boolean;
  compileMs?: number;
  linkMs?: number;
}

function hasCommand(bin: string): boolean {
  try {
    execFileSync(bin, ['--version'], { stdio: 'ignore', timeout: 2000 });
    return true;
  } catch {
    return false;
  }
}

export function detectNativeToolchain(): NativeToolchainInfo {
  return {
    clang: hasCommand('clang'),
    llc: hasCommand('llc'),
    gcc: hasCommand('gcc'),
    strip: hasCommand('strip'),
    targetTriple: defaultHostTargetTriple(),
  };
}

/**
 * Multi-stage Nova Native Build & Linking Driver:
 *   1. Lowers Nova IR -> validated LLVM IR (`.ll`) and `libnovart`-compatible C99 unit.
 *   2. Compiles `libnovart` runtime (`novart.c` -> `libnovart.o`) separately.
 *   3. Compiles program translation unit (`program.ll` via Clang or `program.c` via GCC -> `program.o`).
 *   4. Links `program.o` + `libnovart.o` + `-lm` into a standalone native ELF executable.
 *   5. In `release` profile (`-O3`), applies section GC (`--gc-sections`) and `strip --strip-all`.
 */
export function buildNativeExecutable(
  ir: IRModule,
  options: NativeBuildOptions = {}
): NativeBuildResult {
  const profile: BuildProfile = options.profile ?? 'debug';
  const backendPref: NativeBackendPreference = options.backend ?? 'auto';
  const toolchain = detectNativeToolchain();

  const llvmRes = lowerToLLVMIR(ir, {
    targetTriple: toolchain.targetTriple,
    moduleName: options.moduleName ?? 'nova_program',
  });
  if (!llvmRes.supported || !llvmRes.llvmIR) {
    return {
      ok: false,
      supported: false,
      profile,
      targetTriple: toolchain.targetTriple,
      reason: llvmRes.reason ?? 'Program uses features unsupported by native backend.',
    };
  }
  if (llvmRes.validation && !llvmRes.validation.ok) {
    return {
      ok: false,
      supported: true,
      profile,
      targetTriple: toolchain.targetTriple,
      llvmIR: llvmRes.llvmIR,
      llvmValidation: llvmRes.validation,
      reason: `LLVM IR validation failed: ${llvmRes.validation.errors.join('; ')}`,
    };
  }

  const cRes = generateNativeC(ir);
  if (!cRes.supported || !cRes.cSource) {
    return {
      ok: false,
      supported: false,
      profile,
      targetTriple: toolchain.targetTriple,
      llvmIR: llvmRes.llvmIR,
      llvmValidation: llvmRes.validation,
      reason: cRes.reason ?? 'Native C99 codegen unsupported for this program.',
    };
  }

  // Decide concrete native object compiler
  let backendUsed: 'llvm-clang' | 'gcc-c99';
  if (backendPref === 'llvm') {
    if (!toolchain.clang) {
      return {
        ok: false,
        supported: true,
        profile,
        targetTriple: toolchain.targetTriple,
        llvmIR: llvmRes.llvmIR,
        llvmValidation: llvmRes.validation,
        reason: 'LLVM native backend requested (--backend=llvm), but clang is not installed in this environment. Emit LLVM IR with `nova llvm` or build with `--backend=auto`.',
      };
    }
    backendUsed = 'llvm-clang';
  } else if (backendPref === 'gcc') {
    if (!toolchain.gcc) {
      return {
        ok: false,
        supported: true,
        profile,
        targetTriple: toolchain.targetTriple,
        reason: 'GCC backend requested, but gcc is not installed.',
      };
    }
    backendUsed = 'gcc-c99';
  } else {
    if (toolchain.clang) backendUsed = 'llvm-clang';
    else if (toolchain.gcc) backendUsed = 'gcc-c99';
    else {
      return {
        ok: false,
        supported: true,
        profile,
        targetTriple: toolchain.targetTriple,
        llvmIR: llvmRes.llvmIR,
        llvmValidation: llvmRes.validation,
        reason: 'Neither clang nor gcc is installed on host to assemble and link native executables.',
      };
    }
  }

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-build-'));
  const headerPath = path.join(tmpDir, 'novart.h');
  const runtimeCPath = path.join(tmpDir, 'novart.c');
  const runtimeObjPath = path.join(tmpDir, 'libnovart.o');
  const programLLPath = path.join(tmpDir, 'program.ll');
  const programCPath = path.join(tmpDir, 'program.c');
  const programObjPath = path.join(tmpDir, 'program.o');
  const outBinary = options.outputPath ?? path.join(tmpDir, 'nova_out');

  try {
    fs.writeFileSync(headerPath, NOVA_RUNTIME_H, 'utf8');
    fs.writeFileSync(runtimeCPath, NOVA_RUNTIME_C, 'utf8');
    fs.writeFileSync(programLLPath, llvmRes.llvmIR, 'utf8');
    fs.writeFileSync(programCPath, cRes.cSource, 'utf8');

    if (options.emitLlvmPath) {
      fs.writeFileSync(options.emitLlvmPath, llvmRes.llvmIR, 'utf8');
    }

    const optFlags =
      profile === 'release'
        ? ['-O3', '-DNDEBUG', '-ffunction-sections', '-fdata-sections']
        : ['-O0', '-g'];

    const cc = backendUsed === 'llvm-clang' ? 'clang' : 'gcc';

    // Stage 1: Compile libnovart.o and program.o separately
    const compileStart = performance.now();
    execFileSync(cc, ['-std=c99', ...optFlags, '-I', tmpDir, '-c', runtimeCPath, '-o', runtimeObjPath], {
      timeout: 15000,
      encoding: 'utf8',
    });

    if (backendUsed === 'llvm-clang') {
      execFileSync('clang', [...optFlags, '-Wno-override-module', '-c', programLLPath, '-o', programObjPath], {
        timeout: 15000,
        encoding: 'utf8',
      });
    } else {
      execFileSync('gcc', ['-std=c99', ...optFlags, '-I', tmpDir, '-c', programCPath, '-o', programObjPath], {
        timeout: 15000,
        encoding: 'utf8',
      });
    }
    const compileMs = Number((performance.now() - compileStart).toFixed(2));

    const programObjBytes = fs.statSync(programObjPath).size;
    const runtimeObjBytes = fs.statSync(runtimeObjPath).size;

    // Stage 2: Link program.o + libnovart.o + libm into native ELF binary
    const linkStart = performance.now();
    const linkFlags = profile === 'release' ? ['-Wl,--gc-sections'] : [];
    execFileSync(cc, [programObjPath, runtimeObjPath, ...linkFlags, '-lm', '-o', outBinary], {
      timeout: 15000,
      encoding: 'utf8',
    });

    const preStripBytes = fs.statSync(outBinary).size;
    let stripped = false;
    if (profile === 'release' && toolchain.strip) {
      execFileSync('strip', ['--strip-all', outBinary], { timeout: 5000 });
      stripped = true;
    }
    const linkMs = Number((performance.now() - linkStart).toFixed(2));
    const binaryBytes = fs.statSync(outBinary).size;

    return {
      ok: true,
      supported: true,
      profile,
      backendUsed,
      targetTriple: toolchain.targetTriple,
      outputPath: outBinary,
      emitLlvmPath: options.emitLlvmPath,
      llvmIR: llvmRes.llvmIR,
      llvmValidation: llvmRes.validation,
      cSource: cRes.cSource,
      programObjBytes,
      runtimeObjBytes,
      preStripBytes,
      binaryBytes,
      stripped,
      compileMs,
      linkMs,
    };
  } catch (err) {
    return {
      ok: false,
      supported: true,
      profile,
      backendUsed,
      targetTriple: toolchain.targetTriple,
      llvmIR: llvmRes.llvmIR,
      llvmValidation: llvmRes.validation,
      reason: err instanceof Error ? err.message : String(err),
    };
  } finally {
    // Only remove temp dir if caller supplied an external outputPath
    if (options.outputPath) {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  }
}

export function executeNativeBinary(
  binaryPath: string,
  timeoutMs = 5000
): { exitCode: number; stdout: string[]; stderr: string; execMs: number } {
  const start = performance.now();
  try {
    const rawOut = execFileSync(binaryPath, [], { timeout: timeoutMs, encoding: 'utf8' });
    const execMs = Number((performance.now() - start).toFixed(2));
    const stdout = rawOut.length === 0 ? [] : rawOut.replace(/\n$/, '').split('\n');
    return { exitCode: 0, stdout, stderr: '', execMs };
  } catch (err: unknown) {
    const execMs = Number((performance.now() - start).toFixed(2));
    const e = err as { status?: number; stdout?: string | Buffer; stderr?: string | Buffer };
    const outStr = typeof e.stdout === 'string' ? e.stdout : (e.stdout?.toString('utf8') ?? '');
    const errStr = typeof e.stderr === 'string' ? e.stderr : (e.stderr?.toString('utf8') ?? '');
    return {
      exitCode: e.status ?? 1,
      stdout: outStr.length === 0 ? [] : outStr.replace(/\n$/, '').split('\n'),
      stderr: errStr,
      execMs,
    };
  }
}
