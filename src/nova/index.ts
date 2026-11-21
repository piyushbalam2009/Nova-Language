import { Program } from './ast';
import {
  BytecodeChunk,
  compileToBytecode,
  disassembleBytecode,
  encodeBytecode,
  formatBytecodeHex,
} from './bytecode';
import { generateNativeC, NativeCodegenResult } from './codegen-c';
import { Diagnostic, formatDiagnostic } from './diagnostics';
import { formatProgram } from './formatter';
import { interpretProgram } from './interpreter';
import { formatIR, generateIR, IRModule, OptimizationOptions, OptimizationStats, optimizeIR, validateIR } from './ir';
import { lex, Token, TokenKind } from './lexer';
import { LLVMCodegenResult, lowerToLLVMIR } from './llvm';
import { parse } from './parser';
import { analyze, SemanticResult } from './semantic';
import { BytecodeVerificationResult, verifyBytecode } from './verifier';
import { NovaVM, VMExecutionResult } from './vm';

export type ExecutionEngine = 'vm' | 'interpreter';

export interface CompilationOutput {
  ok: boolean;
  engine: ExecutionEngine;
  tokens: Token[];
  ast: Program;
  semantic?: SemanticResult;
  irRaw?: IRModule;
  irOptimized?: IRModule;
  irText?: string;
  optimizationStats?: OptimizationStats;
  bytecode?: BytecodeChunk;
  verification?: BytecodeVerificationResult;
  disassembly?: string;
  bytecodeHex?: string;
  llvmCodegen?: LLVMCodegenResult;
  nativeCodegen?: NativeCodegenResult;
  execution?: VMExecutionResult;
  diagnostics: Diagnostic[];
  formattedDiagnostics: string[];
}

export const DEFAULT_OPT_OPTIONS: OptimizationOptions = {
  constantFolding: true,
  constantPropagation: true,
  deadCodeElimination: true,
};

export function compileAndRun(
  source: string,
  options: {
    filename?: string;
    execute?: boolean;
    engine?: ExecutionEngine;
    optimizations?: OptimizationOptions;
  } = {}
): CompilationOutput {
  const filename = options.filename ?? 'main.nova';
  const shouldExecute = options.execute ?? true;
  const engine: ExecutionEngine = options.engine ?? 'vm';
  const optConfig = options.optimizations ?? DEFAULT_OPT_OPTIONS;

  const lexRes = lex(source);
  const visibleTokens = lexRes.tokens.filter((t) => t.kind !== TokenKind.Newline);
  const parseRes = parse(lexRes.tokens);

  const diagnostics: Diagnostic[] = [...lexRes.diagnostics, ...parseRes.diagnostics];

  if (diagnostics.length > 0) {
    return {
      ok: false,
      engine,
      tokens: visibleTokens,
      ast: parseRes.program,
      diagnostics,
      formattedDiagnostics: diagnostics.map((d) => formatDiagnostic(d, source, filename)),
    };
  }

  const semantic = analyze(parseRes.program);
  diagnostics.push(...semantic.diagnostics);

  if (diagnostics.length > 0) {
    return {
      ok: false,
      engine,
      tokens: visibleTokens,
      ast: parseRes.program,
      semantic,
      diagnostics,
      formattedDiagnostics: diagnostics.map((d) => formatDiagnostic(d, source, filename)),
    };
  }

  const irRaw = generateIR(parseRes.program);
  const { module: irOptimized, stats: optimizationStats } = optimizeIR(irRaw, optConfig);
  const irDiags = validateIR(irOptimized);
  diagnostics.push(...irDiags);

  if (diagnostics.length > 0) {
    return {
      ok: false,
      engine,
      tokens: visibleTokens,
      ast: parseRes.program,
      semantic,
      irRaw,
      irOptimized,
      irText: formatIR(irOptimized),
      optimizationStats,
      diagnostics,
      formattedDiagnostics: diagnostics.map((d) => formatDiagnostic(d, source, filename)),
    };
  }

  const bytecode = compileToBytecode(irOptimized);
  const verification = verifyBytecode(bytecode);
  diagnostics.push(...verification.diagnostics);

  const encoded = encodeBytecode(bytecode);
  bytecode.byteSize = encoded.byteLength;
  const disassembly = disassembleBytecode(bytecode);
  const bytecodeHex = formatBytecodeHex(encoded, 128);
  const llvmCodegen = lowerToLLVMIR(irOptimized, { moduleName: filename });
  const nativeCodegen = generateNativeC(irOptimized);

  if (diagnostics.length > 0) {
    return {
      ok: false,
      engine,
      tokens: visibleTokens,
      ast: parseRes.program,
      semantic,
      irRaw,
      irOptimized,
      irText: formatIR(irOptimized),
      optimizationStats,
      bytecode,
      verification,
      disassembly,
      bytecodeHex,
      llvmCodegen,
      nativeCodegen,
      diagnostics,
      formattedDiagnostics: diagnostics.map((d) => formatDiagnostic(d, source, filename)),
    };
  }

  let execution: VMExecutionResult | undefined;
  if (shouldExecute) {
    if (engine === 'interpreter') {
      execution = interpretProgram(parseRes.program);
    } else {
      const vm = new NovaVM();
      execution = vm.run(bytecode, { verify: false }); // already verified above
    }
    if (execution.runtimeError) {
      diagnostics.push(execution.runtimeError);
    }
  }

  return {
    ok: diagnostics.length === 0,
    engine,
    tokens: visibleTokens,
    ast: parseRes.program,
    semantic,
    irRaw,
    irOptimized,
    irText: formatIR(irOptimized),
    optimizationStats,
    bytecode,
    verification,
    disassembly,
    bytecodeHex,
    llvmCodegen,
    nativeCodegen,
    execution,
    diagnostics,
    formattedDiagnostics: diagnostics.map((d) => formatDiagnostic(d, source, filename)),
  };
}

export function formatNovaSource(source: string): { ok: boolean; formatted: string; diagnostics: Diagnostic[] } {
  const lexRes = lex(source);
  const parseRes = parse(lexRes.tokens);
  const diagnostics = [...lexRes.diagnostics, ...parseRes.diagnostics];
  if (diagnostics.length > 0) {
    return { ok: false, formatted: source, diagnostics };
  }
  return { ok: true, formatted: formatProgram(parseRes.program), diagnostics: [] };
}

export * from './ai';
export * from './ast';
export * from './bytecode';
export * from './codegen-c';
export * from './diagnostics';
export * from './formatter';
export * from './gc';
export * from './interpreter';
export * from './ir';
export * from './lexer';
export * from './llvm';
export * from './parser';
export * from './runtime-c';
export * from './semantic';
export * from './verifier';
export * from './vm';
