import { ExecutionEngine } from '../nova/index';
import { Program } from '../nova/ast';
import { NativeCodegenResult } from '../nova/codegen-c';
import { Diagnostic } from '../nova/diagnostics';
import { OptimizationOptions, OptimizationStats } from '../nova/ir';
import { Token } from '../nova/lexer';
import { LLVMCodegenResult } from '../nova/llvm';
import { BytecodeVerificationResult } from '../nova/verifier';
import { VMExecutionResult } from '../nova/vm';

export type NativeBuildProfile = 'debug' | 'release';

export interface CompileApiResponse {
  ok: boolean;
  engine?: ExecutionEngine;
  tokens: Token[];
  ast: Program;
  irText: string;
  optimizationStats: OptimizationStats | null;
  disassembly: string;
  bytecodeHex?: string;
  bytecodeMeta: {
    magic: string;
    version?: number;
    byteSize: number;
    constantCount: number;
    functionCount: number;
  } | null;
  bytecodeVerification?: BytecodeVerificationResult | null;
  llvmCodegen?: LLVMCodegenResult | null;
  nativeCodegen: NativeCodegenResult | null;
  execution: VMExecutionResult | null;
  diagnostics: Diagnostic[];
  formattedDiagnostics: string[];
  imports: string[];
  symbolCount: number;
}

export interface NativeVerifyResponse {
  ok: boolean;
  supported: boolean;
  equivalent?: boolean;
  reason?: string;
  profile?: NativeBuildProfile;
  backendUsed?: 'llvm-clang' | 'gcc-c99';
  targetTriple?: string;
  stripped?: boolean;
  programObjBytes?: number;
  runtimeObjBytes?: number;
  preStripBytes?: number;
  vmStdout?: string[];
  interpreterStdout?: string[];
  nativeStdout?: string[];
  compileMs?: number;
  linkMs?: number;
  execMs?: number;
  binaryBytes?: number;
  llvmIR?: string;
  cSource?: string;
  formattedDiagnostics?: string[];
}

export interface WorkspaceFileEntry {
  path: string;
  sizeBytes: number;
  lines: number;
  category: 'manifest' | 'crate' | 'example' | 'docs';
}

export interface ToolchainStatus {
  novaVersion: string;
  node: { available: boolean; version: string };
  gcc: { available: boolean; version: string };
  rustc: { available: boolean; version: string };
  cargo: { available: boolean; version: string };
  llvm: { available: boolean; version: string };
}

export type { OptimizationOptions };
