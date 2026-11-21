import { ArrowRight, CheckCircle2, Cpu, AlertCircle } from 'lucide-react';
import { CompileApiResponse, NativeVerifyResponse } from '../types/api';

interface PipelineViewProps {
  filename: string;
  result: CompileApiResponse | null;
  nativeVerify: NativeVerifyResponse | null;
  isVerifyingNative: boolean;
  onVerifyNative: () => void;
}

export function PipelineView({
  filename,
  result,
  nativeVerify,
  isVerifyingNative,
  onVerifyNative,
}: PipelineViewProps) {
  const stages = [
    {
      num: '01',
      name: 'Lexer',
      crate: 'nova-lexer',
      status: result ? (result.diagnostics.some((d) => d.stage === 'lexer') ? 'Error' : 'Passed') : 'Idle',
      metric: result ? `${result.tokens.length} tokens` : '—',
      detail: 'Spanned tokenization with line/col offsets, escapes, comments, and @ matmul operator.',
    },
    {
      num: '02',
      name: 'Pratt Parser & AST',
      crate: 'nova-parser',
      status: result ? (result.diagnostics.some((d) => d.stage === 'parser') ? 'Error' : 'Passed') : 'Idle',
      metric: result ? `${result.ast.items.length} top-level nodes` : '—',
      detail: 'Recursive-descent statements with Pratt operator-precedence expression parsing.',
    },
    {
      num: '03',
      name: 'Semantic & Type Checker',
      crate: 'nova-semantic',
      status: result ? (result.diagnostics.some((d) => d.stage === 'semantic') ? 'Error' : 'Passed') : 'Idle',
      metric: result ? `${result.symbolCount} symbols` : '—',
      detail: 'Lexical scopes, static type inference, let/mut immutability, arity & return validation.',
    },
    {
      num: '04',
      name: 'Nova IR & Optimizer',
      crate: 'nova-ir',
      status: result?.irText ? 'Passed' : 'Waiting',
      metric: result?.optimizationStats
        ? `${result.optimizationStats.instructionsBefore} -> ${result.optimizationStats.instructionsAfter} insts`
        : '—',
      detail: 'Basic-block IR with constant folding, constant propagation, and dead-code elimination.',
    },
    {
      num: '05',
      name: 'Bytecode & Nova VM',
      crate: 'nova-bytecode / nova-vm',
      status: result?.execution ? (result.execution.exitCode === 0 ? 'Passed' : 'Error') : 'Waiting',
      metric: result?.execution ? `${result.execution.instructionsExecuted} ops (${result.execution.durationMs} ms)` : '—',
      detail: 'Validated stack bytecode executed on call-frame VM with mark-and-sweep heap GC.',
    },
    {
      num: '06',
      name: 'Native Codegen (GCC)',
      crate: 'nova-codegen-c / llvm-planned',
      status: nativeVerify?.supported ? (nativeVerify.equivalent ? 'Equivalent' : 'Mismatch') : 'Ready',
      metric: nativeVerify?.binaryBytes ? `${nativeVerify.binaryBytes.toLocaleString()} B ELF` : 'On demand',
      detail: 'Lowers Nova IR to C99 and builds a native x86_64 binary via GCC for equivalence checks.',
    },
  ];

  return (
    <div className="flex-1 overflow-y-auto bg-[#F8FAFC] p-6 lg:p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-display text-2xl font-bold text-slate-900 tracking-tight">
              Hybrid Compiler Pipeline
            </h1>
            <p className="text-sm text-slate-600 mt-1">
              Live stage telemetry for <span className="font-mono text-slate-900">{filename}</span> across both the Nova VM and Native C99/GCC execution backends.
            </p>
          </div>
          <button
            type="button"
            onClick={onVerifyNative}
            disabled={isVerifyingNative}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 disabled:opacity-60 rounded-lg transition-colors whitespace-nowrap"
          >
            <Cpu className="w-3.5 h-3.5" aria-hidden="true" />
            {isVerifyingNative ? 'Compiling Native Binary...' : 'Run Native Equivalence Check'}
          </button>
        </div>

        {/* 6-Stage Pipeline Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {stages.map((stage) => {
            const isErr = stage.status === 'Error' || stage.status === 'Mismatch';
            const isOk = stage.status === 'Passed' || stage.status === 'Equivalent';
            return (
              <div
                key={stage.num}
                className="bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between gap-2 text-xs mb-2">
                    <span className="font-mono text-slate-400">{stage.num}. {stage.crate}</span>
                    <span
                      className={`font-medium ${
                        isErr
                          ? 'text-red-600'
                          : isOk
                          ? 'text-emerald-700'
                          : 'text-slate-500'
                      }`}
                    >
                      {stage.status}
                    </span>
                  </div>
                  <h2 className="font-display text-base font-semibold text-slate-900">
                    {stage.name}
                  </h2>
                  <p className="text-xs text-slate-600 leading-relaxed mt-1.5">
                    {stage.detail}
                  </p>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-100 font-mono text-xs text-slate-800 tabular-nums">
                  {stage.metric}
                </div>
              </div>
            );
          })}
        </div>

        {/* Side-by-Side Execution Equivalence Comparison */}
        <section className="bg-white border border-slate-200 rounded-xl p-6 space-y-5">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 pb-4">
            <div>
              <h2 className="font-display text-base font-semibold text-slate-900">
                Execution-Equivalence Verification
              </h2>
              <p className="text-xs text-slate-600 mt-0.5">
                Compares bytecode execution on the Nova Virtual Machine against a native x86_64 binary compiled by GCC from the same Nova IR.
              </p>
            </div>
            {nativeVerify && (
              <div className="flex items-center gap-2 text-sm font-semibold">
                {nativeVerify.supported && nativeVerify.equivalent ? (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" aria-hidden="true" />
                    <span className="text-emerald-700">Outputs Identical</span>
                  </>
                ) : (
                  <>
                    <AlertCircle className="w-4 h-4 text-amber-600" aria-hidden="true" />
                    <span className="text-amber-700">
                      {nativeVerify.supported ? 'Output Mismatch' : 'VM-Only Features Used'}
                    </span>
                  </>
                )}
              </div>
            )}
          </div>

          {!nativeVerify ? (
            <div className="py-8 text-center">
              <p className="text-sm text-slate-600">
                Run Native Equivalence Check to compile <span className="font-mono">{filename}</span> with GCC and compare stdout side-by-side.
              </p>
            </div>
          ) : !nativeVerify.supported ? (
            <p className="text-sm text-slate-700 bg-slate-50 border border-slate-200 rounded-lg p-4">
              {nativeVerify.reason}
            </p>
          ) : (
            <div className="space-y-5">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div>
                  <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
                    <span className="font-semibold text-slate-700">Nova VM Stdout</span>
                    <span className="font-mono tabular-nums">{result?.execution?.durationMs ?? 0} ms</span>
                  </div>
                  <pre className="p-4 bg-slate-950 text-slate-100 rounded-lg font-mono text-xs overflow-x-auto min-h-[100px]">
                    {(nativeVerify.vmStdout ?? []).join('\n') || '(no output)'}
                  </pre>
                </div>
                <div>
                  <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
                    <span className="font-semibold text-slate-700">Native x86_64 Binary Stdout (GCC -O2)</span>
                    <span className="font-mono tabular-nums">
                      compile {nativeVerify.compileMs} ms · exec {nativeVerify.execMs} ms
                    </span>
                  </div>
                  <pre className="p-4 bg-slate-950 text-slate-100 rounded-lg font-mono text-xs overflow-x-auto min-h-[100px]">
                    {(nativeVerify.nativeStdout ?? []).join('\n') || '(no output)'}
                  </pre>
                </div>
              </div>

              {nativeVerify.cSource && (
                <div>
                  <div className="flex items-center gap-2 text-xs font-semibold text-slate-600 mb-2">
                    <span>Nova IR</span>
                    <ArrowRight className="w-3.5 h-3.5" aria-hidden="true" />
                    <span>Generated C99 Source</span>
                  </div>
                  <pre className="p-4 bg-slate-50 border border-slate-200 rounded-lg font-mono text-xs text-slate-800 overflow-x-auto max-h-80">
                    {nativeVerify.cSource}
                  </pre>
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
