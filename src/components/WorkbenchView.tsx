import React, { useMemo, useState } from 'react';
import { CheckCircle2, Cpu, FileCode2, RotateCcw, ShieldCheck, Wand2, AlertCircle } from 'lucide-react';
import { NOVA_EXAMPLES, NovaExample } from '../data/examples';
import { CompileApiResponse, NativeBuildProfile, NativeVerifyResponse, OptimizationOptions } from '../types/api';
import { ExecutionEngine } from '../nova/index';

type InspectorTab = 'output' | 'tokens' | 'ast' | 'ir' | 'llvm' | 'bytecode';

interface WorkbenchViewProps {
  activeExample: NovaExample;
  source: string;
  onChangeSource: (next: string) => void;
  onSelectExample: (example: NovaExample) => void;
  engine: ExecutionEngine;
  onChangeEngine: (next: ExecutionEngine) => void;
  buildProfile: NativeBuildProfile;
  onChangeBuildProfile: (next: NativeBuildProfile) => void;
  optimizations: OptimizationOptions;
  onChangeOptimizations: (next: OptimizationOptions) => void;
  result: CompileApiResponse | null;
  nativeVerify: NativeVerifyResponse | null;
  isRunning: boolean;
  isVerifyingNative: boolean;
  onCheckOnly: () => void;
  onFormat: () => void;
  onReset: () => void;
  onVerifyNative: () => void;
}

const INSPECTOR_TABS: Array<{ id: InspectorTab; label: string }> = [
  { id: 'output', label: 'Output' },
  { id: 'tokens', label: 'Tokens' },
  { id: 'ast', label: 'AST' },
  { id: 'ir', label: 'Nova IR' },
  { id: 'llvm', label: 'LLVM IR' },
  { id: 'bytecode', label: 'Bytecode' },
];

export function WorkbenchView({
  activeExample,
  source,
  onChangeSource,
  onSelectExample,
  engine,
  onChangeEngine,
  buildProfile,
  onChangeBuildProfile,
  optimizations,
  onChangeOptimizations,
  result,
  nativeVerify,
  isRunning,
  isVerifyingNative,
  onCheckOnly,
  onFormat,
  onReset,
  onVerifyNative,
}: WorkbenchViewProps) {
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>('output');

  const lines = useMemo(() => source.split('\n'), [source]);
  const errorLines = useMemo(() => {
    const set = new Set<number>();
    for (const d of result?.diagnostics ?? []) {
      if (d.span.line > 0) set.add(d.span.line);
    }
    return set;
  }, [result]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const target = e.currentTarget;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      const next = source.slice(0, start) + '    ' + source.slice(end);
      onChangeSource(next);
      requestAnimationFrame(() => {
        target.selectionStart = target.selectionEnd = start + 4;
      });
    }
  };

  return (
    <div className="flex-1 min-h-0 flex flex-col lg:flex-row bg-[#F8FAFC]">
      {/* Left Sidebar: Examples & Compiler Passes */}
      <aside className="w-full lg:w-[280px] shrink-0 bg-white border-b lg:border-b-0 lg:border-r border-slate-200 flex flex-col overflow-y-auto">
        <div className="p-4 border-b border-slate-200">
          <h2 className="text-xs font-semibold text-slate-500 tracking-tight mb-3">
            Example Programs
          </h2>
          <div className="space-y-1">
            {NOVA_EXAMPLES.map((ex) => {
              const selected = ex.id === activeExample.id;
              return (
                <button
                  key={ex.id}
                  type="button"
                  onClick={() => onSelectExample(ex)}
                  className={`w-full text-left px-3 py-2 rounded-lg focus-visible:outline-2 focus-visible:outline-blue-600 ${
                    selected
                      ? 'bg-slate-900 text-white'
                      : 'text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  <div className="text-sm font-medium truncate">{ex.title}</div>
                  <div
                    className={`text-xs font-mono truncate mt-0.5 flex items-center gap-1.5 ${
                      selected ? 'text-slate-300' : 'text-slate-500'
                    }`}
                  >
                    <span className="truncate">{ex.filename}</span>
                    <span aria-hidden="true">·</span>
                    <span className="shrink-0">{ex.category}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="p-4 border-b border-slate-200">
          <h2 className="text-xs font-semibold text-slate-500 tracking-tight mb-2.5">
            Execution Engine
          </h2>
          <div className="grid grid-cols-2 gap-1 p-1 bg-slate-100 rounded-lg">
            <button
              type="button"
              onClick={() => onChangeEngine('vm')}
              className={`px-2.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                engine === 'vm'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Nova VM
            </button>
            <button
              type="button"
              onClick={() => onChangeEngine('interpreter')}
              className={`px-2.5 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                engine === 'interpreter'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Interpreter
            </button>
          </div>
        </div>

        <div className="p-4 border-b border-slate-200">
          <h2 className="text-xs font-semibold text-slate-500 tracking-tight mb-3">
            IR Optimization Passes
          </h2>
          <div className="space-y-2.5">
            {(
              [
                ['constantFolding', 'Constant Folding'],
                ['constantPropagation', 'Constant Propagation'],
                ['deadCodeElimination', 'Dead-Code Elimination'],
              ] as const
            ).map(([key, label]) => (
              <label key={key} className="flex items-center gap-2.5 text-sm text-slate-700 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={optimizations[key]}
                  onChange={(e) =>
                    onChangeOptimizations({ ...optimizations, [key]: e.target.checked })
                  }
                  className="w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-600"
                />
                <span>{label}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="p-4">
          <h2 className="text-xs font-semibold text-slate-500 tracking-tight mb-2">
            Native Build &amp; Link
          </h2>
          <p className="text-xs text-slate-500 leading-relaxed mb-2.5">
            Lower IR to LLVM, compile &amp; link with <span className="font-mono">libnovart.o</span>, and verify 3-way equivalence.
          </p>
          <div className="grid grid-cols-2 gap-1 p-1 bg-slate-100 rounded-lg mb-2.5">
            <button
              type="button"
              onClick={() => onChangeBuildProfile('debug')}
              className={`px-2 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                buildProfile === 'debug'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Debug (-O0)
            </button>
            <button
              type="button"
              onClick={() => onChangeBuildProfile('release')}
              className={`px-2 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                buildProfile === 'release'
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Release (-O3)
            </button>
          </div>
          <button
            type="button"
            onClick={onVerifyNative}
            disabled={isVerifyingNative}
            className="w-full inline-flex items-center justify-center gap-2 px-3 py-2 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 disabled:opacity-60 rounded-lg border border-slate-200 transition-colors whitespace-nowrap"
          >
            <Cpu className="w-3.5 h-3.5" aria-hidden="true" />
            {isVerifyingNative ? 'Building & Linking...' : 'Build & Verify Native'}
          </button>
        </div>
      </aside>

      {/* Center + Right Split */}
      <div className="flex-1 min-w-0 min-h-0 grid grid-cols-1 xl:grid-cols-12">
        {/* Source Editor Column */}
        <section className="xl:col-span-6 flex flex-col min-h-[420px] xl:min-h-0 border-b xl:border-b-0 xl:border-r border-slate-200 bg-white">
          <div className="h-11 px-4 border-b border-slate-200 flex items-center justify-between gap-3 shrink-0">
            <div className="flex items-center gap-2 min-w-0">
              <FileCode2 className="w-4 h-4 text-slate-500 shrink-0" aria-hidden="true" />
              <span className="font-mono text-xs font-medium text-slate-900 truncate">
                {activeExample.filename}
              </span>
              <span className="text-slate-300" aria-hidden="true">·</span>
              <span className="text-xs text-slate-500 tabular-nums whitespace-nowrap">
                {lines.length} lines
              </span>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <button
                type="button"
                onClick={onCheckOnly}
                disabled={isRunning}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 rounded-md transition-colors whitespace-nowrap"
              >
                <ShieldCheck className="w-3.5 h-3.5" aria-hidden="true" />
                Check
              </button>
              <button
                type="button"
                onClick={onFormat}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 rounded-md transition-colors whitespace-nowrap"
              >
                <Wand2 className="w-3.5 h-3.5" aria-hidden="true" />
                Format
              </button>
              <button
                type="button"
                onClick={onReset}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 rounded-md transition-colors whitespace-nowrap"
              >
                <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
                Reset
              </button>
            </div>
          </div>

          <div className="px-4 py-2 bg-slate-50 border-b border-slate-200 text-xs text-slate-600">
            {activeExample.description}
          </div>

          <div className="flex-1 min-h-0 flex overflow-hidden bg-white">
            {/* Line Numbers Gutter */}
            <div
              aria-hidden="true"
              className="w-12 py-4 pr-3 bg-slate-50 border-r border-slate-200 text-right font-mono text-xs leading-6 text-slate-400 select-none overflow-hidden tabular-nums shrink-0"
            >
              {lines.map((_, idx) => {
                const ln = idx + 1;
                const hasErr = errorLines.has(ln);
                return (
                  <div
                    key={ln}
                    className={hasErr ? 'text-red-600 font-semibold' : undefined}
                  >
                    {ln}
                  </div>
                );
              })}
            </div>

            <textarea
              aria-label="Nova source code editor"
              value={source}
              onChange={(e) => onChangeSource(e.target.value)}
              onKeyDown={handleKeyDown}
              spellCheck={false}
              autoCapitalize="off"
              autoCorrect="off"
              className="code-input flex-1 w-full h-full p-4 bg-white text-slate-900 font-mono text-[13px] leading-6 resize-none focus:outline-none overflow-auto whitespace-pre"
            />
          </div>
        </section>

        {/* Right Inspector Column */}
        <section className="xl:col-span-6 flex flex-col min-h-[420px] xl:min-h-0 bg-white">
          <div className="h-11 px-4 border-b border-slate-200 flex items-center justify-between gap-4 shrink-0">
            <div role="tablist" aria-label="Compiler stage inspector" className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
              {INSPECTOR_TABS.map((t) => {
                const active = inspectorTab === t.id;
                return (
                  <button
                    key={t.id}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    onClick={() => setInspectorTab(t.id)}
                    className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                      active
                        ? 'bg-white text-slate-900 shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {t.label}
                  </button>
                );
              })}
            </div>

            {result && (
              <div className="hidden sm:flex items-center gap-2 text-xs text-slate-500 tabular-nums whitespace-nowrap">
                <span>{result.tokens.length} tokens</span>
                <span aria-hidden="true">·</span>
                <span>{result.diagnostics.length} errors</span>
              </div>
            )}
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto p-5">
            {!result ? (
              <div className="h-full flex flex-col items-center justify-center text-center py-12">
                <p className="text-sm font-medium text-slate-700">Ready to compile</p>
                <p className="text-xs text-slate-500 mt-1">
                  Click Run Program or Check to inspect compilation stages.
                </p>
              </div>
            ) : inspectorTab === 'output' ? (
              <div className="space-y-6">
                {/* Summary Status Bar (No pills, clean semantic text + icon) */}
                <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-200">
                  <div className="flex items-center gap-2">
                    {result.ok ? (
                      <>
                        <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" aria-hidden="true" />
                        <span className="text-sm font-semibold text-slate-900">
                          {result.execution
                            ? result.engine === 'interpreter'
                              ? 'Executed on AST Interpreter'
                              : 'Executed on Nova VM'
                            : 'Type Check Passed'}
                        </span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="w-4 h-4 text-red-600 shrink-0" aria-hidden="true" />
                        <span className="text-sm font-semibold text-red-700">
                          Compilation Failed ({result.diagnostics.length}{' '}
                          {result.diagnostics.length === 1 ? 'error' : 'errors'})
                        </span>
                      </>
                    )}
                  </div>
                  {result.execution && (
                    <div className="flex items-center gap-2 text-xs font-mono text-slate-500 tabular-nums">
                      <span>{result.execution.durationMs} ms</span>
                      <span aria-hidden="true">·</span>
                      <span>
                        {result.execution.instructionsExecuted}{' '}
                        {result.engine === 'interpreter' ? 'AST steps' : 'VM ops'}
                      </span>
                      <span aria-hidden="true">·</span>
                      <span>{result.bytecodeMeta?.byteSize ?? 0} B .nvbc</span>
                    </div>
                  )}
                </div>

                {/* Diagnostics */}
                {result.formattedDiagnostics.length > 0 && (
                  <div className="space-y-3">
                    <h3 className="text-xs font-semibold text-red-700">Compiler Diagnostics</h3>
                    {result.formattedDiagnostics.map((diagText, i) => (
                      <pre
                        key={i}
                        className="p-4 bg-red-50/70 border border-red-200 rounded-lg font-mono text-xs text-red-950 whitespace-pre-wrap break-words leading-relaxed"
                      >
                        {diagText}
                      </pre>
                    ))}
                  </div>
                )}

                {/* Standard Output */}
                {result.execution && (
                  <div>
                    <h3 className="text-xs font-semibold text-slate-500 mb-2">Standard Output</h3>
                    <div
                      data-testid="vm-stdout"
                      className="p-4 bg-slate-950 text-slate-100 rounded-lg font-mono text-[13px] leading-relaxed min-h-[96px] overflow-x-auto"
                    >
                      {result.execution.stdout.length > 0 ? (
                        result.execution.stdout.map((line, idx) => (
                          <div key={idx} className="whitespace-pre-wrap break-words">
                            {line}
                          </div>
                        ))
                      ) : (
                        <span className="text-slate-400">(program produced no output)</span>
                      )}
                    </div>
                  </div>
                )}

                {/* Runtime & Mark-and-Sweep GC Metrics */}
                {result.execution && (
                  <div className="pt-2 border-t border-slate-200">
                    <h3 className="text-xs font-semibold text-slate-500 mb-3">
                      Runtime &amp; Garbage Collector
                    </h3>
                    <dl className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                      <div>
                        <dt className="text-xs text-slate-500">Heap Allocated</dt>
                        <dd className="text-base font-mono font-semibold text-slate-900 tabular-nums mt-0.5">
                          {result.execution.gcStats.totalAllocated} objs
                        </dd>
                      </div>
                      <div>
                        <dt className="text-xs text-slate-500">GC Swept</dt>
                        <dd className="text-base font-mono font-semibold text-slate-900 tabular-nums mt-0.5">
                          {result.execution.gcStats.totalFreed} objs
                        </dd>
                      </div>
                      <div>
                        <dt className="text-xs text-slate-500">GC Cycles</dt>
                        <dd className="text-base font-mono font-semibold text-slate-900 tabular-nums mt-0.5">
                          {result.execution.gcStats.collectionsRun}
                        </dd>
                      </div>
                      <div>
                        <dt className="text-xs text-slate-500">IR Opt Delta</dt>
                        <dd className="text-base font-mono font-semibold text-slate-900 tabular-nums mt-0.5">
                          {result.optimizationStats
                            ? `${result.optimizationStats.instructionsBefore} -> ${result.optimizationStats.instructionsAfter}`
                            : '—'}
                        </dd>
                      </div>
                    </dl>
                  </div>
                )}

                {/* Native Build, Link & 3-Way Equivalence Result */}
                {nativeVerify && (
                  <div className="pt-4 border-t border-slate-200 space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <h3 className="text-xs font-semibold text-slate-500">
                        3-Way Execution Equivalence (Interpreter == VM == Native)
                      </h3>
                      {nativeVerify.supported && nativeVerify.equivalent && (
                        <span className="text-xs font-semibold text-emerald-700">
                          3-Way Equivalence Verified
                        </span>
                      )}
                      {nativeVerify.supported && nativeVerify.equivalent === false && (
                        <span className="text-xs font-semibold text-red-700">
                          Output Mismatch Detected
                        </span>
                      )}
                    </div>

                    {!nativeVerify.supported ? (
                      <p className="text-xs text-slate-600 bg-slate-50 border border-slate-200 rounded-lg p-3">
                        {nativeVerify.reason}
                      </p>
                    ) : (
                      <div className="space-y-3">
                        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-3 bg-slate-50 border border-slate-200 rounded-lg">
                          <div>
                            <dt className="text-[11px] text-slate-500">Profile / Backend</dt>
                            <dd className="text-xs font-mono font-semibold text-slate-900 mt-0.5">
                              {nativeVerify.profile ?? buildProfile} · {nativeVerify.backendUsed ?? 'gcc-c99'}
                            </dd>
                          </div>
                          <div>
                            <dt className="text-[11px] text-slate-500">Objects Linked</dt>
                            <dd className="text-xs font-mono font-semibold text-slate-900 tabular-nums mt-0.5">
                              {((nativeVerify.programObjBytes ?? 0) / 1024).toFixed(1)}K + {((nativeVerify.runtimeObjBytes ?? 0) / 1024).toFixed(1)}K
                            </dd>
                          </div>
                          <div>
                            <dt className="text-[11px] text-slate-500">
                              ELF Binary {nativeVerify.stripped ? '(stripped)' : '(debug)'}
                            </dt>
                            <dd className="text-xs font-mono font-semibold text-slate-900 tabular-nums mt-0.5">
                              {nativeVerify.stripped && nativeVerify.preStripBytes
                                ? `${nativeVerify.preStripBytes.toLocaleString()} -> ${nativeVerify.binaryBytes?.toLocaleString()} B`
                                : `${nativeVerify.binaryBytes?.toLocaleString() ?? 0} B`}
                            </dd>
                          </div>
                          <div>
                            <dt className="text-[11px] text-slate-500">Compile / Link / Run</dt>
                            <dd className="text-xs font-mono font-semibold text-slate-900 tabular-nums mt-0.5">
                              {nativeVerify.compileMs} / {nativeVerify.linkMs ?? 0} / {nativeVerify.execMs} ms
                            </dd>
                          </div>
                        </dl>
                        <pre
                          data-testid="native-stdout"
                          className="p-3 bg-slate-950 text-slate-100 rounded-lg font-mono text-xs overflow-x-auto"
                        >
                          {(nativeVerify.nativeStdout ?? []).join('\n') || '(no output)'}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : inspectorTab === 'tokens' ? (
              <div>
                <h3 className="text-xs font-semibold text-slate-500 mb-3">
                  Lexer Token Stream ({result.tokens.length} tokens)
                </h3>
                <div className="border border-slate-200 rounded-lg overflow-hidden">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="bg-slate-50 border-b border-slate-200 text-xs text-slate-500">
                        <th className="py-2 px-3 font-medium">#</th>
                        <th className="py-2 px-3 font-medium">Token Kind</th>
                        <th className="py-2 px-3 font-medium">Lexeme</th>
                        <th className="py-2 px-3 font-medium text-right">Line:Col</th>
                        <th className="py-2 px-3 font-medium text-right">Span</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 font-mono text-xs">
                      {result.tokens.map((tok, i) => (
                        <tr key={i} className="hover:bg-slate-50">
                          <td className="py-1.5 px-3 text-slate-400 tabular-nums">{i}</td>
                          <td className="py-1.5 px-3 text-blue-700 font-medium">{tok.kind}</td>
                          <td className="py-1.5 px-3 text-slate-900">
                            {tok.lexeme === '' ? 'EOF' : JSON.stringify(tok.lexeme)}
                          </td>
                          <td className="py-1.5 px-3 text-slate-500 text-right tabular-nums">
                            {tok.span.line}:{tok.span.column}
                          </td>
                          <td className="py-1.5 px-3 text-slate-400 text-right tabular-nums">
                            {tok.span.start}..{tok.span.end}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : inspectorTab === 'ast' ? (
              <div>
                <h3 className="text-xs font-semibold text-slate-500 mb-3">
                  Abstract Syntax Tree ({result.ast.items.length} top-level declarations)
                </h3>
                <pre className="p-4 bg-slate-50 border border-slate-200 rounded-lg font-mono text-xs text-slate-800 overflow-x-auto leading-relaxed">
                  {JSON.stringify(result.ast, null, 2)}
                </pre>
              </div>
            ) : inspectorTab === 'ir' ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-xs font-semibold text-slate-500">
                    Nova Intermediate Representation
                  </h3>
                  {result.optimizationStats && (
                    <div className="text-xs font-mono text-slate-500 tabular-nums flex items-center gap-2">
                      <span>Folded: {result.optimizationStats.constantsFolded}</span>
                      <span aria-hidden="true">·</span>
                      <span>Propagated: {result.optimizationStats.constantsPropagated}</span>
                      <span aria-hidden="true">·</span>
                      <span>DCE: {result.optimizationStats.deadInstructionsRemoved}</span>
                    </div>
                  )}
                </div>
                {result.irText ? (
                  <pre className="p-4 bg-slate-950 text-slate-100 rounded-lg font-mono text-xs overflow-x-auto leading-relaxed">
                    {result.irText}
                  </pre>
                ) : (
                  <p className="text-xs text-slate-500">
                    Fix syntax or semantic errors to generate Nova IR.
                  </p>
                )}
              </div>
            ) : inspectorTab === 'llvm' ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-xs font-semibold text-slate-500">
                    LLVM IR (Opaque Pointers · libnovart ABI)
                  </h3>
                  {result.llvmCodegen?.targetTriple && (
                    <span className="text-xs font-mono text-slate-500">
                      target: {result.llvmCodegen.targetTriple}
                    </span>
                  )}
                </div>

                {result.llvmCodegen?.supported && result.llvmCodegen.validation && (
                  <div className="flex flex-wrap items-center justify-between gap-3 px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-lg text-xs">
                    <div className="flex items-center gap-2">
                      {result.llvmCodegen.validation.ok ? (
                        <>
                          <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0" />
                          <span className="font-semibold text-emerald-700">LLVM IR Verified</span>
                        </>
                      ) : (
                        <>
                          <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
                          <span className="font-semibold text-red-700">LLVM Verification Failed</span>
                        </>
                      )}
                    </div>
                    <div className="font-mono text-slate-600 tabular-nums flex flex-wrap items-center gap-2">
                      <span>{result.llvmCodegen.validation.functionsChecked} fns</span>
                      <span aria-hidden="true">·</span>
                      <span>{result.llvmCodegen.validation.blocksChecked} blocks</span>
                      <span aria-hidden="true">·</span>
                      <span>{result.llvmCodegen.validation.instructionsChecked} insts</span>
                    </div>
                  </div>
                )}

                {result.llvmCodegen?.supported && result.llvmCodegen.llvmIR ? (
                  <pre
                    data-testid="llvm-ir-code"
                    className="p-4 bg-slate-950 text-slate-100 rounded-lg font-mono text-xs overflow-x-auto leading-relaxed"
                  >
                    {result.llvmCodegen.llvmIR}
                  </pre>
                ) : result.llvmCodegen && !result.llvmCodegen.supported ? (
                  <p className="text-xs text-slate-600 bg-slate-50 border border-slate-200 rounded-lg p-3.5">
                    {result.llvmCodegen.reason}
                  </p>
                ) : (
                  <p className="text-xs text-slate-500">
                    Fix compilation errors to lower Nova IR to LLVM IR.
                  </p>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-xs font-semibold text-slate-500">
                    Nova VM Bytecode &amp; Verifier
                  </h3>
                  {result.bytecodeMeta && (
                    <div className="text-xs font-mono text-slate-500 tabular-nums flex items-center gap-2">
                      <span>Magic: {result.bytecodeMeta.magic}</span>
                      <span aria-hidden="true">·</span>
                      <span>{result.bytecodeMeta.constantCount} constants</span>
                      <span aria-hidden="true">·</span>
                      <span>{result.bytecodeMeta.byteSize} bytes .nvbc</span>
                    </div>
                  )}
                </div>

                {result.bytecodeVerification && (
                  <div className="flex flex-wrap items-center justify-between gap-3 px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-lg text-xs">
                    <div className="flex items-center gap-2">
                      {result.bytecodeVerification.ok ? (
                        <>
                          <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0" />
                          <span className="font-semibold text-emerald-700">Bytecode Verified</span>
                        </>
                      ) : (
                        <>
                          <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
                          <span className="font-semibold text-red-700">Verification Failed</span>
                        </>
                      )}
                    </div>
                    <div className="font-mono text-slate-600 tabular-nums flex flex-wrap items-center gap-2">
                      <span>{result.bytecodeVerification.functionsVerified} fns</span>
                      <span aria-hidden="true">·</span>
                      <span>{result.bytecodeVerification.instructionsVerified} insts</span>
                      <span aria-hidden="true">·</span>
                      <span>max stack {result.bytecodeVerification.maxStackDepth}</span>
                      <span aria-hidden="true">·</span>
                      <span>{result.bytecodeVerification.checksRun.length} passes</span>
                    </div>
                  </div>
                )}

                {result.disassembly ? (
                  <>
                    <pre className="p-4 bg-slate-950 text-slate-100 rounded-lg font-mono text-xs overflow-x-auto leading-relaxed">
                      {result.disassembly}
                    </pre>
                    {result.bytecodeHex && (
                      <div className="space-y-2 pt-2 border-t border-slate-200">
                        <h4 className="text-xs font-semibold text-slate-500">
                          Binary .nvbc Wire Format (Hex Preview)
                        </h4>
                        <pre className="p-3.5 bg-slate-50 border border-slate-200 text-slate-700 rounded-lg font-mono text-[11px] overflow-x-auto leading-relaxed">
                          {result.bytecodeHex}
                        </pre>
                      </div>
                    )}
                  </>
                ) : (
                  <p className="text-xs text-slate-500">
                    Fix compilation errors to emit Nova VM bytecode.
                  </p>
                )}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
