import { useCallback, useEffect, useState } from 'react';
import { AppView, Header } from './components/Header';
import { PipelineView } from './components/PipelineView';
import { SpecView } from './components/SpecView';
import { WorkbenchView } from './components/WorkbenchView';
import { WorkspaceView } from './components/WorkspaceView';
import { NOVA_EXAMPLES, NovaExample } from './data/examples';
import { DEFAULT_OPT_OPTIONS, ExecutionEngine } from './nova/index';
import { CompileApiResponse, NativeBuildProfile, NativeVerifyResponse, OptimizationOptions } from './types/api';

export default function App() {
  const [activeView, setActiveView] = useState<AppView>('workbench');
  const [activeExample, setActiveExample] = useState<NovaExample>(NOVA_EXAMPLES[0]);
  const [source, setSource] = useState<string>(NOVA_EXAMPLES[0].source);
  const [engine, setEngine] = useState<ExecutionEngine>('vm');
  const [buildProfile, setBuildProfile] = useState<NativeBuildProfile>('release');
  const [optimizations, setOptimizations] = useState<OptimizationOptions>(DEFAULT_OPT_OPTIONS);
  const [result, setResult] = useState<CompileApiResponse | null>(null);
  const [nativeVerify, setNativeVerify] = useState<NativeVerifyResponse | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isVerifyingNative, setIsVerifyingNative] = useState<boolean>(false);

  const runCompile = useCallback(
    async (
      code: string,
      filename: string,
      execute: boolean,
      opts: OptimizationOptions,
      execEngine: ExecutionEngine
    ) => {
      setIsRunning(true);
      try {
        const res = await fetch('/api/compile', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source: code,
            filename,
            execute,
            engine: execEngine,
            optimizations: opts,
          }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as CompileApiResponse;
        setResult(data);
      } catch (err) {
        console.error('Compile request failed:', err);
      } finally {
        setIsRunning(false);
      }
    },
    []
  );

  // Initial execution of hello.nova on first mount
  useEffect(() => {
    runCompile(NOVA_EXAMPLES[0].source, NOVA_EXAMPLES[0].filename, true, DEFAULT_OPT_OPTIONS, 'vm');
  }, [runCompile]);

  const handleSelectExample = (example: NovaExample) => {
    setActiveExample(example);
    setSource(example.source);
    setNativeVerify(null);
    runCompile(example.source, example.filename, true, optimizations, engine);
  };

  const handleChangeEngine = (next: ExecutionEngine) => {
    setEngine(next);
    runCompile(source, activeExample.filename, true, optimizations, next);
  };

  const handleChangeOptimizations = (next: OptimizationOptions) => {
    setOptimizations(next);
    runCompile(source, activeExample.filename, true, next, engine);
  };

  const handleRun = () => {
    setNativeVerify(null);
    runCompile(source, activeExample.filename, true, optimizations, engine);
  };

  const handleCheckOnly = () => {
    setNativeVerify(null);
    runCompile(source, activeExample.filename, false, optimizations, engine);
  };

  const handleFormat = async () => {
    try {
      const res = await fetch('/api/format', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source }),
      });
      const data = (await res.json()) as { ok: boolean; formatted: string };
      if (data.ok) {
        setSource(data.formatted);
      }
    } catch (err) {
      console.error('Format failed:', err);
    }
  };

  const handleReset = () => {
    setSource(activeExample.source);
    setNativeVerify(null);
    runCompile(activeExample.source, activeExample.filename, true, optimizations, engine);
  };

  const handleVerifyNative = async () => {
    setIsVerifyingNative(true);
    try {
      const res = await fetch('/api/native-verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source,
          filename: activeExample.filename,
          profile: buildProfile,
          optimizations,
        }),
      });
      const data = (await res.json()) as NativeVerifyResponse;
      setNativeVerify(data);
    } catch (err) {
      console.error('Native verify failed:', err);
    } finally {
      setIsVerifyingNative(false);
    }
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-[#F8FAFC] text-slate-900">
      <Header
        activeView={activeView}
        onSelectView={setActiveView}
        onRunProgram={handleRun}
        isRunning={isRunning}
      />

      <main className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {activeView === 'workbench' && (
          <WorkbenchView
            activeExample={activeExample}
            source={source}
            onChangeSource={setSource}
            onSelectExample={handleSelectExample}
            engine={engine}
            onChangeEngine={handleChangeEngine}
            buildProfile={buildProfile}
            onChangeBuildProfile={setBuildProfile}
            optimizations={optimizations}
            onChangeOptimizations={handleChangeOptimizations}
            result={result}
            nativeVerify={nativeVerify}
            isRunning={isRunning}
            isVerifyingNative={isVerifyingNative}
            onCheckOnly={handleCheckOnly}
            onFormat={handleFormat}
            onReset={handleReset}
            onVerifyNative={handleVerifyNative}
          />
        )}

        {activeView === 'pipeline' && (
          <PipelineView
            filename={activeExample.filename}
            result={result}
            nativeVerify={nativeVerify}
            isVerifyingNative={isVerifyingNative}
            onVerifyNative={handleVerifyNative}
          />
        )}

        {activeView === 'workspace' && <WorkspaceView />}

        {activeView === 'spec' && <SpecView />}
      </main>
    </div>
  );
}
