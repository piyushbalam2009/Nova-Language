const MILESTONES = [
  {
    version: 'Nova 0.1',
    scope: 'Workspace, Lexer, Pratt Parser, AST, Semantic Analyzer, CLI, Hello World',
    runtimeStatus: 'Implemented & Tested',
    rustStatus: 'Written on disk (awaiting cargo)',
  },
  {
    version: 'Nova 0.2',
    scope: 'Functions, Types, Mutability (let/mut/const), Structs, Loops, Modules, Result<T, E>',
    runtimeStatus: 'Implemented & Tested',
    rustStatus: 'Core subset written on disk',
  },
  {
    version: 'Nova 0.3',
    scope: 'Nova IR, Constant Folding/Propagation/DCE, Bytecode Compiler, Nova VM, Mark-and-Sweep GC',
    runtimeStatus: 'Implemented & Tested',
    rustStatus: 'Written on disk (awaiting cargo)',
  },
  {
    version: 'Nova 0.4',
    scope: 'Native Compilation & Execution Equivalence',
    runtimeStatus: 'Implemented via C99 + GCC (-O2)',
    rustStatus: 'LLVM inkwell backend planned',
  },
  {
    version: 'Nova 0.5',
    scope: 'Standard Library (io, math, string, array, json, time)',
    runtimeStatus: 'Implemented & Tested (core modules)',
    rustStatus: 'fs, http, concurrent planned',
  },
  {
    version: 'Nova 0.6',
    scope: 'Nova AI Engine: CPU Tensors, @ Matmul, Reverse-Mode Autograd, Neural Net Training',
    runtimeStatus: 'Implemented & Tested',
    rustStatus: 'nova-ai tensor crate written',
  },
  {
    version: 'Nova 0.7 – 1.0',
    scope: 'CUDA GPU Backend, Package Manager (nova.toml/nova.lock), LSP, VS Code Extension',
    runtimeStatus: 'Planned',
    rustStatus: 'Planned',
  },
];

export function SpecView() {
  return (
    <div className="flex-1 overflow-y-auto bg-[#F8FAFC] p-6 lg:p-8">
      <div className="max-w-5xl mx-auto space-y-8">
        <div>
          <h1 className="font-display text-2xl font-bold text-slate-900 tracking-tight">
            Nova Language Specification &amp; Status
          </h1>
          <p className="text-sm text-slate-600 mt-1 max-w-2xl leading-relaxed">
            Version 0.1.0 grammar, static typing semantics, hybrid execution contract, and transparent implementation status. No unimplemented feature is reported as working.
          </p>
        </div>

        {/* Implementation Matrix */}
        <section className="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-200">
            <h2 className="font-display text-base font-semibold text-slate-900">
              Milestone Implementation Matrix
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-xs text-slate-500">
                  <th className="py-3 px-6 font-medium">Milestone</th>
                  <th className="py-3 px-6 font-medium">Capabilities</th>
                  <th className="py-3 px-6 font-medium">Executable Runtime / CLI</th>
                  <th className="py-3 px-6 font-medium">Rust Cargo Workspace</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm">
                {MILESTONES.map((m) => (
                  <tr key={m.version} className="hover:bg-slate-50/70">
                    <td className="py-3.5 px-6 font-mono text-xs font-semibold text-slate-900 whitespace-nowrap">
                      {m.version}
                    </td>
                    <td className="py-3.5 px-6 text-xs text-slate-700 leading-relaxed">
                      {m.scope}
                    </td>
                    <td className="py-3.5 px-6 text-xs font-medium whitespace-nowrap">
                      <span
                        className={
                          m.runtimeStatus.startsWith('Implemented')
                            ? 'text-emerald-700'
                            : 'text-slate-500'
                        }
                      >
                        {m.runtimeStatus}
                      </span>
                    </td>
                    <td className="py-3.5 px-6 text-xs text-slate-600 whitespace-nowrap">
                      {m.rustStatus}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Language Grammar Quick Reference */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
            <h2 className="font-display text-base font-semibold text-slate-900">
              01. Bindings &amp; Static Types
            </h2>
            <p className="text-xs text-slate-600 leading-relaxed">
              Immutable bindings use <code className="font-mono text-slate-900">let</code>, mutable bindings require <code className="font-mono text-slate-900">mut</code>, and compile-time constants use <code className="font-mono text-slate-900">const</code>. Types are inferred or explicitly annotated.
            </p>
            <pre className="p-3.5 bg-slate-950 text-slate-100 rounded-lg font-mono text-xs overflow-x-auto leading-relaxed">{`let name = "Nova"
let version: Int = 1
mut counter = 0
counter += 1
const MAX_TOKENS = 4096`}</pre>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
            <h2 className="font-display text-base font-semibold text-slate-900">
              02. Structs &amp; Explicit Error Handling
            </h2>
            <p className="text-xs text-slate-600 leading-relaxed">
              Structures validate every declared field and type at compile time. Fallible functions return <code className="font-mono text-slate-900">Result&lt;T, E&gt;</code> with <code className="font-mono text-slate-900">Ok(...)</code> and <code className="font-mono text-slate-900">Error(...)</code>.
            </p>
            <pre className="p-3.5 bg-slate-950 text-slate-100 rounded-lg font-mono text-xs overflow-x-auto leading-relaxed">{`struct User { name: String, age: Int }

fn divide(a: Float, b: Float) -> Result<Float, String> {
    if b == 0.0 { return Error("Division by zero") }
    return Ok(a / b)
}`}</pre>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
            <h2 className="font-display text-base font-semibold text-slate-900">
              03. Native Tensor &amp; AI Operations
            </h2>
            <p className="text-xs text-slate-600 leading-relaxed">
              The decoupled Nova AI engine provides multi-dimensional CPU tensors, broadcasting, matrix multiplication via <code className="font-mono text-slate-900">@</code>, reverse-mode autograd, and neural network training.
            </p>
            <pre className="p-3.5 bg-slate-950 text-slate-100 rounded-lg font-mono text-xs overflow-x-auto leading-relaxed">{`use nova.ai
let a = tensor([[1.0, 2.0], [3.0, 4.0]])
let b = tensor([[5.0, 6.0], [7.0, 8.0]])
let c = a @ b`}</pre>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
            <h2 className="font-display text-base font-semibold text-slate-900">
              04. Executable CLI Commands
            </h2>
            <p className="text-xs text-slate-600 leading-relaxed">
              Invoke the live compiler, VM, IR emitter, bytecode disassembler, formatter, or native GCC builder directly from the terminal.
            </p>
            <pre className="p-3.5 bg-slate-950 text-slate-100 rounded-lg font-mono text-xs overflow-x-auto leading-relaxed">{`npm run nova -- run nova/examples/hello.nova
npm run nova -- check nova/examples/hello.nova
npm run nova -- ir nova/examples/functions.nova
npm run nova -- build nova/examples/hello.nova --release`}</pre>
          </div>
        </section>
      </div>
    </div>
  );
}
