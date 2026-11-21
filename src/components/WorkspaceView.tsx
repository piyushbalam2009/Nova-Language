import { useEffect, useState } from 'react';
import { Check, Copy, FileText, FolderTree, Terminal } from 'lucide-react';
import { ToolchainStatus, WorkspaceFileEntry } from '../types/api';

const CARGO_VERIFY_CMD = `cd nova
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p nova-cli -- run examples/hello.nova`;

export function WorkspaceView() {
  const [files, setFiles] = useState<WorkspaceFileEntry[]>([]);
  const [selectedPath, setSelectedPath] = useState<string>('Cargo.toml');
  const [fileContent, setFileContent] = useState<string>('');
  const [loadingFile, setLoadingFile] = useState<boolean>(false);
  const [toolchain, setToolchain] = useState<ToolchainStatus | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  useEffect(() => {
    fetch('/api/workspace/files')
      .then((r) => r.json())
      .then((data) => {
        const list = (data.files ?? []) as WorkspaceFileEntry[];
        setFiles(list);
        if (list.length > 0 && !list.some((f) => f.path === selectedPath)) {
          setSelectedPath(list[0].path);
        }
      })
      .catch(() => setFiles([]));

    fetch('/api/toolchain')
      .then((r) => r.json())
      .then((data) => setToolchain(data as ToolchainStatus))
      .catch(() => setToolchain(null));
  }, []);

  useEffect(() => {
    if (!selectedPath) return;
    setLoadingFile(true);
    fetch(`/api/workspace/file?path=${encodeURIComponent(selectedPath)}`)
      .then((r) => r.json())
      .then((data) => setFileContent(typeof data.content === 'string' ? data.content : ''))
      .catch(() => setFileContent('// Failed to load file'))
      .finally(() => setLoadingFile(false));
  }, [selectedPath]);

  const copyVerifyCmd = () => {
    navigator.clipboard?.writeText(CARGO_VERIFY_CMD).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const totalLines = files.reduce((sum, f) => sum + f.lines, 0);

  return (
    <div className="flex-1 min-h-0 flex flex-col lg:flex-row bg-[#F8FAFC]">
      {/* Left File Explorer */}
      <aside className="w-full lg:w-[320px] shrink-0 bg-white border-b lg:border-b-0 lg:border-r border-slate-200 flex flex-col overflow-y-auto">
        <div className="p-4 border-b border-slate-200">
          <div className="flex items-center gap-2 text-slate-900 font-display text-sm font-semibold">
            <FolderTree className="w-4 h-4 text-slate-500" aria-hidden="true" />
            <span>nova/ Rust Workspace</span>
          </div>
          <p className="text-xs text-slate-500 mt-1 tabular-nums">
            {files.length} files · {totalLines.toLocaleString()} lines on disk
          </p>
        </div>

        <div className="p-3 space-y-4 flex-1 overflow-y-auto">
          {(
            [
              ['manifest', 'Workspace Root'],
              ['crate', 'Rust Crates (crates/)'],
              ['example', 'Examples (examples/)'],
              ['docs', 'Documentation (docs/)'],
            ] as const
          ).map(([cat, heading]) => {
            const group = files.filter((f) => f.category === cat);
            if (group.length === 0) return null;
            return (
              <div key={cat}>
                <div className="px-2.5 mb-1.5 text-[11px] font-semibold text-slate-400">
                  {heading}
                </div>
                <div className="space-y-0.5">
                  {group.map((file) => {
                    const active = file.path === selectedPath;
                    const display =
                      cat === 'crate'
                        ? file.path.replace(/^crates\//, '')
                        : cat === 'example'
                        ? file.path.replace(/^examples\//, '')
                        : cat === 'docs'
                        ? file.path.replace(/^docs\//, '')
                        : file.path;
                    return (
                      <button
                        key={file.path}
                        type="button"
                        title={`nova/${file.path}`}
                        onClick={() => setSelectedPath(file.path)}
                        className={`w-full text-left px-2.5 py-1.5 rounded-md flex items-center justify-between gap-2 ${
                          active
                            ? 'bg-slate-900 text-white'
                            : 'text-slate-700 hover:bg-slate-100'
                        }`}
                      >
                        <span className="flex items-center gap-2 min-w-0">
                          <FileText className="w-3.5 h-3.5 shrink-0 opacity-70" aria-hidden="true" />
                          <span className="font-mono text-xs truncate">{display}</span>
                        </span>
                        <span
                          className={`font-mono text-[11px] tabular-nums shrink-0 ${
                            active ? 'text-slate-300' : 'text-slate-400'
                          }`}
                        >
                          {file.lines}L
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </aside>

      {/* Right Code Viewer + Host Toolchain Panel */}
      <div className="flex-1 min-w-0 min-h-0 flex flex-col overflow-y-auto">
        {/* Host Toolchain Status Strip */}
        <div className="bg-white border-b border-slate-200 p-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h1 className="font-display text-lg font-bold text-slate-900">
                Modular Rust Cargo Workspace &amp; Host Toolchains
              </h1>
              <p className="text-xs text-slate-600 mt-1 max-w-2xl leading-relaxed">
                All 9 Rust workspace crates (<span className="font-mono">nova-lexer</span> through <span className="font-mono">nova-cli</span>) are written on disk under <span className="font-mono">nova/</span>. Because this container does not include <span className="font-mono">rustc</span>/<span className="font-mono">cargo</span>, live execution here runs on the full TypeScript Nova compiler, VM, and GCC native backend.
              </p>
            </div>
            <button
              type="button"
              onClick={copyVerifyCmd}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg border border-slate-200 transition-colors whitespace-nowrap shrink-0"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? 'Copied Cargo Commands' : 'Copy Cargo Commands'}
            </button>
          </div>

          {toolchain && (
            <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-2 md:grid-cols-5 gap-4">
              {(
                [
                  ['Node.js', toolchain.node],
                  ['GCC Native', toolchain.gcc],
                  ['rustc', toolchain.rustc],
                  ['cargo', toolchain.cargo],
                  ['LLVM', toolchain.llvm],
                ] as const
              ).map(([label, info]) => (
                <div key={label}>
                  <div className="text-xs text-slate-500">{label}</div>
                  <div
                    className={`font-mono text-xs font-medium mt-0.5 truncate ${
                      info.available ? 'text-emerald-700' : 'text-slate-500'
                    }`}
                    title={info.version}
                  >
                    {info.available ? info.version : 'Not installed'}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Selected File Source */}
        <div className="flex-1 p-6 space-y-6">
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="h-10 px-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
              <span className="font-mono text-xs font-medium text-slate-800">
                nova/{selectedPath}
              </span>
              <span className="font-mono text-xs text-slate-500 tabular-nums">
                {fileContent.split('\n').length} lines
              </span>
            </div>
            <pre className="p-4 bg-slate-950 text-slate-100 font-mono text-xs leading-relaxed overflow-x-auto max-h-[480px]">
              {loadingFile ? 'Loading file...' : fileContent}
            </pre>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-5">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 mb-2">
              <Terminal className="w-4 h-4 text-slate-500" aria-hidden="true" />
              <span>Local Rust Verification Commands</span>
            </div>
            <pre className="p-3 bg-slate-50 border border-slate-200 rounded-lg font-mono text-xs text-slate-800 overflow-x-auto">
              {CARGO_VERIFY_CMD}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
