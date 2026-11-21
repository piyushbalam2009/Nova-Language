import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import express from 'express';
import { createServer as createViteServer } from 'vite';
import { compileAndRun, DEFAULT_OPT_OPTIONS, formatNovaSource } from './src/nova/index';
import {
  buildNativeExecutable,
  executeNativeBinary,
  sanitizeModuleName,
} from './src/nova/native-builder';

const PORT = 3000;
const ROOT_DIR = process.cwd();
const NOVA_DIR = path.resolve(ROOT_DIR, 'nova');
const MAX_SOURCE_CHARS = 64 * 1024;
const MAX_CONCURRENT_NATIVE_BUILDS = 2;
let activeNativeBuilds = 0;

interface WorkspaceFileEntry {
  path: string;
  sizeBytes: number;
  lines: number;
  category: 'manifest' | 'crate' | 'example' | 'docs';
}

function listNovaFiles(dir: string, base: string, acc: WorkspaceFileEntry[]): void {
  if (!fs.existsSync(dir)) return;
  const entries = fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    if (entry.name === 'target' || entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    const rel = path.relative(base, full).replace(/\\/g, '/');
    if (entry.isDirectory()) {
      listNovaFiles(full, base, acc);
    } else {
      const stat = fs.statSync(full);
      const text = fs.readFileSync(full, 'utf8');
      const lines = text.split('\n').length;
      let category: WorkspaceFileEntry['category'] = 'manifest';
      if (rel.startsWith('crates/') || rel.startsWith('runtime/')) category = 'crate';
      else if (rel.startsWith('examples/')) category = 'example';
      else if (rel.startsWith('docs/')) category = 'docs';
      acc.push({ path: rel, sizeBytes: stat.size, lines, category });
    }
  }
}

function probeBinary(bin: string): { available: boolean; version: string } {
  try {
    const out = execFileSync(bin, ['--version'], { encoding: 'utf8', timeout: 2000 });
    return { available: true, version: out.split('\n')[0].trim() };
  } catch {
    return { available: false, version: 'Not installed in container' };
  }
}

async function startServer(): Promise<void> {
  const app = express();
  app.use(express.json({ limit: '512kb' }));

  app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', novaVersion: '0.1.0' });
  });

  app.get('/api/toolchain', (_req, res) => {
    res.json({
      novaVersion: '0.1.0',
      node: { available: true, version: process.version },
      gcc: probeBinary('gcc'),
      rustc: probeBinary('rustc'),
      cargo: probeBinary('cargo'),
      llvm: probeBinary('llvm-config'),
    });
  });

  app.post('/api/compile', (req, res) => {
    const source = typeof req.body?.source === 'string' ? req.body.source : '';
    if (source.length > MAX_SOURCE_CHARS) {
      res.status(413).json({
        ok: false,
        error: `Source exceeds maximum allowed length (${MAX_SOURCE_CHARS} characters).`,
      });
      return;
    }
    const filename = sanitizeModuleName(
      typeof req.body?.filename === 'string' ? req.body.filename : 'main.nova'
    );
    const execute = req.body?.execute !== false;
    const engine = req.body?.engine === 'interpreter' ? 'interpreter' : 'vm';
    const optimizations = {
      constantFolding: req.body?.optimizations?.constantFolding ?? DEFAULT_OPT_OPTIONS.constantFolding,
      constantPropagation: req.body?.optimizations?.constantPropagation ?? DEFAULT_OPT_OPTIONS.constantPropagation,
      deadCodeElimination: req.body?.optimizations?.deadCodeElimination ?? DEFAULT_OPT_OPTIONS.deadCodeElimination,
    };

    const result = compileAndRun(source, { filename, execute, engine, optimizations });
    res.json({
      ok: result.ok,
      engine: result.engine,
      tokens: result.tokens,
      ast: result.ast,
      irText: result.irText ?? '',
      optimizationStats: result.optimizationStats ?? null,
      disassembly: result.disassembly ?? '',
      bytecodeHex: result.bytecodeHex ?? '',
      bytecodeMeta: result.bytecode
        ? {
            magic: result.bytecode.magic,
            version: result.bytecode.version,
            byteSize: result.bytecode.byteSize,
            constantCount: result.bytecode.constants.length,
            functionCount: result.bytecode.functions.length,
          }
        : null,
      bytecodeVerification: result.verification ?? null,
      llvmCodegen: result.llvmCodegen ?? null,
      nativeCodegen: result.nativeCodegen ?? null,
      execution: result.execution ?? null,
      diagnostics: result.diagnostics,
      formattedDiagnostics: result.formattedDiagnostics,
      imports: result.semantic?.imports ?? [],
      symbolCount: result.semantic?.symbols.length ?? 0,
    });
  });

  app.post('/api/format', (req, res) => {
    const source = typeof req.body?.source === 'string' ? req.body.source : '';
    if (source.length > MAX_SOURCE_CHARS) {
      res.status(413).json({
        ok: false,
        error: `Source exceeds maximum allowed length (${MAX_SOURCE_CHARS} characters).`,
      });
      return;
    }
    res.json(formatNovaSource(source));
  });

  app.post('/api/native-verify', (req, res) => {
    const source = typeof req.body?.source === 'string' ? req.body.source : '';
    if (source.length > MAX_SOURCE_CHARS) {
      res.status(413).json({
        ok: false,
        supported: false,
        reason: `Source exceeds maximum allowed length (${MAX_SOURCE_CHARS} characters).`,
      });
      return;
    }
    if (activeNativeBuilds >= MAX_CONCURRENT_NATIVE_BUILDS) {
      res.status(429).json({
        ok: false,
        supported: true,
        reason: 'Native build runner is busy. Please retry shortly.',
      });
      return;
    }
    const filename = sanitizeModuleName(
      typeof req.body?.filename === 'string' ? req.body.filename : 'main.nova'
    );
    const profile = req.body?.profile === 'debug' ? 'debug' : 'release';
    const optimizations = {
      constantFolding: req.body?.optimizations?.constantFolding ?? true,
      constantPropagation: req.body?.optimizations?.constantPropagation ?? true,
      deadCodeElimination: req.body?.optimizations?.deadCodeElimination ?? true,
    };

    const vmResult = compileAndRun(source, { filename, execute: true, engine: 'vm', optimizations });
    const interpResult = compileAndRun(source, { filename, execute: true, engine: 'interpreter', optimizations });

    if (!vmResult.ok || !vmResult.irOptimized) {
      res.json({
        ok: false,
        supported: false,
        reason: 'Source has compile or runtime errors.',
        formattedDiagnostics: vmResult.formattedDiagnostics,
      });
      return;
    }

    activeNativeBuilds += 1;
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nova-native-verify-'));
    const outBin = path.join(tmpDir, 'nova_bin');
    try {
      const build = buildNativeExecutable(vmResult.irOptimized, {
        profile,
        backend: 'auto',
        outputPath: outBin,
        moduleName: filename,
      });

      if (!build.supported) {
        res.json({
          ok: true,
          supported: false,
          reason: build.reason ?? 'Native backend unsupported for this program.',
          vmStdout: vmResult.execution?.stdout ?? [],
          interpreterStdout: interpResult.execution?.stdout ?? [],
        });
        return;
      }

      if (!build.ok || !build.outputPath) {
        res.status(500).json({
          ok: false,
          supported: true,
          reason: build.reason ?? 'Native build/link failed.',
        });
        return;
      }

      const execRes = executeNativeBinary(build.outputPath, 4000);
      const vmStdout = vmResult.execution?.stdout ?? [];
      const interpreterStdout = interpResult.execution?.stdout ?? [];
      const nativeStdout = execRes.stdout;

      const eqVm =
        execRes.exitCode === 0 &&
        vmStdout.length === nativeStdout.length &&
        vmStdout.every((line, idx) => line === nativeStdout[idx]);
      const eqInterp =
        execRes.exitCode === 0 &&
        interpreterStdout.length === nativeStdout.length &&
        interpreterStdout.every((line, idx) => line === nativeStdout[idx]);

      res.json({
        ok: true,
        supported: true,
        equivalent: eqVm && eqInterp,
        profile: build.profile,
        backendUsed: build.backendUsed,
        targetTriple: build.targetTriple,
        stripped: build.stripped,
        programObjBytes: build.programObjBytes,
        runtimeObjBytes: build.runtimeObjBytes,
        preStripBytes: build.preStripBytes,
        binaryBytes: build.binaryBytes,
        compileMs: build.compileMs,
        linkMs: build.linkMs,
        execMs: execRes.execMs,
        vmStdout,
        interpreterStdout,
        nativeStdout,
        llvmIR: build.llvmIR,
        cSource: build.cSource,
      });
    } catch (err) {
      res.status(500).json({
        ok: false,
        supported: true,
        reason: err instanceof Error ? err.message : String(err),
      });
    } finally {
      activeNativeBuilds = Math.max(0, activeNativeBuilds - 1);
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  app.get('/api/workspace/files', (_req, res) => {
    const files: WorkspaceFileEntry[] = [];
    listNovaFiles(NOVA_DIR, NOVA_DIR, files);
    res.json({ root: 'nova', files });
  });

  app.get('/api/workspace/file', (req, res) => {
    const relPath = typeof req.query.path === 'string' ? req.query.path : '';
    if (!relPath || relPath.includes('..') || path.isAbsolute(relPath)) {
      res.status(400).json({ error: 'Invalid workspace file path' });
      return;
    }
    const target = path.resolve(NOVA_DIR, relPath);
    if (!target.startsWith(NOVA_DIR + path.sep)) {
      res.status(403).json({ error: 'Access outside nova workspace denied' });
      return;
    }
    if (!fs.existsSync(target) || fs.statSync(target).isDirectory()) {
      res.status(404).json({ error: 'File not found' });
      return;
    }
    res.json({
      path: relPath,
      content: fs.readFileSync(target, 'utf8'),
    });
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(ROOT_DIR, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Nova Workbench server listening on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start Nova server:', err);
  process.exit(1);
});
