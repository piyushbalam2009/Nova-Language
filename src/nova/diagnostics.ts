export interface Span {
  start: number;
  end: number;
  line: number;
  column: number;
}

export type DiagnosticStage = 'lexer' | 'parser' | 'semantic' | 'ir' | 'bytecode' | 'vm' | 'codegen';

export type DiagnosticSeverity = 'error' | 'warning';

export interface Diagnostic {
  stage: DiagnosticStage;
  severity: DiagnosticSeverity;
  code: string;
  message: string;
  span: Span;
  hint?: string;
}

export function formatDiagnostic(diag: Diagnostic, source: string, filename = '<input>'): string {
  const lines = source.split('\n');
  const lineText = lines[diag.span.line - 1] ?? '';
  const lineNumStr = String(diag.span.line);
  const gutter = ' '.repeat(lineNumStr.length);
  const caretOffset = Math.max(0, diag.span.column - 1);
  const tokenLen = Math.max(1, Math.min(diag.span.end - diag.span.start, Math.max(1, lineText.length - caretOffset)));
  const carets = '^'.repeat(tokenLen);

  const out: string[] = [
    `${diag.severity}[${diag.code}]: ${diag.message}`,
    `  --> ${filename}:${diag.span.line}:${diag.span.column}`,
    `  ${gutter} |`,
    `  ${lineNumStr} | ${lineText}`,
    `  ${gutter} | ${' '.repeat(caretOffset)}${carets}`,
  ];
  if (diag.hint) {
    out.push(`  ${gutter} = help: ${diag.hint}`);
  }
  return out.join('\n');
}
