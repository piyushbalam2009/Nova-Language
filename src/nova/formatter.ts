import { Expr, formatTypeAnnotation, Program, Stmt } from './ast';

export function formatProgram(program: Program): string {
  const indent = (level: number) => '    '.repeat(level);

  const fmtExpr = (expr: Expr): string => {
    switch (expr.kind) {
      case 'IntLiteral': return String(expr.value);
      case 'FloatLiteral': return Number.isInteger(expr.value) ? `${expr.value}.0` : String(expr.value);
      case 'StringLiteral': return JSON.stringify(expr.value);
      case 'CharLiteral': return `'${expr.value}'`;
      case 'BoolLiteral': return expr.value ? 'true' : 'false';
      case 'NoneLiteral': return 'None';
      case 'IdentExpr': return expr.name;
      case 'UnaryExpr': return `${expr.op}${fmtExpr(expr.operand)}`;
      case 'BinaryExpr': return `${fmtExpr(expr.left)} ${expr.op} ${fmtExpr(expr.right)}`;
      case 'RangeExpr': return `${fmtExpr(expr.start)}..${fmtExpr(expr.end)}`;
      case 'CallExpr': return `${fmtExpr(expr.callee)}(${expr.args.map(fmtExpr).join(', ')})`;
      case 'FieldExpr': return `${fmtExpr(expr.object)}.${expr.field}`;
      case 'IndexExpr': return `${fmtExpr(expr.object)}[${fmtExpr(expr.index)}]`;
      case 'ArrayLiteral': return `[${expr.elements.map(fmtExpr).join(', ')}]`;
      case 'MapLiteral':
        return `{ ${expr.entries.map((e) => `${fmtExpr(e.key)}: ${fmtExpr(e.value)}`).join(', ')} }`;
      case 'StructLiteral':
        return `${expr.name} { ${expr.fields.map((f) => `${f.name}: ${fmtExpr(f.value)}`).join(', ')} }`;
      case 'ResultExpr': return `${expr.variant}(${fmtExpr(expr.value)})`;
    }
  };

  const fmtStmt = (stmt: Stmt, level: number): string => {
    const pad = indent(level);
    switch (stmt.kind) {
      case 'UseStmt':
        return `${pad}use ${stmt.path.join('.')}`;
      case 'StructDecl': {
        const fields = stmt.fields.map((f) => `${indent(level + 1)}${f.name}: ${formatTypeAnnotation(f.typeAnnotation)}`).join('\n');
        return `${pad}struct ${stmt.name} {\n${fields}\n${pad}}`;
      }
      case 'FnDecl': {
        const params = stmt.params.map((p) => `${p.name}: ${formatTypeAnnotation(p.typeAnnotation)}`).join(', ');
        const ret = stmt.returnType ? ` -> ${formatTypeAnnotation(stmt.returnType)}` : '';
        const body = stmt.body.map((s) => fmtStmt(s, level + 1)).join('\n');
        return `${pad}fn ${stmt.name}(${params})${ret} {\n${body}\n${pad}}`;
      }
      case 'VarDecl': {
        const ann = stmt.typeAnnotation ? `: ${formatTypeAnnotation(stmt.typeAnnotation)}` : '';
        return `${pad}${stmt.mutability} ${stmt.name}${ann} = ${fmtExpr(stmt.initializer)}`;
      }
      case 'AssignStmt':
        return `${pad}${fmtExpr(stmt.target)} ${stmt.op} ${fmtExpr(stmt.value)}`;
      case 'IfStmt': {
        const thenBody = stmt.thenBranch.map((s) => fmtStmt(s, level + 1)).join('\n');
        let out = `${pad}if ${fmtExpr(stmt.condition)} {\n${thenBody}\n${pad}}`;
        if (stmt.elseBranch) {
          if (Array.isArray(stmt.elseBranch)) {
            const elseBody = stmt.elseBranch.map((s) => fmtStmt(s, level + 1)).join('\n');
            out += ` else {\n${elseBody}\n${pad}}`;
          } else {
            out += ` else ${fmtStmt(stmt.elseBranch, level).trimStart()}`;
          }
        }
        return out;
      }
      case 'WhileStmt': {
        const body = stmt.body.map((s) => fmtStmt(s, level + 1)).join('\n');
        return `${pad}while ${fmtExpr(stmt.condition)} {\n${body}\n${pad}}`;
      }
      case 'ForStmt': {
        const body = stmt.body.map((s) => fmtStmt(s, level + 1)).join('\n');
        return `${pad}for ${stmt.variable} in ${fmtExpr(stmt.iterable)} {\n${body}\n${pad}}`;
      }
      case 'ReturnStmt':
        return stmt.value ? `${pad}return ${fmtExpr(stmt.value)}` : `${pad}return`;
      case 'BreakStmt':
        return `${pad}break`;
      case 'ContinueStmt':
        return `${pad}continue`;
      case 'ExprStmt':
        return `${pad}${fmtExpr(stmt.expr)}`;
    }
  };

  const sections: string[] = [];
  for (let i = 0; i < program.items.length; i++) {
    const item = program.items[i];
    const formatted = fmtStmt(item, 0);
    sections.push(formatted);
    const next = program.items[i + 1];
    if (next && (item.kind === 'FnDecl' || item.kind === 'StructDecl' || (item.kind === 'UseStmt' && next.kind !== 'UseStmt'))) {
      sections.push('');
    }
  }
  return sections.join('\n') + '\n';
}
