import { Expr, FnDecl, Program, Stmt } from './ast';
import { Diagnostic } from './diagnostics';

export type IRConstValue =
  | { kind: 'Int'; value: number }
  | { kind: 'Float'; value: number }
  | { kind: 'Bool'; value: boolean }
  | { kind: 'String'; value: string }
  | { kind: 'Char'; value: string }
  | { kind: 'None' };

export type IRCompareOp = '==' | '!=' | '<' | '<=' | '>' | '>=';

export type IRInstruction =
  | { op: 'CONST'; dst: string; value: IRConstValue }
  | { op: 'LOAD'; dst: string; slot: string }
  | { op: 'STORE'; slot: string; src: string }
  | { op: 'LOAD_GLOBAL'; dst: string; name: string }
  | { op: 'STORE_GLOBAL'; name: string; src: string }
  | { op: 'ADD'; dst: string; left: string; right: string }
  | { op: 'SUB'; dst: string; left: string; right: string }
  | { op: 'MUL'; dst: string; left: string; right: string }
  | { op: 'DIV'; dst: string; left: string; right: string }
  | { op: 'MOD'; dst: string; left: string; right: string }
  | { op: 'MATMUL'; dst: string; left: string; right: string }
  | { op: 'NEG'; dst: string; src: string }
  | { op: 'NOT'; dst: string; src: string }
  | { op: 'AND'; dst: string; left: string; right: string }
  | { op: 'OR'; dst: string; left: string; right: string }
  | { op: 'COMPARE'; dst: string; cmp: IRCompareOp; left: string; right: string }
  | { op: 'CALL'; dst: string; callee: string; args: string[] }
  | { op: 'CALL_METHOD'; dst: string; object: string; method: string; args: string[] }
  | { op: 'ALLOC_STRUCT'; dst: string; typeName: string; fields: Array<{ name: string; src: string }> }
  | { op: 'LOAD_FIELD'; dst: string; object: string; field: string }
  | { op: 'STORE_FIELD'; object: string; field: string; src: string }
  | { op: 'ALLOC_ARRAY'; dst: string; elements: string[] }
  | { op: 'ALLOC_MAP'; dst: string; entries: Array<{ key: string; value: string }> }
  | { op: 'INDEX_GET'; dst: string; object: string; index: string }
  | { op: 'INDEX_SET'; object: string; index: string; src: string }
  | { op: 'MAKE_RESULT'; dst: string; variant: 'Ok' | 'Error'; src: string }
  | { op: 'JUMP'; target: string }
  | { op: 'JUMP_IF_FALSE'; cond: string; target: string }
  | { op: 'RETURN'; src?: string };

export interface IRBasicBlock {
  label: string;
  instructions: IRInstruction[];
}

export interface IRFunction {
  name: string;
  params: string[];
  blocks: IRBasicBlock[];
}

export interface IRModule {
  functions: IRFunction[];
  structs: Array<{ name: string; fields: string[] }>;
}

export interface OptimizationOptions {
  constantFolding: boolean;
  constantPropagation: boolean;
  deadCodeElimination: boolean;
}

export interface OptimizationStats {
  instructionsBefore: number;
  instructionsAfter: number;
  constantsFolded: number;
  constantsPropagated: number;
  deadInstructionsRemoved: number;
}

export function generateIR(program: Program): IRModule {
  const functions: IRFunction[] = [];
  const structs: Array<{ name: string; fields: string[] }> = [];
  const topLevelStmts: Stmt[] = [];
  const globalNames = new Set<string>();

  for (const item of program.items) {
    if (item.kind === 'StructDecl') {
      structs.push({ name: item.name, fields: item.fields.map((f) => f.name) });
    } else if (item.kind === 'VarDecl') {
      globalNames.add(item.name);
      topLevelStmts.push(item);
    } else if (item.kind === 'FnDecl') {
      // Lowered after collecting globalNames
    } else if (item.kind !== 'UseStmt') {
      topLevelStmts.push(item);
    }
  }

  const lowerFunction = (name: string, params: string[], body: Stmt[], isTopLevel: boolean): IRFunction => {
    const blocks: IRBasicBlock[] = [];
    let tempId = 0;
    let blockId = 0;
    let slotUid = 0;

    // Lexical scope stack mapping user variable names -> unique IR local slot names
    const scopeStack: Array<Map<string, string>> = [new Map()];
    for (const p of params) {
      scopeStack[0].set(p, p);
    }

    const pushScope = (): void => {
      scopeStack.push(new Map());
    };
    const popScope = (): void => {
      scopeStack.pop();
    };
    const declareLocal = (varName: string): string => {
      const currentScope = scopeStack[scopeStack.length - 1];
      const existingOuter = lookupLocal(varName);
      const uniqueSlot = existingOuter ? `${varName}_${++slotUid}` : varName;
      currentScope.set(varName, uniqueSlot);
      return uniqueSlot;
    };
    const lookupLocal = (varName: string): string | undefined => {
      for (let i = scopeStack.length - 1; i >= 0; i--) {
        const hit = scopeStack[i].get(varName);
        if (hit !== undefined) return hit;
      }
      return undefined;
    };

    const loopStack: Array<{ breakLabel: string; continueLabel: string }> = [];

    const newTemp = (): string => `%t${tempId++}`;
    const newBlock = (prefix = 'B'): IRBasicBlock => {
      const b: IRBasicBlock = { label: `${prefix}${blockId++}`, instructions: [] };
      blocks.push(b);
      return b;
    };

    let current = newBlock('entry_');

    const emit = (inst: IRInstruction): void => {
      current.instructions.push(inst);
    };

    const isTerminated = (b: IRBasicBlock): boolean => {
      if (b.instructions.length === 0) return false;
      const last = b.instructions[b.instructions.length - 1];
      return last.op === 'RETURN' || last.op === 'JUMP';
    };

    const lowerExpr = (expr: Expr): string => {
      switch (expr.kind) {
        case 'IntLiteral': {
          const dst = newTemp();
          emit({ op: 'CONST', dst, value: { kind: 'Int', value: expr.value } });
          return dst;
        }
        case 'FloatLiteral': {
          const dst = newTemp();
          emit({ op: 'CONST', dst, value: { kind: 'Float', value: expr.value } });
          return dst;
        }
        case 'StringLiteral': {
          const dst = newTemp();
          emit({ op: 'CONST', dst, value: { kind: 'String', value: expr.value } });
          return dst;
        }
        case 'CharLiteral': {
          const dst = newTemp();
          emit({ op: 'CONST', dst, value: { kind: 'Char', value: expr.value } });
          return dst;
        }
        case 'BoolLiteral': {
          const dst = newTemp();
          emit({ op: 'CONST', dst, value: { kind: 'Bool', value: expr.value } });
          return dst;
        }
        case 'NoneLiteral': {
          const dst = newTemp();
          emit({ op: 'CONST', dst, value: { kind: 'None' } });
          return dst;
        }
        case 'IdentExpr': {
          const dst = newTemp();
          const localSlot = lookupLocal(expr.name);
          if (localSlot !== undefined) {
            emit({ op: 'LOAD', dst, slot: localSlot });
          } else {
            emit({ op: 'LOAD_GLOBAL', dst, name: expr.name });
          }
          return dst;
        }
        case 'UnaryExpr': {
          const src = lowerExpr(expr.operand);
          const dst = newTemp();
          emit({ op: expr.op === '-' ? 'NEG' : 'NOT', dst, src });
          return dst;
        }
        case 'BinaryExpr': {
          // Short-circuit evaluation for && and ||
          if (expr.op === '&&' || expr.op === '||') {
            const resSlot = `__sc_${++slotUid}`;
            const left = lowerExpr(expr.left);
            emit({ op: 'STORE', slot: resSlot, src: left });

            const rhsBlock = newBlock('sc_rhs_');
            const endBlock = newBlock('sc_end_');

            if (expr.op === '&&') {
              emit({ op: 'JUMP_IF_FALSE', cond: left, target: endBlock.label });
              emit({ op: 'JUMP', target: rhsBlock.label });
            } else {
              const notLeft = newTemp();
              emit({ op: 'NOT', dst: notLeft, src: left });
              emit({ op: 'JUMP_IF_FALSE', cond: notLeft, target: endBlock.label });
              emit({ op: 'JUMP', target: rhsBlock.label });
            }

            current = rhsBlock;
            const right = lowerExpr(expr.right);
            emit({ op: 'STORE', slot: resSlot, src: right });
            emit({ op: 'JUMP', target: endBlock.label });

            current = endBlock;
            const dst = newTemp();
            emit({ op: 'LOAD', dst, slot: resSlot });
            return dst;
          }

          const left = lowerExpr(expr.left);
          const right = lowerExpr(expr.right);
          const dst = newTemp();
          switch (expr.op) {
            case '+': emit({ op: 'ADD', dst, left, right }); break;
            case '-': emit({ op: 'SUB', dst, left, right }); break;
            case '*': emit({ op: 'MUL', dst, left, right }); break;
            case '/': emit({ op: 'DIV', dst, left, right }); break;
            case '%': emit({ op: 'MOD', dst, left, right }); break;
            case '@': emit({ op: 'MATMUL', dst, left, right }); break;
            case '==':
            case '!=':
            case '<':
            case '<=':
            case '>':
            case '>=':
              emit({ op: 'COMPARE', dst, cmp: expr.op, left, right });
              break;
          }
          return dst;
        }
        case 'CallExpr': {
          const dst = newTemp();
          if (expr.callee.kind === 'FieldExpr') {
            if (
              expr.callee.object.kind === 'IdentExpr' &&
              lookupLocal(expr.callee.object.name) === undefined &&
              !globalNames.has(expr.callee.object.name)
            ) {
              const argTemps = expr.args.map(lowerExpr);
              emit({ op: 'CALL', dst, callee: `${expr.callee.object.name}.${expr.callee.field}`, args: argTemps });
            } else {
              const obj = lowerExpr(expr.callee.object);
              const argTemps = expr.args.map(lowerExpr);
              emit({ op: 'CALL_METHOD', dst, object: obj, method: expr.callee.field, args: argTemps });
            }
          } else if (expr.callee.kind === 'IdentExpr') {
            const argTemps = expr.args.map(lowerExpr);
            emit({ op: 'CALL', dst, callee: expr.callee.name, args: argTemps });
          } else {
            emit({ op: 'CONST', dst, value: { kind: 'None' } });
          }
          return dst;
        }
        case 'StructLiteral': {
          const fields = expr.fields.map((f) => ({ name: f.name, src: lowerExpr(f.value) }));
          const dst = newTemp();
          emit({ op: 'ALLOC_STRUCT', dst, typeName: expr.name, fields });
          return dst;
        }
        case 'FieldExpr': {
          const obj = lowerExpr(expr.object);
          const dst = newTemp();
          emit({ op: 'LOAD_FIELD', dst, object: obj, field: expr.field });
          return dst;
        }
        case 'ArrayLiteral': {
          const elements = expr.elements.map(lowerExpr);
          const dst = newTemp();
          emit({ op: 'ALLOC_ARRAY', dst, elements });
          return dst;
        }
        case 'MapLiteral': {
          const entries = expr.entries.map((e) => ({ key: lowerExpr(e.key), value: lowerExpr(e.value) }));
          const dst = newTemp();
          emit({ op: 'ALLOC_MAP', dst, entries });
          return dst;
        }
        case 'IndexExpr': {
          const obj = lowerExpr(expr.object);
          const idx = lowerExpr(expr.index);
          const dst = newTemp();
          emit({ op: 'INDEX_GET', dst, object: obj, index: idx });
          return dst;
        }
        case 'ResultExpr': {
          const src = lowerExpr(expr.value);
          const dst = newTemp();
          emit({ op: 'MAKE_RESULT', dst, variant: expr.variant, src });
          return dst;
        }
        case 'RangeExpr': {
          const s = lowerExpr(expr.start);
          const e = lowerExpr(expr.end);
          const dst = newTemp();
          emit({ op: 'CALL', dst, callee: 'range', args: [s, e] });
          return dst;
        }
      }
    };

    const lowerStmt = (stmt: Stmt): void => {
      switch (stmt.kind) {
        case 'VarDecl': {
          const val = lowerExpr(stmt.initializer);
          if (isTopLevel) {
            emit({ op: 'STORE_GLOBAL', name: stmt.name, src: val });
          } else {
            const slot = declareLocal(stmt.name);
            emit({ op: 'STORE', slot, src: val });
          }
          return;
        }

        case 'AssignStmt': {
          if (stmt.target.kind === 'IdentExpr') {
            const targetName = stmt.target.name;
            const localSlot = lookupLocal(targetName);
            let rhs = lowerExpr(stmt.value);
            if (stmt.op !== '=') {
              const cur = newTemp();
              if (localSlot !== undefined) {
                emit({ op: 'LOAD', dst: cur, slot: localSlot });
              } else {
                emit({ op: 'LOAD_GLOBAL', dst: cur, name: targetName });
              }
              const res = newTemp();
              const opMap = { '+=': 'ADD', '-=': 'SUB', '*=': 'MUL', '/=': 'DIV' } as const;
              emit({ op: opMap[stmt.op], dst: res, left: cur, right: rhs });
              rhs = res;
            }
            if (localSlot !== undefined) {
              emit({ op: 'STORE', slot: localSlot, src: rhs });
            } else {
              emit({ op: 'STORE_GLOBAL', name: targetName, src: rhs });
            }
          } else if (stmt.target.kind === 'FieldExpr') {
            const obj = lowerExpr(stmt.target.object);
            let rhs = lowerExpr(stmt.value);
            if (stmt.op !== '=') {
              const cur = newTemp();
              emit({ op: 'LOAD_FIELD', dst: cur, object: obj, field: stmt.target.field });
              const res = newTemp();
              const opMap = { '+=': 'ADD', '-=': 'SUB', '*=': 'MUL', '/=': 'DIV' } as const;
              emit({ op: opMap[stmt.op], dst: res, left: cur, right: rhs });
              rhs = res;
            }
            emit({ op: 'STORE_FIELD', object: obj, field: stmt.target.field, src: rhs });
          } else if (stmt.target.kind === 'IndexExpr') {
            const obj = lowerExpr(stmt.target.object);
            const idx = lowerExpr(stmt.target.index);
            let rhs = lowerExpr(stmt.value);
            if (stmt.op !== '=') {
              const cur = newTemp();
              emit({ op: 'INDEX_GET', dst: cur, object: obj, index: idx });
              const res = newTemp();
              const opMap = { '+=': 'ADD', '-=': 'SUB', '*=': 'MUL', '/=': 'DIV' } as const;
              emit({ op: opMap[stmt.op], dst: res, left: cur, right: rhs });
              rhs = res;
            }
            emit({ op: 'INDEX_SET', object: obj, index: idx, src: rhs });
          }
          return;
        }

        case 'IfStmt': {
          const cond = lowerExpr(stmt.condition);
          const thenBlock = newBlock('then_');
          const elseBlock = stmt.elseBranch ? newBlock('else_') : null;
          const mergeBlock = newBlock('merge_');

          emit({ op: 'JUMP_IF_FALSE', cond, target: elseBlock ? elseBlock.label : mergeBlock.label });
          emit({ op: 'JUMP', target: thenBlock.label });

          current = thenBlock;
          pushScope();
          for (const s of stmt.thenBranch) lowerStmt(s);
          popScope();
          if (!isTerminated(current)) emit({ op: 'JUMP', target: mergeBlock.label });

          if (elseBlock && stmt.elseBranch) {
            current = elseBlock;
            pushScope();
            if (Array.isArray(stmt.elseBranch)) {
              for (const s of stmt.elseBranch) lowerStmt(s);
            } else {
              lowerStmt(stmt.elseBranch);
            }
            popScope();
            if (!isTerminated(current)) emit({ op: 'JUMP', target: mergeBlock.label });
          }

          current = mergeBlock;
          return;
        }

        case 'WhileStmt': {
          const condBlock = newBlock('while_cond_');
          const bodyBlock = newBlock('while_body_');
          const endBlock = newBlock('while_end_');

          emit({ op: 'JUMP', target: condBlock.label });

          current = condBlock;
          const cond = lowerExpr(stmt.condition);
          emit({ op: 'JUMP_IF_FALSE', cond, target: endBlock.label });
          emit({ op: 'JUMP', target: bodyBlock.label });

          current = bodyBlock;
          pushScope();
          loopStack.push({ breakLabel: endBlock.label, continueLabel: condBlock.label });
          for (const s of stmt.body) lowerStmt(s);
          loopStack.pop();
          popScope();
          if (!isTerminated(current)) emit({ op: 'JUMP', target: condBlock.label });

          current = endBlock;
          return;
        }

        case 'ForStmt': {
          pushScope();
          if (stmt.iterable.kind === 'RangeExpr') {
            const startTemp = lowerExpr(stmt.iterable.start);
            const endTemp = lowerExpr(stmt.iterable.end);
            const varSlot = declareLocal(stmt.variable);
            const endSlot = `__for_end_${tempId++}`;
            emit({ op: 'STORE', slot: varSlot, src: startTemp });
            emit({ op: 'STORE', slot: endSlot, src: endTemp });

            const condBlock = newBlock('for_cond_');
            const bodyBlock = newBlock('for_body_');
            const stepBlock = newBlock('for_step_');
            const endBlock = newBlock('for_end_');

            emit({ op: 'JUMP', target: condBlock.label });

            current = condBlock;
            const curVal = newTemp();
            const limVal = newTemp();
            const cmp = newTemp();
            emit({ op: 'LOAD', dst: curVal, slot: varSlot });
            emit({ op: 'LOAD', dst: limVal, slot: endSlot });
            emit({ op: 'COMPARE', dst: cmp, cmp: '<', left: curVal, right: limVal });
            emit({ op: 'JUMP_IF_FALSE', cond: cmp, target: endBlock.label });
            emit({ op: 'JUMP', target: bodyBlock.label });

            current = bodyBlock;
            pushScope();
            loopStack.push({ breakLabel: endBlock.label, continueLabel: stepBlock.label });
            for (const s of stmt.body) lowerStmt(s);
            loopStack.pop();
            popScope();
            if (!isTerminated(current)) emit({ op: 'JUMP', target: stepBlock.label });

            current = stepBlock;
            const oldI = newTemp();
            const one = newTemp();
            const nextI = newTemp();
            emit({ op: 'LOAD', dst: oldI, slot: varSlot });
            emit({ op: 'CONST', dst: one, value: { kind: 'Int', value: 1 } });
            emit({ op: 'ADD', dst: nextI, left: oldI, right: one });
            emit({ op: 'STORE', slot: varSlot, src: nextI });
            emit({ op: 'JUMP', target: condBlock.label });

            current = endBlock;
          } else {
            const arrTemp = lowerExpr(stmt.iterable);
            const arrSlot = `__for_arr_${tempId++}`;
            const idxSlot = `__for_idx_${tempId++}`;
            const lenSlot = `__for_len_${tempId++}`;
            const varSlot = declareLocal(stmt.variable);
            emit({ op: 'STORE', slot: arrSlot, src: arrTemp });
            const zero = newTemp();
            emit({ op: 'CONST', dst: zero, value: { kind: 'Int', value: 0 } });
            emit({ op: 'STORE', slot: idxSlot, src: zero });
            const lenVal = newTemp();
            emit({ op: 'CALL', dst: lenVal, callee: 'len', args: [arrTemp] });
            emit({ op: 'STORE', slot: lenSlot, src: lenVal });

            const condBlock = newBlock('for_cond_');
            const bodyBlock = newBlock('for_body_');
            const stepBlock = newBlock('for_step_');
            const endBlock = newBlock('for_end_');

            emit({ op: 'JUMP', target: condBlock.label });
            current = condBlock;
            const iCur = newTemp();
            const lCur = newTemp();
            const cmp = newTemp();
            emit({ op: 'LOAD', dst: iCur, slot: idxSlot });
            emit({ op: 'LOAD', dst: lCur, slot: lenSlot });
            emit({ op: 'COMPARE', dst: cmp, cmp: '<', left: iCur, right: lCur });
            emit({ op: 'JUMP_IF_FALSE', cond: cmp, target: endBlock.label });
            emit({ op: 'JUMP', target: bodyBlock.label });

            current = bodyBlock;
            const arrLoad = newTemp();
            const idxLoad = newTemp();
            const elem = newTemp();
            emit({ op: 'LOAD', dst: arrLoad, slot: arrSlot });
            emit({ op: 'LOAD', dst: idxLoad, slot: idxSlot });
            emit({ op: 'INDEX_GET', dst: elem, object: arrLoad, index: idxLoad });
            emit({ op: 'STORE', slot: varSlot, src: elem });
            pushScope();
            loopStack.push({ breakLabel: endBlock.label, continueLabel: stepBlock.label });
            for (const s of stmt.body) lowerStmt(s);
            loopStack.pop();
            popScope();
            if (!isTerminated(current)) emit({ op: 'JUMP', target: stepBlock.label });

            current = stepBlock;
            const iOld = newTemp();
            const one = newTemp();
            const iNext = newTemp();
            emit({ op: 'LOAD', dst: iOld, slot: idxSlot });
            emit({ op: 'CONST', dst: one, value: { kind: 'Int', value: 1 } });
            emit({ op: 'ADD', dst: iNext, left: iOld, right: one });
            emit({ op: 'STORE', slot: idxSlot, src: iNext });
            emit({ op: 'JUMP', target: condBlock.label });

            current = endBlock;
          }
          popScope();
          return;
        }

        case 'ReturnStmt': {
          const src = stmt.value ? lowerExpr(stmt.value) : undefined;
          emit({ op: 'RETURN', src });
          return;
        }

        case 'BreakStmt': {
          const top = loopStack[loopStack.length - 1];
          if (top) emit({ op: 'JUMP', target: top.breakLabel });
          return;
        }

        case 'ContinueStmt': {
          const top = loopStack[loopStack.length - 1];
          if (top) emit({ op: 'JUMP', target: top.continueLabel });
          return;
        }

        case 'ExprStmt': {
          lowerExpr(stmt.expr);
          return;
        }

        default:
          return;
      }
    };

    for (const s of body) {
      lowerStmt(s);
    }
    if (!isTerminated(current)) {
      emit({ op: 'RETURN' });
    }

    return { name, params, blocks };
  };

  if (topLevelStmts.length > 0) {
    functions.push(lowerFunction('__init__', [], topLevelStmts, true));
  }

  for (const item of program.items) {
    if (item.kind === 'FnDecl') {
      const fn = item as FnDecl;
      functions.push(
        lowerFunction(
          fn.name,
          fn.params.map((p) => p.name),
          fn.body,
          false
        )
      );
    }
  }

  return { functions, structs };
}

function countInstructions(mod: IRModule): number {
  let count = 0;
  for (const fn of mod.functions) {
    for (const b of fn.blocks) {
      count += b.instructions.length;
    }
  }
  return count;
}

function cloneModule(mod: IRModule): IRModule {
  return JSON.parse(JSON.stringify(mod)) as IRModule;
}

function foldBinary(op: string, l: IRConstValue, r: IRConstValue): IRConstValue | null {
  if ((l.kind === 'Int' || l.kind === 'Float') && (r.kind === 'Int' || r.kind === 'Float')) {
    const bothInt = l.kind === 'Int' && r.kind === 'Int';
    const lv = l.value;
    const rv = r.value;
    switch (op) {
      case 'ADD': return bothInt ? { kind: 'Int', value: lv + rv } : { kind: 'Float', value: lv + rv };
      case 'SUB': return bothInt ? { kind: 'Int', value: lv - rv } : { kind: 'Float', value: lv - rv };
      case 'MUL': return bothInt ? { kind: 'Int', value: lv * rv } : { kind: 'Float', value: lv * rv };
      case 'DIV':
        if (rv === 0) return null;
        return bothInt ? { kind: 'Int', value: Math.trunc(lv / rv) } : { kind: 'Float', value: lv / rv };
      case 'MOD':
        if (rv === 0) return null;
        return bothInt ? { kind: 'Int', value: lv % rv } : { kind: 'Float', value: lv % rv };
    }
  }
  if (op === 'ADD' && l.kind === 'String' && r.kind === 'String') {
    return { kind: 'String', value: l.value + r.value };
  }
  if (op === 'AND' && l.kind === 'Bool' && r.kind === 'Bool') {
    return { kind: 'Bool', value: l.value && r.value };
  }
  if (op === 'OR' && l.kind === 'Bool' && r.kind === 'Bool') {
    return { kind: 'Bool', value: l.value || r.value };
  }
  return null;
}

function foldCompare(cmp: IRCompareOp, l: IRConstValue, r: IRConstValue): IRConstValue | null {
  if ((l.kind === 'Int' || l.kind === 'Float') && (r.kind === 'Int' || r.kind === 'Float')) {
    const lv = l.value;
    const rv = r.value;
    switch (cmp) {
      case '==': return { kind: 'Bool', value: lv === rv };
      case '!=': return { kind: 'Bool', value: lv !== rv };
      case '<': return { kind: 'Bool', value: lv < rv };
      case '<=': return { kind: 'Bool', value: lv <= rv };
      case '>': return { kind: 'Bool', value: lv > rv };
      case '>=': return { kind: 'Bool', value: lv >= rv };
    }
  }
  return null;
}

export function optimizeIR(
  input: IRModule,
  options: OptimizationOptions
): { module: IRModule; stats: OptimizationStats } {
  const mod = cloneModule(input);
  const instructionsBefore = countInstructions(mod);
  let constantsFolded = 0;
  let constantsPropagated = 0;
  let deadInstructionsRemoved = 0;

  for (const fn of mod.functions) {
    // Pass 1: Basic-block Constant Propagation + Constant Folding
    if (options.constantPropagation || options.constantFolding) {
      // Find slots written only once (single-assignment locals safe for propagation across block)
      const slotWrites = new Map<string, number>();
      for (const b of fn.blocks) {
        for (const inst of b.instructions) {
          if (inst.op === 'STORE') {
            slotWrites.set(inst.slot, (slotWrites.get(inst.slot) ?? 0) + 1);
          }
        }
      }

      for (const b of fn.blocks) {
        const tempConsts = new Map<string, IRConstValue>();
        const slotConsts = new Map<string, IRConstValue>();

        for (let i = 0; i < b.instructions.length; i++) {
          const inst = b.instructions[i];
          if (inst.op === 'CONST') {
            tempConsts.set(inst.dst, inst.value);
          } else if (inst.op === 'STORE') {
            const c = tempConsts.get(inst.src);
            if (options.constantPropagation && c && slotWrites.get(inst.slot) === 1) {
              slotConsts.set(inst.slot, c);
            } else {
              slotConsts.delete(inst.slot);
            }
          } else if (inst.op === 'LOAD') {
            if (options.constantPropagation && slotConsts.has(inst.slot)) {
              const c = slotConsts.get(inst.slot)!;
              b.instructions[i] = { op: 'CONST', dst: inst.dst, value: c };
              tempConsts.set(inst.dst, c);
              constantsPropagated++;
            }
          } else if (
            options.constantFolding &&
            (inst.op === 'ADD' ||
              inst.op === 'SUB' ||
              inst.op === 'MUL' ||
              inst.op === 'DIV' ||
              inst.op === 'MOD' ||
              inst.op === 'AND' ||
              inst.op === 'OR')
          ) {
            const lc = tempConsts.get(inst.left);
            const rc = tempConsts.get(inst.right);
            if (lc && rc) {
              const folded = foldBinary(inst.op, lc, rc);
              if (folded) {
                b.instructions[i] = { op: 'CONST', dst: inst.dst, value: folded };
                tempConsts.set(inst.dst, folded);
                constantsFolded++;
              }
            }
          } else if (options.constantFolding && inst.op === 'COMPARE') {
            const lc = tempConsts.get(inst.left);
            const rc = tempConsts.get(inst.right);
            if (lc && rc) {
              const folded = foldCompare(inst.cmp, lc, rc);
              if (folded) {
                b.instructions[i] = { op: 'CONST', dst: inst.dst, value: folded };
                tempConsts.set(inst.dst, folded);
                constantsFolded++;
              }
            }
          } else if (options.constantFolding && inst.op === 'NEG') {
            const c = tempConsts.get(inst.src);
            if (c && (c.kind === 'Int' || c.kind === 'Float')) {
              const folded: IRConstValue = { kind: c.kind, value: -c.value };
              b.instructions[i] = { op: 'CONST', dst: inst.dst, value: folded };
              tempConsts.set(inst.dst, folded);
              constantsFolded++;
            }
          } else if (options.constantFolding && inst.op === 'NOT') {
            const c = tempConsts.get(inst.src);
            if (c && c.kind === 'Bool') {
              const folded: IRConstValue = { kind: 'Bool', value: !c.value };
              b.instructions[i] = { op: 'CONST', dst: inst.dst, value: folded };
              tempConsts.set(inst.dst, folded);
              constantsFolded++;
            }
          }
        }
      }
    }

    // Pass 2: Dead-Code Elimination (unused pure temp definitions & instructions after terminator)
    if (options.deadCodeElimination) {
      // Remove instructions after RETURN/JUMP inside each block
      for (const b of fn.blocks) {
        const termIdx = b.instructions.findIndex((inst) => inst.op === 'RETURN' || inst.op === 'JUMP');
        if (termIdx !== -1 && termIdx < b.instructions.length - 1) {
          deadInstructionsRemoved += b.instructions.length - 1 - termIdx;
          b.instructions = b.instructions.slice(0, termIdx + 1);
        }
      }

      // Iteratively remove unused pure temp computations
      let changed = true;
      while (changed) {
        changed = false;
        const usedTemps = new Set<string>();
        const addUse = (t?: string) => { if (t) usedTemps.add(t); };

        for (const b of fn.blocks) {
          for (const inst of b.instructions) {
            switch (inst.op) {
              case 'STORE':
              case 'STORE_GLOBAL':
              case 'NEG':
              case 'NOT':
              case 'MAKE_RESULT':
                addUse(inst.src);
                break;
              case 'ADD':
              case 'SUB':
              case 'MUL':
              case 'DIV':
              case 'MOD':
              case 'MATMUL':
              case 'AND':
              case 'OR':
              case 'COMPARE':
                addUse(inst.left);
                addUse(inst.right);
                break;
              case 'CALL':
                inst.args.forEach(addUse);
                break;
              case 'CALL_METHOD':
                addUse(inst.object);
                inst.args.forEach(addUse);
                break;
              case 'ALLOC_STRUCT':
                inst.fields.forEach((f) => addUse(f.src));
                break;
              case 'LOAD_FIELD':
                addUse(inst.object);
                break;
              case 'STORE_FIELD':
                addUse(inst.object);
                addUse(inst.src);
                break;
              case 'ALLOC_ARRAY':
                inst.elements.forEach(addUse);
                break;
              case 'ALLOC_MAP':
                inst.entries.forEach((e) => { addUse(e.key); addUse(e.value); });
                break;
              case 'INDEX_GET':
                addUse(inst.object);
                addUse(inst.index);
                break;
              case 'INDEX_SET':
                addUse(inst.object);
                addUse(inst.index);
                addUse(inst.src);
                break;
              case 'JUMP_IF_FALSE':
                addUse(inst.cond);
                break;
              case 'RETURN':
                addUse(inst.src);
                break;
            }
          }
        }

        const isPure = (inst: IRInstruction): inst is Extract<IRInstruction, { dst: string }> =>
          inst.op === 'CONST' ||
          inst.op === 'LOAD' ||
          inst.op === 'ADD' ||
          inst.op === 'SUB' ||
          inst.op === 'MUL' ||
          inst.op === 'NEG' ||
          inst.op === 'NOT' ||
          inst.op === 'COMPARE' ||
          inst.op === 'AND' ||
          inst.op === 'OR';

        for (const b of fn.blocks) {
          const next: IRInstruction[] = [];
          for (const inst of b.instructions) {
            if (isPure(inst) && !usedTemps.has(inst.dst)) {
              deadInstructionsRemoved++;
              changed = true;
              continue;
            }
            next.push(inst);
          }
          b.instructions = next;
        }
      }
    }
  }

  return {
    module: mod,
    stats: {
      instructionsBefore,
      instructionsAfter: countInstructions(mod),
      constantsFolded,
      constantsPropagated,
      deadInstructionsRemoved,
    },
  };
}

export function validateIR(mod: IRModule): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  for (const fn of mod.functions) {
    const labels = new Set(fn.blocks.map((b) => b.label));
    for (const b of fn.blocks) {
      for (const inst of b.instructions) {
        if ((inst.op === 'JUMP' || inst.op === 'JUMP_IF_FALSE') && !labels.has(inst.target)) {
          diagnostics.push({
            stage: 'ir',
            severity: 'error',
            code: 'E0301',
            message: `IR validation failed: jump to unknown block '${inst.target}' in function '${fn.name}'`,
            span: { start: 0, end: 0, line: 1, column: 1 },
          });
        }
      }
    }
  }
  return diagnostics;
}

export function formatIR(mod: IRModule): string {
  const lines: string[] = [];
  for (const s of mod.structs) {
    lines.push(`struct ${s.name} { ${s.fields.join(', ')} }`);
  }
  if (mod.structs.length > 0) lines.push('');

  for (const fn of mod.functions) {
    lines.push(`fn @${fn.name}(${fn.params.join(', ')}) {`);
    for (const b of fn.blocks) {
      lines.push(`  ${b.label}:`);
      for (const inst of b.instructions) {
        lines.push(`    ${formatIRInst(inst)}`);
      }
    }
    lines.push('}');
    lines.push('');
  }
  return lines.join('\n').trimEnd();
}

function formatConst(v: IRConstValue): string {
  switch (v.kind) {
    case 'Int': return `i64 ${v.value}`;
    case 'Float': return `f64 ${v.value}`;
    case 'Bool': return `bool ${v.value}`;
    case 'String': return `str ${JSON.stringify(v.value)}`;
    case 'Char': return `char '${v.value}'`;
    case 'None': return 'none';
  }
}

function formatIRInst(inst: IRInstruction): string {
  switch (inst.op) {
    case 'CONST': return `${inst.dst} = CONST ${formatConst(inst.value)}`;
    case 'LOAD': return `${inst.dst} = LOAD $${inst.slot}`;
    case 'STORE': return `STORE $${inst.slot}, ${inst.src}`;
    case 'LOAD_GLOBAL': return `${inst.dst} = LOAD_GLOBAL @${inst.name}`;
    case 'STORE_GLOBAL': return `STORE_GLOBAL @${inst.name}, ${inst.src}`;
    case 'ADD':
    case 'SUB':
    case 'MUL':
    case 'DIV':
    case 'MOD':
    case 'MATMUL':
    case 'AND':
    case 'OR':
      return `${inst.dst} = ${inst.op} ${inst.left}, ${inst.right}`;
    case 'NEG':
    case 'NOT':
      return `${inst.dst} = ${inst.op} ${inst.src}`;
    case 'COMPARE':
      return `${inst.dst} = COMPARE(${inst.cmp}) ${inst.left}, ${inst.right}`;
    case 'CALL':
      return `${inst.dst} = CALL @${inst.callee}(${inst.args.join(', ')})`;
    case 'CALL_METHOD':
      return `${inst.dst} = CALL_METHOD ${inst.object}.${inst.method}(${inst.args.join(', ')})`;
    case 'ALLOC_STRUCT':
      return `${inst.dst} = ALLOC_STRUCT ${inst.typeName} { ${inst.fields.map((f) => `${f.name}: ${f.src}`).join(', ')} }`;
    case 'LOAD_FIELD':
      return `${inst.dst} = LOAD_FIELD ${inst.object}.${inst.field}`;
    case 'STORE_FIELD':
      return `STORE_FIELD ${inst.object}.${inst.field}, ${inst.src}`;
    case 'ALLOC_ARRAY':
      return `${inst.dst} = ALLOC_ARRAY [${inst.elements.join(', ')}]`;
    case 'ALLOC_MAP':
      return `${inst.dst} = ALLOC_MAP { ${inst.entries.map((e) => `${e.key}: ${e.value}`).join(', ')} }`;
    case 'INDEX_GET':
      return `${inst.dst} = INDEX_GET ${inst.object}[${inst.index}]`;
    case 'INDEX_SET':
      return `INDEX_SET ${inst.object}[${inst.index}], ${inst.src}`;
    case 'MAKE_RESULT':
      return `${inst.dst} = ${inst.variant}(${inst.src})`;
    case 'JUMP':
      return `JUMP ${inst.target}`;
    case 'JUMP_IF_FALSE':
      return `JUMP_IF_FALSE ${inst.cond}, ${inst.target}`;
    case 'RETURN':
      return inst.src ? `RETURN ${inst.src}` : 'RETURN';
  }
}
