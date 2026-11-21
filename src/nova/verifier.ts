import type { BytecodeChunk, BytecodeInst, CompiledFunction } from './bytecode';
import { OpCode } from './bytecode';
import { Diagnostic } from './diagnostics';

export const MAX_VERIFIED_STACK_DEPTH = 1024;
export const SUPPORTED_BYTECODE_VERSION = 1;

export interface FunctionVerificationReport {
  name: string;
  instructions: number;
  reachableInstructions: number;
  maxStackDepth: number;
  ok: boolean;
}

export interface BytecodeVerificationResult {
  ok: boolean;
  functionsVerified: number;
  instructionsVerified: number;
  maxStackDepth: number;
  checksRun: string[];
  perFunction: FunctionVerificationReport[];
  diagnostics: Diagnostic[];
}

let cachedValidOpcodes: Set<number> | null = null;
function getValidOpcodes(): Set<number> {
  if (!cachedValidOpcodes) {
    cachedValidOpcodes = new Set<number>(
      Object.values(OpCode).filter((v): v is number => typeof v === 'number')
    );
  }
  return cachedValidOpcodes;
}

const ZERO_SPAN = { start: 0, end: 0, line: 1, column: 1 };

function makeError(code: string, message: string, hint?: string): Diagnostic {
  return {
    stage: 'bytecode',
    severity: 'error',
    code,
    message,
    span: ZERO_SPAN,
    hint,
  };
}

function isNonNegativeInt(v: unknown): v is number {
  return typeof v === 'number' && Number.isInteger(v) && v >= 0;
}

function isNonEmptyString(v: unknown): v is string {
  return typeof v === 'string' && v.length > 0;
}

/**
 * Computes the exact operand stack effect `(pops, pushes)` of a bytecode instruction.
 */
export function instructionStackEffect(inst: BytecodeInst): { pops: number; pushes: number } {
  switch (inst.op) {
    case OpCode.PUSH_CONST:
    case OpCode.LOAD_LOCAL:
    case OpCode.LOAD_GLOBAL:
      return { pops: 0, pushes: 1 };

    case OpCode.STORE_LOCAL:
    case OpCode.STORE_GLOBAL:
    case OpCode.JUMP_IF_FALSE:
    case OpCode.POP:
      return { pops: 1, pushes: 0 };

    case OpCode.NEG:
    case OpCode.NOT:
    case OpCode.LOAD_FIELD:
    case OpCode.MAKE_OK:
    case OpCode.MAKE_ERROR:
      return { pops: 1, pushes: 1 };

    case OpCode.ADD:
    case OpCode.SUB:
    case OpCode.MUL:
    case OpCode.DIV:
    case OpCode.MOD:
    case OpCode.MATMUL:
    case OpCode.AND:
    case OpCode.OR:
    case OpCode.CMP_EQ:
    case OpCode.CMP_NEQ:
    case OpCode.CMP_LT:
    case OpCode.CMP_LTE:
    case OpCode.CMP_GT:
    case OpCode.CMP_GTE:
    case OpCode.INDEX_GET:
      return { pops: 2, pushes: 1 };

    case OpCode.STORE_FIELD:
      return { pops: 2, pushes: 0 };

    case OpCode.INDEX_SET:
      return { pops: 3, pushes: 0 };

    case OpCode.JUMP:
      return { pops: 0, pushes: 0 };

    case OpCode.RETURN:
      return { pops: 1, pushes: 0 };

    case OpCode.CALL: {
      const argc = isNonNegativeInt(inst.arg2) ? inst.arg2 : 0;
      return { pops: argc, pushes: 1 };
    }

    case OpCode.CALL_METHOD: {
      const argc = isNonNegativeInt(inst.arg2) ? inst.arg2 : 0;
      return { pops: argc + 1, pushes: 1 };
    }

    case OpCode.ALLOC_STRUCT: {
      const count = isNonNegativeInt(inst.arg2) ? inst.arg2 : 0;
      return { pops: count, pushes: 1 };
    }

    case OpCode.ALLOC_ARRAY: {
      const count = isNonNegativeInt(inst.arg) ? inst.arg : 0;
      return { pops: count, pushes: 1 };
    }

    case OpCode.ALLOC_MAP: {
      const count = isNonNegativeInt(inst.arg) ? inst.arg : 0;
      return { pops: count * 2, pushes: 1 };
    }

    default:
      return { pops: 0, pushes: 0 };
  }
}

/**
 * Static Nova Bytecode Verifier.
 *
 * Performs full structural, operand, control-flow, and abstract stack-height
 * verification prior to VM execution:
 * - Header magic (`NOVA01`) & version check (E0401)
 * - Constant pool bounds & integrity (E0402, E0406)
 * - Local variable slot bounds & arity fit (E0403, E0406)
 * - Jump target strict bounds (E0404)
 * - Function table / entry point integrity (E0405)
 * - Opcode validity (E0407)
 * - Call target arity check for user functions (E0408)
 * - Operand shape validation for structs, arrays, maps, fields, globals (E0409)
 * - Worklist abstract stack-depth analysis:
 *   - Stack underflow (E0410)
 *   - Stack overflow (E0411)
 *   - Control-flow merge stack-height mismatch (E0412)
 *   - Missing RETURN / fallthrough past end of function (E0413)
 */
export function verifyBytecode(chunk: BytecodeChunk): BytecodeVerificationResult {
  const diagnostics: Diagnostic[] = [];
  const checksRun = [
    'header-magic-version',
    'constant-pool-integrity',
    'function-table-index',
    'opcode-operand-bounds',
    'call-arity-match',
    'cfg-stack-depth-dataflow',
    'termination-guarantee',
  ];

  // 1. Header magic & version
  if (chunk.magic !== 'NOVA01') {
    diagnostics.push(
      makeError('E0401', `Invalid bytecode header magic '${String(chunk.magic)}' (expected 'NOVA01')`)
    );
  }
  const version = chunk.version ?? SUPPORTED_BYTECODE_VERSION;
  if (version !== SUPPORTED_BYTECODE_VERSION) {
    diagnostics.push(
      makeError(
        'E0401',
        `Unsupported bytecode format version ${String(version)} (expected ${SUPPORTED_BYTECODE_VERSION})`
      )
    );
  }

  // 2. Constant pool integrity
  if (!Array.isArray(chunk.constants)) {
    diagnostics.push(makeError('E0406', 'Bytecode chunk is missing a valid constant pool array'));
  } else {
    chunk.constants.forEach((c, idx) => {
      if (!c || typeof c !== 'object') {
        diagnostics.push(makeError('E0406', `Corrupt constant pool entry at #${idx}`));
        return;
      }
      switch (c.kind) {
        case 'Int':
        case 'Float':
          if (typeof c.value !== 'number' || Number.isNaN(c.value)) {
            diagnostics.push(makeError('E0406', `Invalid numeric constant at #${idx}`));
          }
          break;
        case 'Bool':
          if (typeof c.value !== 'boolean') {
            diagnostics.push(makeError('E0406', `Invalid Bool constant at #${idx}`));
          }
          break;
        case 'String':
        case 'Char': {
          const kindLabel = c.kind;
          if (typeof (c as { value: unknown }).value !== 'string') {
            diagnostics.push(makeError('E0406', `Invalid ${kindLabel} constant at #${idx}`));
          }
          break;
        }
        case 'None':
          break;
        default:
          diagnostics.push(makeError('E0406', `Unknown constant kind at #${idx}`));
      }
    });
  }

  // 3. Function table & functionIndex integrity
  const fnByName = new Map<string, CompiledFunction>();
  if (!Array.isArray(chunk.functions)) {
    diagnostics.push(makeError('E0405', 'Bytecode chunk is missing a valid function table'));
    return {
      ok: false,
      functionsVerified: 0,
      instructionsVerified: 0,
      maxStackDepth: 0,
      checksRun,
      perFunction: [],
      diagnostics,
    };
  }

  chunk.functions.forEach((fn, idx) => {
    if (!isNonEmptyString(fn.name)) {
      diagnostics.push(makeError('E0405', `Function at index ${idx} has an empty or invalid name`));
      return;
    }
    if (fnByName.has(fn.name)) {
      diagnostics.push(makeError('E0405', `Duplicate function name '${fn.name}' in bytecode chunk`));
    } else {
      fnByName.set(fn.name, fn);
    }

    if (chunk.functionIndex[fn.name] !== idx) {
      diagnostics.push(
        makeError(
          'E0405',
          `Function index mismatch for '${fn.name}': table has ${chunk.functionIndex[fn.name]}, expected ${idx}`
        )
      );
    }

    if (!isNonNegativeInt(fn.arity) || !isNonNegativeInt(fn.localCount)) {
      diagnostics.push(makeError('E0406', `Invalid arity or localCount in function '${fn.name}'`));
    } else if (fn.arity > fn.localCount) {
      diagnostics.push(
        makeError(
          'E0406',
          `Function '${fn.name}' arity (${fn.arity}) exceeds localCount (${fn.localCount})`
        )
      );
    }
  });

  for (const [name, mappedIdx] of Object.entries(chunk.functionIndex ?? {})) {
    if (!isNonNegativeInt(mappedIdx) || mappedIdx >= chunk.functions.length || chunk.functions[mappedIdx]?.name !== name) {
      diagnostics.push(
        makeError('E0405', `functionIndex entry '${name}' -> ${String(mappedIdx)} does not match function table`)
      );
    }
  }

  if (chunk.entryPoint && !fnByName.has(chunk.entryPoint)) {
    diagnostics.push(
      makeError('E0405', `Declared entryPoint '${chunk.entryPoint}' not found in function table`)
    );
  }

  // 4. Per-function operand + CFG abstract stack-height verification
  let totalInstructions = 0;
  let globalMaxStackDepth = 0;
  const perFunction: FunctionVerificationReport[] = [];

  for (const fn of chunk.functions) {
    const startDiagCount = diagnostics.length;
    totalInstructions += fn.code.length;

    if (fn.code.length === 0) {
      diagnostics.push(
        makeError('E0413', `Function '${fn.name}' has empty bytecode body without RETURN`)
      );
      perFunction.push({
        name: fn.name,
        instructions: 0,
        reachableInstructions: 0,
        maxStackDepth: 0,
        ok: false,
      });
      continue;
    }

    // 4a. Static instruction & operand verification
    const validOpcodes = getValidOpcodes();
    let operandErrors = 0;
    for (let ip = 0; ip < fn.code.length; ip++) {
      const inst = fn.code[ip];
      if (!inst || !validOpcodes.has(inst.op)) {
        diagnostics.push(
          makeError('E0407', `Unknown opcode '${String(inst?.op)}' in '${fn.name}' at ip=${ip}`)
        );
        operandErrors++;
        continue;
      }

      switch (inst.op) {
        case OpCode.PUSH_CONST: {
          if (!isNonNegativeInt(inst.arg) || inst.arg >= chunk.constants.length) {
            diagnostics.push(
              makeError('E0402', `Constant pool index ${String(inst.arg)} out of bounds in '${fn.name}' at ip=${ip}`)
            );
            operandErrors++;
          }
          break;
        }
        case OpCode.LOAD_LOCAL:
        case OpCode.STORE_LOCAL: {
          if (!isNonNegativeInt(inst.arg) || inst.arg >= fn.localCount) {
            diagnostics.push(
              makeError('E0403', `Local slot ${String(inst.arg)} out of bounds in '${fn.name}' at ip=${ip}`)
            );
            operandErrors++;
          }
          break;
        }
        case OpCode.LOAD_GLOBAL:
        case OpCode.STORE_GLOBAL: {
          if (!isNonEmptyString(inst.arg)) {
            diagnostics.push(
              makeError('E0409', `Missing global variable name in '${fn.name}' at ip=${ip}`)
            );
            operandErrors++;
          }
          break;
        }
        case OpCode.JUMP:
        case OpCode.JUMP_IF_FALSE: {
          if (!isNonNegativeInt(inst.arg) || inst.arg >= fn.code.length) {
            diagnostics.push(
              makeError(
                'E0404',
                `Jump target ${String(inst.arg)} out of bounds [0..${fn.code.length - 1}] in '${fn.name}' at ip=${ip}`
              )
            );
            operandErrors++;
          }
          break;
        }
        case OpCode.CALL: {
          if (!isNonEmptyString(inst.arg)) {
            diagnostics.push(makeError('E0409', `CALL missing callee name in '${fn.name}' at ip=${ip}`));
            operandErrors++;
          } else if (!isNonNegativeInt(inst.arg2)) {
            diagnostics.push(makeError('E0409', `CALL '${inst.arg}' has invalid arg count in '${fn.name}' at ip=${ip}`));
            operandErrors++;
          } else {
            const userTarget = fnByName.get(inst.arg);
            if (userTarget && userTarget.arity !== inst.arg2) {
              diagnostics.push(
                makeError(
                  'E0408',
                  `CALL '${inst.arg}' passes ${inst.arg2} argument(s), expected ${userTarget.arity} in '${fn.name}' at ip=${ip}`
                )
              );
              operandErrors++;
            }
          }
          break;
        }
        case OpCode.CALL_METHOD: {
          if (!isNonEmptyString(inst.arg) || !isNonNegativeInt(inst.arg2)) {
            diagnostics.push(
              makeError('E0409', `Invalid CALL_METHOD operands in '${fn.name}' at ip=${ip}`)
            );
            operandErrors++;
          }
          break;
        }
        case OpCode.ALLOC_STRUCT: {
          if (!isNonEmptyString(inst.arg) || !isNonNegativeInt(inst.arg2)) {
            diagnostics.push(
              makeError('E0409', `Invalid ALLOC_STRUCT operands in '${fn.name}' at ip=${ip}`)
            );
            operandErrors++;
          } else if (!Array.isArray(inst.fieldNames) || inst.fieldNames.length !== inst.arg2) {
            diagnostics.push(
              makeError(
                'E0409',
                `ALLOC_STRUCT '${inst.arg}' fieldNames length (${inst.fieldNames?.length ?? 0}) does not match field count (${inst.arg2}) in '${fn.name}' at ip=${ip}`
              )
            );
            operandErrors++;
          }
          break;
        }
        case OpCode.LOAD_FIELD:
        case OpCode.STORE_FIELD: {
          if (!isNonEmptyString(inst.arg)) {
            diagnostics.push(
              makeError('E0409', `Missing field name operand in '${fn.name}' at ip=${ip}`)
            );
            operandErrors++;
          }
          break;
        }
        case OpCode.ALLOC_ARRAY:
        case OpCode.ALLOC_MAP: {
          if (!isNonNegativeInt(inst.arg)) {
            diagnostics.push(
              makeError('E0409', `Invalid allocation count ${String(inst.arg)} in '${fn.name}' at ip=${ip}`)
            );
            operandErrors++;
          }
          break;
        }
      }
    }

    // 4b. Control-flow graph & abstract stack-depth worklist verification
    let fnMaxStack = 0;
    let reachableCount = 0;

    if (operandErrors === 0) {
      const entryDepth = new Map<number, number>();
      const worklist: Array<{ ip: number; depth: number }> = [{ ip: 0, depth: 0 }];
      entryDepth.set(0, 0);

      const enqueue = (targetIp: number, depth: number, fromIp: number): void => {
        if (targetIp >= fn.code.length) {
          diagnostics.push(
            makeError(
              'E0413',
              `Control flow falls off end of function '${fn.name}' after ip=${fromIp} without RETURN`
            )
          );
          return;
        }
        const prev = entryDepth.get(targetIp);
        if (prev === undefined) {
          entryDepth.set(targetIp, depth);
          worklist.push({ ip: targetIp, depth });
        } else if (prev !== depth) {
          diagnostics.push(
            makeError(
              'E0412',
              `Stack height mismatch at merge point ip=${targetIp} in '${fn.name}': incoming depth ${depth} != expected ${prev}`,
              'All control-flow edges targeting the same instruction must carry an identical operand stack height.'
            )
          );
        }
      };

      while (worklist.length > 0) {
        const { ip, depth } = worklist.pop()!;
        const inst = fn.code[ip];
        const { pops, pushes } = instructionStackEffect(inst);

        if (depth < pops) {
          diagnostics.push(
            makeError(
              'E0410',
              `Operand stack underflow in '${fn.name}' at ip=${ip}: stack depth ${depth} < required ${pops}`
            )
          );
          continue;
        }

        const nextDepth = depth - pops + pushes;
        if (nextDepth > fnMaxStack) fnMaxStack = nextDepth;
        if (nextDepth > MAX_VERIFIED_STACK_DEPTH) {
          diagnostics.push(
            makeError(
              'E0411',
              `Operand stack overflow in '${fn.name}' at ip=${ip}: depth ${nextDepth} exceeds limit ${MAX_VERIFIED_STACK_DEPTH}`
            )
          );
          continue;
        }

        if (inst.op === OpCode.RETURN) {
          // Terminates this control-flow path cleanly.
          continue;
        } else if (inst.op === OpCode.JUMP) {
          enqueue(Number(inst.arg), nextDepth, ip);
        } else if (inst.op === OpCode.JUMP_IF_FALSE) {
          enqueue(Number(inst.arg), nextDepth, ip);
          enqueue(ip + 1, nextDepth, ip);
        } else {
          enqueue(ip + 1, nextDepth, ip);
        }
      }

      reachableCount = entryDepth.size;
    }

    if (fnMaxStack > globalMaxStackDepth) globalMaxStackDepth = fnMaxStack;
    fn.maxStackDepth = fnMaxStack;

    perFunction.push({
      name: fn.name,
      instructions: fn.code.length,
      reachableInstructions: reachableCount,
      maxStackDepth: fnMaxStack,
      ok: diagnostics.length === startDiagCount,
    });
  }

  return {
    ok: diagnostics.length === 0,
    functionsVerified: chunk.functions.length,
    instructionsVerified: totalInstructions,
    maxStackDepth: globalMaxStackDepth,
    checksRun,
    perFunction,
    diagnostics,
  };
}
