import { IRFunction, IRModule } from './ir';

export interface LLVMValidationResult {
  ok: boolean;
  functionsChecked: number;
  blocksChecked: number;
  instructionsChecked: number;
  errors: string[];
}

export interface LLVMCodegenResult {
  supported: boolean;
  targetTriple: string;
  llvmIR?: string;
  validation?: LLVMValidationResult;
  functionCount?: number;
  blockCount?: number;
  reason?: string;
}

const UNSUPPORTED_NATIVE_OPS = new Set<string>([]);
const UNSUPPORTED_NATIVE_BUILTINS = new Set([
  'stringify',
  'parse',
  'split',
]);

export function defaultHostTargetTriple(): string {
  if (typeof process !== 'undefined' && process.arch === 'arm64') {
    return process.platform === 'darwin'
      ? 'arm64-apple-macosx14.0.0'
      : 'aarch64-unknown-linux-gnu';
  }
  return 'x86_64-unknown-linux-gnu';
}

function escapeLLVMString(s: string): { escaped: string; byteLen: number } {
  const bytes = new TextEncoder().encode(s);
  let out = '';
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    if (b >= 32 && b <= 126 && b !== 34 && b !== 92) {
      out += String.fromCharCode(b);
    } else {
      out += `\\${b.toString(16).toUpperCase().padStart(2, '0')}`;
    }
  }
  out += '\\00';
  return { escaped: out, byteLen: bytes.length + 1 };
}

function formatLLVMFloat(v: number): string {
  if (!Number.isFinite(v)) return '0.0';
  const s = v.toString();
  if (s.includes('.') || s.includes('e') || s.includes('E')) return s;
  return `${s}.0`;
}

/**
 * Lowers Nova Intermediate Representation (IR) into genuine LLVM 17+ opaque-pointer IR (`.ll`)
 * targeting the `libnovart` C ABI (`%struct.NVal = type { i32, i64, double, i1, ptr }`).
 *
 * The emitted `.ll` module can be compiled directly by `clang` / `llc` and linked with
 * `libnovart.o` + `-lm` into a standalone native executable.
 */
export function lowerToLLVMIR(
  mod: IRModule,
  options: { targetTriple?: string; moduleName?: string } = {}
): LLVMCodegenResult {
  const targetTriple = options.targetTriple ?? defaultHostTargetTriple();
  const moduleName = options.moduleName ?? 'nova_module';

  for (const fn of mod.functions) {
    for (const b of fn.blocks) {
      for (const inst of b.instructions) {
        if (UNSUPPORTED_NATIVE_OPS.has(inst.op)) {
          return {
            supported: false,
            targetTriple,
            reason: `Tensor IR instruction '${inst.op}' executes on the Nova VM + Nova AI Engine.`,
          };
        }
        if (inst.op === 'CALL') {
          const base = inst.callee.includes('.') ? inst.callee.split('.').pop()! : inst.callee;
          if (UNSUPPORTED_NATIVE_BUILTINS.has(base)) {
            return {
              supported: false,
              targetTriple,
              reason: `AI/JSON builtin '${inst.callee}' executes on the Nova VM + Nova AI Engine.`,
            };
          }
        }
      }
    }
  }

  const sanitize = (name: string): string => name.replace(/[^a-zA-Z0-9_]/g, '_');
  const userFns = new Set(mod.functions.map((f) => f.name));
  const globals = new Set<string>();
  const stringPool: Array<{ symbol: string; escaped: string; byteLen: number }> = [];
  const stringMap = new Map<string, { symbol: string; byteLen: number }>();

  const internString = (s: string): { symbol: string; byteLen: number } => {
    const hit = stringMap.get(s);
    if (hit) return hit;
    const { escaped, byteLen } = escapeLLVMString(s);
    const symbol = `@.str.${stringPool.length}`;
    stringPool.push({ symbol, escaped, byteLen });
    const info = { symbol, byteLen };
    stringMap.set(s, info);
    return info;
  };

  for (const fn of mod.functions) {
    for (const b of fn.blocks) {
      for (const inst of b.instructions) {
        if (inst.op === 'LOAD_GLOBAL' || inst.op === 'STORE_GLOBAL') {
          globals.add(inst.name);
        }
      }
    }
  }

  const fnBodies: string[] = [];
  let totalBlocks = 0;

  const lowerFn = (fn: IRFunction): string => {
    let regId = 0;
    const nextReg = (): string => `%r${regId++}`;

    const temps = new Set<string>();
    const slots = new Set<string>();
    for (const b of fn.blocks) {
      for (const inst of b.instructions) {
        if ('dst' in inst && inst.dst) temps.add(inst.dst);
        if (inst.op === 'STORE' || inst.op === 'LOAD') {
          if (!fn.params.includes(inst.slot)) slots.add(inst.slot);
        }
      }
    }

    const tempPtr = (t: string): string => `%tmp_${sanitize(t)}`;
    const slotPtr = (s: string): string => `%var_${sanitize(s)}`;

    const paramsSig = fn.params.map((p) => `%struct.NVal %arg_${sanitize(p)}`).join(', ');
    const out: string[] = [];
    out.push(`define internal %struct.NVal @nova_fn_${sanitize(fn.name)}(${paramsSig}) {`);
    out.push('alloc_entry:');
    totalBlocks++;

    for (const p of fn.params) {
      out.push(`  ${slotPtr(p)} = alloca %struct.NVal, align 8`);
      out.push(`  store %struct.NVal %arg_${sanitize(p)}, ptr ${slotPtr(p)}, align 8`);
    }
    for (const s of slots) {
      out.push(`  ${slotPtr(s)} = alloca %struct.NVal, align 8`);
      const z = nextReg();
      out.push(`  ${z} = call %struct.NVal @nv_none()`);
      out.push(`  store %struct.NVal ${z}, ptr ${slotPtr(s)}, align 8`);
    }
    for (const t of temps) {
      out.push(`  ${tempPtr(t)} = alloca %struct.NVal, align 8`);
    }

    const firstLabel = fn.blocks[0] ? `lbl_${sanitize(fn.blocks[0].label)}` : 'fn_exit';
    out.push(`  br label %${firstLabel}`);

    const loadTemp = (t: string): string => {
      const r = nextReg();
      out.push(`  ${r} = load %struct.NVal, ptr ${tempPtr(t)}, align 8`);
      return r;
    };

    const storeTemp = (t: string, reg: string): void => {
      out.push(`  store %struct.NVal ${reg}, ptr ${tempPtr(t)}, align 8`);
    };

    const emitStrPtr = (s: string): string => {
      const info = internString(s);
      const r = nextReg();
      out.push(`  ${r} = getelementptr inbounds [${info.byteLen} x i8], ptr ${info.symbol}, i64 0, i64 0`);
      return r;
    };

    fn.blocks.forEach((block, bIdx) => {
      totalBlocks++;
      out.push(`lbl_${sanitize(block.label)}:`);
      let terminated = false;

      for (const inst of block.instructions) {
        if (terminated) break;

        switch (inst.op) {
          case 'CONST': {
            const r = nextReg();
            if (inst.value.kind === 'Int') {
              out.push(`  ${r} = call %struct.NVal @nv_int(i64 ${Math.trunc(inst.value.value)})`);
            } else if (inst.value.kind === 'Float') {
              out.push(`  ${r} = call %struct.NVal @nv_float(double ${formatLLVMFloat(inst.value.value)})`);
            } else if (inst.value.kind === 'Bool') {
              out.push(`  ${r} = call %struct.NVal @nv_bool(i1 ${inst.value.value ? 'true' : 'false'})`);
            } else if (inst.value.kind === 'String') {
              const p = emitStrPtr(inst.value.value);
              out.push(`  ${r} = call %struct.NVal @nv_str(ptr ${p})`);
            } else if (inst.value.kind === 'Char') {
              const p = emitStrPtr(inst.value.value);
              out.push(`  ${r} = call %struct.NVal @nv_char(ptr ${p})`);
            } else {
              out.push(`  ${r} = call %struct.NVal @nv_none()`);
            }
            storeTemp(inst.dst, r);
            break;
          }

          case 'LOAD': {
            const r = nextReg();
            out.push(`  ${r} = load %struct.NVal, ptr ${slotPtr(inst.slot)}, align 8`);
            storeTemp(inst.dst, r);
            break;
          }

          case 'STORE': {
            const v = loadTemp(inst.src);
            out.push(`  store %struct.NVal ${v}, ptr ${slotPtr(inst.slot)}, align 8`);
            break;
          }

          case 'LOAD_GLOBAL': {
            const r = nextReg();
            out.push(`  ${r} = load %struct.NVal, ptr @g_${sanitize(inst.name)}, align 8`);
            storeTemp(inst.dst, r);
            break;
          }

          case 'STORE_GLOBAL': {
            const v = loadTemp(inst.src);
            out.push(`  store %struct.NVal ${v}, ptr @g_${sanitize(inst.name)}, align 8`);
            break;
          }

          case 'ADD':
          case 'SUB':
          case 'MUL':
          case 'DIV':
          case 'MOD':
          case 'MATMUL': {
            const l = loadTemp(inst.left);
            const r = loadTemp(inst.right);
            const res = nextReg();
            const fnName = `@nv_${inst.op.toLowerCase()}`;
            out.push(`  ${res} = call %struct.NVal ${fnName}(%struct.NVal ${l}, %struct.NVal ${r})`);
            storeTemp(inst.dst, res);
            break;
          }

          case 'NEG':
          case 'NOT': {
            const v = loadTemp(inst.src);
            const res = nextReg();
            out.push(`  ${res} = call %struct.NVal @nv_${inst.op.toLowerCase()}(%struct.NVal ${v})`);
            storeTemp(inst.dst, res);
            break;
          }

          case 'AND':
          case 'OR': {
            const l = loadTemp(inst.left);
            const r = loadTemp(inst.right);
            const bl = nextReg();
            const br = nextReg();
            const bRes = nextReg();
            const res = nextReg();
            out.push(`  ${bl} = call i1 @nv_truthy(%struct.NVal ${l})`);
            out.push(`  ${br} = call i1 @nv_truthy(%struct.NVal ${r})`);
            out.push(`  ${bRes} = ${inst.op === 'AND' ? 'and' : 'or'} i1 ${bl}, ${br}`);
            out.push(`  ${res} = call %struct.NVal @nv_bool(i1 ${bRes})`);
            storeTemp(inst.dst, res);
            break;
          }

          case 'COMPARE': {
            const cmpFn: Record<string, string> = {
              '==': '@nv_cmp_eq',
              '!=': '@nv_cmp_neq',
              '<': '@nv_cmp_lt',
              '<=': '@nv_cmp_lte',
              '>': '@nv_cmp_gt',
              '>=': '@nv_cmp_gte',
            };
            const l = loadTemp(inst.left);
            const r = loadTemp(inst.right);
            const res = nextReg();
            out.push(`  ${res} = call %struct.NVal ${cmpFn[inst.cmp]}(%struct.NVal ${l}, %struct.NVal ${r})`);
            storeTemp(inst.dst, res);
            break;
          }

          case 'CALL': {
            const argRegs = inst.args.map(loadTemp);
            const res = nextReg();
            if (userFns.has(inst.callee)) {
              const argsList = argRegs.map((a) => `%struct.NVal ${a}`).join(', ');
              out.push(`  ${res} = call %struct.NVal @nova_fn_${sanitize(inst.callee)}(${argsList})`);
            } else {
              const namePtr = emitStrPtr(inst.callee);
              let arrPtr = 'null';
              if (argRegs.length > 0) {
                const arrAlloca = nextReg();
                out.push(`  ${arrAlloca} = alloca [${argRegs.length} x %struct.NVal], align 8`);
                argRegs.forEach((a, idx) => {
                  const elemPtr = nextReg();
                  out.push(
                    `  ${elemPtr} = getelementptr inbounds [${argRegs.length} x %struct.NVal], ptr ${arrAlloca}, i64 0, i64 ${idx}`
                  );
                  out.push(`  store %struct.NVal ${a}, ptr ${elemPtr}, align 8`);
                });
                const decay = nextReg();
                out.push(
                  `  ${decay} = getelementptr inbounds [${argRegs.length} x %struct.NVal], ptr ${arrAlloca}, i64 0, i64 0`
                );
                arrPtr = decay;
              }
              out.push(
                `  ${res} = call %struct.NVal @nv_call_builtin(ptr ${namePtr}, i32 ${argRegs.length}, ptr ${arrPtr})`
              );
            }
            storeTemp(inst.dst, res);
            break;
          }

          case 'CALL_METHOD': {
            const obj = loadTemp(inst.object);
            const argRegs = inst.args.map(loadTemp);
            const methodPtr = emitStrPtr(inst.method);
            let arrPtr = 'null';
            if (argRegs.length > 0) {
              const arrAlloca = nextReg();
              out.push(`  ${arrAlloca} = alloca [${argRegs.length} x %struct.NVal], align 8`);
              argRegs.forEach((a, idx) => {
                const elemPtr = nextReg();
                out.push(
                  `  ${elemPtr} = getelementptr inbounds [${argRegs.length} x %struct.NVal], ptr ${arrAlloca}, i64 0, i64 ${idx}`
                );
                out.push(`  store %struct.NVal ${a}, ptr ${elemPtr}, align 8`);
              });
              const decay = nextReg();
              out.push(
                `  ${decay} = getelementptr inbounds [${argRegs.length} x %struct.NVal], ptr ${arrAlloca}, i64 0, i64 0`
              );
              arrPtr = decay;
            }
            const res = nextReg();
            out.push(
              `  ${res} = call %struct.NVal @nv_method_call(%struct.NVal ${obj}, ptr ${methodPtr}, i32 ${argRegs.length}, ptr ${arrPtr})`
            );
            storeTemp(inst.dst, res);
            break;
          }

          case 'ALLOC_STRUCT': {
            const typePtr = emitStrPtr(inst.typeName);
            const n = inst.fields.length;
            let namesPtr = 'null';
            let valsPtr = 'null';
            if (n > 0) {
              const namesAlloca = nextReg();
              const valsAlloca = nextReg();
              out.push(`  ${namesAlloca} = alloca [${n} x ptr], align 8`);
              out.push(`  ${valsAlloca} = alloca [${n} x %struct.NVal], align 8`);
              inst.fields.forEach((f, idx) => {
                const fnPtr = emitStrPtr(f.name);
                const fv = loadTemp(f.src);
                const np = nextReg();
                const vp = nextReg();
                out.push(`  ${np} = getelementptr inbounds [${n} x ptr], ptr ${namesAlloca}, i64 0, i64 ${idx}`);
                out.push(`  store ptr ${fnPtr}, ptr ${np}, align 8`);
                out.push(`  ${vp} = getelementptr inbounds [${n} x %struct.NVal], ptr ${valsAlloca}, i64 0, i64 ${idx}`);
                out.push(`  store %struct.NVal ${fv}, ptr ${vp}, align 8`);
              });
              const nd = nextReg();
              const vd = nextReg();
              out.push(`  ${nd} = getelementptr inbounds [${n} x ptr], ptr ${namesAlloca}, i64 0, i64 0`);
              out.push(`  ${vd} = getelementptr inbounds [${n} x %struct.NVal], ptr ${valsAlloca}, i64 0, i64 0`);
              namesPtr = nd;
              valsPtr = vd;
            }
            const res = nextReg();
            out.push(
              `  ${res} = call %struct.NVal @nv_alloc_struct(ptr ${typePtr}, i32 ${n}, ptr ${namesPtr}, ptr ${valsPtr})`
            );
            storeTemp(inst.dst, res);
            break;
          }

          case 'LOAD_FIELD': {
            const obj = loadTemp(inst.object);
            const fPtr = emitStrPtr(inst.field);
            const res = nextReg();
            out.push(`  ${res} = call %struct.NVal @nv_load_field(%struct.NVal ${obj}, ptr ${fPtr})`);
            storeTemp(inst.dst, res);
            break;
          }

          case 'STORE_FIELD': {
            const obj = loadTemp(inst.object);
            const val = loadTemp(inst.src);
            const fPtr = emitStrPtr(inst.field);
            out.push(`  call void @nv_store_field(%struct.NVal ${obj}, ptr ${fPtr}, %struct.NVal ${val})`);
            break;
          }

          case 'ALLOC_ARRAY': {
            const n = inst.elements.length;
            let valsPtr = 'null';
            if (n > 0) {
              const arrAlloca = nextReg();
              out.push(`  ${arrAlloca} = alloca [${n} x %struct.NVal], align 8`);
              inst.elements.forEach((el, idx) => {
                const v = loadTemp(el);
                const ep = nextReg();
                out.push(`  ${ep} = getelementptr inbounds [${n} x %struct.NVal], ptr ${arrAlloca}, i64 0, i64 ${idx}`);
                out.push(`  store %struct.NVal ${v}, ptr ${ep}, align 8`);
              });
              const d = nextReg();
              out.push(`  ${d} = getelementptr inbounds [${n} x %struct.NVal], ptr ${arrAlloca}, i64 0, i64 0`);
              valsPtr = d;
            }
            const res = nextReg();
            out.push(`  ${res} = call %struct.NVal @nv_alloc_array(i32 ${n}, ptr ${valsPtr})`);
            storeTemp(inst.dst, res);
            break;
          }

          case 'ALLOC_MAP': {
            const n = inst.entries.length;
            let keysPtr = 'null';
            let valsPtr = 'null';
            if (n > 0) {
              const kAlloca = nextReg();
              const vAlloca = nextReg();
              out.push(`  ${kAlloca} = alloca [${n} x %struct.NVal], align 8`);
              out.push(`  ${vAlloca} = alloca [${n} x %struct.NVal], align 8`);
              inst.entries.forEach((e, idx) => {
                const kv = loadTemp(e.key);
                const vv = loadTemp(e.value);
                const kp = nextReg();
                const vp = nextReg();
                out.push(`  ${kp} = getelementptr inbounds [${n} x %struct.NVal], ptr ${kAlloca}, i64 0, i64 ${idx}`);
                out.push(`  store %struct.NVal ${kv}, ptr ${kp}, align 8`);
                out.push(`  ${vp} = getelementptr inbounds [${n} x %struct.NVal], ptr ${vAlloca}, i64 0, i64 ${idx}`);
                out.push(`  store %struct.NVal ${vv}, ptr ${vp}, align 8`);
              });
              const kd = nextReg();
              const vd = nextReg();
              out.push(`  ${kd} = getelementptr inbounds [${n} x %struct.NVal], ptr ${kAlloca}, i64 0, i64 0`);
              out.push(`  ${vd} = getelementptr inbounds [${n} x %struct.NVal], ptr ${vAlloca}, i64 0, i64 0`);
              keysPtr = kd;
              valsPtr = vd;
            }
            const res = nextReg();
            out.push(`  ${res} = call %struct.NVal @nv_alloc_map(i32 ${n}, ptr ${keysPtr}, ptr ${valsPtr})`);
            storeTemp(inst.dst, res);
            break;
          }

          case 'INDEX_GET': {
            const obj = loadTemp(inst.object);
            const idx = loadTemp(inst.index);
            const res = nextReg();
            out.push(`  ${res} = call %struct.NVal @nv_index_get(%struct.NVal ${obj}, %struct.NVal ${idx})`);
            storeTemp(inst.dst, res);
            break;
          }

          case 'INDEX_SET': {
            const obj = loadTemp(inst.object);
            const idx = loadTemp(inst.index);
            const val = loadTemp(inst.src);
            out.push(`  call void @nv_index_set(%struct.NVal ${obj}, %struct.NVal ${idx}, %struct.NVal ${val})`);
            break;
          }

          case 'MAKE_RESULT': {
            const val = loadTemp(inst.src);
            const res = nextReg();
            const callee = inst.variant === 'Ok' ? '@nv_make_ok' : '@nv_make_error';
            out.push(`  ${res} = call %struct.NVal ${callee}(%struct.NVal ${val})`);
            storeTemp(inst.dst, res);
            break;
          }

          case 'JUMP': {
            out.push(`  br label %lbl_${sanitize(inst.target)}`);
            terminated = true;
            break;
          }

          case 'JUMP_IF_FALSE': {
            const cond = loadTemp(inst.cond);
            const b = nextReg();
            const fallthroughLabel = `ft_${sanitize(block.label)}_${regId++}`;
            out.push(`  ${b} = call i1 @nv_truthy(%struct.NVal ${cond})`);
            out.push(`  br i1 ${b}, label %${fallthroughLabel}, label %lbl_${sanitize(inst.target)}`);
            out.push(`${fallthroughLabel}:`);
            totalBlocks++;
            break;
          }

          case 'RETURN': {
            if (inst.src) {
              const v = loadTemp(inst.src);
              out.push(`  ret %struct.NVal ${v}`);
            } else {
              const v = nextReg();
              out.push(`  ${v} = call %struct.NVal @nv_none()`);
              out.push(`  ret %struct.NVal ${v}`);
            }
            terminated = true;
            break;
          }
        }
      }

      if (!terminated) {
        const nextBlock = fn.blocks[bIdx + 1];
        if (nextBlock) {
          out.push(`  br label %lbl_${sanitize(nextBlock.label)}`);
        } else {
          const v = nextReg();
          out.push(`  ${v} = call %struct.NVal @nv_none()`);
          out.push(`  ret %struct.NVal ${v}`);
        }
      }
    });

    out.push('}');
    return out.join('\n');
  };

  for (const fn of mod.functions) {
    fnBodies.push(lowerFn(fn));
  }

  // Emit complete LLVM IR module
  const header: string[] = [
    `; ModuleID = '${moduleName}'`,
    `source_filename = "${moduleName}"`,
    `target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"`,
    `target triple = "${targetTriple}"`,
    '',
    `; Nova Tagged Value ABI (matches libnovart novart.h)`,
    `%struct.NVal = type { i32, i64, double, i1, ptr }`,
    '',
    `; libnovart Runtime Declarations`,
    'declare %struct.NVal @nv_int(i64)',
    'declare %struct.NVal @nv_float(double)',
    'declare %struct.NVal @nv_bool(i1)',
    'declare %struct.NVal @nv_char(ptr)',
    'declare %struct.NVal @nv_str(ptr)',
    'declare %struct.NVal @nv_none()',
    'declare i1 @nv_truthy(%struct.NVal)',
    'declare %struct.NVal @nv_add(%struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_sub(%struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_mul(%struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_div(%struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_mod(%struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_matmul(%struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_neg(%struct.NVal)',
    'declare %struct.NVal @nv_not(%struct.NVal)',
    'declare %struct.NVal @nv_cmp_eq(%struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_cmp_neq(%struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_cmp_lt(%struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_cmp_lte(%struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_cmp_gt(%struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_cmp_gte(%struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_alloc_struct(ptr, i32, ptr, ptr)',
    'declare %struct.NVal @nv_load_field(%struct.NVal, ptr)',
    'declare void @nv_store_field(%struct.NVal, ptr, %struct.NVal)',
    'declare %struct.NVal @nv_alloc_array(i32, ptr)',
    'declare %struct.NVal @nv_alloc_map(i32, ptr, ptr)',
    'declare %struct.NVal @nv_index_get(%struct.NVal, %struct.NVal)',
    'declare void @nv_index_set(%struct.NVal, %struct.NVal, %struct.NVal)',
    'declare %struct.NVal @nv_make_ok(%struct.NVal)',
    'declare %struct.NVal @nv_make_error(%struct.NVal)',
    'declare %struct.NVal @nv_call_builtin(ptr, i32, ptr)',
    'declare %struct.NVal @nv_method_call(%struct.NVal, ptr, i32, ptr)',
    '',
  ];

  if (stringPool.length > 0) {
    header.push('; Global String Constants');
    for (const s of stringPool) {
      header.push(`${s.symbol} = private unnamed_addr constant [${s.byteLen} x i8] c"${s.escaped}", align 1`);
    }
    header.push('');
  }

  if (globals.size > 0) {
    header.push('; Global Nova Variables');
    for (const g of globals) {
      header.push(`@g_${sanitize(g)} = internal global %struct.NVal zeroinitializer, align 8`);
    }
    header.push('');
  }

  const mainWrapper: string[] = [
    '; Native OS Entry Point',
    'define i32 @main() {',
    'entry:',
  ];
  if (userFns.has('__init__')) {
    mainWrapper.push('  %init_res = call %struct.NVal @nova_fn___init__()');
  }
  if (userFns.has('main')) {
    mainWrapper.push('  %main_res = call %struct.NVal @nova_fn_main()');
  }
  mainWrapper.push('  ret i32 0');
  mainWrapper.push('}');
  totalBlocks++;

  const llvmIR = [...header, ...fnBodies.join('\n\n').split('\n'), '', ...mainWrapper].join('\n');
  const validation = validateLLVMIR(llvmIR);

  return {
    supported: true,
    targetTriple,
    llvmIR,
    validation,
    functionCount: mod.functions.length + 1,
    blockCount: totalBlocks,
  };
}

/**
 * Static LLVM IR structural verifier.
 * Checks target triple, SSA register uniqueness per function, block terminators,
 * and branch label resolution.
 */
export function validateLLVMIR(llvmIR: string): LLVMValidationResult {
  const errors: string[] = [];
  const lines = llvmIR.split('\n');

  if (!lines.some((l) => l.startsWith('target triple = '))) {
    errors.push('Missing LLVM target triple declaration');
  }

  let functionsChecked = 0;
  let blocksChecked = 0;
  let instructionsChecked = 0;

  let inFn: string | null = null;
  let definedLabels = new Set<string>();
  let referencedLabels = new Set<string>();
  let definedRegs = new Set<string>();
  let currentBlock: string | null = null;
  let blockTerminated = false;

  const finishBlock = () => {
    if (inFn && currentBlock && !blockTerminated) {
      errors.push(`LLVM block '${currentBlock}' in ${inFn} does not end with a terminator`);
    }
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith(';')) continue;

    if (line.startsWith('define ')) {
      const m = line.match(/@([a-zA-Z0-9_]+)/);
      inFn = m ? `@${m[1]}` : '@unknown';
      functionsChecked++;
      definedLabels = new Set<string>();
      referencedLabels = new Set<string>();
      definedRegs = new Set<string>();
      currentBlock = null;
      blockTerminated = false;
      continue;
    }

    if (inFn && line === '}') {
      finishBlock();
      for (const target of referencedLabels) {
        if (!definedLabels.has(target)) {
          errors.push(`LLVM branch target '%${target}' not defined in ${inFn}`);
        }
      }
      inFn = null;
      currentBlock = null;
      continue;
    }

    if (!inFn) continue;

    if (/^[a-zA-Z0-9_.]+:$/.test(line)) {
      finishBlock();
      currentBlock = line.slice(0, -1);
      if (definedLabels.has(currentBlock)) {
        errors.push(`Duplicate LLVM block label '${currentBlock}' in ${inFn}`);
      }
      definedLabels.add(currentBlock);
      blockTerminated = false;
      blocksChecked++;
      continue;
    }

    instructionsChecked++;

    const assignMatch = line.match(/^(%[a-zA-Z0-9_]+)\s*=/);
    if (assignMatch) {
      const reg = assignMatch[1];
      if (definedRegs.has(reg)) {
        errors.push(`SSA violation: register '${reg}' redefined in ${inFn}`);
      }
      definedRegs.add(reg);
    }

    if (line.startsWith('ret ') || line === 'unreachable') {
      blockTerminated = true;
    } else if (line.startsWith('br ')) {
      blockTerminated = true;
      const lblMatches = line.matchAll(/label\s+%([a-zA-Z0-9_.]+)/g);
      for (const lm of lblMatches) {
        referencedLabels.add(lm[1]);
      }
    }
  }

  return {
    ok: errors.length === 0,
    functionsChecked,
    blocksChecked,
    instructionsChecked,
    errors,
  };
}
