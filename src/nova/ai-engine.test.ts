import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import {
  AdamOptimizer,
  bceLoss,
  compileAndRun,
  GradVar,
  lowerToLLVMIR,
  mseLoss,
  NovaTensor,
  SequentialModel,
  SGDOptimizer,
  trainModel,
  trainXorNetwork,
  validateLLVMIR,
} from './index';
import { buildNativeExecutable, executeNativeBinary } from './native-builder';

function approxEq(a: number, b: number, tol = 1e-5, label = ''): void {
  assert.ok(
    Math.abs(a - b) <= tol,
    `${label ? label + ': ' : ''}expected ${a} ≈ ${b} (diff=${Math.abs(a - b)}, tol=${tol})`
  );
}

test('1. CPU Tensor Engine: constructors, broadcasting, matmul, dot, transpose, reshape, reductions & activations', () => {
  const a = NovaTensor.fromNested([
    [1, 2],
    [3, 4],
  ]);
  const b = NovaTensor.fromNested([
    [5, 6],
    [7, 8],
  ]);

  const prod = a.matmul(b);
  assert.deepEqual(prod.shape, [2, 2]);
  assert.deepEqual(Array.from(prod.data), [19, 22, 43, 50]);

  // Row broadcasting [2, 2] + [1, 2]
  const bias = NovaTensor.fromNested([[10, -10]]);
  const shifted = a.add(bias);
  assert.deepEqual(Array.from(shifted.data), [11, -8, 13, -6]);

  // Dot product, transpose, reshape
  const u = NovaTensor.fromNested([1, 2, 3]);
  const v = NovaTensor.fromNested([4, 5, 6]);
  approxEq(u.dot(v), 32, 1e-12, 'dot');

  const tr = a.transpose();
  assert.deepEqual(tr.shape, [2, 2]);
  assert.deepEqual(Array.from(tr.data), [1, 3, 2, 4]);

  const r = a.reshape([4, 1]);
  assert.deepEqual(r.shape, [4, 1]);
  assert.deepEqual(Array.from(r.data), [1, 2, 3, 4]);

  // Reductions
  approxEq(a.sum(), 10, 1e-12);
  approxEq(a.mean(), 2.5, 1e-12);
  approxEq(a.max(), 4, 1e-12);
  approxEq(a.min(), 1, 1e-12);
  assert.equal(a.argmax(), 3);

  // Activations & Losses
  const sm = NovaTensor.fromNested([2.0, 1.0, 0.1]).softmax();
  approxEq(sm.sum(), 1.0, 1e-12, 'softmax sum');
  assert.equal(sm.argmax(), 0);

  const th = NovaTensor.fromNested([-1, 0, 1]).tanh();
  approxEq(th.data[1], 0, 1e-12);
  approxEq(th.data[0], -th.data[2], 1e-12);

  const pred = NovaTensor.fromNested([[0.9], [0.1]]);
  const target = NovaTensor.fromNested([[1.0], [0.0]]);
  approxEq(mseLoss(pred, target), 0.01, 1e-12, 'mseLoss');
  assert.ok(bceLoss(pred, target) < 0.15, 'bceLoss low for accurate predictions');

  // Deterministic randUniform
  const r1 = NovaTensor.randUniform([2, 3], 42, 0.5);
  const r2 = NovaTensor.randUniform([2, 3], 42, 0.5);
  assert.deepEqual(Array.from(r1.data), Array.from(r2.data));
});

test('2. Reverse-mode Tensor Autograd: finite-difference gradient check (< 1e-5) for matmul + row-broadcast + tanh/relu/sigmoid + MSE/BCE', () => {
  const xData = [1.0, -2.0, 0.5, 1.5];
  const wData = [0.4, -0.7, 1.2, 0.3];
  const bData = [0.25, -0.15];
  const yTarget = new NovaTensor([2, 2], [0.8, 0.2, 0.1, 0.9]);

  // Analytic gradients for tanh + MSE
  const x = GradVar.leaf(new NovaTensor([2, 2], xData));
  const w = GradVar.leaf(new NovaTensor([2, 2], wData));
  const b = GradVar.leaf(new NovaTensor([1, 2], bData));
  const loss = x.matmul(w).add(b).tanh().mseLoss(yTarget);
  loss.backward();

  const evalLoss = (xd: number[], wd: number[], bd: number[]): number => {
    const xt = new NovaTensor([2, 2], xd);
    const wt = new NovaTensor([2, 2], wd);
    const bt = new NovaTensor([1, 2], bd);
    return mseLoss(xt.matmul(wt).add(bt).tanh(), yTarget);
  };

  const eps = 1e-6;
  for (let i = 0; i < wData.length; i++) {
    const wp = [...wData];
    const wm = [...wData];
    wp[i] += eps;
    wm[i] -= eps;
    const numeric = (evalLoss(xData, wp, bData) - evalLoss(xData, wm, bData)) / (2 * eps);
    approxEq(w.grad.data[i], numeric, 1e-5, `dL/dW[${i}]`);
  }

  for (let i = 0; i < bData.length; i++) {
    const bp = [...bData];
    const bm = [...bData];
    bp[i] += eps;
    bm[i] -= eps;
    const numeric = (evalLoss(xData, wData, bp) - evalLoss(xData, wData, bm)) / (2 * eps);
    approxEq(b.grad.data[i], numeric, 1e-5, `dL/db[${i}]`);
  }

  for (let i = 0; i < xData.length; i++) {
    const xp = [...xData];
    const xm = [...xData];
    xp[i] += eps;
    xm[i] -= eps;
    const numeric = (evalLoss(xp, wData, bData) - evalLoss(xm, wData, bData)) / (2 * eps);
    approxEq(x.grad.data[i], numeric, 1e-5, `dL/dX[${i}]`);
  }

  // Analytic gradient check for sigmoid + BCE
  const sx = GradVar.leaf(new NovaTensor([2, 1], [0.6, -0.8]));
  const sy = new NovaTensor([2, 1], [1.0, 0.0]);
  const bce = sx.sigmoid().bceLoss(sy);
  bce.backward();
  for (let i = 0; i < 2; i++) {
    const p = [0.6, -0.8];
    const m = [0.6, -0.8];
    p[i] += eps;
    m[i] -= eps;
    const lp = bceLoss(new NovaTensor([2, 1], p).sigmoid(), sy);
    const lm = bceLoss(new NovaTensor([2, 1], m).sigmoid(), sy);
    approxEq(sx.grad.data[i], (lp - lm) / (2 * eps), 1e-5, `dBCE/dx[${i}]`);
  }
});

test('3. Neural Network Layers & Optimizers: SGD converges on linear regression and Adam converges on XOR', () => {
  // Linear regression y = 2x + 1 with SGD + momentum
  const linX = new NovaTensor([4, 1], [0, 1, 2, 3]);
  const linY = new NovaTensor([4, 1], [1, 3, 5, 7]);
  const linModel = new SequentialModel().addLinear(1, 1, 7);
  assert.equal(linModel.paramCount(), 2);
  const sgd = new SGDOptimizer(0.05, 0.8);
  const linReport = trainModel(linModel, sgd, linX, linY, 150, 'mse');
  assert.ok(linReport.finalLoss < 1e-4, `SGD linear regression did not converge: ${linReport.finalLoss}`);
  approxEq(linReport.predictions[0], 1.0, 0.05, 'y(0)');
  approxEq(linReport.predictions[3], 7.0, 0.05, 'y(3)');

  // Non-linear XOR with 2-layer Sequential + Adam
  const xorX = new NovaTensor([4, 2], [0, 0, 0, 1, 1, 0, 1, 1]);
  const xorY = new NovaTensor([4, 1], [0, 1, 1, 0]);
  const xorModel = new SequentialModel()
    .addLinear(2, 4, 42)
    .addActivation('tanh')
    .addLinear(4, 1, 137)
    .addActivation('sigmoid');
  assert.equal(xorModel.paramCount(), 2 * 4 + 4 + 4 * 1 + 1);
  const adam = new AdamOptimizer(0.05);
  const xorReport = trainModel(xorModel, adam, xorX, xorY, 400, 'mse');
  assert.ok(xorReport.finalLoss < 0.02, `Adam XOR did not converge: ${xorReport.finalLoss}`);
  assert.ok(xorReport.predictions[0] < 0.2, `XOR(0,0) expected <0.2, got ${xorReport.predictions[0]}`);
  assert.ok(xorReport.predictions[1] > 0.8, `XOR(0,1) expected >0.8, got ${xorReport.predictions[1]}`);
  assert.ok(xorReport.predictions[2] > 0.8, `XOR(1,0) expected >0.8, got ${xorReport.predictions[2]}`);
  assert.ok(xorReport.predictions[3] < 0.2, `XOR(1,1) expected <0.2, got ${xorReport.predictions[3]}`);

  // High-level trainXorNetwork wrapper
  const summary = trainXorNetwork(500, 0.05);
  assert.ok(summary.finalLoss < 0.02);
});

function assertThreeWayEquivalence(
  label: string,
  source: string,
  profile: 'debug' | 'release' = 'debug'
): { stdout: string[]; llvmIR: string } {
  const interp = compileAndRun(source, { filename: `${label}.nova`, engine: 'interpreter', execute: true });
  assert.equal(interp.ok, true, `${label} interpreter errors: ${interp.formattedDiagnostics.join('\n')}`);

  const vm = compileAndRun(source, { filename: `${label}.nova`, engine: 'vm', execute: true });
  assert.equal(vm.ok, true, `${label} VM errors: ${vm.formattedDiagnostics.join('\n')}`);
  assert.deepEqual(vm.execution!.stdout, interp.execution!.stdout, `${label}: VM != Interpreter`);

  // Verify LLVM IR lowers cleanly and passes structural verification
  const llvmRes = lowerToLLVMIR(vm.irOptimized!, { moduleName: `${label}.nova` });
  assert.equal(llvmRes.supported, true, `${label} LLVM lowering failed: ${llvmRes.reason}`);
  const val = validateLLVMIR(llvmRes.llvmIR!);
  assert.equal(val.ok, true, `${label} LLVM IR validation errors: ${val.errors.join('; ')}`);

  // Compile and execute real native ELF binary linked with libnovart
  const built = buildNativeExecutable(vm.irOptimized!, { profile, moduleName: `${label}.nova` });
  assert.equal(built.ok, true, `${label} native build failed: ${built.reason}`);
  const native = executeNativeBinary(built.outputPath!);
  fs.rmSync(path.dirname(built.outputPath!), { recursive: true, force: true });
  assert.equal(native.exitCode, 0, `${label} native exit code ${native.exitCode}: ${native.stderr}`);
  assert.deepEqual(
    native.stdout,
    vm.execution!.stdout,
    `${label} (${profile}): Native ELF != VM\nNative:\n${native.stdout.join('\n')}\nVM:\n${vm.execution!.stdout.join('\n')}`
  );

  return { stdout: vm.execution!.stdout, llvmIR: llvmRes.llvmIR! };
}

test('4. 3-Way Execution Equivalence (Interpreter == Nova VM == Native ELF): tensors.nova (with @nv_matmul)', () => {
  const src = fs.readFileSync(path.resolve('nova/examples/tensors.nova'), 'utf8');
  const { stdout, llvmIR } = assertThreeWayEquivalence('tensors', src, 'debug');
  assert.ok(llvmIR.includes('@nv_matmul'), 'Expected LLVM IR to call @nv_matmul for tensor @ operator');
  assert.ok(stdout[0].includes('Tensor([[19, 22], [43, 50]], shape=[2, 2])'));
  assert.ok(stdout.some((l) => l.includes('"output": 7.5') && l.includes('"dx": 2')));
});

test('5. 3-Way Execution Equivalence (Interpreter == Nova VM == Native ELF): neural_network.nova', () => {
  const src = fs.readFileSync(path.resolve('nova/examples/neural_network.nova'), 'utf8');
  const { stdout } = assertThreeWayEquivalence('neural_network', src, 'release');
  assert.ok(stdout.some((l) => l.includes('Training report:') && l.includes('"epochs": 800')));
});

test('6. 3-Way Execution Equivalence (Interpreter == Nova VM == Native ELF [Debug & Release]): ai_engine.nova', () => {
  const src = fs.readFileSync(path.resolve('nova/examples/ai_engine.nova'), 'utf8');
  const debugRun = assertThreeWayEquivalence('ai_engine_debug', src, 'debug');
  const releaseRun = assertThreeWayEquivalence('ai_engine_release', src, 'release');
  assert.deepEqual(debugRun.stdout, releaseRun.stdout);
  assert.ok(debugRun.stdout.some((l) => l.includes('Sequential(layers=4, params=17)')));
  assert.ok(debugRun.stdout.some((l) => l.includes('"optimizer": adam')));
});

test('7. 3-Way Execution Equivalence: custom SGD + train_step + autograd zero_grad + tensor methods', () => {
  const source = `use nova.ai
use nova.nn

fn main() {
    let t = full([2, 2], 3.0)
    let r = randn([2, 2], 123)
    print("full: " + str(t) + " sum=" + str(t.sum()) + " min=" + str(t.min()) + " max=" + str(t.max()))
    print("randn: " + str(r) + " len=" + str(len(r)) + " type=" + type_of(r))
    print("reshaped: " + str(t.reshape([1, 4])))
    print("t[0]=" + str(t[0]))

    let gx = grad_var(tensor([[2.0, -1.0]]))
    let gw = grad_var(tensor([[1.5], [0.5]]))
    let out = grad_relu(grad_matmul(gx, gw))
    let loss = grad_mse(out, tensor([[4.0]]))
    backward(loss)
    print("grad_w: " + str(grad_of(gw)))
    zero_grad(gw)
    print("zeroed_w: " + str(grad_of(gw)))

    let m = model_new()
    model_add_linear(m, 1, 1, 9)
    let opt = optimizer_sgd(0.05, 0.5)
    let x = tensor([[1.0], [2.0], [3.0]])
    let y = tensor([[2.0], [4.0], [6.0]])
    mut last_loss = 0.0
    for i in 0..40 {
        last_loss = train_step(m, opt, x, y, "mse")
    }
    print("sgd converged: " + str(last_loss < 0.05))
    print("sgd pred: " + str(model_forward(m, tensor([[4.0]]))))
}
`;
  const { stdout } = assertThreeWayEquivalence('custom_ai_sgd', source, 'release');
  assert.ok(stdout.includes('sgd converged: true'));
});
