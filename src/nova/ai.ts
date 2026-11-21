/**
 * Nova AI Engine (Decoupled from Nova VM, exact mathematical parity with Rust `nova-ai` & C99 `libnovart`).
 * - Stage 1: Multi-dimensional CPU Tensor engine (Float64Array storage, broadcasting, matmul, reductions, activations, losses)
 * - Stage 2: Reverse-mode Automatic Differentiation (Dynamic Tensor Autograd graph `GradVar` + scalar `AutogradNode`)
 * - Stage 3: Neural Network Layers (`Linear`, `Sequential`), Optimizers (`SGDOptimizer`, `AdamOptimizer`), and Model Training
 */

/**
 * Deterministic Mulberry32 PRNG matching Rust `nova-ai` and C99 `libnovart`
 * so weight initialization and training trajectories are reproducible across all backends.
 */
export class DeterministicRng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  nextU32(): number {
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  }

  /** Uniform float in [-1.0, 1.0). */
  nextSigned(): number {
    return (this.nextU32() / 4294967296) * 2 - 1;
  }
}

export class NovaTensor {
  public readonly shape: number[];
  public readonly data: Float64Array;

  constructor(shape: number[], data: Float64Array | number[]) {
    const expected = shape.reduce((a, b) => a * b, 1);
    if (data.length !== expected) {
      throw new Error(`Tensor data length ${data.length} does not match shape [${shape.join(', ')}]`);
    }
    this.shape = [...shape];
    this.data = data instanceof Float64Array ? new Float64Array(data) : Float64Array.from(data);
  }

  static fromNested(input: unknown): NovaTensor {
    const shape: number[] = [];
    const flat: number[] = [];

    const inferShape = (node: unknown, depth: number): void => {
      if (Array.isArray(node)) {
        if (shape.length === depth) {
          shape.push(node.length);
        } else if (shape[depth] !== node.length) {
          throw new Error(`Ragged tensor shape at dimension ${depth}`);
        }
        for (const item of node) inferShape(item, depth + 1);
      } else if (typeof node === 'number') {
        flat.push(node);
      } else {
        throw new Error(`Invalid tensor element type: ${typeof node}`);
      }
    };

    inferShape(input, 0);
    if (shape.length === 0) {
      return new NovaTensor([1], [Number(input)]);
    }
    return new NovaTensor(shape, flat);
  }

  static scalar(v: number): NovaTensor {
    return new NovaTensor([1], [v]);
  }

  static zeros(shape: number[]): NovaTensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new NovaTensor(shape, new Float64Array(size));
  }

  static ones(shape: number[]): NovaTensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size).fill(1.0);
    return new NovaTensor(shape, data);
  }

  static full(shape: number[], value: number): NovaTensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size).fill(value);
    return new NovaTensor(shape, data);
  }

  static randUniform(shape: number[], seed = 42, scale = 1.0): NovaTensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const rng = new DeterministicRng(seed);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = rng.nextSigned() * scale;
    }
    return new NovaTensor(shape, data);
  }

  reshape(newShape: number[]): NovaTensor {
    const expected = newShape.reduce((a, b) => a * b, 1);
    if (expected !== this.data.length) {
      throw new Error(
        `Cannot reshape tensor of size ${this.data.length} into [${newShape.join(', ')}]`
      );
    }
    return new NovaTensor(newShape, this.data);
  }

  private static broadcastBinary(a: NovaTensor, b: NovaTensor, fn: (x: number, y: number) => number): NovaTensor {
    // Exact shape match
    if (a.shape.length === b.shape.length && a.shape.every((d, i) => d === b.shape[i])) {
      const out = new Float64Array(a.data.length);
      for (let i = 0; i < out.length; i++) out[i] = fn(a.data[i], b.data[i]);
      return new NovaTensor(a.shape, out);
    }
    // Scalar broadcast
    if (b.data.length === 1) {
      const out = new Float64Array(a.data.length);
      const bv = b.data[0];
      for (let i = 0; i < out.length; i++) out[i] = fn(a.data[i], bv);
      return new NovaTensor(a.shape, out);
    }
    if (a.data.length === 1) {
      const out = new Float64Array(b.data.length);
      const av = a.data[0];
      for (let i = 0; i < out.length; i++) out[i] = fn(av, b.data[i]);
      return new NovaTensor(b.shape, out);
    }
    // Row broadcast [M, N] op [1, N] or [N]
    if (
      a.shape.length === 2 &&
      ((b.shape.length === 1 && b.shape[0] === a.shape[1]) ||
        (b.shape.length === 2 && b.shape[0] === 1 && b.shape[1] === a.shape[1]))
    ) {
      const [rows, cols] = a.shape;
      const out = new Float64Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          out[r * cols + c] = fn(a.data[r * cols + c], b.data[c]);
        }
      }
      return new NovaTensor(a.shape, out);
    }
    throw new Error(`Incompatible tensor shapes for broadcast: [${a.shape.join(', ')}] and [${b.shape.join(', ')}]`);
  }

  add(other: NovaTensor | number): NovaTensor {
    const b = typeof other === 'number' ? new NovaTensor([1], [other]) : other;
    return NovaTensor.broadcastBinary(this, b, (x, y) => x + y);
  }

  sub(other: NovaTensor | number): NovaTensor {
    const b = typeof other === 'number' ? new NovaTensor([1], [other]) : other;
    return NovaTensor.broadcastBinary(this, b, (x, y) => x - y);
  }

  mul(other: NovaTensor | number): NovaTensor {
    const b = typeof other === 'number' ? new NovaTensor([1], [other]) : other;
    return NovaTensor.broadcastBinary(this, b, (x, y) => x * y);
  }

  div(other: NovaTensor | number): NovaTensor {
    const b = typeof other === 'number' ? new NovaTensor([1], [other]) : other;
    return NovaTensor.broadcastBinary(this, b, (x, y) => {
      if (y === 0) throw new Error('Tensor division by zero');
      return x / y;
    });
  }

  dot(other: NovaTensor): number {
    if (this.data.length !== other.data.length) {
      throw new Error(`dot() length mismatch: ${this.data.length} vs ${other.data.length}`);
    }
    let acc = 0;
    for (let i = 0; i < this.data.length; i++) acc += this.data[i] * other.data[i];
    return acc;
  }

  matmul(other: NovaTensor): NovaTensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error(`matmul '@' requires 2D tensors, got [${this.shape.join(', ')}] and [${other.shape.join(', ')}]`);
    }
    const [m, k1] = this.shape;
    const [k2, n] = other.shape;
    if (k1 !== k2) {
      throw new Error(`matmul shape mismatch: [${m}, ${k1}] @ [${k2}, ${n}]`);
    }
    const out = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let k = 0; k < k1; k++) {
        const aik = this.data[i * k1 + k];
        for (let j = 0; j < n; j++) {
          out[i * n + j] += aik * other.data[k * n + j];
        }
      }
    }
    return new NovaTensor([m, n], out);
  }

  transpose(): NovaTensor {
    if (this.shape.length !== 2) {
      throw new Error(`transpose requires 2D tensor, got [${this.shape.join(', ')}]`);
    }
    const [rows, cols] = this.shape;
    const out = new Float64Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        out[c * rows + r] = this.data[r * cols + c];
      }
    }
    return new NovaTensor([cols, rows], out);
  }

  relu(): NovaTensor {
    return new NovaTensor(this.shape, this.data.map((x) => (x > 0 ? x : 0)));
  }

  sigmoid(): NovaTensor {
    return new NovaTensor(this.shape, this.data.map((x) => 1 / (1 + Math.exp(-x))));
  }

  tanh(): NovaTensor {
    return new NovaTensor(this.shape, this.data.map((x) => Math.tanh(x)));
  }

  softmax(): NovaTensor {
    if (this.shape.length === 1) {
      const max = Math.max(...this.data);
      const exps = this.data.map((x) => Math.exp(x - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      return new NovaTensor(this.shape, exps.map((e) => e / sum));
    }
    if (this.shape.length === 2) {
      const [rows, cols] = this.shape;
      const out = new Float64Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        let max = -Infinity;
        for (let c = 0; c < cols; c++) max = Math.max(max, this.data[r * cols + c]);
        let sum = 0;
        for (let c = 0; c < cols; c++) {
          const e = Math.exp(this.data[r * cols + c] - max);
          out[r * cols + c] = e;
          sum += e;
        }
        for (let c = 0; c < cols; c++) out[r * cols + c] /= sum;
      }
      return new NovaTensor(this.shape, out);
    }
    throw new Error('softmax supports 1D and 2D tensors');
  }

  sum(): number {
    let s = 0;
    for (let i = 0; i < this.data.length; i++) s += this.data[i];
    return s;
  }

  mean(): number {
    return this.sum() / Math.max(1, this.data.length);
  }

  max(): number {
    let m = -Infinity;
    for (let i = 0; i < this.data.length; i++) if (this.data[i] > m) m = this.data[i];
    return m;
  }

  min(): number {
    let m = Infinity;
    for (let i = 0; i < this.data.length; i++) if (this.data[i] < m) m = this.data[i];
    return m;
  }

  argmax(): number {
    let best = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < this.data.length; i++) {
      if (this.data[i] > bestVal) {
        bestVal = this.data[i];
        best = i;
      }
    }
    return best;
  }

  unbroadcastTo(targetShape: number[]): NovaTensor {
    if (
      this.shape.length === targetShape.length &&
      this.shape.every((d, i) => d === targetShape[i])
    ) {
      return new NovaTensor(this.shape, this.data);
    }
    const targetLen = targetShape.reduce((a, b) => a * b, 1);
    if (targetLen === 1) {
      return new NovaTensor(targetShape, [this.sum()]);
    }
    if (
      this.shape.length === 2 &&
      ((targetShape.length === 1 && targetShape[0] === this.shape[1]) ||
        (targetShape.length === 2 && targetShape[0] === 1 && targetShape[1] === this.shape[1]))
    ) {
      const [rows, cols] = this.shape;
      const sums = new Float64Array(cols);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          sums[c] += this.data[r * cols + c];
        }
      }
      return new NovaTensor(targetShape, sums);
    }
    throw new Error(
      `Cannot unbroadcast gradient [${this.shape.join(', ')}] to [${targetShape.join(', ')}]`
    );
  }

  toNested(): unknown {
    const build = (dim: number, offset: number): { val: unknown; next: number } => {
      if (dim === this.shape.length - 1) {
        const len = this.shape[dim];
        const arr: number[] = [];
        for (let i = 0; i < len; i++) {
          const rounded = Number(this.data[offset + i].toFixed(4));
          arr.push(Object.is(rounded, -0) ? 0 : rounded);
        }
        return { val: arr, next: offset + len };
      }
      const len = this.shape[dim];
      const arr: unknown[] = [];
      let cur = offset;
      for (let i = 0; i < len; i++) {
        const sub = build(dim + 1, cur);
        arr.push(sub.val);
        cur = sub.next;
      }
      return { val: arr, next: cur };
    };
    return build(0, 0).val;
  }

  toString(): string {
    const formatArr = (node: unknown): string => {
      if (Array.isArray(node)) {
        return `[${node.map(formatArr).join(', ')}]`;
      }
      if (typeof node === 'number') {
        const v = Object.is(node, -0) ? 0 : node;
        return Number.isInteger(v) ? `${v}` : `${Number(v.toFixed(4))}`;
      }
      return String(node);
    };
    return `Tensor(${formatArr(this.toNested())}, shape=[${this.shape.join(', ')}])`;
  }
}

export function mseLoss(pred: NovaTensor, target: NovaTensor): number {
  const diff = pred.sub(target);
  let sum = 0;
  for (let i = 0; i < diff.data.length; i++) sum += diff.data[i] * diff.data[i];
  return sum / Math.max(1, diff.data.length);
}

export function bceLoss(pred: NovaTensor, target: NovaTensor): number {
  if (
    pred.shape.length !== target.shape.length ||
    !pred.shape.every((d, i) => d === target.shape[i])
  ) {
    throw new Error(`bce_loss shape mismatch: [${pred.shape.join(', ')}] vs [${target.shape.join(', ')}]`);
  }
  const eps = 1e-7;
  let sum = 0;
  for (let i = 0; i < pred.data.length; i++) {
    const p = Math.min(1 - eps, Math.max(eps, pred.data[i]));
    const y = target.data[i];
    sum += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
  }
  return sum / Math.max(1, pred.data.length);
}

// ============================================================================
// Stage 2: Reverse-Mode Tensor Autograd (`GradVar`) + Scalar `AutogradNode`
// ============================================================================

export class GradVar {
  public data: NovaTensor;
  public grad: NovaTensor;
  private parents: GradVar[];
  private backwardFn: () => void;

  constructor(data: NovaTensor, parents: GradVar[] = [], backwardFn: () => void = () => {}) {
    this.data = data;
    this.grad = NovaTensor.zeros(data.shape);
    this.parents = parents;
    this.backwardFn = backwardFn;
  }

  static leaf(tensor: NovaTensor): GradVar {
    return new GradVar(new NovaTensor(tensor.shape, tensor.data));
  }

  zeroGrad(): void {
    this.grad.data.fill(0);
  }

  private accumulateGrad(delta: NovaTensor): void {
    const reduced = delta.unbroadcastTo(this.data.shape);
    for (let i = 0; i < this.grad.data.length; i++) {
      this.grad.data[i] += reduced.data[i];
    }
  }

  add(other: GradVar): GradVar {
    const out = new GradVar(this.data.add(other.data), [this, other]);
    out.backwardFn = () => {
      this.accumulateGrad(out.grad);
      other.accumulateGrad(out.grad);
    };
    return out;
  }

  sub(other: GradVar): GradVar {
    const out = new GradVar(this.data.sub(other.data), [this, other]);
    out.backwardFn = () => {
      this.accumulateGrad(out.grad);
      other.accumulateGrad(out.grad.mul(-1));
    };
    return out;
  }

  mul(other: GradVar): GradVar {
    const out = new GradVar(this.data.mul(other.data), [this, other]);
    out.backwardFn = () => {
      this.accumulateGrad(out.grad.mul(other.data));
      other.accumulateGrad(out.grad.mul(this.data));
    };
    return out;
  }

  matmul(other: GradVar): GradVar {
    const out = new GradVar(this.data.matmul(other.data), [this, other]);
    out.backwardFn = () => {
      this.accumulateGrad(out.grad.matmul(other.data.transpose()));
      other.accumulateGrad(this.data.transpose().matmul(out.grad));
    };
    return out;
  }

  relu(): GradVar {
    const out = new GradVar(this.data.relu(), [this]);
    out.backwardFn = () => {
      const gradData = new Float64Array(this.data.data.length);
      for (let i = 0; i < gradData.length; i++) {
        gradData[i] = this.data.data[i] > 0 ? out.grad.data[i] : 0;
      }
      this.accumulateGrad(new NovaTensor(this.data.shape, gradData));
    };
    return out;
  }

  sigmoid(): GradVar {
    const sig = this.data.sigmoid();
    const out = new GradVar(sig, [this]);
    out.backwardFn = () => {
      const gradData = new Float64Array(sig.data.length);
      for (let i = 0; i < gradData.length; i++) {
        const y = sig.data[i];
        gradData[i] = out.grad.data[i] * y * (1 - y);
      }
      this.accumulateGrad(new NovaTensor(sig.shape, gradData));
    };
    return out;
  }

  tanh(): GradVar {
    const th = this.data.tanh();
    const out = new GradVar(th, [this]);
    out.backwardFn = () => {
      const gradData = new Float64Array(th.data.length);
      for (let i = 0; i < gradData.length; i++) {
        const y = th.data[i];
        gradData[i] = out.grad.data[i] * (1 - y * y);
      }
      this.accumulateGrad(new NovaTensor(th.shape, gradData));
    };
    return out;
  }

  mseLoss(target: NovaTensor): GradVar {
    const lossVal = mseLoss(this.data, target);
    const out = new GradVar(NovaTensor.scalar(lossVal), [this]);
    out.backwardFn = () => {
      const n = Math.max(1, this.data.data.length);
      const scale = (2 * out.grad.data[0]) / n;
      const gradData = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        gradData[i] = scale * (this.data.data[i] - target.data[i]);
      }
      this.accumulateGrad(new NovaTensor(this.data.shape, gradData));
    };
    return out;
  }

  bceLoss(target: NovaTensor): GradVar {
    const lossVal = bceLoss(this.data, target);
    const out = new GradVar(NovaTensor.scalar(lossVal), [this]);
    out.backwardFn = () => {
      const eps = 1e-7;
      const n = Math.max(1, this.data.data.length);
      const scale = out.grad.data[0] / n;
      const gradData = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const p = Math.min(1 - eps, Math.max(eps, this.data.data[i]));
        const y = target.data[i];
        gradData[i] = scale * ((p - y) / (p * (1 - p)));
      }
      this.accumulateGrad(new NovaTensor(this.data.shape, gradData));
    };
    return out;
  }

  backward(): void {
    const topo: GradVar[] = [];
    const visited = new Set<GradVar>();
    const build = (n: GradVar) => {
      if (visited.has(n)) return;
      visited.add(n);
      for (const p of n.parents) build(p);
      topo.push(n);
    };
    build(this);
    this.grad.data.fill(1.0);
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i].backwardFn();
    }
  }
}

export class AutogradNode {
  public grad = 0;
  private backwardFn: () => void = () => {};
  private parents: AutogradNode[];

  constructor(public data: number, parents: AutogradNode[] = []) {
    this.parents = parents;
  }

  add(other: AutogradNode): AutogradNode {
    const out = new AutogradNode(this.data + other.data, [this, other]);
    out.backwardFn = () => {
      this.grad += out.grad;
      other.grad += out.grad;
    };
    return out;
  }

  mul(other: AutogradNode): AutogradNode {
    const out = new AutogradNode(this.data * other.data, [this, other]);
    out.backwardFn = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    return out;
  }

  relu(): AutogradNode {
    const out = new AutogradNode(this.data > 0 ? this.data : 0, [this]);
    out.backwardFn = () => {
      this.grad += (this.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  backward(): void {
    const topo: AutogradNode[] = [];
    const visited = new Set<AutogradNode>();
    const build = (n: AutogradNode) => {
      if (visited.has(n)) return;
      visited.add(n);
      for (const p of n.parents) build(p);
      topo.push(n);
    };
    build(this);
    this.grad = 1.0;
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i].backwardFn();
    }
  }
}

// ============================================================================
// Stage 3: Neural Network Layers (`Linear`, `Sequential`), Optimizers (`SGD`, `Adam`) & Trainer
// ============================================================================

export type ActivationKind = 'relu' | 'sigmoid' | 'tanh';

export class LinearLayer {
  public readonly weight: GradVar;
  public readonly bias: GradVar;

  constructor(
    public readonly inFeatures: number,
    public readonly outFeatures: number,
    seed = 42
  ) {
    const limit = Math.sqrt(6 / (inFeatures + outFeatures));
    this.weight = GradVar.leaf(NovaTensor.randUniform([inFeatures, outFeatures], seed, limit));
    this.bias = GradVar.leaf(NovaTensor.zeros([1, outFeatures]));
  }

  forward(x: GradVar): GradVar {
    return x.matmul(this.weight).add(this.bias);
  }

  parameters(): GradVar[] {
    return [this.weight, this.bias];
  }
}

export type SequentialLayer =
  | { kind: 'linear'; layer: LinearLayer }
  | { kind: 'activation'; activation: ActivationKind };

export class SequentialModel {
  public readonly layers: SequentialLayer[] = [];

  addLinear(inFeatures: number, outFeatures: number, seed = 42): this {
    this.layers.push({ kind: 'linear', layer: new LinearLayer(inFeatures, outFeatures, seed) });
    return this;
  }

  addActivation(activation: ActivationKind): this {
    this.layers.push({ kind: 'activation', activation });
    return this;
  }

  forward(x: GradVar): GradVar {
    let cur = x;
    for (const entry of this.layers) {
      if (entry.kind === 'linear') {
        cur = entry.layer.forward(cur);
      } else if (entry.activation === 'relu') {
        cur = cur.relu();
      } else if (entry.activation === 'sigmoid') {
        cur = cur.sigmoid();
      } else {
        cur = cur.tanh();
      }
    }
    return cur;
  }

  predict(x: NovaTensor): NovaTensor {
    return this.forward(GradVar.leaf(x)).data;
  }

  parameters(): GradVar[] {
    const out: GradVar[] = [];
    for (const entry of this.layers) {
      if (entry.kind === 'linear') out.push(...entry.layer.parameters());
    }
    return out;
  }

  paramCount(): number {
    return this.parameters().reduce((acc, p) => acc + p.data.data.length, 0);
  }

  zeroGrad(): void {
    for (const p of this.parameters()) p.zeroGrad();
  }
}

export interface NovaOptimizer {
  readonly name: 'sgd' | 'adam';
  step(params: GradVar[]): void;
}

export class SGDOptimizer implements NovaOptimizer {
  public readonly name = 'sgd' as const;
  private velocities: Float64Array[] = [];

  constructor(
    public readonly lr: number,
    public readonly momentum = 0.0,
    public readonly weightDecay = 0.0
  ) {}

  step(params: GradVar[]): void {
    if (this.velocities.length !== params.length) {
      this.velocities = params.map((p) => new Float64Array(p.data.data.length));
    }
    for (let idx = 0; idx < params.length; idx++) {
      const p = params[idx];
      const vel = this.velocities[idx];
      for (let i = 0; i < p.data.data.length; i++) {
        const g = p.grad.data[i] + this.weightDecay * p.data.data[i];
        vel[i] = this.momentum * vel[i] + g;
        p.data.data[i] -= this.lr * vel[i];
      }
    }
  }
}

export class AdamOptimizer implements NovaOptimizer {
  public readonly name = 'adam' as const;
  private timestep = 0;
  private m: Float64Array[] = [];
  private v: Float64Array[] = [];

  constructor(
    public readonly lr: number,
    public readonly beta1 = 0.9,
    public readonly beta2 = 0.999,
    public readonly eps = 1e-8
  ) {}

  step(params: GradVar[]): void {
    if (this.m.length !== params.length) {
      this.m = params.map((p) => new Float64Array(p.data.data.length));
      this.v = params.map((p) => new Float64Array(p.data.data.length));
    }
    this.timestep++;
    const bc1 = 1 - Math.pow(this.beta1, this.timestep);
    const bc2 = 1 - Math.pow(this.beta2, this.timestep);

    for (let idx = 0; idx < params.length; idx++) {
      const p = params[idx];
      const m = this.m[idx];
      const v = this.v[idx];
      for (let i = 0; i < p.data.data.length; i++) {
        const g = p.grad.data[i];
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g;
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g * g;
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        p.data.data[i] -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }
}

export interface ModelTrainingReport {
  epochs: number;
  initialLoss: number;
  finalLoss: number;
  optimizer: string;
  paramCount: number;
  predictions: number[];
}

export function trainStep(
  model: SequentialModel,
  optimizer: NovaOptimizer,
  x: NovaTensor,
  y: NovaTensor,
  lossKind: 'mse' | 'bce' = 'mse'
): number {
  model.zeroGrad();
  const pred = model.forward(GradVar.leaf(x));
  const loss = lossKind === 'bce' ? pred.bceLoss(y) : pred.mseLoss(y);
  const val = loss.data.data[0];
  loss.backward();
  optimizer.step(model.parameters());
  return val;
}

export function trainModel(
  model: SequentialModel,
  optimizer: NovaOptimizer,
  x: NovaTensor,
  y: NovaTensor,
  epochs: number,
  lossKind: 'mse' | 'bce' = 'mse'
): ModelTrainingReport {
  const clamped = Math.max(1, Math.min(Math.trunc(epochs), 5000));
  let initialLoss = 0;
  let finalLoss = 0;
  for (let ep = 0; ep < clamped; ep++) {
    const loss = trainStep(model, optimizer, x, y, lossKind);
    if (ep === 0) initialLoss = loss;
    finalLoss = loss;
  }
  const preds = model.predict(x);
  return {
    epochs: clamped,
    initialLoss: Number(initialLoss.toFixed(6)),
    finalLoss: Number(finalLoss.toFixed(6)),
    optimizer: optimizer.name,
    paramCount: model.paramCount(),
    predictions: Array.from(preds.data).map((p) => Number(p.toFixed(4))),
  };
}

export interface TrainingSummary {
  epochs: number;
  initialLoss: number;
  finalLoss: number;
  predictions: number[];
}

/**
 * Deterministic 2-layer XOR benchmark powered by Nova's real autograd + Sequential + Adam/SGD stack.
 */
export function trainXorNetwork(epochs: number, learningRate: number): TrainingSummary {
  const x = new NovaTensor([4, 2], [0, 0, 0, 1, 1, 0, 1, 1]);
  const y = new NovaTensor([4, 1], [0, 1, 1, 0]);
  const model = new SequentialModel()
    .addLinear(2, 4, 42)
    .addActivation('tanh')
    .addLinear(4, 1, 137)
    .addActivation('sigmoid');
  const lr = learningRate > 0 ? Math.min(learningRate, 0.1) : 0.05;
  const opt = new AdamOptimizer(lr);
  const report = trainModel(model, opt, x, y, epochs, 'mse');
  return {
    epochs: report.epochs,
    initialLoss: report.initialLoss,
    finalLoss: report.finalLoss,
    predictions: report.predictions,
  };
}
