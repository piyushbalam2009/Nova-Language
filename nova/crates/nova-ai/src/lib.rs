//! Nova Native Rust AI Engine (`nova-ai`).
//!
//! Architecture:
//! - Layer 1: Multi-dimensional CPU `Tensor` with broadcasting, `matmul`, reductions, activations, and losses.
//! - Layer 2: Tape-based reverse-mode Automatic Differentiation (`Tape`, `VarId`) for scalar & tensor graphs.
//! - Layer 3: Neural Network Layers (`Linear`, `Activation`, `Sequential`).
//! - Layer 4: Optimizers (`Sgd` with momentum/weight-decay, `Adam` with bias-corrected moments).
//! - Layer 5: Model Training (`Trainer`, `TrainingReport`, deterministic PRNG for reproducible runs).
//!
//! Zero external dependencies; decoupled from ordinary Nova programs so non-AI workloads stay lightweight.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

/// Deterministic Mulberry32 PRNG matching the Nova VM and `libnovart` native C99 runtime
/// so weight initialization and training trajectories are reproducible across backends.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: u32,
}

impl DeterministicRng {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x6D2B_79F5);
        let mut t = (self.state ^ (self.state >> 15)).wrapping_mul(1 | self.state);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        t ^ (t >> 14)
    }

    /// Uniform float in `[-1.0, 1.0)`.
    pub fn next_signed(&mut self) -> f64 {
        let u = (self.next_u32() as f64) / 4294967296.0;
        u * 2.0 - 1.0
    }
}

// ============================================================================
// 1. MULTI-DIMENSIONAL CPU TENSOR ENGINE
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Result<Self, String> {
        let expected: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        if data.len() != expected {
            return Err(format!(
                "Tensor data length {} does not match shape {:?}",
                data.len(),
                shape
            ));
        }
        Ok(Self { shape, data })
    }

    pub fn scalar(v: f64) -> Self {
        Self {
            shape: vec![1],
            data: vec![v],
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product::<usize>().max(1);
        Self {
            shape,
            data: vec![0.0; len],
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product::<usize>().max(1);
        Self {
            shape,
            data: vec![1.0; len],
        }
    }

    pub fn full(shape: Vec<usize>, value: f64) -> Self {
        let len: usize = shape.iter().product::<usize>().max(1);
        Self {
            shape,
            data: vec![value; len],
        }
    }

    pub fn rand_uniform(shape: Vec<usize>, seed: u32, scale: f64) -> Self {
        let len: usize = shape.iter().product::<usize>().max(1);
        let mut rng = DeterministicRng::new(seed);
        let data = (0..len).map(|_| rng.next_signed() * scale).collect();
        Self { shape, data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = new_shape.iter().product();
        if expected != self.data.len() {
            return Err(format!(
                "Cannot reshape tensor of size {} into {:?}",
                self.data.len(),
                new_shape
            ));
        }
        Ok(Self {
            shape: new_shape,
            data: self.data.clone(),
        })
    }

    fn broadcast_binary<F>(&self, other: &Self, op_name: &str, f: F) -> Result<Self, String>
    where
        F: Fn(f64, f64) -> Result<f64, String>,
    {
        if self.shape == other.shape {
            let mut out = Vec::with_capacity(self.data.len());
            for (&a, &b) in self.data.iter().zip(&other.data) {
                out.push(f(a, b)?);
            }
            return Ok(Self {
                shape: self.shape.clone(),
                data: out,
            });
        }
        if other.data.len() == 1 {
            let b = other.data[0];
            let mut out = Vec::with_capacity(self.data.len());
            for &a in &self.data {
                out.push(f(a, b)?);
            }
            return Ok(Self {
                shape: self.shape.clone(),
                data: out,
            });
        }
        if self.data.len() == 1 {
            let a = self.data[0];
            let mut out = Vec::with_capacity(other.data.len());
            for &b in &other.data {
                out.push(f(a, b)?);
            }
            return Ok(Self {
                shape: other.shape.clone(),
                data: out,
            });
        }
        // Row broadcast: [M, N] op [1, N] or [N]
        if self.shape.len() == 2
            && ((other.shape.len() == 1 && other.shape[0] == self.shape[1])
                || (other.shape.len() == 2
                    && other.shape[0] == 1
                    && other.shape[1] == self.shape[1]))
        {
            let (rows, cols) = (self.shape[0], self.shape[1]);
            let mut out = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                for c in 0..cols {
                    out.push(f(self.data[r * cols + c], other.data[c])?);
                }
            }
            return Ok(Self {
                shape: self.shape.clone(),
                data: out,
            });
        }
        Err(format!(
            "Incompatible shapes for tensor {op_name}: {:?} and {:?}",
            self.shape, other.shape
        ))
    }

    pub fn add(&self, other: &Self) -> Result<Self, String> {
        self.broadcast_binary(other, "add", |a, b| Ok(a + b))
    }

    pub fn sub(&self, other: &Self) -> Result<Self, String> {
        self.broadcast_binary(other, "sub", |a, b| Ok(a - b))
    }

    pub fn mul(&self, other: &Self) -> Result<Self, String> {
        self.broadcast_binary(other, "mul", |a, b| Ok(a * b))
    }

    pub fn div(&self, other: &Self) -> Result<Self, String> {
        self.broadcast_binary(other, "div", |a, b| {
            if b == 0.0 {
                Err("Tensor division by zero".to_string())
            } else {
                Ok(a / b)
            }
        })
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| x * factor).collect(),
        }
    }

    pub fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    pub fn dot(&self, other: &Self) -> Result<f64, String> {
        if self.data.len() != other.data.len() {
            return Err(format!(
                "dot() length mismatch: {} vs {}",
                self.data.len(),
                other.data.len()
            ));
        }
        Ok(self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .sum())
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(format!(
                "matmul requires 2D tensors, got {:?} and {:?}",
                self.shape, other.shape
            ));
        }
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k1 != k2 {
            return Err(format!("matmul shape mismatch: [{m}, {k1}] @ [{k2}, {n}]"));
        }
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for k in 0..k1 {
                let aik = self.data[i * k1 + k];
                for j in 0..n {
                    out[i * n + j] += aik * other.data[k * n + j];
                }
            }
        }
        Ok(Self {
            shape: vec![m, n],
            data: out,
        })
    }

    pub fn transpose(&self) -> Result<Self, String> {
        if self.shape.len() != 2 {
            return Err(format!(
                "transpose requires a 2D tensor, got {:?}",
                self.shape
            ));
        }
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = self.data[r * cols + c];
            }
        }
        Ok(Self {
            shape: vec![cols, rows],
            data: out,
        })
    }

    pub fn relu(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| x.max(0.0)).collect(),
        }
    }

    pub fn sigmoid(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        }
    }

    pub fn tanh(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| x.tanh()).collect(),
        }
    }

    pub fn softmax(&self) -> Result<Self, String> {
        if self.shape.len() == 1 {
            let max_v = self
                .data
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = self.data.iter().map(|&x| (x - max_v).exp()).collect();
            let sum: f64 = exps.iter().sum();
            return Ok(Self {
                shape: self.shape.clone(),
                data: exps.into_iter().map(|e| e / sum).collect(),
            });
        }
        if self.shape.len() == 2 {
            let (rows, cols) = (self.shape[0], self.shape[1]);
            let mut out = vec![0.0; rows * cols];
            for r in 0..rows {
                let row = &self.data[r * cols..(r + 1) * cols];
                let max_v = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for c in 0..cols {
                    let e = (row[c] - max_v).exp();
                    out[r * cols + c] = e;
                    sum += e;
                }
                for c in 0..cols {
                    out[r * cols + c] /= sum;
                }
            }
            return Ok(Self {
                shape: self.shape.clone(),
                data: out,
            });
        }
        Err("softmax supports 1D and 2D tensors".to_string())
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f64 {
        self.sum() / (self.data.len().max(1) as f64)
    }

    pub fn max(&self) -> f64 {
        self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn min(&self) -> f64 {
        self.data.iter().copied().fold(f64::INFINITY, f64::min)
    }

    pub fn argmax(&self) -> usize {
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (i, &v) in self.data.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Reduce a gradient tensor back to `target_shape` (reversing broadcasting).
    pub fn unbroadcast_to(&self, target_shape: &[usize]) -> Result<Self, String> {
        if self.shape == target_shape {
            return Ok(self.clone());
        }
        let target_len: usize = target_shape.iter().product();
        if target_len == 1 {
            return Ok(Self {
                shape: target_shape.to_vec(),
                data: vec![self.sum()],
            });
        }
        if self.shape.len() == 2
            && ((target_shape.len() == 1 && target_shape[0] == self.shape[1])
                || (target_shape.len() == 2
                    && target_shape[0] == 1
                    && target_shape[1] == self.shape[1]))
        {
            let (rows, cols) = (self.shape[0], self.shape[1]);
            let mut sums = vec![0.0; cols];
            for r in 0..rows {
                for c in 0..cols {
                    sums[c] += self.data[r * cols + c];
                }
            }
            return Ok(Self {
                shape: target_shape.to_vec(),
                data: sums,
            });
        }
        Err(format!(
            "Cannot unbroadcast gradient {:?} to target shape {:?}",
            self.shape, target_shape
        ))
    }

    fn format_recursive(&self, dim: usize, offset: usize) -> (String, usize) {
        let len = self.shape[dim];
        if dim + 1 == self.shape.len() {
            let mut parts = Vec::with_capacity(len);
            for i in 0..len {
                let raw = (self.data[offset + i] * 10_000.0).round() / 10_000.0;
                let v = if raw == 0.0 { 0.0 } else { raw };
                if v.fract() == 0.0 {
                    parts.push(format!("{}", v as i64));
                } else {
                    parts.push(format!("{v}"));
                }
            }
            (format!("[{}]", parts.join(", ")), offset + len)
        } else {
            let mut parts = Vec::with_capacity(len);
            let mut cur = offset;
            for _ in 0..len {
                let (s, next) = self.format_recursive(dim + 1, cur);
                parts.push(s);
                cur = next;
            }
            (format!("[{}]", parts.join(", ")), cur)
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (body, _) = self.format_recursive(0, 0);
        let shape_s = self
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Tensor({body}, shape=[{shape_s}])")
    }
}

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Result<f64, String> {
    let diff = pred.sub(target)?;
    let sum: f64 = diff.data.iter().map(|d| d * d).sum();
    Ok(sum / (diff.data.len().max(1) as f64))
}

pub fn bce_loss(pred: &Tensor, target: &Tensor) -> Result<f64, String> {
    if pred.shape != target.shape {
        return Err(format!(
            "bce_loss shape mismatch: {:?} vs {:?}",
            pred.shape, target.shape
        ));
    }
    let eps = 1e-7;
    let sum: f64 = pred
        .data
        .iter()
        .zip(&target.data)
        .map(|(&p, &y)| {
            let pc = p.clamp(eps, 1.0 - eps);
            -(y * pc.ln() + (1.0 - y) * (1.0 - pc).ln())
        })
        .sum();
    Ok(sum / (pred.data.len().max(1) as f64))
}

// ============================================================================
// 2. REVERSE-MODE AUTOMATIC DIFFERENTIATION (TENSOR AUTOGRAD ENGINE)
// ============================================================================

#[derive(Debug, Clone)]
enum GradOp {
    Leaf,
    Add(Var, Var),
    Sub(Var, Var),
    Mul(Var, Var),
    MatMul(Var, Var),
    ReLU(Var),
    Sigmoid(Var),
    Tanh(Var),
    MseLoss(Var, Tensor),
    BceLoss(Var, Tensor),
}

struct VarInner {
    data: Tensor,
    grad: Tensor,
    op: GradOp,
}

/// Differentiable Tensor node forming a dynamic reverse-mode computation graph.
#[derive(Clone)]
pub struct Var(Rc<RefCell<VarInner>>);

impl Var {
    pub fn leaf(data: Tensor) -> Self {
        let grad = Tensor::zeros(data.shape.clone());
        Self(Rc::new(RefCell::new(VarInner {
            data,
            grad,
            op: GradOp::Leaf,
        })))
    }

    pub fn scalar(v: f64) -> Self {
        Self::leaf(Tensor::scalar(v))
    }

    pub fn data(&self) -> Tensor {
        self.0.borrow().data.clone()
    }

    pub fn grad(&self) -> Tensor {
        self.0.borrow().grad.clone()
    }

    pub fn set_data(&self, next: Tensor) {
        self.0.borrow_mut().data = next;
    }

    pub fn zero_grad(&self) {
        let mut inner = self.0.borrow_mut();
        for g in &mut inner.grad.data {
            *g = 0.0;
        }
    }

    fn accumulate_grad(&self, delta: &Tensor) -> Result<(), String> {
        let mut inner = self.0.borrow_mut();
        let reduced = delta.unbroadcast_to(&inner.data.shape)?;
        for (dst, &src) in inner.grad.data.iter_mut().zip(&reduced.data) {
            *dst += src;
        }
        Ok(())
    }

    pub fn add(&self, other: &Var) -> Result<Var, String> {
        let out = self.data().add(&other.data())?;
        let grad = Tensor::zeros(out.shape.clone());
        Ok(Var(Rc::new(RefCell::new(VarInner {
            data: out,
            grad,
            op: GradOp::Add(self.clone(), other.clone()),
        }))))
    }

    pub fn sub(&self, other: &Var) -> Result<Var, String> {
        let out = self.data().sub(&other.data())?;
        let grad = Tensor::zeros(out.shape.clone());
        Ok(Var(Rc::new(RefCell::new(VarInner {
            data: out,
            grad,
            op: GradOp::Sub(self.clone(), other.clone()),
        }))))
    }

    pub fn mul(&self, other: &Var) -> Result<Var, String> {
        let out = self.data().mul(&other.data())?;
        let grad = Tensor::zeros(out.shape.clone());
        Ok(Var(Rc::new(RefCell::new(VarInner {
            data: out,
            grad,
            op: GradOp::Mul(self.clone(), other.clone()),
        }))))
    }

    pub fn matmul(&self, other: &Var) -> Result<Var, String> {
        let out = self.data().matmul(&other.data())?;
        let grad = Tensor::zeros(out.shape.clone());
        Ok(Var(Rc::new(RefCell::new(VarInner {
            data: out,
            grad,
            op: GradOp::MatMul(self.clone(), other.clone()),
        }))))
    }

    pub fn relu(&self) -> Var {
        let out = self.data().relu();
        let grad = Tensor::zeros(out.shape.clone());
        Var(Rc::new(RefCell::new(VarInner {
            data: out,
            grad,
            op: GradOp::ReLU(self.clone()),
        })))
    }

    pub fn sigmoid(&self) -> Var {
        let out = self.data().sigmoid();
        let grad = Tensor::zeros(out.shape.clone());
        Var(Rc::new(RefCell::new(VarInner {
            data: out,
            grad,
            op: GradOp::Sigmoid(self.clone()),
        })))
    }

    pub fn tanh(&self) -> Var {
        let out = self.data().tanh();
        let grad = Tensor::zeros(out.shape.clone());
        Var(Rc::new(RefCell::new(VarInner {
            data: out,
            grad,
            op: GradOp::Tanh(self.clone()),
        })))
    }

    pub fn mse_loss(&self, target: &Tensor) -> Result<Var, String> {
        let loss_val = mse_loss(&self.data(), target)?;
        Ok(Var(Rc::new(RefCell::new(VarInner {
            data: Tensor::scalar(loss_val),
            grad: Tensor::zeros(vec![1]),
            op: GradOp::MseLoss(self.clone(), target.clone()),
        }))))
    }

    pub fn bce_loss(&self, target: &Tensor) -> Result<Var, String> {
        let loss_val = bce_loss(&self.data(), target)?;
        Ok(Var(Rc::new(RefCell::new(VarInner {
            data: Tensor::scalar(loss_val),
            grad: Tensor::zeros(vec![1]),
            op: GradOp::BceLoss(self.clone(), target.clone()),
        }))))
    }

    fn ptr_id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    /// Reverse-mode topological backpropagation from this variable.
    pub fn backward(&self) -> Result<(), String> {
        let mut topo: Vec<Var> = Vec::new();
        let mut visited: Vec<usize> = Vec::new();

        fn build(node: &Var, visited: &mut Vec<usize>, topo: &mut Vec<Var>) {
            let id = node.ptr_id();
            if visited.contains(&id) {
                return;
            }
            visited.push(id);
            let op = node.0.borrow().op.clone();
            match op {
                GradOp::Leaf => {}
                GradOp::Add(a, b)
                | GradOp::Sub(a, b)
                | GradOp::Mul(a, b)
                | GradOp::MatMul(a, b) => {
                    build(&a, visited, topo);
                    build(&b, visited, topo);
                }
                GradOp::ReLU(a)
                | GradOp::Sigmoid(a)
                | GradOp::Tanh(a)
                | GradOp::MseLoss(a, _)
                | GradOp::BceLoss(a, _) => {
                    build(&a, visited, topo);
                }
            }
            topo.push(node.clone());
        }

        build(self, &mut visited, &mut topo);

        // Seed root gradient with ones
        {
            let mut root = self.0.borrow_mut();
            for g in &mut root.grad.data {
                *g = 1.0;
            }
        }

        for node in topo.into_iter().rev() {
            let (out_data, out_grad, op) = {
                let inner = node.0.borrow();
                (inner.data.clone(), inner.grad.clone(), inner.op.clone())
            };
            match op {
                GradOp::Leaf => {}
                GradOp::Add(a, b) => {
                    a.accumulate_grad(&out_grad)?;
                    b.accumulate_grad(&out_grad)?;
                }
                GradOp::Sub(a, b) => {
                    a.accumulate_grad(&out_grad)?;
                    b.accumulate_grad(&out_grad.neg())?;
                }
                GradOp::Mul(a, b) => {
                    let ga = out_grad.mul(&b.data())?;
                    let gb = out_grad.mul(&a.data())?;
                    a.accumulate_grad(&ga)?;
                    b.accumulate_grad(&gb)?;
                }
                GradOp::MatMul(a, b) => {
                    let bt = b.data().transpose()?;
                    let at = a.data().transpose()?;
                    let ga = out_grad.matmul(&bt)?;
                    let gb = at.matmul(&out_grad)?;
                    a.accumulate_grad(&ga)?;
                    b.accumulate_grad(&gb)?;
                }
                GradOp::ReLU(a) => {
                    let a_data = a.data();
                    let mask: Vec<f64> = a_data
                        .data
                        .iter()
                        .zip(&out_grad.data)
                        .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
                        .collect();
                    a.accumulate_grad(&Tensor::new(a_data.shape, mask)?)?;
                }
                GradOp::Sigmoid(a) => {
                    let grad_data: Vec<f64> = out_data
                        .data
                        .iter()
                        .zip(&out_grad.data)
                        .map(|(&y, &g)| g * y * (1.0 - y))
                        .collect();
                    a.accumulate_grad(&Tensor::new(out_data.shape, grad_data)?)?;
                }
                GradOp::Tanh(a) => {
                    let grad_data: Vec<f64> = out_data
                        .data
                        .iter()
                        .zip(&out_grad.data)
                        .map(|(&y, &g)| g * (1.0 - y * y))
                        .collect();
                    a.accumulate_grad(&Tensor::new(out_data.shape, grad_data)?)?;
                }
                GradOp::MseLoss(pred, target) => {
                    let p = pred.data();
                    let scale = 2.0 * out_grad.data[0] / (p.data.len().max(1) as f64);
                    let grad_data: Vec<f64> = p
                        .data
                        .iter()
                        .zip(&target.data)
                        .map(|(&pv, &tv)| scale * (pv - tv))
                        .collect();
                    pred.accumulate_grad(&Tensor::new(p.shape, grad_data)?)?;
                }
                GradOp::BceLoss(pred, target) => {
                    let p = pred.data();
                    let eps = 1e-7;
                    let scale = out_grad.data[0] / (p.data.len().max(1) as f64);
                    let grad_data: Vec<f64> = p
                        .data
                        .iter()
                        .zip(&target.data)
                        .map(|(&pv, &tv)| {
                            let pc = pv.clamp(eps, 1.0 - eps);
                            scale * ((pc - tv) / (pc * (1.0 - pc)))
                        })
                        .collect();
                    pred.accumulate_grad(&Tensor::new(p.shape, grad_data)?)?;
                }
            }
        }
        Ok(())
    }
}

impl fmt::Debug for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.0.borrow();
        write!(f, "Var(data={}, grad={})", inner.data, inner.grad)
    }
}

// ============================================================================
// 3. NEURAL NETWORK LAYERS & SEQUENTIAL MODEL
// ============================================================================

#[derive(Debug, Clone)]
pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub weight: Var,
    pub bias: Var,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, seed: u32) -> Self {
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();
        let weight = Var::leaf(Tensor::rand_uniform(
            vec![in_features, out_features],
            seed,
            limit,
        ));
        let bias = Var::leaf(Tensor::zeros(vec![1, out_features]));
        Self {
            in_features,
            out_features,
            weight,
            bias,
        }
    }

    pub fn forward(&self, x: &Var) -> Result<Var, String> {
        x.matmul(&self.weight)?.add(&self.bias)
    }

    pub fn parameters(&self) -> Vec<Var> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationKind {
    ReLU,
    Sigmoid,
    Tanh,
}

#[derive(Debug, Clone)]
pub enum Layer {
    Linear(Linear),
    Activation(ActivationKind),
}

#[derive(Debug, Clone, Default)]
pub struct Sequential {
    pub layers: Vec<Layer>,
}

impl Sequential {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_linear(&mut self, in_features: usize, out_features: usize, seed: u32) -> &mut Self {
        self.layers
            .push(Layer::Linear(Linear::new(in_features, out_features, seed)));
        self
    }

    pub fn add_activation(&mut self, kind: ActivationKind) -> &mut Self {
        self.layers.push(Layer::Activation(kind));
        self
    }

    pub fn forward(&self, x: &Var) -> Result<Var, String> {
        let mut cur = x.clone();
        for layer in &self.layers {
            cur = match layer {
                Layer::Linear(lin) => lin.forward(&cur)?,
                Layer::Activation(ActivationKind::ReLU) => cur.relu(),
                Layer::Activation(ActivationKind::Sigmoid) => cur.sigmoid(),
                Layer::Activation(ActivationKind::Tanh) => cur.tanh(),
            };
        }
        Ok(cur)
    }

    pub fn predict(&self, x: &Tensor) -> Result<Tensor, String> {
        let out = self.forward(&Var::leaf(x.clone()))?;
        Ok(out.data())
    }

    pub fn parameters(&self) -> Vec<Var> {
        let mut params = Vec::new();
        for layer in &self.layers {
            if let Layer::Linear(lin) = layer {
                params.extend(lin.parameters());
            }
        }
        params
    }

    pub fn param_count(&self) -> usize {
        self.parameters().iter().map(|p| p.data().len()).sum()
    }

    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}

// ============================================================================
// 4. OPTIMIZERS (SGD WITH MOMENTUM & ADAM)
// ============================================================================

#[derive(Debug, Clone)]
pub struct Sgd {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    velocities: Vec<Vec<f64>>,
}

impl Sgd {
    pub fn new(lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            weight_decay: 0.0,
            velocities: Vec::new(),
        }
    }

    pub fn step(&mut self, params: &[Var]) {
        if self.velocities.len() != params.len() {
            self.velocities = params.iter().map(|p| vec![0.0; p.data().len()]).collect();
        }
        for (idx, param) in params.iter().enumerate() {
            let mut data = param.data();
            let grad = param.grad();
            let vel = &mut self.velocities[idx];
            for i in 0..data.data.len() {
                let g = grad.data[i] + self.weight_decay * data.data[i];
                vel[i] = self.momentum * vel[i] + g;
                data.data[i] -= self.lr * vel[i];
            }
            param.set_data(data);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    timestep: u64,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            timestep: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn step(&mut self, params: &[Var]) {
        if self.m.len() != params.len() {
            self.m = params.iter().map(|p| vec![0.0; p.data().len()]).collect();
            self.v = params.iter().map(|p| vec![0.0; p.data().len()]).collect();
        }
        self.timestep += 1;
        let t = self.timestep as i32;
        let bc1 = 1.0 - self.beta1.powi(t);
        let bc2 = 1.0 - self.beta2.powi(t);

        for (idx, param) in params.iter().enumerate() {
            let mut data = param.data();
            let grad = param.grad();
            let m = &mut self.m[idx];
            let v = &mut self.v[idx];
            for i in 0..data.data.len() {
                let g = grad.data[i];
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;
                let m_hat = m[i] / bc1;
                let v_hat = v[i] / bc2;
                data.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
            param.set_data(data);
        }
    }
}

#[derive(Debug, Clone)]
pub enum Optimizer {
    Sgd(Sgd),
    Adam(Adam),
}

impl Optimizer {
    pub fn name(&self) -> &'static str {
        match self {
            Optimizer::Sgd(_) => "sgd",
            Optimizer::Adam(_) => "adam",
        }
    }

    pub fn step(&mut self, params: &[Var]) {
        match self {
            Optimizer::Sgd(s) => s.step(params),
            Optimizer::Adam(a) => a.step(params),
        }
    }
}

// ============================================================================
// 5. MODEL TRAINING ENGINE
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossKind {
    Mse,
    Bce,
}

#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub epochs: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub predictions: Vec<f64>,
    pub optimizer: String,
    pub param_count: usize,
}

pub fn train_step(
    model: &Sequential,
    optimizer: &mut Optimizer,
    x: &Tensor,
    y: &Tensor,
    loss_kind: LossKind,
) -> Result<f64, String> {
    model.zero_grad();
    let pred = model.forward(&Var::leaf(x.clone()))?;
    let loss = match loss_kind {
        LossKind::Mse => pred.mse_loss(y)?,
        LossKind::Bce => pred.bce_loss(y)?,
    };
    let loss_val = loss.data().data[0];
    loss.backward()?;
    let params = model.parameters();
    optimizer.step(&params);
    Ok(loss_val)
}

pub fn train_model(
    model: &Sequential,
    optimizer: &mut Optimizer,
    x: &Tensor,
    y: &Tensor,
    epochs: usize,
    loss_kind: LossKind,
) -> Result<TrainingReport, String> {
    let clamped = epochs.clamp(1, 5000);
    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;
    for ep in 0..clamped {
        let loss = train_step(model, optimizer, x, y, loss_kind)?;
        if ep == 0 {
            initial_loss = loss;
        }
        final_loss = loss;
    }
    let preds = model.predict(x)?;
    Ok(TrainingReport {
        epochs: clamped,
        initial_loss: (initial_loss * 1_000_000.0).round() / 1_000_000.0,
        final_loss: (final_loss * 1_000_000.0).round() / 1_000_000.0,
        predictions: preds
            .data
            .into_iter()
            .map(|p| (p * 10_000.0).round() / 10_000.0)
            .collect(),
        optimizer: optimizer.name().to_string(),
        param_count: model.param_count(),
    })
}

/// Standard 2-layer XOR benchmark using Nova's real autograd + Sequential + Adam/SGD stack.
pub fn train_xor(epochs: usize, learning_rate: f64) -> Result<TrainingReport, String> {
    let x = Tensor::new(vec![4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;
    let y = Tensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0])?;
    let mut model = Sequential::new();
    model
        .add_linear(2, 4, 42)
        .add_activation(ActivationKind::Tanh)
        .add_linear(4, 1, 137)
        .add_activation(ActivationKind::Sigmoid);
    let mut opt = Optimizer::Adam(Adam::new(if learning_rate > 0.0 {
        learning_rate.min(0.1)
    } else {
        0.05
    }));
    train_model(&model, &mut opt, &x, &y, epochs, LossKind::Mse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_tensor_ops_and_broadcasting() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Tensor::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);

        let bias = Tensor::new(vec![1, 2], vec![1.0, -1.0]).unwrap();
        let shifted = c.add(&bias).unwrap();
        assert_eq!(shifted.data, vec![20.0, 21.0, 44.0, 49.0]);
        assert_eq!(shifted.argmax(), 3);
    }

    #[test]
    fn autograd_matches_finite_differences_for_matmul_bias_tanh_mse() {
        let x_data = Tensor::new(vec![2, 2], vec![0.5, -0.3, 1.2, 0.7]).unwrap();
        let w_data = Tensor::new(vec![2, 2], vec![0.4, -0.6, 0.2, 0.9]).unwrap();
        let b_data = Tensor::new(vec![1, 2], vec![0.1, -0.2]).unwrap();
        let target = Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        let x = Var::leaf(x_data.clone());
        let w = Var::leaf(w_data.clone());
        let b = Var::leaf(b_data.clone());
        let loss = x
            .matmul(&w)
            .unwrap()
            .add(&b)
            .unwrap()
            .tanh()
            .mse_loss(&target)
            .unwrap();
        loss.backward().unwrap();
        let analytic_dw = w.grad();
        let analytic_db = b.grad();

        let eps = 1e-6;
        for i in 0..w_data.len() {
            let mut wp = w_data.clone();
            wp.data[i] += eps;
            let mut wm = w_data.clone();
            wm.data[i] -= eps;
            let lp = mse_loss(&x_data.matmul(&wp).unwrap().add(&b_data).unwrap().tanh(), &target).unwrap();
            let lm = mse_loss(&x_data.matmul(&wm).unwrap().add(&b_data).unwrap().tanh(), &target).unwrap();
            let numeric = (lp - lm) / (2.0 * eps);
            assert!(
                (analytic_dw.data[i] - numeric).abs() < 1e-5,
                "dW[{i}] mismatch: analytic={} numeric={numeric}",
                analytic_dw.data[i]
            );
        }
        for i in 0..b_data.len() {
            let mut bp = b_data.clone();
            bp.data[i] += eps;
            let mut bm = b_data.clone();
            bm.data[i] -= eps;
            let lp = mse_loss(&x_data.matmul(&w_data).unwrap().add(&bp).unwrap().tanh(), &target).unwrap();
            let lm = mse_loss(&x_data.matmul(&w_data).unwrap().add(&bm).unwrap().tanh(), &target).unwrap();
            let numeric = (lp - lm) / (2.0 * eps);
            assert!(
                (analytic_db.data[i] - numeric).abs() < 1e-5,
                "db[{i}] mismatch: analytic={} numeric={numeric}",
                analytic_db.data[i]
            );
        }
    }

    #[test]
    fn sequential_model_converges_with_adam_and_sgd() {
        let report = train_xor(400, 0.05).unwrap();
        assert!(report.final_loss < 0.02, "final loss was {}", report.final_loss);
        assert!(report.predictions[0] < 0.2);
        assert!(report.predictions[1] > 0.8);
        assert!(report.predictions[2] > 0.8);
        assert!(report.predictions[3] < 0.2);

        // Linear regression with SGD + momentum: y = 2x + 1
        let x = Tensor::new(vec![4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let y = Tensor::new(vec![4, 1], vec![1.0, 3.0, 5.0, 7.0]).unwrap();
        let mut model = Sequential::new();
        model.add_linear(1, 1, 7);
        let mut sgd = Optimizer::Sgd(Sgd::new(0.05, 0.8));
        let lin_report = train_model(&model, &mut sgd, &x, &y, 300, LossKind::Mse).unwrap();
        assert!(lin_report.final_loss < 1e-4, "SGD linear loss {}", lin_report.final_loss);
    }
}
