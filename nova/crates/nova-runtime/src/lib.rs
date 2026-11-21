//! Nova Runtime values, Mark-and-Sweep Garbage Collector, and shared AI/ML builtin dispatch
//! used identically by both the AST Interpreter (`nova-interpreter`) and Bytecode VM (`nova-vm`).

use nova_ai::{
    bce_loss, mse_loss, train_model, train_step, train_xor, ActivationKind, Adam, LossKind,
    Optimizer, Sequential, Sgd, Tensor, Var,
};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    None,
    HeapRef(usize),
}

#[derive(Debug, Clone)]
pub enum HeapObjKind {
    Str(String),
    Array(Vec<Value>),
    Tensor(Tensor),
    GradVar(Var),
    Model(Sequential),
    Optimizer(Optimizer),
}

#[derive(Debug, Clone)]
pub struct HeapEntry {
    pub kind: HeapObjKind,
    pub marked: bool,
    pub generation: u8,
}

#[derive(Debug, Default)]
pub struct Heap {
    next_id: usize,
    objects: HashMap<usize, HeapEntry>,
    pub total_allocated: usize,
    pub total_freed: usize,
}

impl Heap {
    pub fn new() -> Self {
        Self::default()
    }

    fn alloc_raw(&mut self, kind: HeapObjKind) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.total_allocated += 1;
        self.objects.insert(
            id,
            HeapEntry {
                kind,
                marked: false,
                generation: 0,
            },
        );
        Value::HeapRef(id)
    }

    pub fn alloc_string(&mut self, s: String) -> Value {
        self.alloc_raw(HeapObjKind::Str(s))
    }

    pub fn alloc_array(&mut self, els: Vec<Value>) -> Value {
        self.alloc_raw(HeapObjKind::Array(els))
    }

    pub fn alloc_tensor(&mut self, t: Tensor) -> Value {
        self.alloc_raw(HeapObjKind::Tensor(t))
    }

    pub fn alloc_grad_var(&mut self, v: Var) -> Value {
        self.alloc_raw(HeapObjKind::GradVar(v))
    }

    pub fn alloc_model(&mut self, m: Sequential) -> Value {
        self.alloc_raw(HeapObjKind::Model(m))
    }

    pub fn alloc_optimizer(&mut self, o: Optimizer) -> Value {
        self.alloc_raw(HeapObjKind::Optimizer(o))
    }

    pub fn get(&self, id: usize) -> Option<&HeapEntry> {
        self.objects.get(&id)
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut HeapEntry> {
        self.objects.get_mut(&id)
    }

    pub fn as_tensor(&self, v: &Value) -> Option<&Tensor> {
        if let Value::HeapRef(id) = v {
            if let Some(HeapEntry {
                kind: HeapObjKind::Tensor(t),
                ..
            }) = self.objects.get(id)
            {
                return Some(t);
            }
        }
        None
    }

    pub fn as_grad_var(&self, v: &Value) -> Option<&Var> {
        if let Value::HeapRef(id) = v {
            if let Some(HeapEntry {
                kind: HeapObjKind::GradVar(gv),
                ..
            }) = self.objects.get(id)
            {
                return Some(gv);
            }
        }
        None
    }

    pub fn value_to_tensor(&self, v: &Value) -> Result<Tensor, String> {
        if let Some(t) = self.as_tensor(v) {
            return Ok(t.clone());
        }
        fn extract(heap: &Heap, val: &Value, depth: usize, shape: &mut Vec<usize>, flat: &mut Vec<f64>) -> Result<(), String> {
            match val {
                Value::Int(i) => {
                    flat.push(*i as f64);
                    Ok(())
                }
                Value::Float(f) => {
                    flat.push(*f);
                    Ok(())
                }
                Value::HeapRef(id) => match heap.get(*id).map(|e| &e.kind) {
                    Some(HeapObjKind::Array(els)) => {
                        if shape.len() == depth {
                            shape.push(els.len());
                        } else if shape[depth] != els.len() {
                            return Err(format!("Ragged tensor shape at dimension {depth}"));
                        }
                        for el in els {
                            extract(heap, el, depth + 1, shape, flat)?;
                        }
                        Ok(())
                    }
                    _ => Err("Expected nested numeric array for tensor()".to_string()),
                },
                _ => Err("Expected numeric value in tensor()".to_string()),
            }
        }
        let mut shape = Vec::new();
        let mut flat = Vec::new();
        extract(self, v, 0, &mut shape, &mut flat)?;
        if shape.is_empty() {
            shape.push(1);
        }
        Tensor::new(shape, flat)
    }

    pub fn value_to_usize_vec(&self, v: &Value) -> Result<Vec<usize>, String> {
        if let Value::HeapRef(id) = v {
            if let Some(HeapEntry {
                kind: HeapObjKind::Array(els),
                ..
            }) = self.objects.get(id)
            {
                let mut out = Vec::with_capacity(els.len());
                for el in els {
                    match el {
                        Value::Int(i) if *i > 0 => out.push(*i as usize),
                        Value::Float(f) if *f > 0.0 => out.push(*f as usize),
                        _ => return Err("Shape dimensions must be positive integers".to_string()),
                    }
                }
                return Ok(out);
            }
        }
        Err("Expected integer shape array".to_string())
    }

    pub fn collect(&mut self, roots: &[Value]) -> usize {
        for obj in self.objects.values_mut() {
            obj.marked = false;
        }
        let mut worklist: Vec<usize> = roots
            .iter()
            .filter_map(|v| match v {
                Value::HeapRef(id) => Some(*id),
                _ => None,
            })
            .collect();

        while let Some(id) = worklist.pop() {
            if let Some(entry) = self.objects.get_mut(&id) {
                if entry.marked {
                    continue;
                }
                entry.marked = true;
                entry.generation = entry.generation.saturating_add(1);
                if let HeapObjKind::Array(els) = &entry.kind {
                    for el in els {
                        if let Value::HeapRef(cid) = el {
                            worklist.push(*cid);
                        }
                    }
                }
            }
        }

        let before = self.objects.len();
        self.objects.retain(|_, obj| obj.marked);
        let freed = before - self.objects.len();
        self.total_freed += freed;
        freed
    }

    pub fn format_value(&self, v: &Value) -> String {
        match v {
            Value::Int(i) => i.to_string(),
            Value::Float(f) => {
                if f.fract() == 0.0 && f.is_finite() {
                    format!("{f:.1}")
                } else {
                    f.to_string()
                }
            }
            Value::Bool(b) => b.to_string(),
            Value::None => "None".to_string(),
            Value::HeapRef(id) => match self.objects.get(id) {
                Some(HeapEntry {
                    kind: HeapObjKind::Str(s),
                    ..
                }) => s.clone(),
                Some(HeapEntry {
                    kind: HeapObjKind::Array(els),
                    ..
                }) => {
                    let parts: Vec<String> = els.iter().map(|e| self.format_value(e)).collect();
                    format!("[{}]", parts.join(", "))
                }
                Some(HeapEntry {
                    kind: HeapObjKind::Tensor(t),
                    ..
                }) => t.to_string(),
                Some(HeapEntry {
                    kind: HeapObjKind::GradVar(gv),
                    ..
                }) => format!("GradVar(data={}, grad={})", gv.data(), gv.grad()),
                Some(HeapEntry {
                    kind: HeapObjKind::Model(m),
                    ..
                }) => format!("Sequential(layers={}, params={})", m.layers.len(), m.param_count()),
                Some(HeapEntry {
                    kind: HeapObjKind::Optimizer(o),
                    ..
                }) => format!("Optimizer({})", o.name()),
                None => "<freed>".to_string(),
            },
        }
    }
}

fn as_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Int(i) => Ok(*i as f64),
        Value::Float(f) => Ok(*f),
        _ => Err(format!("Expected numeric argument, got {v:?}")),
    }
}

fn as_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(i) if *i >= 0 => Ok(*i as usize),
        Value::Float(f) if *f >= 0.0 => Ok(*f as usize),
        _ => Err(format!("Expected non-negative integer, got {v:?}")),
    }
}

/// Shared AI/ML builtin dispatcher used by both `nova-interpreter` and `nova-vm`.
/// Returns `Ok(Some(val))` if handled, `Ok(None)` if `name` is not an AI builtin.
pub fn dispatch_ai_builtin(heap: &mut Heap, name: &str, args: &[Value]) -> Result<Option<Value>, String> {
    let base = name.rsplit('.').next().unwrap_or(name);
    match base {
        "tensor" => {
            let t = heap.value_to_tensor(args.first().ok_or("tensor() requires 1 argument")?)?;
            Ok(Some(heap.alloc_tensor(t)))
        }
        "zeros" => {
            let shape = heap.value_to_usize_vec(args.first().ok_or("zeros() requires shape")?)?;
            Ok(Some(heap.alloc_tensor(Tensor::zeros(shape))))
        }
        "ones" => {
            let shape = heap.value_to_usize_vec(args.first().ok_or("ones() requires shape")?)?;
            Ok(Some(heap.alloc_tensor(Tensor::ones(shape))))
        }
        "full" => {
            let shape = heap.value_to_usize_vec(args.first().ok_or("full() requires shape")?)?;
            let val = as_f64(args.get(1).ok_or("full() requires fill value")?)?;
            Ok(Some(heap.alloc_tensor(Tensor::full(shape, val))))
        }
        "randn" => {
            let shape = heap.value_to_usize_vec(args.first().ok_or("randn() requires shape")?)?;
            let seed = args.get(1).map(as_usize).transpose()?.unwrap_or(42) as u32;
            Ok(Some(heap.alloc_tensor(Tensor::rand_uniform(shape, seed, 1.0))))
        }
        "matmul" => {
            let a = heap.as_tensor(args.first().ok_or("matmul() requires 2 tensors")?).ok_or("Expected Tensor")?.clone();
            let b = heap.as_tensor(args.get(1).ok_or("matmul() requires 2 tensors")?).ok_or("Expected Tensor")?.clone();
            Ok(Some(heap.alloc_tensor(a.matmul(&b)?)))
        }
        "dot" => {
            let a = heap.as_tensor(args.first().ok_or("dot() requires 2 tensors")?).ok_or("Expected Tensor")?.clone();
            let b = heap.as_tensor(args.get(1).ok_or("dot() requires 2 tensors")?).ok_or("Expected Tensor")?.clone();
            Ok(Some(Value::Float(a.dot(&b)?)))
        }
        "transpose" => {
            let a = heap.as_tensor(args.first().ok_or("transpose() requires tensor")?).ok_or("Expected Tensor")?.clone();
            Ok(Some(heap.alloc_tensor(a.transpose()?)))
        }
        "reshape" => {
            let a = heap.as_tensor(args.first().ok_or("reshape() requires tensor")?).ok_or("Expected Tensor")?.clone();
            let shape = heap.value_to_usize_vec(args.get(1).ok_or("reshape() requires shape")?)?;
            Ok(Some(heap.alloc_tensor(a.reshape(shape)?)))
        }
        "relu" => {
            let a = heap.as_tensor(args.first().ok_or("relu() requires tensor")?).ok_or("Expected Tensor")?.clone();
            Ok(Some(heap.alloc_tensor(a.relu())))
        }
        "sigmoid" => {
            let a = heap.as_tensor(args.first().ok_or("sigmoid() requires tensor")?).ok_or("Expected Tensor")?.clone();
            Ok(Some(heap.alloc_tensor(a.sigmoid())))
        }
        "tanh" => {
            let a = heap.as_tensor(args.first().ok_or("tanh() requires tensor")?).ok_or("Expected Tensor")?.clone();
            Ok(Some(heap.alloc_tensor(a.tanh())))
        }
        "softmax" => {
            let a = heap.as_tensor(args.first().ok_or("softmax() requires tensor")?).ok_or("Expected Tensor")?.clone();
            Ok(Some(heap.alloc_tensor(a.softmax()?)))
        }
        "mse_loss" => {
            let a = heap.as_tensor(args.first().ok_or("mse_loss() requires 2 tensors")?).ok_or("Expected Tensor")?.clone();
            let b = heap.as_tensor(args.get(1).ok_or("mse_loss() requires 2 tensors")?).ok_or("Expected Tensor")?.clone();
            Ok(Some(Value::Float(mse_loss(&a, &b)?)))
        }
        "bce_loss" => {
            let a = heap.as_tensor(args.first().ok_or("bce_loss() requires 2 tensors")?).ok_or("Expected Tensor")?.clone();
            let b = heap.as_tensor(args.get(1).ok_or("bce_loss() requires 2 tensors")?).ok_or("Expected Tensor")?.clone();
            Ok(Some(Value::Float(bce_loss(&a, &b)?)))
        }
        "grad_var" => {
            let t = heap.value_to_tensor(args.first().ok_or("grad_var() requires tensor")?)?;
            Ok(Some(heap.alloc_grad_var(Var::leaf(t))))
        }
        "grad_add" => {
            let a = heap.as_grad_var(args.first().ok_or("grad_add() requires 2 GradVars")?).ok_or("Expected GradVar")?.clone();
            let b = heap.as_grad_var(args.get(1).ok_or("grad_add() requires 2 GradVars")?).ok_or("Expected GradVar")?.clone();
            Ok(Some(heap.alloc_grad_var(a.add(&b)?)))
        }
        "grad_sub" => {
            let a = heap.as_grad_var(args.first().ok_or("grad_sub() requires 2 GradVars")?).ok_or("Expected GradVar")?.clone();
            let b = heap.as_grad_var(args.get(1).ok_or("grad_sub() requires 2 GradVars")?).ok_or("Expected GradVar")?.clone();
            Ok(Some(heap.alloc_grad_var(a.sub(&b)?)))
        }
        "grad_mul" => {
            let a = heap.as_grad_var(args.first().ok_or("grad_mul() requires 2 GradVars")?).ok_or("Expected GradVar")?.clone();
            let b = heap.as_grad_var(args.get(1).ok_or("grad_mul() requires 2 GradVars")?).ok_or("Expected GradVar")?.clone();
            Ok(Some(heap.alloc_grad_var(a.mul(&b)?)))
        }
        "grad_matmul" => {
            let a = heap.as_grad_var(args.first().ok_or("grad_matmul() requires 2 GradVars")?).ok_or("Expected GradVar")?.clone();
            let b = heap.as_grad_var(args.get(1).ok_or("grad_matmul() requires 2 GradVars")?).ok_or("Expected GradVar")?.clone();
            Ok(Some(heap.alloc_grad_var(a.matmul(&b)?)))
        }
        "grad_relu" => {
            let a = heap.as_grad_var(args.first().ok_or("grad_relu() requires GradVar")?).ok_or("Expected GradVar")?.clone();
            Ok(Some(heap.alloc_grad_var(a.relu())))
        }
        "grad_sigmoid" => {
            let a = heap.as_grad_var(args.first().ok_or("grad_sigmoid() requires GradVar")?).ok_or("Expected GradVar")?.clone();
            Ok(Some(heap.alloc_grad_var(a.sigmoid())))
        }
        "grad_tanh" => {
            let a = heap.as_grad_var(args.first().ok_or("grad_tanh() requires GradVar")?).ok_or("Expected GradVar")?.clone();
            Ok(Some(heap.alloc_grad_var(a.tanh())))
        }
        "grad_mse" => {
            let pred = heap.as_grad_var(args.first().ok_or("grad_mse() requires GradVar")?).ok_or("Expected GradVar")?.clone();
            let target = heap.value_to_tensor(args.get(1).ok_or("grad_mse() requires target tensor")?)?;
            Ok(Some(heap.alloc_grad_var(pred.mse_loss(&target)?)))
        }
        "backward" => {
            let v = heap.as_grad_var(args.first().ok_or("backward() requires GradVar")?).ok_or("Expected GradVar")?.clone();
            v.backward()?;
            Ok(Some(Value::None))
        }
        "grad_of" => {
            let v = heap.as_grad_var(args.first().ok_or("grad_of() requires GradVar")?).ok_or("Expected GradVar")?.clone();
            Ok(Some(heap.alloc_tensor(v.grad())))
        }
        "value_of" => {
            let v = heap.as_grad_var(args.first().ok_or("value_of() requires GradVar")?).ok_or("Expected GradVar")?.clone();
            Ok(Some(heap.alloc_tensor(v.data())))
        }
        "zero_grad" => {
            let v = heap.as_grad_var(args.first().ok_or("zero_grad() requires GradVar")?).ok_or("Expected GradVar")?.clone();
            v.zero_grad();
            Ok(Some(Value::None))
        }
        "model_new" => Ok(Some(heap.alloc_model(Sequential::new()))),
        "model_add_linear" => {
            let model_id = match args.first() {
                Some(Value::HeapRef(id)) => *id,
                _ => return Err("model_add_linear() requires a Sequential model".to_string()),
            };
            let in_f = as_usize(args.get(1).ok_or("Missing in_features")?)?;
            let out_f = as_usize(args.get(2).ok_or("Missing out_features")?)?;
            let seed = args.get(3).map(as_usize).transpose()?.unwrap_or(42) as u32;
            if let Some(HeapEntry {
                kind: HeapObjKind::Model(m),
                ..
            }) = heap.get_mut(model_id)
            {
                m.add_linear(in_f, out_f, seed);
                Ok(Some(Value::HeapRef(model_id)))
            } else {
                Err("Expected Sequential model".to_string())
            }
        }
        "model_add_activation" => {
            let model_id = match args.first() {
                Some(Value::HeapRef(id)) => *id,
                _ => return Err("model_add_activation() requires a Sequential model".to_string()),
            };
            let act_name = heap.format_value(args.get(1).ok_or("Missing activation name")?);
            let kind = match act_name.to_lowercase().as_str() {
                "relu" => ActivationKind::ReLU,
                "sigmoid" => ActivationKind::Sigmoid,
                "tanh" => ActivationKind::Tanh,
                other => return Err(format!("Unsupported activation '{other}'")),
            };
            if let Some(HeapEntry {
                kind: HeapObjKind::Model(m),
                ..
            }) = heap.get_mut(model_id)
            {
                m.add_activation(kind);
                Ok(Some(Value::HeapRef(model_id)))
            } else {
                Err("Expected Sequential model".to_string())
            }
        }
        "model_forward" => {
            let model_id = match args.first() {
                Some(Value::HeapRef(id)) => *id,
                _ => return Err("model_forward() requires a Sequential model".to_string()),
            };
            let x = heap.value_to_tensor(args.get(1).ok_or("model_forward() requires input tensor")?)?;
            let out = match heap.get(model_id).map(|e| &e.kind) {
                Some(HeapObjKind::Model(m)) => m.predict(&x)?,
                _ => return Err("Expected Sequential model".to_string()),
            };
            Ok(Some(heap.alloc_tensor(out)))
        }
        "model_param_count" => {
            let model_id = match args.first() {
                Some(Value::HeapRef(id)) => *id,
                _ => return Err("model_param_count() requires a Sequential model".to_string()),
            };
            let count = match heap.get(model_id).map(|e| &e.kind) {
                Some(HeapObjKind::Model(m)) => m.param_count(),
                _ => return Err("Expected Sequential model".to_string()),
            };
            Ok(Some(Value::Int(count as i64)))
        }
        "optimizer_sgd" => {
            let lr = as_f64(args.first().ok_or("optimizer_sgd() requires lr")?)?;
            let momentum = args.get(1).map(as_f64).transpose()?.unwrap_or(0.0);
            Ok(Some(heap.alloc_optimizer(Optimizer::Sgd(Sgd::new(lr, momentum)))))
        }
        "optimizer_adam" => {
            let lr = as_f64(args.first().ok_or("optimizer_adam() requires lr")?)?;
            Ok(Some(heap.alloc_optimizer(Optimizer::Adam(Adam::new(lr)))))
        }
        "train_step" => {
            let model_id = match args.first() {
                Some(Value::HeapRef(id)) => *id,
                _ => return Err("train_step() requires model".to_string()),
            };
            let opt_id = match args.get(1) {
                Some(Value::HeapRef(id)) => *id,
                _ => return Err("train_step() requires optimizer".to_string()),
            };
            let x = heap.value_to_tensor(args.get(2).ok_or("train_step() requires X")?)?;
            let y = heap.value_to_tensor(args.get(3).ok_or("train_step() requires Y")?)?;
            let loss_str = args.get(4).map(|v| heap.format_value(v)).unwrap_or_else(|| "mse".to_string());
            let loss_kind = if loss_str.eq_ignore_ascii_case("bce") {
                LossKind::Bce
            } else {
                LossKind::Mse
            };
            let model = match heap.get(model_id).map(|e| &e.kind) {
                Some(HeapObjKind::Model(m)) => m.clone(),
                _ => return Err("Expected Sequential model".to_string()),
            };
            let loss_val = match heap.get_mut(opt_id).map(|e| &mut e.kind) {
                Some(HeapObjKind::Optimizer(opt)) => train_step(&model, opt, &x, &y, loss_kind)?,
                _ => return Err("Expected Optimizer".to_string()),
            };
            Ok(Some(Value::Float(loss_val)))
        }
        "train_model" => {
            let model_id = match args.first() {
                Some(Value::HeapRef(id)) => *id,
                _ => return Err("train_model() requires model".to_string()),
            };
            let opt_id = match args.get(1) {
                Some(Value::HeapRef(id)) => *id,
                _ => return Err("train_model() requires optimizer".to_string()),
            };
            let x = heap.value_to_tensor(args.get(2).ok_or("train_model() requires X")?)?;
            let y = heap.value_to_tensor(args.get(3).ok_or("train_model() requires Y")?)?;
            let epochs = as_usize(args.get(4).ok_or("train_model() requires epochs")?)?;
            let loss_str = args.get(5).map(|v| heap.format_value(v)).unwrap_or_else(|| "mse".to_string());
            let loss_kind = if loss_str.eq_ignore_ascii_case("bce") {
                LossKind::Bce
            } else {
                LossKind::Mse
            };
            let model = match heap.get(model_id).map(|e| &e.kind) {
                Some(HeapObjKind::Model(m)) => m.clone(),
                _ => return Err("Expected Sequential model".to_string()),
            };
            let report = match heap.get_mut(opt_id).map(|e| &mut e.kind) {
                Some(HeapObjKind::Optimizer(opt)) => {
                    train_model(&model, opt, &x, &y, epochs, loss_kind)?
                }
                _ => return Err("Expected Optimizer".to_string()),
            };
            let summary = format!(
                "{{\"epochs\": {}, \"initialLoss\": {}, \"finalLoss\": {}, \"optimizer\": \"{}\", \"paramCount\": {}, \"predictions\": {:?}}}",
                report.epochs,
                report.initial_loss,
                report.final_loss,
                report.optimizer,
                report.param_count,
                report.predictions
            );
            Ok(Some(heap.alloc_string(summary)))
        }
        "train_xor" => {
            let epochs = as_usize(args.first().ok_or("train_xor() requires epochs")?)?;
            let lr = as_f64(args.get(1).ok_or("train_xor() requires lr")?)?;
            let report = train_xor(epochs, lr)?;
            let summary = format!(
                "{{\"epochs\": {}, \"initialLoss\": {}, \"finalLoss\": {}, \"predictions\": {:?}}}",
                report.epochs, report.initial_loss, report.final_loss, report.predictions
            );
            Ok(Some(heap.alloc_string(summary)))
        }
        _ => Ok(None),
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweeps_unreachable_strings() {
        let mut heap = Heap::new();
        let kept = heap.alloc_string("Nova".to_string());
        let _garbage = heap.alloc_string("unused".to_string());
        let freed = heap.collect(&[kept]);
        assert_eq!(freed, 1);
    }

    #[test]
    fn dispatches_tensor_and_training_builtins() {
        let mut heap = Heap::new();
        let m = dispatch_ai_builtin(&mut heap, "model_new", &[]).unwrap().unwrap();
        dispatch_ai_builtin(
            &mut heap,
            "model_add_linear",
            &[m.clone(), Value::Int(2), Value::Int(4), Value::Int(42)],
        )
        .unwrap();
        let count = dispatch_ai_builtin(&mut heap, "model_param_count", &[m])
            .unwrap()
            .unwrap();
        assert_eq!(count, Value::Int(12));
    }
}
