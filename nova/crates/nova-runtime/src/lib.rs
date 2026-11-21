//! Nova Runtime values, Mark-and-Sweep Garbage Collector, shared operators,
//! stdlib dispatch, and AI/ML builtin dispatch used identically by both the
//! AST Interpreter (`nova-interpreter`) and Bytecode VM (`nova-vm`).

use nova_ai::{
    bce_loss, mse_loss, train_model, train_step, train_xor, ActivationKind, Adam, LossKind,
    Optimizer, Sequential, Sgd, Tensor, Var,
};
use std::collections::HashMap;
use std::fmt;

const INITIAL_GC_THRESHOLD: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Char(char),
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

#[derive(Debug)]
pub struct Heap {
    next_id: usize,
    objects: HashMap<usize, HeapEntry>,
    pub total_allocated: usize,
    pub total_freed: usize,
    pub collections_run: usize,
    pub gc_threshold: usize,
}

impl Default for Heap {
    fn default() -> Self {
        Self::new()
    }
}

impl Heap {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            objects: HashMap::new(),
            total_allocated: 0,
            total_freed: 0,
            collections_run: 0,
            gc_threshold: INITIAL_GC_THRESHOLD,
        }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn should_collect(&self) -> bool {
        self.objects.len() >= self.gc_threshold
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

    pub fn as_str(&self, v: &Value) -> Option<&str> {
        if let Value::HeapRef(id) = v {
            if let Some(HeapEntry {
                kind: HeapObjKind::Str(s),
                ..
            }) = self.objects.get(id)
            {
                return Some(s.as_str());
            }
        }
        None
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

    pub fn is_truthy(&self, v: &Value) -> bool {
        match v {
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::None => false,
            _ => true,
        }
    }

    pub fn values_equal(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::None, Value::None) => true,
            (Value::Int(x), Value::Int(y)) => x == y,
            (Value::Float(x), Value::Float(y)) => x == y,
            (Value::Int(x), Value::Float(y)) => (*x as f64) == *y,
            (Value::Float(x), Value::Int(y)) => *x == (*y as f64),
            (Value::Bool(x), Value::Bool(y)) => x == y,
            (Value::Char(x), Value::Char(y)) => x == y,
            (Value::HeapRef(ia), Value::HeapRef(ib)) => {
                if ia == ib {
                    return true;
                }
                match (
                    self.objects.get(ia).map(|e| &e.kind),
                    self.objects.get(ib).map(|e| &e.kind),
                ) {
                    (Some(HeapObjKind::Str(sa)), Some(HeapObjKind::Str(sb))) => sa == sb,
                    (Some(HeapObjKind::Array(aa)), Some(HeapObjKind::Array(ab))) => {
                        aa.len() == ab.len()
                            && aa
                                .iter()
                                .zip(ab.iter())
                                .all(|(x, y)| self.values_equal(x, y))
                    }
                    (Some(HeapObjKind::Tensor(ta)), Some(HeapObjKind::Tensor(tb))) => {
                        ta.shape == tb.shape && ta.data == tb.data
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn value_to_tensor(&self, v: &Value) -> Result<Tensor, String> {
        if let Some(t) = self.as_tensor(v) {
            return Ok(t.clone());
        }
        fn extract(
            heap: &Heap,
            val: &Value,
            depth: usize,
            shape: &mut Vec<usize>,
            flat: &mut Vec<f64>,
        ) -> Result<(), String> {
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
        self.collections_run += 1;
        self.gc_threshold = (self.objects.len().saturating_mul(2)).max(INITIAL_GC_THRESHOLD);
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
            Value::Char(c) => c.to_string(),
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
                }) => format!(
                    "Sequential(layers={}, params={})",
                    m.layers.len(),
                    m.param_count()
                ),
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

fn round_dec(v: f64, decimals: i32) -> f64 {
    let scale = 10_f64.powi(decimals);
    let r = (v * scale).round() / scale;
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

// ============================================================================
// Shared Arithmetic, Tensor, and Comparison Semantics (Interpreter == VM)
// ============================================================================

pub fn eval_add(heap: &mut Heap, l: &Value, r: &Value) -> Result<Value, String> {
    if heap.as_str(l).is_some() || heap.as_str(r).is_some() {
        let s = format!("{}{}", heap.format_value(l), heap.format_value(r));
        return Ok(heap.alloc_string(s));
    }
    if heap.as_tensor(l).is_some() || heap.as_tensor(r).is_some() {
        let ta = match heap.as_tensor(l) {
            Some(t) => t.clone(),
            None => Tensor::scalar(as_f64(l)?),
        };
        let tb = match heap.as_tensor(r) {
            Some(t) => t.clone(),
            None => Tensor::scalar(as_f64(r)?),
        };
        return Ok(heap.alloc_tensor(ta.add(&tb)?));
    }
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => a
            .checked_add(*b)
            .map(Value::Int)
            .ok_or_else(|| "Integer overflow in '+'".to_string()),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + *b as f64)),
        _ => Err(format!("Invalid operands for '+': {l:?} and {r:?}")),
    }
}

pub fn eval_sub(heap: &mut Heap, l: &Value, r: &Value) -> Result<Value, String> {
    if heap.as_tensor(l).is_some() || heap.as_tensor(r).is_some() {
        let ta = match heap.as_tensor(l) {
            Some(t) => t.clone(),
            None => Tensor::scalar(as_f64(l)?),
        };
        let tb = match heap.as_tensor(r) {
            Some(t) => t.clone(),
            None => Tensor::scalar(as_f64(r)?),
        };
        return Ok(heap.alloc_tensor(ta.sub(&tb)?));
    }
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => a
            .checked_sub(*b)
            .map(Value::Int)
            .ok_or_else(|| "Integer overflow in '-'".to_string()),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - *b as f64)),
        _ => Err("Invalid operands for '-'".to_string()),
    }
}

pub fn eval_mul(heap: &mut Heap, l: &Value, r: &Value) -> Result<Value, String> {
    if heap.as_tensor(l).is_some() || heap.as_tensor(r).is_some() {
        let ta = match heap.as_tensor(l) {
            Some(t) => t.clone(),
            None => Tensor::scalar(as_f64(l)?),
        };
        let tb = match heap.as_tensor(r) {
            Some(t) => t.clone(),
            None => Tensor::scalar(as_f64(r)?),
        };
        return Ok(heap.alloc_tensor(ta.mul(&tb)?));
    }
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => a
            .checked_mul(*b)
            .map(Value::Int)
            .ok_or_else(|| "Integer overflow in '*'".to_string()),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * *b as f64)),
        _ => Err("Invalid operands for '*'".to_string()),
    }
}

pub fn eval_div(heap: &mut Heap, l: &Value, r: &Value) -> Result<Value, String> {
    if heap.as_tensor(l).is_some() || heap.as_tensor(r).is_some() {
        let ta = match heap.as_tensor(l) {
            Some(t) => t.clone(),
            None => Tensor::scalar(as_f64(l)?),
        };
        let tb = match heap.as_tensor(r) {
            Some(t) => t.clone(),
            None => Tensor::scalar(as_f64(r)?),
        };
        return Ok(heap.alloc_tensor(ta.div(&tb)?));
    }
    match (l, r) {
        (Value::Int(_), Value::Int(0)) => Err("Division by zero".to_string()),
        (Value::Int(a), Value::Int(b)) => a
            .checked_div(*b)
            .map(Value::Int)
            .ok_or_else(|| "Integer overflow in '/'".to_string()),
        (Value::Float(a), Value::Float(b)) => {
            if *b == 0.0 {
                Err("Division by zero".to_string())
            } else {
                Ok(Value::Float(a / b))
            }
        }
        (Value::Int(a), Value::Float(b)) => {
            if *b == 0.0 {
                Err("Division by zero".to_string())
            } else {
                Ok(Value::Float(*a as f64 / b))
            }
        }
        (Value::Float(a), Value::Int(b)) => {
            if *b == 0 {
                Err("Division by zero".to_string())
            } else {
                Ok(Value::Float(a / *b as f64))
            }
        }
        _ => Err("Invalid operands for '/'".to_string()),
    }
}

pub fn eval_mod(l: &Value, r: &Value) -> Result<Value, String> {
    match (l, r) {
        (Value::Int(_), Value::Int(0)) => Err("Modulo by zero".to_string()),
        (Value::Int(a), Value::Int(b)) => a
            .checked_rem(*b)
            .map(Value::Int)
            .ok_or_else(|| "Integer overflow in '%'".to_string()),
        (Value::Float(a), Value::Float(b)) => {
            if *b == 0.0 {
                Err("Modulo by zero".to_string())
            } else {
                Ok(Value::Float(a % b))
            }
        }
        (Value::Int(a), Value::Float(b)) => {
            if *b == 0.0 {
                Err("Modulo by zero".to_string())
            } else {
                Ok(Value::Float((*a as f64) % b))
            }
        }
        (Value::Float(a), Value::Int(b)) => {
            if *b == 0 {
                Err("Modulo by zero".to_string())
            } else {
                Ok(Value::Float(a % (*b as f64)))
            }
        }
        _ => Err("Invalid operands for '%'".to_string()),
    }
}

pub fn eval_matmul(heap: &mut Heap, l: &Value, r: &Value) -> Result<Value, String> {
    let ta = heap
        .as_tensor(l)
        .ok_or_else(|| "Matrix multiplication '@' requires Tensor operands".to_string())?
        .clone();
    let tb = heap
        .as_tensor(r)
        .ok_or_else(|| "Matrix multiplication '@' requires Tensor operands".to_string())?
        .clone();
    Ok(heap.alloc_tensor(ta.matmul(&tb)?))
}

pub fn eval_neg(heap: &mut Heap, v: &Value) -> Result<Value, String> {
    if let Some(t) = heap.as_tensor(v) {
        let negated = t.neg();
        return Ok(heap.alloc_tensor(negated));
    }
    match v {
        Value::Int(i) => i
            .checked_neg()
            .map(Value::Int)
            .ok_or_else(|| "Integer overflow in unary '-'".to_string()),
        Value::Float(f) => Ok(Value::Float(-f)),
        _ => Err(format!("Cannot negate value {v:?}")),
    }
}

pub fn eval_ordering(heap: &Heap, l: &Value, r: &Value) -> Result<std::cmp::Ordering, String> {
    if let (Some(sa), Some(sb)) = (heap.as_str(l), heap.as_str(r)) {
        return Ok(sa.cmp(sb));
    }
    let lv = as_f64(l).map_err(|_| "Ordering comparison requires numeric or string operands".to_string())?;
    let rv = as_f64(r).map_err(|_| "Ordering comparison requires numeric or string operands".to_string())?;
    lv.partial_cmp(&rv)
        .ok_or_else(|| "Cannot order NaN floating-point values".to_string())
}

/// Shared core stdlib builtin dispatcher (math, string, array, conversion) used by both
/// `nova-interpreter` and `nova-vm`.
pub fn dispatch_stdlib_builtin(
    heap: &mut Heap,
    name: &str,
    args: &[Value],
) -> Result<Option<Value>, String> {
    let base = name.rsplit('.').next().unwrap_or(name);
    match base {
        "str" => {
            let s = args
                .first()
                .map(|v| heap.format_value(v))
                .unwrap_or_else(|| "None".to_string());
            Ok(Some(heap.alloc_string(s)))
        }
        "int" => {
            let Some(arg) = args.first() else {
                return Ok(Some(Value::Int(0)));
            };
            let v = match arg {
                Value::Int(i) => *i,
                Value::Float(f) => *f as i64,
                Value::Bool(b) => i64::from(*b),
                _ => heap
                    .format_value(arg)
                    .trim()
                    .parse::<i64>()
                    .map_err(|_| format!("Cannot convert '{}' to Int", heap.format_value(arg)))?,
            };
            Ok(Some(Value::Int(v)))
        }
        "float" => {
            let Some(arg) = args.first() else {
                return Ok(Some(Value::Float(0.0)));
            };
            let v = match arg {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                _ => heap
                    .format_value(arg)
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| format!("Cannot convert '{}' to Float", heap.format_value(arg)))?,
            };
            Ok(Some(Value::Float(v)))
        }
        "len" => {
            let arg = args.first().ok_or("len() requires 1 argument")?;
            if let Value::HeapRef(id) = arg {
                let n = match heap.get(*id).map(|e| &e.kind) {
                    Some(HeapObjKind::Str(s)) => s.chars().count(),
                    Some(HeapObjKind::Array(els)) => els.len(),
                    Some(HeapObjKind::Tensor(t)) => t.data.len(),
                    _ => return Err("len() requires String, Array, or Tensor".to_string()),
                };
                Ok(Some(Value::Int(n as i64)))
            } else {
                Err("len() requires String, Array, or Tensor".to_string())
            }
        }
        "type_of" => {
            let arg = args.first().ok_or("type_of() requires 1 argument")?;
            let ty_name = match arg {
                Value::Int(_) => "Int",
                Value::Float(_) => "Float",
                Value::Bool(_) => "Bool",
                Value::Char(_) => "Char",
                Value::None => "None",
                Value::HeapRef(id) => match heap.get(*id).map(|e| &e.kind) {
                    Some(HeapObjKind::Str(_)) => "String",
                    Some(HeapObjKind::Array(_)) => "Array",
                    Some(HeapObjKind::Tensor(_)) => "Tensor",
                    Some(HeapObjKind::GradVar(_)) => "GradVar",
                    Some(HeapObjKind::Model(_)) => "Model",
                    Some(HeapObjKind::Optimizer(_)) => "Optimizer",
                    None => "None",
                },
            };
            Ok(Some(heap.alloc_string(ty_name.to_string())))
        }
        "abs" => {
            let v = as_f64(args.first().ok_or("abs() requires 1 argument")?)?;
            Ok(Some(Value::Float(v.abs())))
        }
        "sqrt" => {
            let v = as_f64(args.first().ok_or("sqrt() requires 1 argument")?)?;
            Ok(Some(Value::Float(v.sqrt())))
        }
        "pow" => {
            let a = as_f64(args.first().ok_or("pow() requires 2 arguments")?)?;
            let b = as_f64(args.get(1).ok_or("pow() requires 2 arguments")?)?;
            Ok(Some(Value::Float(a.powf(b))))
        }
        "min" => {
            let a = as_f64(args.first().ok_or("min() requires 2 arguments")?)?;
            let b = as_f64(args.get(1).ok_or("min() requires 2 arguments")?)?;
            Ok(Some(Value::Float(a.min(b))))
        }
        "max" => {
            let a = as_f64(args.first().ok_or("max() requires 2 arguments")?)?;
            let b = as_f64(args.get(1).ok_or("max() requires 2 arguments")?)?;
            Ok(Some(Value::Float(a.max(b))))
        }
        "floor" => {
            let v = as_f64(args.first().ok_or("floor() requires 1 argument")?)?;
            Ok(Some(Value::Int(v.floor() as i64)))
        }
        "ceil" => {
            let v = as_f64(args.first().ok_or("ceil() requires 1 argument")?)?;
            Ok(Some(Value::Int(v.ceil() as i64)))
        }
        "sin" => {
            let v = as_f64(args.first().ok_or("sin() requires 1 argument")?)?;
            Ok(Some(Value::Float(v.sin())))
        }
        "cos" => {
            let v = as_f64(args.first().ok_or("cos() requires 1 argument")?)?;
            Ok(Some(Value::Float(v.cos())))
        }
        "exp" => {
            let v = as_f64(args.first().ok_or("exp() requires 1 argument")?)?;
            Ok(Some(Value::Float(v.exp())))
        }
        "log" => {
            let v = as_f64(args.first().ok_or("log() requires 1 argument")?)?;
            Ok(Some(Value::Float(v.ln())))
        }
        "upper" => {
            let s = heap.format_value(args.first().ok_or("upper() requires 1 argument")?);
            Ok(Some(heap.alloc_string(s.to_uppercase())))
        }
        "lower" => {
            let s = heap.format_value(args.first().ok_or("lower() requires 1 argument")?);
            Ok(Some(heap.alloc_string(s.to_lowercase())))
        }
        "trim" => {
            let s = heap.format_value(args.first().ok_or("trim() requires 1 argument")?);
            Ok(Some(heap.alloc_string(s.trim().to_string())))
        }
        "contains" => {
            let hay = heap.format_value(args.first().ok_or("contains() requires 2 arguments")?);
            let needle = heap.format_value(args.get(1).ok_or("contains() requires 2 arguments")?);
            Ok(Some(Value::Bool(hay.contains(&needle))))
        }
        "range" => {
            let s = as_f64(args.first().ok_or("range() requires 2 arguments")?)? as i64;
            let e = as_f64(args.get(1).ok_or("range() requires 2 arguments")?)? as i64;
            let els = if e > s {
                (s..e).map(Value::Int).collect()
            } else {
                Vec::new()
            };
            Ok(Some(heap.alloc_array(els)))
        }
        "sum" => {
            let arg = args.first().ok_or("sum() requires an Array")?;
            if let Value::HeapRef(id) = arg {
                let total = match heap.get(*id).map(|e| &e.kind) {
                    Some(HeapObjKind::Array(els)) => {
                        let mut acc = 0.0;
                        for el in els {
                            acc += as_f64(el)?;
                        }
                        acc
                    }
                    _ => return Err("sum() requires an Array".to_string()),
                };
                Ok(Some(Value::Float(total)))
            } else {
                Err("sum() requires an Array".to_string())
            }
        }
        _ => Ok(None),
    }
}

/// Shared AI/ML builtin dispatcher used by both `nova-interpreter` and `nova-vm`.
/// Returns `Ok(Some(val))` if handled, `Ok(None)` if `name` is not an AI builtin.
pub fn dispatch_ai_builtin(
    heap: &mut Heap,
    name: &str,
    args: &[Value],
) -> Result<Option<Value>, String> {
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
            Ok(Some(
                heap.alloc_tensor(Tensor::rand_uniform(shape, seed, 1.0)),
            ))
        }
        "matmul" => {
            let a = heap
                .as_tensor(args.first().ok_or("matmul() requires 2 tensors")?)
                .ok_or("Expected Tensor")?
                .clone();
            let b = heap
                .as_tensor(args.get(1).ok_or("matmul() requires 2 tensors")?)
                .ok_or("Expected Tensor")?
                .clone();
            Ok(Some(heap.alloc_tensor(a.matmul(&b)?)))
        }
        "dot" => {
            let a = heap
                .as_tensor(args.first().ok_or("dot() requires 2 tensors")?)
                .ok_or("Expected Tensor")?
                .clone();
            let b = heap
                .as_tensor(args.get(1).ok_or("dot() requires 2 tensors")?)
                .ok_or("Expected Tensor")?
                .clone();
            Ok(Some(Value::Float(a.dot(&b)?)))
        }
        "transpose" => {
            let a = heap
                .as_tensor(args.first().ok_or("transpose() requires tensor")?)
                .ok_or("Expected Tensor")?
                .clone();
            Ok(Some(heap.alloc_tensor(a.transpose()?)))
        }
        "reshape" => {
            let a = heap
                .as_tensor(args.first().ok_or("reshape() requires tensor")?)
                .ok_or("Expected Tensor")?
                .clone();
            let shape =
                heap.value_to_usize_vec(args.get(1).ok_or("reshape() requires shape")?)?;
            Ok(Some(heap.alloc_tensor(a.reshape(shape)?)))
        }
        "relu" => {
            let a = heap
                .as_tensor(args.first().ok_or("relu() requires tensor")?)
                .ok_or("Expected Tensor")?
                .clone();
            Ok(Some(heap.alloc_tensor(a.relu())))
        }
        "sigmoid" => {
            let a = heap
                .as_tensor(args.first().ok_or("sigmoid() requires tensor")?)
                .ok_or("Expected Tensor")?
                .clone();
            Ok(Some(heap.alloc_tensor(a.sigmoid())))
        }
        "tanh" => {
            let a = heap
                .as_tensor(args.first().ok_or("tanh() requires tensor")?)
                .ok_or("Expected Tensor")?
                .clone();
            Ok(Some(heap.alloc_tensor(a.tanh())))
        }
        "softmax" => {
            let a = heap
                .as_tensor(args.first().ok_or("softmax() requires tensor")?)
                .ok_or("Expected Tensor")?
                .clone();
            Ok(Some(heap.alloc_tensor(a.softmax()?)))
        }
        "mse_loss" => {
            let a = heap
                .as_tensor(args.first().ok_or("mse_loss() requires 2 tensors")?)
                .ok_or("Expected Tensor")?
                .clone();
            let b = heap
                .as_tensor(args.get(1).ok_or("mse_loss() requires 2 tensors")?)
                .ok_or("Expected Tensor")?
                .clone();
            Ok(Some(Value::Float(mse_loss(&a, &b)?)))
        }
        "bce_loss" => {
            let a = heap
                .as_tensor(args.first().ok_or("bce_loss() requires 2 tensors")?)
                .ok_or("Expected Tensor")?
                .clone();
            let b = heap
                .as_tensor(args.get(1).ok_or("bce_loss() requires 2 tensors")?)
                .ok_or("Expected Tensor")?
                .clone();
            Ok(Some(Value::Float(bce_loss(&a, &b)?)))
        }
        "grad_var" => {
            let t = heap.value_to_tensor(args.first().ok_or("grad_var() requires tensor")?)?;
            Ok(Some(heap.alloc_grad_var(Var::leaf(t))))
        }
        "grad_add" => {
            let a = heap
                .as_grad_var(args.first().ok_or("grad_add() requires 2 GradVars")?)
                .ok_or("Expected GradVar")?
                .clone();
            let b = heap
                .as_grad_var(args.get(1).ok_or("grad_add() requires 2 GradVars")?)
                .ok_or("Expected GradVar")?
                .clone();
            Ok(Some(heap.alloc_grad_var(a.add(&b)?)))
        }
        "grad_sub" => {
            let a = heap
                .as_grad_var(args.first().ok_or("grad_sub() requires 2 GradVars")?)
                .ok_or("Expected GradVar")?
                .clone();
            let b = heap
                .as_grad_var(args.get(1).ok_or("grad_sub() requires 2 GradVars")?)
                .ok_or("Expected GradVar")?
                .clone();
            Ok(Some(heap.alloc_grad_var(a.sub(&b)?)))
        }
        "grad_mul" => {
            let a = heap
                .as_grad_var(args.first().ok_or("grad_mul() requires 2 GradVars")?)
                .ok_or("Expected GradVar")?
                .clone();
            let b = heap
                .as_grad_var(args.get(1).ok_or("grad_mul() requires 2 GradVars")?)
                .ok_or("Expected GradVar")?
                .clone();
            Ok(Some(heap.alloc_grad_var(a.mul(&b)?)))
        }
        "grad_matmul" => {
            let a = heap
                .as_grad_var(args.first().ok_or("grad_matmul() requires 2 GradVars")?)
                .ok_or("Expected GradVar")?
                .clone();
            let b = heap
                .as_grad_var(args.get(1).ok_or("grad_matmul() requires 2 GradVars")?)
                .ok_or("Expected GradVar")?
                .clone();
            Ok(Some(heap.alloc_grad_var(a.matmul(&b)?)))
        }
        "grad_relu" => {
            let a = heap
                .as_grad_var(args.first().ok_or("grad_relu() requires GradVar")?)
                .ok_or("Expected GradVar")?
                .clone();
            Ok(Some(heap.alloc_grad_var(a.relu())))
        }
        "grad_sigmoid" => {
            let a = heap
                .as_grad_var(args.first().ok_or("grad_sigmoid() requires GradVar")?)
                .ok_or("Expected GradVar")?
                .clone();
            Ok(Some(heap.alloc_grad_var(a.sigmoid())))
        }
        "grad_tanh" => {
            let a = heap
                .as_grad_var(args.first().ok_or("grad_tanh() requires GradVar")?)
                .ok_or("Expected GradVar")?
                .clone();
            Ok(Some(heap.alloc_grad_var(a.tanh())))
        }
        "grad_mse" => {
            let pred = heap
                .as_grad_var(args.first().ok_or("grad_mse() requires GradVar")?)
                .ok_or("Expected GradVar")?
                .clone();
            let target =
                heap.value_to_tensor(args.get(1).ok_or("grad_mse() requires target tensor")?)?;
            Ok(Some(heap.alloc_grad_var(pred.mse_loss(&target)?)))
        }
        "backward" => {
            let v = heap
                .as_grad_var(args.first().ok_or("backward() requires GradVar")?)
                .ok_or("Expected GradVar")?
                .clone();
            v.backward()?;
            Ok(Some(Value::None))
        }
        "grad_of" => {
            let v = heap
                .as_grad_var(args.first().ok_or("grad_of() requires GradVar")?)
                .ok_or("Expected GradVar")?
                .clone();
            Ok(Some(heap.alloc_tensor(v.grad())))
        }
        "value_of" => {
            let v = heap
                .as_grad_var(args.first().ok_or("value_of() requires GradVar")?)
                .ok_or("Expected GradVar")?
                .clone();
            Ok(Some(heap.alloc_tensor(v.data())))
        }
        "zero_grad" => {
            let v = heap
                .as_grad_var(args.first().ok_or("zero_grad() requires GradVar")?)
                .ok_or("Expected GradVar")?
                .clone();
            v.zero_grad();
            Ok(Some(Value::None))
        }
        "autograd_demo" => {
            let xv = as_f64(args.first().ok_or("autograd_demo() requires x and w")?)?;
            let wv = as_f64(args.get(1).ok_or("autograd_demo() requires x and w")?)?;
            let x = Var::scalar(xv);
            let w = Var::scalar(wv);
            let b = Var::scalar(1.5);
            let y = x.mul(&w)?.add(&b)?.relu();
            y.backward()?;
            let summary = format!(
                "{{\"output\": {}, \"dx\": {}, \"dw\": {}, \"db\": {}}}",
                round_dec(y.data().data[0], 4),
                round_dec(x.grad().data[0], 4),
                round_dec(w.grad().data[0], 4),
                round_dec(b.grad().data[0], 4),
            );
            Ok(Some(heap.alloc_string(summary)))
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
            let x =
                heap.value_to_tensor(args.get(1).ok_or("model_forward() requires input tensor")?)?;
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
            Ok(Some(
                heap.alloc_optimizer(Optimizer::Sgd(Sgd::new(lr, momentum))),
            ))
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
            let loss_str = args
                .get(4)
                .map(|v| heap.format_value(v))
                .unwrap_or_else(|| "mse".to_string());
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
            let loss_str = args
                .get(5)
                .map(|v| heap.format_value(v))
                .unwrap_or_else(|| "mse".to_string());
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
                "{{\"epochs\": {}, \"initialLoss\": {}, \"finalLoss\": {}, \"optimizer\": {}, \"paramCount\": {}, \"predictions\": {:?}}}",
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
    fn sweeps_unreachable_strings_and_preserves_nested_arrays() {
        let mut heap = Heap::new();
        let inner = heap.alloc_string("alive".to_string());
        let nested = heap.alloc_array(vec![inner.clone()]);
        for i in 0..100 {
            let _garbage = heap.alloc_string(format!("temp-{i}"));
        }
        let freed = heap.collect(&[nested.clone()]);
        assert_eq!(freed, 100);
        assert_eq!(heap.len(), 2);
        assert_eq!(heap.format_value(&nested), "[alive]");
    }

    #[test]
    fn compares_strings_and_mixed_numbers_by_value() {
        let mut heap = Heap::new();
        let s1 = heap.alloc_string("Nova".to_string());
        let s2 = heap.alloc_string("Nova".to_string());
        assert!(heap.values_equal(&s1, &s2));
        assert!(heap.values_equal(&Value::Int(4), &Value::Float(4.0)));
    }

    #[test]
    fn dispatches_tensor_and_training_builtins() {
        let mut heap = Heap::new();
        let m = dispatch_ai_builtin(&mut heap, "model_new", &[])
            .unwrap()
            .unwrap();
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
