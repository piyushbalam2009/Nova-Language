//! Nova 0.1 AST Tree-Walking Interpreter (independent of bytecode VM and LLVM).

use nova_parser::{BinOp, Expr, Mutability, Program, Stmt};
use nova_runtime::{dispatch_ai_builtin, Heap, Value};
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Slot {
    value: Value,
    mutable: bool,
}

#[derive(Debug)]
enum Control {
    Next,
    Return(Value),
}

#[derive(Debug, Clone)]
struct UserFn {
    params: Vec<String>,
    body: Vec<Stmt>,
}

pub struct Interpreter {
    scopes: Vec<HashMap<String, Slot>>,
    functions: HashMap<String, UserFn>,
    pub heap: Heap,
    pub stdout: Vec<String>,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            heap: Heap::new(),
            stdout: Vec::new(),
        }
    }

    pub fn run(&mut self, program: &Program) -> Result<(), String> {
        for item in &program.items {
            if let Stmt::Fn {
                name, params, body, ..
            } = item
            {
                self.functions.insert(
                    name.clone(),
                    UserFn {
                        params: params.iter().map(|(n, _)| n.clone()).collect(),
                        body: body.clone(),
                    },
                );
            }
        }

        for item in &program.items {
            if !matches!(
                item,
                Stmt::Fn { .. } | Stmt::Struct { .. } | Stmt::Use { .. }
            ) {
                self.exec_stmt(item)?;
            }
        }

        if let Some(main_fn) = self.functions.get("main").cloned() {
            self.call_user(&main_fn, Vec::new())?;
        }
        Ok(())
    }

    fn lookup(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(slot) = scope.get(name) {
                return Some(slot.value.clone());
            }
        }
        None
    }

    fn assign(&mut self, name: &str, value: Value) -> Result<(), String> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(slot) = scope.get_mut(name) {
                if !slot.mutable {
                    return Err(format!("Cannot assign to immutable binding '{name}'"));
                }
                slot.value = value;
                return Ok(());
            }
        }
        Err(format!("Undefined variable '{name}'"))
    }

    fn exec_block(&mut self, stmts: &[Stmt]) -> Result<Control, String> {
        self.scopes.push(HashMap::new());
        let mut result = Control::Next;
        for s in stmts {
            match self.exec_stmt(s)? {
                Control::Next => {}
                Control::Return(v) => {
                    result = Control::Return(v);
                    break;
                }
            }
        }
        self.scopes.pop();
        Ok(result)
    }

    fn exec_stmt(&mut self, stmt: &Stmt) -> Result<Control, String> {
        match stmt {
            Stmt::Use { .. } | Stmt::Struct { .. } | Stmt::Fn { .. } => Ok(Control::Next),
            Stmt::VarDecl {
                mutability,
                name,
                init,
                ..
            } => {
                let val = self.eval_expr(init)?;
                if let Some(scope) = self.scopes.last_mut() {
                    scope.insert(
                        name.clone(),
                        Slot {
                            value: val,
                            mutable: matches!(mutability, Mutability::Mut),
                        },
                    );
                }
                Ok(Control::Next)
            }
            Stmt::Assign { name, value, .. } => {
                let val = self.eval_expr(value)?;
                self.assign(name, val)?;
                Ok(Control::Next)
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
                ..
            } => {
                let c = self.eval_expr(cond)?;
                if self.is_truthy(&c) {
                    self.exec_block(then_body)
                } else if let Some(eb) = else_body {
                    self.exec_block(eb)
                } else {
                    Ok(Control::Next)
                }
            }
            Stmt::While { cond, body, .. } => {
                while {
                    let c = self.eval_expr(cond)?;
                    self.is_truthy(&c)
                } {
                    if let Control::Return(v) = self.exec_block(body)? {
                        return Ok(Control::Return(v));
                    }
                }
                Ok(Control::Next)
            }
            Stmt::Return { value, .. } => {
                let v = match value {
                    Some(e) => self.eval_expr(e)?,
                    None => Value::None,
                };
                Ok(Control::Return(v))
            }
            Stmt::Expr(e) => {
                self.eval_expr(e)?;
                Ok(Control::Next)
            }
        }
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value, String> {
        match expr {
            Expr::Int(i, _) => Ok(Value::Int(*i)),
            Expr::Float(f, _) => Ok(Value::Float(*f)),
            Expr::Str(s, _) => Ok(self.heap.alloc_string(s.clone())),
            Expr::Bool(b, _) => Ok(Value::Bool(*b)),
            Expr::None(_) => Ok(Value::None),
            Expr::Ident(name, _) => self
                .lookup(name)
                .ok_or_else(|| format!("Undefined variable '{name}'")),
            Expr::Binary {
                op, left, right, ..
            } => {
                // Short-circuit && and ||
                if matches!(op, BinOp::And) {
                    let l = self.eval_expr(left)?;
                    if !self.is_truthy(&l) {
                        return Ok(Value::Bool(false));
                    }
                    let r = self.eval_expr(right)?;
                    return Ok(Value::Bool(self.is_truthy(&r)));
                }
                if matches!(op, BinOp::Or) {
                    let l = self.eval_expr(left)?;
                    if self.is_truthy(&l) {
                        return Ok(Value::Bool(true));
                    }
                    let r = self.eval_expr(right)?;
                    return Ok(Value::Bool(self.is_truthy(&r)));
                }
                let l = self.eval_expr(left)?;
                let r = self.eval_expr(right)?;
                self.apply_binop(*op, l, r)
            }
            Expr::Call { callee, args, .. } => {
                let mut eval_args = Vec::with_capacity(args.len());
                for a in args {
                    eval_args.push(self.eval_expr(a)?);
                }
                if callee == "print" {
                    let line: Vec<String> = eval_args
                        .iter()
                        .map(|v| self.heap.format_value(v))
                        .collect();
                    self.stdout.push(line.join(" "));
                    return Ok(Value::None);
                }
                if callee == "str" {
                    let s = eval_args
                        .first()
                        .map(|v| self.heap.format_value(v))
                        .unwrap_or_default();
                    return Ok(self.heap.alloc_string(s));
                }
                if let Some(f) = self.functions.get(callee).cloned() {
                    return self.call_user(&f, eval_args);
                }
                if let Some(ai_val) = dispatch_ai_builtin(&mut self.heap, callee, &eval_args)? {
                    return Ok(ai_val);
                }
                Err(format!("Undefined function '{callee}'"))
            }
            Expr::Array(items, _) => {
                let mut els = Vec::with_capacity(items.len());
                for item in items {
                    els.push(self.eval_expr(item)?);
                }
                Ok(self.heap.alloc_array(els))
            }
        }
    }

    fn call_user(&mut self, func: &UserFn, args: Vec<Value>) -> Result<Value, String> {
        let mut fn_scope = HashMap::new();
        for (idx, param) in func.params.iter().enumerate() {
            let val = args.get(idx).cloned().unwrap_or(Value::None);
            fn_scope.insert(
                param.clone(),
                Slot {
                    value: val,
                    mutable: false,
                },
            );
        }
        self.scopes.push(fn_scope);
        let mut ret = Value::None;
        for s in &func.body {
            if let Control::Return(v) = self.exec_stmt(s)? {
                ret = v;
                break;
            }
        }
        self.scopes.pop();
        Ok(ret)
    }

    fn is_truthy(&self, v: &Value) -> bool {
        match v {
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::None => false,
            _ => true,
        }
    }

    fn apply_binop(&mut self, op: BinOp, l: Value, r: Value) -> Result<Value, String> {
        match op {
            BinOp::Add => match (&l, &r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                _ => {
                    let s = format!(
                        "{}{}",
                        self.heap.format_value(&l),
                        self.heap.format_value(&r)
                    );
                    Ok(self.heap.alloc_string(s))
                }
            },
            BinOp::Sub => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
                _ => Err("Invalid operands for '-'".to_string()),
            },
            BinOp::Mul => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
                _ => Err("Invalid operands for '*'".to_string()),
            },
            BinOp::Div => match (l, r) {
                (Value::Int(_), Value::Int(0)) => Err("Division by zero".to_string()),
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
                (Value::Float(a), Value::Float(b)) if b != 0.0 => Ok(Value::Float(a / b)),
                _ => Err("Division by zero or invalid operands".to_string()),
            },
            BinOp::Mod => match (l, r) {
                (Value::Int(_), Value::Int(0)) => Err("Modulo by zero".to_string()),
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a % b)),
                _ => Err("Invalid operands for '%'".to_string()),
            },
            BinOp::Eq => Ok(Value::Bool(l == r)),
            BinOp::NotEq => Ok(Value::Bool(l != r)),
            BinOp::Lt => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
                _ => Err("Invalid operands for '<'".to_string()),
            },
            BinOp::LtEq => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
                _ => Err("Invalid operands for '<='".to_string()),
            },
            BinOp::Gt => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
                _ => Err("Invalid operands for '>'".to_string()),
            },
            BinOp::GtEq => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
                _ => Err("Invalid operands for '>='".to_string()),
            },
            _ => Err(format!("Unsupported binary operator {op:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nova_lexer::tokenize;
    use nova_parser::Parser;

    #[test]
    fn executes_hello_nova() {
        let src = "fn main() {\n    let name = \"Nova\"\n    print(\"Hello, \" + name)\n}";
        let toks = tokenize(src).unwrap();
        let prog = Parser::new(&toks).parse_program().unwrap();
        let mut interp = Interpreter::new();
        interp.run(&prog).unwrap();
        assert_eq!(interp.stdout, vec!["Hello, Nova".to_string()]);
    }

    #[test]
    fn respects_lexical_shadowing_and_short_circuit() {
        let src = "fn main() {\n    let x = 1\n    if true {\n        let x = 2\n        print(x)\n    }\n    print(x)\n    if false && (1 / 0 == 0) {\n        print(\"bad\")\n    }\n}";
        let toks = tokenize(src).unwrap();
        let prog = Parser::new(&toks).parse_program().unwrap();
        let mut interp = Interpreter::new();
        interp.run(&prog).unwrap();
        assert_eq!(interp.stdout, vec!["2".to_string(), "1".to_string()]);
    }
}
