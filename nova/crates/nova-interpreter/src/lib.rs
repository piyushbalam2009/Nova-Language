//! Nova 0.1 AST Tree-Walking Interpreter (independent of bytecode VM and LLVM).

use nova_parser::{BinOp, Expr, Mutability, Program, Stmt, UnaryOp};
use nova_runtime::{
    dispatch_ai_builtin, dispatch_stdlib_builtin, eval_add, eval_div, eval_matmul, eval_mod,
    eval_mul, eval_neg, eval_ordering, eval_sub, Heap, Value,
};
use std::cmp::Ordering;
use std::collections::HashMap;

const MAX_CALL_DEPTH: usize = 256;
const MAX_STEPS: usize = 250_000;

#[derive(Debug, Clone)]
struct Slot {
    value: Value,
    mutable: bool,
}

#[derive(Debug)]
enum Control {
    Next,
    Break,
    Continue,
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
    call_depth: usize,
    pub steps_executed: usize,
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
            call_depth: 0,
            steps_executed: 0,
            heap: Heap::new(),
            stdout: Vec::new(),
        }
    }

    pub fn run(&mut self, program: &Program) -> Result<(), String> {
        self.steps_executed = 0;
        self.call_depth = 0;
        self.stdout.clear();

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
        let roots = self.collect_roots();
        self.heap.collect(&roots);
        Ok(())
    }

    fn collect_roots(&self) -> Vec<Value> {
        let mut roots = Vec::new();
        for scope in &self.scopes {
            for slot in scope.values() {
                roots.push(slot.value.clone());
            }
        }
        roots
    }

    fn maybe_collect_gc(&mut self) {
        if self.heap.should_collect() {
            let roots = self.collect_roots();
            self.heap.collect(&roots);
        }
    }

    fn tick_step(&mut self) -> Result<(), String> {
        self.steps_executed += 1;
        if self.steps_executed > MAX_STEPS {
            return Err(format!(
                "Execution limit exceeded ({MAX_STEPS} interpreter steps)"
            ));
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
            match self.exec_stmt(s) {
                Ok(Control::Next) => {}
                Ok(other) => {
                    result = other;
                    break;
                }
                Err(err) => {
                    self.scopes.pop();
                    return Err(err);
                }
            }
        }
        self.scopes.pop();
        Ok(result)
    }

    fn exec_stmt(&mut self, stmt: &Stmt) -> Result<Control, String> {
        self.tick_step()?;
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
                self.maybe_collect_gc();
                Ok(Control::Next)
            }
            Stmt::Assign { name, value, .. } => {
                let val = self.eval_expr(value)?;
                self.assign(name, val)?;
                self.maybe_collect_gc();
                Ok(Control::Next)
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
                ..
            } => {
                let c = self.eval_expr(cond)?;
                if self.heap.is_truthy(&c) {
                    self.exec_block(then_body)
                } else if let Some(eb) = else_body {
                    self.exec_block(eb)
                } else {
                    Ok(Control::Next)
                }
            }
            Stmt::While { cond, body, .. } => {
                while {
                    self.tick_step()?;
                    let c = self.eval_expr(cond)?;
                    self.heap.is_truthy(&c)
                } {
                    match self.exec_block(body)? {
                        Control::Next | Control::Continue => {}
                        Control::Break => break,
                        Control::Return(v) => return Ok(Control::Return(v)),
                    }
                    self.maybe_collect_gc();
                }
                Ok(Control::Next)
            }
            Stmt::Break(_) => Ok(Control::Break),
            Stmt::Continue(_) => Ok(Control::Continue),
            Stmt::Return { value, .. } => {
                let v = match value {
                    Some(e) => self.eval_expr(e)?,
                    None => Value::None,
                };
                Ok(Control::Return(v))
            }
            Stmt::Expr(e) => {
                self.eval_expr(e)?;
                self.maybe_collect_gc();
                Ok(Control::Next)
            }
        }
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value, String> {
        match expr {
            Expr::Int(i, _) => Ok(Value::Int(*i)),
            Expr::Float(f, _) => Ok(Value::Float(*f)),
            Expr::Str(s, _) => Ok(self.heap.alloc_string(s.clone())),
            Expr::Char(c, _) => Ok(Value::Char(*c)),
            Expr::Bool(b, _) => Ok(Value::Bool(*b)),
            Expr::None(_) => Ok(Value::None),
            Expr::Ident(name, _) => self
                .lookup(name)
                .ok_or_else(|| format!("Undefined variable '{name}'")),
            Expr::Unary { op, expr, .. } => {
                let v = self.eval_expr(expr)?;
                match op {
                    UnaryOp::Neg => eval_neg(&mut self.heap, &v),
                    UnaryOp::Not => Ok(Value::Bool(!self.heap.is_truthy(&v))),
                }
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                // Short-circuit && and ||
                if matches!(op, BinOp::And) {
                    let l = self.eval_expr(left)?;
                    if !self.heap.is_truthy(&l) {
                        return Ok(Value::Bool(false));
                    }
                    let r = self.eval_expr(right)?;
                    return Ok(Value::Bool(self.heap.is_truthy(&r)));
                }
                if matches!(op, BinOp::Or) {
                    let l = self.eval_expr(left)?;
                    if self.heap.is_truthy(&l) {
                        return Ok(Value::Bool(true));
                    }
                    let r = self.eval_expr(right)?;
                    return Ok(Value::Bool(self.heap.is_truthy(&r)));
                }
                let l = self.eval_expr(left)?;
                let r = self.eval_expr(right)?;
                self.apply_binop(*op, &l, &r)
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
                if let Some(f) = self.functions.get(callee).cloned() {
                    return self.call_user(&f, eval_args);
                }
                if let Some(std_val) = dispatch_stdlib_builtin(&mut self.heap, callee, &eval_args)? {
                    return Ok(std_val);
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
        if self.call_depth >= MAX_CALL_DEPTH {
            return Err(format!(
                "Stack overflow: maximum recursion depth ({MAX_CALL_DEPTH}) exceeded"
            ));
        }
        self.call_depth += 1;

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

        let mut ret = Ok(Value::None);
        for s in &func.body {
            match self.exec_stmt(s) {
                Ok(Control::Next) => {}
                Ok(Control::Return(v)) => {
                    ret = Ok(v);
                    break;
                }
                Ok(Control::Break | Control::Continue) => {
                    ret = Err("Invalid loop control outside of loop".to_string());
                    break;
                }
                Err(err) => {
                    ret = Err(err);
                    break;
                }
            }
        }

        self.scopes.pop();
        self.call_depth -= 1;
        ret
    }

    fn apply_binop(&mut self, op: BinOp, l: &Value, r: &Value) -> Result<Value, String> {
        match op {
            BinOp::Add => eval_add(&mut self.heap, l, r),
            BinOp::Sub => eval_sub(&mut self.heap, l, r),
            BinOp::Mul => eval_mul(&mut self.heap, l, r),
            BinOp::Div => eval_div(&mut self.heap, l, r),
            BinOp::Mod => eval_mod(l, r),
            BinOp::MatMul => eval_matmul(&mut self.heap, l, r),
            BinOp::Eq => Ok(Value::Bool(self.heap.values_equal(l, r))),
            BinOp::NotEq => Ok(Value::Bool(!self.heap.values_equal(l, r))),
            BinOp::Lt => Ok(Value::Bool(
                eval_ordering(&self.heap, l, r)? == Ordering::Less,
            )),
            BinOp::LtEq => Ok(Value::Bool(matches!(
                eval_ordering(&self.heap, l, r)?,
                Ordering::Less | Ordering::Equal
            ))),
            BinOp::Gt => Ok(Value::Bool(
                eval_ordering(&self.heap, l, r)? == Ordering::Greater,
            )),
            BinOp::GtEq => Ok(Value::Bool(matches!(
                eval_ordering(&self.heap, l, r)?,
                Ordering::Greater | Ordering::Equal
            ))),
            BinOp::And | BinOp::Or => unreachable!("short-circuit handled in eval_expr"),
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
    fn respects_lexical_shadowing_string_equality_and_short_circuit() {
        let src = r#"
fn main() {
    let x = 1
    if true {
        let x = 2
        print(x)
    }
    print(x)
    let a = "No" + "va"
    print(a == "Nova")
    print(1 + 2.5)
    if false && (1 / 0 == 0) {
        print("bad")
    }
}
"#;
        let toks = tokenize(src).unwrap();
        let prog = Parser::new(&toks).parse_program().unwrap();
        let mut interp = Interpreter::new();
        interp.run(&prog).unwrap();
        assert_eq!(
            interp.stdout,
            vec!["2", "1", "true", "3.5"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn collects_garbage_during_long_loop() {
        let src = r#"
fn main() {
    mut i = 0
    mut last = ""
    while i < 400 {
        last = "item-" + str(i)
        i += 1
    }
    print(last)
}
"#;
        let toks = tokenize(src).unwrap();
        let prog = Parser::new(&toks).parse_program().unwrap();
        let mut interp = Interpreter::new();
        interp.run(&prog).unwrap();
        assert_eq!(interp.stdout, vec!["item-399".to_string()]);
        assert!(interp.heap.collections_run >= 1);
        assert!(interp.heap.total_freed > 200);
    }
}
