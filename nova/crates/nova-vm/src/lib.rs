//! Nova Virtual Machine: verified stack-based bytecode interpreter with explicit IP,
//! GC-rooted call frames, globals, jumps, recursion guards, and mark-and-sweep GC heap.

use nova_bytecode::{verify, Chunk, Op};
use nova_ir::ConstVal;
use nova_runtime::{
    dispatch_ai_builtin, dispatch_stdlib_builtin, eval_add, eval_div, eval_matmul, eval_mod,
    eval_mul, eval_neg, eval_ordering, eval_sub, Heap, Value,
};
use std::cmp::Ordering;
use std::collections::HashMap;

const MAX_CALL_DEPTH: usize = 256;
const MAX_INSTRUCTIONS: usize = 250_000;

#[derive(Debug, Default)]
pub struct Vm {
    stack: Vec<Value>,
    frames: Vec<Vec<Value>>,
    globals: HashMap<String, Value>,
    pub heap: Heap,
    pub stdout: Vec<String>,
    pub instructions_executed: usize,
}

impl Vm {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn execute(&mut self, chunk: &Chunk) -> Result<(), String> {
        verify(chunk).map_err(|errs| {
            errs.into_iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        })?;

        self.stack.clear();
        self.frames.clear();
        self.globals.clear();
        self.stdout.clear();
        self.instructions_executed = 0;

        if let Some(init_idx) = chunk.functions.iter().position(|f| f.name == "__init__") {
            self.call_fn(chunk, init_idx, Vec::new(), 0)?;
        }

        if let Some(main_idx) = chunk.functions.iter().position(|f| f.name == "main") {
            self.call_fn(chunk, main_idx, Vec::new(), 0)?;
        } else if !chunk.functions.iter().any(|f| f.name == "__init__") {
            return Err("Entry function 'main' not found".to_string());
        }

        self.collect_gc();
        Ok(())
    }

    fn collect_gc(&mut self) {
        let mut roots = Vec::with_capacity(
            self.stack.len()
                + self.globals.len()
                + self.frames.iter().map(Vec::len).sum::<usize>(),
        );
        roots.extend(self.stack.iter().cloned());
        for frame in &self.frames {
            roots.extend(frame.iter().cloned());
        }
        roots.extend(self.globals.values().cloned());
        self.heap.collect(&roots);
    }

    fn maybe_collect_gc(&mut self) {
        if self.heap.should_collect() {
            self.collect_gc();
        }
    }

    fn call_fn(
        &mut self,
        chunk: &Chunk,
        fn_idx: usize,
        args: Vec<Value>,
        depth: usize,
    ) -> Result<Value, String> {
        if depth >= MAX_CALL_DEPTH {
            return Err(format!(
                "Stack overflow: maximum recursion depth ({MAX_CALL_DEPTH}) exceeded"
            ));
        }

        let local_count = chunk.functions[fn_idx].local_count;
        let mut locals = vec![Value::None; local_count];
        for (i, arg) in args.into_iter().enumerate() {
            if i < locals.len() {
                locals[i] = arg;
            }
        }
        self.frames.push(locals);
        let frame_idx = self.frames.len() - 1;
        let res = self.run_frame(chunk, fn_idx, frame_idx, depth);
        self.frames.pop();
        res
    }

    fn run_frame(
        &mut self,
        chunk: &Chunk,
        fn_idx: usize,
        frame_idx: usize,
        depth: usize,
    ) -> Result<Value, String> {
        let func = &chunk.functions[fn_idx];
        let mut ip = 0usize;
        while ip < func.code.len() {
            self.instructions_executed += 1;
            if self.instructions_executed > MAX_INSTRUCTIONS {
                return Err(format!(
                    "Execution limit exceeded ({MAX_INSTRUCTIONS} bytecode instructions)"
                ));
            }
            let op = &func.code[ip];
            ip += 1;

            match op {
                Op::PushConst(idx) => {
                    let val = match &chunk.constants[*idx] {
                        ConstVal::Int(i) => Value::Int(*i),
                        ConstVal::Float(f) => Value::Float(*f),
                        ConstVal::Bool(b) => Value::Bool(*b),
                        ConstVal::Char(c) => Value::Char(*c),
                        ConstVal::Str(s) => self.heap.alloc_string(s.clone()),
                        ConstVal::None => Value::None,
                    };
                    self.stack.push(val);
                    self.maybe_collect_gc();
                }
                Op::LoadLocal(slot) => {
                    let val = self.frames[frame_idx][*slot].clone();
                    self.stack.push(val);
                }
                Op::StoreLocal(slot) => {
                    let v = self.stack.pop().ok_or("Stack underflow")?;
                    self.frames[frame_idx][*slot] = v;
                }
                Op::LoadGlobal(name) => {
                    let v = self
                        .globals
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("Undefined global '{name}'"))?;
                    self.stack.push(v);
                }
                Op::StoreGlobal(name) => {
                    let v = self.stack.pop().ok_or("Stack underflow")?;
                    self.globals.insert(name.clone(), v);
                }
                Op::Add => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    let res = eval_add(&mut self.heap, &l, &r)?;
                    self.stack.push(res);
                    self.maybe_collect_gc();
                }
                Op::Sub => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    let res = eval_sub(&mut self.heap, &l, &r)?;
                    self.stack.push(res);
                }
                Op::Mul => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    let res = eval_mul(&mut self.heap, &l, &r)?;
                    self.stack.push(res);
                }
                Op::Div => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    let res = eval_div(&mut self.heap, &l, &r)?;
                    self.stack.push(res);
                }
                Op::Mod => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    let res = eval_mod(&l, &r)?;
                    self.stack.push(res);
                }
                Op::MatMul => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    let res = eval_matmul(&mut self.heap, &l, &r)?;
                    self.stack.push(res);
                    self.maybe_collect_gc();
                }
                Op::Neg => {
                    let v = self.stack.pop().ok_or("Stack underflow")?;
                    let res = eval_neg(&mut self.heap, &v)?;
                    self.stack.push(res);
                }
                Op::Not => {
                    let v = self.stack.pop().ok_or("Stack underflow")?;
                    self.stack.push(Value::Bool(!self.heap.is_truthy(&v)));
                }
                Op::CmpEq | Op::CmpNeq => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    let eq = self.heap.values_equal(&l, &r);
                    self.stack.push(Value::Bool(if matches!(op, Op::CmpEq) {
                        eq
                    } else {
                        !eq
                    }));
                }
                Op::CmpLt | Op::CmpLte | Op::CmpGt | Op::CmpGte => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    let ord = eval_ordering(&self.heap, &l, &r)?;
                    let res = match op {
                        Op::CmpLt => ord == Ordering::Less,
                        Op::CmpLte => matches!(ord, Ordering::Less | Ordering::Equal),
                        Op::CmpGt => ord == Ordering::Greater,
                        Op::CmpGte => matches!(ord, Ordering::Greater | Ordering::Equal),
                        _ => unreachable!(),
                    };
                    self.stack.push(Value::Bool(res));
                }
                Op::AllocArray(count) => {
                    let mut els = vec![Value::None; *count];
                    for i in (0..*count).rev() {
                        els[i] = self.stack.pop().ok_or("Stack underflow")?;
                    }
                    let arr = self.heap.alloc_array(els);
                    self.stack.push(arr);
                    self.maybe_collect_gc();
                }
                Op::Jump(target) => {
                    ip = *target;
                }
                Op::JumpIfFalse(target) => {
                    let cond = self.stack.pop().ok_or("Stack underflow")?;
                    if !self.heap.is_truthy(&cond) {
                        ip = *target;
                    }
                }
                Op::Call { name, argc } => {
                    let mut call_args = vec![Value::None; *argc];
                    for i in (0..*argc).rev() {
                        call_args[i] = self.stack.pop().ok_or("Stack underflow")?;
                    }
                    if name == "print" {
                        let line: Vec<String> = call_args
                            .iter()
                            .map(|a| self.heap.format_value(a))
                            .collect();
                        self.stdout.push(line.join(" "));
                        self.stack.push(Value::None);
                    } else if let Some(idx) = chunk.functions.iter().position(|f| f.name == *name) {
                        let ret = self.call_fn(chunk, idx, call_args, depth + 1)?;
                        self.stack.push(ret);
                    } else if let Some(std_val) =
                        dispatch_stdlib_builtin(&mut self.heap, name, &call_args)?
                    {
                        self.stack.push(std_val);
                    } else if let Some(ai_val) =
                        dispatch_ai_builtin(&mut self.heap, name, &call_args)?
                    {
                        self.stack.push(ai_val);
                    } else {
                        return Err(format!("Unknown function '{name}'"));
                    }
                    self.maybe_collect_gc();
                }
                Op::Return => {
                    return Ok(self.stack.pop().unwrap_or(Value::None));
                }
                Op::Pop => {
                    self.stack.pop().ok_or("Stack underflow")?;
                }
            }
        }
        Ok(Value::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nova_bytecode::compile;
    use nova_interpreter::Interpreter;
    use nova_ir::{fold_constants, lower_program};
    use nova_lexer::tokenize;
    use nova_parser::Parser;

    fn assert_equiv(src: &str, expected: &[&str]) {
        let toks = tokenize(src).unwrap();
        let prog = Parser::new(&toks).parse_program().unwrap();

        let mut interp = Interpreter::new();
        interp.run(&prog).unwrap();

        let mut ir = lower_program(&prog);
        fold_constants(&mut ir);
        let chunk = compile(&ir);
        let mut vm = Vm::new();
        vm.execute(&chunk).unwrap();

        assert_eq!(vm.stdout, interp.stdout, "VM and Interpreter diverged");
        let exp: Vec<String> = expected.iter().map(|s| (*s).to_string()).collect();
        assert_eq!(vm.stdout, exp);
    }

    #[test]
    fn executes_hello_nova_equivalently() {
        let src = "fn main() {\n    let name = \"Nova\"\n    print(\"Hello, \" + name)\n}";
        assert_equiv(src, &["Hello, Nova"]);
    }

    #[test]
    fn vm_and_interpreter_execution_equivalence() {
        let src = r#"
const OFFSET = 3

fn fib(n: Int) -> Int {
    if n <= 1 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}

fn main() {
    let x = 10
    if true {
        let x = 42
        print("inner=" + str(x))
    } else if false {
        print("unreachable")
    }
    print("outer=" + str(x))
    mut sum = 0
    mut i = 0
    while i < 10 {
        i += 1
        if i == 3 {
            continue
        }
        if i > 5 {
            break
        }
        sum += i
    }
    print("sum=" + str(sum))
    print("fib=" + str(fib(7) + OFFSET))
    print("neg=" + str(-OFFSET))
    print("eq=" + str(("No" + "va") == "Nova"))
}
"#;
        assert_equiv(
            src,
            &[
                "inner=42",
                "outer=10",
                "sum=12",
                "fib=16",
                "neg=-3",
                "eq=true",
            ],
        );
    }

    #[test]
    fn vm_and_interpreter_tensor_matmul_and_gc_stress() {
        let src = r#"
fn main() {
    let a = tensor([[1.0, 2.0], [3.0, 4.0]])
    let b = tensor([[2.0, 0.0], [1.0, 2.0]])
    print(a @ b)
    mut i = 0
    mut last = ""
    while i < 350 {
        last = "v" + str(i)
        i += 1
    }
    print(last)
}
"#;
        assert_equiv(
            src,
            &["Tensor([[4, 4], [10, 8]], shape=[2, 2])", "v349"],
        );
    }
}
