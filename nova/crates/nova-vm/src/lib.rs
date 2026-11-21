//! Nova Virtual Machine: verified stack-based bytecode interpreter with explicit IP,
//! call frames, jumps, recursion guards, and mark-and-sweep GC heap.

use nova_bytecode::{verify, Chunk, Op};
use nova_ir::ConstVal;
use nova_runtime::{dispatch_ai_builtin, Heap, Value};

const MAX_CALL_DEPTH: usize = 256;
const MAX_INSTRUCTIONS: usize = 250_000;

#[derive(Debug, Default)]
pub struct Vm {
    stack: Vec<Value>,
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
        self.stdout.clear();
        self.instructions_executed = 0;

        let main_idx = chunk
            .functions
            .iter()
            .position(|f| f.name == "main")
            .ok_or_else(|| "Entry function 'main' not found".to_string())?;
        self.call_fn(chunk, main_idx, Vec::new(), 0)?;
        self.heap.collect(&self.stack);
        Ok(())
    }

    fn as_bool(v: &Value) -> bool {
        match v {
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::None => false,
            _ => true,
        }
    }

    fn values_equal(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::None, Value::None) => true,
            (Value::Int(x), Value::Int(y)) => x == y,
            (Value::Float(x), Value::Float(y)) => x == y,
            (Value::Int(x), Value::Float(y)) => (*x as f64) == *y,
            (Value::Float(x), Value::Int(y)) => *x == (*y as f64),
            (Value::Bool(x), Value::Bool(y)) => x == y,
            (Value::HeapRef(_), Value::HeapRef(_)) => {
                self.heap.format_value(a) == self.heap.format_value(b)
            }
            _ => false,
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

        let func = &chunk.functions[fn_idx];
        let mut locals = vec![Value::None; func.local_count];
        for (i, arg) in args.into_iter().enumerate() {
            if i < locals.len() {
                locals[i] = arg;
            }
        }

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
                        ConstVal::Str(s) => self.heap.alloc_string(s.clone()),
                        ConstVal::None => Value::None,
                    };
                    self.stack.push(val);
                }
                Op::LoadLocal(slot) => self.stack.push(locals[*slot].clone()),
                Op::StoreLocal(slot) => {
                    let v = self.stack.pop().ok_or("Stack underflow")?;
                    locals[*slot] = v;
                }
                Op::Add => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    let res = match (&l, &r) {
                        (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
                        (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
                        (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 + b),
                        (Value::Float(a), Value::Int(b)) => Value::Float(a + *b as f64),
                        _ => {
                            let s = format!(
                                "{}{}",
                                self.heap.format_value(&l),
                                self.heap.format_value(&r)
                            );
                            self.heap.alloc_string(s)
                        }
                    };
                    self.stack.push(res);
                }
                Op::Sub => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    match (l, r) {
                        (Value::Int(a), Value::Int(b)) => self.stack.push(Value::Int(a - b)),
                        (Value::Float(a), Value::Float(b)) => self.stack.push(Value::Float(a - b)),
                        _ => return Err("Invalid operands for '-'".to_string()),
                    }
                }
                Op::Mul => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    match (l, r) {
                        (Value::Int(a), Value::Int(b)) => self.stack.push(Value::Int(a * b)),
                        (Value::Float(a), Value::Float(b)) => self.stack.push(Value::Float(a * b)),
                        _ => return Err("Invalid operands for '*'".to_string()),
                    }
                }
                Op::Div => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    match (l, r) {
                        (Value::Int(_), Value::Int(0)) => {
                            return Err("Division by zero".to_string())
                        }
                        (Value::Int(a), Value::Int(b)) => self.stack.push(Value::Int(a / b)),
                        (Value::Float(a), Value::Float(b)) if b != 0.0 => {
                            self.stack.push(Value::Float(a / b))
                        }
                        _ => return Err("Invalid division".to_string()),
                    }
                }
                Op::Mod => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    match (l, r) {
                        (Value::Int(_), Value::Int(0)) => {
                            return Err("Modulo by zero".to_string())
                        }
                        (Value::Int(a), Value::Int(b)) => self.stack.push(Value::Int(a % b)),
                        (Value::Float(a), Value::Float(b)) if b != 0.0 => {
                            self.stack.push(Value::Float(a % b))
                        }
                        _ => return Err("Invalid modulo".to_string()),
                    }
                }
                Op::Not => {
                    let v = self.stack.pop().ok_or("Stack underflow")?;
                    self.stack.push(Value::Bool(!Self::as_bool(&v)));
                }
                Op::CmpEq | Op::CmpNeq => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    let eq = self.values_equal(&l, &r);
                    self.stack.push(Value::Bool(if matches!(op, Op::CmpEq) {
                        eq
                    } else {
                        !eq
                    }));
                }
                Op::CmpLt | Op::CmpLte | Op::CmpGt | Op::CmpGte => {
                    let r = self.stack.pop().ok_or("Stack underflow")?;
                    let l = self.stack.pop().ok_or("Stack underflow")?;
                    let (lv, rv) = match (l, r) {
                        (Value::Int(a), Value::Int(b)) => (a as f64, b as f64),
                        (Value::Float(a), Value::Float(b)) => (a, b),
                        (Value::Int(a), Value::Float(b)) => (a as f64, b),
                        (Value::Float(a), Value::Int(b)) => (a, b as f64),
                        _ => return Err("Ordering comparison requires numeric operands".to_string()),
                    };
                    let res = match op {
                        Op::CmpLt => lv < rv,
                        Op::CmpLte => lv <= rv,
                        Op::CmpGt => lv > rv,
                        Op::CmpGte => lv >= rv,
                        _ => unreachable!(),
                    };
                    self.stack.push(Value::Bool(res));
                }
                Op::Jump(target) => {
                    ip = *target;
                }
                Op::JumpIfFalse(target) => {
                    let cond = self.stack.pop().ok_or("Stack underflow")?;
                    if !Self::as_bool(&cond) {
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
                    } else if name == "str" {
                        let s = call_args
                            .first()
                            .map(|v| self.heap.format_value(v))
                            .unwrap_or_default();
                        let val = self.heap.alloc_string(s);
                        self.stack.push(val);
                    } else if let Some(idx) = chunk.functions.iter().position(|f| f.name == *name) {
                        let ret = self.call_fn(chunk, idx, call_args, depth + 1)?;
                        self.stack.push(ret);
                    } else if let Some(ai_val) = dispatch_ai_builtin(&mut self.heap, name, &call_args)? {
                        self.stack.push(ai_val);
                    } else {
                        return Err(format!("Unknown function '{name}'"));
                    }
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

    #[test]
    fn executes_hello_nova() {
        let src = "fn main() {\n    let name = \"Nova\"\n    print(\"Hello, \" + name)\n}";
        let toks = tokenize(src).unwrap();
        let prog = Parser::new(&toks).parse_program().unwrap();
        let ir = lower_program(&prog);
        let chunk = compile(&ir);
        let mut vm = Vm::new();
        vm.execute(&chunk).unwrap();
        assert_eq!(vm.stdout, vec!["Hello, Nova".to_string()]);
    }

    #[test]
    fn vm_and_interpreter_execution_equivalence() {
        let src = r#"
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
    }
    print("outer=" + str(x))
    mut sum = 0
    mut i = 1
    while i <= 5 {
        sum = sum + i
        i = i + 1
    }
    print("sum=" + str(sum))
    print("fib=" + str(fib(7)))
}
"#;
        let toks = tokenize(src).unwrap();
        let prog = Parser::new(&toks).parse_program().unwrap();

        let mut interp = Interpreter::new();
        interp.run(&prog).unwrap();

        let mut ir = lower_program(&prog);
        fold_constants(&mut ir);
        let chunk = compile(&ir);
        let mut vm = Vm::new();
        vm.execute(&chunk).unwrap();

        assert_eq!(vm.stdout, interp.stdout);
        assert_eq!(
            vm.stdout,
            vec!["inner=42", "outer=10", "sum=15", "fib=13"]
        );
    }
}
