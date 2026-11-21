//! Nova Intermediate Representation (IR) with lexical scoping, control-flow lowering,
//! short-circuit logical evaluation, globals initialization, and overflow-safe constant folding.

use nova_parser::{BinOp, Expr, Program, Stmt, UnaryOp};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq)]
pub enum ConstVal {
    Int(i64),
    Float(f64),
    Bool(bool),
    Char(char),
    Str(String),
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Inst {
    Const { dst: String, val: ConstVal },
    Load { dst: String, slot: String },
    Store { slot: String, src: String },
    LoadGlobal { dst: String, name: String },
    StoreGlobal { name: String, src: String },
    Add { dst: String, left: String, right: String },
    Sub { dst: String, left: String, right: String },
    Mul { dst: String, left: String, right: String },
    Div { dst: String, left: String, right: String },
    Mod { dst: String, left: String, right: String },
    MatMul { dst: String, left: String, right: String },
    Neg { dst: String, src: String },
    Not { dst: String, src: String },
    Compare { dst: String, cmp: CmpOp, left: String, right: String },
    AllocArray { dst: String, elements: Vec<String> },
    Call { dst: String, callee: String, args: Vec<String> },
    Jump { target: String },
    JumpIfFalse { cond: String, target: String },
    Return { src: Option<String> },
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub label: String,
    pub instructions: Vec<Inst>,
}

#[derive(Debug, Clone)]
pub struct IrFunction {
    pub name: String,
    pub params: Vec<String>,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Debug, Clone)]
pub struct IrModule {
    pub globals: Vec<String>,
    pub functions: Vec<IrFunction>,
}

pub fn lower_program(program: &Program) -> IrModule {
    let mut global_names: Vec<String> = Vec::new();
    let mut global_set: HashSet<String> = HashSet::new();
    let mut top_level_stmts: Vec<&Stmt> = Vec::new();

    for item in &program.items {
        match item {
            Stmt::Fn { .. } | Stmt::Struct { .. } | Stmt::Use { .. } => {}
            Stmt::VarDecl { name, .. } => {
                if global_set.insert(name.clone()) {
                    global_names.push(name.clone());
                }
                top_level_stmts.push(item);
            }
            other => {
                top_level_stmts.push(other);
            }
        }
    }

    let mut functions = Vec::new();

    if !top_level_stmts.is_empty() {
        let mut init_gen = FnBuilder::new(&[], &global_set, true);
        for s in top_level_stmts {
            init_gen.lower_stmt(s);
        }
        if !init_gen.is_current_terminated() {
            init_gen.emit(Inst::Return { src: None });
        }
        functions.push(IrFunction {
            name: "__init__".to_string(),
            params: Vec::new(),
            blocks: init_gen.finish(),
        });
    }

    for item in &program.items {
        if let Stmt::Fn {
            name, params, body, ..
        } = item
        {
            let param_names: Vec<String> = params.iter().map(|(n, _)| n.clone()).collect();
            let mut gen = FnBuilder::new(&param_names, &global_set, false);
            for s in body {
                gen.lower_stmt(s);
            }
            if !gen.is_current_terminated() {
                gen.emit(Inst::Return { src: None });
            }
            functions.push(IrFunction {
                name: name.clone(),
                params: param_names,
                blocks: gen.finish(),
            });
        }
    }

    IrModule {
        globals: global_names,
        functions,
    }
}

pub fn fold_constants(module: &mut IrModule) -> usize {
    let mut folded = 0usize;
    for func in &mut module.functions {
        for block in &mut func.blocks {
            let mut consts: HashMap<String, ConstVal> = HashMap::new();
            for inst in &mut block.instructions {
                match inst.clone() {
                    Inst::Const { dst, val } => {
                        consts.insert(dst, val);
                    }
                    Inst::Neg { dst, src } => {
                        if let Some(v) = consts.get(&src) {
                            let res = match v {
                                ConstVal::Int(a) => a.checked_neg().map(ConstVal::Int),
                                ConstVal::Float(a) => Some(ConstVal::Float(-a)),
                                _ => None,
                            };
                            if let Some(val) = res {
                                *inst = Inst::Const {
                                    dst: dst.clone(),
                                    val: val.clone(),
                                };
                                consts.insert(dst, val);
                                folded += 1;
                            }
                        }
                    }
                    Inst::Not { dst, src } => {
                        if let Some(ConstVal::Bool(b)) = consts.get(&src) {
                            let val = ConstVal::Bool(!*b);
                            *inst = Inst::Const {
                                dst: dst.clone(),
                                val: val.clone(),
                            };
                            consts.insert(dst, val);
                            folded += 1;
                        }
                    }
                    Inst::Add { dst, left, right } => {
                        if let (Some(l), Some(r)) = (consts.get(&left), consts.get(&right)) {
                            let res = match (l, r) {
                                (ConstVal::Int(a), ConstVal::Int(b)) => {
                                    a.checked_add(*b).map(ConstVal::Int)
                                }
                                (ConstVal::Float(a), ConstVal::Float(b)) => {
                                    Some(ConstVal::Float(a + b))
                                }
                                (ConstVal::Str(a), ConstVal::Str(b)) => {
                                    Some(ConstVal::Str(format!("{a}{b}")))
                                }
                                _ => None,
                            };
                            if let Some(val) = res {
                                *inst = Inst::Const {
                                    dst: dst.clone(),
                                    val: val.clone(),
                                };
                                consts.insert(dst, val);
                                folded += 1;
                            }
                        }
                    }
                    Inst::Sub { dst, left, right } => {
                        if let (Some(l), Some(r)) = (consts.get(&left), consts.get(&right)) {
                            let res = match (l, r) {
                                (ConstVal::Int(a), ConstVal::Int(b)) => {
                                    a.checked_sub(*b).map(ConstVal::Int)
                                }
                                (ConstVal::Float(a), ConstVal::Float(b)) => {
                                    Some(ConstVal::Float(a - b))
                                }
                                _ => None,
                            };
                            if let Some(val) = res {
                                *inst = Inst::Const {
                                    dst: dst.clone(),
                                    val: val.clone(),
                                };
                                consts.insert(dst, val);
                                folded += 1;
                            }
                        }
                    }
                    Inst::Mul { dst, left, right } => {
                        if let (Some(l), Some(r)) = (consts.get(&left), consts.get(&right)) {
                            let res = match (l, r) {
                                (ConstVal::Int(a), ConstVal::Int(b)) => {
                                    a.checked_mul(*b).map(ConstVal::Int)
                                }
                                (ConstVal::Float(a), ConstVal::Float(b)) => {
                                    Some(ConstVal::Float(a * b))
                                }
                                _ => None,
                            };
                            if let Some(val) = res {
                                *inst = Inst::Const {
                                    dst: dst.clone(),
                                    val: val.clone(),
                                };
                                consts.insert(dst, val);
                                folded += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    folded
}

struct FnBuilder<'a> {
    temp: usize,
    block_id: usize,
    slot_uid: usize,
    blocks: Vec<BasicBlock>,
    current: usize,
    scopes: Vec<HashMap<String, String>>,
    globals: &'a HashSet<String>,
    is_init: bool,
    loop_stack: Vec<(String, String)>,
}

impl<'a> FnBuilder<'a> {
    fn new(params: &[String], globals: &'a HashSet<String>, is_init: bool) -> Self {
        let mut root_scope = HashMap::new();
        for p in params {
            root_scope.insert(p.clone(), p.clone());
        }
        Self {
            temp: 0,
            block_id: 1,
            slot_uid: 0,
            blocks: vec![BasicBlock {
                label: "entry_0".to_string(),
                instructions: Vec::new(),
            }],
            current: 0,
            scopes: vec![root_scope],
            globals,
            is_init,
            loop_stack: Vec::new(),
        }
    }

    fn finish(self) -> Vec<BasicBlock> {
        self.blocks
    }

    fn next_tmp(&mut self) -> String {
        let name = format!("%t{}", self.temp);
        self.temp += 1;
        name
    }

    fn new_block(&mut self, prefix: &str) -> (usize, String) {
        let label = format!("{prefix}{}", self.block_id);
        self.block_id += 1;
        let idx = self.blocks.len();
        self.blocks.push(BasicBlock {
            label: label.clone(),
            instructions: Vec::new(),
        });
        (idx, label)
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn lookup_local(&self, name: &str) -> Option<String> {
        for scope in self.scopes.iter().rev() {
            if let Some(slot) = scope.get(name) {
                return Some(slot.clone());
            }
        }
        None
    }

    fn declare_local(&mut self, name: &str) -> String {
        let slot = if self.lookup_local(name).is_some() {
            self.slot_uid += 1;
            format!("{name}_{}", self.slot_uid)
        } else {
            name.to_string()
        };
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), slot.clone());
        }
        slot
    }

    fn emit(&mut self, inst: Inst) {
        self.blocks[self.current].instructions.push(inst);
    }

    fn is_current_terminated(&self) -> bool {
        matches!(
            self.blocks[self.current].instructions.last(),
            Some(Inst::Return { .. } | Inst::Jump { .. })
        )
    }

    fn lower_stmt(&mut self, stmt: &Stmt) {
        if self.is_current_terminated() {
            return;
        }
        match stmt {
            Stmt::VarDecl { name, init, .. } => {
                let src = self.lower_expr(init);
                if self.is_init && self.scopes.len() == 1 {
                    self.emit(Inst::StoreGlobal {
                        name: name.clone(),
                        src,
                    });
                } else {
                    let slot = self.declare_local(name);
                    self.emit(Inst::Store { slot, src });
                }
            }
            Stmt::Assign { name, value, .. } => {
                let src = self.lower_expr(value);
                if let Some(slot) = self.lookup_local(name) {
                    self.emit(Inst::Store { slot, src });
                } else if self.globals.contains(name) {
                    self.emit(Inst::StoreGlobal {
                        name: name.clone(),
                        src,
                    });
                } else {
                    self.emit(Inst::Store {
                        slot: name.clone(),
                        src,
                    });
                }
            }
            Stmt::If {
                cond,
                then_body,
                else_body,
                ..
            } => {
                let cond_tmp = self.lower_expr(cond);
                let (then_idx, then_label) = self.new_block("then_");
                let else_target = else_body.as_ref().map(|_| self.new_block("else_"));
                let (merge_idx, merge_label) = self.new_block("merge_");

                let false_label = else_target
                    .as_ref()
                    .map(|(_, l)| l.clone())
                    .unwrap_or_else(|| merge_label.clone());

                self.emit(Inst::JumpIfFalse {
                    cond: cond_tmp,
                    target: false_label,
                });
                self.emit(Inst::Jump { target: then_label });

                self.current = then_idx;
                self.push_scope();
                for s in then_body {
                    self.lower_stmt(s);
                }
                self.pop_scope();
                if !self.is_current_terminated() {
                    self.emit(Inst::Jump {
                        target: merge_label.clone(),
                    });
                }

                if let (Some((else_idx, _)), Some(else_stmts)) = (else_target, else_body) {
                    self.current = else_idx;
                    self.push_scope();
                    for s in else_stmts {
                        self.lower_stmt(s);
                    }
                    self.pop_scope();
                    if !self.is_current_terminated() {
                        self.emit(Inst::Jump {
                            target: merge_label,
                        });
                    }
                }

                self.current = merge_idx;
            }
            Stmt::While { cond, body, .. } => {
                let (cond_idx, cond_label) = self.new_block("while_cond_");
                let (body_idx, body_label) = self.new_block("while_body_");
                let (end_idx, end_label) = self.new_block("while_end_");

                self.emit(Inst::Jump {
                    target: cond_label.clone(),
                });

                self.current = cond_idx;
                let cond_tmp = self.lower_expr(cond);
                self.emit(Inst::JumpIfFalse {
                    cond: cond_tmp,
                    target: end_label.clone(),
                });
                self.emit(Inst::Jump { target: body_label });

                self.current = body_idx;
                self.push_scope();
                self.loop_stack.push((cond_label.clone(), end_label));
                for s in body {
                    self.lower_stmt(s);
                }
                self.loop_stack.pop();
                self.pop_scope();
                if !self.is_current_terminated() {
                    self.emit(Inst::Jump { target: cond_label });
                }

                self.current = end_idx;
            }
            Stmt::Break(_) => {
                if let Some((_, break_label)) = self.loop_stack.last().cloned() {
                    self.emit(Inst::Jump {
                        target: break_label,
                    });
                }
            }
            Stmt::Continue(_) => {
                if let Some((continue_label, _)) = self.loop_stack.last().cloned() {
                    self.emit(Inst::Jump {
                        target: continue_label,
                    });
                }
            }
            Stmt::Return { value, .. } => {
                let src = value.as_ref().map(|e| self.lower_expr(e));
                self.emit(Inst::Return { src });
            }
            Stmt::Expr(e) => {
                self.lower_expr(e);
            }
            _ => {}
        }
    }

    fn lower_expr(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Int(v, _) => {
                let dst = self.next_tmp();
                self.emit(Inst::Const {
                    dst: dst.clone(),
                    val: ConstVal::Int(*v),
                });
                dst
            }
            Expr::Float(v, _) => {
                let dst = self.next_tmp();
                self.emit(Inst::Const {
                    dst: dst.clone(),
                    val: ConstVal::Float(*v),
                });
                dst
            }
            Expr::Str(s, _) => {
                let dst = self.next_tmp();
                self.emit(Inst::Const {
                    dst: dst.clone(),
                    val: ConstVal::Str(s.clone()),
                });
                dst
            }
            Expr::Char(c, _) => {
                let dst = self.next_tmp();
                self.emit(Inst::Const {
                    dst: dst.clone(),
                    val: ConstVal::Char(*c),
                });
                dst
            }
            Expr::Bool(b, _) => {
                let dst = self.next_tmp();
                self.emit(Inst::Const {
                    dst: dst.clone(),
                    val: ConstVal::Bool(*b),
                });
                dst
            }
            Expr::None(_) => {
                let dst = self.next_tmp();
                self.emit(Inst::Const {
                    dst: dst.clone(),
                    val: ConstVal::None,
                });
                dst
            }
            Expr::Ident(name, _) => {
                let dst = self.next_tmp();
                if let Some(slot) = self.lookup_local(name) {
                    self.emit(Inst::Load {
                        dst: dst.clone(),
                        slot,
                    });
                } else if self.globals.contains(name) {
                    self.emit(Inst::LoadGlobal {
                        dst: dst.clone(),
                        name: name.clone(),
                    });
                } else {
                    self.emit(Inst::Load {
                        dst: dst.clone(),
                        slot: name.clone(),
                    });
                }
                dst
            }
            Expr::Unary { op, expr, .. } => {
                let src = self.lower_expr(expr);
                let dst = self.next_tmp();
                match op {
                    UnaryOp::Neg => self.emit(Inst::Neg {
                        dst: dst.clone(),
                        src,
                    }),
                    UnaryOp::Not => self.emit(Inst::Not {
                        dst: dst.clone(),
                        src,
                    }),
                }
                dst
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                if matches!(op, BinOp::And | BinOp::Or) {
                    self.slot_uid += 1;
                    let res_slot = format!("__sc_{}", self.slot_uid);
                    let l = self.lower_expr(left);
                    let (rhs_idx, rhs_label) = self.new_block("sc_rhs_");
                    let (end_idx, end_label) = self.new_block("sc_end_");

                    // Normalize short-circuit result to Bool (parity with AST interpreter)
                    let short_val = self.next_tmp();
                    self.emit(Inst::Const {
                        dst: short_val.clone(),
                        val: ConstVal::Bool(*op == BinOp::Or),
                    });
                    self.emit(Inst::Store {
                        slot: res_slot.clone(),
                        src: short_val,
                    });

                    if *op == BinOp::And {
                        self.emit(Inst::JumpIfFalse {
                            cond: l,
                            target: end_label.clone(),
                        });
                        self.emit(Inst::Jump { target: rhs_label });
                    } else {
                        let not_l = self.next_tmp();
                        self.emit(Inst::Not {
                            dst: not_l.clone(),
                            src: l,
                        });
                        self.emit(Inst::JumpIfFalse {
                            cond: not_l,
                            target: end_label.clone(),
                        });
                        self.emit(Inst::Jump { target: rhs_label });
                    }

                    self.current = rhs_idx;
                    let r = self.lower_expr(right);
                    let not_r = self.next_tmp();
                    let bool_r = self.next_tmp();
                    self.emit(Inst::Not {
                        dst: not_r.clone(),
                        src: r,
                    });
                    self.emit(Inst::Not {
                        dst: bool_r.clone(),
                        src: not_r,
                    });
                    self.emit(Inst::Store {
                        slot: res_slot.clone(),
                        src: bool_r,
                    });
                    self.emit(Inst::Jump { target: end_label });

                    self.current = end_idx;
                    let dst = self.next_tmp();
                    self.emit(Inst::Load {
                        dst: dst.clone(),
                        slot: res_slot,
                    });
                    return dst;
                }

                let l = self.lower_expr(left);
                let r = self.lower_expr(right);
                let dst = self.next_tmp();
                let inst = match op {
                    BinOp::Add => Inst::Add { dst: dst.clone(), left: l, right: r },
                    BinOp::Sub => Inst::Sub { dst: dst.clone(), left: l, right: r },
                    BinOp::Mul => Inst::Mul { dst: dst.clone(), left: l, right: r },
                    BinOp::Div => Inst::Div { dst: dst.clone(), left: l, right: r },
                    BinOp::Mod => Inst::Mod { dst: dst.clone(), left: l, right: r },
                    BinOp::MatMul => Inst::MatMul { dst: dst.clone(), left: l, right: r },
                    BinOp::Eq => Inst::Compare { dst: dst.clone(), cmp: CmpOp::Eq, left: l, right: r },
                    BinOp::NotEq => Inst::Compare { dst: dst.clone(), cmp: CmpOp::NotEq, left: l, right: r },
                    BinOp::Lt => Inst::Compare { dst: dst.clone(), cmp: CmpOp::Lt, left: l, right: r },
                    BinOp::LtEq => Inst::Compare { dst: dst.clone(), cmp: CmpOp::LtEq, left: l, right: r },
                    BinOp::Gt => Inst::Compare { dst: dst.clone(), cmp: CmpOp::Gt, left: l, right: r },
                    BinOp::GtEq => Inst::Compare { dst: dst.clone(), cmp: CmpOp::GtEq, left: l, right: r },
                    BinOp::And | BinOp::Or => unreachable!("handled above"),
                };
                self.emit(inst);
                dst
            }
            Expr::Call { callee, args, .. } => {
                let arg_tmps = args.iter().map(|a| self.lower_expr(a)).collect();
                let dst = self.next_tmp();
                self.emit(Inst::Call {
                    dst: dst.clone(),
                    callee: callee.clone(),
                    args: arg_tmps,
                });
                dst
            }
            Expr::Array(items, _) => {
                let elements = items.iter().map(|e| self.lower_expr(e)).collect();
                let dst = self.next_tmp();
                self.emit(Inst::AllocArray {
                    dst: dst.clone(),
                    elements,
                });
                dst
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nova_lexer::tokenize;
    use nova_parser::Parser;

    #[test]
    fn folds_constant_arithmetic_safely() {
        let src = "fn main() {\n    let x = 2 + 3 * 4\n    let y = -5\n}";
        let toks = tokenize(src).unwrap();
        let prog = Parser::new(&toks).parse_program().unwrap();
        let mut ir = lower_program(&prog);
        let folded = fold_constants(&mut ir);
        assert_eq!(folded, 3);
    }

    #[test]
    fn lowers_matmul_and_array_without_miscompilation() {
        let src = r#"
fn main() {
    let a = tensor([[1.0, 2.0]])
    let b = tensor([[3.0], [4.0]])
    let c = a @ b
}
"#;
        let toks = tokenize(src).unwrap();
        let prog = Parser::new(&toks).parse_program().unwrap();
        let ir = lower_program(&prog);
        let insts: Vec<&Inst> = ir.functions[0]
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .collect();
        assert!(insts.iter().any(|i| matches!(i, Inst::MatMul { .. })));
        assert!(insts.iter().any(|i| matches!(i, Inst::AllocArray { .. })));
        assert!(!insts.iter().any(|i| matches!(i, Inst::Div { .. })));
    }
}
