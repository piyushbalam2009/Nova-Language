//! Nova Intermediate Representation (IR) with lexical scoping, control-flow lowering,
//! short-circuit logical evaluation, and constant folding.

use nova_parser::{BinOp, Expr, Program, Stmt};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum ConstVal {
    Int(i64),
    Float(f64),
    Bool(bool),
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
    Add { dst: String, left: String, right: String },
    Sub { dst: String, left: String, right: String },
    Mul { dst: String, left: String, right: String },
    Div { dst: String, left: String, right: String },
    Mod { dst: String, left: String, right: String },
    Not { dst: String, src: String },
    Compare { dst: String, cmp: CmpOp, left: String, right: String },
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
    pub functions: Vec<IrFunction>,
}

pub fn lower_program(program: &Program) -> IrModule {
    let mut functions = Vec::new();
    for item in &program.items {
        if let Stmt::Fn {
            name, params, body, ..
        } = item
        {
            let param_names: Vec<String> = params.iter().map(|(n, _)| n.clone()).collect();
            let mut gen = FnBuilder::new(&param_names);
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
    IrModule { functions }
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
                    Inst::Add { dst, left, right } => {
                        if let (Some(l), Some(r)) = (consts.get(&left), consts.get(&right)) {
                            let res = match (l, r) {
                                (ConstVal::Int(a), ConstVal::Int(b)) => Some(ConstVal::Int(a + b)),
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
                        if let (Some(ConstVal::Int(a)), Some(ConstVal::Int(b))) =
                            (consts.get(&left), consts.get(&right))
                        {
                            let val = ConstVal::Int(a - b);
                            *inst = Inst::Const {
                                dst: dst.clone(),
                                val: val.clone(),
                            };
                            consts.insert(dst, val);
                            folded += 1;
                        }
                    }
                    Inst::Mul { dst, left, right } => {
                        if let (Some(ConstVal::Int(a)), Some(ConstVal::Int(b))) =
                            (consts.get(&left), consts.get(&right))
                        {
                            let val = ConstVal::Int(a * b);
                            *inst = Inst::Const {
                                dst: dst.clone(),
                                val: val.clone(),
                            };
                            consts.insert(dst, val);
                            folded += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    folded
}

struct FnBuilder {
    temp: usize,
    block_id: usize,
    slot_uid: usize,
    blocks: Vec<BasicBlock>,
    current: usize,
    scopes: Vec<HashMap<String, String>>,
}

impl FnBuilder {
    fn new(params: &[String]) -> Self {
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
        match stmt {
            Stmt::VarDecl { name, init, .. } => {
                let src = self.lower_expr(init);
                let slot = self.declare_local(name);
                self.emit(Inst::Store { slot, src });
            }
            Stmt::Assign { name, value, .. } => {
                let src = self.lower_expr(value);
                let slot = self.lookup_local(name).unwrap_or_else(|| name.clone());
                self.emit(Inst::Store { slot, src });
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
                    target: end_label,
                });
                self.emit(Inst::Jump { target: body_label });

                self.current = body_idx;
                self.push_scope();
                for s in body {
                    self.lower_stmt(s);
                }
                self.pop_scope();
                if !self.is_current_terminated() {
                    self.emit(Inst::Jump { target: cond_label });
                }

                self.current = end_idx;
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
                let slot = self.lookup_local(name).unwrap_or_else(|| name.clone());
                self.emit(Inst::Load {
                    dst: dst.clone(),
                    slot,
                });
                dst
            }
            Expr::Binary {
                op, left, right, ..
            } => {
                if matches!(op, BinOp::And | BinOp::Or) {
                    self.slot_uid += 1;
                    let res_slot = format!("__sc_{}", self.slot_uid);
                    let l = self.lower_expr(left);
                    self.emit(Inst::Store {
                        slot: res_slot.clone(),
                        src: l.clone(),
                    });
                    let (rhs_idx, rhs_label) = self.new_block("sc_rhs_");
                    let (end_idx, end_label) = self.new_block("sc_end_");

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
                    self.emit(Inst::Store {
                        slot: res_slot.clone(),
                        src: r,
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
                    BinOp::Eq => Inst::Compare { dst: dst.clone(), cmp: CmpOp::Eq, left: l, right: r },
                    BinOp::NotEq => Inst::Compare { dst: dst.clone(), cmp: CmpOp::NotEq, left: l, right: r },
                    BinOp::Lt => Inst::Compare { dst: dst.clone(), cmp: CmpOp::Lt, left: l, right: r },
                    BinOp::LtEq => Inst::Compare { dst: dst.clone(), cmp: CmpOp::LtEq, left: l, right: r },
                    BinOp::Gt => Inst::Compare { dst: dst.clone(), cmp: CmpOp::Gt, left: l, right: r },
                    BinOp::GtEq => Inst::Compare { dst: dst.clone(), cmp: CmpOp::GtEq, left: l, right: r },
                    _ => Inst::Div { dst: dst.clone(), left: l, right: r },
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
            Expr::Array(_, _) => {
                let dst = self.next_tmp();
                self.emit(Inst::Const {
                    dst: dst.clone(),
                    val: ConstVal::None,
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
    fn folds_constant_arithmetic() {
        let src = "fn main() {\n    let x = 2 + 3 * 4\n}";
        let toks = tokenize(src).unwrap();
        let prog = Parser::new(&toks).parse_program().unwrap();
        let mut ir = lower_program(&prog);
        let folded = fold_constants(&mut ir);
        assert_eq!(folded, 2);
    }
}
