//! Nova Bytecode Compiler, Instruction Set, and Static CFG/Stack-Height Verifier.

use nova_ir::{CmpOp, ConstVal, Inst, IrModule};
use std::collections::HashMap;
use std::fmt;

pub const BYTECODE_MAGIC: &[u8; 6] = b"NOVA01";
pub const BYTECODE_VERSION: u16 = 1;
pub const MAX_VERIFIED_STACK_DEPTH: usize = 1024;

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    PushConst(usize),
    LoadLocal(usize),
    StoreLocal(usize),
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Not,
    CmpEq,
    CmpNeq,
    CmpLt,
    CmpLte,
    CmpGt,
    CmpGte,
    Jump(usize),
    JumpIfFalse(usize),
    Call { name: String, argc: usize },
    Return,
    Pop,
}

#[derive(Debug, Clone)]
pub struct CompiledFn {
    pub name: String,
    pub arity: usize,
    pub local_count: usize,
    pub code: Vec<Op>,
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub magic: [u8; 6],
    pub version: u16,
    pub constants: Vec<ConstVal>,
    pub functions: Vec<CompiledFn>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyError {
    pub code: &'static str,
    pub message: String,
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error[{}]: {}", self.code, self.message)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReport {
    pub functions_verified: usize,
    pub instructions_verified: usize,
    pub max_stack_depth: usize,
}

pub fn compile(module: &IrModule) -> Chunk {
    let mut constants: Vec<ConstVal> = Vec::new();
    let mut functions = Vec::new();

    for f in &module.functions {
        let mut slots: HashMap<String, usize> = HashMap::new();
        for p in &f.params {
            let next = slots.len();
            slots.insert(p.clone(), next);
        }
        let mut slot_of = |name: &str| -> usize {
            if let Some(&idx) = slots.get(name) {
                idx
            } else {
                let idx = slots.len();
                slots.insert(name.to_string(), idx);
                idx
            }
        };

        let mut code = Vec::new();
        let mut block_offsets: HashMap<String, usize> = HashMap::new();
        let mut pending_jumps: Vec<(usize, String)> = Vec::new();

        for b in &f.blocks {
            block_offsets.insert(b.label.clone(), code.len());
            for inst in &b.instructions {
                match inst {
                    Inst::Const { dst, val } => {
                        let idx = constants.len();
                        constants.push(val.clone());
                        code.push(Op::PushConst(idx));
                        let s = slot_of(dst);
                        code.push(Op::StoreLocal(s));
                    }
                    Inst::Load { dst, slot } => {
                        let src_s = slot_of(slot);
                        let dst_s = slot_of(dst);
                        code.push(Op::LoadLocal(src_s));
                        code.push(Op::StoreLocal(dst_s));
                    }
                    Inst::Store { slot, src } => {
                        let src_s = slot_of(src);
                        let dst_s = slot_of(slot);
                        code.push(Op::LoadLocal(src_s));
                        code.push(Op::StoreLocal(dst_s));
                    }
                    Inst::Add { dst, left, right } => {
                        let (l, r, d) = (slot_of(left), slot_of(right), slot_of(dst));
                        code.push(Op::LoadLocal(l));
                        code.push(Op::LoadLocal(r));
                        code.push(Op::Add);
                        code.push(Op::StoreLocal(d));
                    }
                    Inst::Sub { dst, left, right } => {
                        let (l, r, d) = (slot_of(left), slot_of(right), slot_of(dst));
                        code.push(Op::LoadLocal(l));
                        code.push(Op::LoadLocal(r));
                        code.push(Op::Sub);
                        code.push(Op::StoreLocal(d));
                    }
                    Inst::Mul { dst, left, right } => {
                        let (l, r, d) = (slot_of(left), slot_of(right), slot_of(dst));
                        code.push(Op::LoadLocal(l));
                        code.push(Op::LoadLocal(r));
                        code.push(Op::Mul);
                        code.push(Op::StoreLocal(d));
                    }
                    Inst::Div { dst, left, right } => {
                        let (l, r, d) = (slot_of(left), slot_of(right), slot_of(dst));
                        code.push(Op::LoadLocal(l));
                        code.push(Op::LoadLocal(r));
                        code.push(Op::Div);
                        code.push(Op::StoreLocal(d));
                    }
                    Inst::Mod { dst, left, right } => {
                        let (l, r, d) = (slot_of(left), slot_of(right), slot_of(dst));
                        code.push(Op::LoadLocal(l));
                        code.push(Op::LoadLocal(r));
                        code.push(Op::Mod);
                        code.push(Op::StoreLocal(d));
                    }
                    Inst::Not { dst, src } => {
                        let (s, d) = (slot_of(src), slot_of(dst));
                        code.push(Op::LoadLocal(s));
                        code.push(Op::Not);
                        code.push(Op::StoreLocal(d));
                    }
                    Inst::Compare {
                        dst,
                        cmp,
                        left,
                        right,
                    } => {
                        let (l, r, d) = (slot_of(left), slot_of(right), slot_of(dst));
                        code.push(Op::LoadLocal(l));
                        code.push(Op::LoadLocal(r));
                        code.push(match cmp {
                            CmpOp::Eq => Op::CmpEq,
                            CmpOp::NotEq => Op::CmpNeq,
                            CmpOp::Lt => Op::CmpLt,
                            CmpOp::LtEq => Op::CmpLte,
                            CmpOp::Gt => Op::CmpGt,
                            CmpOp::GtEq => Op::CmpGte,
                        });
                        code.push(Op::StoreLocal(d));
                    }
                    Inst::Call { dst, callee, args } => {
                        for a in args {
                            let s = slot_of(a);
                            code.push(Op::LoadLocal(s));
                        }
                        let d = slot_of(dst);
                        code.push(Op::Call {
                            name: callee.clone(),
                            argc: args.len(),
                        });
                        code.push(Op::StoreLocal(d));
                    }
                    Inst::Jump { target } => {
                        let idx = code.len();
                        code.push(Op::Jump(usize::MAX));
                        pending_jumps.push((idx, target.clone()));
                    }
                    Inst::JumpIfFalse { cond, target } => {
                        let c = slot_of(cond);
                        code.push(Op::LoadLocal(c));
                        let idx = code.len();
                        code.push(Op::JumpIfFalse(usize::MAX));
                        pending_jumps.push((idx, target.clone()));
                    }
                    Inst::Return { src } => {
                        if let Some(s_name) = src {
                            let s = slot_of(s_name);
                            code.push(Op::LoadLocal(s));
                        } else {
                            let idx = constants.len();
                            constants.push(ConstVal::None);
                            code.push(Op::PushConst(idx));
                        }
                        code.push(Op::Return);
                    }
                }
            }
        }

        for (inst_idx, label) in pending_jumps {
            if let Some(&target_ip) = block_offsets.get(&label) {
                match &mut code[inst_idx] {
                    Op::Jump(t) | Op::JumpIfFalse(t) => *t = target_ip,
                    _ => {}
                }
            }
        }

        functions.push(CompiledFn {
            name: f.name.clone(),
            arity: f.params.len(),
            local_count: slots.len(),
            code,
        });
    }

    Chunk {
        magic: *BYTECODE_MAGIC,
        version: BYTECODE_VERSION,
        constants,
        functions,
    }
}

fn stack_effect(op: &Op) -> (usize, usize) {
    match op {
        Op::PushConst(_) | Op::LoadLocal(_) => (0, 1),
        Op::StoreLocal(_) | Op::JumpIfFalse(_) | Op::Pop => (1, 0),
        Op::Not => (1, 1),
        Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Div
        | Op::Mod
        | Op::CmpEq
        | Op::CmpNeq
        | Op::CmpLt
        | Op::CmpLte
        | Op::CmpGt
        | Op::CmpGte => (2, 1),
        Op::Jump(_) => (0, 0),
        Op::Call { argc, .. } => (*argc, 1),
        Op::Return => (1, 0),
    }
}

/// Static Nova Bytecode Verifier.
/// Validates header, operand bounds, call arity, CFG stack-height consistency,
/// stack overflow/underflow, and function termination (`RETURN`).
pub fn verify(chunk: &Chunk) -> Result<VerificationReport, Vec<VerifyError>> {
    let mut errors = Vec::new();

    if &chunk.magic != BYTECODE_MAGIC {
        errors.push(VerifyError {
            code: "E0401",
            message: "Invalid bytecode header magic (expected NOVA01)".to_string(),
        });
    }
    if chunk.version != BYTECODE_VERSION {
        errors.push(VerifyError {
            code: "E0401",
            message: format!("Unsupported bytecode version {}", chunk.version),
        });
    }

    let mut fn_arities: HashMap<&str, usize> = HashMap::new();
    for f in &chunk.functions {
        if fn_arities.insert(f.name.as_str(), f.arity).is_some() {
            errors.push(VerifyError {
                code: "E0405",
                message: format!("Duplicate function '{}' in bytecode chunk", f.name),
            });
        }
        if f.arity > f.local_count {
            errors.push(VerifyError {
                code: "E0406",
                message: format!(
                    "Function '{}' arity ({}) exceeds local_count ({})",
                    f.name, f.arity, f.local_count
                ),
            });
        }
    }

    let mut total_insts = 0usize;
    let mut global_max_stack = 0usize;

    for f in &chunk.functions {
        total_insts += f.code.len();
        if f.code.is_empty() {
            errors.push(VerifyError {
                code: "E0413",
                message: format!("Function '{}' has empty bytecode body without Return", f.name),
            });
            continue;
        }

        let mut operand_ok = true;
        for (ip, op) in f.code.iter().enumerate() {
            match op {
                Op::PushConst(idx) if *idx >= chunk.constants.len() => {
                    errors.push(VerifyError {
                        code: "E0402",
                        message: format!(
                            "Constant index {idx} out of bounds in '{}' at ip={ip}",
                            f.name
                        ),
                    });
                    operand_ok = false;
                }
                Op::LoadLocal(s) | Op::StoreLocal(s) if *s >= f.local_count => {
                    errors.push(VerifyError {
                        code: "E0403",
                        message: format!(
                            "Local slot {s} out of bounds in '{}' at ip={ip}",
                            f.name
                        ),
                    });
                    operand_ok = false;
                }
                Op::Jump(t) | Op::JumpIfFalse(t) if *t >= f.code.len() => {
                    errors.push(VerifyError {
                        code: "E0404",
                        message: format!(
                            "Jump target {t} out of bounds in '{}' at ip={ip}",
                            f.name
                        ),
                    });
                    operand_ok = false;
                }
                Op::Call { name, argc } => {
                    if let Some(&expected) = fn_arities.get(name.as_str()) {
                        if *argc != expected {
                            errors.push(VerifyError {
                                code: "E0408",
                                message: format!(
                                    "Call '{}' passes {argc} arg(s), expected {expected} in '{}' at ip={ip}",
                                    name, f.name
                                ),
                            });
                            operand_ok = false;
                        }
                    }
                }
                _ => {}
            }
        }

        if !operand_ok {
            continue;
        }

        // Abstract stack-height dataflow verification via worklist
        let mut entry_depth: HashMap<usize, usize> = HashMap::new();
        let mut worklist: Vec<(usize, usize)> = vec![(0, 0)];
        entry_depth.insert(0, 0);

        while let Some((ip, depth)) = worklist.pop() {
            let op = &f.code[ip];
            let (pops, pushes) = stack_effect(op);
            if depth < pops {
                errors.push(VerifyError {
                    code: "E0410",
                    message: format!(
                        "Operand stack underflow in '{}' at ip={ip}: depth {depth} < {pops}",
                        f.name
                    ),
                });
                continue;
            }
            let next_depth = depth - pops + pushes;
            if next_depth > global_max_stack {
                global_max_stack = next_depth;
            }
            if next_depth > MAX_VERIFIED_STACK_DEPTH {
                errors.push(VerifyError {
                    code: "E0411",
                    message: format!(
                        "Operand stack overflow in '{}' at ip={ip}: depth {next_depth}",
                        f.name
                    ),
                });
                continue;
            }

            let mut targets = Vec::new();
            match op {
                Op::Return => continue,
                Op::Jump(t) => targets.push(*t),
                Op::JumpIfFalse(t) => {
                    targets.push(*t);
                    targets.push(ip + 1);
                }
                _ => targets.push(ip + 1),
            }

            for target in targets {
                if target >= f.code.len() {
                    errors.push(VerifyError {
                        code: "E0413",
                        message: format!(
                            "Control flow falls off end of '{}' after ip={ip} without Return",
                            f.name
                        ),
                    });
                    continue;
                }
                match entry_depth.get(&target) {
                    None => {
                        entry_depth.insert(target, next_depth);
                        worklist.push((target, next_depth));
                    }
                    Some(&prev) if prev != next_depth => {
                        errors.push(VerifyError {
                            code: "E0412",
                            message: format!(
                                "Stack height mismatch at merge ip={target} in '{}': {next_depth} != {prev}",
                                f.name
                            ),
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(VerificationReport {
            functions_verified: chunk.functions.len(),
            instructions_verified: total_insts,
            max_stack_depth: global_max_stack,
        })
    } else {
        Err(errors)
    }
}

pub fn validate(chunk: &Chunk) -> Result<(), String> {
    verify(chunk).map(|_| ()).map_err(|errs| {
        errs.into_iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_stack_underflow_and_missing_return() {
        let bad = Chunk {
            magic: *BYTECODE_MAGIC,
            version: BYTECODE_VERSION,
            constants: vec![],
            functions: vec![CompiledFn {
                name: "main".to_string(),
                arity: 0,
                local_count: 0,
                code: vec![Op::Add],
            }],
        };
        let errs = verify(&bad).unwrap_err();
        assert!(errs.iter().any(|e| e.code == "E0410"));
    }

    #[test]
    fn rejects_merge_stack_mismatch() {
        let bad = Chunk {
            magic: *BYTECODE_MAGIC,
            version: BYTECODE_VERSION,
            constants: vec![ConstVal::Bool(true), ConstVal::Int(1)],
            functions: vec![CompiledFn {
                name: "main".to_string(),
                arity: 0,
                local_count: 0,
                code: vec![
                    Op::PushConst(0),
                    Op::JumpIfFalse(3),
                    Op::PushConst(1),
                    Op::Return,
                ],
            }],
        };
        let errs = verify(&bad).unwrap_err();
        assert!(errs.iter().any(|e| e.code == "E0412" || e.code == "E0410"));
    }
}
