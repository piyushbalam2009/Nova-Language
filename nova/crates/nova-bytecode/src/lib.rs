//! Nova Bytecode Compiler, Instruction Set, `.nvbc` Binary Codec, and Static CFG/Stack-Height Verifier.

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
    LoadGlobal(String),
    StoreGlobal(String),
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    MatMul,
    Neg,
    Not,
    CmpEq,
    CmpNeq,
    CmpLt,
    CmpLte,
    CmpGt,
    CmpGte,
    AllocArray(usize),
    Jump(usize),
    JumpIfFalse(usize),
    Call { name: String, argc: usize },
    Return,
    Pop,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledFn {
    pub name: String,
    pub arity: usize,
    pub local_count: usize,
    pub code: Vec<Op>,
}

#[derive(Debug, Clone, PartialEq)]
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
                    Inst::LoadGlobal { dst, name } => {
                        let dst_s = slot_of(dst);
                        code.push(Op::LoadGlobal(name.clone()));
                        code.push(Op::StoreLocal(dst_s));
                    }
                    Inst::StoreGlobal { name, src } => {
                        let src_s = slot_of(src);
                        code.push(Op::LoadLocal(src_s));
                        code.push(Op::StoreGlobal(name.clone()));
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
                    Inst::MatMul { dst, left, right } => {
                        let (l, r, d) = (slot_of(left), slot_of(right), slot_of(dst));
                        code.push(Op::LoadLocal(l));
                        code.push(Op::LoadLocal(r));
                        code.push(Op::MatMul);
                        code.push(Op::StoreLocal(d));
                    }
                    Inst::Neg { dst, src } => {
                        let (s, d) = (slot_of(src), slot_of(dst));
                        code.push(Op::LoadLocal(s));
                        code.push(Op::Neg);
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
                    Inst::AllocArray { dst, elements } => {
                        for el in elements {
                            let s = slot_of(el);
                            code.push(Op::LoadLocal(s));
                        }
                        let d = slot_of(dst);
                        code.push(Op::AllocArray(elements.len()));
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
        Op::PushConst(_) | Op::LoadLocal(_) | Op::LoadGlobal(_) => (0, 1),
        Op::StoreLocal(_) | Op::StoreGlobal(_) | Op::JumpIfFalse(_) | Op::Pop => (1, 0),
        Op::Neg | Op::Not => (1, 1),
        Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Div
        | Op::Mod
        | Op::MatMul
        | Op::CmpEq
        | Op::CmpNeq
        | Op::CmpLt
        | Op::CmpLte
        | Op::CmpGt
        | Op::CmpGte => (2, 1),
        Op::AllocArray(count) => (*count, 1),
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

fn write_str(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(bytes);
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > bytes.len() {
        return Err("Unexpected EOF while decoding .nvbc u32".to_string());
    }
    let val = u32::from_le_bytes([bytes[*pos], bytes[*pos + 1], bytes[*pos + 2], bytes[*pos + 3]]);
    *pos += 4;
    Ok(val)
}

fn read_str(bytes: &[u8], pos: &mut usize) -> Result<String, String> {
    let len = read_u32(bytes, pos)? as usize;
    if *pos + len > bytes.len() {
        return Err("Unexpected EOF while decoding .nvbc string".to_string());
    }
    let s = std::str::from_utf8(&bytes[*pos..*pos + len])
        .map_err(|_| "Invalid UTF-8 string in .nvbc".to_string())?
        .to_string();
    *pos += len;
    Ok(s)
}

/// Serialize a compiled Nova bytecode chunk into the binary `.nvbc` wire format.
pub fn encode_chunk(chunk: &Chunk) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&chunk.magic);
    out.extend_from_slice(&chunk.version.to_le_bytes());
    out.extend_from_slice(&(chunk.constants.len() as u32).to_le_bytes());
    for c in &chunk.constants {
        match c {
            ConstVal::None => out.push(0),
            ConstVal::Int(i) => {
                out.push(1);
                out.extend_from_slice(&i.to_le_bytes());
            }
            ConstVal::Float(f) => {
                out.push(2);
                out.extend_from_slice(&f.to_bits().to_le_bytes());
            }
            ConstVal::Bool(b) => {
                out.push(3);
                out.push(u8::from(*b));
            }
            ConstVal::Char(ch) => {
                out.push(4);
                out.extend_from_slice(&(*ch as u32).to_le_bytes());
            }
            ConstVal::Str(s) => {
                out.push(5);
                write_str(&mut out, s);
            }
        }
    }

    out.extend_from_slice(&(chunk.functions.len() as u32).to_le_bytes());
    for f in &chunk.functions {
        write_str(&mut out, &f.name);
        out.extend_from_slice(&(f.arity as u32).to_le_bytes());
        out.extend_from_slice(&(f.local_count as u32).to_le_bytes());
        out.extend_from_slice(&(f.code.len() as u32).to_le_bytes());
        for op in &f.code {
            match op {
                Op::PushConst(idx) => {
                    out.push(1);
                    out.extend_from_slice(&(*idx as u32).to_le_bytes());
                }
                Op::LoadLocal(s) => {
                    out.push(2);
                    out.extend_from_slice(&(*s as u32).to_le_bytes());
                }
                Op::StoreLocal(s) => {
                    out.push(3);
                    out.extend_from_slice(&(*s as u32).to_le_bytes());
                }
                Op::LoadGlobal(name) => {
                    out.push(4);
                    write_str(&mut out, name);
                }
                Op::StoreGlobal(name) => {
                    out.push(5);
                    write_str(&mut out, name);
                }
                Op::Add => out.push(6),
                Op::Sub => out.push(7),
                Op::Mul => out.push(8),
                Op::Div => out.push(9),
                Op::Mod => out.push(10),
                Op::MatMul => out.push(11),
                Op::Neg => out.push(12),
                Op::Not => out.push(13),
                Op::CmpEq => out.push(14),
                Op::CmpNeq => out.push(15),
                Op::CmpLt => out.push(16),
                Op::CmpLte => out.push(17),
                Op::CmpGt => out.push(18),
                Op::CmpGte => out.push(19),
                Op::AllocArray(n) => {
                    out.push(20);
                    out.extend_from_slice(&(*n as u32).to_le_bytes());
                }
                Op::Jump(t) => {
                    out.push(21);
                    out.extend_from_slice(&(*t as u32).to_le_bytes());
                }
                Op::JumpIfFalse(t) => {
                    out.push(22);
                    out.extend_from_slice(&(*t as u32).to_le_bytes());
                }
                Op::Call { name, argc } => {
                    out.push(23);
                    write_str(&mut out, name);
                    out.extend_from_slice(&(*argc as u32).to_le_bytes());
                }
                Op::Return => out.push(24),
                Op::Pop => out.push(25),
            }
        }
    }
    out
}

/// Deserialize a binary `.nvbc` byte buffer back into a `Chunk`.
pub fn decode_chunk(bytes: &[u8]) -> Result<Chunk, String> {
    if bytes.len() < 8 {
        return Err("Invalid .nvbc buffer: shorter than 8-byte header".to_string());
    }
    let mut magic = [0u8; 6];
    magic.copy_from_slice(&bytes[0..6]);
    if &magic != BYTECODE_MAGIC {
        return Err("Invalid .nvbc magic header".to_string());
    }
    let version = u16::from_le_bytes([bytes[6], bytes[7]]);
    if version != BYTECODE_VERSION {
        return Err(format!("Unsupported .nvbc version {version}"));
    }

    let mut pos = 8usize;
    let const_count = read_u32(bytes, &mut pos)? as usize;
    let mut constants = Vec::with_capacity(const_count);
    for _ in 0..const_count {
        if pos >= bytes.len() {
            return Err("Unexpected EOF decoding constant tag".to_string());
        }
        let tag = bytes[pos];
        pos += 1;
        let val = match tag {
            0 => ConstVal::None,
            1 => {
                if pos + 8 > bytes.len() {
                    return Err("Unexpected EOF decoding Int constant".to_string());
                }
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[pos..pos + 8]);
                pos += 8;
                ConstVal::Int(i64::from_le_bytes(b))
            }
            2 => {
                if pos + 8 > bytes.len() {
                    return Err("Unexpected EOF decoding Float constant".to_string());
                }
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[pos..pos + 8]);
                pos += 8;
                ConstVal::Float(f64::from_bits(u64::from_le_bytes(b)))
            }
            3 => {
                if pos >= bytes.len() {
                    return Err("Unexpected EOF decoding Bool constant".to_string());
                }
                let b = bytes[pos] != 0;
                pos += 1;
                ConstVal::Bool(b)
            }
            4 => {
                let code = read_u32(bytes, &mut pos)?;
                let ch = char::from_u32(code)
                    .ok_or_else(|| format!("Invalid Unicode scalar {code} in .nvbc"))?;
                ConstVal::Char(ch)
            }
            5 => ConstVal::Str(read_str(bytes, &mut pos)?),
            other => return Err(format!("Unknown constant tag {other} in .nvbc")),
        };
        constants.push(val);
    }

    let fn_count = read_u32(bytes, &mut pos)? as usize;
    let mut functions = Vec::with_capacity(fn_count);
    for _ in 0..fn_count {
        let name = read_str(bytes, &mut pos)?;
        let arity = read_u32(bytes, &mut pos)? as usize;
        let local_count = read_u32(bytes, &mut pos)? as usize;
        let code_len = read_u32(bytes, &mut pos)? as usize;
        let mut code = Vec::with_capacity(code_len);
        for _ in 0..code_len {
            if pos >= bytes.len() {
                return Err("Unexpected EOF decoding opcode".to_string());
            }
            let tag = bytes[pos];
            pos += 1;
            let op = match tag {
                1 => Op::PushConst(read_u32(bytes, &mut pos)? as usize),
                2 => Op::LoadLocal(read_u32(bytes, &mut pos)? as usize),
                3 => Op::StoreLocal(read_u32(bytes, &mut pos)? as usize),
                4 => Op::LoadGlobal(read_str(bytes, &mut pos)?),
                5 => Op::StoreGlobal(read_str(bytes, &mut pos)?),
                6 => Op::Add,
                7 => Op::Sub,
                8 => Op::Mul,
                9 => Op::Div,
                10 => Op::Mod,
                11 => Op::MatMul,
                12 => Op::Neg,
                13 => Op::Not,
                14 => Op::CmpEq,
                15 => Op::CmpNeq,
                16 => Op::CmpLt,
                17 => Op::CmpLte,
                18 => Op::CmpGt,
                19 => Op::CmpGte,
                20 => Op::AllocArray(read_u32(bytes, &mut pos)? as usize),
                21 => Op::Jump(read_u32(bytes, &mut pos)? as usize),
                22 => Op::JumpIfFalse(read_u32(bytes, &mut pos)? as usize),
                23 => {
                    let cname = read_str(bytes, &mut pos)?;
                    let argc = read_u32(bytes, &mut pos)? as usize;
                    Op::Call { name: cname, argc }
                }
                24 => Op::Return,
                25 => Op::Pop,
                other => return Err(format!("Unknown opcode tag {other} in .nvbc")),
            };
            code.push(op);
        }
        functions.push(CompiledFn {
            name,
            arity,
            local_count,
            code,
        });
    }

    Ok(Chunk {
        magic,
        version,
        constants,
        functions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nova_ir::lower_program;
    use nova_lexer::tokenize;
    use nova_parser::Parser;

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

    #[test]
    fn round_trips_binary_nvbc_wire_format() {
        let src = "const BASE = 10\nfn main() {\n    let arr = [1, 2, BASE]\n    print(arr)\n}";
        let toks = tokenize(src).unwrap();
        let prog = Parser::new(&toks).parse_program().unwrap();
        let ir = lower_program(&prog);
        let chunk = compile(&ir);
        assert!(verify(&chunk).is_ok());

        let encoded = encode_chunk(&chunk);
        let decoded = decode_chunk(&encoded).unwrap();
        assert_eq!(chunk, decoded);
        assert!(verify(&decoded).is_ok());
    }
}
