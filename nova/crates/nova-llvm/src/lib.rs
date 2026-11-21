//! Nova IR -> LLVM 17+ opaque-pointer IR (`.ll`) lowering, structural verifier,
//! and native object compilation, linking, and release-strip build driver.

use nova_ir::{CmpOp, ConstVal, Inst, IrFunction, IrModule};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildProfile {
    Debug,
    Release,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlvmValidationReport {
    pub ok: bool,
    pub functions_checked: usize,
    pub blocks_checked: usize,
    pub instructions_checked: usize,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LlvmLowerResult {
    pub target_triple: String,
    pub llvm_ir: String,
    pub validation: LlvmValidationReport,
}

#[derive(Debug, Clone)]
pub struct NativeBuildArtifact {
    pub output_path: PathBuf,
    pub profile: BuildProfile,
    pub stripped: bool,
    pub binary_bytes: u64,
    pub llvm_ir_path: Option<PathBuf>,
}

pub fn host_target_triple() -> &'static str {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        "arm64-apple-macosx14.0.0"
    }
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        "aarch64-unknown-linux-gnu"
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "aarch64", target_os = "linux")
    )))]
    {
        "x86_64-unknown-linux-gnu"
    }
}

fn sanitize_ident(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('x');
    }
    out
}

fn escape_llvm_bytes(s: &str) -> (String, usize) {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len() + 4);
    for &b in bytes {
        if (32..=126).contains(&b) && b != b'"' && b != b'\\' {
            out.push(b as char);
        } else {
            out.push_str(&format!("\\{:02X}", b));
        }
    }
    out.push_str("\\00");
    (out, bytes.len() + 1)
}

fn format_llvm_f64(v: f64) -> String {
    if !v.is_finite() {
        return "0.0".to_string();
    }
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

struct StringPool {
    entries: BTreeMap<String, (String, usize, String)>,
    next_id: usize,
}

impl StringPool {
    fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            next_id: 0,
        }
    }

    fn intern(&mut self, value: &str) -> (String, usize) {
        if let Some((sym, len, _)) = self.entries.get(value) {
            return (sym.clone(), *len);
        }
        let sym = format!("@.str.{}", self.next_id);
        self.next_id += 1;
        let (escaped, len) = escape_llvm_bytes(value);
        self.entries
            .insert(value.to_string(), (sym.clone(), len, escaped));
        (sym, len)
    }

    fn emit_globals(&self, out: &mut Vec<String>) {
        for (sym, len, escaped) in self.entries.values() {
            out.push(format!(
                "{sym} = private unnamed_addr constant [{len} x i8] c\"{escaped}\", align 1"
            ));
        }
    }
}

/// Lower a `nova_ir::IrModule` to verified LLVM 17+ opaque-pointer IR targeting `libnovart`.
pub fn lower_to_llvm(module: &IrModule) -> LlvmLowerResult {
    let target_triple = host_target_triple().to_string();
    let mut pool = StringPool::new();
    let user_fns: HashSet<String> = module.functions.iter().map(|f| f.name.clone()).collect();
    let mut fn_defs: Vec<String> = Vec::new();

    for func in &module.functions {
        fn_defs.push(lower_function(func, &user_fns, &mut pool));
    }

    let mut lines: Vec<String> = Vec::new();
    lines.push("; ModuleID = 'nova_module'".to_string());
    lines.push("source_filename = \"program.nova\"".to_string());
    lines.push(format!("target triple = \"{target_triple}\""));
    lines.push(String::new());
    lines.push("%struct.NVal = type { i32, i64, double, i1, ptr }".to_string());
    lines.push(String::new());

    // Runtime C ABI declarations (libnovart)
    let decls = [
        "declare %struct.NVal @nv_none()",
        "declare %struct.NVal @nv_int(i64)",
        "declare %struct.NVal @nv_float(double)",
        "declare %struct.NVal @nv_bool(i1)",
        "declare %struct.NVal @nv_str_copy(ptr)",
        "declare i1 @nv_truthy(%struct.NVal)",
        "declare %struct.NVal @nv_add(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_sub(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_mul(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_div(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_mod(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_matmul(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_not(%struct.NVal)",
        "declare %struct.NVal @nv_eq(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_neq(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_lt(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_lte(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_gt(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_gte(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_call_builtin(ptr, i32, ptr)",
        "declare %struct.NVal @nv_builtin_print(ptr, i32)",
        "declare %struct.NVal @nv_builtin_str(%struct.NVal)",
        "declare %struct.NVal @nv_builtin_int(%struct.NVal)",
        "declare %struct.NVal @nv_builtin_float(%struct.NVal)",
        "declare %struct.NVal @nv_builtin_abs(%struct.NVal)",
        "declare %struct.NVal @nv_builtin_sqrt(%struct.NVal)",
        "declare %struct.NVal @nv_builtin_pow(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_builtin_min(%struct.NVal, %struct.NVal)",
        "declare %struct.NVal @nv_builtin_max(%struct.NVal, %struct.NVal)",
    ];
    for d in decls {
        lines.push(d.to_string());
    }
    lines.push(String::new());

    pool.emit_globals(&mut lines);
    if !pool.entries.is_empty() {
        lines.push(String::new());
    }

    for def in fn_defs {
        lines.push(def);
        lines.push(String::new());
    }

    // C ABI main() entrypoint
    lines.push("define i32 @main() {".to_string());
    lines.push("entry:".to_string());
    if user_fns.contains("main") {
        lines.push("  %_entry_ret = call %struct.NVal @nova_fn_main()".to_string());
    }
    lines.push("  ret i32 0".to_string());
    lines.push("}".to_string());

    let llvm_ir = lines.join("\n");
    let validation = validate_llvm_ir(&llvm_ir);
    LlvmLowerResult {
        target_triple,
        llvm_ir,
        validation,
    }
}

fn lower_function(func: &IrFunction, user_fns: &HashSet<String>, pool: &mut StringPool) -> String {
    let mut locals: BTreeSet<String> = BTreeSet::new();
    let mut temps: BTreeSet<String> = BTreeSet::new();

    for p in &func.params {
        locals.insert(p.clone());
    }
    for block in &func.blocks {
        for inst in &block.instructions {
            match inst {
                Inst::Const { dst, .. } => {
                    temps.insert(dst.clone());
                }
                Inst::Load { dst, slot } => {
                    temps.insert(dst.clone());
                    locals.insert(slot.clone());
                }
                Inst::Store { slot, src } => {
                    locals.insert(slot.clone());
                    temps.insert(src.clone());
                }
                Inst::Add { dst, left, right }
                | Inst::Sub { dst, left, right }
                | Inst::Mul { dst, left, right }
                | Inst::Div { dst, left, right }
                | Inst::Mod { dst, left, right }
                | Inst::Compare {
                    dst, left, right, ..
                } => {
                    temps.insert(dst.clone());
                    temps.insert(left.clone());
                    temps.insert(right.clone());
                }
                Inst::Not { dst, src } => {
                    temps.insert(dst.clone());
                    temps.insert(src.clone());
                }
                Inst::Call { dst, args, .. } => {
                    temps.insert(dst.clone());
                    for a in args {
                        temps.insert(a.clone());
                    }
                }
                Inst::JumpIfFalse { cond, .. } => {
                    temps.insert(cond.clone());
                }
                Inst::Return { src } => {
                    if let Some(s) = src {
                        temps.insert(s.clone());
                    }
                }
                Inst::Jump { .. } => {}
            }
        }
    }

    let params_sig = func
        .params
        .iter()
        .map(|p| format!("%struct.NVal %arg_{}", sanitize_ident(p)))
        .collect::<Vec<_>>()
        .join(", ");

    let mut out = Vec::new();
    out.push(format!(
        "define internal %struct.NVal @nova_fn_{}({params_sig}) {{",
        sanitize_ident(&func.name)
    ));
    out.push("alloc_entry:".to_string());

    let mut reg_id: usize = 0;
    let mut next_reg = |prefix: &str| {
        reg_id += 1;
        format!("%{prefix}_{reg_id}")
    };

    for l in &locals {
        out.push(format!("  %var_{} = alloca %struct.NVal, align 8", sanitize_ident(l)));
    }
    for t in &temps {
        out.push(format!("  %tmp_{} = alloca %struct.NVal, align 8", sanitize_ident(t)));
    }

    if !locals.is_empty() || !temps.is_empty() {
        let init_none = next_reg("none");
        out.push(format!("  {init_none} = call %struct.NVal @nv_none()"));
        for l in &locals {
            out.push(format!(
                "  store %struct.NVal {init_none}, ptr %var_{}, align 8",
                sanitize_ident(l)
            ));
        }
        for t in &temps {
            out.push(format!(
                "  store %struct.NVal {init_none}, ptr %tmp_{}, align 8",
                sanitize_ident(t)
            ));
        }
    }

    for p in &func.params {
        let sp = sanitize_ident(p);
        out.push(format!(
            "  store %struct.NVal %arg_{sp}, ptr %var_{sp}, align 8"
        ));
    }

    let first_label = func
        .blocks
        .first()
        .map(|b| sanitize_ident(&b.label))
        .unwrap_or_else(|| "exit_ret".to_string());
    out.push(format!("  br label %{first_label}"));

    if func.blocks.is_empty() {
        out.push("exit_ret:".to_string());
        let r = next_reg("ret");
        out.push(format!("  {r} = call %struct.NVal @nv_none()"));
        out.push(format!("  ret %struct.NVal {r}"));
        out.push("}".to_string());
        return out.join("\n");
    }

    for (bi, block) in func.blocks.iter().enumerate() {
        let label = sanitize_ident(&block.label);
        out.push(format!("{label}:"));
        let mut terminated = false;

        for inst in &block.instructions {
            if terminated {
                break;
            }
            match inst {
                Inst::Const { dst, val } => {
                    let r = next_reg("c");
                    match val {
                        ConstVal::Int(v) => {
                            out.push(format!("  {r} = call %struct.NVal @nv_int(i64 {v})"));
                        }
                        ConstVal::Float(v) => {
                            out.push(format!(
                                "  {r} = call %struct.NVal @nv_float(double {})",
                                format_llvm_f64(*v)
                            ));
                        }
                        ConstVal::Bool(b) => {
                            out.push(format!(
                                "  {r} = call %struct.NVal @nv_bool(i1 {})",
                                if *b { 1 } else { 0 }
                            ));
                        }
                        ConstVal::Str(s) => {
                            let (sym, len) = pool.intern(s);
                            let gep = next_reg("strptr");
                            out.push(format!(
                                "  {gep} = getelementptr inbounds [{len} x i8], ptr {sym}, i64 0, i64 0"
                            ));
                            out.push(format!("  {r} = call %struct.NVal @nv_str_copy(ptr {gep})"));
                        }
                        ConstVal::None => {
                            out.push(format!("  {r} = call %struct.NVal @nv_none()"));
                        }
                    }
                    out.push(format!(
                        "  store %struct.NVal {r}, ptr %tmp_{}, align 8",
                        sanitize_ident(dst)
                    ));
                }
                Inst::Load { dst, slot } => {
                    let r = next_reg("ld");
                    out.push(format!(
                        "  {r} = load %struct.NVal, ptr %var_{}, align 8",
                        sanitize_ident(slot)
                    ));
                    out.push(format!(
                        "  store %struct.NVal {r}, ptr %tmp_{}, align 8",
                        sanitize_ident(dst)
                    ));
                }
                Inst::Store { slot, src } => {
                    let r = next_reg("st");
                    out.push(format!(
                        "  {r} = load %struct.NVal, ptr %tmp_{}, align 8",
                        sanitize_ident(src)
                    ));
                    out.push(format!(
                        "  store %struct.NVal {r}, ptr %var_{}, align 8",
                        sanitize_ident(slot)
                    ));
                }
                Inst::Add { dst, left, right }
                | Inst::Sub { dst, left, right }
                | Inst::Mul { dst, left, right }
                | Inst::Div { dst, left, right }
                | Inst::Mod { dst, left, right } => {
                    let helper = match inst {
                        Inst::Add { .. } => "@nv_add",
                        Inst::Sub { .. } => "@nv_sub",
                        Inst::Mul { .. } => "@nv_mul",
                        Inst::Div { .. } => "@nv_div",
                        Inst::Mod { .. } => "@nv_mod",
                        _ => unreachable!(),
                    };
                    let lv = next_reg("l");
                    let rv = next_reg("r");
                    let res = next_reg("bin");
                    out.push(format!(
                        "  {lv} = load %struct.NVal, ptr %tmp_{}, align 8",
                        sanitize_ident(left)
                    ));
                    out.push(format!(
                        "  {rv} = load %struct.NVal, ptr %tmp_{}, align 8",
                        sanitize_ident(right)
                    ));
                    out.push(format!(
                        "  {res} = call %struct.NVal {helper}(%struct.NVal {lv}, %struct.NVal {rv})"
                    ));
                    out.push(format!(
                        "  store %struct.NVal {res}, ptr %tmp_{}, align 8",
                        sanitize_ident(dst)
                    ));
                }
                Inst::Not { dst, src } => {
                    let v = next_reg("u");
                    let res = next_reg("not");
                    out.push(format!(
                        "  {v} = load %struct.NVal, ptr %tmp_{}, align 8",
                        sanitize_ident(src)
                    ));
                    out.push(format!(
                        "  {res} = call %struct.NVal @nv_not(%struct.NVal {v})"
                    ));
                    out.push(format!(
                        "  store %struct.NVal {res}, ptr %tmp_{}, align 8",
                        sanitize_ident(dst)
                    ));
                }
                Inst::Compare {
                    dst,
                    cmp,
                    left,
                    right,
                } => {
                    let helper = match cmp {
                        CmpOp::Eq => "@nv_eq",
                        CmpOp::NotEq => "@nv_neq",
                        CmpOp::Lt => "@nv_lt",
                        CmpOp::LtEq => "@nv_lte",
                        CmpOp::Gt => "@nv_gt",
                        CmpOp::GtEq => "@nv_gte",
                    };
                    let lv = next_reg("cl");
                    let rv = next_reg("cr");
                    let res = next_reg("cmp");
                    out.push(format!(
                        "  {lv} = load %struct.NVal, ptr %tmp_{}, align 8",
                        sanitize_ident(left)
                    ));
                    out.push(format!(
                        "  {rv} = load %struct.NVal, ptr %tmp_{}, align 8",
                        sanitize_ident(right)
                    ));
                    out.push(format!(
                        "  {res} = call %struct.NVal {helper}(%struct.NVal {lv}, %struct.NVal {rv})"
                    ));
                    out.push(format!(
                        "  store %struct.NVal {res}, ptr %tmp_{}, align 8",
                        sanitize_ident(dst)
                    ));
                }
                Inst::Call { dst, callee, args } => {
                    let mut loaded_args = Vec::with_capacity(args.len());
                    for a in args {
                        let av = next_reg("arg");
                        out.push(format!(
                            "  {av} = load %struct.NVal, ptr %tmp_{}, align 8",
                            sanitize_ident(a)
                        ));
                        loaded_args.push(av);
                    }
                    let res = next_reg("call");
                    if callee == "print" {
                        let argc = loaded_args.len();
                        let arr = next_reg("pargs");
                        let alloc_len = argc.max(1);
                        out.push(format!(
                            "  {arr} = alloca [{alloc_len} x %struct.NVal], align 8"
                        ));
                        for (idx, av) in loaded_args.iter().enumerate() {
                            let gep = next_reg("pgep");
                            out.push(format!(
                                "  {gep} = getelementptr inbounds [{alloc_len} x %struct.NVal], ptr {arr}, i64 0, i64 {idx}"
                            ));
                            out.push(format!("  store %struct.NVal {av}, ptr {gep}, align 8"));
                        }
                        let base = next_reg("pbase");
                        out.push(format!(
                            "  {base} = getelementptr inbounds [{alloc_len} x %struct.NVal], ptr {arr}, i64 0, i64 0"
                        ));
                        out.push(format!(
                            "  {res} = call %struct.NVal @nv_builtin_print(ptr {base}, i32 {argc})"
                        ));
                    } else if user_fns.contains(callee) {
                        let sig = loaded_args
                            .iter()
                            .map(|a| format!("%struct.NVal {a}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        out.push(format!(
                            "  {res} = call %struct.NVal @nova_fn_{}({sig})",
                            sanitize_ident(callee)
                        ));
                    } else {
                        let helper = match callee.as_str() {
                            "str" => Some("@nv_builtin_str"),
                            "int" => Some("@nv_builtin_int"),
                            "float" => Some("@nv_builtin_float"),
                            "abs" => Some("@nv_builtin_abs"),
                            "sqrt" => Some("@nv_builtin_sqrt"),
                            "pow" => Some("@nv_builtin_pow"),
                            "min" => Some("@nv_builtin_min"),
                            "max" => Some("@nv_builtin_max"),
                            _ => None,
                        };
                        if let Some(h) = helper {
                            let sig = loaded_args
                                .iter()
                                .map(|a| format!("%struct.NVal {a}"))
                                .collect::<Vec<_>>()
                                .join(", ");
                            out.push(format!("  {res} = call %struct.NVal {h}({sig})"));
                        } else {
                            let (sym, slen) = pool.intern(callee);
                            let name_ptr = next_reg("bname");
                            out.push(format!(
                                "  {name_ptr} = getelementptr inbounds [{slen} x i8], ptr {sym}, i64 0, i64 0"
                            ));
                            let argc = loaded_args.len();
                            if argc == 0 {
                                out.push(format!(
                                    "  {res} = call %struct.NVal @nv_call_builtin(ptr {name_ptr}, i32 0, ptr null)"
                                ));
                            } else {
                                let arr = next_reg("bargs");
                                out.push(format!("  {arr} = alloca [{argc} x %struct.NVal], align 8"));
                                for (idx, av) in loaded_args.iter().enumerate() {
                                    let gep = next_reg("bgep");
                                    out.push(format!(
                                        "  {gep} = getelementptr inbounds [{argc} x %struct.NVal], ptr {arr}, i64 0, i64 {idx}"
                                    ));
                                    out.push(format!("  store %struct.NVal {av}, ptr {gep}, align 8"));
                                }
                                let base = next_reg("bbase");
                                out.push(format!(
                                    "  {base} = getelementptr inbounds [{argc} x %struct.NVal], ptr {arr}, i64 0, i64 0"
                                ));
                                out.push(format!(
                                    "  {res} = call %struct.NVal @nv_call_builtin(ptr {name_ptr}, i32 {argc}, ptr {base})"
                                ));
                            }
                        }
                    }
                    out.push(format!(
                        "  store %struct.NVal {res}, ptr %tmp_{}, align 8",
                        sanitize_ident(dst)
                    ));
                }
                Inst::Jump { target } => {
                    out.push(format!("  br label %{}", sanitize_ident(target)));
                    terminated = true;
                }
                Inst::JumpIfFalse { cond, target } => {
                    let cv = next_reg("condv");
                    let tb = next_reg("truthy");
                    let cont_label = format!("cont_{}", {
                        reg_id += 1;
                        reg_id
                    });
                    out.push(format!(
                        "  {cv} = load %struct.NVal, ptr %tmp_{}, align 8",
                        sanitize_ident(cond)
                    ));
                    out.push(format!("  {tb} = call i1 @nv_truthy(%struct.NVal {cv})"));
                    out.push(format!(
                        "  br i1 {tb}, label %{cont_label}, label %{}",
                        sanitize_ident(target)
                    ));
                    out.push(format!("{cont_label}:"));
                }
                Inst::Return { src } => {
                    if let Some(s) = src {
                        let rv = next_reg("retv");
                        out.push(format!(
                            "  {rv} = load %struct.NVal, ptr %tmp_{}, align 8",
                            sanitize_ident(s)
                        ));
                        out.push(format!("  ret %struct.NVal {rv}"));
                    } else {
                        let rv = next_reg("retn");
                        out.push(format!("  {rv} = call %struct.NVal @nv_none()"));
                        out.push(format!("  ret %struct.NVal {rv}"));
                    }
                    terminated = true;
                }
            }
        }

        if !terminated {
            if let Some(next_blk) = func.blocks.get(bi + 1) {
                out.push(format!("  br label %{}", sanitize_ident(&next_blk.label)));
            } else {
                let rv = next_reg("fallret");
                out.push(format!("  {rv} = call %struct.NVal @nv_none()"));
                out.push(format!("  ret %struct.NVal {rv}"));
            }
        }
    }

    out.push("}".to_string());
    out.join("\n")
}

/// Static structural verifier for emitted LLVM IR modules.
pub fn validate_llvm_ir(ir: &str) -> LlvmValidationReport {
    let mut errors = Vec::new();
    if !ir.contains("target triple = \"") {
        errors.push("Missing LLVM target triple declaration.".to_string());
    }

    let mut functions_checked = 0usize;
    let mut blocks_checked = 0usize;
    let mut instructions_checked = 0usize;

    let mut in_fn = false;
    let mut cur_fn = String::new();
    let mut cur_block: Option<String> = None;
    let mut block_terminated = false;
    let mut defined_labels: HashSet<String> = HashSet::new();
    let mut referenced_labels: Vec<String> = Vec::new();
    let mut ssa_regs: HashSet<String> = HashSet::new();

    for raw in ir.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with(';') {
            continue;
        }
        if line.starts_with("define ") && line.ends_with('{') {
            in_fn = true;
            functions_checked += 1;
            cur_fn = line
                .split('@')
                .nth(1)
                .and_then(|s| s.split('(').next())
                .unwrap_or("unknown")
                .to_string();
            cur_block = None;
            block_terminated = false;
            defined_labels.clear();
            referenced_labels.clear();
            ssa_regs.clear();
            continue;
        }
        if in_fn && line == "}" {
            if let Some(ref blk) = cur_block {
                if !block_terminated {
                    errors.push(format!(
                        "Function @{cur_fn}: block '{blk}' is missing a terminator instruction."
                    ));
                }
            }
            for target in &referenced_labels {
                if !defined_labels.contains(target) {
                    errors.push(format!(
                        "Function @{cur_fn}: branch references undefined label %{target}."
                    ));
                }
            }
            in_fn = false;
            continue;
        }
        if !in_fn {
            continue;
        }
        if let Some(label) = line.strip_suffix(':') {
            if !label.contains(' ') {
                if let Some(ref prev) = cur_block {
                    if !block_terminated {
                        errors.push(format!(
                            "Function @{cur_fn}: block '{prev}' is missing a terminator instruction."
                        ));
                    }
                }
                blocks_checked += 1;
                defined_labels.insert(label.to_string());
                cur_block = Some(label.to_string());
                block_terminated = false;
                continue;
            }
        }

        instructions_checked += 1;
        if let Some(eq_pos) = line.find(" = ") {
            let lhs = line[..eq_pos].trim();
            if lhs.starts_with('%') && !ssa_regs.insert(lhs.to_string()) {
                errors.push(format!(
                    "Function @{cur_fn}: duplicate SSA register assignment to {lhs}."
                ));
            }
        }
        if line.starts_with("br ") || line.starts_with("ret ") || line == "unreachable" {
            block_terminated = true;
            if line.starts_with("br ") {
                for part in line.split("label %").skip(1) {
                    let lbl: String = part
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.')
                        .collect();
                    if !lbl.is_empty() {
                        referenced_labels.push(lbl);
                    }
                }
            }
        }
    }

    LlvmValidationReport {
        ok: errors.is_empty(),
        functions_checked,
        blocks_checked,
        instructions_checked,
        errors,
    }
}

/// Compile emitted LLVM IR `.ll` + a `novart.c` runtime file to a native binary via `clang`
/// (and run `strip --strip-all` for `BuildProfile::Release`).
pub fn build_native_with_clang(
    module: &IrModule,
    novart_c_path: &Path,
    output_path: &Path,
    profile: BuildProfile,
    emit_llvm_path: Option<&Path>,
) -> Result<NativeBuildArtifact, String> {
    let lowered = lower_to_llvm(module);
    if !lowered.validation.ok {
        return Err(format!(
            "LLVM IR validation failed: {}",
            lowered.validation.errors.join("; ")
        ));
    }

    let tmp_dir = std::env::temp_dir().join(format!("nova-llvm-build-{}", std::process::id()));
    fs::create_dir_all(&tmp_dir).map_err(|e| format!("failed to create temp build dir: {e}"))?;

    let ll_file = tmp_dir.join("program.ll");
    let prog_obj = tmp_dir.join("program.o");
    let rt_obj = tmp_dir.join("libnovart.o");
    fs::write(&ll_file, &lowered.llvm_ir)
        .map_err(|e| format!("failed to write LLVM IR file: {e}"))?;

    if let Some(emit_p) = emit_llvm_path {
        fs::write(emit_p, &lowered.llvm_ir)
            .map_err(|e| format!("failed to emit LLVM IR to {}: {e}", emit_p.display()))?;
    }

    let opt_flag = match profile {
        BuildProfile::Debug => "-O0",
        BuildProfile::Release => "-O3",
    };

    let mut prog_cmd = Command::new("clang");
    prog_cmd
        .arg("-c")
        .arg(opt_flag)
        .arg(&ll_file)
        .arg("-o")
        .arg(&prog_obj);
    if profile == BuildProfile::Debug {
        prog_cmd.arg("-g");
    }
    let status = prog_cmd
        .status()
        .map_err(|e| format!("failed to invoke clang (is LLVM/clang installed on PATH?): {e}"))?;
    if !status.success() {
        let _ = fs::remove_dir_all(&tmp_dir);
        return Err("clang failed while compiling LLVM IR to object file".to_string());
    }

    let mut rt_cmd = Command::new("clang");
    rt_cmd
        .arg("-c")
        .arg("-std=c99")
        .arg(opt_flag)
        .arg(novart_c_path)
        .arg("-o")
        .arg(&rt_obj);
    let rt_status = rt_cmd
        .status()
        .map_err(|e| format!("failed to invoke clang for libnovart.o: {e}"))?;
    if !rt_status.success() {
        let _ = fs::remove_dir_all(&tmp_dir);
        return Err("clang failed while compiling libnovart.o".to_string());
    }

    let mut link_cmd = Command::new("clang");
    link_cmd
        .arg(opt_flag)
        .arg(&prog_obj)
        .arg(&rt_obj)
        .arg("-o")
        .arg(output_path)
        .arg("-lm");
    let link_status = link_cmd
        .status()
        .map_err(|e| format!("failed to link native executable: {e}"))?;
    let _ = fs::remove_dir_all(&tmp_dir);
    if !link_status.success() {
        return Err("native linker failed".to_string());
    }

    let mut stripped = false;
    if profile == BuildProfile::Release {
        if let Ok(st) = Command::new("strip")
            .arg("--strip-all")
            .arg(output_path)
            .status()
        {
            stripped = st.success();
        }
    }

    let binary_bytes = fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    Ok(NativeBuildArtifact {
        output_path: output_path.to_path_buf(),
        profile,
        stripped,
        binary_bytes,
        llvm_ir_path: emit_llvm_path.map(|p| p.to_path_buf()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nova_ir::{fold_constants, lower_program};
    use nova_lexer::tokenize;
    use nova_parser::Parser;

    #[test]
    fn lowers_and_validates_hello_world_to_llvm_ir() {
        let src = r#"
fn add(a: Int, b: Int) -> Int {
    return a + b
}

fn main() {
    let x = add(10, 32)
    print(x)
}
"#;
        let tokens = tokenize(src).unwrap();
        let program = Parser::new(&tokens).parse_program().unwrap();
        let mut ir = lower_program(&program);
        fold_constants(&mut ir);

        let lowered = lower_to_llvm(&ir);
        assert!(lowered.validation.ok, "errors: {:?}", lowered.validation.errors);
        assert!(lowered.llvm_ir.contains("%struct.NVal = type { i32, i64, double, i1, ptr }"));
        assert!(lowered.llvm_ir.contains("define internal %struct.NVal @nova_fn_add("));
        assert!(lowered.llvm_ir.contains("define i32 @main()"));
    }

    #[test]
    fn rejects_broken_llvm_ir_in_validator() {
        let broken = "define i32 @main() {\nentry:\n  %x = call %struct.NVal @nv_int(i64 1)\n  %x = call %struct.NVal @nv_int(i64 2)\n  br label %missing_block\n}\n";
        let report = validate_llvm_ir(broken);
        assert!(!report.ok);
        assert!(report.errors.len() >= 3);
    }
}
