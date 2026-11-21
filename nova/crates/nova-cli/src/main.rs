//! Nova CLI executable (`nova run [--engine=interpreter|vm]`, `nova llvm`, `nova build [--release]`, `nova verify`, `nova check`, `nova version`).

use nova_bytecode::{compile, verify};
use nova_interpreter::Interpreter;
use nova_ir::{fold_constants, lower_program};
use nova_lexer::tokenize;
use nova_llvm::{lower_to_llvm, BuildProfile};
use nova_parser::Parser;
use nova_semantic::check_program;
use nova_vm::Vm;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: nova <run|llvm|build|verify|check|version> [file.nova] [--engine=interpreter|vm] [--debug|--release]"
        );
        process::exit(1);
    }

    match args[1].as_str() {
        "version" | "--version" => {
            println!("nova {VERSION}");
        }
        "check" | "verify" | "llvm" | "build" | "run" => {
            let cmd = args[1].as_str();
            let use_vm = args.iter().any(|a| a == "--engine=vm");
            let profile = if args.iter().any(|a| a == "--debug") {
                BuildProfile::Debug
            } else {
                BuildProfile::Release
            };
            let path = match args.iter().skip(2).find(|a| !a.starts_with("--")) {
                Some(p) => p,
                None => {
                    eprintln!("error: expected a .nova source file");
                    process::exit(1);
                }
            };
            let source = match fs::read_to_string(path) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("error: failed to read '{path}': {err}");
                    process::exit(1);
                }
            };
            let tokens = match tokenize(&source) {
                Ok(t) => t,
                Err(errs) => {
                    for e in errs {
                        eprintln!("{e}");
                    }
                    process::exit(1);
                }
            };
            let program = match Parser::new(&tokens).parse_program() {
                Ok(p) => p,
                Err(errs) => {
                    for e in errs {
                        eprintln!("{e}");
                    }
                    process::exit(1);
                }
            };
            if let Err(errs) = check_program(&program) {
                for e in errs {
                    eprintln!("{e}");
                }
                process::exit(1);
            }

            if cmd == "check" {
                println!("OK: {path}");
                return;
            }

            if cmd == "llvm" {
                let mut ir = lower_program(&program);
                fold_constants(&mut ir);
                let lowered = lower_to_llvm(&ir);
                if !lowered.validation.ok {
                    for err in &lowered.validation.errors {
                        eprintln!("llvm verification error: {err}");
                    }
                    process::exit(1);
                }
                println!("{}", lowered.llvm_ir);
                return;
            }

            if cmd == "build" {
                let mut ir = lower_program(&program);
                fold_constants(&mut ir);
                let lowered = lower_to_llvm(&ir);
                if !lowered.validation.ok {
                    for err in &lowered.validation.errors {
                        eprintln!("llvm verification error: {err}");
                    }
                    process::exit(1);
                }
                let stem = PathBuf::from(path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("out")
                    .to_string();
                let ll_out = PathBuf::from(format!("{stem}.ll"));
                if let Err(err) = fs::write(&ll_out, &lowered.llvm_ir) {
                    eprintln!("error: failed to write {}: {err}", ll_out.display());
                    process::exit(1);
                }
                let profile_label = match profile {
                    BuildProfile::Debug => "debug (-O0 -g)",
                    BuildProfile::Release => "release (-O3 + strip)",
                };
                println!(
                    "EMITTED LLVM IR ({profile_label}): {} ({} fns, {} blocks, {} insts)",
                    ll_out.display(),
                    lowered.validation.functions_checked,
                    lowered.validation.blocks_checked,
                    lowered.validation.instructions_checked
                );
                return;
            }

            if cmd == "verify" {
                let mut ir = lower_program(&program);
                fold_constants(&mut ir);
                let chunk = compile(&ir);
                match verify(&chunk) {
                    Ok(report) => {
                        println!(
                            "VERIFIED: {path} ({} functions, {} instructions, max stack {})",
                            report.functions_verified,
                            report.instructions_verified,
                            report.max_stack_depth
                        );
                    }
                    Err(errs) => {
                        for e in errs {
                            eprintln!("{e}");
                        }
                        process::exit(1);
                    }
                }
                return;
            }

            if use_vm {
                let mut ir = lower_program(&program);
                fold_constants(&mut ir);
                let chunk = compile(&ir);
                if let Err(errs) = verify(&chunk) {
                    for e in errs {
                        eprintln!("{e}");
                    }
                    process::exit(1);
                }
                let mut vm = Vm::new();
                if let Err(err) = vm.execute(&chunk) {
                    eprintln!("runtime error: {err}");
                    process::exit(1);
                }
                for line in vm.stdout {
                    println!("{line}");
                }
            } else {
                let mut interp = Interpreter::new();
                if let Err(err) = interp.run(&program) {
                    eprintln!("runtime error: {err}");
                    process::exit(1);
                }
                for line in interp.stdout {
                    println!("{line}");
                }
            }
        }
        other => {
            eprintln!("error: unknown command '{other}'");
            process::exit(1);
        }
    }
}
