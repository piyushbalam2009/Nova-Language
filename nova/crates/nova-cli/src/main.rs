//! Nova CLI executable:
//!   nova run <file.nova> [--engine=vm|interpreter]
//!   nova check <file.nova>
//!   nova verify <file.nova>
//!   nova llvm <file.nova> [-o out.ll]
//!   nova build <file.nova> [--debug|--release] [--emit-llvm] [-o out]
//!   nova version

use nova_bytecode::{compile, verify};
use nova_interpreter::Interpreter;
use nova_ir::{fold_constants, lower_program};
use nova_lexer::tokenize;
use nova_llvm::{build_native_executable, lower_to_llvm, BuildProfile};
use nova_parser::Parser;
use nova_semantic::check_program;
use nova_vm::Vm;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Engine {
    Vm,
    Interpreter,
}

struct ParsedArgs {
    file: String,
    engine: Engine,
    profile: BuildProfile,
    output: Option<PathBuf>,
    emit_llvm: bool,
}

fn print_usage() {
    eprintln!(
        "Nova Programming Language CLI v{VERSION}\n\n\
Usage:\n  \
nova run <file.nova> [--engine=vm|interpreter]\n  \
nova check <file.nova>\n  \
nova verify <file.nova>\n  \
nova llvm <file.nova> [-o out.ll]\n  \
nova build <file.nova> [--debug|--release] [--emit-llvm] [-o out]\n  \
nova version"
    );
}

fn parse_subcommand_args(raw: &[String]) -> Result<ParsedArgs, String> {
    let mut file: Option<String> = None;
    let mut engine = Engine::Vm;
    let mut profile = BuildProfile::Debug;
    let mut output: Option<PathBuf> = None;
    let mut emit_llvm = false;

    let mut i = 0;
    while i < raw.len() {
        let arg = &raw[i];
        if arg == "-o" {
            i += 1;
            let out_val = raw
                .get(i)
                .ok_or_else(|| "expected output path after '-o'".to_string())?;
            output = Some(PathBuf::from(out_val));
        } else if let Some(val) = arg.strip_prefix("--engine=") {
            engine = match val {
                "vm" => Engine::Vm,
                "interpreter" => Engine::Interpreter,
                other => {
                    return Err(format!(
                        "invalid --engine value '{other}' (expected 'vm' or 'interpreter')"
                    ));
                }
            };
        } else if arg == "--release" {
            profile = BuildProfile::Release;
        } else if arg == "--debug" {
            profile = BuildProfile::Debug;
        } else if arg == "--emit-llvm" {
            emit_llvm = true;
        } else if arg.starts_with('-') {
            return Err(format!("unknown flag '{arg}'"));
        } else if file.is_none() {
            file = Some(arg.clone());
        } else {
            return Err(format!("unexpected extra positional argument '{arg}'"));
        }
        i += 1;
    }

    let file = file.ok_or_else(|| "expected a .nova source file".to_string())?;
    Ok(ParsedArgs {
        file,
        engine,
        profile,
        output,
        emit_llvm,
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "version" | "--version" | "-V" => {
            println!("nova {VERSION}");
        }
        "--help" | "-h" | "help" => {
            print_usage();
        }
        cmd @ ("check" | "verify" | "llvm" | "build" | "run") => {
            let parsed = match parse_subcommand_args(&args[2..]) {
                Ok(p) => p,
                Err(msg) => {
                    eprintln!("error: {msg}");
                    process::exit(1);
                }
            };

            let source = match fs::read_to_string(&parsed.file) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("error: failed to read '{}': {err}", parsed.file);
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
                println!("OK: {}", parsed.file);
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
                if let Some(out_path) = parsed.output {
                    if let Err(err) = fs::write(&out_path, &lowered.llvm_ir) {
                        eprintln!("error: failed to write {}: {err}", out_path.display());
                        process::exit(1);
                    }
                    println!(
                        "Emitted verified LLVM IR: {} ({}, {} fns, {} blocks)",
                        out_path.display(),
                        lowered.target_triple,
                        lowered.validation.functions_checked,
                        lowered.validation.blocks_checked
                    );
                } else {
                    println!("{}", lowered.llvm_ir);
                }
                return;
            }

            if cmd == "build" {
                let mut ir = lower_program(&program);
                fold_constants(&mut ir);
                let stem = PathBuf::from(&parsed.file)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("out")
                    .to_string();
                let out_path = parsed.output.unwrap_or_else(|| PathBuf::from(&stem));
                let emit_ll = if parsed.emit_llvm {
                    Some(PathBuf::from(format!("{}.ll", out_path.display())))
                } else {
                    None
                };

                match build_native_executable(&ir, &out_path, parsed.profile, emit_ll.as_deref()) {
                    Ok(artifact) => {
                        let profile_label = match artifact.profile {
                            BuildProfile::Debug => "debug",
                            BuildProfile::Release => "release",
                        };
                        let strip_label = if artifact.stripped { ", stripped" } else { "" };
                        println!(
                            "Built native executable: {} ({profile_label}, {}{strip_label}, {} bytes)",
                            artifact.output_path.display(),
                            artifact.target_triple,
                            artifact.binary_bytes
                        );
                        if let Some(ll) = artifact.llvm_ir_path {
                            println!("Emitted LLVM IR: {}", ll.display());
                        }
                    }
                    Err(err) => {
                        eprintln!("error: native build failed: {err}");
                        process::exit(1);
                    }
                }
                return;
            }

            if cmd == "verify" {
                let mut ir = lower_program(&program);
                fold_constants(&mut ir);
                let chunk = compile(&ir);
                match verify(&chunk) {
                    Ok(report) => {
                        println!(
                            "VERIFIED: {} ({} functions, {} instructions, max stack {})",
                            parsed.file,
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

            match parsed.engine {
                Engine::Vm => {
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
                }
                Engine::Interpreter => {
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
        }
        other => {
            eprintln!("error: unknown command '{other}'");
            print_usage();
            process::exit(1);
        }
    }
}
