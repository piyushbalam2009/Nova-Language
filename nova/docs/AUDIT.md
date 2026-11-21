# Nova Language — Repository Engineering Audit & Stabilization Log

## 1. Repository Architecture Overview
- **Canonical Implementation**: Rust workspace under `nova/` (11 crates: `nova-lexer`, `nova-parser`, `nova-semantic`, `nova-ir`, `nova-bytecode`, `nova-interpreter`, `nova-vm`, `nova-runtime`, `nova-llvm`, `nova-ai`, `nova-cli`) + embedded C99 runtime (`nova/runtime/novart.h`, `nova/runtime/novart.c`).
- **Web Workbench & Bridge**: Express + Vite UI (`server.ts`, `src/components/*`) currently backed by the transitional TypeScript implementation (`src/nova/*`, `bin/nova.ts`), slated for progressive migration to a Rust backend service once the canonical Rust compiler stabilizes in CI.

## 2. Engineering Audit Findings & Stabilization Status

### Fixed in This Stabilization Pass
1. **Cargo Workspace Test Dependencies (`nova-ir`, `nova-interpreter`, `nova-bytecode`)**:
   - `nova-ir/Cargo.toml` and `nova-interpreter/Cargo.toml` were missing `[dev-dependencies] nova-lexer`, which prevented `cargo test --workspace` from compiling unit tests. Fixed.
2. **Lexer (`nova-lexer`)**:
   - Added character literal lexing (`'a'`, escape sequences), block comments (`/* ... */`), hex (`0x`) / binary (`0b`) integers, underscores in fractional floats, and scientific notation (`1.5e-2`).
   - Fixed escape-sequence error span calculation (previously pointed at string opening quote).
3. **Parser (`nova-parser`)**:
   - Fixed an infinite loop in `parse_type` when encountering unexpected tokens inside generic brackets (e.g., `Array<123>`).
   - Added unary operators (`-`, `!`), `else if` chains, `struct` declarations, `break`/`continue`, `Char` literals, compound assignment (`+=`, `-=`, `*=`, `/=`), and cleaner error recovery in `parse_fn`/`parse_use`.
4. **Semantic Analyzer (`nova-semantic`)**:
   - Registered all Nova 0.1 stdlib and AI/ML builtins (preserving user function shadowing).
   - Added duplicate function/struct/parameter checks, call argument type checking, binary/unary operand validation (including `Tensor` operations), definite-return path checking for non-Void functions, and `break`/`continue` loop-depth validation.
5. **IR Lowering (`nova-ir`)**:
   - Fixed silent miscompilation where `BinOp::MatMul` (`@`) fell through to `Inst::Div`.
   - Fixed `Expr::Array` lowering (previously emitted `ConstVal::None`, breaking `tensor([[...]])` in VM/LLVM) by adding `Inst::AllocArray`.
   - Normalized short-circuit `&&`/`||` to `Bool` for exact parity with `nova-interpreter`.
   - Added `Inst::Neg`, `Inst::MatMul`, `Inst::LoadGlobal`/`StoreGlobal`, `__init__` lowering for top-level bindings, `break`/`continue` loop targets, and checked integer constant folding.
6. **Bytecode & Verifier (`nova-bytecode`)**:
   - Added `Op::Neg`, `Op::MatMul`, `Op::AllocArray`, `Op::LoadGlobal`, `Op::StoreGlobal` with verifier stack-effect checks.
   - Implemented binary `.nvbc` wire format serializer/deserializer (`encode_chunk` / `decode_chunk`) with round-trip tests.
7. **Runtime, Interpreter & VM Parity (`nova-runtime`, `nova-interpreter`, `nova-vm`, `nova-ai`)**:
   - Consolidated value equality (`"Nova" == "Nova"` now `true` in both engines), mixed `Int`/`Float` arithmetic, `Tensor` operators, and stdlib/AI dispatch in `nova-runtime`.
   - Fixed `nova-vm` call-frame GC rooting (`frames: Vec<Vec<Value>>` + `globals`) and added adaptive mid-execution mark-and-sweep GC + recursion/step limits to both `nova-interpreter` and `nova-vm`.
   - Replaced `O(N^2)` `Vec::contains` in `nova-ai` `Var::backward` with `HashSet<usize>` (`O(N)`).
8. **LLVM Native Backend & CLI (`nova-llvm`, `nova-cli`, `nova/runtime`)**:
   - Extracted and embedded the canonical `libnovart` C99 runtime (`nova/runtime/novart.h` and `novart.c`).
   - Unified `nova-llvm` C ABI declarations with `libnovart` (`@nv_str`, `@nv_cmp_eq`, `@nv_alloc_array`, `@nv_call_builtin`, etc.).
   - Fixed `nova build` in `nova-cli` so it compiles and links a real standalone native executable via `clang` + `libnovart` (rather than emitting a `.ll` file and claiming it was built), with proper `-o`, `--debug`, `--release`, `--emit-llvm`, and `--engine=` parsing.
9. **Security & Server Hardening (`server.ts`, `src/nova/native-builder.ts`, `package.json`)**:
   - Stripped host environment secrets from spawned native compilation and execution child processes.
   - Added strict source size caps (64 KB), filename/module-name sanitization, concurrent native build throttling, `/api/health`, and `npm test` script.

### Remaining Roadmap Items (Future Milestones)
- Full `for` loops, `match`, struct field access/instantiation, array/map indexing, and `Result<T, E>` in the Rust crates (`nova-parser` -> `nova-semantic` -> `nova-ir` -> `nova-vm` -> `nova-llvm`).
- Module resolver (`nova.toml` + `use` graph) and standard library modules (`std.io`, `std.math`, `std.fs`, `std.time`, `std.collections`).
- Progressive migration of the Express web workbench endpoints to a Rust backend service once CI runs `cargo` + `clang`.
