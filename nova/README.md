# Nova Programming Language (v0.1.0)

Nova is a statically typed, Python-inspired, hybrid-execution programming language designed for general-purpose systems, backend automation, and native AI/tensor computation.

## Workspace Crates

| Crate | Role |
|---|---|
| `nova-lexer` | Tokenizer with line, column, span tracking and diagnostics |
| `nova-parser` | Recursive-descent & Pratt expression parser producing the Nova AST |
| `nova-semantic` | Lexical scopes, symbol tables, static type checker & inference |
| `nova-ir` | Backend-independent basic-block IR with constant folding & DCE passes |
| `nova-bytecode` | Compact bytecode compiler, validator, and disassembler |
| `nova-runtime` | Value representation and mark-and-sweep heap garbage collector |
| `nova-vm` | Stack-based Nova Virtual Machine executing compiled bytecode |
| `nova-ai` | Decoupled CPU Tensor engine (`@` matmul, broadcasting) & reverse-mode Autograd |
| `nova-cli` | Executable `nova` CLI (`run`, `check`, `ir`, `bytecode`, `version`) |

## Building & Testing with Rust

```bash
cd nova
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p nova-cli -- run examples/hello.nova
cargo run -p nova-cli -- check examples/hello.nova
```
