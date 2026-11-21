# Nova Language Specification (v0.1.0)

## 1. Bindings & Mutability
- `let x = 10` — immutable binding (type inferred or explicit `let x: Int = 10`).
- `mut count = 0` — mutable binding allowing reassignment (`count = count + 1`, `count += 1`).
- `const MAX_TOKENS = 4096` — compile-time constant.

## 2. Primitive & Composite Types
`Int`, `Float`, `Bool`, `String`, `Char`, `Array<T>`, `Map<K, V>`, `Tensor<T>`, `Option<T>`, `Result<T, E>`, `None`, and user-defined `struct`.

## 3. Control Flow & Functions
- `fn name(a: Int, b: Int) -> Int { return a + b }`
- `if cond { ... } else if cond2 { ... } else { ... }`
- `for i in 0..10 { ... }` and `for item in items { ... }`
- `while cond { ... }` with `break` and `continue`.

## 4. Error Handling
Functions return `Result<T, E>` using `Ok(value)` and `Error(message)` constructors, inspected with `is_ok`, `is_error`, and `unwrap`.

## 5. Native AI & Tensor Extensions
- `use nova.ai` / `use nova.nn`
- `let a = tensor([[1.0, 2.0], [3.0, 4.0]])`
- Matrix multiplication operator `@`: `let c = a @ b`
- Activations & training: `relu`, `sigmoid`, `softmax`, `mse_loss`, `autograd_demo`, `train_xor`.
