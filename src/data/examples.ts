export interface NovaExample {
  id: string;
  filename: string;
  title: string;
  category: string;
  description: string;
  source: string;
}

export const NOVA_EXAMPLES: NovaExample[] = [
  {
    id: 'hello',
    filename: 'hello.nova',
    title: 'Hello World',
    category: 'Nova 0.1',
    description: 'First execution target: immutable string binding, concatenation, and print.',
    source: `fn main() {
    let name = "Nova"
    print("Hello, " + name)
}
`,
  },
  {
    id: 'functions',
    filename: 'functions.nova',
    title: 'Functions & Loops',
    category: 'Nova 0.2',
    description: 'Typed functions, recursion, compile-time const, mutable bindings, and range loops.',
    source: `const MAX_DEPTH = 10

fn add(a: Int, b: Int) -> Int {
    return a + b
}

fn fib(n: Int) -> Int {
    if n <= 1 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}

fn main() {
    let sum = add(10, 20)
    print("add(10, 20) = " + str(sum))

    mut total = 0
    for i in 0..MAX_DEPTH {
        total += fib(i)
    }
    print("Sum of first 10 Fibonacci numbers: " + str(total))
}
`,
  },
  {
    id: 'structs_results',
    filename: 'structs_results.nova',
    title: 'Structs & Result<T, E>',
    category: 'Nova 0.2',
    description: 'User-defined structs, field access, and explicit Ok/Error result handling.',
    source: `struct User {
    name: String
    age: Int
}

fn divide(a: Float, b: Float) -> Result<Float, String> {
    if b == 0.0 {
        return Error("Division by zero")
    }
    return Ok(a / b)
}

fn main() {
    let user = User {
        name: "Alex",
        age: 22
    }
    print("User: " + user.name + ", age " + str(user.age))

    let ok_res = divide(42.0, 6.0)
    let err_res = divide(10.0, 0.0)

    print("divide(42.0, 6.0) -> " + str(ok_res))
    print("divide(10.0, 0.0) -> " + str(err_res))

    if is_ok(ok_res) {
        print("Unwrapped value: " + str(unwrap(ok_res)))
    }
}
`,
  },
  {
    id: 'ir_opt',
    filename: 'ir_optimizer.nova',
    title: 'IR Optimization Demo',
    category: 'Nova 0.3',
    description: 'Demonstrates constant folding, constant propagation, and dead-code elimination.',
    source: `fn compute(scale: Int) -> Int {
    let dead_value = 999 * 42
    let base = 2 + 3 * 4
    let offset = base + 6
    return offset * scale
}

fn main() {
    let result = compute(5)
    print("Computed result: " + str(result))
}
`,
  },
  {
    id: 'tensors',
    filename: 'tensors.nova',
    title: 'Tensors & Autograd',
    category: 'Nova AI',
    description: 'Matrix multiplication (@), ReLU, Softmax, and reverse-mode automatic differentiation.',
    source: `use nova.ai

fn main() {
    let a = tensor([[1.0, 2.0], [3.0, 4.0]])
    let b = tensor([[5.0, 6.0], [7.0, 8.0]])

    let product = a @ b
    let activated = relu(product - 30.0)
    let probs = softmax(tensor([2.0, 1.0, 0.1]))

    print("A @ B = " + str(product))
    print("Shape: " + str(product.shape()))
    print("ReLU(A @ B - 30) = " + str(activated))
    print("Softmax([2.0, 1.0, 0.1]) = " + str(probs))

    let grads = autograd_demo(3.0, 2.0)
    print("Autograd y=relu(x*w+1.5) at x=3,w=2: " + str(grads))
}
`,
  },
  {
    id: 'neural_network',
    filename: 'neural_network.nova',
    title: 'Neural Network Training',
    category: 'Nova AI',
    description: 'Trains a real 2-layer Dense neural network with backpropagation on the XOR dataset.',
    source: `use nova.nn

fn main() {
    print("Training 2-layer Dense neural network on XOR dataset...")
    let report = train_xor(800, 0.8)
    print("Training report: " + str(report))
}
`,
  },
  {
    id: 'ai_engine',
    filename: 'ai_engine.nova',
    title: 'AI Engine & Adam Training',
    category: 'Nova AI',
    description: 'Full CPU tensor ops, composable reverse-mode tensor autograd, Sequential layers, and Adam training.',
    source: `use nova.ai
use nova.nn

fn main() {
    print("=== Nova AI Engine: Tensors, Autograd & Training ===")

    // 1. CPU Tensor Operations (matmul, transpose, broadcasting, reductions)
    let x_mat = tensor([[1.0, 2.0], [3.0, 4.0]])
    let bias = tensor([[0.5, -0.5]])
    let shifted = x_mat + bias
    print("Broadcast X + bias = " + str(shifted))
    print("Transpose(X) = " + str(x_mat.transpose()))
    print("dot([1,2,3], [4,5,6]) = " + str(dot(tensor([1.0, 2.0, 3.0]), tensor([4.0, 5.0, 6.0]))))
    print("argmax(X) = " + str(x_mat.argmax()))

    // 2. Composable Reverse-Mode Tensor Autograd
    let vx = grad_var(tensor([[1.0, -2.0]]))
    let vw = grad_var(tensor([[0.5, -1.0], [1.5, 0.5]]))
    let vb = grad_var(tensor([[0.25, -0.25]]))
    let target = tensor([[1.0, 0.0]])

    let pred = grad_tanh(grad_add(grad_matmul(vx, vw), vb))
    let loss = grad_mse(pred, target)
    backward(loss)

    print("Autograd forward pred = " + str(value_of(pred)))
    print("Autograd MSE loss = " + str(value_of(loss)))
    print("dL/dW = " + str(grad_of(vw)))
    print("dL/db = " + str(grad_of(vb)))

    // 3. Sequential Neural Network + Adam Optimizer on XOR
    let xor_x = tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    let xor_y = tensor([[0.0], [1.0], [1.0], [0.0]])

    let model = model_new()
    model_add_linear(model, 2, 4, 42)
    model_add_activation(model, "tanh")
    model_add_linear(model, 4, 1, 137)
    model_add_activation(model, "sigmoid")

    print("Model: " + str(model) + " (params=" + str(model_param_count(model)) + ")")

    let opt = optimizer_adam(0.05)
    let report = train_model(model, opt, xor_x, xor_y, 350, "mse")
    print("Adam XOR report = " + str(report))
    print("Trained forward(xor_x) = " + str(model_forward(model, xor_x)))
}
`,
  },
  {
    id: 'type_errors',
    filename: 'type_error_demo.nova',
    title: 'Type & Mutability Errors',
    category: 'Diagnostics',
    description: 'Verifies static type checking, immutability enforcement, and arity validation.',
    source: `fn add(a: Int, b: Int) -> Int {
    return a + b
}

fn main() {
    let age: Int = "Hello"
    let fixed = 10
    fixed = 20
    let bad = add(5)
    print(unknown_symbol)
}
`,
  },
];
