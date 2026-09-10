from yara.optimizer import GradientDescent, SGDMomentum, Adam

def rosebrock(params):
    x, y = params
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]

def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    history = [params[:]]
    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history

start = [-1.0, 1.0]

gd_history = optimize(GradientDescent(learning_rate=0.0005), rosebrock, rosenbrock_grad, start)
sgd_history = optimize(SGDMomentum(learning_rate=0.0001, momentum=0.9), rosebrock, rosenbrock_grad, start)
adam_history = optimize(Adam(learning_rate=0.0005), rosebrock, rosenbrock_grad, start)

for name, history in [("gd", gd_history), ("sgd+m", sgd_history), ("adam", adam_history)]:
    final = history[-1]
    loss = rosebrock(final)
    print(f"{name:6s} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")
