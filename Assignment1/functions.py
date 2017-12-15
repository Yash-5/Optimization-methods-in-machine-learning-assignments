from math import log, e

def quadratic(x, y, grad=False):
    if grad:
        grad_x = 2.25 * x + 0.5 * y+ 2
        grad_y = 0.5 * x + 1.5 * y + 2
        grad_norm = (grad_x ** 2 + grad_y ** 2) ** 0.5
        return grad_x / grad_norm, grad_y / grad_norm
    else:
        return 1.125 * x**2 + 0.5 * x * y + 0.75 * y**2 + 2 * x + 2 * y

def ridge_reg(x, y, grad=False):
    if grad:
        grad_x = x + (10 * e**(0.2 * x)) / (1 + e**(0.2 * x))
        grad_y = y - (25 * e**(-0.5 * y)) / (1 + e**(-0.5 * y))
        grad_norm = (grad_x ** 2 + grad_y ** 2) ** 0.5
        return grad_x / grad_norm, grad_y / grad_norm
    else:
        return 0.5 * (x**2 + y**2) + 50 * log(1 + e**(-0.5 * y)) + 50 * log(1 + e**(0.2 * x))

def himmelblaus(x, y, grad=False):
    if grad:
        grad_x = 0.4 * x * (x**2 + y - 11) + 0.2 * (x + y**2 - 7)
        grad_y = 0.2 * (x**2 + y - 11) + 0.4 * y * (x + y**2 - 7)
        grad_norm = (grad_x ** 2 + grad_y ** 2) ** 0.5
        return grad_x / grad_norm, grad_y / grad_norm
    else:
        return 0.1 * (x**2 + y - 11)**2 + 0.1 * (x + y**2 - 7)**2

def rosenbrock(x, y, grad=False):
    if grad:
        grad_x = -0.004 * (1 - x) - 0.8 * x * (y - x**2)
        grad_y = 0.4 * (y - x**2)
        grad_norm = (grad_x ** 2 + grad_y ** 2) ** 0.5
        return grad_x / grad_norm, grad_y / grad_norm
    else:
        return 0.002 * (1 - x)**2 + 0.2 * (y - x**2)**2
