import numpy as np
from random import uniform
from numpy.linalg import norm

def out_of_domain(curr_pt, x_range, y_range):
        if curr_pt[0] < x_range[0] or curr_pt[0] > x_range[1]\
            or curr_pt[1] < y_range[0] or curr_pt[1] > y_range[1]:
                return True
        return False

def grad_descent(x_range, y_range, func, stepsize=0.01, adaptive=False, start_pt=(0, 0), iterations=1000):
    curr_pt = start_pt

    curr_iter = 1
    x_seq = [curr_pt[0]]
    y_seq = [curr_pt[1]]
    while curr_iter <= iterations:
        if adaptive:
            stepsize = 1 / curr_iter
        delta = -stepsize * np.array(func(*curr_pt, grad=True))
        curr_pt = (curr_pt[0] + delta[0], curr_pt[1] + delta[1])
        if out_of_domain(curr_pt, x_range, y_range):
            return x_seq, y_seq
        x_seq.append(curr_pt[0])
        y_seq.append(curr_pt[1])
        curr_iter += 1
    return x_seq, y_seq

def grad_descent_backtrack(x_range, y_range, func, start_pt=(0, 0), init_step_size=1, \
                           backtrack_ratio=0.5, slope_ratio=0.5, max_backtracks=10, iterations=1000):
    curr_pt = start_pt

    curr_iter = 1
    x_seq = [curr_pt[0]]
    y_seq = [curr_pt[1]]
    while curr_iter <= iterations:
        curr_backtracks = 1
        stepsize = init_step_size
        delta = np.array(func(*curr_pt, grad=True))
        while curr_backtracks <= max_backtracks:
            if func(curr_pt[0] - stepsize * delta[0], curr_pt[1] - stepsize * delta[1]) <=\
                    func(*curr_pt) - stepsize * slope_ratio * norm(delta)\
                    and not out_of_domain((curr_pt[0] - stepsize * delta[0], curr_pt[1] - stepsize * delta[1]), x_range, y_range):
                break
            stepsize *= backtrack_ratio
            curr_backtracks += 1
        curr_pt = (curr_pt[0] - stepsize * delta[0], curr_pt[1] - stepsize * delta[1])
        if out_of_domain(curr_pt, x_range, y_range):
            return x_seq, y_seq
        x_seq.append(curr_pt[0])
        y_seq.append(curr_pt[1])
        curr_iter += 1
    return x_seq, y_seq
