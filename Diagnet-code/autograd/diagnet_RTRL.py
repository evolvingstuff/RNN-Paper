"""Implements the long-short term memory character model.
This version vectorizes over multiple examples, but each string
has a fixed length."""

from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
from autograd import grad

def main():
    print('RTRL experiment')

    seq = [1.1, -0.3, -0.05]

    r = 0.9
    w = 1.1

    def f_r(r):
        c = 0.0
        for i in seq:
            c = np.abs(c*r + i*w)
        return c

    def f_w(w):
        c = 0.0
        for i in seq:
            c = np.abs(c*r + i*w)
        return c

    def rtrl_f_r(r):
        c = 0.0
        dfdr = 0.0
        for i in seq:
            s = c * r + i * w
            if s >= 0:
                sgn = 1.0
            else:
                sgn = -1.0
            dfdr = sgn*c + r*dfdr
            c = np.abs(s)
        return dfdr

    def rtrl_f_w(w):
        c = 0.0
        dfdw = 0.0
        for i in seq:
            s = c * r + i * w
            if s >= 0:
                sgn = 1.0
            else:
                sgn = -1.0
            dfdw = sgn*i + r*dfdw
            c = np.abs(s)
        return dfdw

    print('----------------------')
    print('f_r')
    grad_f_r = grad(f_r)
    print(str(grad_f_r(1.0)))
    print(str(grad_f_r(1.00001)))
    print(str(grad_f_r(1.1)))

    print('----------------------')
    print(str(rtrl_f_r(1.0)))
    print(str(rtrl_f_r(1.00001)))
    print(str(rtrl_f_r(1.1)))

    print('----------------------')
    print('f_w')
    grad_f_w = grad(f_w)
    print(str(grad_f_w(1.0)))
    print(str(grad_f_w(1.00001)))
    print(str(grad_f_w(1.1)))

    print('----------------------')
    print(str(rtrl_f_w(1.0)))
    print(str(rtrl_f_w(1.00001)))
    print(str(rtrl_f_w(1.1)))

if __name__ == '__main__':
    main()


