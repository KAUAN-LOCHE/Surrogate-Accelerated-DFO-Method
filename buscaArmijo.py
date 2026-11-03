import numpy as np

ALPHA = 1
CUT = 0.5
TOL = 1e-5
P = -1
X = np.array([1])
C=0.1

def f(x):
    return np.sum(x**4)

def gf(x):
    return 4*x**3

while True:
    gfx = gf(X)
    if np.sum(gfx) < TOL:
        print(f'X: {X}')
        break
    direction = P*gfx
    xnew = X + ALPHA*direction
    condition = f(X)+ALPHA*np.sum(gfx*direction)*C
    if f(xnew) <= condition:
        X = xnew
    else:
        ALPHA *= 0.5
    print(f'ALPHA: {ALPHA}, gfx: {np.sum(gfx)}, xnew: {xnew}, direction: {direction} condition: {condition}')

