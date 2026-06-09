import numpy as np

def rosenbrock(POP) -> tuple:
    POP = np.array(POP)
    if POP.ndim == 1:
        POP = POP.reshape(1, -1)
    
    numVAR = POP.shape[1]
    x1 = POP[:, 0]
    x2 = POP[:, 1]

    if numVAR != 2:
        print('NÚMERO DE VARIÁVEIS INCORRETO')
        return None, None

    FX = (1 - x1)**2 + 100 * (x2 - x1**2)**2 # f(x1, x2) = (1 - x1)^2 + 100 * (x2 - x1^2)^2
    GX = np.maximum(x1**2 + x2**2 - 1, 0) # x^2 + y^2 <= 1 ---> g(x1, x2) = max(x1^2 + x2^2 - 1, 0)
    
    return FX, GX
