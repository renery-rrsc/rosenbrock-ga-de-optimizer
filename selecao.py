import numpy as np

def elitismo(POP, FX, GX, tamPOP) -> tuple:
    ind = np.argsort(FX)
    ind = ind[:tamPOP]
    
    return POP[ind, :], FX[ind], GX[ind]

def torneio(POP, FX, GX, tamPOP) -> tuple:
    ind = np.zeros(tamPOP, dtype=int)
    num_pop = POP.shape[0]
    
    for i in range(tamPOP):
        r = np.random.choice(num_pop, 2, replace=False)
        if FX[r[0]] <= FX[r[1]]:
            ind[i] = r[0]
        else:
            ind[i] = r[1]
            
    return POP[ind, :], FX[ind], GX[ind]

def roleta(POP, FX, GX, tamPOP) -> tuple:
    ind = np.zeros(tamPOP, dtype=int)
    
    # Transforma os valores por se tratar de um problema de minimização
    FXnorm = 1.0 / (FX + 1e-10) 
    FXnorm = FXnorm / np.sum(FXnorm)
    
    for i in range(tamPOP):
        r = np.random.rand()
        soma = 0.0
        cont = 0
        while r >= soma and cont < len(FXnorm):
            soma += FXnorm[cont]
            cont += 1
        ind[i] = cont - 1
        
    return POP[ind, :], FX[ind], GX[ind]

def selecao(POP, FX, GX, tamPOP) -> tuple:
    tamELITE = 3
    
    POP = np.array(POP)
    FX = np.array(FX)
    GX = np.array(GX)
    
    POPelitismo, FXelitismo, GXelitismo = elitismo(POP, FX, GX, tamELITE)
    POPtorneio, FXtorneio, GXtorneio = torneio(POP, FX, GX, tamPOP - tamELITE)
    
    POP_new = np.vstack((POPelitismo, POPtorneio))
    FX_new = np.concatenate((FXelitismo, FXtorneio))
    GX_new = np.concatenate((GXelitismo, GXtorneio))
    
    return POP_new, FX_new, GX_new
