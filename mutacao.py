import numpy as np
from calculaFX import calculaFX

# ======> REGISTRO DE ALTERAÇÕES <======
#
# Data: 14-05-2026
# 1. Integrar o algoritmo de recozimento simulado visto em aula para melhorar a exploração do espaço de busca e a convergência global

def mutacao(POPnovo, FXNovo, GXNovo, xmin, xmax, T) -> tuple:
    tamPOP, numVar = POPnovo.shape

    # Aplica rúido na população
    ruido = 0.2 * (xmax - xmin) * (np.random.rand(tamPOP, numVar) - 0.5)
    POP_vizinha = POPnovo + ruido
    # Garantir que a população vizinha esteja dentro dos limites
    POP_vizinha = np.clip(POP_vizinha, xmin, xmax)

    # Avaliar função objetivo para nova população vizinha
    FX_vizinha, GX_vizinha = calculaFX(POP_vizinha)

    dF = FX_vizinha - FXNovo

    # Critério de aceitação baseado na diferença de função objetivo
    res_melhor = (dF <= 0)

    # Recozimento simulado com o decaimento da temperatura
    T = max(T, 1e-8)
    p = np.exp(-dF / T)
    res_inf = (dF > 0) & (np.random.rand(tamPOP) < p)

    mask = res_melhor | res_inf

    # Atualizar mutações dos individuos aceitos
    POPnovo[mask] = POP_vizinha[mask]
    FXNovo[mask] = FX_vizinha[mask]
    GXNovo[mask] = GX_vizinha[mask]

    # POPnovo = np.array(POPnovo)
    # tamPOP, numVAR = POPnovo.shape
    
    # for i in range(tamPOP):
    #     if np.random.rand() <= 0.5:  # Probabilidade de mutação
    #         POPnovo[i, :] = POPnovo[i, :] + 0.5 * (1 * np.random.rand(numVAR) - 0.5) * (xmax - xmin)
            
    # POPnovo = np.maximum(POPnovo, xmin)
    # POPnovo = np.minimum(POPnovo, xmax)
    
    return POPnovo, FXNovo, GXNovo
