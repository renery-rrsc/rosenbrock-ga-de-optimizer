import numpy as np
# ======> REGISTRO DE ALTERAÇÕES <======
#
# Data: 15-05-2026
# 1. Implementação de um cruzamento com transformação da base do sistema para alinhar as coordenadas com o fundo do vale com base na matriz de
# covariância dos indivíduos da população, visando acelerar a convergência para o mínimo global.

def cruzamento_rotacionado(POP, FX, xmin, xmax) -> np.ndarray:
    POP = np.array(POP)
    tamPOP, numVAR = POP.shape

    # rotação
    # filtrando os melhores indivíduos para estimar a direção do vale
    num_best = max(2, tamPOP // 4)  # seleciona os melhores 25% ou pelo menos 2 indivíduos
    idx_best = np.argsort(FX)[:num_best]
    POP_best = POP[idx_best, :]

    cov_matrix = np.cov(POP_best, rowvar=False)

    if np.all(cov_matrix == 0):
        cov_matrix = np.eye(numVAR)

    _, evec = np.linalg.eigh(cov_matrix)

    POP_rot = np.dot(POP - np.mean(POP, axis=0), evec) # rotaciona a população CENTRALIZADA para trasnformação ocorrer na origem

    # cruzamento
    r1 = np.random.choice(tamPOP, size=tamPOP)
    r2 = np.random.choice(tamPOP, size=tamPOP)
    idem = r1 == r2
    while np.any(r1 == r2):
        r2[idem] = np.random.choice(tamPOP, size=np.sum(idem))
        idem = r1 == r2

    POPnovo_rot = POP_rot[r1, :] + (2 * np.random.rand(tamPOP, numVAR) - 0.5) * (POP_rot[r2, :] - POP_rot[r1, :])
    POPnovo = np.dot(POPnovo_rot, evec.T) + np.mean(POP, axis=0) # rotaciona de volta para o sistema original
    POPnovo = np.maximum(POPnovo, xmin)
    POPnovo = np.minimum(POPnovo, xmax)

    return POPnovo

def cruzamento(POP, xmin, xmax) -> np.ndarray:
    POP = np.array(POP)
    tamPOP, numVAR = POP.shape
    POPnovo = np.copy(POP)
    
    for i in range(tamPOP):
        r = np.random.choice(tamPOP, 2, replace=False)
        POPnovo[i, :] = POP[r[0], :] + (2 * np.random.rand(numVAR) - 0.5) * (POP[r[1], :] - POP[r[0], :])
        
    POPnovo = np.maximum(POPnovo, xmin)
    POPnovo = np.minimum(POPnovo, xmax)
    
    return POPnovo
