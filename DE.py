import numpy as np
import matplotlib.pyplot as plt
from calculaFX import calculaFX

# ======> REGISTRO DE ALTERAÇÕES <======
#
# Data: 14-05-2026
# 1. Utilizar o melhor indivíduo da geração para guiar a mutação dos demais para acelerar a convergência

xmin = -1.5
xmax = 1.5

tamPOP = 50
numGER = int(10000 / tamPOP)
numVAR = 2
numEXEC = 10
minFX = np.zeros(numEXEC)

plt.ion()
fig, ax = plt.subplots()
scatter, = ax.plot([], [], 'ro')
ax.set_xlim([xmin, xmax])
ax.set_ylim([xmin, xmax])
ax.grid(True)

for t in range(numEXEC):
    POP = xmin + np.random.rand(tamPOP, numVAR) * (xmax - xmin)
    FX, GX = calculaFX(POP)
    
    for g in range(2, numGER + 1):
        melhor_individuo = np.argmin(FX)
        X_best = POP[melhor_individuo, :]
        
        for i in range(tamPOP):
            j = np.random.choice(numVAR)
            C = 0.9 + 0.2 * np.random.rand()
            r = np.random.choice(tamPOP, 3, replace=False)
            Pnovo = X_best + C * (POP[r[0], :] - POP[r[1], :]) # melhor_individuo + C * (individuo_aleatorio1 - individuo_aleatorio2)
            
            p = np.random.rand(numVAR) # probabilidade de mutação para cada variável
            mask = p <= 0.5
            mask[j] = True

            Pnovo = np.where(mask, Pnovo, POP[i, :]) # mantém as variáveis que não foram selecionadas para mutação iguais ao indivíduo atual
            Pnovo = np.clip(Pnovo, xmin, xmax) # garante que o novo indivíduo esteja dentro dos limites

            FXnovo, GXnovo = calculaFX(np.array([Pnovo]))
            
            if FXnovo[0] <= FX[i]:
                POP[i, :] = Pnovo
                FX[i] = FXnovo[0]
                GX[i] = GXnovo[0]
                
        x1 = POP[:, 0]
        x2 = POP[:, 1]
        scatter.set_data(x1, x2)
        ax.set_title(f'Generation: {g}')
        fig.canvas.draw()
        fig.canvas.flush_events()

        # for i in range(tamPOP):
        #     j = np.random.choice(numVAR)
        #     C = 0.9 + 0.2 * np.random.rand()
        #     r = np.random.choice(tamPOP, 3, replace=False)
        #     Pnovo = POP[r[0], :] + C * (POP[r[2], :] - POP[r[1], :])
            
        #     for d in range(numVAR):
        #         if np.random.rand() <= 0.5 and d != j:
        #             Pnovo[d] = POP[i, d]
            
        #     FXnovo, GXnovo = calculaFX(np.array([Pnovo]))
            
        #     if FXnovo[0] <= FX[i]:
        #         POP[i, :] = Pnovo
        #         FX[i] = FXnovo[0]
        #         GX[i] = GXnovo[0]
                
        # ax.clear()
        # ax.plot(POP[:, 0], POP[:, 1], 'ro')
        # ax.set_xlim([xmin, xmax])
        # ax.set_ylim([xmin, xmax])
        # ax.set_xlabel(str(g))
        # ax.grid(True)
        # plt.pause(0.01)
        
    minFX[t] = np.min(FX)

plt.ioff()
plt.show()
print("Minimum FX per execution:", minFX)
