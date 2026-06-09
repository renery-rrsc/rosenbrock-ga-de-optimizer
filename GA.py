import time # Módulo adicionado para monitoramento de performance
import numpy as np
import matplotlib.pyplot as plt
from calculaFX import calculaFX
from cruzamento import cruzamento, cruzamento_rotacionado
from mutacao import mutacao
from selecao import selecao

# ======> REGISTRO DE ALTERAÇÕES <======
# Data: 14-05-2026
# 1. Integrar o algoritmo de recozimento simulado visto em aula para melhorar a exploração do espaço de busca e a convergência global
# 2. Atualização do script GA para chamar a nova função que transforma a base da população
# 3. Adição de monitoramento de tempo de execução com time.perf_counter()

xmin = -1.5
xmax = 1.5

# tamPOP * numGER <= 10000
tamPOP = 100
numGER = int(10000 / tamPOP)
numVAR = 2
numEXEC = 30
minFX = np.zeros(numEXEC)

# Array para armazenar o tempo gasto em cada uma das execuções
exec_times = np.zeros(numEXEC) 

# Parâmetros para o recozimento simulado
T0 = 10000  # Temperatura inicial
a = 0.95  # Fator de resfriamento

plt.ion()
fig, ax = plt.subplots()

if numVAR == 2:
    scatter, = ax.plot([], [], 'ro')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([xmin, xmax])
    ax.grid(True)

# Marca o tempo do início de todo o processo
tempo_total_inicio = time.perf_counter()

for t in range(numEXEC):
    # Inicia o cronômetro para esta execução específica
    inicio_execucao = time.perf_counter() 
    
    POP = xmin + np.random.rand(tamPOP, numVAR) * (xmax - xmin) # gera pop inicial
    FX, GX = calculaFX(POP) # avalia pop inicial
    
    # reinicia temperatura
    T = T0

    for g in range(2, numGER + 1):
        # cruzamento -> avaliação -> mutação -> reavaliação da população mutante -> seleção -> elitismo
        POPnovo = cruzamento_rotacionado(POP, FX, xmin, xmax)
        FXnovo, GXnovo = calculaFX(POPnovo)
        POPnovo, FXnovo, GXnovo = mutacao(POPnovo, FX, GX, xmin, xmax, T)
        _, GXnovo = calculaFX(POPnovo)
        
        # elitismo
        POP = np.vstack((POP, POPnovo))
        FX = np.concatenate((FX, FXnovo)) 
        GX = np.concatenate((GX, GXnovo))
        POP, FX, GX = selecao(POP, FX, GX, tamPOP)
        
        # resfriamento
        T *= a

        if numVAR == 2:
            x1 = POP[:, 0]
            x2 = POP[:, 1]
            scatter.set_data(x1, x2)
            ax.set_title(f'Execução: {t+1}/{numEXEC} | Generation: {g} | Temp: {T:.2f}')
            fig.canvas.draw()
            fig.canvas.flush_events()
        
    minFX[t] = np.min(FX)
    
    # Para o cronômetro da execução e salva o tempo (em segundos)
    exec_times[t] = time.perf_counter() - inicio_execucao 

# Finaliza a plotagem dinâmica
plt.ioff()
plt.show()

# Para o cronômetro total
tempo_total_fim = time.perf_counter() - tempo_total_inicio

# ================= RESULTADOS =================
print("\n" + "="*40)
print("             MÉTRICAS DA FUNÇÃO")
print("="*40)
print("Mínimo FX por execução:\n", minFX)
print(f"Média dos Mínimos: {np.mean(minFX)}")
print(f"Melhor solução global (Variáveis): {POP[np.argmin(FX)]}")
print(f"Melhor FX global: {np.min(minFX)}")

print("\n" + "="*40)
print("             MÉTRICAS DE TEMPO")
print("="*40)
print("Tempo de cada execução (em segundos):\n", np.round(exec_times, 4))
print(f"Tempo médio por execução: {np.mean(exec_times):.4f} s")
print(f"Tempo da execução mais rápida: {np.min(exec_times):.4f} s")
print(f"Tempo total do script: {tempo_total_fim:.4f} s")