# Rosenbrock Optimization Experiments

Este diretório contém implementações e experimentos com algoritmos de otimização aplicados à função de Rosenbrock.

## Visão geral

O objetivo principal deste projeto é explorar métodos heurísticos para minimizar a função de Rosenbrock em um domínio bidimensional limitado a `[-1.5, 1.5]` para cada variável.

A pasta inclui versões de:
- Algoritmo genético (GA)
- Evolução diferencial (DE)
- Operadores de cruzamento, mutação e seleção
- Avaliação da função objetivo com restrição de região viável

## Conteúdo principal

- `GA.py` — fluxo principal de um algoritmo genético híbrido, incluindo cruzamento rotacionado, mutação com recozimento simulado e seleção elitista/por torneio.
- `DE.py` — experimento com evolução diferencial para buscar soluções de mínima energia.
- `rosenbrock.py` — implementação da função objetivo de Rosenbrock e da restrição de viabilidade.
- `calculaFX.py` — wrapper que avalia a população usando `rosenbrock.py`.
- `cruzamento.py` — operadores de cruzamento padrão e rotacionado.
- `mutacao.py` — operadores de mutação incluindo versão com recozimento simulado.
- `selecao.py` — estratégias de seleção: elitismo, torneio e roleta.
- `explore.ipynb` — notebook exploratório para testes e visualizações.
- `MATLab/` — arquivos de suporte em MATLAB relacionados ao problema.
- `requirements.txt` — dependências Python do projeto.
- `rosenbrock_renery.zip` — arquivo de backup/arquivamento do projeto.
- `stats_weapons.py` — utilitário estatístico adicional presente na pasta.

## Requisitos

Recomenda-se criar um ambiente virtual e instalar as dependências locais:

```powershell
cd c:\Projetos\rosenbrock
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Dependências principais:
- numpy
- matplotlib
- pandas
- scipy
- scikit-learn
- statsmodels

## Como executar

Para rodar o algoritmo genético principal:

```powershell
python GA.py
```

Para testar a evolução diferencial:

```powershell
python DE.py
```

Para abrir a análise interativa:

- Abra `explore.ipynb` em Jupyter ou VS Code.

## Observações

- `GA.py` usa gráficos interativos para exibir a evolução dos indivíduos em cada geração.
- `mutacao_nova` aplica ruído gaussiano e aceita soluções subótimas com base no critério do recozimento simulado.
- `cruzamento_rotacionado` tenta alinhar o crossover ao vale da função usando a covariância dos melhores indivíduos.

## Autoria

Conteúdo desenvolvido por Renery Carvalho.
