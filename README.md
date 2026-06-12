# multiobjective-ids-generalization

Repositório experimental associado à dissertação sobre Sistemas de Detecção de Intrusão baseados em Aprendizado de Máquina, com foco em generalização cross-dataset e otimização multiobjetivo.

## Descrição

Este projeto implementa um pipeline experimental para avaliação de modelos de detecção de intrusão baseados em aprendizado de máquina, utilizando datasets NetFlow. A abordagem considera a preparação dos dados, higienização, balanceamento, seleção de características e avaliação de desempenho em múltiplos conjuntos de dados.

A estrutura experimental busca investigar a capacidade de generalização cross-dataset dos modelos, analisando o desempenho em diferentes domínios de dados. Para isso, o código utiliza redes neurais do tipo MLP e otimização multiobjetivo com NSGA-II, considerando métricas associadas ao desempenho F1 em diferentes datasets.

## Objetivo

Avaliar estratégias de detecção de intrusão baseadas em aprendizado de máquina que combinem:

- generalização cross-dataset;
- seleção de características;
- otimização multiobjetivo;
- avaliação comparativa entre datasets NetFlow;
- redução de custo computacional por meio da escolha otimizada de atributos e arquitetura do modelo.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow/Keras
- Pymoo
- FastAI
- Pickle

## Estrutura experimental

O código realiza:

1. carregamento dos datasets NetFlow;
2. remoção de atributos não utilizados;
3. higienização dos dados;
4. balanceamento das classes;
5. redução de uso de memória;
6. padronização dos dados;
7. treinamento de modelo MLP;
8. otimização multiobjetivo com NSGA-II;
9. avaliação em múltiplos datasets;
10. salvamento de checkpoints e resultados.

## Tema da pesquisa

Sistemas de Detecção de Intrusão Baseados em Aprendizado de Máquina com Generalização Cross-Dataset e Otimização Multiobjetivo.

## Estrutura do projeto (Algoritmos 1 e 2 do Capítulo 4)

```
├── Datasets/            # cache local (vazio no git); dados oficiais no Drive em mestrado/Datasets
├── Modulos/             # pacote reutilizável
│   ├── drive_loader.py        # localização/validação dos datasets (Colab/local)
│   ├── preprocessamento.py    # Seção 4.1 (limpeza, vazamento, Min-Max, estratificação)
│   ├── classificadores.py     # fábrica dos 8 classificadores (Seção 5.2)
│   ├── avaliacao.py           # Algoritmo 2 - AvaliarFitness(x, D, h)
│   ├── otimizacao.py          # Algoritmo 1 - NSGA-II (pymoo)
│   └── selecao_rfe.py         # baseline RFE (Capítulo 5, monobjetivo)
├── src/
│   ├── config.yaml                # parâmetros centrais dos experimentos
│   ├── setup_datasets_drive.py    # download oficial UQ -> MyDrive/mestrado/Datasets (Colab)
│   ├── algoritmo1_otimizacao.py   # executável do Algoritmo 1
│   └── algoritmo2_avaliacao.py    # executável do Algoritmo 2 (baseline ou solução do P*)
├── Resultados/          # pareto/, metricas/, logs/
└── tests/               # suíte pytest (cobertura mínima 85%, atual ~94%)
```

## Como executar

```bash
pip install -r requirements.txt

# 1) Uma única vez, no Colab: baixar as bases para o Drive
python src/setup_datasets_drive.py

# 2) Otimização multiobjetivo (Algoritmo 1)
python src/algoritmo1_otimizacao.py

# 3) Avaliação final de uma solução do P* com dados completos (Algoritmo 2)
python src/algoritmo2_avaliacao.py --pareto Resultados/pareto/<arquivo>.json --solucao 0 --amostra 0

# Baseline RFE do Capítulo 5 (Tabela 3: --todos avalia os 8 classificadores)
python src/baseline_rfe.py --todos

# Testes com cobertura
pytest
```

O classificador h, os parâmetros do NSGA-II (N_pop, N_gen, pc, pm) e a
subamostragem da fase de busca são configurados em `src/config.yaml`.
