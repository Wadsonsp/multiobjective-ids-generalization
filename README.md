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
