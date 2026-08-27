# Seleção multiobjetivo de features para IDS cross-dataset

Projeto experimental da dissertação sobre generalização cross-dataset em
Sistemas de Detecção de Intrusão. Eu uso o NSGA-II como **pré-filtro (Fase 1)**
para encontrar compromissos entre desempenho cross-dataset e quantidade de
features. A escolha e a análise detalhada das soluções acontecem na **Fase 2**.

## Formulação biobjetivo

Cada solução é um vetor real `x` com um gene em `[0, 1]` para cada feature. Eu
mantenho a feature quando `x >= 0.5` e minimizo:

1. `1 - média(F1-macro UNSW→ToN, F1-macro ToN→UNSW)`;
2. `número de features selecionadas / número total de features`.

O F1 de cada direção, o diagnóstico de incompatibilidade de taxonomia e o erro
nas classes conhecidas permanecem separados nos artefatos. A quantidade de
soluções não dominadas não é definida previamente.

## Estrutura

```text
Datasets/                  dois Parquets locais completos
Modulos/
  avaliacao.py             avaliações intra e cross-dataset
  carregamento.py          leitura local e validação dos Parquets
  checkpoint.py            cache e progresso local do NSGA-II
  classificadores.py       fábrica de classificadores
  otimizacao.py            problema biobjetivo e NSGA-II
  preprocessamento.py      limpeza, taxonomia e alinhamento
src/
  algoritmo1_otimizacao.py Fase 1: pré-filtro NSGA-II
  algoritmo2_avaliacao.py  Fase 2: avaliação detalhada
  config.yaml              configuração central
  graficos_cross.py        figuras da análise
tests/                     testes unitários e de integração sintética
Resultados/                Pareto, métricas, checkpoints, logs e figuras
```

## Dados locais

Antes de executar, estes arquivos devem existir:

```text
Datasets/NF-UNSW-NB15-V2.parquet
Datasets/NF-ToN-IoT-V2.parquet
```

O pipeline lê os dois arquivos completos. Não há parâmetro de subamostragem no
fluxo científico.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Execução

Fase 1, com população 24, 15 gerações e operadores padrão do pymoo:

```bash
python src/algoritmo1_otimizacao.py
```

O cache fica em `Resultados/checkpoints`. Para retomar uma interrupção, eu
executo novamente o mesmo comando.

Fase 2, avaliando todas as soluções encontradas:

```bash
python src/algoritmo2_avaliacao.py \
  --pareto Resultados/pareto/pareto_<execucao>.json \
  --todas
```

Também posso avaliar uma solução ou a máscara completa:

```bash
python src/algoritmo2_avaliacao.py --pareto <arquivo.json> --solucao 0
python src/algoritmo2_avaliacao.py --mascara cheia
```

Depois da Fase 2:

```bash
python src/graficos_cross.py
```

## Testes

```bash
pytest
```

Os testes usam bases sintéticas pequenas. Eles não reduzem nem substituem os
datasets usados na execução científica.

## Referência dos datasets

Sarhan, Layeghy e Portmann, *Towards a Standard Feature Set for Network
Intrusion Detection System Datasets*, Mobile Networks and Applications, 2022.
