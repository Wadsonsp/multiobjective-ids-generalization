# E1 — Árvore de decisão, transferência, 2 bases, 2 objetivos

**Nome-padrão (fase B):** `experimentos/E1_arvore_transferencia_2bases_2obj/` · **Situação:** concluído em 07/09/2026 · [Voltar ao mapa](../MAPA_DA_PESQUISA.md)

Este experimento ocupa as pastas genéricas de `Resultados/` (`pareto/`, `metricas/`, `figuras/`), por isso esta ficha fica na raiz de `Resultados/`.

## Pergunta

A seleção multiobjetivo de atributos melhora a transferência de um detector entre duas redes (UNSW ↔ ToN)?

## Desenho

- Classificador: árvore de decisão multiclasse (coluna `Attack`), profundidade 8, pesos de classe balanceados.
- Protocolo: transferência — treina numa base e testa na outra (UNSW→ToN e ToN→UNSW). Classes do destino ausentes no treino ficam no código −1.
- Objetivos: `1 − média(F1-macro UNSW→ToN, F1-macro ToN→UNSW)` e `k/37`.
- NSGA-II: população 24, 15 gerações, seed 42, operadores padrão do pymoo.
- Fase 1 = busca; Fase 2 = avaliação intra e cross das soluções e da máscara completa, com 5 folds.

## Resultado principal

| Alternativa | Atributos | UNSW→ToN | ToN→UNSW | F1 cross médio |
|---|---|---|---|---|
| Baseline | 37 | 0,016712 | 0,077545 | 0,047128 |
| S1 | 3 | 0,054537 | 0,079990 | 0,067264 |

S1 usa `SERVER_TCP_FLAGS`, `TCP_WIN_MAX_IN` e `DNS_QUERY_ID` (−92% de atributos). O hipervolume foi de 0,100282 para 0,170428 e ainda cresceu 0,003375 na última geração.

## Limitações

As bases de destino participaram da seleção (não é teste externo); 15 gerações sem estabilização; uma seed; os F1 absolutos são muito baixos.

## Onde está

| O quê | Caminho |
|---|---|
| Configuração | [src/config.yaml](../src/config.yaml) |
| Pareto | [pareto/pareto_20260907_000921_decision_tree.json](pareto/pareto_20260907_000921_decision_tree.json) |
| Histórico da busca | [metricas/historico_20260907_000921_decision_tree.json](metricas/historico_20260907_000921_decision_tree.json) |
| Avaliações da Fase 2 | [metricas/pareto_20260907_000921_decision_tree/](metricas/pareto_20260907_000921_decision_tree/) |
| Figuras finais | [figuras/orientadores_final/](figuras/orientadores_final/) e `figuras/orientadores_final.zip` |
| Saída padrão do relatório | [figuras/orientadores_parcial/](figuras/orientadores_parcial/) |
| Serviços | `deploy/systemd/ids-otimizacao.service`, `ids-analise.service`, `ids-relatorio.service` e `.path` |

## Como reproduzir

```bash
python src/algoritmo1_otimizacao.py
python src/algoritmo2_avaliacao.py --pareto Resultados/pareto/pareto_20260907_000921_decision_tree.json --todas
.venv/bin/python src/relatorio_orientadores.py
```

## Commits e textos

- Resultados publicados em `2b2db1a` (07/09/2026); figuras atualizadas em `8e84850` e `3ea171e`.
- Protocolo: [README](../README.md), seções “Como formulei o experimento concluído com duas bases” e “Como reproduzo a execução com duas bases”.
- Roteiro: [roteiro_completo_orientadores.md](../docs/roteiro_completo_orientadores.md), seção 6.
- Diário: Entradas 001 a 003 do [diário de pesquisa](../docs/diario_de_pesquisa.md).

## E0 — Teste de execução

Em 27/08/2026, eu rodei a mesma busca com população 1 e 1 geração para testar a execução: [pareto/pareto_20260827_184529_decision_tree.json](pareto/pareto_20260827_184529_decision_tree.json) e [metricas/historico_20260827_184529_decision_tree.json](metricas/historico_20260827_184529_decision_tree.json). O código rejeita esse resultado como preliminar (`encontrar_pareto_final`, testado em `tests/test_relatorio.py`). Ele não entra na dissertação.
