# E2 — Árvore de decisão, transferência, 3 bases, 2 objetivos

**Nome-padrão (fase B):** `experimentos/E2_arvore_transferencia_3bases_2obj/` · **Situação:** concluído em 10/09/2026, resultados publicados em 17/09/2026 · [Voltar ao mapa](../../MAPA_DA_PESQUISA.md)

## Pergunta

O compromisso entre transferência e número de atributos se mantém quando acrescento uma terceira rede (CIC)?

## O que mudou em relação ao E1

Só as bases: 2 → 3, ou seja, de 2 para 6 direções de transferência (UNSW→ToN, ToN→UNSW, UNSW→CIC, CIC→UNSW, ToN→CIC e CIC→ToN). Classificador, objetivos, NSGA-II (população 24, 15 gerações, seed 42) e tratamento dos dados são os mesmos. O primeiro objetivo passa a ser `1 − média dos F1-macro das seis direções`.

## Resultado principal

| Alternativa | Atributos | Redução | F1 cross médio |
|---|---|---|---|
| Baseline | 37 | 0% | 0,035290 |
| S1 | 7 | 81% | 0,046460 |
| S2 | 8 | 78% | 0,058466 |
| S3 | 11 | 70% | 0,058526 |
| S4 | 12 | 68% | 0,061615 |

- S4 tem a maior média, mas perde para o baseline em ToN→UNSW (0,055 vs 0,078). S2 supera o baseline nas seis direções.
- S4 melhora o cross, mas reduz o F1 intra nas três bases (ex.: UNSW 0,407 vs 0,473 do baseline).
- O hipervolume foi de 0,1061 para 0,1465 em 15 gerações.

## Revisão dos eixos (21/09/2026)

Para preparar a apresentação do E3, eu redesenhei o Pareto do E2 com `1 − F1` no eixo vertical, sem alterar os resultados, e encontrei um candidato a joelho em S2: [revisao_pareto_anterior/](../revisao_pareto_anterior/), gerada por `src/revisar_pareto_anterior.py`.

## Limitações

As três bases participaram da seleção (não há teste externo); 15 gerações; uma seed; F1 cross absoluto entre 0,01 e 0,10; F1 intra e cross usam vocabulários de classes diferentes.

## Onde está

| O quê | Caminho |
|---|---|
| Configuração | [src/config_tres_datasets.yaml](../../src/config_tres_datasets.yaml) |
| Pareto | [pareto/pareto_20260910_053058_decision_tree.json](pareto/pareto_20260910_053058_decision_tree.json) |
| Histórico e avaliações | [metricas/](metricas/) |
| Figuras e tabelas | [figuras/](figuras/) (inclui `resumo_solucoes.csv` e `convergencia_nsga2.csv`) e `figuras.zip` |
| Revisão dos eixos | [../revisao_pareto_anterior/](../revisao_pareto_anterior/) |
| Serviço | `deploy/systemd/ids-tres-datasets.service` |

## Como reproduzir

```bash
.venv/bin/python -u src/executar_experimento.py --config src/config_tres_datasets.yaml
```

## Commits e textos

- Código em `b5ae8ed` (07/09/2026); resultados em `da1025f` e `33f3321` (17/09/2026).
- Protocolo: [experimento_tres_datasets.md](../../docs/experimento_tres_datasets.md).
- Roteiros: [roteiro_tres_datasets.md](../../docs/roteiro_tres_datasets.md) e [roteiro_completo_orientadores.md](../../docs/roteiro_completo_orientadores.md), seções 7 e 8.
- Diário: Entradas 001 a 003 do [diário de pesquisa](../../docs/diario_de_pesquisa.md).
