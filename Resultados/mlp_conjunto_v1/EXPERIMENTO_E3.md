# E3 — MLP, treino conjunto, 3 bases, 3 objetivos

**Nome-padrão (fase B):** `experimentos/E3_mlp_conjunto_3bases_3obj/` · **Situação:** concluído — 100/100 gerações, escolhas fixadas na validação e teste reservado avaliado (`analise.concluida.json`); resultado final registrado em 18/11/2026 (commit `79033ff`) · [Voltar ao mapa](../../MAPA_DA_PESQUISA.md)

## Pergunta

Treinando com dados das três redes juntas, consigo reduzir atributos e tempo de inferência de uma MLP mantendo ou melhorando o desempenho? Este desenho não é transferência para uma rede que não participou do treino.

## O que mudou em relação ao E2

Três fatores ao mesmo tempo, além do orçamento da busca:

- **classificador:** árvore de decisão → MLP com 3 camadas de `2k` neurônios (ReLU, Adam, taxa 0,001, lote 1024, até 30 épocas, parada após 5 sem melhora, pesos de classe balanceados);
- **protocolo:** transferência → treino conjunto, com divisão treino/validação/teste de ~70/15/15% por grupos de registros idênticos (observado: 72,0/12,7/15,3%);
- **objetivos:** 2 → 3 (`1 − F1-macro` de validação, `k/37` e mediana do tempo por evento);
- **NSGA-II:** 15 → 100 gerações (população 24, seed 42).

## Resultado final (geração 100, validação e teste reservado)

Escolhas fixadas na validação ([escolhas_antes_do_teste.json](escolhas_antes_do_teste.json)) e avaliadas uma única vez no teste reservado ([teste/](teste/)). Os identificadores S1–S22 referem-se às máscaras da geração 100 e não coincidem com os da geração 73.

| Escolha (critério na validação) | Atributos | F1-macro validação | F1-macro teste | Mediana de inferência (validação) |
|---|---:|---:|---:|---:|
| S22 — maior F1 | 21 (−43%) | 0,551 | 0,533 | 1,32 µs/evento (−19%) |
| S10 — compromisso | 8 (−78%) | 0,464 | 0,463 | 0,87 µs/evento (−47%) |
| S4 — mais rápida | 4 | 0,233 | 0,184 | 0,65 µs/evento |
| S1 — mais compacta | 2 | 0,136 | 0,102 | 0,71 µs/evento |
| Baseline — 37 atributos | 37 | 0,516 | 0,533 | 1,63 µs/evento |

- As 22 soluções da geração 100 são não dominadas nos três objetivos. No teste, S22 e o baseline têm F1 numericamente iguais (0,533); a diferença é uma execução com seed 42 e não estabelece equivalência estatística.
- Na validação, cerca de 76% da vantagem de S22 sobre o baseline vem de `ftp-bruteforce`, classe com um único exemplo nessa partição (ver `figuras_extras/`, g7 e g8). É um alerta de interpretação, não uma causa comprovada.
- Candidato geométrico a joelho (critério: desvio abaixo da reta entre extremos, calculado só na projeção erro × atributos): S7, 6 atributos, F1 0,427, 1,42 µs/evento — um dos tempos mais altos. Nas outras projeções não há cálculo de joelho ([figuras/diagnostico_pareto.json](figuras/diagnostico_pareto.json)).
- O desenho não demonstra transferência para uma rede ausente do treino, invariância de atributos nem robustez entre seeds.

### Resultado parcial anterior (geração 73, só validação — mantido como histórico)

- As 22 soluções eram não dominadas nos três objetivos; cinco dominavam o baseline (37 atributos, F1 0,516, 1,63 µs/evento).
- S22: F1 0,551 com 21 atributos (−43%) e 1,32 µs (−19%). S12 (compromisso): F1 0,458 com 8 atributos (−78%) e 0,84 µs (−48%).
- Hipervolume perto de 0,50 desde a geração 40.
- Problemas identificados: TNR baixo, com cerca de metade dos alertas falsos, também no baseline (Entrada 006); F1 oscila de 0,07 a 0,11 entre épocas e há uma única seed (Entrada 007).

## Onde está

| O quê | Caminho |
|---|---|
| Configuração | [src/config_mlp_conjunto.yaml](../../src/config_mlp_conjunto.yaml) |
| Identidade e preparação | [experimento.json](experimento.json), [preparacao_resumo.json](preparacao_resumo.json), [LEIA-ME.md](LEIA-ME.md) |
| Baseline e calibração de tempo | [baseline_validacao.json](baseline_validacao.json), [calibracao_tempo.json](calibracao_tempo.json) |
| Execução (só local, fora do git) | `dados/`, `modelos/`, `geracoes/`, `progresso.json`, `logs/` |
| Figuras finais (geração 100, com teste) | [figuras/](figuras/) — `01`–`03` Pareto e hipervolume, `04` validação × teste, `05` matrizes de confusão, [relatorio_mlp.pdf](figuras/relatorio_mlp.pdf) |
| Escolhas e teste reservado | [escolhas_antes_do_teste.json](escolhas_antes_do_teste.json), [teste/](teste/), [analise.concluida.json](analise.concluida.json); `resultado_final.json` e `pareto_validacao.json` em Git LFS |
| Gráficos complementares (gerados à mão, geração 100) | `figuras_extras/` — `src/graficos_mlp_extras.py` |
| Figuras congeladas | [../revisao_pareto_tres_objetivos/](../revisao_pareto_tres_objetivos/) (g72) e [orientadores_g073/](orientadores_g073/) (g73), cada uma com `estado_gNNN.json` para refazer as figuras |
| Serviços | `deploy/systemd/ids-mlp-baseline.service` e `ids-mlp-conjunto.service` |

## Como reproduzir

```bash
.venv/bin/python -u src/executar_experimento_mlp.py --etapa preparar
.venv/bin/python -u src/executar_experimento_mlp.py --etapa baseline
.venv/bin/python -u src/executar_experimento_mlp.py --etapa completo
.venv/bin/python src/graficos_mlp_extras.py
```

## O que falta

A execução está concluída. Ficam em aberto, como propostas no mapa: estudo de seeds, controles (árvore no protocolo do E3, seleção simples), outro esquema de pesos de classe e avaliação em base externa ao treino.

## Commits e textos

- Código em `302f81c` (21/09/2026); nova codificação do Pareto em `92525f5` e registros em `5c9620b` (23/10/2026, branch `pareto-tres-objetivos-orientadores`); figuras da geração 82 em `fc6f34a` (27/10/2026); execução concluída e teste em `79033ff`, figuras finais e handoff em `0ef7e8d` (18/11/2026).
- Protocolo: [experimento_mlp_conjunto.md](../../docs/experimento_mlp_conjunto.md).
- Roteiros: [roteiro_mlp_conjunto.md](../../docs/roteiro_mlp_conjunto.md) e [roteiro_completo_orientadores.md](../../docs/roteiro_completo_orientadores.md), seções 10 a 14.
- Mensagens rápidas: [explicacoes_rapidas_orientadores.md](../../docs/explicacoes_rapidas_orientadores.md). Proposta: [proposta_estudo_seeds.md](../../docs/proposta_estudo_seeds.md).
- Diário: Entradas 004 a 007 do [diário de pesquisa](../../docs/diario_de_pesquisa.md).
