# E3 — MLP, treino conjunto, 3 bases, 3 objetivos

**Nome-padrão (fase B):** `experimentos/E3_mlp_conjunto_3bases_3obj/` · **Situação:** em execução desde 21/09/2026; geração 73/100 em 23/10/2026; teste ainda fechado · [Voltar ao mapa](../../MAPA_DA_PESQUISA.md)

## Pergunta

Treinando com dados das três redes juntas, consigo reduzir atributos e tempo de inferência de uma MLP mantendo ou melhorando o desempenho? Este desenho não é transferência para uma rede que não participou do treino.

## O que mudou em relação ao E2

Três fatores ao mesmo tempo, além do orçamento da busca:

- **classificador:** árvore de decisão → MLP com 3 camadas de `2k` neurônios (ReLU, Adam, taxa 0,001, lote 1024, até 30 épocas, parada após 5 sem melhora, pesos de classe balanceados);
- **protocolo:** transferência → treino conjunto, com divisão treino/validação/teste de ~70/15/15% por grupos de registros idênticos (observado: 72,0/12,7/15,3%);
- **objetivos:** 2 → 3 (`1 − F1-macro` de validação, `k/37` e mediana do tempo por evento);
- **NSGA-II:** 15 → 100 gerações (população 24, seed 42).

## Resultado parcial (geração 73, só validação)

- As 22 soluções são não dominadas nos três objetivos; cinco dominam o baseline (37 atributos, F1 0,516, 1,63 µs/evento).
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
| Figuras vivas (sobrescritas a cada geração) | [figuras/](figuras/) |
| Gráficos complementares (gerados à mão) | `figuras_extras/` — `src/graficos_mlp_extras.py` |
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

Concluir as 100 gerações, fixar as escolhas em `escolhas_antes_do_teste.json`, avaliar o teste e confirmar a conclusão por `analise.concluida.json`.

## Commits e textos

- Código em `302f81c` (21/09/2026); nova codificação do Pareto em `92525f5` e registros em `5c9620b` (23/10/2026, branch `pareto-tres-objetivos-orientadores`).
- Protocolo: [experimento_mlp_conjunto.md](../../docs/experimento_mlp_conjunto.md).
- Roteiros: [roteiro_mlp_conjunto.md](../../docs/roteiro_mlp_conjunto.md) e [roteiro_completo_orientadores.md](../../docs/roteiro_completo_orientadores.md), seções 10 a 14.
- Mensagens rápidas: [explicacoes_rapidas_orientadores.md](../../docs/explicacoes_rapidas_orientadores.md). Proposta: [proposta_estudo_seeds.md](../../docs/proposta_estudo_seeds.md).
- Diário: Entradas 004 a 007 do [diário de pesquisa](../../docs/diario_de_pesquisa.md).
