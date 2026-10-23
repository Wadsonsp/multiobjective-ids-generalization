# Mapa da minha pesquisa

Eu uso este mapa para achar cada experimento, seus resultados e seus textos, na ordem em que fiz a pesquisa. Atualizado em 23/10/2026.

Os arquivos ainda estão nas pastas originais (fase A da reorganização). Cada experimento tem uma ficha com tudo o que lhe pertence. O nome-padrão indica para onde ele irá na fase B, quando eu mover as pastas para `experimentos/`.

## Linha do tempo

| Exp. | Período | Classificador | Protocolo | Objetivos | Bases | NSGA-II | Situação | Ficha |
|---|---|---|---|---|---|---|---|---|
| E0 | 27/08/2026 | árvore de decisão | transferência | 2 | 2 | pop. 1, 1 geração | teste de execução, descartado | [E1](Resultados/EXPERIMENTO_E1.md#e0--teste-de-execução) |
| E1 | 07/09/2026 | árvore de decisão | transferência | 2 | 2 | pop. 24, 15 gerações | concluído | [E1](Resultados/EXPERIMENTO_E1.md) |
| E2 | 07–17/09/2026 | árvore de decisão | transferência | 2 | **3** | pop. 24, 15 gerações | concluído | [E2](Resultados/tres_datasets_v1/EXPERIMENTO_E2.md) |
| E3 | 21/09/2026 → | **MLP** | **treino conjunto** | **3** | 3 | pop. 24, **100** gerações | em execução (geração 73 em 23/10) | [E3](Resultados/mlp_conjunto_v1/EXPERIMENTO_E3.md) |

Todos usam seed 42, os mesmos 37 atributos candidatos da família NetFlow v2 e a remoção de portas e TTL por risco de vazamento.

- **Objetivos com 2:** erro de transferência (`1 − F1-macro cross médio`) e proporção de atributos (`k/37`).
- **Objetivos com 3:** erro na validação conjunta (`1 − F1-macro`), proporção de atributos e tempo de inferência por evento.
- **Bases:** NF-UNSW-NB15-v2 (UNSW), NF-ToN-IoT-v2 (ToN) e, a partir do E2, NF-CSE-CIC-IDS2018-v2 (CIC).

## O que mudou de um experimento para o outro

| Passagem | O que mudou | O que isso permite concluir |
|---|---|---|
| E0 → E1 | só o orçamento da busca (pop. 1 → 24; 1 → 15 gerações) | nada sobre o método; o E0 só testou a execução |
| E1 → E2 | **bases 2 → 3** (2 → 6 direções de transferência) | o efeito de acrescentar uma rede, com o resto igual |
| E2 → E3 | **classificador, protocolo e objetivos ao mesmo tempo**, além de 15 → 100 gerações e da nova divisão treino/validação/teste | uma mudança de pergunta; a diferença de desempenho não pode ser atribuída a um fator isolado |

Uma leitura possível do fio condutor, ainda a discutir com os orientadores: no E1 e no E2, a transferência entre redes falha (F1 cross de 0,01 a 0,10) e a seleção de atributos melhora pouco; o E3 responde treinando com as três redes juntas e acrescenta o custo de inferência.

## Experimentos propostos, ainda não executados

| Proposta | O que isolaria | Onde está descrita |
|---|---|---|
| Estudo de seeds do modelo (E3) | variação do F1 causada só pelo treino | [proposta_estudo_seeds.md](docs/proposta_estudo_seeds.md), diário Entrada 007 |
| Árvore no protocolo do E3 (ponte) | efeito do classificador contra efeito do protocolo | ainda só sugerida; sem documento |
| Outro esquema de pesos de classe | causa dos falsos alarmes (TNR baixo) | diário Entrada 006 |
| MLP treinada em duas bases e testada na terceira | generalização externa da MLP | ainda só sugerida; sem documento |
| Comparação com métodos simples de seleção | se o NSGA-II é necessário | ainda só sugerida; sem documento |

## Onde encontrar cada tipo de coisa

| Procuro | E1 | E2 | E3 |
|---|---|---|---|
| Configuração | [src/config.yaml](src/config.yaml) | [src/config_tres_datasets.yaml](src/config_tres_datasets.yaml) | [src/config_mlp_conjunto.yaml](src/config_mlp_conjunto.yaml) |
| Protocolo | [README](README.md), seções “Como formulei o experimento concluído com duas bases” e “Como reproduzo a execução com duas bases” | [experimento_tres_datasets.md](docs/experimento_tres_datasets.md) | [experimento_mlp_conjunto.md](docs/experimento_mlp_conjunto.md) |
| Roteiro para orientadores | [roteiro_completo_orientadores.md](docs/roteiro_completo_orientadores.md), seção 6 | [roteiro_tres_datasets.md](docs/roteiro_tres_datasets.md); roteiro completo, seções 7 e 8 | [roteiro_mlp_conjunto.md](docs/roteiro_mlp_conjunto.md); roteiro completo, seções 10 a 14 |
| Pareto e métricas | `Resultados/pareto/`, `Resultados/metricas/` | `Resultados/tres_datasets_v1/pareto/`, `metricas/` | `Resultados/mlp_conjunto_v1/` (execução local) |
| Figuras principais | [figuras/orientadores_final/](Resultados/figuras/orientadores_final/) | [tres_datasets_v1/figuras/](Resultados/tres_datasets_v1/figuras/) | [figuras/](Resultados/mlp_conjunto_v1/figuras/) (vivas) |
| Figuras congeladas e revisões | [figuras/orientadores_parcial/](Resultados/figuras/orientadores_parcial/) | [revisao_pareto_anterior/](Resultados/revisao_pareto_anterior/) | [revisao_pareto_tres_objetivos/](Resultados/revisao_pareto_tres_objetivos/) (g72), [orientadores_g073/](Resultados/mlp_conjunto_v1/orientadores_g073/) (g73) |
| Mensagens rápidas | — | — | [explicacoes_rapidas_orientadores.md](docs/explicacoes_rapidas_orientadores.md) |
| Diário | Entradas 001–003 | Entradas 001–003 | Entradas 004–007 |

O [diário de pesquisa](docs/diario_de_pesquisa.md) é único e cronológico para todos os experimentos. O [roteiro completo](docs/roteiro_completo_orientadores.md) cobre E1, E2 e E3 numa só apresentação.

## Antes do E0: desenvolvimento do pipeline (maio a agosto de 2026)

Entre maio e agosto, eu montei o pipeline: scripts iniciais (removidos no commit `2e9b777`), suporte a XGBoost, análise da queda de desempenho cross-dataset (`src/graficos_cross.py`, commit `228112e`) e a refatoração do NSGA-II para execução local (`7cb748b`). Não há resultados dessa fase preservados no projeto; o histórico está no git.

## Código compartilhado

O código fica fora das pastas de experimento, porque os experimentos o compartilham: `Modulos/` (dados, modelos, otimização, avaliação), `src/` (pontos de entrada, relatórios e gráficos) e `tests/`. Os serviços que rodam cada experimento em segundo plano estão em [deploy/systemd/](deploy/systemd/).
