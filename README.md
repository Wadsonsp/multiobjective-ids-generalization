# Seleção multiobjetivo de features para IDS cross-dataset

Neste projeto da minha dissertação de mestrado, investigo a generalização
cross-dataset em Sistemas de Detecção de Intrusão. Eu uso o NSGA-II como **pré-filtro (Fase 1)**
para encontrar compromissos entre desempenho cross-dataset e quantidade de
features. Eu faço a escolha e a análise detalhada das soluções na **Fase 2**.

## Meu novo experimento com MLP conjunta — 21/09/2026

Eu implementei um protocolo independente com treino conjunto das três bases,
MLP com três camadas de `2k` neurônios, NSGA-II com 100 gerações e seed 42,
e três objetivos: `1 − F1-macro`, `k/d` e tempo por evento.
Eu apresento o Pareto somente em **2D**, verificando sua orientação e um possível
joelho sem impor formato de curva. Eu separo treino, validação e teste por grupos
de entradas idênticas e avalio o teste apenas após fixar as escolhas.

Eu descrevo configuração, execução e limitações no
[protocolo da MLP](docs/experimento_mlp_conjunto.md) e preparo minha fala no
[novo roteiro](docs/roteiro_mlp_conjunto.md). Eu preservo abaixo os resultados
anteriores com árvore e seis transferências. Eu não os apresento como resultados
da MLP. A configuração nova está em `src/config_mlp_conjunto.yaml`.

```bash
.venv/bin/python -u src/executar_experimento_mlp.py --etapa preparar
.venv/bin/python -u src/executar_experimento_mlp.py --etapa baseline
.venv/bin/python -u src/executar_experimento_mlp.py --etapa completo
```

## Meu roteiro e os resultados atuais com três bases

Eu reuni a explicação das mudanças no código, os resultados e sua contribuição
para minha dissertação no [meu roteiro de apresentação](docs/roteiro_tres_datasets.md),
atualizado em 17/09/2026. Eu confirmei a conclusão desta rodada em 10/09/2026:
15 gerações, quatro soluções e uma avaliação com todos os 37 atributos.
Eu encontrei seleções de 7 a 12 atributos com F1-macro cross médio entre
0,046460 e 0,061615; no baseline, obtive 0,035290. Eu reconheço os valores
absolutos baixos e a ausência de teste externo independente.

Eu consulto o [relatório de três bases](Resultados/tres_datasets_v1/figuras/relatorio_graficos.pdf)
e as [avaliações detalhadas](Resultados/tres_datasets_v1/metricas/pareto_20260910_053058_decision_tree/).
Eu preservo abaixo o planejamento e as observações datadas do início da rodada;
eu uso o roteiro para consultar a síntese atual dos resultados.

## Como registro minhas decisões e descobertas

Eu mantenho um [diário de pesquisa](docs/diario_de_pesquisa.md) para apresentar
aos meus orientadores as decisões metodológicas, suas justificativas, as
evidências observadas e as questões ainda em aberto. Eu registro ali por que
mantive inicialmente 15 gerações e por que não considero esse orçamento uma
prova de convergência. Eu preservo fotografias datadas da execução em
[docs/registros](docs/registros/), distinguindo acompanhamento parcial de
resultados concluídos.

## Onde explico os mecanismos da minha metodologia

Eu descrevo a limpeza e o alinhamento dos datasets, os pesos de classe no
treinamento, os dois objetivos do NSGA-II e a penalização `[1, 1]` para máscaras
sem atributos na [metodologia atual](docs/experimento_tres_datasets.md#como-aplico-os-mecanismos-da-metodologia-atual).
Eu também explico como agrupo classes desconhecidas em `-1` no cálculo do F1
e quais limitações reconheço. Eu registrei essa verificação na entrada 003 do
[meu diário](docs/diario_de_pesquisa.md).

## O que pretendo ampliar agora

Eu pretendo passar de dois para três datasets, incluindo o
**NF-CSE-CIC-IDS2018-v2** na otimização e na avaliação. Vou investigar seis
direções de transferência, mantendo a identidade de cada origem e destino.
Eu descrevo minha proposta, as hipóteses e as mudanças necessárias no
[protocolo do experimento com três datasets](docs/experimento_tres_datasets.md).

Eu converti integralmente o CSV oficial em Parquet e conferi 18.893.708 linhas
na entrada e na saída. Eu adaptei o pipeline para três bases, com configuração
própria em `src/config_tres_datasets.yaml`. Eu preservo `src/config.yaml` para
reproduzir o experimento com duas bases. Não trato a nova rodada como concluída
antes de verificar seu marcador e suas métricas.

Na ampliação, proponho minimizar `1 - média dos seis F1-macro direcionais` e
`k/d`. Vou usar todos os registros válidos, avaliar as soluções e o baseline,
e separar configuração, checkpoints, serviços e saídas por experimento.
Vou continuar investigando a convergência: como ainda observei melhoria na
última geração da rodada anterior, não concluo que 15 gerações sejam suficientes.

## Como executo minha rodada com três bases

Eu iniciei a rodada com 34.015.950 registros válidos e 37 atributos comuns.
Eu uso `src/executar_experimento.py` para encadear otimização, avaliação de todas
as soluções, baseline e relatórios. Eu identifico o protocolo pela configuração
e pelos hashes SHA-256 dos três Parquets. Eu não reaproveito o cache da rodada
com duas bases. Eu armazeno as saídas em `Resultados/tres_datasets_v1/`.

Eu acompanho o serviço com:

```bash
systemctl --user status ids-tres-datasets.service
tail -f ~/multiobjective-ids-generalization/Resultados/tres_datasets_v1/logs/execucao.log
```

Eu verifico `Resultados/tres_datasets_v1/checkpoints/analise.concluida` para
confirmar que concluí todas as etapas. Eu retomo uma interrupção com o mesmo
serviço; preservo as avaliações detalhadas já salvas e refaço somente a solução
que estava em andamento.

Eu instalo o serviço, quando necessário, com:

```bash
mkdir -p Resultados/tres_datasets_v1/logs ~/.config/systemd/user
cp deploy/systemd/ids-tres-datasets.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger "$USER"
systemctl --user enable --now ids-tres-datasets.service
```

Eu posso parar a rodada com `systemctl --user stop ids-tres-datasets.service`.
Como alternativa ao serviço, eu executo o mesmo pipeline no terminal:

```bash
.venv/bin/python -u src/executar_experimento.py --config src/config_tres_datasets.yaml
```

Eu uso um lock por experimento para evitar duas execuções simultâneas desse
pipeline. Eu não inicio os scripts individuais em paralelo ao serviço.
Eu preservo as contagens após limpeza em
`Resultados/tres_datasets_v1/checkpoints/dados_preprocessamento.json` e consulto
os relatórios em `Resultados/tres_datasets_v1/figuras/` quando forem produzidos.

## Como interpreto a tabela de acompanhamento do NSGA-II

Eu acompanho a tabela no log da execução. Cada linha registra uma geração
concluída; durante o cálculo da geração seguinte, posso continuar vendo a
última linha sem que isso signifique uma interrupção.

```text
n_gen  |  n_eval  | n_nds  |      eps      |   indicator
     1 |       24 |      3 |             - |             -
     2 |       48 |      5 |  0.3944093989 |         ideal
```

| Coluna que acompanho | Como eu a interpreto |
|---|---|
| `n_gen` | Eu leio o número da geração concluída. Nesta rodada, configurei 15 gerações; a primeira corresponde à avaliação da população inicial. |
| `n_eval` | Eu leio o total acumulado de indivíduos avaliados pelo algoritmo. Não interpreto esse número como quantidade de linhas dos datasets nem de treinamentos novos: uma máscara repetida pode recuperar critérios do cache. |
| `n_nds` | Eu leio a quantidade de indivíduos não dominados no conjunto corrente `algorithm.opt`. Eu considero não dominado um indivíduo quando nenhum outro do conjunto comparado é pelo menos tão bom nos dois objetivos e estritamente melhor em um deles. Não interpreto esse conjunto como a fronteira ótima global conhecida. |
| `eps` | Eu acompanho uma medida de mudança entre gerações no espaço dos objetivos, conforme o componente identificado em `indicator`. Eu não a interpreto como F1, hipervolume, erro de classificação ou porcentagem de melhoria. |
| `indicator` | Eu identifico qual componente originou o `eps` mostrado: `ideal`, `nadir` ou `f`. |

### Como distingo os valores de `indicator`

- **`ideal`:** eu acompanho a mudança normalizada dos melhores valores de cada
  objetivo no conjunto corrente. Esses melhores valores podem pertencer a
  soluções diferentes; eu não suponho que uma única solução alcance todos eles.
- **`nadir`:** eu acompanho a mudança normalizada dos maiores valores de cada
  objetivo no conjunto não dominado corrente. Eu trato esse ponto como a
  estimativa corrente usada pelo pymoo, não como o nadir exato de uma fronteira
  ótima conhecida.
- **`f`:** eu acompanho uma distância entre as fronteiras de gerações sucessivas,
  calculada por IGD após a normalização interna. Eu não confundo esse cálculo
  entre gerações com IGD em relação a uma fronteira ótima verdadeira.
- **`-`:** eu reconheço que ainda não há comparação disponível, como na primeira
  geração, quando não existe uma geração anterior para essa medida.

Eu conferi essa explicação na implementação local do **pymoo 0.6.2**, nas
classes `MultiObjectiveOutput`, `NumberOfNondominatedSolutions` e
`MultiObjectiveSpaceTermination`. Nessa implementação, eu observo que a exibição
prioriza `ideal` quando sua mudança supera a tolerância padrão `0.0025`;
caso contrário, verifica `nadir` e, por último, apresenta `f`. Eu não interpreto
`indicator` como uma escolha obrigatória do maior entre os três valores.

Eu observo que valores pequenos de `eps` indicam pequena mudança no componente
mostrado, mas não comprovam ótimo global ou convergência definitiva. Como o
componente pode mudar, eu não leio a sequência de `eps` como uma única métrica
homogênea. Na minha execução, eu uso **15 gerações como critério de parada**;
a tolerância usada para exibir essa tabela não encerra automaticamente a busca.
Eu examino separadamente a curva de hipervolume para complementar o diagnóstico.

### Como relaciono indivíduos, máscaras e checkpoints

Eu uso genes reais e aplico o limiar `0.5` para obter máscaras binárias.
Assim, indivíduos distintos podem selecionar os mesmos atributos. Por isso,
eu não equiparo `n_nds` à quantidade de máscaras únicas que salvo no Pareto
final. Também não interpreto mais indivíduos não dominados como melhoria
obrigatória: uma nova solução pode dominar várias anteriores.

Eu gravo cada avaliação nova concluída no cache, enquanto a linha da tabela
só aparece ao concluir a geração. Assim, posso ter 56 máscaras únicas salvas
mesmo quando a última linha mostra `n_eval=48`. Eu uso o serviço, o log e os
checkpoints em conjunto para acompanhar o trabalho.

### O que observei em 07/09/2026 às 17h24

Eu confirmei o serviço ativo, sem reinícios por falha, **duas de 15 gerações
concluídas** e a terceira em andamento. Eu contei **56 avaliações únicas salvas**,
com última gravação às **17h15**. Eu registro esses valores como uma fotografia
datada; eles não representam o estado atual permanente do projeto.

Na segunda linha do exemplo, eu interpreto `2 | 48 | 5 | 0.3944093989 | ideal`
como a conclusão da geração 2, com 48 indivíduos avaliados desde o início,
cinco indivíduos não dominados e mudança normalizada do ponto ideal. Eu não
interpreto `0.3944093989` como uma melhoria de 39,44% no desempenho do detector.
Eu ainda não concluí a otimização, a avaliação detalhada ou o baseline desta
rodada com três bases na data dessa observação.

## Resultados disponíveis no repositório

Eu concluí a execução com duas bases em 7 de setembro de 2026: 15 gerações do
NSGA-II, avaliação detalhada da solução final e baseline com os 37 atributos.

- [Relatório completo em PDF](Resultados/figuras/orientadores_final/relatorio_graficos.pdf)
- [Pacote para download: gráficos, tabelas e dados utilizados](Resultados/figuras/orientadores_final.zip)
- [Curva de hipervolume por geração](Resultados/figuras/orientadores_final/00_convergencia_hipervolume.png)
- [Comparação intra/cross com baseline](Resultados/figuras/orientadores_final/06_intra_vs_cross.png)
- [Notas metodológicas e interpretação](Resultados/figuras/orientadores_final/LEIA-ME.md)
- [Pareto final](Resultados/pareto/pareto_20260907_000921_decision_tree.json)
- [Avaliações detalhadas](Resultados/metricas/pareto_20260907_000921_decision_tree/)

Eu abro o HTML localmente após extrair o ZIP. Incluí o diagnóstico de
convergência e interpreto a melhora do hipervolume como progresso da busca,
sem tomá-la como prova de ótimo global ou generalização externa.

## Como formulei o experimento concluído com duas bases

Eu represento cada solução por um vetor real `x`, com um gene em `[0, 1]` para
cada feature. Eu
mantenho a feature quando `x >= 0.5` e minimizo:

1. `1 - média(F1-macro UNSW→ToN, F1-macro ToN→UNSW)`;
2. `número de features selecionadas / número total de features`.

Eu preservo separadamente o F1 de cada direção, a incompatibilidade de taxonomia
e o erro nas classes conhecidas. Não defino previamente a quantidade de soluções
não dominadas; observo o conjunto que a busca produz.

## Como organizo o projeto

```text
Datasets/                  meus Parquets e o ZIP da terceira base
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
Resultados/                meus artefatos de cada execução
docs/                      meu protocolo de ampliação para três datasets
```

## Dados locais

No experimento concluído, eu utilizo estes arquivos:

```text
Datasets/NF-UNSW-NB15-V2.parquet
Datasets/NF-ToN-IoT-V2.parquet
```

Eu leio os dois arquivos locais completos e removo apenas os registros inválidos
conforme o pré-processamento. Não aplico subamostragem no fluxo científico.
Eu registro a preparação da terceira base em [Datasets/README.md](Datasets/README.md).

## Como preparo meu ambiente

Eu preparo o ambiente Python com:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Como reproduzo a execução com duas bases

Eu executo a Fase 1 com população 24, 15 gerações e operadores padrão do pymoo:

```bash
python src/algoritmo1_otimizacao.py
```

Eu mantenho o cache em `Resultados/checkpoints`. Para retomar uma interrupção, eu
executo novamente o mesmo comando.

Eu avalio todas as soluções encontradas na Fase 2:

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

Depois da Fase 2, eu gero os gráficos:

```bash
python src/graficos_cross.py
```

## Como verifico a implementação

Eu executo meus testes com:

```bash
pytest
```

Eu uso bases sintéticas pequenas nos testes. Com elas, verifico a implementação
sem reduzir ou substituir os datasets da execução científica.

## Referência que utilizo para os datasets

Sarhan, Layeghy e Portmann, *Towards a Standard Feature Set for Network
Intrusion Detection System Datasets*, Mobile Networks and Applications, 2022.

## Como reproduzo a execução com duas bases independente da sessão

Eu configurei `ids-otimizacao.service` para executar a Fase 1 com
`src/config.yaml`, reutilizar checkpoints e reiniciar após falhas em 60 segundos.
Eu registro a conclusão em `Resultados/checkpoints/otimizacao.concluida` para
evitar repetir o experimento após o término ou após reiniciar o computador.

Eu instalo o serviço para meu usuário, com o projeto em `~/multiobjective-ids-generalization`:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/ids-otimizacao.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger "$USER"
systemctl --user enable --now ids-otimizacao.service
```

Eu habilito `enable-linger` para manter o gerenciador após logout e iniciar
no boot sem login; dependendo da máquina, preciso de autorização administrativa.
Eu retomo a execução quando a máquina volta, aproveitando as avaliações do cache.

Eu acompanho ou paro o experimento com:

```bash
systemctl --user status ids-otimizacao.service
tail -f Resultados/logs/nsga2.log
systemctl --user stop ids-otimizacao.service
```

Eu desativo também a inicialização automática com:

```bash
systemctl --user disable --now ids-otimizacao.service
```

Eu verifico o código de saída zero e o marcador para reconhecer uma conclusão
bem-sucedida, mesmo quando vejo o serviço inativo. Para repetir a mesma rodada,
posso remover apenas seu marcador de conclusão e iniciar o serviço novamente.
Eu uso um lock no script e evito iniciar o Python diretamente em paralelo.
Para três bases, vou criar uma identidade de execução própria; não basta remover
o marcador antigo ou alterar o YAML.

Eu também posso executar manualmente com o serviço desativado:

```bash
nohup bash src/executar_otimizacao.sh >> Resultados/logs/nsga2.log 2>&1 < /dev/null &
```

Eu uso `nohup` apenas para proteção contra desconexão do terminal; para retomada
automática após falha ou reboot, uso os serviços supervisionados.

## Material para os orientadores

Eu gero o relatório do checkpoint ou da execução concluída sem repetir treinamentos:

```bash
.venv/bin/python src/relatorio_orientadores.py
```

Eu consulto `Resultados/figuras/orientadores_parcial/index.html` ou o PDF nessa
pasta para a saída padrão do comando. Para compartilhar a execução concluída,
uso o pacote `orientadores_final` vinculado no início deste README. Eu incluo
figuras PNG/PDF, CSV, uma cópia dos dados e a identificação do estágio da análise.

Eu configurei a continuação para aguardar a Fase 1, gerar o Pareto, avaliar
cada solução e a máscara completa com cinco folds, e atualizar o pacote final.
Eu salvo atomicamente cada solução concluída. Após uma interrupção, reaproveito
as soluções já salvas e refaço a interrompida; retomo por solução, não por fold.

```bash
cp deploy/systemd/ids-analise.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ids-analise.service
tail -f Resultados/logs/analise.log
```

Eu paro a avaliação e a geração automática com:

```bash
systemctl --user disable --now ids-analise.service
```

Eu mantenho a otimização no serviço separado `ids-otimizacao.service` e salvo
as avaliações em `Resultados/metricas/pareto_<execucao>/`. Eu verifico o marcador
`analise.concluida` para distinguir a conclusão de todas as etapas de um pacote
ainda atualizado parcialmente.

### Convergência do NSGA-II

Eu incluo a curva `00_convergencia_hipervolume.png` e seu PDF, uma página de
diagnóstico e `convergencia_nsga2.csv`. Eu calculo o hipervolume da fronteira
sobrevivente em cada geração com referência fixa `(1.1, 1.1)` e objetivos
`1 - F1_cross_medio` e `k/d`, ambos em `[0,1]`. Não normalizo por geração nem
substituo essa fronteira pelo conjunto acumulado de soluções visitadas.

Eu reconstruo as gerações com a mesma seed, população e operadores, consultando
uma cópia do cache. Não leio Parquets, retreino modelos ou altero checkpoints
nesse procedimento. Eu interrompo a reconstrução quando falta uma máscara e
confiro a fronteira reconstruída contra a salva antes de publicar a curva.
Não interpreto a posição de uma linha do cache como o número de uma geração.

Eu interpreto o diagnóstico como progresso nos objetivos, sem concluir ótimo
global, estabilidade entre seeds ou generalização independente. Eu mantenho
a referência fixa e consulto a seguinte base metodológica:
https://pymoo.org/getting_started/part_4.html

Na rodada `20260907_000921`, eu usei `ids-relatorio.path` para atualizar o
pacote após o baseline. Essa rodada já terminou. Eu mantenho o diagnóstico
integrado ao gerador e vou adaptar os caminhos dos serviços antes da rodada
com três bases.
