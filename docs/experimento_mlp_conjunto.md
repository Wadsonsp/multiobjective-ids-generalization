# Meu novo experimento: MLP conjunta e três objetivos

Eu defini este protocolo em 21/09/2026, seguindo as orientações do Eduardo e a decisão de apresentar somente gráficos de Pareto em duas dimensões. Eu preservo o experimento anterior com árvore e seis transferências como outro protocolo.

## Minha pergunta

Eu investigo se consigo reduzir os atributos e o tempo de inferência de uma MLP, mantendo ou melhorando o desempenho em registros de teste reservados das três bases. Eu treino com dados de UNSW, ToN e CSE juntos. Eu não apresento esse desenho como transferência para uma base que não participou do treinamento.

## Como eu atendo às orientações

| Orientação que recebi | Decisão que implementei |
|---|---|
| Aumentar 15 para 100 e manter seed 42 | Eu uso 100 gerações do NSGA-II, população 24 e seed 42. |
| Juntar as três bases | Eu reúno suas partes de treino, preservando a origem para as análises. |
| Usar uma MLP com três camadas | Eu uso três camadas escondidas com `2k` neurônios cada, sendo `k` a quantidade selecionada. |
| Acrescentar tempo por evento | Eu minimizo a mediana de medições repetidas de padronização + previsão, divididas pelo número de eventos. |
| Descobrir o tempo máximo | Eu registro o maior tempo médio por evento observado nas repetições de cada solução. Eu não afirmo um máximo universal nem latência máxima individual. |
| Rever a fronteira | Eu mostro somente projeções 2D, com os sentidos dos objetivos explícitos. Eu verifico um candidato geométrico a joelho sem impor curvatura. |
| Avaliar cada classe | Eu salvo matrizes de contagens e percentuais por linha, precisão, recall, F1 e suporte. |

## Como eu separo treino, validação e teste

Eu planejo aproximadamente 70%/15%/15%. Eu não separo linhas idênticas aleatoriamente: calculo um hash dos 37 atributos comuns em `float32`, antes da seleção, e atribuo o grupo inteiro a uma parte com seed 42. Eu uso a mesma regra entre as bases e entre rótulos, para que entradas idênticas não atravessem as partições. Eu preservo os duplicados dentro de sua partição; não aplico subamostragem para treino ou cálculo de métricas.

Eu aceito proporções aproximadas, pois grupos inteiros podem concentrar registros. Eu salvo as contagens reais por base, classe e partição no manifesto. Eu não chamo esse procedimento de divisão estratificada exata. Se uma classe não tiver nenhum exemplo no treino, eu interrompo a preparação e reviso o protocolo antes da busca.

Eu não encontrei identificadores de sessão ou tempo comuns aos três Parquets. Meu agrupamento reduz vazamento por entradas idênticas, mas não demonstra independência entre sessões, capturas ou períodos, nem elimina quase duplicatas. Eu reconheço essa limitação na dissertação.

Eu leio os dados em lotes e armazeno as matrizes em disco. Eu ajusto média e desvio dos atributos somente com o treino. Eu transformo validação e teste com esses parâmetros. A MLP e o NSGA-II não consultam as previsões ou métricas do teste durante a busca.

Eu normalizo espaços e capitalização dos rótulos, mantendo os nomes existentes. Eu não uno automaticamente `dos` com categorias específicas de DoS de outra base. Eu registro o vocabulário e as contagens para revisão da taxonomia. Eu reconheço que nomes iguais não comprovam, por si só, equivalência semântica entre datasets.

## Como eu treino a MLP

Eu uso a arquitetura `k → 2k → 2k → 2k → classes`, ReLU, Adam, taxa inicial 0,001, regularização alpha 0,0001 e minibatches de 1.024 registros. Eu embaralho conjuntamente os índices de todo o treino a cada época, mantendo as três origens no mesmo processo de treinamento.

Eu configurei um limite inicial de 30 épocas, paciência de 5 épocas e melhora mínima de 0,0001 no F1-macro de validação. Eu preservo o modelo da melhor época segundo esse critério. Eu uso pesos por classe calculados apenas nas contagens do treino. A implementação exige scikit-learn >= 1.7 para pesos na MLP incremental.

Eu distingo as 100 gerações do NSGA-II das épocas de cada MLP. Cada época visita todos os registros de treino. Eu não uso a validação interna automática do `MLPClassifier`; uso a partição já preparada tanto para acompanhar o treinamento quanto para avaliar as máscaras.

Eu salvo o estado do modelo e do otimizador Adam ao fim de cada época. Se houver interrupção no meio de uma época, eu refaço somente essa época a partir do estado anterior. Eu guardo os modelos concluídos para avaliar as escolhas no teste sem retreiná-los nem misturar a validação ao treino.

Quando reduzo `k`, eu também reduzo o tamanho das camadas. Portanto, uma mudança no tempo pode refletir os atributos e a arquitetura. Eu não atribuo o efeito exclusivamente à seleção de atributos.

## Meus três objetivos

Eu minimizo simultaneamente:

1. `1 − F1-macro de validação` sobre o conjunto unido;
2. `k / d`, com `d` igual ao número de atributos candidatos comuns;
3. mediana dos `segundos de inferência / eventos` nas repetições.

Eu mantenho o mesmo vocabulário de classes do treino para calcular o macro-F1; uma classe sem suporte e sem previsão contribui com zero. Eu declaro essa convenção também nas métricas por base, nas quais algumas categorias não existem. Eu forneço o suporte para que essa situação fique visível. Eu registro a acurácia geral como medida complementar.

Eu trato a máscara vazia como solução inviável, sem treinar um modelo. Eu incluo a máscara completa na população inicial e a uso como baseline, com os mesmos dados e regras de treinamento.

## Como eu meço tempo e máximo observado

Eu uso os mesmos até 65.536 registros de validação, escolhidos com seed 42, em todas as máscaras. Esse conjunto menor serve somente ao benchmark de tempo; o F1 e as matrizes usam a partição inteira. Eu processo lotes de 4.096 eventos, faço dois aquecimentos e sete repetições. Eu fixo duas threads e avalio as máscaras sequencialmente.

Eu cronometro a padronização dos atributos selecionados e a chamada de previsão. Eu excluo leitura de disco, seleção inicial dos registros do benchmark e treinamento. Eu divido cada tempo total pelo número de eventos e salvo todos os valores, mediana, mínimo, máximo e desvio.

Eu interpreto o máximo como **máximo observado das médias por evento em lote**. Eu não o chamo de pior latência possível de um evento. Eu não escolho o máximo como objetivo principal, porque uma medição isolada pode refletir interrupções do sistema. Eu reconheço que carga e frequência da CPU também afetam as medições, mesmo com seed fixa.

Eu congelo a primeira medição concluída de cada máscara e a reutilizo na retomada. Eu registro ambiente, versões, arquivos e código na identidade do experimento para evitar misturar medições incompatíveis.

## Como eu apresento o Pareto somente em 2D

No gráfico principal, eu coloco **quantidade de atributos no eixo horizontal** e **1 − F1-macro no vertical**. Ambos melhoram em direção ao canto inferior esquerdo. Eu uso a cor para mostrar o tempo de inferência. Eu também apresento erro versus tempo e atributos versus tempo em dois painéis 2D.

Eu não altero os valores para produzir uma barriga. Eu traço segmentos somente entre os pontos não dominados na projeção de dois objetivos. Eu explico que esses segmentos não são soluções intermediárias medidas. Um ponto pode parecer dominado em dois eixos e permanecer não dominado quando considero o tempo.

Para verificar a curvatura, eu normalizo os dois eixos pelos extremos da fronteira projetada e comparo seus pontos com a reta que liga esses extremos. Se existir desvio em direção à origem, eu destaco o maior como candidato geométrico a joelho. Com menos de três pontos, eu informo que não há pontos suficientes. Eu não apresento esse indicador como teste estatístico, prova de convergência ou decisão automática.

Na apresentação anterior, eu mostrava F1 crescente versus atributos, misturando um eixo a maximizar e outro a minimizar. O sentido visual diferente não comprovava um erro do algoritmo. Agora eu explicito dois eixos a minimizar para facilitar a leitura solicitada.

## Como eu acompanho a busca

Eu salvo a fronteira ao fim de cada geração. Eu calculo o hipervolume dos três objetivos, embora os gráficos de Pareto sejam exclusivamente 2D.

Para o HV, eu transformo o tempo por `t / (t + t_ref)`, usando como referência fixa a mediana do baseline medida antes da busca. A transformação é monotônica, fica abaixo de 1 e não corta tempos maiores que o baseline. Eu uso o ponto de referência `(1.1, 1.1, 1.1)`. No NSGA-II, o terceiro objetivo continua em segundos por evento, sem essa transformação.

Eu não suponho que o baseline seja um limite máximo. Eu não normalizo pelo máximo de cada geração, pois isso mudaria a escala do acompanhamento. Eu não comparo esse HV diretamente ao da rodada de dois objetivos.

## Como eu escolho o que vai ao teste

Ao terminar a busca, eu fixo, usando apenas a validação, até quatro representantes: maior F1, menor quantidade de atributos, menor tempo e compromisso. Eu acrescento o baseline e removo máscaras repetidas.

Para o compromisso, eu uso a menor distância à origem dos objetivos `[1 − F1, k/d, t/(t+t_ref)]`, com pesos iguais. Eu registro essa regra antes do teste; ela é uma preferência explícita e pode escolher uma solução diferente do joelho da projeção.

Eu salvo `escolhas_antes_do_teste.json` e só então calculo as métricas de teste. Eu apresento todas essas escolhas e não seleciono uma vencedora olhando o teste. Eu avalio o teste conjunto e cada origem separadamente, com o mesmo modelo.

Nas matrizes, eu apresento contagens e normalização por linha. A diagonal normalizada corresponde ao recall de cada classe. Eu mostro um traço quando não há exemplos daquela classe no painel. Eu incluo tabelas CSV com precisão, recall, F1 e suporte.

## Como eu executo e acompanho

Eu uso uma configuração independente em `src/config_mlp_conjunto.yaml` e saídas em `Resultados/mlp_conjunto_v1/`. Eu executo os comandos a partir da raiz do projeto.

```bash
# Eu preparo todos os dados sem iniciar a busca.
.venv/bin/python -u src/executar_experimento_mlp.py --etapa preparar

# Eu meço o custo do baseline e obtenho sua validação antes da busca longa.
.venv/bin/python -u src/executar_experimento_mlp.py --etapa baseline

# Eu executo ou retomo as 100 gerações, a seleção e o teste final.
.venv/bin/python -u src/executar_experimento_mlp.py --etapa completo
```

Eu posso executar somente a busca com `--etapa busca`. Eu consulto `progresso.json`, os registros em `geracoes/` e o relatório parcial em `figuras/index.html`. Eu confirmo a conclusão de todo o fluxo por `analise.concluida.json`, não apenas pela existência de um gráfico.

Eu posso iniciar somente o baseline em segundo plano antes das 100 gerações:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/ids-mlp-baseline.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start ids-mlp-baseline.service
tail -f Resultados/mlp_conjunto_v1/logs/baseline.log
```

Eu confirmo o término pelo arquivo `Resultados/mlp_conjunto_v1/baseline_validacao.json`
e pelo resultado do serviço. O baseline sozinho não acessa o teste nem inicia as
100 gerações. Depois, o fluxo completo reutiliza seu modelo e suas medições.

Se eu usar o serviço para continuar após fechar o terminal:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/ids-mlp-conjunto.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start ids-mlp-conjunto.service
tail -f Resultados/mlp_conjunto_v1/logs/execucao.log
```

Eu paro com `systemctl --user stop ids-mlp-conjunto.service`. A continuidade após logout depende da configuração de persistência do gerenciador de usuário da máquina. Eu não inicio o terminal e o serviço simultaneamente; o lock recusa uma segunda execução.

Se eu modificar configuração, dados, código científico ou ambiente, eu uso um novo nome de experimento e uma nova pasta. Eu não edito o manifesto para forçar a reutilização.

## O que eu preciso verificar antes da sexta-feira

Eu verifico o fluxo com dados sintéticos, preparo os dados reais e meço o custo do baseline. Eu uso esse custo para discutir o orçamento: população 24 por 100 gerações representa até aproximadamente 2.400 avaliações de indivíduos, com possível reutilização de máscaras. Cada avaliação pode exigir várias épocas sobre milhões de registros.

Eu não prometo que a execução termine na sexta sem medir sua duração. Se a busca ainda estiver em andamento, eu apresento o protocolo, a geração concluída, o Pareto de validação parcial e suas limitações. Eu não apresento resultados sintéticos como científicos, nem consulto o teste antecipadamente para escolher soluções.

## O que eu já validei em 21/09/2026

Eu preparei integralmente os três Parquets e confirmei 34.015.950 registros válidos,
37 atributos comuns e 31 classes no vocabulário conjunto. Eu obtive estas divisões:

| Minha partição | Registros | Proporção observada |
|---|---:|---:|
| Treino | 24.500.089 | 72,03% |
| Validacao | 4.304.724 | 12,66% |
| Teste | 5.211.137 | 15,32% |

Eu auditei todos os hashes dos grupos e encontrei **zero grupos compartilhados**
entre treino, validação e teste. Eu obtive 8.088.577 grupos distintos no treino,
1.730.517 na validação e 1.733.094 no teste. Eu não confundo essa auditoria com
independência temporal ou por sessão. A concentração de cópias em alguns grupos
explica por que as proporções observadas não são exatamente as planejadas.

Eu registrei as contagens por base e classe no
[resumo da preparação](../Resultados/mlp_conjunto_v1/preparacao_resumo.json).
Eu também concluí os 72 testes do projeto, com 97,71% de cobertura dos módulos.
As execuções da MLP nesses testes usaram dados sintéticos; eu ainda não executei
o baseline real nem as 100 gerações no momento deste registro.

Eu revisei o [Pareto anterior em dois painéis](../Resultados/revisao_pareto_anterior/comparacao_eixos.png)
e observei S2 como candidato geométrico a joelho ao mostrar erro versus atributos.
Eu identifico essa figura como resultado da árvore anterior, e não da MLP.
Eu posso reproduzi-la com `.venv/bin/python src/revisar_pareto_anterior.py`.

## Minhas referências de implementação

Eu consulto a [MLP do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), a [padronização incremental](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), as [orientações contra vazamento](https://scikit-learn.org/stable/common_pitfalls.html), o [NSGA-II do pymoo](https://pymoo.org/algorithms/moo/nsga2.html) e os [indicadores multiobjetivo](https://pymoo.org/misc/indicators.html).
