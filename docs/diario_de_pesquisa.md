# Meu diário de pesquisa: decisões, experimentos e descobertas

## Como organizo este registro

Eu mantenho este diário para explicar aos meus orientadores como desenvolvo
os experimentos e que evidências utilizo na dissertação. Eu separo o que
**decidi e implementei**, o que **observei** e o que **ainda pretendo investigar**.
Não trato uma hipótese como resultado, nem um orçamento de execução como prova
de convergência.

Eu iniciei este diário em **7 de setembro de 2026**, no horário de Brasília.
Nesta primeira entrada, reconstruo retrospectivamente decisões discutidas ao
longo da preparação. Quando não disponho da data original ou de uma justificativa
contemporânea, eu explicito essa limitação em vez de inventar uma cronologia.
Nas próximas entradas, vou registrar a decisão no momento em que a tomar e
preservar as entradas anteriores, acrescentando correções identificadas.

Eu uso o [protocolo com três datasets](experimento_tres_datasets.md) para a
descrição do método e este diário para acompanhar sua evolução. Eu reúno as
evidências desta entrada no [registro de 7 de setembro](registros/2026-09-07_tres_datasets.json).
Esse registro é uma fotografia do estado observado, não um painel atualizado
em tempo real.

## Entrada 001 — 07/09/2026: minha evolução de duas para três bases

### D01 — Por que mantive inicialmente 15 gerações

**Minha decisão adotada:** eu mantive população 24, 15 gerações e seed 42 na
primeira rodada com três bases, preservando esses parâmetros da configuração
anterior.

**Minha justificativa:** eu quis iniciar a ampliação sem mudar simultaneamente
a quantidade de bases, a população, a seed e o orçamento de gerações. Eu trato
15 gerações como um orçamento exploratório, correspondente a aproximadamente
360 avaliações do NSGA-II, incluindo possíveis máscaras repetidas.

**O que eu não consigo justificar retrospectivamente:** eu não tenho, nos
registros consultados, uma análise que demonstre por que 15 foi escolhido
originalmente ou que esse valor seja ótimo. Minha justificativa documentada
refere-se à manutenção desse orçamento na ampliação para três bases. Eu não
atribuo esse número a uma exigência do NSGA-II ou a uma conclusão da literatura.

**Evidência que me fez questionar o orçamento:** na rodada concluída com duas
bases, observei aumento do hipervolume de 0,100282 para 0,170428. A última geração
ainda acrescentou aproximadamente 0,003375. Por isso, eu não afirmo que a busca
estabilizou ao atingir 15 gerações.

**Consequência para minha dissertação:** eu vou relatar o orçamento e o
comportamento observado separadamente. Manter os parâmetros facilita comparar
as configurações, mas não torna idênticas as tarefas de duas e três bases.

**Minha próxima questão, ainda sem decisão implementada:** depois de examinar
a curva da rodada atual, vou discutir com meus orientadores se ampliarei o
orçamento e se realizarei outras seeds. Eu ainda não defini um limiar de
estabilização, uma janela de gerações ou um novo limite de execução. Antes de
uma nova rodada, pretendo registrar esses critérios, o custo disponível e se
estou fazendo uma análise exploratória ou confirmatória.

Eu vinculo esta decisão à [configuração da rodada anterior](../src/config.yaml),
à [configuração com três bases](../src/config_tres_datasets.yaml) e aos
[valores de hipervolume do experimento concluído](../Resultados/figuras/orientadores_final/convergencia_nsga2.csv).

### D02 — Como ampliei a pergunta experimental

**Minha decisão adotada:** eu incluí NF-CSE-CIC-IDS2018-v2 junto de
NF-UNSW-NB15-v2 e NF-ToN-IoT-v2 na otimização e na avaliação.

**Minha justificativa:** eu quis investigar se a seleção de atributos mantém
um compromisso útil ao considerar uma terceira base da família NetFlow v2.
Eu preservei as bases separadas e passei de duas para seis transferências:
UNSW→ToN, ToN→UNSW, UNSW→CSE, CSE→UNSW, ToN→CSE e CSE→ToN.

**Alternativa que considerei:** inicialmente, considerei reservar a terceira
base apenas como teste externo. Ao decidir incluí-la na otimização, mudei a
pergunta desta rodada. Eu já não posso apresentá-la como teste externo
independente. Para investigar essa condição, precisarei selecionar em duas
bases e avaliar na terceira sem usá-la na seleção ou no ajuste de parâmetros.

Eu detalho esse desenho no [protocolo experimental](experimento_tres_datasets.md).

### D03 — Como mantive dois objetivos com três datasets

**Minha decisão adotada:** eu mantive a formulação biobjetivo:

```text
f1 = 1 - média dos seis F1-macro direcionais
f2 = número de atributos selecionados / número de atributos comuns
```

**Minha justificativa:** eu atribuí o mesmo peso a cada direção, para que uma
base maior não determine diretamente o peso da média. Eu preservei os seis
F1 individuais para investigar perdas que uma média agregada pode esconder.

**Minha limitação de interpretação:** eu reconheço que a composição de `f1`
mudou em relação à rodada de duas bases. Mesmo mantendo escala e referência,
eu não comparo diretamente seus hipervolumes como se medissem a mesma tarefa.

Eu implemento essa formulação em [Modulos/otimizacao.py](../Modulos/otimizacao.py).

### D04 — Como preparei e conferi os arquivos de dados

**Minha decisão adotada:** eu converti o CSV oficial do CSE diretamente do ZIP
para Parquet em lotes, sem subamostragem, e conferi 18.893.708 registros em ambos
os formatos. Eu mantive os números em `float64` no arquivo convertido e usei
`float32` nas features após a limpeza, para reduzir o consumo de memória.
Eu preservei os hashes SHA-256 de origem e destino no registro local de conversão.

**O que observei no carregamento real:** eu obtive 37 atributos comuns e as
seguintes contagens:

| Minha base | Linhas do arquivo | Linhas que removi | Linhas válidas que utilizei |
|---|---:|---:|---:|
| NF-UNSW-NB15-v2 | 1.986.745 | 0 | 1.986.745 |
| NF-ToN-IoT-v2 | 13.135.881 | 131 | 13.135.750 |
| NF-CSE-CIC-IDS2018-v2 | 18.893.708 | 253 | 18.893.455 |
| Total | 34.016.334 | 384 | 34.015.950 |

Eu removi registros com valores ausentes, infinitos ou fora do intervalo
numérico admitido. Eu não reduzi os arquivos por amostragem. Essas contagens
comprovam o uso dos registros válidos dos arquivos locais; eu não as trato,
isoladamente, como comprovação de correspondência integral com toda versão
publicada desses datasets.

Eu descrevo a preparação em [Datasets/README.md](../Datasets/README.md) e
preservo as contagens no [registro desta entrada](registros/2026-09-07_tres_datasets.json).

### D05 — Por que acrescentei o diagnóstico de convergência

**Minha decisão adotada:** eu acrescentei hipervolume por geração, variação do
hipervolume e uma interpretação escrita ao material dos orientadores. Eu
reconheci que o gráfico de Pareto, sozinho, não demonstrava a evolução da busca.

**Meu procedimento:** eu reconstruí as gerações consultando somente o cache,
com os mesmos parâmetros, e conferi a fronteira reconstruída contra a salva.
Eu utilizei referência fixa `(1.1, 1.1)` e os objetivos em `[0,1]`, sem
normalização distinta a cada geração. Eu medi a fronteira sobrevivente, não
um conjunto acumulado de todas as soluções visitadas.

**Minha interpretação:** eu interpreto aumento do hipervolume como progresso
nos objetivos definidos. Não o trato como prova de ótimo global, estabilidade
entre seeds ou generalização em dados externos.

Eu vinculo este procedimento ao [código do diagnóstico](../src/diagnostico_nsga2.py)
e ao [PDF do experimento concluído](../Resultados/figuras/orientadores_final/relatorio_graficos.pdf).
Eu consultei a [análise de convergência do pymoo](https://pymoo.org/getting_started/part_4.html)
como referência metodológica, sem atribuir a ela a escolha específica de 15 gerações.

### D06 — Como limitei minhas conclusões sobre taxonomia e domain shift

**Minha decisão adotada:** eu preservei a definição atual do F1 cross e
explicitei que classes do destino ausentes no treino são agrupadas no código
`-1`. Eu não interpreto esse cálculo como macro-F1 individual de todas as
categorias desconhecidas originais.

**Correção que fiz na apresentação:** eu substituí a atribuição automática de
erros em classes conhecidas a “domain shift” por uma descrição do erro nesse
grupo. Eu reconheço que a partição não isola causalmente domain shift e não
constitui uma decomposição do F1.

**Questão que deixo em aberto:** se eu estudar outra definição de F1, uma tarefa
binária ou uma taxonomia harmonizada, vou registrar outro protocolo e não
misturar os resultados ou caches com os desta rodada.

Eu explico esses limites nas [notas do relatório anterior](../Resultados/figuras/orientadores_final/LEIA-ME.md).

### D07 — Como mantive continuidade e rastreabilidade

**Minha decisão adotada:** eu utilizei `systemd` com reinício após falhas e
persistência após logout. Eu separei as saídas em `Resultados/tres_datasets_v1/`
e identifiquei o experimento pela configuração e pelos hashes dos Parquets.

**Minha justificativa:** eu quis preservar a rodada de duas bases e evitar
reaproveitar um cache agregado cujo objetivo mudou. Eu inicio uma nova busca
para seis direções. Eu uso checkpoints na Fase 1 e salvo cada solução detalhada
concluída na Fase 2, antes de seguir para a próxima.

**O que verifiquei:** eu testei as seis direções, a média dos objetivos, a
separação de identidades, a preparação dos dados e a reconstrução do Pareto.
Também executei o fluxo completo com dados sintéticos até baseline e gráficos,
e confirmei que uma segunda execução reconheceu a conclusão sem recomputar.
Eu interpreto esses testes como verificação do software, não como validação
das hipóteses científicas.

Eu utilizo [executar_experimento.py](../src/executar_experimento.py) e a
[unidade do serviço](../deploy/systemd/ids-tres-datasets.service).

### O que observei no acompanhamento desta entrada

Às **15h49 de 07/09/2026**, eu registrei o serviço ativo, sem reinícios, duas
gerações concluídas e 50 avaliações únicas salvas no cache. Eu ainda não concluí
a otimização com três bases. Eu não confundo as avaliações únicas do cache com
o contador de indivíduos avaliados pelo NSGA-II, pois máscaras podem se repetir.

Na consulta das 14h03, eu havia observado 44 avaliações em aproximadamente
12 horas. A partir desse ritmo, estimei cerca de 100 horas totais para a Fase 1,
mais o tempo da avaliação detalhada. Eu trato essa conta como uma projeção
operacional provisória: custo por máscara, cache e carga da máquina podem
alterá-la. Eu não a utilizo como duração final medida na dissertação.

Eu preservo a configuração, o commit do código, as contagens, a fronteira parcial
e o trecho do log no [registro de evidências](registros/2026-09-07_tres_datasets.json).
Eu identifico o código desta observação pelo commit `b5ae8ed`; esse identificador
antecede a criação do próprio diário.

### Como posso redigir esta decisão na dissertação

> Eu adotei inicialmente população 24, 15 gerações e seed 42, mantendo os
> parâmetros da rodada anterior ao ampliar a análise para três datasets.
> Tratei o número de gerações como um orçamento exploratório, sem pressupor
> convergência. Para examinar a evolução da busca, acompanhei o hipervolume
> com escala e ponto de referência fixos. No experimento anterior, com duas
> bases, observei melhoria também na última geração, o que motivou a discussão
> sobre ampliar o orçamento em investigações posteriores. Na rodada com três
> bases, passei a agregar seis direções de transferência, preservando seus
> resultados individuais para evitar que a média ocultasse diferenças.

Eu considero esse parágrafo uma proposta de redação baseada nos registros
atuais. Antes de incorporá-lo à dissertação, vou ajustá-lo aos resultados finais
e à discussão com meus orientadores.

## Entrada 002 — 07/09/2026: como interpretei o log do algoritmo

Às 17h24, eu confirmei duas gerações concluídas, 56 máscaras únicas salvas e
nenhum reinício por falha. Eu esclareci que a linha do log é atualizada por
geração concluída, enquanto o cache recebe avaliações durante a geração.

Eu consultei as classes de exibição e de variação dos objetivos no pymoo 0.6.2
instalado e documentei `n_gen`, `n_eval`, `n_nds`, `eps` e `indicator` no
[README](../README.md). Eu distingui o número de indivíduos das máscaras únicas
e a mudança indicada por `eps` do hipervolume. Eu também registrei que
`ideal`, `nadir` e `f` seguem uma prioridade de exibição e que a tolerância
interna da tabela não substitui meu critério de parada de 15 gerações.

Eu não alterei parâmetros nem reiniciei a execução ao acrescentar essa
explicação. Eu usei essa leitura para tornar o acompanhamento compreensível
sem afirmar convergência ou melhoria percentual a partir de `eps`.

## Entrada 003 — 07/09/2026: como distingui os mecanismos e a penalização

Eu investiguei se a penalização estava sendo utilizada nos experimentos.
Eu conferi o código de pré-processamento, classificação, avaliação e otimização,
além da configuração de três bases. Eu registrei a explicação completa na
[metodologia atual](experimento_tres_datasets.md#como-aplico-os-mecanismos-da-metodologia-atual).

Eu confirmei a penalização explícita da máscara sem atributos: eu atribuo
`[1, 1]` aos objetivos e não treino o classificador. Para máscaras não vazias,
eu minimizo `1 - média dos seis F1 macro` e `k/37`. Eu esclareci que a quantidade
de atributos é um objetivo separado, sem coeficiente adicional somado ao F1.
Eu não alterei essa formulação durante a execução.

Eu distingui a limpeza de registros inválidos, a exclusão e o alinhamento dos
atributos, a normalização textual dos rótulos e o uso de `float32` da penalização
na busca. Eu também registrei que trato o desbalanceamento com pesos de classe
na árvore, sem subamostrar as bases.

Eu verifiquei que as classes ausentes na origem recebem o código `-1` no teste.
Eu reconheci que esse agrupamento afeta o F1 e não equivale a avaliar cada
categoria desconhecida separadamente. Eu não o interpreto como uma penalização
extra nem como reconhecimento aprendido de classes desconhecidas.

Eu confirmei a existência da regra para máscaras vazias, mas não medi sua
frequência: essas máscaras retornam antes da gravação no cache. Eu não deduzo
sua ausência na busca a partir da ausência no cache. Eu também não atribuo
benefícios empíricos a cada mecanismo sem experimentos comparativos.

Eu mantenho como próximo passo examinar as seis direções, os diagnósticos de
taxonomia, o baseline e a evolução do hipervolume quando os resultados estiverem
disponíveis. Eu acrescentei documentação sem modificar o código ou reiniciar
o serviço. Eu uso como referência de implementação o commit `b5ae8ed` e a
configuração `src/config_tres_datasets.yaml`.

## Entrada 004 — 21/09/2026: minha nova orientação com MLP

Eu recebi do Eduardo a orientação de passar para 100 gerações, manter seed 42,
unir as três bases no treinamento, usar uma MLP com três camadas de duas vezes
a quantidade de atributos e acrescentar o tempo de inferência por evento.
Eu também preciso rever a leitura do Pareto e apresentar resultados por classe.

Eu implementei um protocolo separado, com treino/validação/teste aproximados de
70%/15%/15%, agrupando entradas idênticas antes da divisão. Eu interpretei o
tamanho das camadas como `2k`, dependente da seleção. Eu registrei os parâmetros
adicionais como decisões de implementação, e não como escolhas expressas pelo orientador.

Eu apresento somente gráficos de Pareto 2D, conforme solicitado. Eu verifico o
sentido dos eixos e um candidato geométrico a joelho; eu não garanto uma barriga.
Eu preservo os registros da árvore e não comparo diretamente seu hipervolume
com o novo indicador de três objetivos.

Eu detalho os procedimentos e limites no [novo protocolo](experimento_mlp_conjunto.md)
e no [roteiro para orientação](roteiro_mlp_conjunto.md). Eu registro a conclusão
científica da rodada somente depois da execução real, separando-a dos testes de software.

## Entrada 005 — 23/10/2026: como passei a representar os três objetivos em 2D

**Pergunta que investiguei.** Eu me perguntei se um gráfico de Pareto em 3D
representaria melhor os três objetivos da MLP (erro `1 − F1-macro`, proporção
de atributos e tempo por evento) e se fazia sentido gerar o Pareto de erro e o
de tempo em figuras separadas.

**O que observei antes da mudança.** No gráfico erro × atributos, eu ligava só
os pontos não dominados daquela projeção e mostrava o tempo apenas na cor.
Soluções rápidas, como S2, S5 e S8, pareciam dominadas e, portanto, ruins, embora
continuem não dominadas nos três objetivos. Na figura de tempo, os dois painéis
mostravam só pontos azuis iguais, sem fronteira, sem o objetivo omitido e sem o
baseline. Assim, as três projeções não eram lidas da mesma forma.

**Alternativas que considerei.**

1. Pareto 3D estático: eu o rejeitei para PNG/PDF. A perspectiva impede ler
   valores, os rótulos se sobrepõem e uma superfície exigiria interpolar entre
   soluções, o que contraria meu critério de não suavizar nem sugerir resultados
   intermediários medidos. Com o tempo num eixo, eu também daria a ele o mesmo
   peso visual que ao erro, embora sua amplitude seja pequena (≈0,65–1,6 µs/evento).
2. Pareto 3D interativo no `index.html`: eu o considerei útil para explorar,
   mas não para a dissertação impressa. Eu não o adotei nesta entrada.
3. Matriz única com as três projeções: eu a rejeitei porque os painéis ficariam
   pequenos para 22 rótulos e para o diagnóstico de joelho.
4. Coordenadas paralelas: eu constatei que já as gero em
   `figuras_extras/g1_coordenadas_paralelas`, com os três objetivos e mais seis
   métricas; eu não criei uma versão reduzida a três eixos.

**Decisão que tomei.** Eu mantive somente gráficos 2D e as duas figuras
separadas: a figura 01 (erro × atributos) continua como principal, porque é o
trade-off central e é nela que calculo o candidato a joelho. Eu passei a usar a
mesma codificação nas três projeções:

- a cor mostra o objetivo que ficou fora dos eixos, com uma única matiz fixa por
  objetivo: azul para tempo, verde para atributos e roxo para erro. Mais escuro
  é sempre pior, e a barra de cores diz “Cor = <objetivo>, mais escuro = pior”;
- círculos estão na fronteira daquela projeção, ligados por segmentos;
- losangos ficam fora da fronteira projetada, mas não são dominados nos três
  objetivos. Eles vencem, no objetivo da cor, as soluções que os dominam na
  projeção;
- um “x” cinza marcaria uma solução dominada nos três objetivos, se houvesse;
- o baseline aparece como X cinza nos três painéis;
- a legenda de cada painel explica as cores dos pontos: dois quadrados mostram
  o tom claro e o escuro, com o nome do objetivo e o valor de cada extremo
  (por exemplo, “Cor clara: tempo = 0.65 µs/evento (melhor)”). Eu acrescentei
  isso porque só a barra de cores não deixava claro, à primeira leitura, o que
  a cor dos pontos representava. Na figura 02, eu pus a legenda abaixo de cada
  painel para não cobrir pontos;
- cada painel tem um título no formato “Y × X — cor: objetivo”, por exemplo
  “Erro × atributos — cor: tempo”. Eu troquei o título genérico da figura 01
  porque ele não dizia o que estava em cada eixo, e isso levou a uma leitura
  errada (atributos × tempo em vez de atributos × erro).

Eu troquei o `viridis` por matizes únicas porque cada barra de cores agora
representa um objetivo diferente; com o mesmo mapa, a mesma cor significaria
coisas diferentes em cada painel.

**Justificativa e evidências.** Na geração 72/100 (relatório PARCIAL, seed 42,
somente validação), as 22 soluções salvas são não dominadas nos três objetivos.
Nas projeções, 15 estão na fronteira erro × atributos, 11 na fronteira
erro × tempo e só 3 (S1, S2 e S5) na fronteira atributos × tempo. S11 e S18
não estão em nenhuma das três fronteiras projetadas e continuam não dominadas:
elas equilibram os três objetivos. Esse caso mostra que nenhuma projeção 2D,
sozinha ou em conjunto, substitui a dominância nos três objetivos, e justifica
marcar a dominância tridimensional explicitamente em todos os painéis.

**Configuração, experimento e código.** `src/config_mlp_conjunto.yaml`,
`Resultados/mlp_conjunto_v1/`. Eu alterei `src/relatorio_mlp.py`: criei
`nao_dominados` e `painel_pareto` e reescrevi as figuras 01 e 02 sobre essa
função. Acrescentei `test_dominancia_em_tres_objetivos` em
`tests/test_mlp_conjunto.py`. A mudança parte do commit `a67b962` e ainda não
estava commitada quando a registrei. Ela altera somente a apresentação: não
muda objetivos, busca, seleção ou medições.

**Limitações que reconheci.** Os números acima são parciais e mudarão até a
geração 100. Soluções com o mesmo número de atributos têm a mesma arquitetura
(três camadas de `2k` neurônios), mas tempos bem diferentes: com seis atributos,
S9 tem 1,42 µs/evento e S10, 0,83 µs/evento. Eu atribuo essa diferença
principalmente às condições da medição, e não ao modelo, e por isso não
interpreto diferenças pequenas de tempo entre soluções de tamanho parecido. A fronteira projetada depende de empates: com atributos e erro
iguais, eu mantenho a primeira solução pela ordem. A distinção entre círculo e
losango é geométrica e não indica qual solução é melhor. A execução em andamento
importou o módulo antigo ao iniciar; até ela terminar ou ser reiniciada, a pasta
`figuras/` continua recebendo a codificação anterior, e eu preciso regenerar o
relatório depois.

**Próximo passo.** Regenerar o relatório ao fim das 100 gerações, conferir de
novo quantas soluções ficam fora de todas as fronteiras projetadas e decidir com
o orientador se vale incluir um 3D interativo apenas como material de apoio.

**Arquivos que preservo como evidência.** `src/relatorio_mlp.py` e a cópia
das figuras da geração 72, reconstruída de `geracoes/072.json`, em `Resultados/revisao_pareto_tres_objetivos/`
(figuras 01 e 02 em PNG/PDF, com LEIA-ME). Eu salvei essa cópia à parte
porque a execução em andamento sobrescreve `mlp_conjunto_v1/figuras/`.

**Discussão com os orientadores.** Ainda não discuti esta mudança com eles.

## Entrada 006 — 23/10/2026: o que os gráficos mostram na geração 73 e o problema de peso

**Pergunta que investiguei.** Eu precisei explicar aos orientadores, de forma
rápida, o que cada gráfico mostra sobre os três objetivos da dissertação e o
que o desempenho por base revela sobre a detecção do tráfego benigno.

**Decisão que tomei.** Eu congelei a geração 73/100 (PARCIAL, seed 42,
somente validação) e gerei com o mesmo estado as figuras 01, 02, g1 e g3 e a
tabela `metricas_solucoes.csv` em `Resultados/mlp_conjunto_v1/orientadores_g073/`.
Eu reconstruí o estado com as soluções de `geracoes/073.json` e o histórico de
`progresso.json` até essa geração. Eu escrevi cinco mensagens curtas em
[explicacoes_rapidas_orientadores.md](explicacoes_rapidas_orientadores.md),
no formato do WhatsApp.

**Resultado que observei (geração 73).**

- As 22 soluções são não dominadas nos três objetivos. Nas projeções, 16 estão
  na fronteira erro × atributos, 12 na erro × tempo e 3 na tempo × atributos.
  S10 e S18 não estão em nenhuma fronteira 2D.
- O erro cai rápido até cerca de oito atributos (F1-macro de 0,102 com 2
  atributos para 0,458 com 8) e depois a curva achata: de 17 para 21 atributos,
  o F1 vai de 0,544 para 0,551. O candidato geométrico a joelho é S11, com 7
  atributos.
- Cinco soluções dominam o baseline (37 atributos, F1 0,516, 1,63 µs/evento):
  S17, S19, S20, S21 e S22. A de maior F1, S22, tem F1 0,551 com 21 atributos
  (−43%) e 1,32 µs/evento (−19%). A de compromisso, S12, tem 8 atributos (−78%)
  e 0,84 µs/evento (−48%), com perda de 0,058 de F1.
- Por base, S22 supera o baseline em UNSW (0,43 vs 0,39) e CIC (0,72 vs 0,65)
  e fica próxima em ToN (0,56 vs 0,58).
- O hipervolume está perto de 0,50 desde a geração 40 (0,4979 na 40; 0,4997 na
  73; máximo de 0,5014 na 65).

**O problema de peso que identifiquei.** No g3, o TPR (ataque classificado como
algum ataque) fica entre 0,97 e 1,00 em todas as bases e soluções, mas o TNR
(benigno reconhecido como benigno) é baixo: de 0,04 a 0,46 na validação
conjunta e de 0,005 a 0,32 em ToN. Na validação conjunta, cerca de metade dos
alertas é falso alarme: 52% para S22 e 53% para o baseline.

Minha hipótese, ainda não testada, é o peso por classe no treino
(`pesos_balanceados: true`, `pesos = n / (31 × contagem)` em
`Modulos/modelo_mlp.py`). O benigno é 63,5% dos 24.500.089 exemplos de
treino, mas recebe 1/31 (3,2%) do peso total. Cada exemplo benigno pesa 0,051,
enquanto um exemplo de `worms`, a classe mais rara (130 exemplos), pesa cerca de
6.079. Errar benigno como ataque custa pouco para a rede. O F1-macro também
atribui ao benigno 1/31 do objetivo, embora os falsos alarmes reduzam
indiretamente a precisão das classes de ataque. Como o baseline com 37
atributos tem o mesmo comportamento, eu atribuo o problema ao protocolo de
treino e não à seleção de atributos. A proporção de benigno no treino varia por
base: 96,2% em UNSW, 26,1% em ToN e 88,7% em CIC.

**Alternativas que considerei, sem adotar.** Pesos suavizados (por exemplo,
raiz quadrada da razão), um peso maior para o benigno, ou a inclusão da taxa
de falso alarme como objetivo ou restrição. Qualquer uma delas exige um novo
experimento, com nova pasta, sem misturar com `mlp_conjunto_v1`.

**Limitações que reconheci.** Os resultados são de validação, com uma seed,
numa geração intermediária. O TPR não exige acertar o tipo do ataque. A
relação entre os pesos e o TNR baixo é uma hipótese compatível com os dados,
não uma causa demonstrada. O script dos gráficos complementares
(`src/graficos_mlp_extras.py`) ainda não estava versionado quando gerei estas
figuras.

**Próximo passo.** Enviar as mensagens aos orientadores, discutir se testamos
outro esquema de pesos num novo experimento e repetir esta análise na geração 100.

**Arquivos que preservo como evidência.**
`Resultados/mlp_conjunto_v1/orientadores_g073/` (figuras 01, 02, g1 e g3 em
PNG/PDF e `metricas_solucoes.csv`), `geracoes/073.json`,
`preparacao_resumo.json` (contagens por classe) e
`docs/explicacoes_rapidas_orientadores.md`.

**Ajuste no g3.** Os painéis da direita não mostravam os valores do eixo Y,
porque compartilham a escala com os da esquerda, e isso gerou dúvida na
leitura. Eu passei a escrever o nome da métrica no eixo Y de cada painel, como
“TPR (0–1, maior é melhor)”, e a repetir os valores nos quatro painéis. Também
registrei que, no painel de F1-macro, a barra “Conjunto” é o F1-macro geral com
as 31 classes, o mesmo do objetivo, e não a média das três bases.

**Discussão com os orientadores.** Mensagens preparadas em 23/10/2026; ainda
sem retorno registrado.

## Entrada 007 — 23/10/2026: o que as curvas de treino revelam sobre as seeds

**Pergunta que investiguei.** Eu quis entender o gráfico de evolução por época
(g6) e sua relação com a seed única do experimento.

**O que o gráfico mostra.** Para cada solução escolhida e o baseline, o painel
esquerdo traz o F1-macro de validação após cada época (uma passada pelos
24.500.089 registros de treino), e o direito, a loss de treino (entropia
cruzada com pesos de classe). O treino para após 5 épocas sem melhora maior que
0,0001 no F1 de validação (máximo de 30) e mantém o modelo da melhor época.
Na geração 73: S22 teve a melhor época na 11 e parou na 16 (cerca de 40 min);
S12, 11 e 16 (24 min); S1, 12 e 17 (20 min); S4, 6 e 11 (12 min); baseline,
6 e 11 (33 min).

**Resultado que observei.**

- O F1 de validação oscila de 0,07 a 0,11 entre épocas consecutivas. S22 foi de
  0,437 na época 10 para 0,551 na 11 e voltou a 0,48 na 16; S4 caiu de 0,233
  para 0,159 de uma época para a seguinte.
- O F1 que atribuo a cada solução é o pico escolhido na mesma validação que
  define o objetivo do NSGA-II; portanto, tende a ser otimista.
- Na S22, a loss continua caindo após a época 11 enquanto o F1 cai, o que
  justifica parar pelo F1 e não pela loss.
- A loss é irregular, com picos nas épocas 7 e 10. Uma hipótese, não testada, é
  a instabilidade causada pelos pesos de classe extremos (Entrada 006).
- S1, com 2 atributos, mantém loss perto de 2 e F1 perto de 0,10: não aprende o
  suficiente.

**Relação com as seeds.** A seed 42 controla quatro coisas no código: a divisão
dos dados (`Modulos/dados_mlp.py`), os pesos iniciais da MLP e a ordem dos
dados por época (`Modulos/modelo_mlp.py`) e o NSGA-II (`Modulos/busca_mlp.py`).
Cada curva é uma única realização. Pela oscilação entre épocas, eu espero que a
variação entre seeds tenha ordem parecida, maior que a diferença entre S22 e o
baseline (0,035) e muito maior que a diferença entre S20, S21 e S22 (menos de
0,01). Com uma seed, eu não afirmo que essas diferenças de F1 sejam reais.

**Proposta que ainda não adotei.** Retreinar as quatro soluções escolhidas e o
baseline com cinco seeds de modelo, mantendo a divisão dos dados, sem repetir a
busca: cerca de 20 treinos e 10 horas. Isso exige separar a seed da divisão da
seed do modelo, num experimento novo. Eu detalhei o desenho em
[proposta_estudo_seeds.md](proposta_estudo_seeds.md) e acrescentei a mensagem 6
em [explicacoes_rapidas_orientadores.md](explicacoes_rapidas_orientadores.md).

**Limitações que reconheci.** A estimativa da variação entre seeds é uma
inferência a partir da oscilação entre épocas, não uma medição. O estudo
proposto não mede a variação da busca nem da divisão dos dados.

**Arquivos que preservo como evidência.**
`Resultados/mlp_conjunto_v1/orientadores_g073/g6_curvas_treino.png`,
`curvas_treino.csv` e `estado_g073.json` na mesma pasta. Eu salvei esse estado
porque `geracoes/` e `progresso.json` não são versionados; com ele, as figuras
da geração 73 podem ser refeitas. Fiz o mesmo para a geração 72 em
`Resultados/revisao_pareto_tres_objetivos/estado_g072.json`.

**Discussão com os orientadores.** Proposta preparada em 23/10/2026; ainda sem
decisão.

## Como vou acrescentar minhas próximas entradas

Eu vou registrar mudanças de dataset, métrica, taxonomia, seed, população,
orçamento de gerações, pré-processamento ou critério de parada antes de iniciar
a rodada correspondente. Também vou acrescentar conclusões de execução,
interrupções relevantes e correções de interpretação. Eu não vou sobrescrever
uma decisão passada para fazê-la parecer prevista desde o início.

Para cada entrada, vou preencher este roteiro:

```text
Entrada e data em que registrei:
Data do evento, se for diferente:
Pergunta que investiguei:
Decisão que tomei (ou proposta que ainda não adotei):
Alternativas que considerei:
Justificativa e evidências que consultei:
Configuração, experimento e commit que utilizei:
Resultado que observei:
Limitações que reconheci:
Próximo passo que pretendo executar:
Arquivos que preservo como evidência:
Discussão com os orientadores, quando ocorrer:
```

Eu ainda não registro neste diário uma aprovação dos orientadores para ampliar
as gerações ou alterar a métrica, porque essa discussão não foi documentada aqui.
