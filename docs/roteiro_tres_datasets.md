# Meu roteiro para explicar a ampliação para três datasets

Eu preservo este documento como registro do experimento com árvore. Em 21/09/2026,
eu iniciei a implementação de um [novo protocolo com MLP e treino conjunto](experimento_mlp_conjunto.md),
com um [roteiro próprio](roteiro_mlp_conjunto.md). Os números abaixo pertencem ao protocolo anterior.


Eu atualizei este roteiro em 17/09/2026, com base no código e nos resultados salvos de `tres_datasets_v1`. Eu confirmei no marcador local a conclusão da análise em **10/09/2026, às 08h21, no horário UTC−3**. Eu concluí esta rodada experimental; ainda considero suas conclusões exploratórias dentro da dissertação.

Eu detalho a leitura de todas as figuras na [seção 6](#6-como-eu-explico-cada-gráfico-durante-a-apresentação), incluindo os quatro painéis da partição do erro, as seis matrizes de S1 e exemplos numéricos para minha fala.

## 1. O problema que estou investigando

Na minha dissertação, eu investigo se consigo selecionar um conjunto menor de atributos de tráfego de rede que ajude um detector de intrusão a funcionar quando muda a base de dados. Um atributo pode representar, por exemplo, a quantidade de bytes ou uma informação do protocolo utilizado na comunicação.

Eu separo duas situações. Na avaliação **intra-dataset**, eu treino e avalio dentro da mesma base, usando divisões diferentes dos registros. Na avaliação **cross-dataset**, eu treino em uma base e avalio em outra. Minha preocupação é entender quanto do desempenho se mantém nessa transferência.

Eu uso o NSGA-II para procurar combinações de atributos que equilibrem dois interesses: melhorar o F1-macro de transferência e usar menos atributos. Eu uso o F1-macro como uma medida que combina precisão e recall por classe e calcula uma média entre as classes consideradas. Eu não interpreto esse valor como a porcentagem de registros classificados corretamente.

## 2. Por que eu passei de duas para três bases

Antes, eu trabalhava com NF-UNSW-NB15-v2 e NF-ToN-IoT-v2. Agora, eu incluí NF-CSE-CIC-IDS2018-v2. Neste roteiro, eu uso UNSW, ToN e CSE como abreviações dessas três bases.

Com duas bases, eu observava duas transferências. Com três, eu observo seis: UNSW→ToN, ToN→UNSW, UNSW→CSE, CSE→UNSW, ToN→CSE e CSE→ToN. Em cada seta, eu treino na base da esquerda e avalio na base da direita. Eu mantenho as bases separadas, sem juntar todos os registros em um único treinamento.

Essa mudança amplia minha pergunta: eu quero saber se uma mesma seleção de atributos consegue equilibrar o desempenho em mais cenários. Eu também consigo observar se uma melhora na média esconde uma perda em determinada direção.

Eu preciso deixar claro que utilizei as três bases durante a seleção dos atributos. Portanto, a terceira base não funciona como teste externo independente nesta rodada. Eu ampliei os cenários usados na busca, mas ainda preciso de outro desenho experimental para investigar uma base que não participou da seleção.

## 3. O que eu mudei no código e por que isso importa

Eu não criei um novo NSGA-II nem um novo classificador. Eu adaptei o fluxo experimental para ampliar a pergunta que consigo investigar. Eu conferi as mudanças da ampliação no commit `b5ae8ed` e distingui os mecanismos novos daqueles que já existiam.

| Parte do projeto que examinei | O que eu fiz | Como isso ajuda minha pesquisa |
|---|---|---|
| [Preparação da terceira base](../src/preparar_terceiro_dataset.py) | Eu converti o CSV do ZIP em Parquet por lotes, conferi as contagens e registrei hashes dos arquivos. | Eu consigo documentar de onde vieram os dados e conferir a conversão sem carregar todo o CSV de uma vez. |
| [Configuração de três bases](../src/config_tres_datasets.yaml) | Eu acrescentei o CSE e defini pastas próprias para esta rodada. | Eu preservo a execução anterior e identifico qual configuração produziu cada resultado. |
| [Preparação e identidade do experimento](../Modulos/experimento.py) | Eu centralizei o carregamento, registrei as linhas removidas e identifiquei o protocolo pela configuração e pelos arquivos. | Eu consigo conferir o conjunto usado e evitar reaproveitar resultados de outro experimento. |
| [Otimização](../Modulos/otimizacao.py) | Eu retirei a restrição de exatamente duas bases e passei a exigir `n × (n − 1)` resultados direcionais. Com três bases, eu agrego seis F1. | Eu faço a seleção considerar todas as transferências previstas. |
| [Avaliação](../Modulos/avaliacao.py) | Eu aproveitei o laço que já percorria os pares de bases e mantive cada direção registrada separadamente. | Eu consigo examinar perdas que a média não mostra. Eu não apresento esse laço como uma criação nova da ampliação. |
| [Execução completa](../src/executar_experimento.py) | Eu encadeei otimização, avaliação das soluções, comparação com todos os atributos e relatórios, com retomada. | Eu reduzo a necessidade de executar etapas manualmente e preservo avaliações já concluídas. |
| [Relatório](../src/relatorio_orientadores.py) | Eu adaptei a apresentação para seis direções e três bases na comparação intra/cross. | Eu consigo mostrar os resultados de forma consistente com o novo experimento. |

Eu mantive a limpeza de registros inválidos e o alinhamento dos atributos na mesma ordem. Eu também mantive a exclusão das colunas de rótulo dos atributos e o descarte configurado de portas e TTL. Eu não atribuo a cada escolha um benefício isolado, porque ainda não fiz comparações que retirem um mecanismo por vez.

Eu mantenho testes com bases sintéticas para conferir seis transferências, três avaliações intra, a média dos objetivos, a identidade dos arquivos, a retomada e o relatório. Eu uso esses testes para verificar o software; os resultados científicos que apresento abaixo vêm da execução com os arquivos completos.

## 4. Como eu realizei esta rodada

Eu utilizei **34.015.950 registros válidos e 37 atributos comuns**, sem subamostragem. Eu conferi estas contagens no registro local de pré-processamento e no [registro de acompanhamento](registros/2026-09-07_tres_datasets.json):

| Base que utilizei | Registros originais | Registros removidos | Registros válidos |
|---|---:|---:|---:|
| UNSW | 1.986.745 | 0 | 1.986.745 |
| ToN | 13.135.881 | 131 | 13.135.750 |
| CSE | 18.893.708 | 253 | 18.893.455 |
| Total | 34.016.334 | 384 | 34.015.950 |

Eu removi registros com valores ausentes, infinitos ou fora do intervalo numérico admitido. Eu uso `float32` nos atributos após a limpeza para reduzir a memória necessária. Eu não interpreto o volume de dados, sozinho, como garantia de qualidade ou generalização.

Eu mantive uma árvore de decisão com profundidade máxima 8 e pesos de classe balanceados. No NSGA-II, eu usei população 24, 15 gerações e seed 42. Eu mantive esses parâmetros da rodada anterior para não mudar tudo ao mesmo tempo; não tenho evidência de que 15 gerações sejam suficientes para estabilizar a busca.

Eu continuo minimizando dois objetivos:

```text
Meu objetivo de desempenho = 1 − média dos seis F1-macro direcionais
Meu objetivo de redução = quantidade de atributos selecionados / 37
```

Eu atribuo o mesmo peso a cada direção na média. Eu aplico a mesma seleção de atributos em todas as bases, mas treino um classificador na origem de cada transferência. Se a seleção não contém nenhum atributo, eu mantenho a penalização `[1, 1]` e não treino o modelo.

Na primeira fase, eu procuro as seleções com o NSGA-II. Na segunda, eu avalio as quatro soluções finais e a referência com todos os atributos, que chamo de **baseline**. Eu faço validação cruzada estratificada em cinco partes dentro de cada base e registro as seis transferências.

## 5. O que eu obtive até agora

Eu concluí as 15 gerações e as **cinco avaliações detalhadas: quatro soluções e o baseline**. Eu encontrei quatro seleções na fronteira final de Pareto. Eu entendo essa fronteira como os compromissos encontrados pela busca: dentro desse conjunto, eu não melhoro um objetivo sem piorar o outro. Eu não afirmo que encontrei todas as melhores combinações possíveis.

### Minha comparação com todos os atributos

Eu calculei a média dos seis F1 a partir das avaliações salvas. Eu apresento os valores na escala de 0 a 1, arredondados:

| Alternativa que avaliei | Atributos | Redução em relação a 37 | F1-macro cross médio |
|---|---:|---:|---:|
| Baseline | 37 | 0,00% | 0,035290 |
| S1 | 7 | 81,08% | 0,046460 |
| S2 | 8 | 78,38% | 0,058466 |
| S3 | 11 | 70,27% | 0,058526 |
| S4 | 12 | 67,57% | 0,061615 |

Eu observei que as quatro seleções usam menos atributos e apresentam média cross maior que a do baseline nesta rodada. Para S4, eu observo uma diferença de aproximadamente **0,026324 na escala do F1** em relação ao baseline. Eu considero esse resultado favorável à investigação da seleção de atributos, mas reconheço que o desempenho absoluto continua baixo.

Eu vejo S1 como a alternativa mais compacta e S4 como a de maior média cross entre as quatro. Eu também noto que S2 usa oito atributos e fica muito próxima de S3, que usa onze. Eu não declaro uma vencedora sem definir quanto valorizo compactação, desempenho médio e comportamento por direção.

Eu não concluo que menos atributos necessariamente reduzem a latência na mesma proporção. Para sustentar uma vantagem de tempo ou memória, eu precisaria analisar essas medidas com um procedimento de comparação adequado.

### O que eu percebo quando separo as seis direções

Eu comparo abaixo o baseline, S2 e S4 para mostrar por que a média não conta toda a história:

| Direção que avaliei | Baseline | S2 — 8 atributos | S4 — 12 atributos |
|---|---:|---:|---:|
| UNSW→ToN | 0,016712 | 0,023226 | 0,068919 |
| UNSW→CSE | 0,016091 | 0,084913 | 0,096281 |
| ToN→UNSW | 0,077545 | 0,079008 | 0,055070 |
| ToN→CSE | 0,030505 | 0,049755 | 0,031116 |
| CSE→UNSW | 0,058885 | 0,085746 | 0,085172 |
| CSE→ToN | 0,012004 | 0,028148 | 0,033129 |

Eu observei que S4 melhora cinco direções em relação ao baseline, mas piora ToN→UNSW. Já S2 supera numericamente o baseline nas seis direções nesta execução. Eu ainda não testei a estabilidade dessas diferenças em outras seeds.

Esse resultado me ajuda a justificar por que preservo as métricas individuais. Se eu mostrasse apenas a maior média, deixaria de explicar a perda de S4 em uma das transferências.

### O que eu observo dentro e fora da mesma base

Eu uso S4 como exemplo para comparar o desempenho intra-dataset com o baseline:

| Base em que fiz validação cruzada | F1-macro intra do baseline | F1-macro intra de S4 |
|---|---:|---:|
| UNSW | 0,472851 | 0,407256 |
| ToN | 0,719982 | 0,651292 |
| CSE | 0,619720 | 0,572915 |

Eu observei que S4 melhora a média cross, mas reduz o F1 intra nas três bases. Isso mostra um compromisso relevante: a escolha que favorece minha média de transferência não é necessariamente a melhor dentro de cada base.

Eu vejo valores intra bem maiores que os cross nesse exemplo, mas reconheço que os conjuntos de classes considerados mudam entre as avaliações. Eu não interpreto a diferença como uma medida pura de mudança de distribuição. Eu também não considero essa validação cruzada uma avaliação independente de todo o processo de seleção, porque as bases já participaram da escolha dos atributos.

### O que eu aprendo com as classes desconhecidas

Eu normalizei espaços e letras maiúsculas ou minúsculas dos rótulos, mas não fiz uma equivalência semântica completa entre as categorias de ataque. Quando uma classe do destino não existe no treinamento, eu a agrupo no código `-1` para calcular o F1. O classificador não aprendeu a prever esse código.

Eu observei que aproximadamente **67,48% dos registros de ToN têm rótulos ausentes na origem UNSW**. Quando a origem é CSE e o destino é ToN, essa proporção chega a **72,58%**. Eu estou falando de proporções de registros, não de porcentagens de categorias.

Esses números me ajudam a discutir uma dificuldade do protocolo: parte dos registros pertence a classes que o modelo não viu no treino. Eu não atribuo toda a queda de desempenho a isso, nem chamo todo erro nas classes conhecidas de mudança de distribuição. Eu também não interpreto o agrupamento em `-1` como avaliação individual de cada categoria desconhecida ou como detecção aprendida de ataques novos.

### O que a evolução da busca me mostra

Eu acompanhei o hipervolume, que resume o conjunto de compromissos nos dois objetivos. Com a mesma escala e referência durante esta rodada, eu observo progresso quando ele aumenta.

Eu registrei **0,106119 na geração 1 e 0,146452 na geração 15**. Da geração 14 para a 15, eu ainda observei um aumento de aproximadamente **0,000254**. Eu consultei o histórico reconstruído pelo cache e validado contra o Pareto salvo.

Eu concluo que houve progresso nos objetivos, mas não tenho base para afirmar estabilização, ótimo global ou repetibilidade com outras seeds. Eu também não comparo diretamente esse hipervolume ao da rodada de duas bases: agora meu objetivo agrega seis transferências, e a tarefa mudou.

## 6. Como eu explico cada gráfico durante a apresentação

Eu uso esta parte como minha fala ao abrir as figuras. Eu começo explicando os eixos e a legenda, aponto um resultado concreto e digo como ele ajuda a responder minha pergunta de pesquisa. Eu apresento valores arredondados; consulto os arquivos de métricas quando preciso de mais casas decimais.

### Gráfico 00 — Como minha busca evoluiu

Eu abro a [curva de hipervolume](../Resultados/tres_datasets_v1/figuras/00_convergencia_hipervolume.png).

**Como eu leio:** “No eixo horizontal, eu acompanho as 15 gerações. No painel de cima, eu observo o hipervolume, que resume os compromissos encontrados entre desempenho e quantidade de atributos. No painel de baixo, eu vejo quanto esse valor mudou em relação à geração anterior. Eu uso a mesma referência e a mesma escala durante toda a rodada.”

**O que eu aponto:** “Eu comecei com 0,106119 e terminei com 0,146452. Na última geração, eu ainda tive um ganho de aproximadamente 0,000254. Eu vejo progresso, mas não afirmo que a busca parou de melhorar.”

**Como eu relaciono à dissertação:** “Esse gráfico me ajuda a justificar por que trato 15 gerações como um orçamento inicial. Eu preciso investigar outras seeds e um orçamento maior antes de afirmar estabilidade. Eu não interpreto hipervolume como F1, porcentagem de acertos ou prova de generalização externa.”

### Gráfico 01 — Quais compromissos eu encontrei no Pareto

Eu abro o [gráfico de Pareto](../Resultados/tres_datasets_v1/figuras/01_pareto.png).

**Como eu leio:** “No eixo horizontal, eu vejo a quantidade de atributos; mais à esquerda significa uma seleção menor. No vertical, eu vejo o F1-macro médio das seis transferências; mais acima significa maior média. Eu mostro em cinza as máscaras avaliadas no cache e em azul as quatro soluções da fronteira final. Cada ponto representa uma seleção de atributos, não um registro de tráfego.”

**O que eu aponto:** “Eu encontrei S1 com sete atributos e média 0,046460. Em S4, eu uso doze atributos e obtenho 0,061615. Eu também observo que S2, com oito atributos, fica muito próxima de S3, com onze: as médias são 0,058466 e 0,058526.”

**Como eu relaciono à dissertação:** “Eu tenho alternativas para discutir o custo de acrescentar atributos em troca de desempenho. Eu não escolho S4 automaticamente só por ela estar mais acima. Eu também preciso olhar a redução de atributos e as seis direções. A linha azul apenas liga as soluções encontradas; eu não a interpreto como resultado medido para todas as quantidades intermediárias.”

Eu consulto a tabela da seção 5 para comparar com o baseline de 37 atributos e média 0,035290. Eu não afirmo que esse baseline está desenhado como um ponto específico deste gráfico, pois a figura mostra a busca e sua fronteira.

### Gráfico 02 — Por que eu olho cada direção de transferência

Eu abro o [F1 por direção](../Resultados/tres_datasets_v1/figuras/02_f1_por_direcao.png).

**Como eu leio:** “No eixo horizontal, eu vejo S1 a S4 e a quantidade de atributos de cada solução. Cada cor representa uma das seis transferências. No eixo vertical, eu vejo o F1-macro daquela direção. Eu comparo a mesma cor entre soluções para descobrir como uma transferência mudou.”

**O que eu aponto:** “Na transferência CSE→UNSW, eu obtenho aproximadamente 0,127633 com S3, enquanto S4 fica em 0,085172. Mesmo assim, S4 tem a maior média das seis direções. Isso me mostra que a melhor média não significa o melhor resultado em toda transferência.”

Eu complemento a figura com a tabela do baseline: S2 supera essa referência nas seis direções, enquanto S4 melhora cinco e piora ToN→UNSW. Eu faço essa comparação com as métricas salvas, porque este gráfico apresenta somente S1 a S4.

**Como eu relaciono à dissertação:** “Eu consigo mostrar que a direção importa. Treinar em uma base e avaliar em outra não produz necessariamente o mesmo resultado quando inverto a seta. Por isso, eu preservo os seis valores em vez de apresentar apenas uma média.”

### Gráfico 03 — Como eu explico a partição do erro por grupo de classes

Eu abro a [partição do erro corrigida](../Resultados/tres_datasets_v1/figuras/03_erro_por_grupo.png).

**O que eu mudei na apresentação:** “Antes, eu reunia as 24 barras em um único painel e os nomes ficavam sobrepostos. Agora, eu separei S1, S2, S3 e S4 em quatro painéis com a mesma escala. Eu coloquei os nomes completos em linhas e deixei a origem acima da seta e o destino abaixo. Eu alterei a organização visual; os valores continuam sendo os mesmos.”

**Como eu leio os eixos:** “Em cada painel, eu tenho seis barras, uma por direção de transferência. No eixo vertical, eu vejo frações de todos os registros da base de destino. Por exemplo, 0,20 representa 20% desses registros. Eu não estou lendo porcentagens de tipos de ataque.”

**Como eu explico as cores:** “Em roxo, eu mostro a fração de registros cujas classes não estavam no treinamento. Neste protocolo, eu não consigo acertar essas classes, porque o modelo não aprendeu a prevê-las. Em laranja, eu mostro a fração de todos os registros do destino que pertencem a classes conhecidas, mas foram classificados incorretamente.”

Eu calculo a parte laranja multiplicando a fração de registros de classes conhecidas pela taxa de erro dentro desse grupo. Eu não leio a altura laranja diretamente como a taxa de erro calculada somente entre as classes conhecidas.

Eu uso dois exemplos para tornar a leitura concreta:

| Exemplo que apresento | Parte roxa | Parte laranja | Soma das partes |
|---|---:|---:|---:|
| S1, UNSW→ToN | 67,48% | 23,55% | 91,03% |
| S2, UNSW→CSE | 11,95% | 14,45% | 26,40% |

“Na primeira barra, eu tenho aproximadamente 91,03% de registros classificados incorretamente, juntando os dois grupos. A parte que falta para chegar a 100%, cerca de 8,97%, corresponde aos acertos. Eu não concluo que o F1 seja 0,0897: F1-macro e proporção de acertos são medidas diferentes.”

Eu também comparo UNSW→CSE entre S1 e S2. Eu observo que a parte roxa permanece em 11,95%, enquanto a parte laranja cai de 76,42% para 14,45%. Eu entendo essa repetição do roxo: as classes disponíveis na origem e no destino não mudam quando troco apenas os atributos selecionados. Já os erros nas classes conhecidas podem mudar com a seleção.

**Como eu relaciono à dissertação:** “Esse gráfico me ajuda a separar duas situações que aparecem juntas no erro total: registros de classes ausentes no treinamento e erros nas classes presentes. Eu vejo onde a seleção mudou o comportamento do modelo e onde permaneceu uma limitação do conjunto de rótulos. Eu não apresento essa figura como uma decomposição do F1 nem como prova de que o laranja mede apenas mudança de distribuição.”

### Gráfico 04 — Quais atributos aparecem nas soluções

Eu abro o [mapa dos atributos selecionados](../Resultados/tres_datasets_v1/figuras/04_atributos.png).

**Como eu leio:** “No eixo horizontal, eu vejo os nomes dos atributos que apareceram em pelo menos uma solução. No vertical, eu vejo S1 a S4. Um quadrado azul indica que eu selecionei aquele atributo; um quadrado claro indica que não o selecionei. Eu não estou mostrando intensidade de importância: a informação é presença ou ausência.”

**O que eu aponto:** “Eu encontrei seis atributos nas quatro soluções: `OUT_BYTES`, `RETRANSMITTED_OUT_BYTES`, `TCP_WIN_MAX_IN`, `ICMP_TYPE`, `ICMP_IPV4_TYPE` e `DNS_QUERY_ID`. Em S1, eu acrescento `IN_BYTES` a esse conjunto e chego aos sete atributos da solução mais compacta.”

**Como eu relaciono à dissertação:** “Eu consigo examinar quais atributos se repetem e quais mudam entre os compromissos encontrados. Essa recorrência me dá uma pista para novas análises, mas não prova que esses atributos sejam universalmente importantes. Eu preciso verificar outras execuções e, se quiser medir a contribuição individual, realizar comparações específicas. Eu também não suponho que S2, S3 e S4 sejam apenas S1 com atributos adicionados; as combinações podem trocar atributos.”

### Gráficos 05 — Onde o modelo acerta e confunde as classes

Eu tenho seis matrizes, uma para cada direção. **Eu apresento as matrizes de S1, com sete atributos**, porque o relatório usa a solução mais compacta. Eu não as atribuo a S4 nem ao baseline.

**Como eu leio:** “No eixo vertical, eu vejo a classe verdadeira dos registros da base de destino. No horizontal, eu vejo a classe prevista, apresentada nos rótulos do destino. Eu normalizo cada linha separadamente: uma célula com 0,80 significa 80% dos registros daquela classe verdadeira. Na diagonal, eu vejo acertos; fora dela, eu vejo confusões. Quanto mais escuro o azul, maior a proporção.”

Eu não leio 0,80 em uma célula como 80% de acurácia do modelo inteiro. Eu também não interpreto a ausência de um número escrito como zero exato: o gráfico só anota células com proporção de pelo menos 0,01.

**Como eu explico “sem correspondência”:** “Nessa coluna, eu agrupo previsões de rótulos aprendidos na origem que não existem no vocabulário do destino. Eu não confundo essa coluna com o roxo do gráfico anterior: lá eu observo classes verdadeiras do destino ausentes na origem; aqui eu observo previsões da origem sem um rótulo correspondente no destino.”

Para começar a leitura das seis figuras, eu acompanho a mesma classe, `benign`, e observo sua diagonal:

| Matriz de S1 que eu abro | Registros `benign` do destino previstos como `benign` |
|---|---:|
| [UNSW→ToN](../Resultados/tres_datasets_v1/figuras/05_matriz_NF-UNSW-NB15-v2_para_NF-ToN-IoT-v2.png) | 15,82% |
| [UNSW→CSE](../Resultados/tres_datasets_v1/figuras/05_matriz_NF-UNSW-NB15-v2_para_NF-CSE-CIC-IDS2018-v2.png) | 13,20% |
| [ToN→UNSW](../Resultados/tres_datasets_v1/figuras/05_matriz_NF-ToN-IoT-v2_para_NF-UNSW-NB15-v2.png) | 18,05% |
| [ToN→CSE](../Resultados/tres_datasets_v1/figuras/05_matriz_NF-ToN-IoT-v2_para_NF-CSE-CIC-IDS2018-v2.png) | 51,11% |
| [CSE→UNSW](../Resultados/tres_datasets_v1/figuras/05_matriz_NF-CSE-CIC-IDS2018-v2_para_NF-UNSW-NB15-v2.png) | 78,92% |
| [CSE→ToN](../Resultados/tres_datasets_v1/figuras/05_matriz_NF-CSE-CIC-IDS2018-v2_para_NF-ToN-IoT-v2.png) | 56,92% |

Eu calculo esses percentuais com as contagens da matriz, antes do arredondamento das células na imagem. Eu os interpreto como recall da classe `benign`, não como F1-macro ou desempenho em todas as classes.

Eu também aponto um erro concreto em UNSW→ToN: aproximadamente **96,00% dos registros rotulados como `ddos` no destino foram previstos como `benign` em S1**. Eu confiro que `ddos` não aparece no vocabulário de treino dessa direção. Esse exemplo me ajuda a mostrar por que preciso examinar as classes e não apenas o valor médio.

**Como eu relaciono à dissertação:** “Eu uso as matrizes para localizar as confusões que os números agregados escondem. Eu consigo formular perguntas sobre o comportamento por classe e sobre a correspondência dos rótulos. Eu não uso apenas a linha de tráfego benigno para concluir que o detector reconhece bem os ataques.”

### Gráfico 06 — O que muda quando eu saio da mesma base

Eu abro a [comparação intra/cross](../Resultados/tres_datasets_v1/figuras/06_intra_vs_cross.png).

**Como eu leio:** “Eu tenho um painel por base de destino. No eixo horizontal, eu comparo o baseline e S1 a S4; no vertical, eu vejo o F1-macro. A barra intra representa a média dos cinco folds dentro daquela base. As duas barras cross representam modelos treinados nas outras duas bases e avaliados na base indicada no título do painel.”

“Por exemplo, no painel CSE, eu comparo a validação dentro do CSE com UNSW→CSE e ToN→CSE. Eu não estou lendo CSE→UNSW nesse painel, porque sua base de destino é UNSW.”

Eu explico que as hastes da barra intra mostram o desvio padrão entre folds. Eu não as apresento como intervalo de confiança nem como variação entre seeds do NSGA-II. As barras cross não têm essa avaliação de dispersão entre execuções.

**O que eu aponto:** “No painel ToN, eu observo F1 intra de aproximadamente 0,719982 no baseline e 0,651292 em S4. Para S4, as transferências que chegam a ToN ficam em 0,068919, vindo de UNSW, e 0,033129, vindo de CSE. Eu vejo que obter um valor maior dentro da própria base não garante manter esse desempenho ao transferir o modelo.”

Eu também observo uma exceção à ideia de que selecionar atributos sempre piora o desempenho intra: em CSE, S1 chega a 0,643834, enquanto o baseline fica em 0,619720. Eu não generalizo esse ganho para as outras bases ou seleções.

**Como eu relaciono à dissertação:** “Eu uso esse gráfico para mostrar por que preciso avaliar a transferência diretamente. A avaliação dentro de uma base responde a uma pergunta diferente. Como as bases participaram da seleção dos atributos e os rótulos considerados variam, eu trato esses resultados como diagnóstico do protocolo atual, não como teste externo independente.”

### Como eu conecto as figuras em uma única explicação

“Eu começo mostrando que a busca progrediu, mas ainda não demonstrei estabilidade. Depois, eu apresento os compromissos de Pareto e mostro que a maior média não vence em toda direção. Eu uso a partição do erro e as matrizes para entender melhor as falhas, e o mapa de atributos para mostrar quais combinações encontrei. Por fim, eu comparo intra e cross para discutir o limite da transferência. Assim, eu conecto os gráficos à minha pergunta de pesquisa, em vez de apenas descrever barras e cores.”

## 7. Como esse trabalho contribui para minha dissertação

Eu organizo minha contribuição atual em três partes.

**Na metodologia**, eu passei a investigar uma mesma seleção de atributos em seis transferências, mantendo as bases separadas e os resultados por direção. Eu consigo explicar melhor o que significa buscar um compromisso entre compactação e desempenho em diferentes cenários.

**Nos resultados**, eu encontrei seleções de 7 a 12 atributos que superam a média cross da referência com 37 atributos nesta rodada. Ao mesmo tempo, eu mostrei que maior média pode esconder perdas direcionais e que melhorar a transferência pode acompanhar uma queda intra-dataset. Eu uso essas observações para discutir os limites da seleção, não apenas seus ganhos.

**Na organização da pesquisa**, eu passei a registrar configuração, identidade dos arquivos, contagens de limpeza, avaliações e gráficos de forma ligada ao experimento. Isso me ajuda a explicar de onde vieram os números e a repetir ou ampliar o procedimento. Eu reconheço essa organização como suporte à pesquisa, sem confundi-la com uma inovação no algoritmo.

Minha conclusão atual é que encontrei evidências exploratórias de que a seleção pode produzir conjuntos mais compactos e melhorar o critério médio de transferência usado na busca. Eu ainda não demonstrei generalização para uma base independente, superioridade sobre outros métodos de seleção ou desempenho suficiente para uso em uma rede real.

## 8. O que eu pretendo investigar depois

Eu pretendo discutir com meus orientadores os próximos passos antes de iniciar novas rodadas:

1. Eu quero definir um orçamento maior e um critério de acompanhamento da estabilização, porque a busca ainda melhorava ao encerrar.
2. Eu quero repetir o experimento com outras seeds para verificar a variação das soluções e dos resultados. Eu não trato cinco folds de uma avaliação intra como cinco execuções independentes do NSGA-II.
3. Eu quero selecionar em duas bases e avaliar na terceira sem usá-la na seleção ou no ajuste. Eu posso alternar qual base fica de fora para investigar essa pergunta com as três bases disponíveis.
4. Eu quero comparar a seleção com alternativas simples, como seleções aleatórias com a mesma quantidade de atributos. Assim, posso investigar o que ganho com o procedimento de busca além da redução do número de atributos.
5. Eu quero examinar uma taxonomia harmonizada ou uma tarefa binária em protocolos separados. Se eu mudar a definição da tarefa ou da métrica, vou gerar novos resultados e evitar misturá-los com os desta rodada.

Eu ainda considero esses itens propostas de continuidade, não experimentos já realizados.

## 9. Como eu encerraria minha apresentação

“Com a inclusão da terceira base, eu passei a avaliar seis transferências e concluí uma primeira rodada com quatro seleções de atributos. Eu encontrei alternativas mais compactas e com média cross maior que a referência com todos os atributos. Porém, os valores ainda são baixos, e a melhora não acontece da mesma forma em todas as direções. Para minha dissertação, esse trabalho me ajuda a mostrar tanto o potencial da seleção quanto os limites do protocolo atual. Meu próximo passo é investigar a estabilidade desses resultados e separar melhor a seleção dos atributos da avaliação em uma base independente.”

## Onde eu consulto as evidências durante a apresentação

Eu apoio as tabelas deste roteiro nas [cinco avaliações detalhadas](../Resultados/tres_datasets_v1/metricas/pareto_20260910_053058_decision_tree/), no [resumo das soluções](../Resultados/tres_datasets_v1/figuras/resumo_solucoes.csv) e no [histórico de hipervolume](../Resultados/tres_datasets_v1/figuras/convergencia_nsga2.csv). Eu uso S1, S2, S3 e S4 para os arquivos de solução 00, 01, 02 e 03, respectivamente.

Eu mostraria o [Pareto](../Resultados/tres_datasets_v1/figuras/01_pareto.png) ao explicar compactação e desempenho, o [F1 por direção](../Resultados/tres_datasets_v1/figuras/02_f1_por_direcao.png) ao discutir as diferenças entre transferências, a [comparação intra/cross](../Resultados/tres_datasets_v1/figuras/06_intra_vs_cross.png) e a [curva de hipervolume](../Resultados/tres_datasets_v1/figuras/00_convergencia_hipervolume.png).

Eu também posso abrir o [relatório completo em PDF](../Resultados/tres_datasets_v1/figuras/relatorio_graficos.pdf). Eu mantenho o [protocolo](experimento_tres_datasets.md) e o [diário](diario_de_pesquisa.md) como registros da evolução do trabalho; as observações datadas de 07/09 descrevem o acompanhamento daquela data.
