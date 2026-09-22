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
