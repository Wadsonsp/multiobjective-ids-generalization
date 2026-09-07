# Minha investigação com três datasets NetFlow v2

## Onde acompanho minhas decisões

Eu mantenho o histórico das escolhas, suas justificativas e as observações
posteriores no [meu diário de pesquisa](diario_de_pesquisa.md). Eu uso este
protocolo para descrever o desenho experimental e o diário para explicar sua
evolução, incluindo a manutenção inicial de 15 gerações e as limitações dessa
escolha. Eu não reinterpreto uma decisão exploratória como convergência comprovada.

## O que pretendo investigar

Na minha dissertação, procuro entender se a seleção multiobjetivo de atributos
me ajuda a transferir um detector de intrusão entre redes diferentes. Depois
de concluir o experimento com NF-UNSW-NB15-v2 e NF-ToN-IoT-v2, pretendo incluir
NF-CSE-CIC-IDS2018-v2 na seleção e na avaliação das soluções.

Nesta etapa, proponho utilizar as três bases durante a otimização. Com isso,
eu deixo de reservar a terceira base como teste externo independente. Para
investigar generalização externa posteriormente, precisarei de outro protocolo,
como selecionar em duas bases e avaliar na terceira sem utilizá-la na escolha
de atributos ou parâmetros.

Eu não pretendo concatenar as três bases em uma única tabela de treinamento.
Quero preservar cada origem e destino para examinar onde a transferência
funciona e onde ela falha.

## O que já observei e como avancei

No experimento com duas bases, concluí 15 gerações, população 24 e seed 42.
Encontrei uma solução final com três atributos: `SERVER_TCP_FLAGS`,
`TCP_WIN_MAX_IN` e `DNS_QUERY_ID`. Também concluí a avaliação intra/cross dessa
solução e do baseline com todos os 37 atributos candidatos daquele experimento.

Reconstruí o histórico pelo cache e conferi a fronteira obtida contra o Pareto
salvo. Observei que o hipervolume passou de 0,100282 para 0,170428 e ainda cresceu
na geração 15. Interpreto isso como progresso nos objetivos definidos; não
concluo que a busca estabilizou ou que alcancei o ótimo global.

Já baixei e verifiquei a integridade do ZIP oficial do terceiro dataset. Quando defini este protocolo, eu ainda não havia convertido o CSV nem iniciado
a rodada ampliada. Eu registro a implementação na seção seguinte e mantenho
separados os resultados concluídos de duas bases e os da nova investigação.

## O que implementei após definir o protocolo

Eu converti o CSV diretamente do ZIP em lotes e confirmei 18.893.708 linhas
antes e depois da conversão. Eu preservei o arquivo oficial e gravei hashes
SHA-256 de origem e destino. Eu adaptei a otimização para `n*(n-1)` direções,
criei `src/config_tres_datasets.yaml` e separei as saídas em
`Resultados/tres_datasets_v1/`.

Eu implementei `src/executar_experimento.py` e o serviço
`ids-tres-datasets.service` para encadear todas as etapas com retomada. Eu
valido a configuração e os hashes dos arquivos antes de reutilizar resultados.
Eu adaptei os gráficos para seis direções e três painéis intra/cross, mantendo
o diagnóstico de hipervolume. Eu guardo as contagens de limpeza de cada base.

Eu leio uma base de cada vez durante a preparação e libero sua cópia bruta.
Eu armazeno os números em `float64` na conversão e, depois da limpeza, converto
as features para `float32` para reduzir memória. Eu mantenho todos os registros
válidos e não aplico subamostragem. Registro abaixo o plano original para
explicar as escolhas que orientaram essa implementação; os achados da nova
rodada dependerão dos resultados efetivamente gerados.

## O que verifiquei ao iniciar a execução completa

Eu iniciei o serviço `ids-tres-datasets.service` com população 24, 15 gerações
e seed 42. Eu confirmei 37 atributos comuns e seis direções. Registrei as
seguintes contagens após o pré-processamento:

| Minha base | Registros do arquivo | Registros que removi | Registros que utilizei |
|---|---:|---:|---:|
| NF-UNSW-NB15-v2 | 1.986.745 | 0 | 1.986.745 |
| NF-ToN-IoT-v2 | 13.135.881 | 131 | 13.135.750 |
| NF-CSE-CIC-IDS2018-v2 | 18.893.708 | 253 | 18.893.455 |
| Total | 34.016.334 | 384 | 34.015.950 |

Eu removi registros com valores ausentes, infinitos ou fora do intervalo
numérico admitido. Eu iniciei a busca com zero avaliações recuperadas, em um
cache separado. Eu ainda não tenho resultados finais dessa rodada.

Eu validei a média dos seis F1, as seis transferências, o alinhamento dos dados,
a identidade dos arquivos, a rejeição de cache incompatível e a reconstrução
do Pareto. Também executei o pipeline completo com três bases sintéticas até
baseline e gráficos; ao executá-lo novamente, confirmei que ele reconheceu a
conclusão sem recomputar. Eu uso esses testes para verificar a implementação,
e não para substituir os dados completos da investigação.

## Como vou preparar os dados

1. Vou extrair o CSV do ZIP `Datasets/NF-CSE-CIC-IDS2018-v2.zip` e preservar as
   informações de origem e os manifestos que acompanham o download.
2. Vou converter o CSV em lotes para
   `Datasets/NF-CSE-CIC-IDS2018-V2.parquet`, sem amostrar ou eliminar linhas na
   conversão. Vou conferir se a contagem de entrada coincide com a de saída.
3. Vou verificar nomes, tipos, rótulos e valores inválidos antes de executar a
   limpeza. Vou registrar o total original, o total removido e o total utilizado
   em cada base. Não vou confundir a leitura integral de um arquivo local com
   a comprovação de que ele contém todos os registros da publicação original.
4. Vou aplicar as regras existentes de remoção de valores ausentes, infinitos
   e fora do intervalo `float32`, além do descarte de atributos previsto na
   configuração. Vou alinhar os atributos comuns nas três bases e registrar a
   ordem exata. Só depois dessa verificação vou definir o novo valor de `d`;
   não vou assumir antecipadamente que continuará sendo 37.
5. Vou conferir as categorias de ataque. Vou preservar a normalização de
   espaços e capitalização e registrar quais classes estão ausentes em cada
   origem, sem declarar categorias distintas equivalentes apenas pelo nome.

Vou manter os dados brutos no ambiente local. Para compartilhar a pesquisa,
pretendo versionar o protocolo, as configurações, as métricas e os gráficos.

## Quais transferências vou avaliar

Eu vou calcular os seis pares ordenados entre as três bases:

| Minha base de treinamento | Minha base de teste |
|---|---|
| NF-UNSW-NB15-v2 | NF-ToN-IoT-v2 |
| NF-ToN-IoT-v2 | NF-UNSW-NB15-v2 |
| NF-UNSW-NB15-v2 | NF-CSE-CIC-IDS2018-v2 |
| NF-CSE-CIC-IDS2018-v2 | NF-UNSW-NB15-v2 |
| NF-ToN-IoT-v2 | NF-CSE-CIC-IDS2018-v2 |
| NF-CSE-CIC-IDS2018-v2 | NF-ToN-IoT-v2 |

Em cada par, vou treinar na origem e testar no destino, usando a mesma máscara
e a mesma ordem de atributos. Vou usar todos os registros válidos de cada
arquivo, sem subamostragem. Na avaliação intra-dataset, vou realizar cinco folds
estratificados em cada uma das três bases.

## Como proponho ampliar os objetivos

Vou manter dois objetivos no NSGA-II, mesmo trabalhando com três datasets.
Para uma máscara `m`, proponho minimizar:

```text
f1(m) = 1 - (soma dos seis F1-macro direcionais / 6)
f2(m) = k(m) / d
```

Vou atribuir o mesmo peso a cada direção. Dessa forma, o tamanho de uma base
não determina diretamente o peso dela na média. Vou preservar os seis valores
individuais nos artefatos porque uma média melhor pode esconder uma direção
com desempenho pior. Continuarei aplicando o limiar 0,5 aos genes reais e a
penalização prevista para máscaras vazias.

Para a primeira execução ampliada, proponho manter a árvore de decisão de
profundidade 8, `class_weight=balanced`, população 24, seed 42 e orçamento inicial
de 15 gerações, para evitar mudar simultaneamente todos os fatores. Vou tratar
esse orçamento como exploratório: o resultado anterior não demonstra que 15
gerações bastam. Antes de ampliar o orçamento ou acrescentar seeds, vou registrar
o critério e os limites computacionais da nova rodada.

Vou manter explícita a definição atual do F1 cross: agrupo classes do destino
ausentes no treinamento no código `-1`, e calculo a média macro sobre a união
dos códigos observados e previstos. Isso não equivale ao macro-F1 separado de
cada categoria desconhecida original. Se eu investigar outra definição ou uma
taxonomia harmonizada, vou tratá-la como outro protocolo e outro cache.

## O que preciso adaptar no código

| Onde vou trabalhar | O que preciso alterar |
|---|---|
| `src/config.yaml` e configuração do novo experimento | Vou preservar a configuração de duas bases e criar uma configuração identificada para três bases e suas saídas. |
| `Modulos/otimizacao.py` | Vou substituir as exigências de exatamente duas bases e dois F1 por validação de `n >= 2` e `n*(n-1)` direções, calculando a média correspondente. |
| `Modulos/avaliacao.py` | Vou validar o laço existente de pares com três bases e conferir que não inclui transferência de uma base para ela mesma. |
| `src/algoritmo1_otimizacao.py` e checkpoints | Vou identificar bases, arquivos, schema, métrica e parâmetros no contexto da nova execução. |
| `src/algoritmo2_avaliacao.py` | Vou conferir a avaliação das três bases e salvar a identidade completa do experimento em cada solução e baseline. |
| `src/graficos_cross.py` e `src/relatorio_orientadores.py` | Vou adaptar cores, agrupamentos, painéis e legendas atualmente limitados a duas direções ou duas bases. |
| `src/diagnostico_nsga2.py` | Vou validar a reconstrução para três bases e remover textos ou pressupostos específicos de duas bases. |
| `src/continuar_analise.py`, scripts e unidades em `deploy/systemd/` | Vou selecionar artefatos pelo contexto da execução e separar caminhos, locks e marcadores, incluindo a unidade vinculada à execução antiga. |

Eu identifiquei que não basta acrescentar um terceiro nome ao YAML:
antes da adaptação, `Modulos/otimizacao.py` rejeitava uma quantidade de bases diferente de dois.
Também encontrei nomes de cache, seleção de Pareto e marcadores de conclusão
que precisam ser separados para não reutilizar indevidamente a execução anterior.

## Como vou preservar os experimentos e a retomada

Vou conservar os resultados já publicados de duas bases. Vou criar uma pasta
própria para a rodada com três bases, com configuração, checkpoints, logs,
Pareto, avaliações e relatórios identificados pelo experimento.

Vou iniciar uma nova busca para os seis pares. Não vou apresentar a população
ou o cache agregado da busca anterior como continuação equivalente, porque o
primeiro objetivo mudou. Se eu decidir reaproveitar avaliações direcionais no
futuro, precisarei validar individualmente máscara, schema, arquivos, classificador,
seed e definição de métrica; esse reaproveitamento não faz parte da proposta inicial.

Vou usar execução supervisionada e checkpoints. Na Fase 2, vou salvar cada
solução concluída antes de passar à seguinte. Vou conferir que o marcador da
rodada antiga não impede o início da nova e que um reinício não mistura resultados.
Como passarei de duas para seis direções e incluirei uma base adicional, espero
maior custo de execução; vou medi-lo, sem prometer uma duração antes da medição.

## Quais testes vou usar antes da execução completa

- Vou usar três bases sintéticas para verificar seis direções únicas e ausência
  de pares com origem igual ao destino.
- Vou conferir numericamente a média de seis F1 e a proporção `k/d`.
- Vou testar o alinhamento e a rejeição de máscaras ou schemas incompatíveis.
- Vou verificar que o contexto de duas bases não é aceito como cache de três bases.
- Vou testar retomada, escrita dos resultados, baseline e geração dos gráficos
  com três bases, sem reduzir os dados da execução científica.
- Vou conferir que o histórico reconstruído reproduz o Pareto salvo e que o
  hipervolume usa escala e referência fixas em todas as gerações.
- Vou manter os testes do caso de duas bases para conferir a compatibilidade.

## O que pretendo apresentar aos orientadores

Vou apresentar a fronteira de Pareto, o hipervolume e sua variação por geração,
os seis F1 direcionais, a comparação intra/cross nas três bases, a seleção de
atributos e as matrizes de confusão por direção. Vou comparar todas as soluções
finais com o baseline de todos os atributos comuns.

Vou manter a referência do hipervolume em `(1.1, 1.1)` e os objetivos em `[0,1]`.
Ainda assim, não vou comparar diretamente o HV de duas bases com o de três como
se medissem a mesma tarefa: a composição do primeiro objetivo mudou.

Vou procurar respostas para estas perguntas:

1. A seleção de atributos mantém um compromisso favorável quando incluo outra rede?
2. Uma melhoria da média aparece nas seis direções ou esconde perdas específicas?
3. Que atributos voltam a aparecer nas soluções encontradas?
4. A busca continua melhorando ao atingir o orçamento de gerações?
5. Que limitações de taxonomia permanecem mesmo quando uso atributos comuns?

Vou relatar os resultados que eu observar, inclusive se a seleção não superar
o baseline. Não vou atribuir todo erro em classes conhecidas a domain shift,
nem interpretar uma única seed como demonstração de robustez.

## Minhas referências de procedimento

Vou usar o [portal oficial de datasets da UQ](https://staff.itee.uq.edu.au/marius/NIDS_datasets/)
e a publicação de Sarhan, Layeghy e Portmann, *Towards a Standard Feature Set for
Network Intrusion Detection System Datasets* (2022), para documentar a origem
das bases. Para o diagnóstico, vou seguir a
[análise de convergência do pymoo](https://pymoo.org/getting_started/part_4.html).
