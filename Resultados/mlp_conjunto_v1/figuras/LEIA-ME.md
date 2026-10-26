# Meu relatório da MLP conjunta

PARCIAL | geração 80/100 | seed 42

Eu treino uma MLP com dados das três bases. Eu separo treino, validação e teste por grupos de atributos idênticos.
Eu seleciono os atributos na validação. Eu uso o teste somente após fixar as escolhas.

## Como eu leio o Pareto

Eu minimizo erro (1 − F1-macro), proporção de atributos e segundos por evento.
Eu mostro somente gráficos 2D. Melhorar erro e quantidade de atributos significa avançar para o canto inferior esquerdo.
Eu desenho as três projeções com a mesma codificação. A cor mostra o objetivo que ficou fora dos eixos, com uma matiz
fixa por objetivo: azul para tempo, verde para atributos e roxo para erro; mais escuro é sempre pior.
Círculos estão na fronteira daquela projeção; losangos ficam fora dela, mas continuam não dominados nos três objetivos
porque vencem no objetivo da cor quem os domina na projeção. O baseline aparece como X cinza em todos os painéis.
Na legenda, dois quadrados mostram o tom claro e o escuro com o valor do objetivo da cor em cada extremo.
Uma solução pode ficar fora das três fronteiras projetadas e continuar não dominada: ela equilibra os três objetivos.
Eu não uso gráfico 3D estático: a perspectiva impede ler valores, e uma superfície exigiria interpolar entre soluções.
Para ver os três objetivos lado a lado sem projeção, eu uso as coordenadas paralelas (g1) em figuras_extras.
Eu ligo apenas os pontos não dominados da projeção; os segmentos não são resultados intermediários medidos.
Eu não imponho formato de barriga nem suavizo os pontos.

Minha verificação geométrica: Há um candidato geométrico a joelho abaixo da reta entre extremos. Não é prova de ótimo, convergência ou escolha automática.

Eu calculo o HV nos três objetivos, com tempo transformado por t/(t+t_ref), referência fixa (1.1, 1.1, 1.1)
e t_ref=1.63122935e-06 s/evento, medido no baseline. Na otimização eu uso o tempo original.
Eu não comparo esse HV ao do experimento anterior de dois objetivos.

## Como eu leio as matrizes e o tempo

Eu mostro contagens e percentuais por linha; a diagonal percentual é recall da classe, não acurácia geral.
Eu incluo precisão, recall, F1 e suporte nos CSVs. Um traço identifica classe sem exemplos no teste daquele painel.
Eu mantenho o vocabulário de classes fixo do treino no macro-F1, inclusive nas tabelas por base.
Eu meço padronização + previsão, com lotes e repetições fixos, excluindo leitura de disco.
Eu registro mediana e máximo observado do tempo médio por evento em lote; isso não é a maior latência individual.
Eu não afirmo generalização para uma quarta base nem estabilidade entre seeds.
