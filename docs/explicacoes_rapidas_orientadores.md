# Explicações rápidas para os orientadores (WhatsApp)

Eu preparei estas mensagens em 23/10/2026 para enviar pelo WhatsApp (mensagens 1 a 6). Todos os números são da **geração 73/100** (PARCIAL, seed 42, somente validação). As figuras correspondentes estão congeladas em `Resultados/mlp_conjunto_v1/orientadores_g073/`, porque a pasta `figuras/` continua mudando enquanto a busca roda.

Como enviar: anexe a imagem indicada e cole o texto dentro do bloco `~~~` logo abaixo. O texto já usa a formatação do WhatsApp (`*negrito*` e ```` ``` ```` para a tabela em fonte fixa). A justificativa completa está na Entrada 006 do [diário](diario_de_pesquisa.md).

---

## Mensagem 1 — Tabela de referência

Anexo: nenhum (ou `metricas_solucoes.csv`, se quiserem a planilha).

~~~
*MLP conjunta — resultado parcial (geração 73/100, seed 42, só validação)*

Tabela de referência das 22 soluções do Pareto atual. Todas são não dominadas nos 3 objetivos (erro, nº de atributos, tempo).

```
Sol   k  F1    µs   TNR  Fr
S1*   2  0,102 0,95 0,04 ac
S2    3  0,228 0,70 0,15 ac
S3    4  0,354 0,73 0,28 ab
S4*   4  0,233 0,65 0,31 bc
S5    4  0,324 0,68 0,27 b
S6    5  0,363 0,77 0,35 ab
S7    5  0,268 0,68 0,22 b
S8    6  0,427 1,42 0,40 a
S9    6  0,393 0,83 0,31 b
S10   7  0,421 1,17 0,46 —
S11   7  0,457 1,28 0,43 a
S12*  8  0,458 0,84 0,42 ab
S13   9  0,470 1,26 0,48 a
S14  10  0,481 0,87 0,43 ab
S15  11  0,492 0,88 0,32 ab
S16  12  0,499 1,15 0,45 a
S17  14  0,527 1,21 0,46 a
S18  15  0,514 1,03 0,45 —
S19  16  0,537 1,40 0,45 a
S20  17  0,544 1,03 0,47 ab
S21  19  0,548 1,14 0,45 ab
S22* 21  0,551 1,32 0,46 ab
Base 37  0,516 1,63 0,43 -
```
k = nº de atributos (de 37) | F1 = F1-macro | µs = mediana por evento | TNR = fração do benigno reconhecida como benigna
* = escolhidas antes do teste: S22 maior F1, S12 compromisso, S1 mais compacta, S4 mais rápida
Fr = fronteira 2D em que aparece: a = erro×atributos (gráfico 01), b = erro×tempo (02 esq.), c = tempo×atributos (02 dir.)

Os rótulos S são renumerados a cada geração, e os números podem mudar até a geração 100.
~~~

---

## Mensagem 2 — Gráfico 01: erro × atributos

Anexo: `orientadores_g073/01_pareto_erro_atributos.png`

~~~
*Gráfico 01 — Erro × atributos (cor: tempo)*

É o trade-off central da dissertação: reduzir atributos sem perder detecção.
Eixo X = nº de atributos; eixo Y = erro (1 − F1-macro). Nos dois, menor é melhor, então o melhor fica no canto inferior esquerdo. A cor azul é o 3º objetivo, o tempo; mais escuro = mais lento.

Leitura:
- O erro cai rápido até ~8 atributos (F1 0,10 → 0,46) e depois a curva achata: de 17 para 21 atributos o F1 só vai de 0,544 para 0,551.
- Círculos ligados = fronteira desta projeção (16 soluções). Losangos = parecem dominados aqui, mas são mais rápidos que quem os domina, logo continuam não dominados nos 3 objetivos.
- Círculo vermelho (S11, 7 atributos) = candidato geométrico a joelho. É indicação visual, não escolha automática.
- X cinza = baseline com os 37 atributos (F1 0,516; 1,63 µs). Cinco soluções o dominam nos 3 objetivos: S17, S19, S20, S21 e S22. Ex.: S20 tem 17 atributos, F1 0,544 e 1,03 µs.
~~~

---

## Mensagem 3 — Gráfico 02: projeções com tempo

Anexo: `orientadores_g073/02_pareto_tempo.png`

~~~
*Gráfico 02 — As duas projeções restantes*

Assim cada par de objetivos aparece nos eixos uma vez. Em cada painel, a cor é o objetivo que ficou fora dos eixos; mais escuro = pior.

*Esquerda — Erro × tempo (verde = atributos):* mostra o custo de inferência para reduzir o erro. São 12 soluções na fronteira; ganhar F1 exige redes maiores (verde mais escuro) e mais lentas. S20 (F1 0,544; 1,03 µs) já supera o baseline em erro e em tempo.

*Direita — Tempo × atributos (roxo = erro):* o tempo cresce com o nº de atributos porque a rede tem 3 camadas de 2k neurônios. Só 3 soluções estão na fronteira (S1, S2, S4), todas com erro alto (roxo escuro). Ou seja, ser mínimo em atributos e em tempo custa caro em detecção.

Cuidado: soluções com o mesmo nº de atributos têm a mesma rede, mas o tempo medido varia (S8 e S9, ambas com 6 atributos: 1,42 vs 0,83 µs). Não interpreto diferenças pequenas de tempo.

Leitura conjunta 01 + 02: S10 e S18 não estão em nenhuma fronteira 2D e mesmo assim são não dominadas, porque equilibram os 3 objetivos. Por isso não me apoio em uma única projeção.
~~~

---

## Mensagem 4 — Gráfico g1: consolidação parcial

Anexo: `orientadores_g073/g1_coordenadas_paralelas.png`

~~~
*Gráfico g1 — Coordenadas paralelas: consolidação parcial*

Cada linha é uma solução e cada eixo é uma métrica, todos orientados para "para cima é melhor". É o único gráfico que mostra os 3 objetivos juntos, mais acurácia, TNR, TPR e F1 por base. Destaquei as 4 escolhas fixadas na validação e o baseline.

Evidências parciais:
1. Não existe solução ótima em tudo: as linhas se cruzam nos 3 primeiros eixos. S22 tem o maior F1 (0,551), mas 21 atributos; S1 tem 2 atributos, mas F1 0,10; S4 é a mais rápida (0,65 µs), com F1 0,23.
2. Redução sem perda: S22 domina o baseline, com F1 0,551 vs 0,516, 21 vs 37 atributos (−43%) e 1,32 vs 1,63 µs (−19%).
3. Compromisso: S12 usa 8 atributos (−78%) e 0,84 µs (−48%), perdendo 0,058 de F1 em relação ao baseline.
4. Por base, S22 supera o baseline em UNSW (0,43 vs 0,39) e CIC (0,72 vs 0,65) e fica próxima em ToN (0,56 vs 0,58).
5. O hipervolume está estável perto de 0,50 desde a geração ~40 (0,498 → 0,500). É sinal de estabilização da busca, não prova de convergência.

Limites: só validação, uma seed, geração 73/100. O teste só será aberto depois de fixar as escolhas.
~~~

---

## Mensagem 5 — Gráfico g3: problema de peso na detecção

Anexo: `orientadores_g073/g3_desempenho_por_base.png`

~~~
*Gráfico g3 — Desempenho por base: um problema de peso na detecção*

*O que é o eixo X:* em todos os painéis, inclusive os dois da direita, ele não é numérico. São as partes da validação separadas pela base de origem (UNSW, ToN, CIC) e "Conjunto" = toda a validação junta. O eixo Y é a métrica, de 0 a 1. Cada cor é uma das soluções escolhidas.
- Superior direito: F1-macro em cada base, só com as classes presentes nela. Em "Conjunto", é o F1-macro geral com as 31 classes, o mesmo usado no objetivo (erro = 1 − F1); por isso não é a média das três bases.
- Inferior direito: TPR = fração dos ataques classificada como algum ataque (não exige acertar o tipo).
*O que é o eixo Y:* o valor da métrica indicada no próprio eixo (acurácia, F1-macro, TNR ou TPR). Os quatro painéis usam a mesma escala de 0 a 1, e maior é melhor em todos.

*O problema:*
- TPR ≈ 0,97–1,00 em todas as bases e soluções: quase nenhum ataque escapa.
- TNR (inferior esquerdo) é baixo: no Conjunto, só 4% a 46% do tráfego benigno é reconhecido como benigno; em ToN, de 0,5% a 32%.
- Na prática, cerca de metade dos alertas é falso alarme: 52% na S22 e 53% no baseline, na validação.

*Hipótese (ainda não testada):* os pesos balanceados por classe no treino. O benigno é 63,5% dos exemplos de treino, mas recebe 1/31 do peso total (3,2%), porque cada uma das 31 classes recebe o mesmo peso total. Um exemplo benigno pesa 0,05; um de "worms" pesa ~6.080. Para a rede, errar benigno como ataque sai barato. O F1-macro também dá ao benigno só 1/31 do objetivo, embora os falsos alarmes reduzam indiretamente a precisão das classes de ataque.

Como o baseline com 37 atributos tem o mesmo comportamento, o problema vem do protocolo de treino, não da seleção de atributos.

*Proposta para discutirmos (não adotada):* testar pesos suavizados (ex.: raiz quadrada), um peso maior para o benigno, ou incluir a taxa de falso alarme como objetivo ou restrição. Qualquer mudança seria um novo experimento, sem misturar com esta rodada.
~~~

---

## Mensagem 6 — Gráfico g6: treino por época e seeds

Anexo: `orientadores_g073/g6_curvas_treino.png`

~~~
*Gráfico g6 — Treino por época e por que preciso de mais seeds*

Uma época = uma passada da MLP pelos 24,5 milhões de registros de treino. Depois de cada época, avalio na validação.
- Esquerda: F1-macro de validação por época. O marcador é a melhor época, e é esse modelo que fica guardado.
- Direita: loss de treino (entropia cruzada com pesos de classe); menor é melhor.
- Parada: 5 épocas sem melhora no F1 de validação (máx. 30). Ex.: S22 teve a melhor época na 11 e parou na 16.

O que o gráfico mostra:
1. O F1 oscila muito entre épocas: S22 foi de 0,437 (época 10) para 0,551 (época 11) e caiu para 0,48 na 16. As oscilações chegam a 0,07–0,11.
2. O valor guardado é um pico escolhido na própria validação, que também é o objetivo do NSGA-II. Logo, tende a ser otimista; o teste reservado vai mostrar quanto se mantém.
3. A loss continua caindo depois da melhor época enquanto o F1 cai, por isso a parada é pelo F1.
4. A loss é irregular. Uma hipótese são os pesos de classe extremos (o mesmo problema do g3).

*Relação com as seeds:* a seed 42 fixa a divisão dos dados, os pesos iniciais, a ordem dos dados por época e o NSGA-II. Cada curva é uma única realização. Se o F1 oscila 0,07–0,11 entre épocas, a variação entre seeds deve ter ordem parecida, maior que a diferença S22 vs baseline (0,551 vs 0,516).

*Proposta (não adotada):* retreinar as 4 soluções escolhidas e o baseline com 5 seeds de modelo, mantendo a divisão dos dados. São ~20 treinos, cerca de 10 h, sem refazer a busca. Isso diria se a vantagem supera o ruído do treino.
~~~
