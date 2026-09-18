# Resultados para orientação

FINAL DA FASE 1 — geração 15/15.

Fonte: Resultados/pareto/pareto_20260907_000921_decision_tree.json

- Eu observei o seguinte na busca: Hipervolume: 0.100282 na geração 1 e 0.170428 na geração 15. Variação na última geração: +0.003375. Houve melhora na última geração; a curva não sustenta afirmar estabilização ao encerrar.
- Hipervolume da fronteira sobrevivente por geração; referência fixa (1.1, 1.1), objetivos em [0,1]. Histórico reconstruído apenas pelo cache e validado contra a fronteira salva. A curva não comprova ótimo global. Método: https://pymoo.org/getting_started/part_4.html
- Eu utilizei os Parquets completos, sem subamostragem, e uma árvore de decisão com profundidade 8 e pesos balanceados.
- NSGA-II: população 24, 15 gerações previstas, seed 42. A fronteira mostrada pertence à geração registrada; o cache pode conter avaliações posteriores ainda sem uma geração concluída.
- Eu agrupo classes do destino ausentes no treino no código -1 e calculo o F1-macro sobre a união dos rótulos observados e previstos. Não calculo separadamente cada categoria desconhecida original.
- Eu interpreto a partição de erros como frações de amostras, sem decompor o F1 ou atribuir isoladamente os erros a domain shift.
- Eu apresento as matrizes da solução mais compacta e agrupo na coluna sem correspondência as previsões fora do vocabulário do destino.
- Eu utilizei as bases de transferência na seleção das máscaras e interpreto esses valores como critérios de seleção, sem afirmar teste externo independente.
- Eu não afirmo estabilidade entre execuções com uma única seed e não declaro uma solução vencedora sem um critério adicional.
- Eu concluí as seguintes avaliações detalhadas: 2. Barras de dispersão mostram o desvio padrão entre folds, não intervalo de confiança.
