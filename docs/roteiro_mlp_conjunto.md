# Meu roteiro para apresentar o novo experimento ao Eduardo

## O que eu mudei

“Eu reorganizei o experimento para treinar uma MLP com registros das três bases juntos. Eu passei o NSGA-II para 100 gerações, mantive a seed 42 e acrescentei o tempo por evento como terceiro objetivo. Eu preservo o trabalho anterior como outra etapa da pesquisa.”

## Como eu separo os dados

“Eu separo aproximadamente 70% para treino, 15% para validação e 15% para teste. Eu mantenho entradas idênticas na mesma parte, inclusive quando aparecem em bases diferentes. Eu registro as proporções reais. Como não tenho identificadores de sessão comuns às três bases, eu reconheço que esse cuidado não elimina toda possível dependência entre registros.”

“Eu reúno as partes de treino das três bases e ajusto a padronização somente nelas. Eu uso a validação durante a busca. Eu deixo o teste para depois de escolher as soluções. Eu não chamo esse teste de avaliação em uma quarta rede desconhecida.”

## Como eu construo a rede

“Se uma solução selecionar dez atributos, eu treino uma MLP com três camadas escondidas de vinte neurônios cada. Eu sigo a regra de duas vezes a quantidade selecionada. Eu também reconheço que reduzir os atributos reduz a rede, de modo que ambos podem influenciar o tempo.”

## O que eu procuro melhorar

“Eu procuro três coisas ao mesmo tempo: reduzir `1 − F1-macro`, reduzir a proporção de atributos e reduzir o tempo médio por evento. Eu meço o tempo repetidamente e uso a mediana na busca. Eu também registro o máximo observado, sem afirmar que ele seja a maior latência possível de um evento individual.”

## Como eu explico o Pareto e a barriga

“Eu apresento o Pareto somente em duas dimensões. No gráfico principal, eu mostro quantidade de atributos e erro `1 − F1`. Agora, os dois eixos melhoram para o canto inferior esquerdo. Eu uso a cor para indicar o tempo, que continua sendo o terceiro objetivo do algoritmo.”

“Eu verifico se aparece um ponto de compromisso com desvio em direção à origem em relação à reta entre os extremos. Eu posso destacar esse candidato a joelho, mas não modifico os resultados para criar uma barriga. Se houver poucos pontos ou se a curvatura não aparecer, eu relato isso.”

“Eu também mostro erro versus tempo e atributos versus tempo em gráficos 2D. Um ponto que parece pior em dois eixos pode continuar relevante pelo terceiro objetivo. Eu não concluo que o algoritmo esteja errado apenas pelo formato visual.”

## Como eu explico os resultados por classe

“Eu apresento matrizes de contagens e de percentuais por linha. Na diagonal percentual, eu vejo quanto de cada classe foi reconhecido corretamente: esse valor é o recall da classe. Eu mostro também precisão, F1 e quantidade de exemplos, porque acurácia geral sozinha não descreve todas as categorias.”

## Como eu comparo com o baseline

“Eu comparo as soluções com uma MLP treinada com todos os atributos, usando o mesmo protocolo. Eu fixo minhas escolhas com os dados de validação e só depois calculo o teste. Eu procuro verificar se consigo reduzir atributos e tempo mantendo um desempenho próximo ou superior, sem pressupor que esse resultado acontecerá.”

## Como eu informo o estado da execução

“Eu separo o que já implementei do que já executei nos dados reais. Eu informo a geração concluída e se o relatório é parcial. Eu só apresento métricas de teste depois de concluir a busca e fixar as escolhas. Eu não uso os números dos testes sintéticos como resultados da dissertação.”

Eu consulto os parâmetros, comandos e limitações no [meu protocolo completo](experimento_mlp_conjunto.md). Eu uso os arquivos de `Resultados/mlp_conjunto_v1/` para preencher os números da apresentação quando estiverem disponíveis.

## O que eu já posso mostrar na reunião

“Eu já implementei e testei o novo fluxo, preparei os dados reais e conferi que
nenhum grupo de entradas idênticas atravessa treino, validação e teste. Eu tenho
37 atributos e 31 classes. No registro de 21/09, eu ainda não iniciei o treinamento
real da MLP ou as 100 gerações; eu separo essa situação dos testes de software.”

“Na [revisão visual do Pareto anterior](../Resultados/revisao_pareto_anterior/comparacao_eixos.png),
eu observei um candidato a joelho em S2. Eu obtive essa leitura mostrando 1 − F1
no eixo vertical, sem mudar os resultados. Eu não garanto que a nova fronteira
da MLP terá a mesma forma.”
