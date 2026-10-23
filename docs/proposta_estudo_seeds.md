# Minha proposta de estudo de variabilidade por seed (MLP conjunta)

Proposta registrada em 23/10/2026 para discutir com os orientadores. **Não está adotada nem implementada.** A motivação está na Entrada 007 do [diário](diario_de_pesquisa.md).

## Pergunta

A diferença de F1-macro entre as soluções escolhidas e o baseline é maior que a variação causada apenas pela inicialização e pela ordem de treino da MLP?

## Por que preciso dela

- No g6 (geração 73), o F1-macro de validação oscila de 0,07 a 0,11 entre épocas consecutivas da mesma rede. A S22 foi de 0,437 na época 10 para 0,551 na época 11.
- As diferenças que eu quero afirmar são menores que essa oscilação: S22 contra baseline é 0,551 contra 0,516 (0,035); S20, S21 e S22 diferem entre si em menos de 0,01.
- O F1 de cada solução é o da melhor época, escolhida na mesma validação que define o objetivo; portanto, ele tende a ser otimista.
- Hoje a seed 42 controla ao mesmo tempo a divisão dos dados, os pesos iniciais, a ordem dos dados por época e o NSGA-II.

## Desenho proposto

- **O que muda:** só a seed do modelo, que controla os pesos iniciais e a ordem dos dados por época.
- **O que fica fixo:** a divisão treino/validação/teste (seed 42), as máscaras de atributos, a arquitetura `2k`, os hiperparâmetros, a regra de parada e a medição de tempo.
- **Soluções:** as quatro escolhidas antes do teste (maior F1, compromisso, mais compacta, mais rápida) e o baseline, fixadas ao final da geração 100.
- **Seeds:** 42 (a atual) e mais quatro, definidas antes de rodar, por exemplo 1, 2, 3 e 4.
- **O que registro:** média, desvio-padrão, mínimo e máximo do F1-macro de validação e de teste, TNR, TPR e tempo; e a melhor época de cada treino.
- **Custo estimado:** 5 soluções × 4 seeds novas = 20 treinos de 12 a 40 minutos, cerca de 10 horas de máquina. A busca do NSGA-II não precisa ser repetida.

## O que exige no código

Separar `seed_divisao` (usada em `Modulos/dados_mlp.py`) de `seed_modelo` (usada em `Modulos/modelo_mlp.py`). Isso altera a identidade experimental, então o estudo vai para uma pasta nova, por exemplo `Resultados/mlp_conjunto_v1_seeds/`, sem tocar em `mlp_conjunto_v1`.

## Como vou ler o resultado

- Se o intervalo das seeds da S22 ficar acima do intervalo do baseline, posso afirmar a vantagem para esta divisão de dados.
- Se os intervalos se sobrepuserem, eu relato que a diferença não supera a variação de treino. A redução de atributos e de tempo continua válida, mas a afirmação de ganho em F1, não.
- Este estudo não mede a variação da busca do NSGA-II nem da divisão dos dados. Para isso seriam necessárias rodadas completas com outras seeds, que custam muito mais.
