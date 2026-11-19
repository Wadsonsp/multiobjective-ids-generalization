# Histórico da conversa e briefing para Claude

**Atualizado em:** 18/11/2026
**Projeto:** dissertação sobre seleção multiobjetivo de atributos para IDS e avaliação de generalização.
**Finalidade:** fornecer a Claude o histórico de decisões, os achados verificados, o estado dos arquivos e o próximo trabalho. Este documento é um handoff cronológico e técnico, não uma transcrição literal de todas as mensagens.

## Instrução de continuidade

Continue a análise e a revisão dos gráficos da MLP usando o repositório local. Comece por `MAPA_DA_PESQUISA.md` e este histórico. Preserve os dados e resultados já obtidos. Não treine modelos novamente para refazer figuras. Confira o estado atual do Git antes de editar, porque há alterações locais ainda não commitadas. O ponto de atenção pendente é a clareza das duas projeções de Pareto e de como nelas se comunica o candidato a joelho.

O usuário prefere explicações e entregas em português, com afirmações apoiadas pelos resultados e limites descritos com clareza. `.venv`, `.pytest_cache`, `.github` e `Datasets` foram excluídos da inspeção original; os binários de imagem também não foram abertos durante aquela primeira etapa. Mais tarde, o usuário autorizou leitura dos gráficos e solicitou atualizações.

## Histórico cronológico

1. **O usuário pediu um commit local.** Foram registrados resultados novos da MLP.
2. **O usuário pediu ajuda para sincronizar e fazer push.** A `main` estava três commits à frente do GitHub. Dois JSONs de aproximadamente 259 MiB excediam o limite normal de blobs do GitHub. Foram adicionados ao Git LFS e o último commit foi refeito como `79033ff`; criou-se também o backup local `backup/main-before-lfs-30869fb`. O push não terminou porque o terminal não tinha autenticação GitHub disponível.
3. **O usuário pediu uma análise da pesquisa em duas etapas:** primeiro, ler o mapa, apresentar o entendimento e o plano sem alterações; depois, prosseguir após autorização.
4. **O usuário apontou o PDF `seminarioPPGIav5.pdf`** em `Downloads` e pediu para identificar a pergunta principal e as acessórias, cobrir a trajetória E1–E3, separar mudança de método de revisão gráfica e verificar estudos ainda propostos.
5. **Foram encontrados resultados finais da MLP** posteriores ao estado descrito no mapa: geração 100/100, arquivo de conclusão, escolhas pré-teste e avaliações do teste reservado. A ficha E3 e o mapa não foram atualizados e ainda refletem geração 73/100.
6. **O usuário perguntou sobre disponibilidade do modelo.** Foram encontrados 1.768 arquivos `modelo.joblib` locais em `Resultados/mlp_conjunto_v1/modelos/`; essa pasta é ignorada pelo Git. Métricas e figuras estão versionáveis, mas os pesos do modelo não foram enviados no repositório.
7. **O usuário perguntou se os resultados eram melhores e mais compatíveis com a pesquisa.** A conclusão foi que a MLP dá evidência de trade-off sob treino conjunto, mas não demonstra transferência para rede ausente do treino nem robustez entre seeds.
8. **O usuário pediu atualização dos gráficos.** As oito figuras complementares foram atualizadas de geração 82 para geração 100, junto com tabelas CSV e PDF consolidado em `Resultados/mlp_conjunto_v1/figuras_extras/`. Foram conferidas 23 linhas (22 soluções e baseline), identificadores e valores contra a geração 100.
9. **O usuário pediu explicação do gráfico g1.** Foi explicado que é um gráfico de coordenadas paralelas, não uma série temporal ou “sismógrafo”: cada eixo é uma métrica, cada linha é uma solução, e todos os eixos foram orientados para que cima seja melhor. As soluções destacadas são S22, S10, S1, S4 e baseline.
10. **O usuário perguntou sobre o teste e onde ele aparece.** O teste reservado já está feito para S22, S1, S4, S10 e baseline. O gráfico de comparação é `figuras/04_validacao_teste.png`; matrizes de confusão estão em `figuras/05_matriz_*.png`.
11. **O usuário apontou que o título de `04_validacao_teste.png` estava estranho.** O título foi alterado no gerador para “F1-macro na validação e no teste reservado”. PNG, PDF da figura, relatório completo, HTML e ZIP das figuras foram regenerados. A alteração está local, sem commit.
12. **O usuário perguntou se os gráficos de Pareto estavam confusos e quem seria o candidato a joelho em 02.** As figuras foram revistas e os dados da geração 100 foram analisados. A conclusão e o diagnóstico exploratório estão detalhados abaixo; os gráficos de Pareto ainda não foram modificados para resolver a confusão.
13. **O usuário pediu o objetivo da dissertação e as perguntas da apresentação, relacionados à trajetória.** Foram comparados o escopo original do seminário e o que E1–E3 de fato testaram. A orientação foi preservar o histórico da proposta, explicitar a mudança de pergunta no E3 e não afirmar que a MLP conjunta prova generalização cross-dataset.
14. **O usuário pediu um arquivo Markdown da conversa** e esclareceu que deveria conter o histórico para dar continuidade no Claude. Este arquivo foi refeito como esse histórico e briefing.

## Objetivo original e perguntas da apresentação

O objetivo geral apresentado foi:

> Desenvolver IDS com generalização cross-dataset considerando a seleção de features invariantes via meta-heurísticas, demonstrando acurácia e eficiência através da análise de trade-offs (desempenho-custo).

A pergunta principal foi:

> Como desenvolver um IDS baseado em ML que seja validado por generalização cross-dataset mantendo a eficiência computacional e a performance preditiva?

As perguntas acessórias foram:

1. Quais atributos selecionados por algoritmos genéticos demonstram mais estabilidade e poder discriminativo quando transferidos entre datasets de IDS?
2. Qual é a relação quantitativa entre o número de atributos, o desempenho preditivo e o custo computacional, medido pelo tempo de inferência?
3. Quais modelos de ML equilibram melhor acurácia multiclasse, generalização cross-dataset e tempo de inferência?

A apresentação também propunha estudar atributos invariantes, usar NSGA-II, comparar desempenho e custo, avaliar dados reais de NetFlow do STJ e seguir quatro fases: preparação; análise e seleção; avaliação cross-dataset; consolidação. Os resultados preliminares de modelos da página 10 pertencem à proposta e não devem ser tratados como comparação controlada com E1–E3.

## Trajetória e interpretação científica

| Etapa | Desenho | Relação com as perguntas | Resultado e limite |
|---|---|---|---|
| E0 | Árvore; população 1; uma geração; teste operacional | Verificação de pipeline | Descartado como evidência científica. |
| E1 | Árvore; transferência UNSW ↔ ToN; objetivos `1 − F1 cross médio` e `k/37` | Primeira avaliação direta da pergunta cross-dataset | S1 reduziu atributos de 37 para 3 e melhorou o F1 cross médio de 0,0471 para 0,0673; valores absolutos baixos. Uma seed; os domínios participaram da seleção. |
| E2 | Árvore; três bases; seis direções; mesmos dois objetivos | Examina a ampliação de transferência para uma terceira base | S2 (8 atributos) supera o baseline nas seis direções; F1 cross médio 0,0585 contra 0,0353. Ainda não é teste externo independente; F1 cross baixo. |
| E3 | MLP treinada conjuntamente em UNSW, ToN e CSE; objetivos `1 − F1 validação`, `k/37` e mediana de segundos/evento | Responde principalmente à pergunta de trade-off desempenho/atributos/tempo | Há teste reservado, mas das mesmas três bases usadas no treino. Não demonstra transferência para uma quarta rede, invariância de atributos ou robustez entre seeds. |

E2 → E3 altera simultaneamente classificador, protocolo, objetivos, divisão dos dados e orçamento de gerações. Logo, a diferença entre os resultados não isola o efeito de trocar árvore por MLP.

### Resultados finais conhecidos de E3

| Escolha feita pela validação | Atributos | F1 validação | F1 teste reservado | Mediana medida na validação |
|---|---:|---:|---:|---:|
| S22 — maior F1 | 21 | 0,5511 | 0,5332 | 1,317 µs/evento |
| S10 — compromisso | 8 | 0,4643 | 0,4632 | 0,871 µs/evento |
| S4 — mais rápida | 4 | 0,2329 | 0,1843 | 0,654 µs/evento |
| S1 — mais compacta | 2 | 0,1361 | 0,1024 | 0,708 µs/evento |
| Baseline — todos os atributos | 37 | 0,5156 | 0,5325 | 1,631 µs/evento |

A S22 reduz o conjunto de atributos em aproximadamente 43% e o tempo mediano medido em validação em aproximadamente 19% frente ao baseline; seu F1 de teste é numericamente próximo. Isso descreve uma única execução, seed 42, e não estabelece equivalência estatística ou robustez.

O F1 observado na validação dá peso igual às 31 classes. A decomposição complementar indica que cerca de 76% da diferença de F1 da S22 sobre o baseline na validação se deve a `ftp-bruteforce`, que tinha somente um exemplo nessa partição. Isso é um alerta interpretativo; não prova, isoladamente, uma causa.

As principais lacunas da proposta original continuam sendo: avaliação em domínio externo; estudo de seeds; comparação controlada de modelos e de seletores simples; teste de pesos de classe; e validação com dados NetFlow do STJ. O estudo de seeds e os controles são propostas no repositório, sem execução identificada.

## Gráficos e questão do candidato a joelho

### Figuras de E3

- `Resultados/mlp_conjunto_v1/figuras/01_pareto_erro_atributos.png`: erro versus atributos; a cor mostra tempo. Este é o gráfico em que o código calcula e marca o candidato a joelho.
- `Resultados/mlp_conjunto_v1/figuras/02_pareto_tempo.png`: dois painéis, erro versus tempo (cor = atributos) e tempo versus atributos (cor = erro). Mostra projeções, não um cálculo oficial de joelho.
- `Resultados/mlp_conjunto_v1/figuras/03_hipervolume.png`: evolução do hipervolume da busca.
- `Resultados/mlp_conjunto_v1/figuras/04_validacao_teste.png`: comparação final entre validação e teste reservado; título atualizado para “F1-macro na validação e no teste reservado”.
- `Resultados/mlp_conjunto_v1/figuras/05_matriz_*.png`: matrizes de confusão das quatro escolhas e baseline, no conjunto e em cada base.
- `Resultados/mlp_conjunto_v1/figuras_extras/g1_coordenadas_paralelas.png`: métricas de validação em eixos paralelos; mostra cada solução como um perfil multivariado, não como sinal temporal.
- As figuras complementares restantes são g3 desempenho por base, g4 classes, g6 curvas de treino, g7/g8 análise de classes raras/contribuição e suas versões para slides.

### Diagnóstico geométrico das projeções

O código em `src/relatorio_mlp.py` aplica a função `diagnosticar_curvatura()` somente à projeção de erro versus atributos. Ela encontra o ponto da fronteira com maior desvio normalizado em direção à origem em relação à reta entre os extremos. O resultado oficial para a figura 01 é **S7**, com seis atributos, F1 de validação 0,427 e mediana de tempo 1,417 µs/evento. O critério usa só erro e quantidade de atributos; não prova ótimo e não considera o terceiro objetivo.

Reaplicando exploratoriamente a mesma regra aos valores da geração 100:

- Na projeção erro × tempo do gráfico 02, a regra escolhe **S20** como máximo desvio geométrico.
- Na projeção tempo × atributos do gráfico 02, a regra não identifica joelho.

Esses cálculos adicionais não fazem parte do relatório nem foram adotados como conclusões científicas. No gráfico 02, S7 pode aparecer com outro marcador porque o formato indica sua posição na fronteira daquela projeção, não porque seja outro modelo ou candidato a joelho.

**Problema apontado pelo usuário:** as figuras 01 e 02, seus títulos e as regras dos marcadores são confusos. O próximo trabalho deve propor uma apresentação que deixe explícito que S7 é candidato apenas no gráfico 01, que 02 contém as demais projeções e que os identificadores S1–S22 se referem às máscaras da geração mostrada. Discutir/revisar a forma de apresentar o diagnóstico; não rotular S20 como joelho científico sem justificar e documentar a decisão.

## Estado dos arquivos e artefatos

Trabalho executado localmente nesta conversa:

- Atualizadas para geração 100 as oito figuras complementares e tabelas CSV em `Resultados/mlp_conjunto_v1/figuras_extras/`.
- Alterado o título em `src/relatorio_mlp.py` e regenerados os artefatos principais em `Resultados/mlp_conjunto_v1/figuras/`, seu relatório PDF, página HTML e ZIP.
- Criado este documento de handoff.

As alterações são locais e **não foram commitadas**. Houve outras alterações locais de resultados antes disso; portanto, inspecione `git status` e preserve todo o trabalho do usuário antes de editar ou fazer commits. A atualização dos gráficos não requer retreinar a MLP. O treinamento local está em `Resultados/mlp_conjunto_v1/modelos/`, pasta ignorada pelo Git; há 1.768 arquivos de modelo.

O mapa e `Resultados/mlp_conjunto_v1/EXPERIMENTO_E3.md` continuam com informação antiga de geração 73 e teste fechado. Ao revisar, compare a documentação com os resultados finais e sinalize o descompasso; não altere os registros científicos sem deixar rastreabilidade.

## Arquivos de referência

- `MAPA_DA_PESQUISA.md`
- `docs/diario_de_pesquisa.md`
- `docs/experimento_mlp_conjunto.md`
- `docs/proposta_estudo_seeds.md`
- `Resultados/EXPERIMENTO_E1.md`
- `Resultados/tres_datasets_v1/EXPERIMENTO_E2.md`
- `Resultados/mlp_conjunto_v1/EXPERIMENTO_E3.md`
- `src/relatorio_mlp.py`
- `src/graficos_mlp_extras.py`
- `Modulos/busca_mlp.py`
- `Resultados/mlp_conjunto_v1/resultado_final.json`
- `Resultados/mlp_conjunto_v1/escolhas_antes_do_teste.json`
- `Resultados/mlp_conjunto_v1/teste/`

O PDF original foi fornecido em `/home/wadson.pereira/Downloads/seminarioPPGIav5.pdf`; essa localização pode não existir no ambiente do Claude. As perguntas e o objetivo originais foram transcritos neste arquivo para permitir continuar sem depender do PDF.
