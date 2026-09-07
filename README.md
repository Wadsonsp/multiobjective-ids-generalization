# Seleção multiobjetivo de features para IDS cross-dataset

Projeto experimental da dissertação sobre generalização cross-dataset em
Sistemas de Detecção de Intrusão. Eu uso o NSGA-II como **pré-filtro (Fase 1)**
para encontrar compromissos entre desempenho cross-dataset e quantidade de
features. A escolha e a análise detalhada das soluções acontecem na **Fase 2**.

## Resultados disponíveis no repositório

A execução de 7 de setembro de 2026 foi concluída: 15 gerações do NSGA-II,
avaliação detalhada da solução final e baseline com os 37 atributos.

- [Relatório completo em PDF](Resultados/figuras/orientadores_final/relatorio_graficos.pdf)
- [Pacote para download: gráficos, tabelas e dados utilizados](Resultados/figuras/orientadores_final.zip)
- [Curva de hipervolume por geração](Resultados/figuras/orientadores_final/00_convergencia_hipervolume.png)
- [Comparação intra/cross com baseline](Resultados/figuras/orientadores_final/06_intra_vs_cross.png)
- [Notas metodológicas e interpretação](Resultados/figuras/orientadores_final/LEIA-ME.md)
- [Pareto final](Resultados/pareto/pareto_20260907_000921_decision_tree.json)
- [Avaliações detalhadas](Resultados/metricas/pareto_20260907_000921_decision_tree/)

O HTML do pacote pode ser aberto localmente após extrair o ZIP. Os resultados
incluem o diagnóstico de convergência; melhora do hipervolume não certifica
ótimo global nem generalização em um teste externo independente.

## Formulação biobjetivo

Cada solução é um vetor real `x` com um gene em `[0, 1]` para cada feature. Eu
mantenho a feature quando `x >= 0.5` e minimizo:

1. `1 - média(F1-macro UNSW→ToN, F1-macro ToN→UNSW)`;
2. `número de features selecionadas / número total de features`.

O F1 de cada direção, o diagnóstico de incompatibilidade de taxonomia e o erro
nas classes conhecidas permanecem separados nos artefatos. A quantidade de
soluções não dominadas não é definida previamente.

## Estrutura

```text
Datasets/                  dois Parquets locais completos
Modulos/
  avaliacao.py             avaliações intra e cross-dataset
  carregamento.py          leitura local e validação dos Parquets
  checkpoint.py            cache e progresso local do NSGA-II
  classificadores.py       fábrica de classificadores
  otimizacao.py            problema biobjetivo e NSGA-II
  preprocessamento.py      limpeza, taxonomia e alinhamento
src/
  algoritmo1_otimizacao.py Fase 1: pré-filtro NSGA-II
  algoritmo2_avaliacao.py  Fase 2: avaliação detalhada
  config.yaml              configuração central
  graficos_cross.py        figuras da análise
tests/                     testes unitários e de integração sintética
Resultados/                Pareto, métricas, checkpoints, logs e figuras
```

## Dados locais

Antes de executar, estes arquivos devem existir:

```text
Datasets/NF-UNSW-NB15-V2.parquet
Datasets/NF-ToN-IoT-V2.parquet
```

O pipeline lê os dois arquivos completos. Não há parâmetro de subamostragem no
fluxo científico.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Execução

Fase 1, com população 24, 15 gerações e operadores padrão do pymoo:

```bash
python src/algoritmo1_otimizacao.py
```

O cache fica em `Resultados/checkpoints`. Para retomar uma interrupção, eu
executo novamente o mesmo comando.

Fase 2, avaliando todas as soluções encontradas:

```bash
python src/algoritmo2_avaliacao.py \
  --pareto Resultados/pareto/pareto_<execucao>.json \
  --todas
```

Também posso avaliar uma solução ou a máscara completa:

```bash
python src/algoritmo2_avaliacao.py --pareto <arquivo.json> --solucao 0
python src/algoritmo2_avaliacao.py --mascara cheia
```

Depois da Fase 2:

```bash
python src/graficos_cross.py
```

## Testes

```bash
pytest
```

Os testes usam bases sintéticas pequenas. Eles não reduzem nem substituem os
datasets usados na execução científica.

## Referência dos datasets

Sarhan, Layeghy e Portmann, *Towards a Standard Feature Set for Network
Intrusion Detection System Datasets*, Mobile Networks and Applications, 2022.

## Execução independente da sessão

O serviço `ids-otimizacao.service` executa a Fase 1 com a configuração de
`src/config.yaml`, reutiliza os checkpoints e reinicia após falhas em 60 segundos.
Após conclusão normal, grava `Resultados/checkpoints/otimizacao.concluida` e
não repete o experimento, inclusive após reiniciar o computador.

Instalação para o usuário atual (projeto em `~/multiobjective-ids-generalization`):

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/ids-otimizacao.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger "$USER"
systemctl --user enable --now ids-otimizacao.service
```

O `enable-linger` mantém o gerenciador do usuário após logout e permite iniciar
no boot sem login; dependendo da máquina, exige autorização administrativa.
Nenhum processo continua executando com a máquina desligada. O serviço retoma
quando ela volta, reaproveitando as avaliações salvas no cache.

Acompanhamento e parada:

```bash
systemctl --user status ids-otimizacao.service
tail -f Resultados/logs/nsga2.log
systemctl --user stop ids-otimizacao.service
```

Para desativar também a inicialização automática:

```bash
systemctl --user disable --now ids-otimizacao.service
```

Uma conclusão bem-sucedida deixa o serviço inativo, com código de saída zero.
Para repetir intencionalmente uma execução concluída, remova apenas o marcador
`Resultados/checkpoints/otimizacao.concluida` e inicie o serviço novamente.
O script usa um lock para impedir execuções simultâneas iniciadas por ele;
não rode o Python diretamente enquanto o serviço estiver executando.

Como alternativa manual, com o serviço desativado:

```bash
nohup bash src/executar_otimizacao.sh >> Resultados/logs/nsga2.log 2>&1 < /dev/null &
```

O `nohup` protege contra desconexão do terminal, mas não fornece reinício
automático após falha ou reboot.

## Material para os orientadores

O relatório do checkpoint atual é gerado sem repetir treinamentos:

```bash
.venv/bin/python src/relatorio_orientadores.py
```

Abra `Resultados/figuras/orientadores_parcial/index.html` ou o
`relatorio_graficos.pdf` nessa pasta. O ZIP ao lado inclui figuras PNG/PDF,
tabela CSV e uma cópia dos dados utilizados. O pacote parcial identifica
explicitamente a geração registrada e não mistura o Pareto do teste antigo.

A continuação automática aguarda o marcador de conclusão da Fase 1, gera os
gráficos finais de Pareto, avalia cada solução e a máscara completa com cinco
folds nos datasets completos, e acrescenta a comparação intra/cross ao pacote
`Resultados/figuras/orientadores_final`. Cada solução detalhada concluída é
salva atomicamente e reutilizada após uma interrupção. A solução interrompida
é recalculada; a retomada da Fase 2 ocorre por solução, não por fold.

```bash
cp deploy/systemd/ids-analise.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ids-analise.service
tail -f Resultados/logs/analise.log
```

Para parar a avaliação e a geração automática:

```bash
systemctl --user disable --now ids-analise.service
```

A otimização usa o serviço separado `ids-otimizacao.service`. As avaliações
ficam em `Resultados/metricas/pareto_<execucao>/`; o marcador
`analise.concluida` nessa pasta indica que todas as soluções, o baseline e os
gráficos foram finalizados. Enquanto isso, o pacote final pode conter apenas
as avaliações detalhadas já concluídas.

### Convergência do NSGA-II

O pacote inclui `00_convergencia_hipervolume.png` (e PDF), uma página de
diagnóstico no relatório e `convergencia_nsga2.csv`. O hipervolume é calculado
sobre a fronteira sobrevivente de cada geração, com referência fixa `(1.1, 1.1)`
e os objetivos originais `1 - F1_cross_medio` e `k/d`, ambos em `[0,1]`.
Não há normalização por geração nem uso de um arquivo acumulado de soluções.

As gerações são reconstruídas com a mesma seed, população e operadores,
consultando somente uma cópia em memória do cache. Não há leitura dos Parquets,
retreinamento ou escrita nos checkpoints. Máscaras ausentes interrompem o
procedimento; a fronteira reconstruída precisa coincidir com a salva antes
que a curva seja publicada. A reconstrução não usa a posição da linha do cache
como se fosse o número da geração.

O diagnóstico mostra progresso nos objetivos; não certifica ótimo global,
estabilidade entre seeds ou generalização independente. O ponto de referência
é mantido fixo em todas as gerações. Base metodológica:
https://pymoo.org/getting_started/part_4.html

Para a análise já em execução, `ids-relatorio.path` observa o marcador de
conclusão da avaliação de 20260907_000921 e atualiza o pacote final com o
baseline, sem interromper o treinamento em curso. As próximas execuções do
gerador já incluem o diagnóstico diretamente.
