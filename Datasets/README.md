# Meus datasets locais

Eu mantenho meus arquivos de dados em `Datasets/`, resolvido a partir da raiz
do projeto. Nesta máquina, eu os acesso em:

```text
/home/wadson.pereira/multiobjective-ids-generalization/Datasets/
```

## O que já utilizei e o que estou preparando

| Base que utilizo na investigação | Meu arquivo local | Situação que verifiquei |
|---|---|---|
| NF-UNSW-NB15-v2 | `NF-UNSW-NB15-V2.parquet` | Eu concluí o experimento com duas bases. |
| NF-ToN-IoT-v2 | `NF-ToN-IoT-V2.parquet` | Eu concluí o experimento com duas bases. |
| NF-CSE-CIC-IDS2018-v2 | `NF-CSE-CIC-IDS2018-v2.zip` | Eu validei o ZIP e converti seu CSV integralmente em `NF-CSE-CIC-IDS2018-V2.parquet`. |

Eu conferi 1.986.745 linhas no Parquet local do UNSW e 13.135.881 no do ToN-IoT.
Após a limpeza, utilizei 1.986.745 e 13.135.750 registros, respectivamente. Eu
removi 131 registros inválidos no segundo arquivo e não apliquei subamostragem.
Essas contagens descrevem meus arquivos locais e não substituem a verificação
de procedência contra as versões publicadas.

## Como vou preparar a terceira base

Eu baixei o [ZIP oficial NF-CSE-CIC-IDS2018-v2](https://rdm.uq.edu.au/files/ce5161d0-ef9c-11ed-827d-e762de186848),
com 612.059.380 bytes. Eu verifiquei que ele contém o CSV
`data/NF-CSE-CIC-IDS2018-v2.csv`, com 3.221.378.977 bytes, dentro da pasta do pacote,
além de `NetFlow_v2_Features.csv` e manifestos.

Eu li o CSV diretamente do ZIP e converti em lotes para
`NF-CSE-CIC-IDS2018-V2.parquet`, sem manter outra cópia descompactada em disco.
Eu conferi 18.893.708 registros em ambos os formatos e gravei os hashes SHA-256
em `NF-CSE-CIC-IDS2018-V2.proveniencia.json`. Eu gravei as colunas numéricas em
`float64` no Parquet; após a limpeza, uso `float32` nas features da nova rodada
para reduzir memória, compatível com a entrada interna da árvore. Eu registro
as remoções e o schema comum em `dados_preprocessamento.json` do experimento.

Eu mantenho em `src/config.yaml` a configuração ativa do experimento com duas
bases. Eu preparo a nova rodada com `src/config_tres_datasets.yaml`; eu seleciono
explicitamente essa configuração no serviço `ids-tres-datasets.service`.

Eu descrevo as seis direções de transferência, a mudança do objetivo e a
separação dos checkpoints no [meu protocolo com três datasets](../docs/experimento_tres_datasets.md).

## Referência que utilizo

Sarhan, Layeghy e Portmann, *Towards a Standard Feature Set for Network
Intrusion Detection System Datasets*, Mobile Networks and Applications, 2022.

## O que observei no carregamento das três bases

Eu confirmei 18.893.455 registros válidos no novo dataset após remover 253
registros inválidos. Somando as três bases, eu utilizo 34.015.950 registros
válidos e 37 atributos comuns. Eu iniciei a otimização com seis direções no
serviço `ids-tres-datasets.service`, em uma pasta própria para esta rodada.
