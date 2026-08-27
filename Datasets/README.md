# Datasets locais

Eu mantenho nesta pasta os dois datasets completos usados pelo pipeline:

| Base | Arquivo esperado |
|---|---|
| NF-UNSW-NB15-v2 | `NF-UNSW-NB15-V2.parquet` |
| NF-ToN-IoT-v2 | `NF-ToN-IoT-V2.parquet` |

O código lê os Parquets diretamente e não aplica subamostragem. Os nomes podem
ser alterados em `src/config.yaml` quando necessário.

Referência: Sarhan, Layeghy e Portmann, *Towards a Standard Feature Set for
Network Intrusion Detection System Datasets*, Mobile Networks and Applications,
2022.
