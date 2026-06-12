# Datasets

Esta pasta fica **vazia no repositório** (apenas cache local opcional). Os arquivos oficiais ficam no Google Drive em `MyDrive/mestrado/Datasets`.

| Base | Arquivo | Fonte oficial (UQ) |
|---|---|---|
| NF-UNSW-NB15-v2 | `NF-UNSW-NB15-v2.csv` | https://rdm.uq.edu.au/files/8c6e2a00-ef9c-11ed-827d-e762de186848 |
| NF-ToN-IoT-v2 | `NF-ToN-IoT-v2.csv` | https://rdm.uq.edu.au/files/a4ad7080-ef9c-11ed-a964-b70596e96ad5 |

Para popular o Drive, rodar **uma vez no Google Colab**:

```
!git clone https://github.com/Wadsonsp/multiobjective-ids-generalization.git
%cd multiobjective-ids-generalization
!python src/setup_datasets_drive.py
```

Citação obrigatória (uso acadêmico): Sarhan, Layeghy e Portmann, *Towards a Standard Feature Set for Network Intrusion Detection System Datasets*, Mobile Networks and Applications, 2022.
