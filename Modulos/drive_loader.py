# -*- coding: utf-8 -*-
"""Carregamento dos datasets NF-v2 a partir do Google Drive.

A lógica de localização segue a ordem:
1. Se estiver rodando no Colab, monto o Drive e leio direto de
   MyDrive/mestrado/Datasets (pasta oficial dos dados do mestrado).
2. Fora do Colab, procuro um cache local na pasta Datasets/ do projeto.
3. Se o arquivo não existir em nenhum dos dois, oriento a execução do
   script src/setup_datasets_drive.py (download das fontes oficiais da UQ).

Mantive a validação de schema aqui para falhar cedo: se o CSV não tiver as
colunas NF-v2 esperadas, não adianta seguir para o pré-processamento.
"""

import os

import pandas as pd

# Colunas mínimas que preciso encontrar em qualquer base NF-v2 para
# considerar o arquivo válido (identificadores, rótulos e alguns atributos
# de fluxo característicos do schema de 43 atributos).
COLUNAS_OBRIGATORIAS_NFV2 = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "IN_BYTES",
    "OUT_BYTES",
    "FLOW_DURATION_MILLISECONDS",
    "MIN_TTL",
    "MAX_TTL",
    "Label",
    "Attack",
]


def _esta_no_colab():
    """Detecto o ambiente Colab pela presença do módulo google.colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def montar_drive_se_necessario():
    """Monta o Google Drive quando estou no Colab (idempotente)."""
    if _esta_no_colab():
        from google.colab import drive
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
        return True
    return False


def localizar_arquivo(nome_arquivo, pasta_drive, pasta_local):
    """Retorna o caminho do dataset, priorizando o Drive (Colab).

    A ideia é poder rodar o mesmo código no Colab e na máquina local sem
    mudar nada: só a origem do arquivo muda.
    """
    candidatos = []
    if _esta_no_colab():
        montar_drive_se_necessario()
        candidatos.append(os.path.join(pasta_drive, nome_arquivo))
    candidatos.append(os.path.join(pasta_local, nome_arquivo))

    for caminho in candidatos:
        if os.path.exists(caminho):
            return caminho

    raise FileNotFoundError(
        f"Dataset '{nome_arquivo}' não encontrado em {candidatos}. "
        "Execute src/setup_datasets_drive.py no Colab para baixar as bases "
        "oficiais para MyDrive/mestrado/Datasets, ou copie o CSV para a "
        "pasta Datasets/ do projeto."
    )


def validar_schema_nfv2(df, nome_base=""):
    """Falha cedo se o CSV não tiver o schema NF-v2 esperado."""
    faltantes = [c for c in COLUNAS_OBRIGATORIAS_NFV2 if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"O arquivo da base {nome_base} não parece ser NF-v2: "
            f"colunas ausentes: {faltantes}"
        )
    return True


def carregar_dataset(nome_base, config, nrows=None):
    """Carrega um dataset NF-v2 pelo nome definido no config.yaml.

    Parameters
    ----------
    nome_base : str
        Chave em config['datasets']['bases'] (ex.: 'NF-UNSW-NB15-v2').
    config : dict
        Configuração carregada do config.yaml.
    nrows : int, optional
        Limite de linhas (uso apenas para testes rápidos/exploração).
    """
    cfg_ds = config["datasets"]
    info = cfg_ds["bases"][nome_base]
    caminho = localizar_arquivo(
        info["arquivo"], cfg_ds["pasta_drive"], cfg_ds["pasta_local"]
    )
    # low_memory=False evita inferência de dtype em chunks, que gera
    # colunas mistas nas bases grandes.
    df = pd.read_csv(caminho, nrows=nrows, low_memory=False)
    validar_schema_nfv2(df, nome_base)
    return df


def carregar_todas_as_bases(config, nrows=None):
    """Carrega o dicionário D = {nome: DataFrame} usado nos Algoritmos 1 e 2."""
    return {
        nome: carregar_dataset(nome, config, nrows=nrows)
        for nome in config["datasets"]["bases"]
    }
