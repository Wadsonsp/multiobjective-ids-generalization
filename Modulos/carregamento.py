# -*- coding: utf-8 -*-
"""Eu carrego localmente os datasets NF-v2 usados no experimento.

Eu resolvo caminhos relativos a partir da raiz do projeto, reconheço CSV e
Parquet pela extensão e valido o schema logo após a leitura. Dessa forma, eu
interrompo a execução com uma mensagem clara antes de iniciar uma otimização
longa com um arquivo incorreto.
"""

from pathlib import Path

import pandas as pd


RAIZ_PROJETO = Path(__file__).resolve().parents[1]

# Eu valido identificadores, rótulos e atributos característicos do schema
# NF-v2 antes de permitir que a execução avance.
COLUNAS_OBRIGATORIAS_NFV2 = [
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


def localizar_arquivo(nome_arquivo, pasta_local):
    """Eu localizo um dataset dentro da pasta local configurada."""
    pasta = Path(pasta_local)
    if not pasta.is_absolute():
        pasta = RAIZ_PROJETO / pasta
    caminho = pasta / nome_arquivo
    if caminho.is_file():
        return str(caminho)
    raise FileNotFoundError(
        f"Eu não encontrei o dataset '{nome_arquivo}' em '{pasta}'. "
        "Copie os dois arquivos Parquet para a pasta Datasets do projeto "
        "ou corrija datasets.pasta_local em src/config.yaml."
    )


def validar_schema_nfv2(df, nome_base=""):
    """Eu interrompo cedo quando o arquivo não possui o schema NF-v2."""
    faltantes = [c for c in COLUNAS_OBRIGATORIAS_NFV2 if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"O arquivo da base {nome_base} não parece ser NF-v2: "
            f"colunas ausentes: {faltantes}"
        )
    return True


def carregar_dataset(nome_base, config, nrows=None):
    """Eu carrego um dataset pelo nome definido no arquivo de configuração.

    Eu uso ``nrows`` somente nos testes automatizados. Os executáveis do
    experimento não informam esse argumento e, portanto, leem todas as linhas.
    """
    cfg_ds = config["datasets"]
    info = cfg_ds["bases"][nome_base]
    caminho = localizar_arquivo(info["arquivo"], cfg_ds["pasta_local"])
    extensao = Path(caminho).suffix.lower()

    # Eu escolho o leitor pela extensão para usar diretamente os Parquets locais.
    if extensao in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        arquivo_parquet = pq.ParquetFile(caminho)
        nomes_schema = arquivo_parquet.schema_arrow.names
        # Eu valido o schema sem carregar milhões de linhas na memória.
        validar_schema_nfv2(pd.DataFrame(columns=nomes_schema), nome_base)

        # Eu não leio as colunas que o pré-processamento descartaria logo em
        # seguida. Todas as linhas continuam sendo usadas no experimento.
        cfg_pre = config.get("preprocessamento", {})
        ignoradas = set(cfg_pre.get("atributos_vazamento", []))
        coluna_binaria = cfg_pre.get("coluna_binaria")
        if coluna_binaria:
            ignoradas.add(coluna_binaria)
        colunas_leitura = [c for c in nomes_schema if c not in ignoradas]

        if nrows is None:
            df = pd.read_parquet(caminho, columns=colunas_leitura)
        else:
            # Eu leio somente o primeiro lote nos testes e verificações rápidas.
            lotes = arquivo_parquet.iter_batches(
                batch_size=nrows, columns=colunas_leitura
            )
            df = next(lotes).to_pandas().head(nrows).copy()
    elif extensao == ".csv":
        # Eu mantenho CSV apenas para testes e pequenos arquivos legados.
        df = pd.read_csv(caminho, nrows=nrows, low_memory=False)
        validar_schema_nfv2(df, nome_base)
    else:
        raise ValueError(
            f"Eu não sei ler a extensão '{extensao}' do dataset {nome_base}. "
            "Use um arquivo .parquet, .pq ou .csv."
        )
    return df


def carregar_todas_as_bases(config, nrows=None):
    """Eu monto o dicionário com exatamente as bases configuradas."""
    return {
        nome: carregar_dataset(nome, config, nrows=nrows)
        for nome in config["datasets"]["bases"]
    }
