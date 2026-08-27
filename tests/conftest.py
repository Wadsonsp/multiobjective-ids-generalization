# -*- coding: utf-8 -*-
"""Fixtures dos testes: bases sintéticas que imitam o schema NF-v2.

Crio duas "bases" pequenas e separáveis (papel do NF-UNSW-NB15-v2 e do
NF-ToN-IoT-v2) com as mesmas colunas, incluindo os identificadores que
o pré-processamento precisa descartar (IPs, portas, TTL). Assim os
testes exercitam o pipeline completo sem carregar os Parquets reais.
"""

import numpy as np
import pandas as pd
import pytest

# Atributos numéricos "candidatos" das bases sintéticas
N_ATRIBUTOS = 8
CLASSES = ["Benign", "DoS", "Scanning"]

CONFIG_PRE_TESTE = {
    "atributos_vazamento": [
        "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
        "L4_SRC_PORT", "L4_DST_PORT",
        "MIN_TTL", "MAX_TTL",
    ],
    "coluna_rotulo": "Attack",
    "coluna_binaria": "Label",
}


def _base_sintetica(n=600, seed=0, deslocamento=0.0):
    """Gera uma base com classes separáveis por deslocamento de média.

    O parâmetro 'deslocamento' simula o domain shift entre as bases:
    mesma estrutura, distribuições diferentes.
    """
    rng = np.random.RandomState(seed)
    linhas = []
    for i, classe in enumerate(CLASSES):
        n_classe = n // len(CLASSES)
        centro = i * 2.0 + deslocamento
        feats = rng.normal(loc=centro, scale=0.5, size=(n_classe, N_ATRIBUTOS))
        for f in feats:
            linhas.append(list(f) + [classe])

    colunas_feat = [f"FEAT_{j}" for j in range(N_ATRIBUTOS)]
    df = pd.DataFrame(linhas, columns=colunas_feat + ["Attack"])

    # Colunas identificadoras que DEVEM ser removidas pelo pré-processamento
    df["IPV4_SRC_ADDR"] = "10.0.0.1"
    df["IPV4_DST_ADDR"] = "10.0.0.2"
    df["L4_SRC_PORT"] = rng.randint(1024, 65535, size=len(df))
    df["L4_DST_PORT"] = rng.randint(1, 1024, size=len(df))
    df["MIN_TTL"] = 64
    df["MAX_TTL"] = 128
    df["Label"] = (df["Attack"] != "Benign").astype(int)

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture
def config_pre():
    return dict(CONFIG_PRE_TESTE)


@pytest.fixture
def base_a():
    """Base sintética no papel do NF-UNSW-NB15-v2."""
    return _base_sintetica(n=600, seed=1, deslocamento=0.0)


@pytest.fixture
def base_b():
    """Base sintética no papel do NF-ToN-IoT-v2 (com domain shift leve)."""
    return _base_sintetica(n=600, seed=2, deslocamento=0.3)


@pytest.fixture
def bases_Xy(base_a, base_b, config_pre):
    """Dicionário D = {nome: (X, y)} já pré-processado e alinhado."""
    from Modulos.preprocessamento import alinhar_colunas, preprocessar_base

    bases = {
        "BASE_A": preprocessar_base(base_a, config_pre),
        "BASE_B": preprocessar_base(base_b, config_pre),
    }
    bases, _ = alinhar_colunas(bases)
    return bases
