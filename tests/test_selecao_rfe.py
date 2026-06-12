# -*- coding: utf-8 -*-
"""Testes do baseline RFE (Capítulo 5).

O que valido aqui:
- a máscara tem exatamente n_atributos posições ligadas;
- o formato é compatível com o Algoritmo 2 (mesmo d, binária);
- o RFE elimina preferencialmente atributos de ruído puro;
- reprodutibilidade com seed;
- caso degenerado (alvo >= d) devolve a máscara cheia;
- integração: a máscara do RFE passa pelo AvaliarFitness sem ajustes.
"""

import numpy as np
import pandas as pd
import pytest

from Modulos.avaliacao import avaliar_fitness
from Modulos.selecao_rfe import ajustar_mascara_rfe


@pytest.fixture
def base_com_ruido(bases_Xy):
    """Base A acrescida de 4 colunas de ruído puro (sem relação com y).

    Servem para verificar se o RFE de fato descarta atributos inúteis.
    """
    X, y = bases_Xy["BASE_A"]
    rng = np.random.RandomState(99)
    X = X.copy()
    for j in range(4):
        X[f"RUIDO_{j}"] = rng.normal(size=len(X))
    return X, y


class TestMascaraRFE:
    def test_seleciona_exatamente_n_atributos(self, base_com_ruido):
        X, y = base_com_ruido
        mascara, _ = ajustar_mascara_rfe(X, y, n_atributos=5, seed=42)
        assert mascara.shape[0] == X.shape[1]
        assert int(mascara.sum()) == 5
        assert set(np.unique(mascara)).issubset({0, 1})

    def test_elimina_ruido_antes_dos_informativos(self, base_com_ruido):
        # Com 8 atributos informativos + 4 de ruído, pedir 8 deve
        # descartar majoritariamente o ruído
        X, y = base_com_ruido
        mascara, _ = ajustar_mascara_rfe(X, y, n_atributos=8, seed=42)
        colunas_mantidas = [c for c, b in zip(X.columns, mascara) if b]
        n_ruido_mantido = sum(1 for c in colunas_mantidas if c.startswith("RUIDO_"))
        assert n_ruido_mantido <= 1, (
            f"RFE manteve ruído demais: {colunas_mantidas}"
        )

    def test_reprodutivel_com_seed(self, base_com_ruido):
        X, y = base_com_ruido
        m1, _ = ajustar_mascara_rfe(X, y, n_atributos=5, seed=7)
        m2, _ = ajustar_mascara_rfe(X, y, n_atributos=5, seed=7)
        assert np.array_equal(m1, m2)

    def test_alvo_maior_que_d_retorna_mascara_cheia(self, bases_Xy):
        # Caso degenerado: nada a eliminar (equivale ao baseline sem RFE)
        X, y = bases_Xy["BASE_A"]
        mascara, ranking = ajustar_mascara_rfe(X, y, n_atributos=X.shape[1] + 10)
        assert int(mascara.sum()) == X.shape[1]
        assert all(r == 1 for r in ranking)

    def test_subamostra_para_o_ajuste(self, base_com_ruido):
        # O fit com subamostra precisa funcionar e manter o formato
        X, y = base_com_ruido
        mascara, _ = ajustar_mascara_rfe(
            X, y, n_atributos=5, seed=42, amostra=200
        )
        assert int(mascara.sum()) == 5


class TestIntegracaoComAlgoritmo2:
    def test_mascara_rfe_passa_pelo_avaliar_fitness(self, bases_Xy):
        # A comparação baseline x Pareto exige o MESMO protocolo:
        # a máscara do RFE entra direto no Algoritmo 2
        X, y = bases_Xy["BASE_A"]
        mascara, _ = ajustar_mascara_rfe(X, y, n_atributos=4, seed=42)
        criterios = avaliar_fitness(mascara, bases_Xy, "lda", cv_folds=2, seed=42)
        assert criterios["numero_atributos"] == 4
        assert len(criterios["f1_macro_cross_por_direcao"]) == 2
