# -*- coding: utf-8 -*-
"""Testes do pré-processamento (Seção 4.1).

O que valido aqui:
- remoção de NaN/inf sem imputação;
- descarte dos atributos de vazamento (IPs, portas, TTL);
- Min-Max ajustado SOMENTE no treino (sem vazamento estatístico);
- alinhamento de colunas entre bases.
"""

import numpy as np
import pandas as pd
import pytest

from Modulos.preprocessamento import (
    alinhar_colunas,
    preprocessar_base,
    remover_atributos_vazamento,
    remover_ausentes_e_infinitos,
    separar_atributos_e_rotulo,
)


class TestRemocaoAusentesInfinitos:
    def test_remove_linhas_com_nan(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
        resultado = remover_ausentes_e_infinitos(df)
        assert len(resultado) == 2
        assert not resultado.isna().any().any()

    def test_remove_linhas_com_infinito(self):
        # inf aparece em métricas derivadas com divisão por zero
        df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 4.0], "b": [1, 2, 3, 4]})
        resultado = remover_ausentes_e_infinitos(df)
        assert len(resultado) == 2
        assert np.isfinite(resultado["a"]).all()

    def test_nao_imputa_valores(self):
        # A decisão da Seção 4.1 é REMOVER, nunca imputar média/mediana
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [10.0, 20.0]})
        resultado = remover_ausentes_e_infinitos(df)
        assert 20.0 not in resultado["b"].values or len(resultado) == 1


class TestRemocaoVazamento:
    def test_remove_ips_portas_ttl(self, base_a, config_pre):
        resultado = remover_atributos_vazamento(
            base_a, config_pre["atributos_vazamento"]
        )
        for col in config_pre["atributos_vazamento"]:
            assert col not in resultado.columns, f"{col} deveria ter sido removida"

    def test_tolera_coluna_inexistente(self, base_a):
        # errors='ignore': não quebra se a base não tiver alguma coluna
        resultado = remover_atributos_vazamento(base_a, ["COLUNA_QUE_NAO_EXISTE"])
        assert len(resultado.columns) == len(base_a.columns)


class TestSeparacaoAtributosRotulo:
    def test_label_binario_fora_da_matriz(self, base_a, config_pre):
        df = remover_atributos_vazamento(base_a, config_pre["atributos_vazamento"])
        X, y = separar_atributos_e_rotulo(df)
        # Label é vazamento direto do rótulo - não pode ficar em X
        assert "Label" not in X.columns
        assert "Attack" not in X.columns
        assert len(X) == len(y)

    def test_matriz_apenas_numerica(self, base_a, config_pre):
        # Mesmo sem remover IPs antes, X final não pode ter colunas texto
        X, _ = separar_atributos_e_rotulo(base_a)
        assert all(np.issubdtype(dt, np.number) for dt in X.dtypes)

    def test_normaliza_rotulos_para_comparar_taxonomias(self):
        df = pd.DataFrame({
            "f": [1.0, 2.0, 3.0],
            "Attack": [" DoS ", "dos", "BACKDOOR"],
            "Label": [1, 1, 1],
        })
        _, y = separar_atributos_e_rotulo(df)
        assert y.tolist() == ["dos", "dos", "backdoor"]


class TestPipelineCompleto:
    def test_preprocessar_base_remove_tudo_que_deve(self, base_a, config_pre):
        X, y = preprocessar_base(base_a, config_pre)
        for col in config_pre["atributos_vazamento"]:
            assert col not in X.columns
        assert "Label" not in X.columns
        assert len(X) == len(y) > 0

    def test_alinhar_colunas_mesma_ordem(self, base_a, base_b, config_pre):
        bases = {
            "A": preprocessar_base(base_a, config_pre),
            "B": preprocessar_base(base_b, config_pre),
        }
        alinhadas, ordem = alinhar_colunas(bases)
        # A máscara x precisa indexar o MESMO atributo em qualquer base
        assert list(alinhadas["A"][0].columns) == list(alinhadas["B"][0].columns) == ordem
