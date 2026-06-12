# -*- coding: utf-8 -*-
"""Testes do pré-processamento (Seção 4.1).

O que valido aqui:
- remoção de NaN/inf sem imputação;
- descarte dos atributos de vazamento (IPs, portas, TTL);
- Min-Max ajustado SOMENTE no treino (sem vazamento estatístico);
- amostragem estratificada preservando proporções e classes minoritárias;
- alinhamento de colunas entre bases.
"""

import numpy as np
import pandas as pd
import pytest

from Modulos.preprocessamento import (
    ajustar_e_aplicar_minmax,
    alinhar_colunas,
    amostra_estratificada,
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


class TestMinMax:
    def test_treino_no_intervalo_01(self):
        X_tr = pd.DataFrame({"a": [0.0, 5.0, 10.0]})
        X_te = pd.DataFrame({"a": [2.5, 7.5]})
        tr, te, _ = ajustar_e_aplicar_minmax(X_tr, X_te)
        assert tr.min() >= 0.0 and tr.max() <= 1.0

    def test_parametros_ajustados_so_no_treino(self):
        # Se o scaler visse o teste, o 20.0 viraria o novo máximo (=1.0).
        # Como ajusto só no treino, 20.0 extrapola e é clipado em 1.0,
        # e o 10.0 do treino continua sendo o máximo (=1.0).
        X_tr = pd.DataFrame({"a": [0.0, 10.0]})
        X_te = pd.DataFrame({"a": [20.0]})
        tr, te, scaler = ajustar_e_aplicar_minmax(X_tr, X_te)
        assert scaler.data_max_[0] == 10.0  # máximo veio do TREINO
        assert te[0, 0] == 1.0              # teste extrapolado foi clipado

    def test_clip_em_cenario_cross_dataset(self):
        # Valores fora do range do treino são comuns no cross-dataset
        X_tr = pd.DataFrame({"a": [1.0, 2.0]})
        X_te = pd.DataFrame({"a": [-50.0, 100.0]})
        _, te, _ = ajustar_e_aplicar_minmax(X_tr, X_te)
        assert te.min() >= 0.0 and te.max() <= 1.0


class TestAmostraEstratificada:
    def test_preserva_proporcoes(self, base_a):
        amostra = amostra_estratificada(base_a, 300, coluna_rotulo="Attack", seed=42)
        prop_original = base_a["Attack"].value_counts(normalize=True)
        prop_amostra = amostra["Attack"].value_counts(normalize=True)
        for classe in prop_original.index:
            # tolerância de 5 pontos percentuais para a base pequena
            assert abs(prop_original[classe] - prop_amostra[classe]) < 0.05

    def test_preserva_classes_minoritarias(self):
        # Piso de 1 amostra por classe: Ransomware/MITM não podem sumir
        df = pd.DataFrame({
            "f": range(1000),
            "Attack": ["Benign"] * 990 + ["Raro"] * 10,
        })
        amostra = amostra_estratificada(df, 100, coluna_rotulo="Attack", seed=42)
        assert "Raro" in amostra["Attack"].values

    def test_n_maior_que_base_retorna_tudo(self, base_a):
        amostra = amostra_estratificada(base_a, 10_000_000, coluna_rotulo="Attack")
        assert len(amostra) == len(base_a)

    def test_reprodutivel_com_seed(self, base_a):
        a1 = amostra_estratificada(base_a, 300, coluna_rotulo="Attack", seed=7)
        a2 = amostra_estratificada(base_a, 300, coluna_rotulo="Attack", seed=7)
        pd.testing.assert_frame_equal(a1, a2)


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
