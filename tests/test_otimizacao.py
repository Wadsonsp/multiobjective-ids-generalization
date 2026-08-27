# -*- coding: utf-8 -*-
"""Eu testo o problema biobjetivo, o NSGA-II e o carregamento local."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from Modulos.carregamento import (
    COLUNAS_OBRIGATORIAS_NFV2,
    carregar_dataset,
    carregar_todas_as_bases,
    localizar_arquivo,
    validar_schema_nfv2,
)
from Modulos.otimizacao import ProblemaSelecaoCaracteristicas, executar_otimizacao


CLF_RAPIDO = "decision_tree"


def _domina(f_a, f_b):
    """Eu verifico a definição matemática de dominância de Pareto."""
    return np.all(f_a <= f_b) and np.any(f_a < f_b)


class TestProblemaBiobjetivo:
    def test_dimensoes_e_limites(self, bases_Xy):
        problema = ProblemaSelecaoCaracteristicas(bases_Xy, CLF_RAPIDO)
        d = bases_Xy["BASE_A"][0].shape[1]
        assert problema.n_var == d
        assert problema.n_obj == 2
        assert np.all(problema.xl == 0)
        assert np.all(problema.xu == 1)

    def test_limiar_de_meio_seleciona_features(self, bases_Xy):
        problema = ProblemaSelecaoCaracteristicas(bases_Xy, CLF_RAPIDO)
        x = np.array([0.49, 0.50, 0.90] + [0.1] * (problema.d - 3))
        assert problema.decidir_features_mantidas(x).tolist() == [1, 2]
        assert problema.criar_mascara_binaria(x).sum() == 2

    def test_mascara_vazia_recebe_penalidade(self, bases_Xy):
        problema = ProblemaSelecaoCaracteristicas(bases_Xy, CLF_RAPIDO)
        out = {}
        problema._evaluate(np.zeros(problema.d), out)
        assert out["F"] == [1.0, 1.0]
        assert problema.historico[-1]["avaliacao_valida"] is False

    def test_formula_dos_dois_objetivos(self, bases_Xy, monkeypatch):
        import Modulos.otimizacao as otm

        def avaliacao_controlada(mascara, bases, classificador, seed):
            return {
                "f1_macro_cross_por_direcao": {"A->B": 0.70, "B->A": 0.90},
                "diagnostico_cross": {},
                "tempo_medio_inferencia": 0.001,
                "numero_atributos": int(np.sum(mascara)),
            }

        monkeypatch.setattr(otm, "avaliar_fase1_cross_dataset", avaliacao_controlada)
        problema = otm.ProblemaSelecaoCaracteristicas(bases_Xy, CLF_RAPIDO)
        x = np.zeros(problema.d)
        x[: problema.d // 2] = 0.8
        out = {}
        problema._evaluate(x, out)
        assert out["F"][0] == pytest.approx(0.20)
        assert out["F"][1] == pytest.approx((problema.d // 2) / problema.d)
        assert problema.historico[-1]["f1_macro_cross_medio"] == pytest.approx(0.80)

    def test_exige_exatamente_dois_datasets(self, bases_Xy):
        apenas_uma = {"BASE_A": bases_Xy["BASE_A"]}
        with pytest.raises(ValueError, match="exatamente dois datasets"):
            ProblemaSelecaoCaracteristicas(apenas_uma, CLF_RAPIDO)


class TestExecucaoNSGA2:
    def test_retorna_mascaras_binarias_e_dois_objetivos(self, bases_Xy):
        resultado = executar_otimizacao(
            bases_Xy, CLF_RAPIDO, n_pop=6, n_gen=2,
            seed=42, verbose=False,
        )
        assert resultado["mascaras"].ndim == 2
        assert set(np.unique(resultado["mascaras"])).issubset({0, 1})
        assert resultado["objetivos"].shape[1] == 2
        assert len({tuple(m) for m in resultado["mascaras"]}) == len(
            resultado["mascaras"]
        )

    def test_pareto_mutuamente_nao_dominado(self, bases_Xy):
        resultado = executar_otimizacao(
            bases_Xy, CLF_RAPIDO, n_pop=8, n_gen=3,
            seed=42, verbose=False,
        )
        objetivos = resultado["objetivos"]
        for i in range(len(objetivos)):
            for j in range(len(objetivos)):
                if i != j:
                    assert not _domina(objetivos[i], objetivos[j])

    def test_reprodutivel_com_seed(self, bases_Xy):
        kwargs = dict(n_pop=6, n_gen=2, seed=123, verbose=False)
        r1 = executar_otimizacao(bases_Xy, CLF_RAPIDO, **kwargs)
        r2 = executar_otimizacao(bases_Xy, CLF_RAPIDO, **kwargs)
        assert np.array_equal(r1["mascaras"], r2["mascaras"])
        assert np.allclose(r1["objetivos"], r2["objetivos"])


class TestCarregamentoLocal:
    def _config_fake(self, pasta_local, arquivo="mini.csv"):
        return {
            "datasets": {
                "pasta_local": str(pasta_local),
                "bases": {"MINI-NFV2": {"arquivo": arquivo}},
            }
        }

    def _base_nfv2_sintetica(self):
        return pd.DataFrame({c: [1, 2, 3] for c in COLUNAS_OBRIGATORIAS_NFV2})

    def test_localiza_arquivo_local(self, tmp_path):
        arquivo = tmp_path / "base.csv"
        arquivo.write_text("a,b\n1,2\n")
        assert localizar_arquivo("base.csv", str(tmp_path)) == str(arquivo)

    def test_arquivo_inexistente_orienta_pasta_local(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Datasets"):
            localizar_arquivo("nao_existe.parquet", str(tmp_path))

    def test_valida_schema_nfv2(self):
        assert validar_schema_nfv2(self._base_nfv2_sintetica()) is True
        with pytest.raises(ValueError, match="NF-v2"):
            validar_schema_nfv2(pd.DataFrame({"qualquer": [1]}))

    def test_carrega_csv_local(self, tmp_path):
        self._base_nfv2_sintetica().to_csv(tmp_path / "mini.csv", index=False)
        df = carregar_dataset("MINI-NFV2", self._config_fake(tmp_path))
        assert len(df) == 3

    def test_carrega_parquet_local(self, tmp_path):
        self._base_nfv2_sintetica().to_parquet(tmp_path / "mini.parquet", index=False)
        config = self._config_fake(tmp_path, arquivo="mini.parquet")
        df = carregar_dataset("MINI-NFV2", config, nrows=2)
        assert len(df) == 2

    def test_carrega_dicionario_de_bases(self, tmp_path):
        self._base_nfv2_sintetica().to_csv(tmp_path / "mini.csv", index=False)
        bases = carregar_todas_as_bases(self._config_fake(tmp_path), nrows=2)
        assert set(bases) == {"MINI-NFV2"}
        assert len(bases["MINI-NFV2"]) == 2


class TestIntegracao:
    def test_pipeline_sintetico_ate_pareto_salvo(
        self, base_a, base_b, config_pre, tmp_path
    ):
        from Modulos.preprocessamento import alinhar_colunas, preprocessar_base

        bases = {
            "BASE_A": preprocessar_base(base_a, config_pre),
            "BASE_B": preprocessar_base(base_b, config_pre),
        }
        bases, ordem = alinhar_colunas(bases)
        resultado = executar_otimizacao(
            bases, CLF_RAPIDO, n_pop=6, n_gen=2,
            seed=42, verbose=False,
        )
        caminho = tmp_path / "pareto_teste.json"
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump({
                "atributos": ordem,
                "mascaras": resultado["mascaras"].tolist(),
                "objetivos": resultado["objetivos"].tolist(),
            }, f)
        assert os.path.exists(caminho)
        with open(caminho, encoding="utf-8") as f:
            salvo = json.load(f)
        assert len(salvo["mascaras"]) == len(salvo["objetivos"]) >= 1
