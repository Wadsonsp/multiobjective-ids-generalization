# -*- coding: utf-8 -*-
"""Testes do Algoritmo 1 (NSGA-II), do drive_loader e smoke test de
integração do pipeline completo.

O que valido aqui:
- população inicial binária com a forma correta;
- reparo de máscara vazia;
- soluções de P* mutuamente não-dominadas;
- reprodutibilidade com seed fixa;
- localização/validação de arquivos do drive_loader (com arquivos
  temporários, sem rede);
- pipeline ponta a ponta em escala mínima (smoke test).
"""

import os

import numpy as np
import pandas as pd
import pytest

from Modulos.drive_loader import (
    COLUNAS_OBRIGATORIAS_NFV2,
    localizar_arquivo,
    validar_schema_nfv2,
)
from Modulos.otimizacao import (
    ProblemaSelecaoCaracteristicas,
    executar_otimizacao,
    reparar_mascara_vazia,
)

CLF_RAPIDO = "lda"


def _domina(f_a, f_b):
    """a domina b se for <= em tudo e < em pelo menos um objetivo."""
    return np.all(f_a <= f_b) and np.any(f_a < f_b)


class TestReparoMascara:
    def test_mascara_vazia_ganha_um_bit(self):
        rng = np.random.RandomState(0)
        reparada = reparar_mascara_vazia(np.zeros(10, dtype=bool), rng)
        assert reparada.sum() == 1

    def test_mascara_valida_nao_muda(self):
        rng = np.random.RandomState(0)
        original = np.array([True, False, True])
        assert np.array_equal(reparar_mascara_vazia(original, rng), original)


class TestProblema:
    def test_dimensoes_do_problema(self, bases_Xy):
        problema = ProblemaSelecaoCaracteristicas(bases_Xy, CLF_RAPIDO, cv_folds=2)
        d = bases_Xy["BASE_A"][0].shape[1]
        # tri-objetivo (F1 intra, F1 cross, custo) em {0,1}^d
        assert problema.n_var == d
        assert problema.n_obj == 3

    def test_avaliacao_alimenta_historico(self, bases_Xy):
        problema = ProblemaSelecaoCaracteristicas(bases_Xy, CLF_RAPIDO, cv_folds=2)
        out = {}
        x = np.ones(problema.d, dtype=bool)
        problema._evaluate(x, out)
        assert len(out["F"]) == 3
        # Todos os critérios do Algoritmo 2 ficam no histórico
        assert "f1_macro_cross_por_direcao" in problema.historico[0]


class TestExecucaoNSGA2:
    @pytest.fixture(scope="class")
    def resultado(self, request):
        # Execução mínima compartilhada entre os testes da classe
        bases_Xy = request.getfixturevalue("bases_Xy")
        return executar_otimizacao(
            bases_Xy, CLF_RAPIDO, n_pop=8, n_gen=3,
            pc=0.9, pm=0.1, seed=42, cv_folds=2, verbose=False,
        )

    # getfixturevalue não resolve fixtures de função em escopo de classe,
    # então materializo a fixture aqui
    @pytest.fixture(autouse=True)
    def _injeta_bases(self, bases_Xy, request):
        request.cls._bases = bases_Xy

    def test_retorna_solucoes_binarias(self):
        r = executar_otimizacao(
            self._bases, CLF_RAPIDO, n_pop=8, n_gen=2,
            seed=42, cv_folds=2, verbose=False,
        )
        assert r["mascaras"].ndim == 2
        assert set(np.unique(r["mascaras"])).issubset({0, 1})
        assert r["mascaras"].shape[1] == self._bases["BASE_A"][0].shape[1]

    def test_pareto_mutuamente_nao_dominado(self):
        # Definição de P*: nenhuma solução domina outra (linhas 13-14)
        r = executar_otimizacao(
            self._bases, CLF_RAPIDO, n_pop=8, n_gen=3,
            seed=42, cv_folds=2, verbose=False,
        )
        F = r["objetivos"]
        for i in range(len(F)):
            for j in range(len(F)):
                if i != j:
                    assert not _domina(F[i], F[j]), (
                        f"Solução {i} domina {j}: P* inválido"
                    )

    def test_reprodutivel_com_seed(self):
        r1 = executar_otimizacao(
            self._bases, CLF_RAPIDO, n_pop=6, n_gen=2,
            seed=123, cv_folds=2, verbose=False,
        )
        r2 = executar_otimizacao(
            self._bases, CLF_RAPIDO, n_pop=6, n_gen=2,
            seed=123, cv_folds=2, verbose=False,
        )
        assert np.array_equal(r1["mascaras"], r2["mascaras"])

    def test_historico_cobre_todas_as_avaliacoes(self):
        r = executar_otimizacao(
            self._bases, CLF_RAPIDO, n_pop=6, n_gen=2,
            seed=42, cv_folds=2, verbose=False,
        )
        # No mínimo a população inicial inteira foi avaliada
        assert len(r["historico"]) >= 6


class TestDriveLoader:
    def test_localiza_arquivo_local(self, tmp_path):
        arquivo = tmp_path / "base.csv"
        arquivo.write_text("a,b\n1,2\n")
        caminho = localizar_arquivo("base.csv", "/drive/inexistente", str(tmp_path))
        assert caminho == str(arquivo)

    def test_arquivo_inexistente_orienta_setup(self, tmp_path):
        # A mensagem precisa apontar para o script de setup do Drive
        with pytest.raises(FileNotFoundError, match="setup_datasets_drive"):
            localizar_arquivo("nao_existe.csv", "/drive/x", str(tmp_path))

    def test_valida_schema_nfv2_completo(self):
        df = pd.DataFrame({c: [0] for c in COLUNAS_OBRIGATORIAS_NFV2})
        assert validar_schema_nfv2(df) is True

    def test_rejeita_schema_invalido(self):
        df = pd.DataFrame({"qualquer_coisa": [1]})
        with pytest.raises(ValueError, match="NF-v2"):
            validar_schema_nfv2(df, nome_base="TESTE")


class TestIntegracao:
    def test_pipeline_ponta_a_ponta(self, base_a, base_b, config_pre, tmp_path):
        """Smoke test: bases brutas -> pré-processamento -> Algoritmo 1 ->
        P* salvo em disco. Escala mínima só para validar o encadeamento."""
        import json

        from Modulos.preprocessamento import alinhar_colunas, preprocessar_base

        bases = {
            "BASE_A": preprocessar_base(base_a, config_pre),
            "BASE_B": preprocessar_base(base_b, config_pre),
        }
        bases, ordem = alinhar_colunas(bases)

        r = executar_otimizacao(
            bases, CLF_RAPIDO, n_pop=6, n_gen=2,
            seed=42, cv_folds=2, verbose=False,
        )

        # Simulo a gravação do P* como no src/algoritmo1_otimizacao.py
        caminho = tmp_path / "pareto_teste.json"
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump({
                "atributos": ordem,
                "mascaras": r["mascaras"].tolist(),
                "objetivos": r["objetivos"].tolist(),
            }, f)

        assert os.path.exists(caminho)
        with open(caminho, encoding="utf-8") as f:
            salvo = json.load(f)
        assert len(salvo["mascaras"]) == len(salvo["objetivos"]) >= 1
        assert len(salvo["atributos"]) == len(salvo["mascaras"][0])


class TestCarregamentoDataset:
    def _config_fake(self, pasta_local):
        # Config mínimo apontando para um CSV sintético em pasta temporária
        return {
            "datasets": {
                "pasta_drive": "/content/drive/inexistente",
                "pasta_local": str(pasta_local),
                "bases": {"MINI-NFV2": {"arquivo": "mini.csv"}},
            }
        }

    def _csv_nfv2_sintetico(self, caminho):
        df = pd.DataFrame({c: [1, 2, 3] for c in COLUNAS_OBRIGATORIAS_NFV2})
        df.to_csv(caminho, index=False)

    def test_carregar_dataset_valida_e_retorna(self, tmp_path):
        from Modulos.drive_loader import carregar_dataset

        self._csv_nfv2_sintetico(tmp_path / "mini.csv")
        df = carregar_dataset("MINI-NFV2", self._config_fake(tmp_path))
        assert len(df) == 3
        assert "Attack" in df.columns

    def test_carregar_todas_as_bases_monta_dicionario_D(self, tmp_path):
        # D = {nome: DataFrame}, entrada dos Algoritmos 1 e 2
        from Modulos.drive_loader import carregar_todas_as_bases

        self._csv_nfv2_sintetico(tmp_path / "mini.csv")
        D = carregar_todas_as_bases(self._config_fake(tmp_path), nrows=2)
        assert set(D.keys()) == {"MINI-NFV2"}
        assert len(D["MINI-NFV2"]) == 2
