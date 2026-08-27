# -*- coding: utf-8 -*-
"""Testes do checkpoint (cache de avaliações + progresso por geração).

O que valido aqui:
- gravação imediata e releitura do cache (sobrevive a "nova sessão");
- linha truncada (queda no meio da escrita) ignorada sem corromper;
- contexto experimental divergente é rejeitado com erro claro;
- RETOMADA: reexecutar a mesma otimização não recomputa nada (zero
  avaliações novas) e devolve o MESMO P*;
- progresso por geração gravado de forma atômica.
"""

import json
import os

import numpy as np
import pytest

import Modulos.otimizacao as otm
from Modulos.checkpoint import CacheAvaliacoes, RegistroProgresso, chave_da_mascara

CONTEXTO = {"classificador": "decision_tree", "seed": 42}
CLF_RAPIDO = "decision_tree"


class TestCacheAvaliacoes:
    def test_registra_e_recupera(self, tmp_path):
        cache = CacheAvaliacoes(str(tmp_path / "c.jsonl"), CONTEXTO)
        mascara = np.array([1, 0, 1])
        cache.registrar(mascara, {"f1": 0.9})
        assert cache.obter(mascara) == {"f1": 0.9}
        assert cache.obter(np.array([0, 1, 1])) is None

    def test_sobrevive_a_nova_sessao(self, tmp_path):
        # Eu simulo uma nova execução: outro objeto, mesmo arquivo local.
        caminho = str(tmp_path / "c.jsonl")
        CacheAvaliacoes(caminho, CONTEXTO).registrar([1, 1, 0], {"f1": 0.8})
        cache2 = CacheAvaliacoes(caminho, CONTEXTO)
        assert len(cache2) == 1
        assert cache2.obter([1, 1, 0]) == {"f1": 0.8}

    def test_linha_truncada_e_ignorada(self, tmp_path):
        # Queda NO MEIO de uma escrita: a última linha fica pela metade
        caminho = str(tmp_path / "c.jsonl")
        cache = CacheAvaliacoes(caminho, CONTEXTO)
        cache.registrar([1, 0], {"f1": 0.7})
        with open(caminho, "a", encoding="utf-8") as f:
            f.write('{"mascara": "01", "criterios": {"f1": 0.')  # truncada
        cache2 = CacheAvaliacoes(caminho, CONTEXTO)
        assert len(cache2) == 1            # só a íntegra sobrevive
        assert cache2.obter([0, 1]) is None  # a truncada será recalculada

    def test_contexto_divergente_e_rejeitado(self, tmp_path):
        # Misturar caches de experimentos diferentes corromperia tudo
        caminho = str(tmp_path / "c.jsonl")
        CacheAvaliacoes(caminho, CONTEXTO)
        with pytest.raises(ValueError, match="OUTRO contexto"):
            CacheAvaliacoes(caminho, {"classificador": "outro", "seed": 42})

    def test_chave_canonica(self):
        assert chave_da_mascara([1, 0, 1, 1]) == "1011"
        assert chave_da_mascara(np.array([True, False])) == "10"


class TestRetomadaExecucao:
    def test_segunda_execucao_nao_recomputa_e_repete_pareto(
        self, bases_Xy, tmp_path, monkeypatch
    ):
        """O coração da retomada: com seed fixa + cache, reexecutar é um
        replay. Conto as chamadas reais ao Algoritmo 2: a 2a execução
        completa deve fazer ZERO avaliações novas e devolver o mesmo P*."""
        caminho = str(tmp_path / "cache.jsonl")
        chamadas = {"n": 0}
        original = otm.avaliar_fase1_cross_dataset

        def avaliar_contando(*args, **kwargs):
            chamadas["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(otm, "avaliar_fase1_cross_dataset", avaliar_contando)

        cache1 = CacheAvaliacoes(caminho, CONTEXTO)
        r1 = otm.executar_otimizacao(
            bases_Xy, CLF_RAPIDO, n_pop=6, n_gen=2, seed=42,
            verbose=False, cache=cache1,
        )
        n_primeira = chamadas["n"]
        assert n_primeira > 0
        assert len(cache1) == n_primeira  # tudo que computou foi gravado

        # "Nova sessão": cache recarregado do disco, mesma configuração
        chamadas["n"] = 0
        cache2 = CacheAvaliacoes(caminho, CONTEXTO)
        r2 = otm.executar_otimizacao(
            bases_Xy, CLF_RAPIDO, n_pop=6, n_gen=2, seed=42,
            verbose=False, cache=cache2,
        )
        assert chamadas["n"] == 0, "retomada recomputou avaliações do cache"
        assert np.array_equal(r1["mascaras"], r2["mascaras"])
        assert np.allclose(r1["objetivos"], r2["objetivos"])


class TestRegistroProgresso:
    class _OptFake:
        def get(self, campo):
            return {"X": np.array([[1, 0], [0, 1]]),
                    "F": np.array([[0.1, 0.2], [0.2, 0.1]])}[campo]

    class _AlgoritmoFake:
        n_gen = 7

        class evaluator:
            n_eval = 42

        opt = None

    def test_grava_fronteira_parcial(self, tmp_path):
        caminho = str(tmp_path / "progresso.json")
        registro = RegistroProgresso(caminho)
        alg = self._AlgoritmoFake()
        alg.opt = self._OptFake()
        registro(alg)  # pymoo chama o callback como função

        with open(caminho, encoding="utf-8") as f:
            estado = json.load(f)
        assert estado["geracao"] == 7
        assert estado["avaliacoes"] == 42
        assert len(estado["mascaras_parciais"]) == 2
        # Escrita atômica: o temporário não pode sobrar
        assert not os.path.exists(caminho + ".tmp")

    def test_integra_com_nsga2_real(self, bases_Xy, tmp_path):
        # Progresso gravado durante uma execução real do pymoo
        caminho = str(tmp_path / "progresso.json")
        otm.executar_otimizacao(
            bases_Xy, CLF_RAPIDO, n_pop=6, n_gen=3, seed=42,
            verbose=False, registro_progresso=RegistroProgresso(caminho),
        )
        with open(caminho, encoding="utf-8") as f:
            estado = json.load(f)
        assert estado["geracao"] == 3  # última geração registrada
        assert len(estado["mascaras_parciais"]) >= 1
