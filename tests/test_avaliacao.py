# -*- coding: utf-8 -*-
"""Eu testo a avaliação detalhada e a árvore usada no experimento.

O que valido aqui:
- aplicação correta da máscara binária (linha 1 do pseudocódigo);
- k(x) = soma da máscara;
- CV estratificada e métricas intra-dataset (linhas 2-4);
- cross-dataset registrado POR DIREÇÃO, sem média única (linhas 5-10);
- custo computacional medido (linhas 11-13);
- estrutura completa da saída (linhas 14-18);
- configuração exata da árvore de decisão do problema original.
"""

import numpy as np
import pytest

from Modulos.avaliacao import (
    aplicar_mascara,
    avaliar_cross_dataset,
    avaliar_fitness,
    avaliar_intra_dataset,
    numero_de_atributos,
)
from Modulos.classificadores import NOMES_VALIDOS, criar_classificador

CLF_RAPIDO = "decision_tree"


class TestMascara:
    def test_aplica_mascara_seleciona_colunas_certas(self, bases_Xy):
        X, _ = bases_Xy["BASE_A"]
        mascara = np.zeros(X.shape[1], dtype=int)
        mascara[0] = 1
        mascara[2] = 1
        X_sel = aplicar_mascara(X, mascara)
        assert list(X_sel.columns) == [X.columns[0], X.columns[2]]

    def test_mascara_vazia_levanta_erro(self, bases_Xy):
        X, _ = bases_Xy["BASE_A"]
        with pytest.raises(ValueError):
            aplicar_mascara(X, np.zeros(X.shape[1], dtype=int))

    def test_mascara_tamanho_errado_levanta_erro(self, bases_Xy):
        X, _ = bases_Xy["BASE_A"]
        with pytest.raises(ValueError):
            aplicar_mascara(X, np.ones(X.shape[1] + 3, dtype=int))

    def test_k_de_x_e_a_soma_da_mascara(self):
        # k(x) = somatório de x_j (Seção 4.2)
        assert numero_de_atributos([1, 0, 1, 1, 0]) == 3
        assert numero_de_atributos(np.ones(10)) == 10


class TestIntraDataset:
    def test_metricas_no_intervalo_valido(self, bases_Xy):
        X, y = bases_Xy["BASE_A"]
        mascara = np.ones(X.shape[1], dtype=int)
        r = avaliar_intra_dataset(mascara, X, y, CLF_RAPIDO, cv_folds=3, seed=42)
        assert 0.0 <= r["f1_macro"] <= 1.0
        assert 0.0 <= r["precisao_macro"] <= 1.0
        assert 0.0 <= r["recall_macro"] <= 1.0

    def test_registra_matriz_de_confusao(self, bases_Xy):
        # Linha 4 do pseudocódigo exige a matriz de confusão registrada
        X, y = bases_Xy["BASE_A"]
        mascara = np.ones(X.shape[1], dtype=int)
        r = avaliar_intra_dataset(mascara, X, y, CLF_RAPIDO, cv_folds=3, seed=42)
        n_classes = len(r["classes"])
        assert len(r["matriz_confusao"]) == n_classes
        # A matriz acumulada dos folds soma o total de amostras
        assert int(np.sum(r["matriz_confusao"])) == len(y)

    def test_numero_de_folds_respeitado(self, bases_Xy):
        X, y = bases_Xy["BASE_A"]
        mascara = np.ones(X.shape[1], dtype=int)
        r = avaliar_intra_dataset(mascara, X, y, CLF_RAPIDO, cv_folds=4, seed=42)
        assert len(r["f1_macro_por_fold"]) == 4

    def test_base_separavel_tem_f1_alto(self, bases_Xy):
        # As bases sintéticas são separáveis: F1 baixo indicaria bug
        X, y = bases_Xy["BASE_A"]
        mascara = np.ones(X.shape[1], dtype=int)
        r = avaliar_intra_dataset(mascara, X, y, CLF_RAPIDO, cv_folds=3, seed=42)
        assert r["f1_macro"] > 0.9


class TestCrossDataset:
    def test_registra_as_duas_direcoes_separadas(self, bases_Xy):
        # Linha 9: F1-macro POR DIREÇÃO, sem reduzir a média única
        X, _ = bases_Xy["BASE_A"]
        mascara = np.ones(X.shape[1], dtype=int)
        resultados, _, _ = avaliar_cross_dataset(mascara, bases_Xy, CLF_RAPIDO, seed=42)
        assert "BASE_A->BASE_B" in resultados
        assert "BASE_B->BASE_A" in resultados
        assert len(resultados) == 2  # i != j, nada de i == j

    def test_tempo_de_inferencia_positivo(self, bases_Xy):
        # Linha 12: tempo médio de inferência precisa ser medido
        X, _ = bases_Xy["BASE_A"]
        mascara = np.ones(X.shape[1], dtype=int)
        _, tempo, _ = avaliar_cross_dataset(mascara, bases_Xy, CLF_RAPIDO, seed=42)
        assert tempo > 0.0


class TestAvaliarFitnessCompleto:
    def test_estrutura_da_saida(self, bases_Xy):
        # Linhas 14-18: critérios separados na saída
        X, _ = bases_Xy["BASE_A"]
        mascara = np.ones(X.shape[1], dtype=int)
        c = avaliar_fitness(mascara, bases_Xy, CLF_RAPIDO, cv_folds=3, seed=42)
        assert set(c["f1_macro_intra"].keys()) == {"BASE_A", "BASE_B"}
        assert set(c["f1_macro_cross_por_direcao"].keys()) == {
            "BASE_A->BASE_B", "BASE_B->BASE_A"
        }
        assert c["tempo_medio_inferencia"] > 0
        assert c["numero_atributos"] == X.shape[1]

    def test_mascara_parcial_reduz_k(self, bases_Xy):
        X, _ = bases_Xy["BASE_A"]
        mascara = np.zeros(X.shape[1], dtype=int)
        mascara[:3] = 1
        c = avaliar_fitness(mascara, bases_Xy, CLF_RAPIDO, cv_folds=3, seed=42)
        assert c["numero_atributos"] == 3


class TestFabricaClassificadores:
    def test_cria_todos_os_classificadores(self):
        for nome in NOMES_VALIDOS:
            modelo = criar_classificador(nome, seed=42)
            assert modelo is not None

    def test_nome_invalido_levanta_erro(self):
        with pytest.raises(ValueError):
            criar_classificador("svm_que_nao_existe")

    def test_configuracao_da_arvore_original(self):
        dt = criar_classificador("decision_tree")
        assert dt.max_depth == 8 and dt.class_weight == "balanced"

    def test_balanceamento_fica_no_classificador(self):
        assert criar_classificador("decision_tree").class_weight == "balanced"


class TestDiagnosticoCross:
    """Decomposição da queda cross-dataset em domain shift vs taxonomia."""

    def _bases_taxonomia_diferente(self):
        import pandas as pd
        def base(seed, classes, desloc):
            r = np.random.RandomState(seed)
            partes = []
            for i, c in enumerate(classes):
                f = r.normal(i * 2 + desloc, 0.5, size=(120, 5))
                df = pd.DataFrame(f, columns=[f"F{j}" for j in range(5)])
                df["Attack"] = c
                partes.append(df)
            df = pd.concat(partes).sample(frac=1.0, random_state=seed)
            return (df.drop(columns=["Attack"]).reset_index(drop=True),
                    df["Attack"].reset_index(drop=True))
        # BASE_A tem Benign/DoS/Scan; BASE_B troca Scan por Ransomware
        # (classe que A nunca viu) -> incompatibilidade de taxonomia
        return {
            "A": base(1, ["Benign", "DoS", "Scan"], 0.0),
            "B": base(2, ["Benign", "DoS", "Ransomware"], 0.3),
        }

    def test_diagnostico_presente_na_saida(self, bases_Xy):
        X, _ = bases_Xy["BASE_A"]
        mascara = np.ones(X.shape[1], dtype=int)
        c = avaliar_fitness(mascara, bases_Xy, CLF_RAPIDO, cv_folds=2, seed=42)
        assert "diagnostico_cross" in c
        for chave in ("BASE_A->BASE_B", "BASE_B->BASE_A"):
            d = c["diagnostico_cross"][chave]
            assert "matriz_confusao" in d
            assert "prop_desconhecidas" in d
            assert "acuracia_classes_conhecidas" in d

    def test_detecta_classe_desconhecida(self):
        # Treinar em A (sem Ransomware) e testar em B (com Ransomware):
        # a partição desconhecida deve ser detectada e quantificada
        from Modulos.avaliacao import avaliar_cross_dataset
        bases = self._bases_taxonomia_diferente()
        X, _ = bases["A"]
        mascara = np.ones(X.shape[1], dtype=int)
        _, _, diag = avaliar_cross_dataset(mascara, bases, CLF_RAPIDO, seed=42)
        d = diag["A->B"]
        # ~1/3 das amostras de B são Ransomware, ausente em A
        assert d["n_classes_desconhecidas"] > 0
        assert 0.2 < d["prop_desconhecidas"] < 0.45

    def test_matriz_soma_total_de_amostras(self, bases_Xy):
        X, _ = bases_Xy["BASE_A"]
        mascara = np.ones(X.shape[1], dtype=int)
        c = avaliar_fitness(mascara, bases_Xy, CLF_RAPIDO, cv_folds=2, seed=42)
        d = c["diagnostico_cross"]["BASE_A->BASE_B"]
        assert int(np.sum(d["matriz_confusao"])) == d["n_total"]
