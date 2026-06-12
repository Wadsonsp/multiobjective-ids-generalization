# -*- coding: utf-8 -*-
"""Testes do Algoritmo 2 (AvaliarFitness) e da fábrica de classificadores.

O que valido aqui:
- aplicação correta da máscara binária (linha 1 do pseudocódigo);
- k(x) = soma da máscara;
- CV estratificada e métricas intra-dataset (linhas 2-4);
- cross-dataset registrado POR DIREÇÃO, sem média única (linhas 5-10);
- custo computacional medido (linhas 11-13);
- estrutura completa da saída (linhas 14-18);
- configurações exatas dos 8 classificadores da Seção 5.2.
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
from Modulos.classificadores import NOMES_VALIDOS, criar_classificador, usa_sample_weight

# Uso classificadores leves nos testes para a suíte rodar rápido;
# o XGBoost entra no smoke test de integração.
CLF_RAPIDO = "lda"


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
        resultados, _ = avaliar_cross_dataset(mascara, bases_Xy, CLF_RAPIDO, seed=42)
        assert "BASE_A->BASE_B" in resultados
        assert "BASE_B->BASE_A" in resultados
        assert len(resultados) == 2  # i != j, nada de i == j

    def test_tempo_de_inferencia_positivo(self, bases_Xy):
        # Linha 12: tempo médio de inferência precisa ser medido
        X, _ = bases_Xy["BASE_A"]
        mascara = np.ones(X.shape[1], dtype=int)
        _, tempo = avaliar_cross_dataset(mascara, bases_Xy, CLF_RAPIDO, seed=42)
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
    def test_cria_todos_os_8(self):
        for nome in NOMES_VALIDOS:
            modelo = criar_classificador(nome, seed=42)
            assert modelo is not None

    def test_nome_invalido_levanta_erro(self):
        with pytest.raises(ValueError):
            criar_classificador("svm_que_nao_existe")

    def test_configuracoes_da_secao_5_2(self):
        # Verifico os hiperparâmetros que reportei no texto da dissertação
        rf = criar_classificador("random_forest")
        assert rf.n_estimators == 100 and rf.criterion == "gini"
        assert rf.max_depth is None and rf.max_features == "sqrt"

        xgb = criar_classificador("xgboost")
        assert xgb.learning_rate == 0.3 and xgb.max_depth == 6
        assert xgb.min_child_weight == 1 and xgb.subsample == 1.0

        lgbm = criar_classificador("lightgbm")
        assert lgbm.n_estimators == 100 and lgbm.learning_rate == 0.1
        assert lgbm.num_leaves == 31

        lr = criar_classificador("logistic_regression")
        assert lr.C == 1.0 and lr.solver == "lbfgs" and lr.max_iter == 100

        knn = criar_classificador("knn")
        assert knn.n_neighbors == 5 and knn.weights == "uniform" and knn.p == 2

        mlp = criar_classificador("mlp")
        assert mlp.hidden_layer_sizes == (100,) and mlp.activation == "relu"
        assert mlp.learning_rate_init == 0.001 and mlp.max_iter == 200

    def test_balanceamento_conforme_secao_4_1(self):
        # class_weight balanceado em RF, LightGBM e LR; XGBoost via sample_weight
        assert criar_classificador("random_forest").class_weight == "balanced"
        assert criar_classificador("lightgbm").class_weight == "balanced"
        assert criar_classificador("logistic_regression").class_weight == "balanced"
        assert usa_sample_weight("xgboost") is True
        assert usa_sample_weight("lda") is False
