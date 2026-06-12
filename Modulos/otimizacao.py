# -*- coding: utf-8 -*-
"""Algoritmo 1 - Otimização multiobjetivo para seleção de características
(Figura 5 da dissertação), implementado com NSGA-II (Deb et al. [12])
via pymoo.

Correspondência com o pseudocódigo:
 1: Pré-processar os datasets em D  -> feito antes, em src/algoritmo1_otimizacao.py
 2-3: Inicializar população P com N_pop indivíduos binários x em {0,1}^d
      -> amostragem BinaryRandomSampling
 4-7: para cada geração, para cada x em P: critérios(x) <- AvaliarFitness(x, D, h)
      -> ProblemaSelecaoCaracteristicas._evaluate chama Modulos.avaliacao
 8: Ordenar P por não-dominância e diversidade -> non-dominated sorting +
    crowding distance internos do NSGA-II
 9-11: Selecionar, aplicar crossover (pc) e mutação (pm), gerar nova população
      -> TwoPointCrossover(prob=pc) e BitflipMutation(prob=pm)
12-14: Retornar P* (conjunto de soluções não-dominadas) -> res.X / res.F

Formulação tri-objetivo (cronograma do Capítulo 6: F1 intra, cross, custo).
O pymoo MINIMIZA, então converto os F1 para (1 - F1):
  f1 = 1 - média do F1-macro intra-dataset
  f2 = 1 - média do F1-macro cross-dataset (média só para guiar a busca;
       os valores POR DIREÇÃO ficam registrados no histórico de critérios)
  f3 = custo: k(x)/d (indicador estrutural, default) ou tempo de inferência
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from Modulos.avaliacao import avaliar_fitness


def reparar_mascara_vazia(mascara, rng):
    """Garante pelo menos 1 atributo selecionado.

    O bit-flip pode zerar a máscara inteira; um classificador sem
    atributos não faz sentido, então ligo uma posição aleatória.
    """
    mascara = np.asarray(mascara).astype(bool).copy()
    if not mascara.any():
        mascara[rng.randint(0, mascara.shape[0])] = True
    return mascara


class ProblemaSelecaoCaracteristicas(ElementwiseProblem):
    """Problema binário tri-objetivo avaliado pelo Algoritmo 2."""

    def __init__(self, bases_Xy, nome_clf, cv_folds=5, seed=42,
                 objetivo_custo="k"):
        # d = número de atributos candidatos (colunas já alinhadas)
        self.d = next(iter(bases_Xy.values()))[0].shape[1]
        self.bases_Xy = bases_Xy
        self.nome_clf = nome_clf
        self.cv_folds = cv_folds
        self.seed = seed
        self.objetivo_custo = objetivo_custo
        self.rng = np.random.RandomState(seed)
        # Histórico com TODOS os critérios de cada solução avaliada
        # (inclusive F1 cross por direção), para análise posterior
        self.historico = []
        super().__init__(n_var=self.d, n_obj=3, n_constr=0, xl=0, xu=1, vtype=bool)

    def _evaluate(self, x, out, *args, **kwargs):
        # Linha 6 do Algoritmo 1: critérios(x) <- AvaliarFitness(x, D, h)
        mascara = reparar_mascara_vazia(x, self.rng)
        criterios = avaliar_fitness(
            mascara, self.bases_Xy, self.nome_clf, self.cv_folds, self.seed
        )

        f1_intra_medio = float(np.mean(list(criterios["f1_macro_intra"].values())))
        f1_cross_medio = float(
            np.mean(list(criterios["f1_macro_cross_por_direcao"].values()))
        )

        if self.objetivo_custo == "tempo":
            custo = criterios["tempo_medio_inferencia"]
        else:
            # k(x)/d normaliza o custo estrutural para [0,1]
            custo = criterios["numero_atributos"] / self.d

        out["F"] = [1.0 - f1_intra_medio, 1.0 - f1_cross_medio, custo]

        self.historico.append({"mascara": mascara.astype(int).tolist(), **criterios})


def executar_otimizacao(bases_Xy, nome_clf, n_pop=40, n_gen=30,
                        pc=0.9, pm=0.05, seed=42, cv_folds=5,
                        objetivo_custo="k", verbose=True):
    """Executa o Algoritmo 1 e retorna o conjunto Pareto P*.

    Returns
    -------
    dict com:
      mascaras   : matriz binária (n_solucoes x d) das soluções de P*
      objetivos  : valores (1-F1_intra, 1-F1_cross, custo) de cada solução
      historico  : critérios completos de todas as avaliações
    """
    problema = ProblemaSelecaoCaracteristicas(
        bases_Xy, nome_clf, cv_folds=cv_folds, seed=seed,
        objetivo_custo=objetivo_custo,
    )

    # Linhas 2-3: população inicial binária; linhas 9-11: operadores genéticos
    algoritmo = NSGA2(
        pop_size=n_pop,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=pc),
        mutation=BitflipMutation(prob=pm),
        eliminate_duplicates=True,
    )

    # Linhas 4-12: laço de gerações (critério de parada = N_gen)
    res = minimize(
        problema,
        algoritmo,
        ("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
        save_history=False,
    )

    # Linhas 13-14: P* = conjunto de soluções não-dominadas
    mascaras = np.atleast_2d(res.X).astype(int)
    objetivos = np.atleast_2d(res.F)

    return {
        "mascaras": mascaras,
        "objetivos": objetivos,
        "historico": problema.historico,
        "nomes_objetivos": ["1 - F1_intra_medio", "1 - F1_cross_medio",
                            f"custo ({objetivo_custo})"],
    }
