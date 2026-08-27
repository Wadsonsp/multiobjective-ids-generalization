# -*- coding: utf-8 -*-
"""Eu implemento a Fase 1: pré-filtro de features com NSGA-II.

Eu sigo a formulação didática apresentada no código original: cada variável
de decisão fica no intervalo [0, 1] e eu seleciono uma feature quando o gene
é maior ou igual a 0,5. O pymoo minimiza os dois objetivos:

1. erro cross-dataset = 1 - média dos F1-macro nas duas direções;
2. proporção de features = número selecionado / número total.

Eu uso o NSGA-II padrão do pymoo. Informo somente o tamanho da população;
amostragem, crossover, mutação e seleção permanecem com os padrões da
biblioteca. O número de gerações é apenas o critério de parada do experimento.
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from Modulos.avaliacao import avaliar_fase1_cross_dataset


class ProblemaSelecaoCaracteristicas(ElementwiseProblem):
    """Eu defino o problema cross-dataset com exatamente dois objetivos.

    Para cada solução, eu uso a mesma máscara nos dois datasets. Eu treino
    em uma base completa e testo na outra completa, nas duas direções. Em
    seguida, eu combino os dois F1-macro em um único objetivo de desempenho
    e mantenho o número de features como o segundo objetivo.
    """

    def __init__(self, bases_Xy, nome_clf, seed=42, cache=None):
        # Eu guardo os dois datasets já alinhados para acessá-los em _evaluate.
        if len(bases_Xy) != 2:
            raise ValueError(
                "Eu preciso de exatamente dois datasets para avaliar as duas direções."
            )
        self.d = next(iter(bases_Xy.values()))[0].shape[1]
        self.bases_Xy = bases_Xy
        self.nome_clf = nome_clf
        self.seed = seed

        # Eu uso o cache para não treinar novamente uma máscara já avaliada.
        self.cache = cache

        # Eu mantenho o histórico completo para explicar cada ponto de Pareto.
        self.historico = []

        # Eu crio um gene em [0, 1] para cada feature e declaro dois objetivos.
        super().__init__(
            n_var=self.d,
            n_obj=2,
            n_constr=0,
            xl=np.zeros(self.d),
            xu=np.ones(self.d),
        )

    def decidir_features_mantidas(self, x):
        """Eu devolvo os índices cujos genes atingiram o limiar de 0,5."""
        return np.where(np.asarray(x) >= 0.5)[0]

    def criar_mascara_binaria(self, x):
        """Eu converto os genes contínuos para a máscara usada nas bases."""
        mascara = np.zeros(self.d, dtype=int)
        mascara[self.decidir_features_mantidas(x)] = 1
        return mascara

    def _evaluate(self, x, out, *args, **kwargs):
        # Eu transformo os genes na mesma máscara binária para as duas bases.
        mascara = self.criar_mascara_binaria(x)
        numero_selecionadas = int(mascara.sum())

        # Eu penalizo a solução vazia porque nenhum classificador pode ser
        # treinado sem features. Assim reproduzo a regra do código original.
        if numero_selecionadas == 0:
            out["F"] = [1.0, 1.0]
            self.historico.append({
                "mascara": mascara.tolist(),
                "avaliacao_valida": False,
                "motivo": "nenhuma feature selecionada",
                "f1_macro_cross_medio": 0.0,
                "numero_atributos": 0,
            })
            return

        # Eu procuro primeiro no cache porque duas soluções contínuas podem
        # produzir exatamente a mesma máscara após a aplicação do limiar.
        criterios = self.cache.obter(mascara) if self.cache is not None else None
        if criterios is None:
            criterios = avaliar_fase1_cross_dataset(
                mascara, self.bases_Xy, self.nome_clf, self.seed
            )
            if self.cache is not None:
                # Eu gravo imediatamente para poder retomar uma execução local.
                self.cache.registrar(mascara, criterios)

        # Eu preservo cada direção no histórico e uso somente a média como
        # valor agregado do primeiro objetivo.
        valores_cross = list(criterios["f1_macro_cross_por_direcao"].values())
        if len(valores_cross) != 2:
            raise RuntimeError(
                "Eu esperava exatamente dois valores de F1 cross-dataset."
            )
        f1_cross_medio = float(
            np.mean(valores_cross)
        )

        # Eu normalizo a quantidade de features como no código fornecido.
        proporcao_features = numero_selecionadas / self.d

        # Eu entrego ao pymoo os dois objetivos que ele deve minimizar.
        out["F"] = [1.0 - f1_cross_medio, proporcao_features]

        self.historico.append({
            "mascara": mascara.tolist(),
            "avaliacao_valida": True,
            "f1_macro_cross_medio": f1_cross_medio,
            **criterios,
        })


def executar_otimizacao(bases_Xy, nome_clf, n_pop=24, n_gen=15,
                        seed=42, verbose=True,
                        cache=None, registro_progresso=None):
    """Executa o Algoritmo 1 e retorna o conjunto Pareto P*.

    Returns
    -------
    dict com:
      mascaras   : matriz binária (n_solucoes x d) das soluções de P*
      objetivos  : valores (1-F1_cross_médio, proporção de features)
      historico  : critérios completos de todas as avaliações
    """
    problema = ProblemaSelecaoCaracteristicas(
        bases_Xy, nome_clf, seed=seed, cache=cache,
    )

    # Eu informo apenas a população e preservo os operadores padrão do NSGA-II.
    algoritmo = NSGA2(pop_size=n_pop)

    # Linhas 4-12: laço de gerações (critério de parada = N_gen)
    # Eu gravo a fronteira parcial localmente ao fim de cada geração; o
    # callback só entra nos kwargs quando existe, porque o pymoo invoca
    # o que for passado (None explícito quebraria a chamada interna)
    kwargs_minimize = {}
    if registro_progresso is not None:
        kwargs_minimize["callback"] = registro_progresso

    res = minimize(
        problema,
        algoritmo,
        ("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
        save_history=False,
        **kwargs_minimize,
    )

    # Eu converto os genes de Pareto em máscaras usando o mesmo limiar do
    # problema e removo máscaras repetidas sem definir uma quantidade final.
    genes = np.atleast_2d(res.X)
    mascaras_brutas = (genes >= 0.5).astype(int)
    objetivos_brutos = np.atleast_2d(res.F)
    indices_unicos = []
    chaves_vistas = set()
    for i, mascara in enumerate(mascaras_brutas):
        chave = tuple(mascara.tolist())
        if chave not in chaves_vistas:
            chaves_vistas.add(chave)
            indices_unicos.append(i)
    mascaras = mascaras_brutas[indices_unicos]
    objetivos = objetivos_brutos[indices_unicos]

    # Eu ordeno a saída da solução mais compacta para a menos compacta.
    ordem = np.lexsort((objetivos[:, 0], objetivos[:, 1]))
    mascaras = mascaras[ordem]
    objetivos = objetivos[ordem]

    return {
        "mascaras": mascaras,
        "objetivos": objetivos,
        "historico": problema.historico,
        "nomes_objetivos": [
            "1 - F1_cross_medio",
            "proporcao_features_selecionadas",
        ],
    }
