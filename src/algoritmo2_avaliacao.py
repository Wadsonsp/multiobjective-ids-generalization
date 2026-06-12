# -*- coding: utf-8 -*-
"""Execução isolada do Algoritmo 2 - AvaliarFitness(x, D, h) (Figura 6).

Serve para dois cenários:
1. Avaliação FINAL (dados completos) de uma solução escolhida do P*
   gerado pelo Algoritmo 1, já que a busca usa subamostragem;
2. Baseline com todos os atributos (máscara cheia), equivalente à
   análise preliminar do Capítulo 5.

Uso:
    # baseline com todos os atributos e dados completos
    python src/algoritmo2_avaliacao.py --mascara cheia --amostra 0

    # avalia a solução 3 de um arquivo Pareto salvo
    python src/algoritmo2_avaliacao.py --pareto Resultados/pareto/pareto_X.json --solucao 3

    # máscara manual (string binária, d posições)
    python src/algoritmo2_avaliacao.py --mascara 101100111...
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modulos.avaliacao import avaliar_fitness  # noqa: E402
from Modulos.drive_loader import carregar_todas_as_bases  # noqa: E402
from Modulos.preprocessamento import alinhar_colunas, preprocessar_base  # noqa: E402

CONFIG_PADRAO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def carregar_config(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def montar_mascara(args, d):
    """Resolve a máscara a partir de --mascara ou --pareto/--solucao."""
    if args.pareto:
        with open(args.pareto, "r", encoding="utf-8") as f:
            pareto = json.load(f)
        return np.array(pareto["mascaras"][args.solucao], dtype=int)

    if args.mascara == "cheia":
        # Baseline: todos os atributos selecionados (k(x) = d)
        return np.ones(d, dtype=int)

    bits = [int(c) for c in args.mascara.strip()]
    if len(bits) != d:
        raise ValueError(f"Máscara com {len(bits)} bits para d={d} atributos.")
    return np.array(bits, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Algoritmo 2 - AvaliarFitness")
    parser.add_argument("--config", default=CONFIG_PADRAO)
    parser.add_argument("--mascara", default="cheia",
                        help="'cheia' ou string binária com d posições")
    parser.add_argument("--pareto", default=None,
                        help="arquivo JSON de P* gerado pelo Algoritmo 1")
    parser.add_argument("--solucao", type=int, default=0,
                        help="índice da solução dentro do arquivo Pareto")
    parser.add_argument("--classificador", default=None)
    parser.add_argument("--amostra", type=int, default=None,
                        help="subamostra estratificada (0 = dados completos)")
    args = parser.parse_args()

    config = carregar_config(args.config)
    cfg_pre = config["preprocessamento"]
    seed = config["avaliacao"]["seed"]
    nome_clf = args.classificador or config["classificador"]["nome"]

    # Pré-processamento idêntico ao do Algoritmo 1 (Seção 4.1)
    brutas = carregar_todas_as_bases(config)
    bases_Xy = {}
    for nome, df in brutas.items():
        n_base = config["datasets"]["bases"][nome].get("amostra_estratificada")
        X, y = preprocessar_base(df, cfg_pre, n_amostras=n_base, seed=seed)
        bases_Xy[nome] = (X, y)
    bases_Xy, ordem = alinhar_colunas(bases_Xy)

    # Subamostra opcional (uso 0 para a avaliação final completa)
    amostra = args.amostra
    if amostra:
        import pandas as pd
        from Modulos.preprocessamento import amostra_estratificada
        for nome, (X, y) in bases_Xy.items():
            df_xy = pd.concat([X, y], axis=1)
            df_xy = amostra_estratificada(
                df_xy, amostra, coluna_rotulo=cfg_pre["coluna_rotulo"], seed=seed
            )
            bases_Xy[nome] = (
                df_xy.drop(columns=[cfg_pre["coluna_rotulo"]]),
                df_xy[cfg_pre["coluna_rotulo"]],
            )

    mascara = montar_mascara(args, d=len(ordem))
    print(f"[config] h={nome_clf} | k(x)={int(mascara.sum())} de d={len(ordem)}")

    criterios = avaliar_fitness(
        mascara, bases_Xy, nome_clf,
        cv_folds=config["avaliacao"]["cv_folds"], seed=seed,
    )

    # Saída com os critérios separados, como definido na Figura 6
    print("\n===== Critérios de avaliação =====")
    print("F1-macro intra-dataset:")
    for nome, f1 in criterios["f1_macro_intra"].items():
        print(f"  {nome}: {f1:.4f}")
    print("F1-macro cross-dataset por direção:")
    for direcao, f1 in criterios["f1_macro_cross_por_direcao"].items():
        print(f"  {direcao}: {f1:.4f}")
    print(f"Tempo médio de inferência: {criterios['tempo_medio_inferencia'] * 1000:.4f} ms/amostra")
    print(f"Número de atributos selecionados k(x): {criterios['numero_atributos']}")

    pasta = config["resultados"]["pasta_metricas"]
    os.makedirs(pasta, exist_ok=True)
    rotulo = time.strftime("%Y%m%d_%H%M%S") + f"_{nome_clf}_k{int(mascara.sum())}"
    caminho = os.path.join(pasta, f"avaliacao_{rotulo}.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump({"mascara": mascara.tolist(), "atributos": ordem, **criterios},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[salvo] {caminho}")


if __name__ == "__main__":
    main()
