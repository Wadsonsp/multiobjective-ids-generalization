# -*- coding: utf-8 -*-
"""Eu executo a Fase 2: avaliação detalhada das máscaras pré-filtradas.

Serve para dois cenários:
1. Avaliação detalhada de uma ou de todas as soluções do P*;
2. Baseline com todos os atributos (máscara cheia), equivalente à
   análise preliminar do Capítulo 5.

Uso:
    # baseline com todos os atributos e dados completos
    python src/algoritmo2_avaliacao.py --mascara cheia

    # avalia a solução 3 de um arquivo Pareto salvo
    python src/algoritmo2_avaliacao.py --pareto Resultados/pareto/pareto_X.json --solucao 3

    # Eu avalio todas as soluções não dominadas sem definir sua quantidade
    python src/algoritmo2_avaliacao.py --pareto Resultados/pareto/pareto_X.json --todas

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
from Modulos.carregamento import carregar_todas_as_bases  # noqa: E402
from Modulos.preprocessamento import alinhar_colunas, preprocessar_base  # noqa: E402

CONFIG_PADRAO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def carregar_config(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_caminho_local(caminho):
    """Eu resolvo saídas relativas a partir da raiz do repositório."""
    return caminho if os.path.isabs(caminho) else os.path.join(RAIZ_PROJETO, caminho)


def montar_mascaras(args, d, ordem_atributos):
    """Eu resolvo uma ou todas as máscaras solicitadas pela linha de comando."""
    if args.pareto:
        with open(args.pareto, "r", encoding="utf-8") as f:
            pareto = json.load(f)
        if pareto.get("atributos") != ordem_atributos:
            raise ValueError(
                "As features do arquivo Pareto não coincidem com os datasets atuais."
            )
        indices = range(len(pareto["mascaras"])) if args.todas else [args.solucao]
        return [
            (int(i), np.array(pareto["mascaras"][i], dtype=int)) for i in indices
        ]

    if args.mascara == "cheia":
        # Baseline: todos os atributos selecionados (k(x) = d)
        return [(0, np.ones(d, dtype=int))]

    bits = [int(c) for c in args.mascara.strip()]
    if len(bits) != d:
        raise ValueError(f"Máscara com {len(bits)} bits para d={d} atributos.")
    return [(0, np.array(bits, dtype=int))]


def imprimir_criterios(criterios):
    """Eu apresento as métricas de uma máscara de forma legível."""
    print("F1-macro intra-dataset:")
    for nome, f1 in criterios["f1_macro_intra"].items():
        print(f"  {nome}: {f1:.4f}")
    print("F1-macro cross-dataset por direção:")
    for direcao, f1 in criterios["f1_macro_cross_por_direcao"].items():
        print(f"  {direcao}: {f1:.4f}")
    print(
        "Tempo médio de inferência: "
        f"{criterios['tempo_medio_inferencia'] * 1000:.4f} ms/amostra"
    )
    print(f"Número de atributos selecionados k(x): {criterios['numero_atributos']}")


def main():
    parser = argparse.ArgumentParser(description="Algoritmo 2 - AvaliarFitness")
    parser.add_argument("--config", default=CONFIG_PADRAO)
    parser.add_argument("--mascara", default="cheia",
                        help="'cheia' ou string binária com d posições")
    parser.add_argument("--pareto", default=None,
                        help="arquivo JSON de P* gerado pelo Algoritmo 1")
    parser.add_argument("--solucao", type=int, default=0,
                        help="índice da solução dentro do arquivo Pareto")
    parser.add_argument("--todas", action="store_true",
                        help="avalia todas as soluções do arquivo Pareto")
    parser.add_argument("--classificador", default=None)
    args = parser.parse_args()

    config = carregar_config(args.config)
    cfg_pre = config["preprocessamento"]
    seed = config["avaliacao"]["seed"]
    nome_clf = args.classificador or config["classificador"]["nome"]

    # Eu repito o pré-processamento da Fase 1 usando todas as linhas.
    brutas = carregar_todas_as_bases(config)
    bases_Xy = {}
    for nome, df in brutas.items():
        X, y = preprocessar_base(df, cfg_pre, nome_base=nome)
        bases_Xy[nome] = (X, y)
    bases_Xy, ordem = alinhar_colunas(bases_Xy)

    mascaras = montar_mascaras(args, d=len(ordem), ordem_atributos=ordem)
    avaliacoes = []
    for indice, mascara in mascaras:
        print(
            f"\n===== Solução {indice} | h={nome_clf} | "
            f"k(x)={int(mascara.sum())} de d={len(ordem)} ====="
        )
        criterios = avaliar_fitness(
            mascara,
            bases_Xy,
            nome_clf,
            cv_folds=config["avaliacao"]["cv_folds"],
            seed=seed,
        )
        imprimir_criterios(criterios)
        avaliacoes.append({
            "indice_solucao": indice,
            "mascara": mascara.tolist(),
            "atributos_selecionados": [
                atributo for atributo, bit in zip(ordem, mascara) if bit
            ],
            **criterios,
        })

    pasta = resolver_caminho_local(config["resultados"]["pasta_metricas"])
    os.makedirs(pasta, exist_ok=True)
    rotulo = time.strftime("%Y%m%d_%H%M%S") + f"_{nome_clf}"
    caminho = os.path.join(pasta, f"avaliacao_{rotulo}.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fase": "Fase 2 - avaliacao detalhada",
                "classificador": nome_clf,
                "atributos": ordem,
                "avaliacoes": avaliacoes,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[salvo] {caminho}")


if __name__ == "__main__":
    main()
