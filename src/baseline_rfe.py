# -*- coding: utf-8 -*-
"""Baseline RFE (Capítulo 5) avaliado pelo mesmo Algoritmo 2 da proposta.

Fluxo:
1. Pré-processamento idêntico ao dos Algoritmos 1 e 2 (Seção 4.1);
2. Ajuste do RFE com Random Forest como estimador de referência na base
   escolhida (--base), reduzindo para --n-atributos (30 no texto);
3. Avaliação da máscara RFE E da máscara cheia pelo AvaliarFitness
   (Algoritmo 2), gerando a comparação no formato da Tabela 3:
   F1-macro e tempo de inferência antes/depois da redução, com variação
   percentual.

Como o baseline e a proposta tri-objetivo passam pelo MESMO protocolo de
avaliação, a comparação RFE x soluções do P* fica homogênea.

Uso:
    # baseline padrão (RFE 30 atributos, ajustado no NF-UNSW-NB15-v2)
    python src/baseline_rfe.py

    # comparação completa nos 8 classificadores (Tabela 3)
    python src/baseline_rfe.py --todos

    # ajuste do RFE na outra base, com subamostra de busca menor
    python src/baseline_rfe.py --base NF-ToN-IoT-v2 --amostra 100000
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
from Modulos.classificadores import NOMES_VALIDOS  # noqa: E402
from Modulos.drive_loader import carregar_todas_as_bases  # noqa: E402
from Modulos.preprocessamento import (  # noqa: E402
    alinhar_colunas,
    amostra_estratificada,
    preprocessar_base,
)
from Modulos.selecao_rfe import ajustar_mascara_rfe  # noqa: E402

CONFIG_PADRAO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def carregar_config(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preparar_bases(config, amostra=None):
    """Mesmo pré-processamento dos Algoritmos 1 e 2 (Seção 4.1)."""
    brutas = carregar_todas_as_bases(config)
    cfg_pre = config["preprocessamento"]
    seed = config["avaliacao"]["seed"]

    bases_Xy = {}
    for nome, df in brutas.items():
        n_base = config["datasets"]["bases"][nome].get("amostra_estratificada")
        X, y = preprocessar_base(df, cfg_pre, n_amostras=n_base, seed=seed)
        if amostra:
            import pandas as pd
            df_xy = pd.concat([X, y], axis=1)
            df_xy = amostra_estratificada(
                df_xy, amostra, coluna_rotulo=cfg_pre["coluna_rotulo"], seed=seed
            )
            y = df_xy[cfg_pre["coluna_rotulo"]]
            X = df_xy.drop(columns=[cfg_pre["coluna_rotulo"]])
        bases_Xy[nome] = (X, y)
        print(f"[base] {nome}: {X.shape[0]} fluxos, {X.shape[1]} atributos")

    return alinhar_colunas(bases_Xy)


def variacao_pct(antes, depois):
    """Variação percentual no formato da Tabela 3."""
    if antes == 0:
        return 0.0
    return (depois - antes) / antes * 100.0


def comparar(nome_clf, mascara_cheia, mascara_rfe, bases_Xy, cv_folds, seed):
    """Avalia as duas máscaras pelo Algoritmo 2 e monta a linha comparativa."""
    cheia = avaliar_fitness(mascara_cheia, bases_Xy, nome_clf, cv_folds, seed)
    rfe = avaliar_fitness(mascara_rfe, bases_Xy, nome_clf, cv_folds, seed)

    linha = {"classificador": nome_clf}
    for nome_base in bases_Xy:
        f1_a = cheia["f1_macro_intra"][nome_base]
        f1_d = rfe["f1_macro_intra"][nome_base]
        linha[f"f1_intra_{nome_base}_cheia"] = f1_a
        linha[f"f1_intra_{nome_base}_rfe"] = f1_d
        linha[f"var_pct_f1_{nome_base}"] = variacao_pct(f1_a, f1_d)

    # Cross-dataset por direção (registro separado, Seção 4.3)
    linha["f1_cross_cheia"] = cheia["f1_macro_cross_por_direcao"]
    linha["f1_cross_rfe"] = rfe["f1_macro_cross_por_direcao"]

    t_a = cheia["tempo_medio_inferencia"]
    t_d = rfe["tempo_medio_inferencia"]
    linha["tempo_inferencia_cheia"] = t_a
    linha["tempo_inferencia_rfe"] = t_d
    linha["var_pct_tempo"] = variacao_pct(t_a, t_d)

    linha["k_cheia"] = cheia["numero_atributos"]
    linha["k_rfe"] = rfe["numero_atributos"]
    return linha


def imprimir_linha(linha, bases):
    print(f"\n--- {linha['classificador']} ---")
    for nome_base in bases:
        print(
            f"  F1 intra {nome_base}: "
            f"{linha[f'f1_intra_{nome_base}_cheia']:.4f} -> "
            f"{linha[f'f1_intra_{nome_base}_rfe']:.4f} "
            f"({linha[f'var_pct_f1_{nome_base}']:+.2f}%)"
        )
    for direcao in linha["f1_cross_cheia"]:
        print(
            f"  F1 cross {direcao}: "
            f"{linha['f1_cross_cheia'][direcao]:.4f} -> "
            f"{linha['f1_cross_rfe'][direcao]:.4f}"
        )
    print(
        f"  Tempo inferência: {linha['tempo_inferencia_cheia'] * 1000:.4f} -> "
        f"{linha['tempo_inferencia_rfe'] * 1000:.4f} ms/amostra "
        f"({linha['var_pct_tempo']:+.2f}%)"
    )
    print(f"  k(x): {linha['k_cheia']} -> {linha['k_rfe']}")


def main():
    parser = argparse.ArgumentParser(description="Baseline RFE (Capítulo 5)")
    parser.add_argument("--config", default=CONFIG_PADRAO)
    parser.add_argument("--n-atributos", type=int, default=30,
                        help="alvo de atributos do RFE (30 no texto)")
    parser.add_argument("--base", default=None,
                        help="base usada no ajuste do RFE (default: a primeira)")
    parser.add_argument("--classificador", default=None,
                        help="classificador da avaliação (default: config.yaml)")
    parser.add_argument("--todos", action="store_true",
                        help="avalia os 8 classificadores (formato da Tabela 3)")
    parser.add_argument("--amostra", type=int, default=None,
                        help="subamostra estratificada da avaliação (0 = completa)")
    parser.add_argument("--amostra-rfe", type=int, default=200000,
                        help="subamostra usada SÓ no ajuste do RFE")
    args = parser.parse_args()

    config = carregar_config(args.config)
    seed = config["avaliacao"]["seed"]
    cv_folds = config["avaliacao"]["cv_folds"]
    amostra = (
        args.amostra if args.amostra is not None
        else config["avaliacao"]["amostra_busca"]
    )
    if amostra == 0:
        amostra = None

    bases_Xy, ordem = preparar_bases(config, amostra)
    d = len(ordem)

    # Ajuste do RFE na base de referência escolhida
    nome_base_rfe = args.base or next(iter(bases_Xy))
    if nome_base_rfe not in bases_Xy:
        raise ValueError(f"Base '{nome_base_rfe}' não existe: {list(bases_Xy)}")
    X_ref, y_ref = bases_Xy[nome_base_rfe]

    print(f"\n[rfe] ajustando em {nome_base_rfe}: {d} -> {args.n_atributos} atributos")
    inicio = time.time()
    mascara_rfe, ranking = ajustar_mascara_rfe(
        X_ref, y_ref, n_atributos=args.n_atributos,
        seed=seed, amostra=args.amostra_rfe,
    )
    print(f"[rfe] concluído em {(time.time() - inicio) / 60:.1f} min")
    selecionados = [a for a, bit in zip(ordem, mascara_rfe) if bit]
    print(f"[rfe] atributos mantidos ({int(mascara_rfe.sum())}): {selecionados}")

    mascara_cheia = np.ones(d, dtype=int)

    # Avaliação homogênea: as duas máscaras passam pelo Algoritmo 2
    classificadores = (
        list(NOMES_VALIDOS) if args.todos
        else [args.classificador or config["classificador"]["nome"]]
    )

    linhas = []
    for nome_clf in classificadores:
        linha = comparar(nome_clf, mascara_cheia, mascara_rfe,
                         bases_Xy, cv_folds, seed)
        imprimir_linha(linha, bases_Xy.keys())
        linhas.append(linha)

    pasta = config["resultados"]["pasta_metricas"]
    os.makedirs(pasta, exist_ok=True)
    rotulo = time.strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join(pasta, f"baseline_rfe_{rotulo}.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_ajuste_rfe": nome_base_rfe,
                "n_atributos_alvo": args.n_atributos,
                "atributos": ordem,
                "mascara_rfe": mascara_rfe.tolist(),
                "ranking_rfe": [int(r) for r in ranking],
                "atributos_selecionados": selecionados,
                "comparacao": linhas,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n[salvo] {caminho}")


if __name__ == "__main__":
    main()
