# -*- coding: utf-8 -*-
"""Execução do Algoritmo 1 - Otimização multiobjetivo (Figura 5).

Pipeline completo: carrega as bases do Drive, aplica o pré-processamento
da Seção 4.1, executa o NSGA-II com o AvaliarFitness do Algoritmo 2 e
salva o conjunto Pareto P* e o histórico de critérios em Resultados/.

Uso:
    python src/algoritmo1_otimizacao.py                       # config padrão
    python src/algoritmo1_otimizacao.py --n-pop 60 --n-gen 50 # sobrescreve
    python src/algoritmo1_otimizacao.py --amostra-busca 50000 # busca mais leve

Observação importante de custo: cada avaliação de fitness treina o
classificador em CV + nas duas direções cross-dataset. Para a fase de
BUSCA uso a subamostragem estratificada (avaliacao.amostra_busca do
config); a avaliação FINAL das soluções de P* deve ser refeita com os
dados completos (--amostra-busca 0).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

# Permito rodar tanto da raiz do projeto quanto de dentro de src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modulos.checkpoint import CacheAvaliacoes, RegistroProgresso  # noqa: E402
from Modulos.drive_loader import carregar_todas_as_bases  # noqa: E402
from Modulos.otimizacao import executar_otimizacao  # noqa: E402
from Modulos.preprocessamento import alinhar_colunas, preprocessar_base  # noqa: E402

CONFIG_PADRAO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def carregar_config(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preparar_bases(config, amostra_busca=None):
    """Linha 1 do Algoritmo 1: pré-processar os datasets em D."""
    brutas = carregar_todas_as_bases(config)
    cfg_pre = config["preprocessamento"]
    seed = config["avaliacao"]["seed"]

    bases_Xy = {}
    for nome, df in brutas.items():
        # Amostra fixa da base (ex.: 2M do ToN-IoT, Seção 5.1) +
        # subamostra adicional da fase de busca, quando configurada
        n_base = config["datasets"]["bases"][nome].get("amostra_estratificada")
        X, y = preprocessar_base(df, cfg_pre, n_amostras=n_base, seed=seed)
        if amostra_busca:
            from Modulos.preprocessamento import amostra_estratificada
            import pandas as pd
            df_xy = pd.concat([X, y], axis=1)
            df_xy = amostra_estratificada(
                df_xy, amostra_busca, coluna_rotulo=cfg_pre["coluna_rotulo"], seed=seed
            )
            y = df_xy[cfg_pre["coluna_rotulo"]]
            X = df_xy.drop(columns=[cfg_pre["coluna_rotulo"]])
        bases_Xy[nome] = (X, y)
        print(f"[base] {nome}: {X.shape[0]} fluxos, {X.shape[1]} atributos")

    # Alinho colunas entre as bases: a máscara x precisa indexar o mesmo
    # atributo em qualquer base (vantagem do schema NF-v2 comum)
    bases_Xy, ordem = alinhar_colunas(bases_Xy)
    print(f"[schema] {len(ordem)} atributos candidatos (d={len(ordem)})")
    return bases_Xy, ordem


def salvar_resultados(resultado, ordem_atributos, config, rotulo_execucao):
    """Salva P* (máscaras + objetivos) e o histórico completo de critérios."""
    pasta_pareto = config["resultados"]["pasta_pareto"]
    pasta_metricas = config["resultados"]["pasta_metricas"]
    os.makedirs(pasta_pareto, exist_ok=True)
    os.makedirs(pasta_metricas, exist_ok=True)

    caminho_pareto = os.path.join(pasta_pareto, f"pareto_{rotulo_execucao}.json")
    with open(caminho_pareto, "w", encoding="utf-8") as f:
        json.dump(
            {
                "atributos": ordem_atributos,
                "nomes_objetivos": resultado["nomes_objetivos"],
                "mascaras": resultado["mascaras"].tolist(),
                "objetivos": resultado["objetivos"].tolist(),
                "atributos_selecionados_por_solucao": [
                    [a for a, bit in zip(ordem_atributos, m) if bit]
                    for m in resultado["mascaras"]
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    caminho_hist = os.path.join(pasta_metricas, f"historico_{rotulo_execucao}.json")
    with open(caminho_hist, "w", encoding="utf-8") as f:
        json.dump(resultado["historico"], f, ensure_ascii=False, indent=2)

    print(f"[salvo] P* -> {caminho_pareto}")
    print(f"[salvo] histórico de critérios -> {caminho_hist}")


def main():
    parser = argparse.ArgumentParser(description="Algoritmo 1 - NSGA-II")
    parser.add_argument("--config", default=CONFIG_PADRAO)
    parser.add_argument("--n-pop", type=int, default=None)
    parser.add_argument("--n-gen", type=int, default=None)
    parser.add_argument("--pc", type=float, default=None)
    parser.add_argument("--pm", type=float, default=None)
    parser.add_argument("--classificador", default=None)
    parser.add_argument(
        "--amostra-busca", type=int, default=None,
        help="Subamostra estratificada da fase de busca (0 = dados completos)",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Pasta (idealmente no Drive) para cache de avaliações e "
             "progresso por geração. Para RETOMAR após queda do Colab, "
             "basta reexecutar o MESMO comando com a mesma pasta.",
    )
    args = parser.parse_args()

    config = carregar_config(args.config)
    cfg_otm = config["otimizacao"]

    # Argumentos de linha de comando sobrescrevem o config.yaml
    n_pop = args.n_pop or cfg_otm["n_pop"]
    n_gen = args.n_gen or cfg_otm["n_gen"]
    pc = args.pc if args.pc is not None else cfg_otm["pc"]
    pm = args.pm if args.pm is not None else cfg_otm["pm"]
    nome_clf = args.classificador or config["classificador"]["nome"]
    amostra_busca = (
        args.amostra_busca
        if args.amostra_busca is not None
        else config["avaliacao"]["amostra_busca"]
    )
    if amostra_busca == 0:
        amostra_busca = None

    print(f"[config] h={nome_clf} | N_pop={n_pop} | N_gen={n_gen} | "
          f"pc={pc} | pm={pm} | amostra_busca={amostra_busca}")

    bases_Xy, ordem = preparar_bases(config, amostra_busca)

    # Checkpoint: cache de avaliações + progresso por geração no Drive.
    # O contexto identifica o experimento; mudar qualquer item daria
    # critérios diferentes, então o cache é segregado por contexto.
    cache, registro = None, None
    if args.checkpoint_dir:
        contexto = {
            "classificador": nome_clf,
            "bases": sorted(bases_Xy.keys()),
            "d": len(ordem),
            "cv_folds": config["avaliacao"]["cv_folds"],
            "amostra_busca": amostra_busca,
            "seed": cfg_otm["seed"],
            "objetivo_custo": cfg_otm["objetivo_custo"],
        }
        sufixo = f"{nome_clf}_a{amostra_busca or 'full'}_s{cfg_otm['seed']}"
        cache = CacheAvaliacoes(
            os.path.join(args.checkpoint_dir, f"cache_{sufixo}.jsonl"), contexto
        )
        registro = RegistroProgresso(
            os.path.join(args.checkpoint_dir, f"progresso_{sufixo}.json")
        )
        print(f"[checkpoint] {len(cache)} avaliações recuperadas do cache "
              f"em {args.checkpoint_dir}")

    inicio = time.time()
    resultado = executar_otimizacao(
        bases_Xy,
        nome_clf,
        n_pop=n_pop,
        n_gen=n_gen,
        pc=pc,
        pm=pm,
        seed=cfg_otm["seed"],
        cv_folds=config["avaliacao"]["cv_folds"],
        objetivo_custo=cfg_otm["objetivo_custo"],
        cache=cache,
        registro_progresso=registro,
    )
    duracao = time.time() - inicio

    print(f"\n[fim] {len(resultado['mascaras'])} soluções não-dominadas em P* "
          f"({duracao / 60:.1f} min)")

    rotulo = time.strftime("%Y%m%d_%H%M%S") + f"_{nome_clf}"
    salvar_resultados(resultado, ordem, config, rotulo)


if __name__ == "__main__":
    main()
