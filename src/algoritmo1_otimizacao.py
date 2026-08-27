# -*- coding: utf-8 -*-
"""Eu executo a Fase 1: pré-filtro de features com NSGA-II.

Eu carrego os dois Parquets locais completos, aplico o pré-processamento,
executo o problema biobjetivo e salvo localmente o conjunto Pareto e o
histórico de todas as máscaras avaliadas.

Uso:
    python src/algoritmo1_otimizacao.py
    python src/algoritmo1_otimizacao.py --n-pop 24 --n-gen 15
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
from Modulos.carregamento import carregar_todas_as_bases  # noqa: E402
from Modulos.otimizacao import executar_otimizacao  # noqa: E402
from Modulos.preprocessamento import alinhar_colunas, preprocessar_base  # noqa: E402

CONFIG_PADRAO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def carregar_config(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolver_caminho_local(caminho):
    """Eu resolvo saídas relativas a partir da raiz do repositório."""
    return caminho if os.path.isabs(caminho) else os.path.join(RAIZ_PROJETO, caminho)


def preparar_bases(config):
    """Eu leio e pré-processo integralmente os dois datasets locais."""
    brutas = carregar_todas_as_bases(config)
    cfg_pre = config["preprocessamento"]
    bases_Xy = {}
    for nome, df in brutas.items():
        # Eu aplico somente limpeza e separação: nenhuma linha é amostrada.
        X, y = preprocessar_base(df, cfg_pre, nome_base=nome)
        bases_Xy[nome] = (X, y)
        print(f"[base] {nome}: {X.shape[0]} fluxos, {X.shape[1]} atributos")
        print(f"[classes] {nome}: {sorted(str(c) for c in y.unique())}")

    # Eu alinho as colunas para cada gene representar a mesma feature.
    bases_Xy, ordem = alinhar_colunas(bases_Xy)
    print(f"[schema] {len(ordem)} atributos candidatos (d={len(ordem)})")
    return bases_Xy, ordem


def salvar_resultados(
    resultado, ordem_atributos, config, rotulo_execucao, configuracao_execucao
):
    """Eu salvo P* e o histórico completo em arquivos JSON locais."""
    pasta_pareto = resolver_caminho_local(config["resultados"]["pasta_pareto"])
    pasta_metricas = resolver_caminho_local(config["resultados"]["pasta_metricas"])
    os.makedirs(pasta_pareto, exist_ok=True)
    os.makedirs(pasta_metricas, exist_ok=True)

    caminho_pareto = os.path.join(pasta_pareto, f"pareto_{rotulo_execucao}.json")
    with open(caminho_pareto, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fase": "Fase 1 - pre-filtro NSGA-II",
                "configuracao": configuracao_execucao,
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
    parser.add_argument("--classificador", default=None)
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="pasta local para cache e progresso (default: config.yaml)",
    )
    args = parser.parse_args()

    config = carregar_config(args.config)
    cfg_otm = config["otimizacao"]

    # Argumentos de linha de comando sobrescrevem o config.yaml
    n_pop = args.n_pop or cfg_otm["n_pop"]
    n_gen = args.n_gen or cfg_otm["n_gen"]
    nome_clf = args.classificador or config["classificador"]["nome"]
    checkpoint_dir = resolver_caminho_local(
        args.checkpoint_dir or config["resultados"]["pasta_checkpoints"]
    )

    print(
        f"[config] h={nome_clf} | N_pop={n_pop} | N_gen={n_gen} | "
        "dados completos | operadores padrão do NSGA-II"
    )

    bases_Xy, ordem = preparar_bases(config)

    # Eu mantenho cache de avaliações e progresso por geração no disco local.
    # O contexto identifica o experimento; mudar qualquer item daria
    # critérios diferentes, então o cache é segregado por contexto.
    # Eu sempre habilito o checkpoint local porque cada avaliação completa é cara.
    contexto = {
        "formulacao": "biobjetivo_cross_medio_features_v2_float32",
        "classificador": nome_clf,
        "bases": sorted(bases_Xy.keys()),
        "arquivos": {
            nome: info["arquivo"]
            for nome, info in config["datasets"]["bases"].items()
        },
        "atributos": ordem,
        "d": len(ordem),
        "dados_completos": True,
        "seed": cfg_otm["seed"],
    }
    sufixo = f"{nome_clf}_full_v2_s{cfg_otm['seed']}"
    cache = CacheAvaliacoes(
        os.path.join(checkpoint_dir, f"cache_{sufixo}.jsonl"), contexto
    )
    registro = RegistroProgresso(
        os.path.join(checkpoint_dir, f"progresso_{sufixo}.json")
    )
    print(
        f"[checkpoint] {len(cache)} avaliações recuperadas em {checkpoint_dir}"
    )

    inicio = time.time()
    resultado = executar_otimizacao(
        bases_Xy,
        nome_clf,
        n_pop=n_pop,
        n_gen=n_gen,
        seed=cfg_otm["seed"],
        cache=cache,
        registro_progresso=registro,
    )
    duracao = time.time() - inicio

    print(f"\n[fim] {len(resultado['mascaras'])} soluções não-dominadas em P* "
          f"({duracao / 60:.1f} min)")

    rotulo = time.strftime("%Y%m%d_%H%M%S") + f"_{nome_clf}"
    salvar_resultados(
        resultado,
        ordem,
        config,
        rotulo,
        {
            "classificador": nome_clf,
            "n_pop": n_pop,
            "n_gen": n_gen,
            "seed": cfg_otm["seed"],
            "dados_completos": True,
            "operadores_nsga2": "padrao_pymoo",
        },
    )


if __name__ == "__main__":
    main()
