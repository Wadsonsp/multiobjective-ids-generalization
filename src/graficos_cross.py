# -*- coding: utf-8 -*-
"""Eu gero os gráficos locais da análise de generalização cross-dataset.

Lê os JSONs de avaliação gerados por src/algoritmo2_avaliacao.py (campo
'diagnostico_cross') e produz quatro figuras que sustentam a discussão
sobre as DUAS causas da queda de desempenho na transferência:

1. matriz_confusao  : heatmap da matriz de confusão cross-dataset, com a
   coluna extra "sem correspondência" evidenciando a taxonomia incompatível.
2. intra_vs_cross   : barras agrupadas comparando F1 intra x cross por
   solução, deixando visível o abismo entre os dois regimes.
3. decomposicao     : barras empilhadas separando, por direção, a parcela
   da queda atribuível a domain shift x incompatibilidade de taxonomia.
4. tradeoff         : dispersão k(x) x F1, mostrando o compromisso
   desempenho/parcimônia ao longo das soluções reavaliadas.

Uso local, após reavaliar as soluções:
    python src/graficos_cross.py --metricas Resultados/metricas \
        --saida Resultados/figuras
"""

import argparse
import glob
import json
import os

import numpy as np

# Eu uso um backend não interativo para salvar PNG sem abrir uma janela.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def carregar_avaliacoes(pasta_metricas):
    """Eu reúno as avaliações detalhadas salvas pela Fase 2."""
    avaliacoes = []
    for caminho in sorted(glob.glob(os.path.join(pasta_metricas, "avaliacao_*.json"))):
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
        # Eu aceito o formato atual, que agrupa várias soluções por arquivo.
        for avaliacao in dados.get("avaliacoes", []):
            if "diagnostico_cross" in avaliacao:
                avaliacao["__arquivo__"] = os.path.basename(caminho)
                avaliacoes.append(avaliacao)
    return avaliacoes


def plot_matriz_confusao(diag, direcao, caminho_saida):
    """Heatmap da matriz de confusão de uma direção de transferência.

    Linhas = classe verdadeira (base de destino); colunas = classe prevista
    (vocabulário da base de origem) + 1 coluna "sem correspondência".
    """
    info = diag[direcao]
    cm = np.array(info["matriz_confusao"], dtype=float)
    classes_teste = info["classes_teste"]
    rotulos_col = list(classes_teste) + ["(sem corresp.)"]

    # Normalizo por linha (recall por classe) para leitura independente de
    # tamanho de classe - cada linha soma 1
    soma_linha = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, soma_linha, out=np.zeros_like(cm), where=soma_linha > 0)
    # Removo colunas totalmente vazias para enxugar a figura (exceto a última)
    cm_norm = cm_norm[: len(classes_teste), :]

    fig, ax = plt.subplots(figsize=(max(6, len(rotulos_col) * 0.7),
                                    max(4, len(classes_teste) * 0.6)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(rotulos_col)))
    ax.set_yticks(range(len(classes_teste)))
    ax.set_xticklabels(rotulos_col, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes_teste, fontsize=8)
    ax.set_xlabel("Classe prevista (vocabulário da origem)")
    ax.set_ylabel("Classe verdadeira (base de destino)")
    ax.set_title(f"Matriz de confusão cross-dataset\n{direcao}", fontsize=10)

    # Anoto os valores nas células com proporção relevante
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            v = cm_norm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)

    fig.colorbar(im, ax=ax, label="Proporção (recall por classe)")
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return caminho_saida


def plot_intra_vs_cross(avaliacoes, caminho_saida):
    """Barras agrupadas: F1 intra (médio) x F1 cross (médio) por solução."""
    rotulos, f1_intra, f1_cross = [], [], []
    for a in avaliacoes:
        k = a.get("numero_atributos", "?")
        rotulos.append(f"k={k}")
        f1_intra.append(float(np.mean(list(a["f1_macro_intra"].values()))))
        f1_cross.append(float(np.mean(list(a["f1_macro_cross_por_direcao"].values()))))

    x = np.arange(len(rotulos))
    largura = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(rotulos) * 1.1), 4.5))
    ax.bar(x - largura / 2, f1_intra, largura, label="Intra-dataset", color="#2E5496")
    ax.bar(x + largura / 2, f1_cross, largura, label="Cross-dataset", color="#C0504D")

    ax.set_xticks(x)
    ax.set_xticklabels(rotulos)
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 1)
    ax.set_title("Desempenho intra-dataset vs. cross-dataset por solução")
    ax.legend()
    for i in range(len(rotulos)):
        ax.text(i - largura / 2, f1_intra[i] + 0.02, f"{f1_intra[i]:.2f}",
                ha="center", fontsize=8)
        ax.text(i + largura / 2, f1_cross[i] + 0.02, f"{f1_cross[i]:.2f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return caminho_saida


def plot_decomposicao(avaliacoes, caminho_saida):
    """Barras empilhadas: parcela da queda por causa, por direção.

    Para cada direção, decomponho o que "falta" para o desempenho perfeito:
    - parcela de incompatibilidade de taxonomia = proporção de amostras de
      classes que a origem nunca viu (limite estrutural, inevitável);
    - parcela atribuível a domain shift = erro restante nas classes
      conhecidas (o modelo poderia ter acertado, mas a distribuição mudou).
    Isso separa a queda "inevitável" da queda "de modelo".
    """
    direcoes, taxonomia, shift = [], [], []
    for a in avaliacoes:
        k = a.get("numero_atributos", "?")
        for direcao, info in a["diagnostico_cross"].items():
            origem = direcao.split("->")[0].replace("NF-", "").replace("-v2", "")
            destino = direcao.split("->")[1].replace("NF-", "").replace("-v2", "")
            direcoes.append(f"{origem}->{destino}\n(k={k})")
            p_desc = info["prop_desconhecidas"]
            acc_conh = info["acuracia_classes_conhecidas"]
            # fração de erro nas conhecidas, ponderada pela fração conhecida
            erro_shift = (1 - p_desc) * (1 - acc_conh)
            taxonomia.append(p_desc)
            shift.append(erro_shift)

    x = np.arange(len(direcoes))
    fig, ax = plt.subplots(figsize=(max(6, len(direcoes) * 1.3), 4.8))
    b1 = ax.bar(x, taxonomia, 0.55, label="Incompatibilidade de taxonomia\n(classe ausente no treino)",
                color="#8064A2")
    ax.bar(x, shift, 0.55, bottom=taxonomia,
           label="Domain shift\n(erro em classe conhecida)", color="#C0504D")

    ax.set_xticks(x)
    ax.set_xticklabels(direcoes, fontsize=8)
    ax.set_ylabel("Fração das amostras de teste")
    ax.set_ylim(0, 1)
    ax.set_title("Decomposição da queda cross-dataset por causa")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return caminho_saida


def plot_tradeoff(avaliacoes, caminho_saida):
    """Dispersão k(x) x F1, com F1 intra e cross sobrepostos."""
    ks, f1_intra, f1_cross = [], [], []
    for a in avaliacoes:
        ks.append(a.get("numero_atributos", 0))
        f1_intra.append(float(np.mean(list(a["f1_macro_intra"].values()))))
        f1_cross.append(float(np.mean(list(a["f1_macro_cross_por_direcao"].values()))))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(ks, f1_intra, s=70, color="#2E5496", label="Intra-dataset", zorder=3)
    ax.scatter(ks, f1_cross, s=70, color="#C0504D", label="Cross-dataset", zorder=3)
    for i, k in enumerate(ks):
        ax.annotate(f"k={k}", (k, f1_intra[i]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("Número de características selecionadas k(x)")
    ax.set_ylabel("F1-macro (médio)")
    ax.set_ylim(0, 1)
    ax.set_title("Compromisso entre número de características e desempenho")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)
    return caminho_saida


def main():
    parser = argparse.ArgumentParser(description="Gráficos da análise cross-dataset")
    parser.add_argument("--metricas", default="Resultados/metricas")
    parser.add_argument("--saida", default="Resultados/figuras")
    args = parser.parse_args()

    os.makedirs(args.saida, exist_ok=True)
    avaliacoes = carregar_avaliacoes(args.metricas)
    if not avaliacoes:
        print("Nenhuma avaliação com diagnóstico cross encontrada em",
              args.metricas, "- reavalie as soluções com a versão nova do código.")
        return

    print(f"{len(avaliacoes)} avaliação(ões) carregada(s).")

    # Gráficos agregados (todas as soluções juntas)
    print("[fig]", plot_intra_vs_cross(avaliacoes,
          os.path.join(args.saida, "intra_vs_cross.png")))
    print("[fig]", plot_decomposicao(avaliacoes,
          os.path.join(args.saida, "decomposicao_causas.png")))
    print("[fig]", plot_tradeoff(avaliacoes,
          os.path.join(args.saida, "tradeoff_k_f1.png")))

    # Matrizes de confusão: uma por direção, da solução de menor k
    # (a mais enxuta costuma ser a mais interessante de inspecionar)
    enxuta = min(avaliacoes, key=lambda a: a.get("numero_atributos", 999))
    k = enxuta.get("numero_atributos", "?")
    for direcao in enxuta["diagnostico_cross"]:
        nome = direcao.replace("->", "_para_").replace("NF-", "").replace("-v2", "")
        print("[fig]", plot_matriz_confusao(
            enxuta["diagnostico_cross"], direcao,
            os.path.join(args.saida, f"matriz_k{k}_{nome}.png")))


if __name__ == "__main__":
    main()
