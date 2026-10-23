"""Eu gero gráficos complementares da MLP a partir do progresso salvo, só com validação.

Eu não treino, não abro o teste e não altero o experimento em execução.
G1: coordenadas paralelas; G3: desempenho por base; G4: classes; G6: curvas de treino.
"""
import argparse
import csv
from pathlib import Path
import json
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from Modulos.busca_mlp import escolher_solucoes

# Eu fixo cor e marcador pelo critério de escolha; a ordem passou na checagem de
# daltonismo para todos os pares (ΔE OKLab×100 ≥ 13 simulado, ≥ 19 normal).
ESTILOS = {'maior_f1': ('Maior F1', '#2a78d6', 'o'),
           'compromisso': ('Compromisso', '#008300', 'D'),
           'mais_compacta': ('Mais compacta', '#e87ba4', 's'),
           'mais_rapida': ('Mais rápida', '#eda100', '^'),
           'baseline': ('Baseline', '#52514e', 'X')}
TEXTO, TEXTO_2, GRADE, PARETO = '#0b0b0b', '#52514e', '#e4e3df', '#c9c8c3'
AZUIS = LinearSegmentedColormap.from_list('azuis', ['#f4f8fd', '#cde2fb', '#9ec5f4', '#6da7ec',
                                                    '#3987e5', '#256abf', '#184f95', '#0d366b'])
ABREV = {'NF-UNSW-NB15-v2': 'UNSW', 'NF-ToN-IoT-v2': 'ToN', 'NF-CSE-CIC-IDS2018-v2': 'CIC'}

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': TEXTO_2, 'axes.labelcolor': TEXTO,
                     'xtick.color': TEXTO_2, 'ytick.color': TEXTO_2, 'axes.titlesize': 10,
                     'legend.frameon': False})


def indice_benigno(classes):
    return [c.casefold() for c in classes].index('benign')


def metricas_binarias(cm, benigno):
    """TNR = benigno previsto como benigno; TPR = ataque previsto como qualquer ataque."""
    cm = np.asarray(cm, dtype=float)
    ataques = np.arange(len(cm)) != benigno
    tn, n_benigno = cm[benigno, benigno], cm[benigno].sum()
    tp, n_ataque = cm[np.ix_(ataques, ataques)].sum(), cm[ataques].sum()
    return dict(tnr=tn/n_benigno if n_benigno else np.nan, tpr=tp/n_ataque if n_ataque else np.nan)


def f1_macro_presentes(cm):
    """Macro-F1 só sobre classes com suporte na matriz; classes ausentes não contam como zero."""
    cm = np.asarray(cm, dtype=float)
    suporte, previstos, acertos = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    presentes = suporte > 0
    if not presentes.any():
        return np.nan
    return float((2*acertos[presentes]/(suporte[presentes]+previstos[presentes])).mean())


def medidas(sol, benigno):
    """Eu derivo tudo das matrizes de confusão de validação já salvas."""
    geral = sol['validacao']['geral']
    linha = dict(id=sol['id'], k=sol['k'], erro=1-geral['f1_macro'], f1_macro=geral['f1_macro'],
                 us_evento=sol['tempo']['mediana_s']*1e6, acuracia=geral['acuracia'],
                 **metricas_binarias(geral['matriz_confusao'], benigno))
    for base, m in sol['validacao']['por_base'].items():
        a = ABREV.get(base, base)
        cm = np.asarray(m['matriz_confusao'])
        linha[f'acuracia_{a}'] = m['acuracia']
        linha[f'f1_{a}'] = f1_macro_presentes(cm)
        for nome, v in metricas_binarias(cm, benigno).items():
            linha[f'{nome}_{a}'] = v
    return linha


def destacadas(estado):
    """Eu uso as mesmas escolhas fixadas antes do teste, na mesma ordem de estilos."""
    por_id = {s['id']: s for s in estado['solucoes']}
    saida = []
    for escolha in escolher_solucoes(estado):
        sol = dict(estado['baseline'], id='Baseline') if escolha['id'] == 'Baseline' else por_id[escolha['id']]
        criterio = min(escolha['criterios'], key=list(ESTILOS).index)
        nome = ' / '.join(ESTILOS[c][0] for c in sorted(escolha['criterios'], key=list(ESTILOS).index))
        rotulo = nome if escolha['id'] == 'Baseline' else f"{nome} ({escolha['id']})"
        saida.append(dict(sol=sol, criterio=criterio, rotulo=rotulo))
    return sorted(saida, key=lambda d: list(ESTILOS).index(d['criterio']))


def salvar_csv(caminho, linhas):
    with Path(caminho).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader(); w.writerows(linhas)


def coordenadas_paralelas(linhas, destaques, bases):
    """G1: cada linha é uma solução; cada eixo tem escala própria e para cima é melhor."""
    eixos = [('erro', '1 − F1-macro', True, '.3f'), ('k', 'Atributos (k)', True, '.0f'),
             ('us_evento', 'µs/evento', True, '.2f'), ('acuracia', 'Acurácia', False, '.2f'),
             ('tnr', 'TNR\n(benigno)', False, '.2f'), ('tpr', 'TPR\n(ataque)', False, '.3f')]
    eixos += [(f'f1_{ABREV.get(b, b)}', f'F1-macro\n{ABREV.get(b, b)}', False, '.2f') for b in bases]
    ids = [l['id'] for l in linhas]
    V = np.array([[l[e] for e, *_ in eixos] for l in linhas], dtype=float)
    lo, hi = np.nanmin(V, axis=0), np.nanmax(V, axis=0)
    amplitude = np.where(hi > lo, hi-lo, 1.)
    N = (V-lo)/amplitude
    menor = np.array([m for _, _, m, _ in eixos])
    N[:, menor] = 1-N[:, menor]
    xs = np.arange(len(eixos))
    fig, ax = plt.subplots(figsize=(14, 5.2))
    marcados = {d['sol']['id'] for d in destaques}
    for i, n in enumerate(N):
        if ids[i] not in marcados:
            ax.plot(xs, n, color=PARETO, lw=.9, zorder=1)
    ax.plot([], [], color=PARETO, lw=.9, label='Demais soluções do Pareto')
    for d in destaques:
        _, cor, marcador = ESTILOS[d['criterio']]
        ax.plot(xs, N[ids.index(d['sol']['id'])], '--', color=cor, lw=2, marker=marcador, ms=9,
                markeredgecolor='white', markeredgewidth=1.2, label=d['rotulo'], zorder=3)
    for j, (_, rotulo, m, fmt) in enumerate(eixos):
        ax.axvline(j, color=TEXTO_2, lw=.8, zorder=0)
        for t in np.linspace(0, 1, 5):
            valor = lo[j] + (1-t if m else t)*(hi[j]-lo[j])
            ax.text(j-.15, t, format(valor, fmt), ha='right', va='center', fontsize=7.5, color=TEXTO_2,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=.6), zorder=4)
        ax.text(j, 1.06, f"{rotulo} {'↓' if m else '↑'}", ha='center', va='bottom', fontsize=9, color=TEXTO)
    ax.set(xlim=(-.45, len(eixos)-.6), ylim=(-.04, 1.04), xticks=[], yticks=[])
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.02), ncol=3, fontsize=8.5)
    ax.set_title('Soluções do Pareto em todas as métricas — para cima é melhor em todos os eixos '
                 '(↓/↑ indicam a direção original)', pad=44)
    return fig


def desempenho_por_base(linhas_dest, destaques, bases):
    """G3: barras agrupadas por base; um painel por métrica, mesma escala 0–1."""
    grupos = [ABREV.get(b, b) for b in bases] + ['Conjunto']
    paineis = [('acuracia', 'Acurácia', 'Acurácia'), ('f1', 'F1-macro (classes presentes na base)', 'F1-macro'),
               ('tnr', 'TNR — benigno classificado como benigno', 'TNR'),
               ('tpr', 'TPR — ataque detectado como ataque', 'TPR')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharey=True)
    largura = .8/len(destaques)
    xs = np.arange(len(grupos))
    for ax, (chave, titulo, rotulo_y) in zip(axes.flat, paineis):
        for i, (d, l) in enumerate(zip(destaques, linhas_dest)):
            _, cor, _ = ESTILOS[d['criterio']]
            geral = l['f1_macro'] if chave == 'f1' else l[chave]
            valores = [l[f'{chave}_{g}'] for g in grupos[:-1]] + [geral]
            ax.bar(xs-.4+largura*(i+.5), valores, largura, color=cor, edgecolor='white',
                   linewidth=1.5, label=d['rotulo'], zorder=2)
        ax.set(title=titulo, xticks=xs, xticklabels=grupos, ylim=(0, 1),
               ylabel=f'{rotulo_y} (0–1, maior é melhor)')
        # Eu repito os valores do eixo nos painéis da direita, que o sharey esconderia.
        ax.tick_params(axis='y', labelleft=True)
        ax.yaxis.grid(color=GRADE, lw=.8, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(destaques), fontsize=8.5,
               bbox_to_anchor=(.5, .985))
    fig.suptitle('Desempenho por base na validação — soluções escolhidas antes do teste', y=1.02)
    return fig


def mapa_classes(destaques, classes, bases, benigno):
    """G4: recall e F1 por classe; linhas agrupadas pela base com mais exemplos da classe."""
    suporte_base = np.array([np.asarray(m['matriz_confusao']).sum(axis=1)
                             for m in destaques[0]['sol']['validacao']['por_base'].values()])
    principal = suporte_base.argmax(axis=0)
    suporte = suporte_base.sum(axis=0)
    ordem = [benigno] + sorted((c for c in range(len(classes)) if c != benigno),
                               key=lambda c: (principal[c], -suporte[c]))
    rotulos = [f"{'todas' if c == benigno else ABREV.get(bases[principal[c]], bases[principal[c]])} · "
               f"{classes[c]} (n={suporte[c]:,})".replace(',', '.') for c in ordem]
    colunas = [d['rotulo'].replace(' / ', '/\n').replace(' (', '\n(').replace(' ', '\n', 1)
               if len(d['rotulo']) > 12 else d['rotulo'] for d in destaques]
    fig, axes = plt.subplots(1, 2, figsize=(14, 11), sharey=True, layout='constrained')
    tabelas = {}
    for ax, chave, titulo in [(axes[0], 'recall', 'Recall por classe (quanto da classe foi reconhecido)'),
                              (axes[1], 'f1', 'F1 por classe (recall e precisão juntos)')]:
        M = np.array([[d['sol']['validacao']['geral']['por_classe'][c][chave] for d in destaques]
                      for c in ordem])
        tabelas[chave] = M
        im = ax.imshow(M, cmap=AZUIS, vmin=0, vmax=1, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if M[i, j] > .55 else TEXTO)
        ax.set(title=titulo, xticks=np.arange(len(colunas)), yticks=np.arange(len(ordem)))
        ax.set_xticklabels(colunas, fontsize=7.5)
        ax.set_yticklabels(rotulos, fontsize=7.5)
        ax.tick_params(length=0)
        quebras = [i for i in range(2, len(ordem)) if principal[ordem[i]] != principal[ordem[i-1]]]
        for q in [1] + quebras:
            ax.axhline(q-.5, color='white', lw=3)
        for lado in ax.spines.values():
            lado.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=.4, label='0 = nunca acerta · 1 = perfeito')
    fig.suptitle('Classes na validação conjunta — prefixo indica a base com mais exemplos da classe')
    return fig, ordem, tabelas


def curvas_treino(destaques):
    """G6: F1 de validação e loss de treino por época; painéis separados, sem eixo duplo."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for d in destaques:
        _, cor, marcador = ESTILOS[d['criterio']]
        t = d['sol']['treinamento']
        ep = [h['epoca'] for h in t['historico']]
        for ax, chave in [(axes[0], 'f1_validacao'), (axes[1], 'loss')]:
            v = [h[chave] for h in t['historico']]
            ax.plot(ep, v, color=cor, lw=2, label=d['rotulo'], zorder=2)
            melhor = ep.index(t['melhor_epoca'])
            ax.plot(ep[melhor], v[melhor], marcador, color=cor, ms=10,
                    markeredgecolor='white', markeredgewidth=1.2, zorder=3)
    axes[0].set(title='F1-macro de validação por época (marcador = melhor época, modelo mantido)',
                xlabel='Época', ylabel='F1-macro')
    axes[1].set(title='Loss de treino por época', xlabel='Época', ylabel='Loss (entropia cruzada ponderada)')
    for ax in axes:
        ax.grid(color=GRADE, lw=.8, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8)
    return fig


def gerar(estado, saida):
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    classes = estado['baseline']['validacao']['geral']['classes']
    bases = list(estado['baseline']['validacao']['por_base'])
    benigno = indice_benigno(classes)
    destaques = destacadas(estado)
    linhas = [medidas(s, benigno) for s in estado['solucoes'] + [dict(estado['baseline'], id='Baseline')]]
    por_id = {l['id']: l for l in linhas}
    linhas_dest = [por_id[d['sol']['id']] for d in destaques]
    rodape = (f"{estado['status']} | geração {estado['historico'][-1]['geracao']}/"
              f"{estado['geracoes_previstas']} | seed 42 | somente validação")
    figuras = []
    with PdfPages(saida/'graficos_extras.pdf') as pdf:
        def salvar(fig, nome, y=-.02):
            fig.text(.5, y, rodape, ha='center', fontsize=7.5, color=TEXTO_2)
            fig.savefig(saida/f'{nome}.png', dpi=160, bbox_inches='tight')
            fig.savefig(saida/f'{nome}.pdf', bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            figuras.append(nome)
        salvar(coordenadas_paralelas(linhas, destaques, bases), 'g1_coordenadas_paralelas', y=-.06)
        salvar(desempenho_por_base(linhas_dest, destaques, bases), 'g3_desempenho_por_base')
        fig, ordem, tabelas = mapa_classes(destaques, classes, bases, benigno)
        salvar(fig, 'g4_classes')
        salvar(curvas_treino(destaques), 'g6_curvas_treino')
    criterio = {d['sol']['id']: d['rotulo'] for d in destaques}
    salvar_csv(saida/'metricas_solucoes.csv', [dict(l, destaque=criterio.get(l['id'], '')) for l in linhas])
    salvar_csv(saida/'classes_validacao.csv',
               [dict(classe=classes[c], solucao=d['rotulo'], recall=tabelas['recall'][i, j],
                     f1=tabelas['f1'][i, j]) for i, c in enumerate(ordem) for j, d in enumerate(destaques)])
    salvar_csv(saida/'curvas_treino.csv',
               [dict(solucao=d['rotulo'], **h) for d in destaques for h in d['sol']['treinamento']['historico']])
    return figuras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pasta = RAIZ/'Resultados/mlp_conjunto_v1'
    parser.add_argument('--progresso', default=str(pasta/'progresso.json'))
    parser.add_argument('--saida', default=str(pasta/'figuras_extras'))
    args = parser.parse_args()
    figuras = gerar(json.loads(Path(args.progresso).read_text()), args.saida)
    print(f'[fim] {len(figuras)} gráficos em {args.saida}', flush=True)


if __name__ == '__main__':
    main()
