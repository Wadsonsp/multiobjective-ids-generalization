"""Eu gero gráficos complementares da MLP a partir do progresso salvo, só com validação.

Eu não treino, não abro o teste e não altero o experimento em execução.
G1: coordenadas paralelas; G3: desempenho por base; G4: classes; G6: curvas de treino;
G7: classes raras; G8: quanto cada classe move o F1-macro e a acurácia.
G7b e G8b: as mesmas mensagens para slides, para quem não é da área.
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
# Eu chamo de rara a classe com menos exemplos que isto na validação conjunta.
LIMITE_RARAS = 1000
RARA, COMUM = '#256abf', PARETO
# Eu uso letras maiores nos gráficos simples, feitos para slides 16:9.
SLIDE = {'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 13,
         'xtick.labelsize': 12, 'ytick.labelsize': 13, 'legend.fontsize': 12}

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


def classes_raras(sol, limite=LIMITE_RARAS):
    """Classes com 1 a `limite`−1 exemplos na validação conjunta, da menor para a maior."""
    por_classe = sol['validacao']['geral']['por_classe']
    return sorted((c for c, p in enumerate(por_classe) if 0 < p['suporte'] < limite),
                  key=lambda c: por_classe[c]['suporte'])


def rotulo_classe(classes, por_classe, c):
    return f"{classes[c]} (n={por_classe[c]['suporte']:,})".replace(',', '.')


def f1_classes_raras(destaques, classes, raras):
    """G7: F1, precisão e recall das classes raras, só para maior F1, compromisso e baseline."""
    escolhidos = [d for d in destaques if d['criterio'] in ('maior_f1', 'compromisso', 'baseline')]
    por_classe = escolhidos[0]['sol']['validacao']['geral']['por_classe']
    ys = np.arange(len(raras))
    # Eu desloco os marcadores na vertical para que valores iguais não se escondam.
    desvios = np.linspace(-.22, .22, len(escolhidos)) if len(escolhidos) > 1 else [0.]
    fig, axes = plt.subplots(1, 3, figsize=(14, .42*len(raras)+2.4), sharey=True)
    for ax, chave, titulo in [(axes[0], 'f1', 'F1 por classe'),
                              (axes[1], 'precisao', 'Precisão (do que previ como a classe, quanto era dela)'),
                              (axes[2], 'recall', 'Recall (quanto da classe foi reconhecido)')]:
        for d, dy in zip(escolhidos, desvios):
            _, cor, marcador = ESTILOS[d['criterio']]
            pc = d['sol']['validacao']['geral']['por_classe']
            ax.scatter([pc[c][chave] for c in raras], ys+dy, color=cor, marker=marcador, s=48,
                       edgecolors='white', linewidths=1, label=d['rotulo'], zorder=3)
        ax.set(title=titulo, xlim=(-.03, 1.03), xlabel='0–1, maior é melhor')
        ax.xaxis.grid(color=GRADE, lw=.8, zorder=0)
        for y in ys[:-1]:
            ax.axhline(y+.5, color=GRADE, lw=.6, zorder=0)
        ax.tick_params(axis='y', length=0)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set(yticks=ys, ylim=(len(raras)-.5, -.5))
    axes[0].set_yticklabels([rotulo_classe(classes, por_classe, c) for c in raras])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(escolhidos), fontsize=8.5,
               bbox_to_anchor=(.5, 1.0))
    fig.suptitle(f'Classes raras (menos de {LIMITE_RARAS:,} exemplos na validação) — '.replace(',', '.')
                 + f'cada classe vale 1/{len(classes)} do F1-macro, qualquer que seja o n', y=1.05)
    return fig


def decompor_diferenca(escolhida, baseline):
    """Contribuição exata de cada classe para a diferença de F1-macro e de acurácia.

    ΔF1-macro = Σ ΔF1_c / C, porque o F1-macro é a média simples das C classes;
    Δacurácia = Σ Δacertos_c / N, porque a acurácia conta exemplos.
    """
    ga, gb = escolhida['validacao']['geral'], baseline['validacao']['geral']
    cma, cmb = np.asarray(ga['matriz_confusao']), np.asarray(gb['matriz_confusao'])
    if not np.array_equal(cma.sum(axis=1), cmb.sum(axis=1)):
        raise ValueError('As duas matrizes precisam vir da mesma validação (mesmo suporte por classe).')
    d_f1 =np.array([pa['f1']-pb['f1'] for pa, pb in zip(ga['por_classe'], gb['por_classe'])])/len(cmb)
    d_acuracia = (np.diag(cma)-np.diag(cmb))/cmb.sum()
    return d_f1, d_acuracia


def contribuicao_classes(rotulo, d_f1, d_acuracia, classes, por_classe, raras):
    """G8: barras por classe; à esquerda move o F1-macro, à direita move a acurácia."""
    ordem = sorted(range(len(classes)), key=lambda c: por_classe[c]['suporte'])
    ys = np.arange(len(ordem))
    rara = np.isin(ordem, raras)
    cores = np.where(rara, RARA, COMUM)
    fig, axes = plt.subplots(1, 2, figsize=(14, .3*len(ordem)+2.6), sharey=True)
    for ax, valores, nome, fmt in [(axes[0], d_f1, 'F1-macro', '+.4f'),
                                   (axes[1], d_acuracia, 'Acurácia', '+.6f')]:
        v = valores[ordem]
        ax.barh(ys, v, color=cores, height=.7, zorder=2)
        limite = max(np.abs(v).max(), 1/len(classes) if nome == 'F1-macro' else 0) * 1.45 or 1.
        for y, x in zip(ys, v):
            texto = f' {x:+.4f} ' if round(x, 4) else ' ≈0 '
            ax.text(x, y, texto, ha='left' if x >= 0 else 'right', va='center', fontsize=6.5, color=TEXTO_2)
        ax.axvline(0, color=TEXTO_2, lw=.8, zorder=1)
        ax.set(xlim=(-limite, limite), xlabel=f'Contribuição para a diferença de {nome}',
               title=f'{nome}: {valores.sum():{fmt}} no total, '
                     f'{valores[raras].sum():{fmt}} vindo das classes raras')
        ax.xaxis.grid(color=GRADE, lw=.8, zorder=0)
        ax.tick_params(axis='y', length=0)
        ax.spines[['top', 'right']].set_visible(False)
    for x in (-1/len(classes), 1/len(classes)):
        axes[0].axvline(x, color=TEXTO_2, ls='--', lw=.8, zorder=1)
    axes[0].set(yticks=ys, ylim=(len(ordem)-.5, -.5))
    axes[0].set_yticklabels([rotulo_classe(classes, por_classe, c) for c in ordem], fontsize=7.5)
    for marca, cor in zip(axes[0].get_yticklabels(), cores):
        marca.set_color(TEXTO if cor == RARA else TEXTO_2)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=RARA),
                        plt.Rectangle((0, 0), 1, 1, color=COMUM),
                        plt.Line2D([], [], color=TEXTO_2, ls='--', lw=.8)],
               labels=[f'Classe rara (n < {LIMITE_RARAS:,})'.replace(',', '.'), 'Demais classes',
                       f'±1/{len(classes)}: o máximo que uma classe move o F1-macro'],
               loc='upper center', ncol=3, fontsize=8.5, bbox_to_anchor=(.5, .985))
    fig.suptitle(f'Quanto cada classe move as métricas: {rotulo} − Baseline (validação)\n'
                 'A soma das barras de cada painel é exatamente a diferença total da métrica', y=1.02)
    return fig


def br(x, fmt='.1f'):
    """Eu escrevo números com vírgula decimal e sinal de menos tipográfico."""
    return format(x, fmt).replace('.', ',').replace('-', '−')


def por_cento(x):
    """Porcentagem inteira; abaixo de 1% eu escrevo '<1%' para não sugerir zero."""
    return '<1%' if 0 < x < 1 else f'{x:.0f}%'


def milhar(n):
    return f'{int(n):,}'.replace(',', '.')


def exemplos(n):
    return 'um único exemplo' if n == 1 else f'{milhar(n)} exemplos'


def registros(n):
    return f"{milhar(n)} registro{'' if n == 1 else 's'}"


def titulos_slide(fig, titulo, subtitulo):
    linhas = subtitulo.count('\n') + 1
    fig.suptitle(titulo, fontsize=18, fontweight='bold', y=1.02+.04*linhas)
    fig.text(.5, .985, subtitulo, ha='center', va='bottom', fontsize=13, color=TEXTO_2, linespacing=1.4)


def classes_raras_simples(baseline, classes, raras):
    """G7b (slides): no baseline, quanto de cada classe rara o modelo identificou e quanto acertou ao apontá-la.

    Eu chamo de "apontar" a previsão do modelo. Um apontamento errado pode ser tráfego normal
    (alarme falso) ou outro ataque com o tipo trocado; por isso não uso a palavra "alarme".
    """
    geral = baseline['validacao']['geral']
    pc, cm = geral['por_classe'], np.asarray(geral['matriz_confusao'])
    benigno = indice_benigno(classes)
    encontrou = np.array([pc[c]['recall'] for c in raras])*100
    certos = np.array([pc[c]['precisao'] for c in raras])*100
    ys = np.arange(len(raras))
    with plt.rc_context(SLIDE):
        fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), sharey=True)
        for ax, v, cor, titulo, explicacao in [
                (axes[0], encontrou, '#2a78d6', 'Identificou',
                 'dos registros da classe, quantos o modelo apontou como dela'),
                (axes[1], certos, '#eda100', 'Acertou ao apontar',
                 'dos registros apontados como a classe, quantos eram dela')]:
            ax.barh(ys, v, color=cor, height=.65, zorder=2)
            for y, x in zip(ys, v):
                ax.text(x+1.5, y, por_cento(x), va='center', color=TEXTO, zorder=3,
                        bbox=dict(facecolor='white', edgecolor='none', pad=1))
            ax.set_title(titulo, fontweight='bold', loc='left')
            ax.set(xlim=(0, 118), xticks=[0, 25, 50, 75, 100], xlabel=explicacao)
            ax.set_xticklabels([f'{t}%' for t in (0, 25, 50, 75, 100)])
            ax.xaxis.grid(color=GRADE, lw=.8, zorder=0)
            ax.tick_params(axis='y', length=0)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].set(yticks=ys, ylim=(len(raras)-.5, -.5))
        axes[0].set_yticklabels([f"{classes[c]} ({exemplos(pc[c]['suporte'])})" for c in raras])
        axes[1].axvline(10, color=TEXTO_2, ls='--', lw=1.2, zorder=1)
        axes[1].text(11, len(raras)-.55, '10%', color=TEXTO_2, fontsize=11)
        # Eu explico, com as contagens da matriz, a classe em que identificar e acertar mais se
        # distanciam. Exijo 100 exemplos para que a porcentagem não descreva um ou dois registros.
        suporte = np.array([pc[c]['suporte'] for c in raras])
        i = int(np.argmax(np.where(suporte >= 100, encontrou-certos, -np.inf)))
        c = raras[i]
        acertos, apontados = cm[c, c], cm[:, c].sum()
        axes[1].annotate(f'“{classes[c]}”: identificou {milhar(acertos)} dos {registros(cm[c].sum())}\n'
                         f'da classe, mas, dos {registros(apontados)} que apontou\n'
                         f'como “{classes[c]}”, só {milhar(acertos)} eram dela',
                         xy=(certos[i]+9, ys[i]), xytext=(30, ys[i]), va='center', fontsize=11.5,
                         arrowprops=dict(arrowstyle='->', color=TEXTO_2))
        errados = sum(cm[:, r].sum()-cm[r, r] for r in raras)
        do_benigno = sum(cm[benigno, r] for r in raras)
        subtitulo = (f"Modelo com todos os {baseline['k']} atributos (baseline). Classe rara: menos de "
                     f'{milhar(LIMITE_RARAS)} exemplos nos dados de validação.')
        if errados:
            subtitulo += (f'\nDos {registros(errados)} apontados por engano como uma dessas classes, '
                          f'{br(100*do_benigno/errados)}% eram tráfego normal (alarme falso) e o resto, outro ataque.')
        titulos_slide(fig, f'Em {int((certos < 10).sum())} das {len(raras)} classes raras, menos de 10% '
                           'do que o modelo apontou como a classe era dela', subtitulo)
    return fig


def peso(p):
    """Uma casa decimal, para que as partes de cada barra somem 100%; abaixo de 0,05% eu escrevo '<0,1%'."""
    return '<0,1%' if 0 < p < .05 else f'{br(p)}%'


def contribuicao_simples(maior, baseline, d_f1, classes, raras, benigno, n_solucoes):
    """G8b (slides): de onde vem a vantagem da solução de maior F1 e por que a acurácia não a vê."""
    C = len(classes)
    suporte = np.array([p['suporte'] for p in baseline['validacao']['geral']['por_classe']], dtype=float)
    cm_base = np.asarray(baseline['validacao']['geral']['matriz_confusao'])
    cm_maior = np.asarray(maior['sol']['validacao']['geral']['matriz_confusao'])
    topo = max(raras, key=lambda c: abs(d_f1[c]))
    outras = [c for c in raras if c != topo]
    comuns = [c for c in range(C) if c not in raras]
    passos = 100*np.array([d_f1[topo], d_f1[outras].sum(), d_f1[comuns].sum()])
    total = passos.sum()
    ident = maior['sol']['id']
    f1_base = 100*baseline['validacao']['geral']['f1_macro']
    f1_maior = 100*maior['sol']['validacao']['geral']['f1_macro']
    with plt.rc_context(SLIDE):
        fig, (esq, dir_) = plt.subplots(1, 2, figsize=(13.33, 7.5), gridspec_kw=dict(width_ratios=[1.25, 1]))
        # Cascata: cada passo começa onde o anterior terminou; a última barra é o total.
        xs = np.arange(4)
        acumulado = np.cumsum(passos)
        inicio = np.concatenate([[0], acumulado[:-1]])
        esq.bar(xs[:3], passos, bottom=inicio, color=['#e87ba4', RARA, COMUM], width=.6, zorder=2)
        esq.bar(xs[3], total, color=TEXTO, width=.6, zorder=2)
        for x, fim, v in zip(xs, list(acumulado)+[total], list(passos)+[total]):
            esq.annotate(br(v, '+.1f'), (x, fim), xytext=(0, 5 if v >= 0 else -5), textcoords='offset points',
                         ha='center', va='bottom' if v >= 0 else 'top', fontsize=14, fontweight='bold')
        for x, nivel in zip(xs[:3], acumulado):
            esq.plot([x+.3, x+.7], [nivel, nivel], color=TEXTO_2, lw=1, ls=':', zorder=1)
        esq.axhline(0, color=TEXTO_2, lw=.8)
        alto, baixo = max(acumulado.max(), total, 0), min(acumulado.min(), inicio.min(), total, 0)
        folga = .15*(alto-baixo)+.1
        esq.set_ylim(baixo-folga if baixo < 0 else 0, alto+folga)
        esq.set_xticks(xs)
        esq.set_xticklabels([f"{classes[topo]}\n({exemplos(int(suporte[topo])).replace(' exemplo', chr(10)+'exemplo')})",
                             f'outras {len(outras)}\nraras', f'{len(comuns)} não raras\n(com tráfego\nnormal)',
                             'vantagem\ntotal'], fontsize=11.5)
        esq.set_xlabel('Valores arredondados; a soma exata dos passos é a vantagem total',
                       fontsize=11, color=TEXTO_2, labelpad=10)
        esq.set(ylabel='Pontos na nota F1-macro (escala 0 a 100)',
                title=f'Nota F1-macro: baseline {br(f1_base)} → {ident} {br(f1_maior)}')
        esq.yaxis.grid(color=GRADE, lw=.8, zorder=0)
        esq.tick_params(axis='x', length=0)
        esq.spines[['top', 'right']].set_visible(False)
        # Barras de 100%: quanto cada grupo pesa em cada nota.
        ataques_comuns = [c for c in comuns if c != benigno]
        grupos = [('tráfego normal', [benigno], TEXTO_2), (classes[topo], [topo], '#e87ba4'),
                  (f'outras {len(outras)} raras', outras, RARA),
                  (f'{len(ataques_comuns)} ataques\nnão raros', ataques_comuns, COMUM)]
        pesos = [[100*len(cs)/C for _, cs, _ in grupos],
                 [100*suporte[cs].sum()/suporte.sum() for _, cs, _ in grupos]]
        for y, ps in enumerate(pesos):
            esquerda, notas = 0, []
            for (nome, _, cor), p in zip(grupos, ps):
                dir_.barh(y, p, left=esquerda, color=cor, height=.5, edgecolor='white', lw=1.5)
                if p >= 9:
                    dir_.text(esquerda+p/2, y, f'{nome}\n{peso(p)}', ha='center', va='center', fontsize=11.5,
                              color='white' if cor in (TEXTO_2, RARA) else TEXTO)
                else:
                    notas.append(f"{nome.replace(chr(10), ' ')}: {peso(p)}")
                esquerda += p
            dir_.text(0, y+.32, ' · '.join(notas), va='top', fontsize=11.5, color=TEXTO_2)
        dir_.set(xlim=(0, 100), ylim=(1.75, -.5), yticks=[0, 1], xticks=[],
                 title='Quanto cada grupo pesa em cada nota')
        dir_.set_yticklabels(['F1-macro\n(cada tipo de\ntráfego vale\no mesmo)', 'Acurácia\n(cada registro\nvale o mesmo)'])
        dir_.tick_params(axis='y', length=0)
        for lado in dir_.spines.values():
            lado.set_visible(False)
        if total > 0 and passos[0] > total/2:
            titulo = (f'{100*passos[0]/total:.0f}% da vantagem da {ident} vem de uma classe com '
                      f'{exemplos(int(suporte[topo]))} ({classes[topo]})')
        else:
            titulo = f'De onde vem a diferença entre a {ident} e o baseline'
        # Eu descrevo a classe do primeiro passo pelas contagens, sem dizer "acertou": nos dois
        # modelos o que muda pode ser o acerto, os apontamentos errados ou ambos.
        def apontou(cm):
            certos = cm[topo, topo]
            return f"{registros(cm[:, topo].sum())} ({milhar(certos)} correto{'' if certos == 1 else 's'})"
        raras_pct = 100*suporte[raras].sum()/suporte.sum()
        titulos_slide(fig, titulo,
                      f'{ident}: maior F1-macro entre as {n_solucoes} soluções atuais da busca.\n'
                      f'Como {classes[topo]}, o baseline apontou {apontou(cm_base)}; '
                      f'a {ident}, {apontou(cm_maior)}.\n'
                      f'Na acurácia, as {len(raras)} classes raras somam {br(raras_pct, ".2f")}% dos registros; '
                      'por isso ela quase não as enxerga.')
        fig.subplots_adjust(wspace=.45)
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
    ultima = estado['historico'][-1]['geracao']
    rodape_slide = (f"{'Resultado parcial' if estado['status'] == 'PARCIAL' else 'Resultado'}: "
                    f"geração {ultima} de {estado['geracoes_previstas']} da busca · dados de validação · "
                    'uma única execução (seed 42)')
    figuras = []
    with PdfPages(saida/'graficos_extras.pdf') as pdf:
        def salvar(fig, nome, y=-.02, texto=rodape, tamanho=7.5):
            fig.text(.5, y, texto, ha='center', fontsize=tamanho, color=TEXTO_2)
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
        baseline = dict(estado['baseline'], id='Baseline')
        por_classe = baseline['validacao']['geral']['por_classe']
        raras = classes_raras(baseline)
        maior = next(d for d in destaques if d['criterio'] == 'maior_f1')
        d_f1, d_acuracia = decompor_diferenca(maior['sol'], baseline)
        if raras:
            salvar(f1_classes_raras(destaques, classes, raras), 'g7_classes_raras', y=-.03)
        salvar(contribuicao_classes(maior['rotulo'], d_f1, d_acuracia, classes, por_classe, raras),
               'g8_contribuicao_classes')
        if raras:
            salvar(classes_raras_simples(baseline, classes, raras), 'g7b_classes_raras_simples',
                   y=-.03, texto=rodape_slide, tamanho=11)
            salvar(contribuicao_simples(maior, baseline, d_f1, classes, raras, benigno, len(estado['solucoes'])),
                   'g8b_contribuicao_simples', y=-.08, texto=rodape_slide, tamanho=11)
    criterio = {d['sol']['id']: d['rotulo'] for d in destaques}
    salvar_csv(saida/'metricas_solucoes.csv', [dict(l, destaque=criterio.get(l['id'], '')) for l in linhas])
    salvar_csv(saida/'classes_validacao.csv',
               [dict(classe=classes[c], solucao=d['rotulo'], recall=tabelas['recall'][i, j],
                     f1=tabelas['f1'][i, j]) for i, c in enumerate(ordem) for j, d in enumerate(destaques)])
    salvar_csv(saida/'curvas_treino.csv',
               [dict(solucao=d['rotulo'], **h) for d in destaques for h in d['sol']['treinamento']['historico']])
    if raras:
        salvar_csv(saida/'classes_raras.csv',
                   [dict(classe=classes[c], suporte=por_classe[c]['suporte'], solucao=d['rotulo'],
                         **{m: d['sol']['validacao']['geral']['por_classe'][c][m]
                            for m in ('f1', 'precisao', 'recall')})
                    for c in raras for d in destaques if d['criterio'] in ('maior_f1', 'compromisso', 'baseline')])
    salvar_csv(saida/'contribuicao_classes.csv',
               [dict(classe=classes[c], suporte=por_classe[c]['suporte'], rara=c in raras,
                     comparacao=f"{maior['rotulo']} − Baseline", delta_f1_macro=d_f1[c],
                     delta_acuracia=d_acuracia[c]) for c in range(len(classes))])
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
