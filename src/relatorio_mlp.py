"""Eu apresento o experimento MLP em 2D, sem impor curvatura ao Pareto."""
import csv
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from Modulos.dados_mlp import salvar_json


# Eu dou uma única matiz a cada objetivo, nas três figuras; mais escuro é sempre pior.
OBJETIVOS = dict(atributos=('Atributos selecionados', 'Greens', 'atributos = {:.0f}'),
                 erro=('1 − F1-macro de validação', 'Purples', 'erro 1 − F1 = {:.3f}'),
                 tempo=('Mediana de inferência (µs/evento em lote)', 'Blues', 'tempo = {:.2f} µs/evento'))


def mapa_objetivo(nome):
    """Eu corto o trecho mais claro da matiz para nenhum ponto sumir no fundo branco."""
    return LinearSegmentedColormap.from_list(nome, plt.get_cmap(OBJETIVOS[nome][1])(np.linspace(.3, 1, 256)))


def painel_pareto(fig, ax, valores, eixo_x, eixo_y, objetivo_cor, livre, sols, base):
    """Eu desenho uma projeção com a mesma codificação em todas as figuras."""
    xx, yy, cor = valores[eixo_x], valores[eixo_y], valores[objetivo_cor]
    ind = fronteira_2d(xx, yy)
    na_projecao = np.isin(np.arange(len(sols)), ind)
    mapa = mapa_objetivo(objetivo_cor)
    # Eu nomeio na legenda o objetivo da cor e o valor dos tons extremos.
    for fracao, tom, sentido, valor in [(0., 'clara', 'melhor', cor.min()), (1., 'escura', 'pior', cor.max())]:
        ax.scatter([], [], marker='s', s=70, color=mapa(fracao), edgecolors='#334155', linewidths=.7,
                   label=f'Cor {tom}: {OBJETIVOS[objetivo_cor][2].format(valor)} ({sentido})')
    def pontos(mascara, **estilo):
        return ax.scatter(xx[mascara], yy[mascara], c=cor[mascara], cmap=mapa,
                          vmin=cor.min(), vmax=cor.max(), edgecolors='#334155', linewidths=.7,
                          zorder=3, **estilo)
    sc = pontos(na_projecao, marker='o', s=70)
    pontos(~na_projecao & livre, marker='D', s=55)
    dominadas = ~na_projecao & ~livre
    if dominadas.any():
        ax.scatter(xx[dominadas], yy[dominadas], marker='x', s=60, c='#64748b', zorder=3,
                   label='Dominada nos três objetivos')
    ax.scatter([], [], marker='o', s=70, c='#cbd5e1', edgecolors='#334155', linewidths=.7,
               label='Na fronteira desta projeção')
    ax.scatter([], [], marker='D', s=55, c='#cbd5e1', edgecolors='#334155', linewidths=.7,
               label='Fora da fronteira 2D, não dominada nos 3 objetivos\n(compensa no objetivo da cor)')
    ax.plot(xx[ind], yy[ind], color='#334155', lw=1.2, zorder=2,
            label='Fronteira nesta projeção (segmentos entre pontos)')
    for s, xi, yi in zip(sols, xx, yy):
        ax.annotate(s['id'], (xi, yi), xytext=(5, 6), textcoords='offset points', fontsize=8)
    ax.scatter([base[eixo_x]], [base[eixo_y]], marker='X', s=100, c='#52514e', zorder=3, label='Baseline')
    fig.colorbar(sc, ax=ax, label=f'Cor = {OBJETIVOS[objetivo_cor][0]}\nmais escuro = pior')
    ax.set(xlabel=f'{OBJETIVOS[eixo_x][0]} — menor é melhor', ylabel=f'{OBJETIVOS[eixo_y][0]} — menor é melhor',
           title=f'{eixo_y.capitalize()} × {eixo_x} — cor: {objetivo_cor}')
    ax.grid(alpha=.2)
    return ind


def fronteira_2d(x, y):
    """Índices não dominados na projeção de dois objetivos minimizados."""
    x, y = np.asarray(x), np.asarray(y)
    indices = []
    melhor = float('inf')
    for i in sorted(range(len(x)), key=lambda i: (x[i], y[i])):
        if y[i] < melhor:
            indices.append(i)
            melhor = y[i]
    return indices


def nao_dominados(F):
    """Máscara das linhas de F não dominadas em todos os objetivos minimizados."""
    F = np.asarray(F, dtype=float)
    return np.array([not ((F <= f).all(axis=1) & (F < f).any(axis=1)).any() for f in F], dtype=bool)


def diagnosticar_curvatura(x, y):
    indices = fronteira_2d(x, y)
    resultado = dict(indices=indices, joelho_indice=None, desvio_normalizado=0.,
                     conclusao='Menos de três pontos na fronteira 2D; não avalio curvatura.')
    if len(indices) < 3:
        return resultado
    pontos = np.column_stack([x, y])[indices].astype(float)
    amplitude = np.ptp(pontos, axis=0)
    if (amplitude == 0).any():
        return resultado
    normal = (pontos-pontos.min(axis=0))/amplitude
    ganhos = (1-normal.sum(axis=1))/np.sqrt(2)
    i = int(np.argmax(ganhos))
    if ganhos[i] > 1e-8:
        resultado.update(joelho_indice=int(indices[i]), desvio_normalizado=float(ganhos[i]),
                         conclusao='Há um candidato geométrico a joelho abaixo da reta entre extremos. '
                                   'Não é prova de ótimo, convergência ou escolha automática.')
    else:
        resultado['conclusao'] = 'Não observei desvio em direção à origem abaixo da reta entre extremos.'
    return resultado


def gerar(estado, saida, meta):
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    sols = estado['solucoes']
    x = np.array([s['k'] for s in sols])
    erro = np.array([s['objetivos'][0] for s in sols])
    tempos = np.array([s['tempo']['mediana_s']*1e6 for s in sols])
    diag = diagnosticar_curvatura(x, erro)
    salvar_json(saida/'diagnostico_pareto.json', diag)
    figuras = []
    titulo_estado = f"{estado['status']} | geração {estado['historico'][-1]['geracao']}/{estado['geracoes_previstas']} | seed 42"
    with PdfPages(saida/'relatorio_mlp.pdf') as pdf:
        def salvar(fig, nome):
            fig.text(.5, .015, titulo_estado, ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .045, 1, 1))
            fig.savefig(saida/f'{nome}.png', dpi=160)
            fig.savefig(saida/f'{nome}.pdf')
            pdf.savefig(fig)
            plt.close(fig)
            figuras.append(nome)

        livre = nao_dominados([s['objetivos'] for s in sols])
        base = estado['baseline']
        base_valores = dict(atributos=base['k'], erro=base['objetivos'][0], tempo=base['tempo']['mediana_s']*1e6)
        valores = dict(atributos=x, erro=erro, tempo=tempos)

        fig, ax = plt.subplots(figsize=(10, 6))
        ind = painel_pareto(fig, ax, valores, 'atributos', 'erro', 'tempo', livre, sols, base_valores)
        if len(ind) >= 2:
            ax.plot(x[[ind[0], ind[-1]]], erro[[ind[0], ind[-1]]], '--', color='#94a3b8',
                    label='Reta entre extremos — referência geométrica')
        if diag['joelho_indice'] is not None:
            j = diag['joelho_indice']
            ax.scatter([x[j]], [erro[j]], marker='o', s=220, facecolors='none',
                       edgecolors='#dc2626', label='Candidato a joelho na projeção')
        ax.legend(fontsize=8)
        salvar(fig, '01_pareto_erro_atributos')

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        painel_pareto(fig, axes[0], valores, 'tempo', 'erro', 'atributos', livre, sols, base_valores)
        painel_pareto(fig, axes[1], valores, 'atributos', 'tempo', 'erro', livre, sols, base_valores)
        for ax in axes:
            ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2)
        fig.suptitle('Projeções 2D restantes — em cada painel, a cor mostra o objetivo fora dos eixos')
        salvar(fig, '02_pareto_tempo')

        hist = estado['historico']
        geracoes = [h['geracao'] for h in hist]
        valores = [h['hipervolume'] for h in hist]
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(geracoes, valores, 'o-'); axes[0].set(ylabel='Hipervolume', title='Evolução da busca de três objetivos')
        axes[1].bar(geracoes[1:], np.diff(valores)); axes[1].set(xlabel='Geração', ylabel='Variação do HV')
        for ax in axes: ax.grid(alpha=.2)
        salvar(fig, '03_hipervolume')

        avaliacoes = estado.get('teste', [])
        if avaliacoes:
            labels = [s['id'] + (' / Baseline' if 'baseline' in s['criterios'] and s['id'] != 'Baseline' else '') for s in avaliacoes]
            xx = np.arange(len(avaliacoes))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(xx-.2, [s['validacao']['validacao']['geral']['f1_macro'] for s in avaliacoes], .4, label='Validação usada na seleção')
            ax.bar(xx+.2, [s['medidas']['geral']['f1_macro'] for s in avaliacoes], .4, label='Teste reservado')
            ax.set(xticks=xx, xticklabels=labels, ylim=(0, 1), ylabel='F1-macro', title='Avaliação final das escolhas fixadas antes do teste')
            ax.legend(); salvar(fig, '04_validacao_teste')
            for sol in avaliacoes:
                medidas = [('geral', sol['medidas']['geral'])] + list(sol['medidas']['por_base'].items())
                for nome, medidas_base in medidas:
                    cm = np.array(medidas_base['matriz_confusao'])
                    suporte = cm.sum(axis=1, keepdims=True)
                    normal = np.divide(cm, suporte, out=np.zeros_like(cm, dtype=float), where=suporte > 0)
                    n = len(meta['classes'])
                    fig, axes = plt.subplots(1, 2, figsize=(max(13, n*.8), max(6, n*.35)))
                    for ax, valores_cm, subtitulo, percentual in [
                            (axes[0], cm, 'Contagens', False),
                            (axes[1], normal, 'Percentual por classe verdadeira; diagonal = recall', True)]:
                        im = ax.imshow(valores_cm, cmap='Blues', aspect='auto',
                                       vmin=0, vmax=1 if percentual else None)
                        ax.set(xticks=np.arange(n), yticks=np.arange(n), xticklabels=meta['classes'],
                               yticklabels=meta['classes'], xlabel='Classe prevista', ylabel='Classe verdadeira', title=subtitulo)
                        ax.tick_params(axis='x', labelrotation=90, labelsize=7)
                        ax.tick_params(axis='y', labelsize=7)
                        for i in range(n):
                            for j in range(n):
                                v = valores_cm[i, j]
                                if (v >= .01 or i == j) if percentual else v > 0:
                                    txt = ('—' if suporte[i, 0] == 0 else f'{v:.0%}') if percentual else str(v)
                                    ax.text(j, i, txt, ha='center', va='center', fontsize=6,
                                            color='white' if v > np.max(valores_cm)*.5 else 'black')
                        fig.colorbar(im, ax=ax, shrink=.7)
                    fig.suptitle(f"{sol['id']} | teste | {nome}")
                    seguro = nome.replace('/', '_')
                    salvar(fig, f"05_matriz_{sol['id']}_{seguro}")
                    with (saida/f"classes_{sol['id']}_{seguro}.csv").open('w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=['classe', 'suporte', 'precisao', 'recall', 'f1'])
                        w.writeheader(); w.writerows(medidas_base['por_classe'])
    with (saida/'resumo_validacao.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['solucao', 'atributos', 'f1_macro', 'mediana_us_evento', 'maximo_observado_us_evento'])
        for s in sols + [dict(estado['baseline'], id='Baseline')]:
            w.writerow([s['id'], s['k'], 1-s['objetivos'][0], s['tempo']['mediana_s']*1e6,
                        s['tempo']['maximo_observado_s']*1e6])
    texto = f'''# Meu relatório da MLP conjunta

{titulo_estado}

Eu treino uma MLP com dados das três bases. Eu separo treino, validação e teste por grupos de atributos idênticos.
Eu seleciono os atributos na validação. Eu uso o teste somente após fixar as escolhas.

## Como eu leio o Pareto

Eu minimizo erro (1 − F1-macro), proporção de atributos e segundos por evento.
Eu mostro somente gráficos 2D. Melhorar erro e quantidade de atributos significa avançar para o canto inferior esquerdo.
Eu desenho as três projeções com a mesma codificação. A cor mostra o objetivo que ficou fora dos eixos, com uma matiz
fixa por objetivo: azul para tempo, verde para atributos e roxo para erro; mais escuro é sempre pior.
Círculos estão na fronteira daquela projeção; losangos ficam fora dela, mas continuam não dominados nos três objetivos
porque vencem no objetivo da cor quem os domina na projeção. O baseline aparece como X cinza em todos os painéis.
Na legenda, dois quadrados mostram o tom claro e o escuro com o valor do objetivo da cor em cada extremo.
Uma solução pode ficar fora das três fronteiras projetadas e continuar não dominada: ela equilibra os três objetivos.
Eu não uso gráfico 3D estático: a perspectiva impede ler valores, e uma superfície exigiria interpolar entre soluções.
Para ver os três objetivos lado a lado sem projeção, eu uso as coordenadas paralelas (g1) em figuras_extras.
Eu ligo apenas os pontos não dominados da projeção; os segmentos não são resultados intermediários medidos.
Eu não imponho formato de barriga nem suavizo os pontos.

Minha verificação geométrica: {diag['conclusao']}

Eu calculo o HV nos três objetivos, com tempo transformado por t/(t+t_ref), referência fixa (1.1, 1.1, 1.1)
e t_ref={estado['referencia_tempo_s']:.9g} s/evento, medido no baseline. Na otimização eu uso o tempo original.
Eu não comparo esse HV ao do experimento anterior de dois objetivos.

## Como eu leio as matrizes e o tempo

Eu mostro contagens e percentuais por linha; a diagonal percentual é recall da classe, não acurácia geral.
Eu incluo precisão, recall, F1 e suporte nos CSVs. Um traço identifica classe sem exemplos no teste daquele painel.
Eu mantenho o vocabulário de classes fixo do treino no macro-F1, inclusive nas tabelas por base.
Eu meço padronização + previsão, com lotes e repetições fixos, excluindo leitura de disco.
Eu registro mediana e máximo observado do tempo médio por evento em lote; isso não é a maior latência individual.
Eu não afirmo generalização para uma quarta base nem estabilidade entre seeds.
'''
    (saida/'LEIA-ME.md').write_text(texto)
    pagina = '<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>MLP conjunta</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}</style>'
    pagina += '<pre>'+html.escape(texto)+'</pre>'
    pagina += ''.join(f'<img src="{n}.png" alt="{n}">' for n in figuras)
    (saida/'index.html').write_text(pagina+'</html>')
    shutil.make_archive(str(saida), 'zip', saida)
    return diag
