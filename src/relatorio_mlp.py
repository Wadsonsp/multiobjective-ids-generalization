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
import numpy as np

from Modulos.dados_mlp import salvar_json


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

        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(x, erro, c=tempos, cmap='viridis', s=70, zorder=3)
        ind = diag['indices']
        ax.plot(x[ind], erro[ind], color='#0369a1', label='Fronteira nesta projeção (segmentos entre pontos)')
        if len(ind) >= 2:
            ax.plot(x[[ind[0], ind[-1]]], erro[[ind[0], ind[-1]]], '--', color='#94a3b8',
                    label='Reta entre extremos — referência geométrica')
        for s, xi, yi in zip(sols, x, erro):
            ax.annotate(s['id'], (xi, yi), xytext=(5, 6), textcoords='offset points')
        if diag['joelho_indice'] is not None:
            j = diag['joelho_indice']
            ax.scatter([x[j]], [erro[j]], marker='o', s=220, facecolors='none',
                       edgecolors='#dc2626', label='Candidato a joelho na projeção')
        base = estado['baseline']
        ax.scatter([base['k']], [base['objetivos'][0]], marker='X', s=100, c='#d97706', label='Baseline')
        fig.colorbar(sc, ax=ax, label='Mediana de inferência (µs/evento em lote)')
        ax.set(xlabel='Atributos selecionados — menor é melhor',
               ylabel='1 − F1-macro de validação — menor é melhor',
               title='Pareto em 2D — melhoria em direção ao canto inferior esquerdo')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
        salvar(fig, '01_pareto_erro_atributos')

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, xx, yy, xlabel, ylabel in [
            (axes[0], tempos, erro, 'µs/evento — menor é melhor', '1 − F1-macro — menor é melhor'),
            (axes[1], x, tempos, 'Atributos — menor é melhor', 'µs/evento — menor é melhor')]:
            ax.scatter(xx, yy)
            for s, xi, yi in zip(sols, xx, yy):
                ax.annotate(s['id'], (xi, yi), xytext=(4, 4), textcoords='offset points')
            ax.set(xlabel=xlabel, ylabel=ylabel); ax.grid(alpha=.2)
        fig.suptitle('Projeções 2D das soluções de três objetivos')
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
Eu uso a cor para representar o tempo. Uma solução pode parecer dominada em dois eixos e continuar não dominada nos três.
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
