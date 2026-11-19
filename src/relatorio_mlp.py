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


# Eu traduzo os critérios de escolha gravados em escolhas_antes_do_teste.json para o rótulo da figura.
CRITERIOS = dict(maior_f1='maior F1', compromisso='compromisso', mais_rapida='mais rápida',
                 mais_compacta='mais compacta', baseline='baseline')


def rotular(ax, xs, ys, textos, negritos, fontsize=8):
    """Eu escrevo um rótulo por ponto, testando posições até não cobrir outro rótulo nem outro marcador."""
    ax.figure.canvas.draw()
    escala = 72/ax.figure.dpi  # eu trabalho em pontos, a mesma unidade dos deslocamentos dos rótulos
    pontos = [tuple(ax.transData.transform((x, y))*escala) for x, y in zip(xs, ys)]
    raio = 11  # eu reservo o anel de destaque (s=260 ≈ 8 pt de raio) em torno de cada marcador
    caixas = []
    candidatos = [(6, 6), (6, -13), (-8, 6), (-8, -13), (6, 18), (-8, 18), (6, -25), (-8, -25),
                  (16, 2), (-18, 2), (6, 30), (-8, 30), (6, -37), (-8, -37), (16, 14), (-18, 14), (16, -22), (-18, -22)]
    # Em regiões cheias eu aceito posições mais distantes, ligadas ao ponto por uma linha-guia.
    candidatos += [(round(d*np.cos(a)), round(d*np.sin(a)))
                   for d in (30, 44, 60, 78) for a in np.arange(0, 2*np.pi, np.pi/6)]
    renderer = ax.figure.canvas.get_renderer()
    medidas = []
    for texto, negrito in zip(textos, negritos):
        # Eu meço o texto com o renderizador real, em pontos, em vez de estimar pela contagem de caracteres.
        t = ax.text(0, 0, texto, fontsize=fontsize, fontweight='bold' if negrito else 'normal', linespacing=1.15)
        bb = t.get_window_extent(renderer)
        medidas.append((bb.width*escala, bb.height*escala))
        t.remove()
    def caixa(i, dx, dy):
        largura, altura = medidas[i]
        px, py = pontos[i]
        x0 = px+dx if dx >= 0 else px+dx-largura
        # Eu ancoro o rótulo pela base quando ele fica acima do ponto e pelo topo quando fica abaixo.
        y0 = py+dy-1 if dy >= 0 else py+dy-altura
        return (x0, y0, x0+largura, y0+altura)
    def guia_cruza_marcador(i, c):
        # Eu rejeito a posição se o segmento entre o ponto e o rótulo passar por cima de outro marcador.
        (ax0, ay0), (bx, by) = pontos[i], ((c[0]+c[2])/2, (c[1]+c[3])/2)
        for j, (px, py) in enumerate(pontos):
            if j == i:
                continue
            t = max(0., min(1., ((px-ax0)*(bx-ax0)+(py-ay0)*(by-ay0)) / max(1e-9, (bx-ax0)**2+(by-ay0)**2)))
            if ((ax0+t*(bx-ax0)-px)**2 + (ay0+t*(by-ay0)-py)**2) ** .5 < raio:
                return True
        return False
    def cruza(a, b):
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
    limites = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted()).bounds
    limites = [limites[0]*72, limites[1]*72, (limites[0]+limites[2])*72, (limites[1]+limites[3])*72]
    def fora_do_painel(c):
        return c[0] < limites[0] or c[2] > limites[2] or c[1] < limites[1] or c[3] > limites[3]
    def cobre_marcador(c, i):
        return fora_do_painel(c) or any(j != i and c[0]-raio < px < c[2]+raio and c[1]-raio < py < c[3]+raio
                                        for j, (px, py) in enumerate(pontos))
    # Eu posiciono primeiro os destaques, para que os rótulos comuns desviem deles.
    ordem = sorted(range(len(xs)), key=lambda i: (not negritos[i], xs[i], ys[i]))
    for i in ordem:
        escolha = None
        for dx, dy in candidatos:
            c = caixa(i, dx, dy)
            if not any(cruza(c, o) for o in caixas) and not cobre_marcador(c, i) and not guia_cruza_marcador(i, c):
                escolha = (dx, dy, c)
                break
        if escolha is None:
            # Eu aceito a posição com menos conflitos (rótulos e marcadores) quando nenhuma está livre.
            def conflitos(d):
                c = caixa(i, *d)
                return (sum(cruza(c, o) for o in caixas) + 2*cobre_marcador(c, i)
                        + 2*guia_cruza_marcador(i, c) + 10*fora_do_painel(c))
            dx, dy = min(candidatos, key=conflitos)
            escolha = (dx, dy, caixa(i, dx, dy))
        dx, dy, c = escolha
        caixas.append(c)
        longe = abs(dx) > 10 or abs(dy) > 20
        ax.annotate(textos[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords='offset points', fontsize=fontsize,
                    ha='left' if dx >= 0 else 'right', va='bottom' if dy >= 0 else 'top', linespacing=1.15,
                    fontweight='bold' if negritos[i] else 'normal',
                    arrowprops=dict(arrowstyle='-', color='#94a3b8', lw=.6, shrinkB=4) if longe else None)


def painel_pareto(fig, ax, valores, eixo_x, eixo_y, objetivo_cor, livre, sols, base,
                  destaques=None, joelho=None, lembrete_joelho=None):
    """Eu desenho uma projeção com a mesma codificação em todas as figuras.

    destaques: {id: critério} das escolhas avaliadas no teste; ganham borda grossa e rótulo em negrito.
    joelho: {id, criterio} para desenhar o anel vermelho nesta projeção, somente onde o cálculo foi feito.
    lembrete_joelho: id a marcar com anel cinza tracejado, para lembrar que ele é candidato em outra projeção.
    """
    destaques = destaques or {}
    xx, yy, cor = valores[eixo_x], valores[eixo_y], valores[objetivo_cor]
    ids = [s['id'] for s in sols]
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
    # Eu realço as escolhas levadas ao teste com a mesma borda em todos os painéis.
    em_destaque = np.array([i in destaques for i in ids])
    def anel(mascara, cor, lw, rotulo, s=260, **estilo):
        # Eu desenho o anel em tamanho real e ponho na legenda uma cópia do tamanho dos outros símbolos.
        ax.scatter(xx[mascara], yy[mascara], marker='o', s=s, facecolors='none', edgecolors=cor,
                   linewidths=lw, zorder=4, **estilo)
        ax.scatter([], [], marker='o', s=80, facecolors='none', edgecolors=cor, linewidths=lw,
                   label=rotulo, **estilo)
    if em_destaque.any():
        anel(em_destaque, '#0f172a', 1.8, 'Escolha avaliada no teste reservado (rótulo em negrito)', s=150)
    if joelho is not None:
        anel(np.array([i == joelho['id'] for i in ids]), '#dc2626', 1.8,
             f"{joelho['id']} — candidato a joelho ({joelho['criterio']})")
    if lembrete_joelho is not None and lembrete_joelho in ids:
        anel(np.array([i == lembrete_joelho for i in ids]), '#9ca3af', 1.2,
             f'{lembrete_joelho} — candidato a joelho apenas na figura 01;\naqui não há cálculo de joelho',
             linestyle='--')
    textos = list(ids)  # o critério de cada escolha vai no rodapé, para o rótulo caber nas regiões cheias
    # Eu deixo os rótulos para depois do layout final, quando as distâncias entre pontos já são as definitivas.
    ax._rotulos_pendentes = (xx, yy, textos, em_destaque)
    ax.scatter([base[eixo_x]], [base[eixo_y]], marker='X', s=100, c='#52514e', zorder=3, label='Baseline')
    if len(ind) <= 2:
        ax.text(.98, .02, f"Fronteira 2D com apenas {len(ind)} ponto(s): {', '.join(ids[i] for i in ind)}.\n"
                'As demais soluções seguem não dominadas nos três objetivos.',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
                bbox=dict(boxstyle='round', fc='white', ec='#cbd5e1'))
    fig.colorbar(sc, ax=ax, label=f'Cor = {OBJETIVOS[objetivo_cor][0]}\nmais escuro = pior')
    ax.set(xlabel=f'{OBJETIVOS[eixo_x][0]} — menor é melhor', ylabel=f'{OBJETIVOS[eixo_y][0]} — menor é melhor',
           title=f'{eixo_y.capitalize()} × {eixo_x} — cor: {objetivo_cor}')
    ax.grid(alpha=.2)
    ax.margins(x=.08, y=.1)
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


CRITERIO_JOELHO = 'maior desvio abaixo da reta entre extremos, só em erro × atributos'
NOTA_IDS = ('S1–S22 identificam as máscaras não dominadas da geração mostrada, numeradas por quantidade '
            'de atributos; os números mudam se a geração mudar.')


def diagnosticar_projecoes(valores, ids):
    """Eu registro o diagnóstico oficial (erro × atributos) e, só como exploração, o das outras projeções."""
    oficial = diagnosticar_curvatura(valores['atributos'], valores['erro'])
    oficial.update(projecao='erro × atributos', criterio=CRITERIO_JOELHO,
                   joelho_id=None if oficial['joelho_indice'] is None else ids[oficial['joelho_indice']])
    exploratorio = {}
    for nome, ex, ey in [('erro × tempo', 'tempo', 'erro'), ('tempo × atributos', 'atributos', 'tempo')]:
        d = diagnosticar_curvatura(valores[ex], valores[ey])
        exploratorio[nome] = dict(joelho_id=None if d['joelho_indice'] is None else ids[d['joelho_indice']],
                                  desvio_normalizado=d['desvio_normalizado'], conclusao=d['conclusao'])
    oficial['projecoes_exploratorias'] = dict(
        adotado=False, resultados=exploratorio,
        nota='Mesma regra geométrica aplicada às outras projeções apenas para registro. '
             'Não aparece nas figuras nem fundamenta conclusão; o candidato oficial vale só para erro × atributos.')
    return oficial


def carregar_escolhas(estado, saida):
    """Eu leio as escolhas fixadas antes do teste; sem o arquivo, uso as já testadas; senão, nada."""
    arquivo = Path(saida).parent/'escolhas_antes_do_teste.json'
    if arquivo.exists():
        escolhas = json.loads(arquivo.read_text(encoding='utf-8'))
    else:
        escolhas = estado.get('teste', [])
    return {e['id']: e['criterios'][0] for e in escolhas if e['id'] != 'Baseline' and e.get('criterios')}


def figuras_pareto(estado, diag, salvar, destaques):
    """Eu desenho as figuras 01 e 02 com a mesma codificação e deixo explícito onde há diagnóstico de joelho."""
    sols = estado['solucoes']
    ids = [s['id'] for s in sols]
    x = np.array([s['k'] for s in sols])
    erro = np.array([s['objetivos'][0] for s in sols])
    tempos = np.array([s['tempo']['mediana_s']*1e6 for s in sols])
    livre = nao_dominados([s['objetivos'] for s in sols])
    base = estado['baseline']
    base_valores = dict(atributos=base['k'], erro=base['objetivos'][0], tempo=base['tempo']['mediana_s']*1e6)
    valores = dict(atributos=x, erro=erro, tempo=tempos)
    joelho = None if diag['joelho_id'] is None else dict(id=diag['joelho_id'], criterio=CRITERIO_JOELHO)
    criterios = ', '.join(f'{i} ({CRITERIOS.get(c, c)})' for i, c in destaques.items())
    rodape = NOTA_IDS + (f'\nBorda grossa e negrito: escolhas fixadas na validação e avaliadas no teste reservado — {criterios}.'
                         if destaques else '')

    fig, ax = plt.subplots(figsize=(11, 6.8))
    ind = painel_pareto(fig, ax, valores, 'atributos', 'erro', 'tempo', livre, sols, base_valores,
                        destaques=destaques, joelho=joelho)
    if len(ind) >= 2:
        ax.plot(x[[ind[0], ind[-1]]], erro[[ind[0], ind[-1]]], '--', color='#94a3b8',
                label='Reta entre extremos — referência geométrica')
    if joelho is not None:
        j = ids.index(joelho['id'])
        ax.annotate(f"{joelho['id']}: {x[j]:.0f} atributos, F1 {1-erro[j]:.3f}, {tempos[j]:.2f} µs/evento",
                    (x[j], erro[j]), xytext=(28, 28), textcoords='offset points', fontsize=8, color='#dc2626',
                    arrowprops=dict(arrowstyle='-', color='#dc2626', lw=.8))
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_title('Figura 01 — erro × atributos (cor: tempo). '
                 'Única projeção em que calculo o candidato geométrico a joelho.', fontsize=11)
    salvar(fig, '01_pareto_erro_atributos', rodape=rodape)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))
    lembrete = None if joelho is None else joelho['id']
    painel_pareto(fig, axes[0], valores, 'tempo', 'erro', 'atributos', livre, sols, base_valores,
                  destaques=destaques, lembrete_joelho=lembrete)
    painel_pareto(fig, axes[1], valores, 'atributos', 'tempo', 'erro', livre, sols, base_valores,
                  destaques=destaques, lembrete_joelho=lembrete)
    for ax in axes:
        ax.legend(fontsize=7.2, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2)
    sem_joelho = ('Não calculo joelho nestas projeções' if joelho is None else
                  f"Não calculo joelho nestas projeções; {joelho['id']} é candidato apenas na figura 01")
    fig.suptitle('Figura 02 — projeções erro × tempo e tempo × atributos (cor: objetivo fora dos eixos). '
                 f'{sem_joelho}.', fontsize=11)
    salvar(fig, '02_pareto_tempo', rodape=rodape)
    return valores


def gerar(estado, saida, meta):
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    sols = estado['solucoes']
    ids = [s['id'] for s in sols]
    valores_iniciais = dict(atributos=np.array([s['k'] for s in sols]),
                            erro=np.array([s['objetivos'][0] for s in sols]),
                            tempo=np.array([s['tempo']['mediana_s']*1e6 for s in sols]))
    diag = diagnosticar_projecoes(valores_iniciais, ids)
    salvar_json(saida/'diagnostico_pareto.json', diag)
    destaques = carregar_escolhas(estado, saida)
    figuras = []
    titulo_estado = f"{estado['status']} | geração {estado['historico'][-1]['geracao']}/{estado['geracoes_previstas']} | seed 42"
    with PdfPages(saida/'relatorio_mlp.pdf') as pdf:
        def salvar(fig, nome, rodape=''):
            if rodape:
                fig.text(.5, .035, rodape, ha='center', va='bottom', fontsize=7.5, color='#334155')
            fig.text(.5, .012, titulo_estado, ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .085 if rodape else .045, 1, 1))
            for ax in fig.axes:
                if hasattr(ax, '_rotulos_pendentes'):
                    rotular(ax, *ax._rotulos_pendentes)
            fig.savefig(saida/f'{nome}.png', dpi=160)
            fig.savefig(saida/f'{nome}.pdf')
            pdf.savefig(fig)
            plt.close(fig)
            figuras.append(nome)

        figuras_pareto(estado, diag, salvar, destaques)

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
            ax.set(xticks=xx, xticklabels=labels, ylim=(0, 1), ylabel='F1-macro', title='F1-macro na validação e no teste reservado')
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
                    with (saida/f"classes_{sol['id']}_{seguro}.csv").open('w', newline='', encoding='utf-8') as f:
                        w = csv.DictWriter(f, fieldnames=['classe', 'suporte', 'precisao', 'recall', 'f1'])
                        w.writeheader(); w.writerows(medidas_base['por_classe'])
    with (saida/'resumo_validacao.csv').open('w', newline='', encoding='utf-8') as f:
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
{NOTA_IDS}
As escolhas fixadas na validação e avaliadas no teste reservado aparecem com borda grossa e rótulo em negrito
nos três painéis, para ligar as figuras de Pareto à figura 04.

## Onde eu calculo o joelho

Eu aplico a verificação geométrica somente à figura 01 (erro × atributos): tomo a fronteira dessa projeção,
traço a reta entre seus extremos e marco o ponto de maior desvio normalizado em direção à origem.
Minha verificação geométrica: {diag['conclusao']}
Candidato: {diag['joelho_id'] or 'nenhum'} (critério: {CRITERIO_JOELHO}).
O critério ignora o tempo; o candidato pode ser lento, e por isso escrevo seu tempo ao lado do anel vermelho.
Na figura 02 eu não calculo joelho. {diag['joelho_id'] or 'O candidato'} aparece ali com um anel cinza tracejado
só para lembrar que é candidato na figura 01; seu marcador (círculo ou losango) segue a regra de cada projeção.
Aplicar a mesma regra às outras projeções é exploratório: eu registro o resultado em diagnostico_pareto.json
com `adotado: false` e não o uso nas figuras nem como conclusão.

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
    (saida/'LEIA-ME.md').write_text(texto, encoding='utf-8')
    pagina = '<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>MLP conjunta</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}</style>'
    pagina += '<pre>'+html.escape(texto)+'</pre>'
    pagina += ''.join(f'<img src="{n}.png" alt="{n}">' for n in figuras)
    (saida/'index.html').write_text(pagina+'</html>', encoding='utf-8')
    shutil.make_archive(str(saida), 'zip', saida)
    return diag
