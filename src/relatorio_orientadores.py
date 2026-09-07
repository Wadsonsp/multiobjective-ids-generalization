"""Gera figuras e relatório rastreável do checkpoint ou do Pareto final."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import yaml

from graficos_cross import plot_matriz_confusao
from diagnostico_nsga2 import reconstruir

RAIZ = Path(__file__).resolve().parents[1]


def encontrar_pareto_final(raiz=RAIZ):
    config = yaml.safe_load((raiz / 'src/config.yaml').read_text())
    esperado = dict(classificador=config['classificador']['nome'],
                    n_pop=config['otimizacao']['n_pop'],
                    n_gen=config['otimizacao']['n_gen'],
                    seed=config['otimizacao']['seed'], dados_completos=True)
    candidatos = []
    for p in (raiz / 'Resultados/pareto').glob('pareto_*.json'):
        dados = json.loads(p.read_text())
        if all(dados.get('configuracao', {}).get(k) == v for k, v in esperado.items()):
            candidatos.append(p)
    return max(candidatos, key=lambda p: p.stat().st_mtime) if candidatos else None


def carregar_fonte(pareto=None, raiz=RAIZ):
    config = yaml.safe_load((raiz / 'src/config.yaml').read_text())
    clf, seed = config['classificador']['nome'], config['otimizacao']['seed']
    pasta = raiz / 'Resultados/checkpoints'
    cache_path = pasta / f'cache_{clf}_full_v2_s{seed}.jsonl'
    linhas = cache_path.read_text().splitlines()
    contexto = json.loads(linhas[0])['__contexto__']
    cache = {}
    for linha in linhas[1:]:
        try:
            item = json.loads(linha)
        except json.JSONDecodeError:
            continue  # A execução pode estar escrevendo a última linha.
        cache[item['mascara']] = item['criterios']
    if pareto:
        fonte = Path(pareto)
        estado = json.loads(fonte.read_text())
        if estado['atributos'] != contexto['atributos']:
            raise ValueError('Schema do Pareto diverge do cache.')
        mascaras = estado['mascaras']
        geracao = estado['configuracao']['n_gen']
        status = 'FINAL DA FASE 1'
    else:
        fonte = pasta / f'progresso_{clf}_full_v2_s{seed}.json'
        estado = json.loads(fonte.read_text())
        mascaras = estado['mascaras_parciais']
        geracao = estado['geracao']
        status = 'PARCIAL — FASE 1 EM ANDAMENTO'
    # O checkpoint pode repetir uma máscara quando genes diferentes geram a mesma seleção.
    chaves = sorted(set(''.join(map(str, m)) for m in mascaras), key=lambda m: (m.count('1'), m))
    solucoes = [dict(id=f'S{i+1}', mascara=list(map(int, m)), **cache[m])
                for i, m in enumerate(chaves)]
    return dict(status=status, geracao=geracao, geracoes_previstas=config['otimizacao']['n_gen'],
                fonte=str(fonte), contexto=contexto, configuracao=config,
                estado=estado, cache=cache, solucoes=solucoes)


def gerar(pareto=None, saida=None, metricas=None):
    if pareto is None and (RAIZ/'Resultados/checkpoints/otimizacao.concluida').exists():
        pareto = encontrar_pareto_final()
        if pareto is not None and metricas is None:
            metricas = RAIZ/'Resultados/metricas'/pareto.stem
    dados = carregar_fonte(pareto)
    diagnostico = reconstruir(dados)
    dados['diagnostico_nsga2'] = diagnostico
    saida = Path(saida or RAIZ / 'Resultados/figuras/orientadores_parcial')
    saida.mkdir(parents=True, exist_ok=True)
    (saida / 'dados_utilizados.json').write_text(json.dumps(dados, ensure_ascii=False, indent=2))
    sols = dados['solucoes']
    d = dados['contexto']['d']
    dirs = list(sols[0]['f1_macro_cross_por_direcao'])
    curto = lambda s: s.replace('NF-', '').replace('-v2', '').replace('->', ' → ')
    rotulos = [f"{s['id']} · k={s['numero_atributos']}" for s in sols]
    ks = [s['numero_atributos'] for s in sols]
    f1s = [np.mean(list(s['f1_macro_cross_por_direcao'].values())) for s in sols]
    avaliacoes=[]
    if metricas:
        for p in sorted(Path(metricas).glob('avaliacao_*.json')):
            avaliacoes.extend(json.loads(p.read_text())['avaliacoes'])
    dados['avaliacoes_detalhadas'] = avaliacoes
    (saida / 'dados_utilizados.json').write_text(json.dumps(dados, ensure_ascii=False, indent=2))
    subtitulo = (f"{dados['status']} | geração {dados['geracao']}/{dados['geracoes_previstas']}"
                 f" | Fase 2: {len(avaliacoes)}/{len(sols)+1} avaliações (inclui baseline)")
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    figuras = []
    with PdfPages(saida / 'relatorio_graficos.pdf') as pdf:
        fig = plt.figure(figsize=(11.7,8.3))
        fig.text(.08,.91,'Generalização cross-dataset em IDS',fontsize=23,weight='bold')
        fig.text(.08,.85,subtitulo,fontsize=12,color='#0369a1')
        pontos = [
            f"Fronteira registrada: {len(sols)} soluções, de {min(ks)} a {max(ks)} atributos entre {d} candidatos. F1-macro cross médio entre {min(f1s):.4f} e {max(f1s):.4f}.",
            "Método: árvore de decisão com profundidade 8 e pesos balanceados, dados completos das duas bases e transferência nas duas direções. A média das direções e a proporção de atributos são os objetivos do NSGA-II.",
            "Leitura do F1: na implementação atual, classes do destino ausentes no treino são agrupadas no código -1. Não se calcula o F1 individual de cada categoria desconhecida original.",
            "A partição do erro é por fração de amostras; não decompõe o F1 e não identifica causalmente domain shift. A coluna sem correspondência nas matrizes agrupa previsões fora do vocabulário do destino.",
            "As bases da transferência participam da seleção de atributos. Estes resultados não constituem avaliação em teste externo independente. Foi utilizada uma única seed.",
            "Os gráficos intra-dataset aparecem apenas quando houver avaliações detalhadas concluídas. O pacote parcial não contém essa etapa. Os resultados parciais podem mudar ao concluir a geração 15.",
            "Rastreabilidade: veja dados_utilizados.json, resumo_solucoes.csv e LEIA-ME.md no mesmo pacote. Fonte: " + dados['fonte'],
        ]
        y=.76
        for ponto in pontos:
            linhas=textwrap.fill(ponto,112)
            fig.text(.08,y,linhas,fontsize=11,va='top',linespacing=1.5)
            y-=.032*(linhas.count('\n')+1)+.025
        pdf.savefig(fig)
        plt.close(fig)
        def salvar(fig, nome):
            fig.text(.5, .015, subtitulo, ha='center', fontsize=8, color='#475569')
            fig.tight_layout(rect=(0,.045,1,1))
            fig.savefig(saida / f'{nome}.png', dpi=200)
            fig.savefig(saida / f'{nome}.pdf')
            pdf.savefig(fig)
            plt.close(fig)
            figuras.append(nome)

        hist = diagnostico['historico']
        fig, axes = plt.subplots(2,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        geracoes = [h['geracao'] for h in hist]
        valores = [h['hipervolume'] for h in hist]
        axes[0].plot(geracoes,valores,'o-',color='#0369a1')
        axes[0].set(ylabel='Hipervolume (maior é melhor)', title='Convergência do NSGA-II — referência fixa (1,1; 1,1)')
        axes[0].set_ylim(bottom=0)
        axes[0].grid(alpha=.25)
        axes[0].annotate(f'{valores[-1]:.6f}',(geracoes[-1],valores[-1]),xytext=(-65,12),textcoords='offset points')
        axes[1].bar(geracoes[1:],np.diff(valores),color='#d97706')
        axes[1].axhline(0,color='#475569',linewidth=.7)
        axes[1].set(xlabel='Geração',ylabel='Variação do HV',xticks=geracoes)
        axes[1].grid(axis='y',alpha=.25)
        salvar(fig,'00_convergencia_hipervolume')
        fig = plt.figure(figsize=(11.7,8.3))
        fig.text(.08,.9,'Diagnóstico da busca multiobjetivo',fontsize=22,weight='bold')
        textos = [diagnostico['conclusao'],
                  'Objetivos minimizados: f1 = 1 − F1 cross médio; f2 = k/37. Ambos já estão em [0,1]. Referência fixa r = (1.1, 1.1), pior que todos os valores possíveis. Não há normalização diferente por geração.',
                  'A curva mede o hipervolume da fronteira da população sobrevivente em cada geração, não de um arquivo acumulado de todas as soluções já visitadas. O NSGA-II não otimiza diretamente o hipervolume; ele pode oscilar.',
                  diagnostico['metodo'] + ' Nenhum modelo foi retreinado para recuperar esta curva. O cache não foi alterado.',
                  'Melhora do hipervolume mostra progresso nos objetivos definidos. Não comprova ótimo global nem melhora de generalização em teste independente. Uma única seed e 15 gerações não estabelecem robustez.',
                  'Fonte metodológica: pymoo — Analysis of Convergence: https://pymoo.org/getting_started/part_4.html',
                  'Os valores por geração e as fronteiras reconstruídas estão em convergencia_nsga2.csv e dados_utilizados.json.']
        y=.79
        for texto in textos:
            linhas=textwrap.fill(texto,110)
            fig.text(.08,y,linhas,fontsize=11,va='top',linespacing=1.5)
            y-=.032*(linhas.count('\n')+1)+.035
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10,6))
        cache = list(dados['cache'].values())
        ax.scatter([c['numero_atributos'] for c in cache],
                   [np.mean(list(c['f1_macro_cross_por_direcao'].values())) for c in cache],
                   color='#94a3b8', alpha=.45, s=25, label=f"Máscaras avaliadas no cache ({len(cache)})")
        ax.plot(ks,f1s,'o-',color='#0369a1',label='Fronteira da geração registrada')
        for s,k,f in zip(sols,ks,f1s):
            ax.annotate(s['id'],(k,f),xytext=(5,8),textcoords='offset points')
        ax.set(xlabel=f'Número de atributos selecionados (de {d})',ylabel='Média do F1-macro cross nas duas direções',
               title='Compromisso entre parcimônia e desempenho cross-dataset')
        ax.grid(alpha=.2); ax.legend(loc='lower right'); salvar(fig,'01_pareto')

        fig, ax = plt.subplots(figsize=(10,6))
        x=np.arange(len(sols))
        for j,direcao in enumerate(dirs):
            vals=[s['f1_macro_cross_por_direcao'][direcao] for s in sols]
            bars=ax.bar(x+(j-.5)*.36,vals,.36,label=curto(direcao),color=['#0369a1','#d97706'][j])
            ax.bar_label(bars,fmt='%.4f',padding=4,fontsize=9)
        ax.set(xticks=x,xticklabels=rotulos,ylabel='F1-macro',ylim=(0,max(f1s+[max(s['f1_macro_cross_por_direcao'].values()) for s in sols])*1.4),
               title='A direção da transferência altera o desempenho')
        ax.legend(); ax.grid(axis='y',alpha=.2); salvar(fig,'02_f1_por_direcao')

        fig, ax = plt.subplots(figsize=(11,6))
        desconhecidas=[]; erros=[]; labels=[]
        for s in sols:
            for direcao in dirs:
                info=s['diagnostico_cross'][direcao]
                p=info['prop_desconhecidas']
                desconhecidas.append(p); erros.append((1-p)*(1-info['acuracia_classes_conhecidas']))
                labels.append(f"{s['id']}\n{curto(direcao)}")
        x=np.arange(len(labels))
        ax.bar(x,desconhecidas,color='#7c3aed',label='Amostras de classes ausentes no treino')
        ax.bar(x,erros,bottom=desconhecidas,color='#d97706',label='Erros em classes presentes no treino')
        ax.set(xticks=x,xticklabels=labels,ylim=(0,1),ylabel='Fração das amostras do destino',
               title='Partição do erro de classificação por grupo de classes')
        ax.tick_params(axis='x',labelsize=8,rotation=25)
        ax.legend(loc='upper center', bbox_to_anchor=(.5,1.02), fontsize=9)
        ax.set_ylim(0,1.16)
        ax.set_yticks(np.arange(0,1.01,.2))
        salvar(fig,'03_erro_por_grupo')

        fig, ax = plt.subplots(figsize=(12,max(4,len(sols)*.65+2)))
        selecao=np.array([s['mascara'] for s in sols])
        indices=np.where(selecao.any(axis=0))[0]
        ax.imshow(selecao[:,indices],cmap='Blues',vmin=0,vmax=1,aspect='auto')
        ax.set_xticks(range(len(indices)),[dados['contexto']['atributos'][i] for i in indices],rotation=60,ha='right',fontsize=9)
        ax.set_yticks(range(len(sols)),rotulos)
        ax.set_title('Atributos selecionados em cada solução (azul = selecionado)')
        salvar(fig,'04_atributos')

        for direcao in dirs:
            nome='05_matriz_'+direcao.replace('->','_para_')
            caminho=saida/f'{nome}.png'
            plot_matriz_confusao(sols[0]['diagnostico_cross'],direcao,str(caminho))
            # Reabre a figura pronta para incluí-la no PDF com a identificação da solução e estágio.
            fig,ax=plt.subplots(figsize=(11,8))
            ax.imshow(plt.imread(caminho)); ax.axis('off')
            fig.suptitle(f"Solução {sols[0]['id']} — {ks[0]} atributos")
            salvar(fig,nome)

        if avaliacoes:
            fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
            nomes=list(avaliacoes[0]['f1_macro_intra'])
            labels=[]
            for a in avaliacoes:
                chave=a['mascara']
                sid=next((s['id'] for s in sols if s['mascara']==chave),'Baseline')
                labels.append(f"{sid}\nk={a['numero_atributos']}")
            for ax,base in zip(axes,nomes):
                direcao=next(di for di in dirs if di.split('->')[1]==base)
                intra=[a['f1_macro_intra'][base] for a in avaliacoes]
                desvio=[np.std(a['detalhes_intra'][base]['f1_macro_por_fold'],ddof=1) for a in avaliacoes]
                cross=[a['f1_macro_cross_por_direcao'][direcao] for a in avaliacoes]
                x=np.arange(len(avaliacoes))
                ax.bar(x-.18,intra,.36,yerr=desvio,capsize=3,label='Intra: média ± DP dos folds',color='#0369a1')
                ax.bar(x+.18,cross,.36,label='Cross: treino na outra base',color='#d97706')
                ax.set(xticks=x,xticklabels=labels,ylim=(0,1),title=curto(base),ylabel='F1-macro')
                ax.legend(fontsize=8)
            salvar(fig,'06_intra_vs_cross')

    with (saida/'convergencia_nsga2.csv').open('w',newline='',encoding='utf-8-sig') as arq:
        w=csv.DictWriter(arq,fieldnames=['geracao','avaliacoes','hipervolume','solucoes_unicas'])
        w.writeheader()
        w.writerows({k:h[k] for k in w.fieldnames} for h in diagnostico['historico'])
    campos=['solucao','numero_atributos','reducao_atributos_pct','f1_cross_medio']+dirs+['atributos']
    rows=[]
    for s,k,f in zip(sols,ks,f1s):
        rows.append(dict(solucao=s['id'],numero_atributos=k,reducao_atributos_pct=100*(1-k/d),
                         f1_cross_medio=float(f),**s['f1_macro_cross_por_direcao'],
                         atributos=', '.join(a for a,b in zip(dados['contexto']['atributos'],s['mascara']) if b)))
    with (saida/'resumo_solucoes.csv').open('w',newline='',encoding='utf-8-sig') as arq:
        w=csv.DictWriter(arq,fieldnames=campos); w.writeheader(); w.writerows(rows)
    tabela=''.join('<tr>'+''.join(f'<td>{html.escape(str(row[c]))}</td>' for c in campos[:-1])+'</tr>' for row in rows)
    notas=[
        diagnostico['conclusao'],
        'Hipervolume da fronteira sobrevivente por geração; referência fixa (1.1, 1.1), objetivos em [0,1]. Histórico reconstruído apenas pelo cache e validado contra a fronteira salva. A curva não comprova ótimo global. Método: https://pymoo.org/getting_started/part_4.html',
        'Os dados são os Parquets completos; não foi aplicada subamostragem. Classificador: árvore de decisão, profundidade 8 e pesos balanceados.',
        f"NSGA-II: população {dados['configuracao']['otimizacao']['n_pop']}, {dados['geracoes_previstas']} gerações previstas, seed 42. A fronteira mostrada pertence à geração registrada; o cache pode conter avaliações posteriores ainda sem uma geração concluída.",
        'O F1-macro cross segue a implementação atual: classes do destino ausentes no treino são agrupadas no código -1; a média usa a união dos rótulos observados e previstos. Não equivale a calcular macro-F1 individualmente para cada categoria original desconhecida.',
        'A partição de erros representa frações de amostras, não uma decomposição do F1. Erros nas classes conhecidas não comprovam isoladamente domain shift.',
        'As matrizes representam a solução mais compacta. A coluna sem correspondência agrupa previsões de classes da origem ausentes no vocabulário do destino.',
        'As bases usadas na transferência também participam da seleção das máscaras; estes valores são critérios de seleção, não estimativas em teste externo independente.',
        'Uma única seed não permite afirmar estabilidade entre execuções. O conjunto apresentado não determina uma solução vencedora sem um critério adicional.',
        ('A avaliação intra-dataset ainda não está disponível neste pacote.' if not avaliacoes else f'Avaliações detalhadas disponíveis: {len(avaliacoes)}. Barras de dispersão mostram o desvio padrão entre folds, não intervalo de confiança.')]
    texto=f"# Resultados para orientação\n\n{dados['status']} — geração {dados['geracao']}/{dados['geracoes_previstas']}.\n\nFonte: {dados['fonte']}\n\n"
    texto+='\n'.join(f'- {n}' for n in notas)+'\n'
    (saida/'LEIA-ME.md').write_text(texto)
    pagina='<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Resultados IDS</title><style>body{font:17px system-ui;max-width:1150px;margin:40px auto;color:#183047;padding:20px}img{width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ddd}aside{background:#eef5fa;padding:18px}li{margin:12px 0}</style>'
    pagina+=f'<h1>Generalização cross-dataset em IDS</h1><aside><strong>{html.escape(subtitulo)}</strong><p>Fonte: {html.escape(dados["fonte"])}</p></aside>'
    pagina+='<h2>Soluções da fronteira</h2><table><tr>'+''.join(f'<th>{html.escape(c)}</th>' for c in campos[:-1])+'</tr>'+tabela+'</table>'
    pagina+='<h2>Leitura e limites dos resultados</h2><ul>'+''.join(f'<li>{html.escape(n)}</li>' for n in notas)+'</ul>'
    pagina+=''.join(f'<img src="{nome}.png" alt="{nome}">' for nome in figuras)
    (saida/'index.html').write_text(pagina)
    shutil.make_archive(str(saida),'zip',saida)
    print(f'[relatorio] {saida} ({len(figuras)} figuras)',flush=True)
    return saida


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pareto'); p.add_argument('--saida'); p.add_argument('--metricas')
    a=p.parse_args(); gerar(a.pareto,a.saida,a.metricas)
