"""Eu reviso os eixos do Pareto anterior sem alterar resultados científicos."""
import csv
import hashlib
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Modulos.dados_mlp import salvar_json
from relatorio_mlp import diagnosticar_curvatura


def main():
    fonte = RAIZ/'Resultados/tres_datasets_v1/figuras/resumo_solucoes.csv'
    with fonte.open(encoding='utf-8-sig') as f:
        linhas = list(csv.DictReader(f))
    x = [int(r['numero_atributos']) for r in linhas]
    f1 = [float(r['f1_cross_medio']) for r in linhas]
    erro = [1-f for f in f1]
    diag = diagnosticar_curvatura(x, erro)
    saida = RAIZ/'Resultados/revisao_pareto_anterior'
    saida.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(x, f1, 'o-', color='#0369a1')
    axes[0].set(xlabel='Atributos — menor é melhor', ylabel='F1-macro cross — maior é melhor',
                title='Apresentação anterior: sentidos diferentes')
    axes[1].plot(x, erro, 'o-', color='#0369a1', label='Resultados observados')
    axes[1].plot([x[0], x[-1]], [erro[0], erro[-1]], '--', color='#94a3b8', label='Reta entre extremos')
    if diag['joelho_indice'] is not None:
        i = diag['joelho_indice']
        axes[1].scatter([x[i]], [erro[i]], s=240, facecolors='none', edgecolors='#dc2626',
                        label=f"{linhas[i]['solucao']}: candidato a joelho")
    axes[1].set(xlabel='Atributos — menor é melhor', ylabel='1 − F1-macro cross — menor é melhor',
                title='Revisão: melhoria para baixo e para a esquerda')
    for ax, valores in zip(axes, [f1, erro]):
        for r, xi, yi in zip(linhas, x, valores):
            ax.annotate(r['solucao'], (xi, yi), xytext=(5, 7), textcoords='offset points')
        ax.grid(alpha=.2)
        ax.margins(.12)
    axes[1].legend(fontsize=8)
    fig.suptitle('Revisão visual do experimento ANTERIOR — árvore, três bases, dois objetivos')
    fig.text(.5, .015, 'Mesmos resultados; eixo vertical em detalhe. Segmentos não representam soluções adicionais. Não são resultados da MLP.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for extensao in ['png', 'pdf']:
        fig.savefig(saida/f'comparacao_eixos.{extensao}', dpi=180)
    plt.close(fig)
    salvar_json(saida/'diagnostico.json', dict(fonte=str(fonte.relative_to(RAIZ)),
                sha256=hashlib.sha256(fonte.read_bytes()).hexdigest(),
                experimento='árvore anterior; não MLP', diagnostico=diag,
                pontos=[dict(id=r['solucao'], atributos=k, f1=f, erro=1-f)
                        for r, k, f in zip(linhas, x, f1)]))
    (saida/'LEIA-ME.md').write_text('''# Minha revisão dos eixos do Pareto anterior

Eu preservei os quatro resultados da árvore e troquei somente a apresentação do eixo vertical de F1 para 1 − F1.
Eu observei um candidato geométrico a joelho em S2, com oito atributos. Isso não demonstra ótimo global nem convergência.
Eu uso eixos verticais em detalhe para mostrar as diferenças; os valores absolutos de F1 continuam baixos.
Eu não alterei os pontos nem ajustei uma curva suave. A reta tracejada apenas liga os extremos.
Eu não apresento esta figura como resultado da nova MLP ou como garantia de sua futura curvatura.
''')
    print(saida)


if __name__ == '__main__':
    main()
