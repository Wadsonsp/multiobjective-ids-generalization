"""Eu verifico as métricas derivadas e a geração dos gráficos complementares da MLP."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from Modulos.modelo_mlp import metricas_matriz
from graficos_mlp_extras import f1_macro_presentes, gerar, metricas_binarias

CLASSES = ['benign', 'dos', 'scan']


def test_binarias_juntam_ataques():
    cm = [[6, 3, 1], [0, 4, 1], [2, 0, 3]]
    m = metricas_binarias(cm, 0)
    assert m['tnr'] == pytest.approx(.6)
    # Ataque confundido com outro ataque continua detectado.
    assert m['tpr'] == pytest.approx(8/10)


def test_f1_ignora_classe_ausente_na_base():
    cm = np.array([[5, 1, 0], [1, 3, 0], [0, 0, 0]])
    assert f1_macro_presentes(cm) == pytest.approx(metricas_matriz(cm[:2, :2], CLASSES[:2])['f1_macro'])
    assert metricas_matriz(cm, CLASSES)['f1_macro'] < f1_macro_presentes(cm)


def solucao(i, k, rng):
    def validacao():
        cm = rng.integers(0, 50, size=(3, 3)) + np.eye(3, dtype=int)*100
        return metricas_matriz(cm, CLASSES)
    geral = validacao()
    return dict(id=f'S{i}', k=k, mascara=[1]*k + [0]*(4-k),
                objetivos=[1-geral['f1_macro'], k/4, 1e-6*k],
                validacao=dict(geral=geral, por_base={'A': validacao(), 'B': validacao()}),
                tempo=dict(mediana_s=1e-6*k),
                treinamento=dict(melhor_epoca=2, historico=[
                    dict(epoca=e, f1_validacao=.5+.1*e, loss=1/e) for e in (1, 2, 3)]))


def test_gera_quatro_graficos_e_tabelas(tmp_path):
    rng = np.random.default_rng(0)
    sols = [solucao(i, k, rng) for i, k in enumerate([1, 2, 3], start=1)]
    baseline = dict(solucao(9, 4, rng), id='baseline')
    estado = dict(status='PARCIAL', geracoes_previstas=10, referencia_tempo_s=4e-6,
                  historico=[dict(geracao=3)], solucoes=sols, baseline=baseline)
    figuras = gerar(estado, tmp_path)
    assert figuras == ['g1_coordenadas_paralelas', 'g3_desempenho_por_base', 'g4_classes', 'g6_curvas_treino']
    for nome in figuras:
        assert (tmp_path/f'{nome}.png').stat().st_size > 1000
    for tabela in ['metricas_solucoes.csv', 'classes_validacao.csv', 'curvas_treino.csv']:
        assert len((tmp_path/tabela).read_text().splitlines()) > 1
