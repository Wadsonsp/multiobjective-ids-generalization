"""Eu verifico as métricas derivadas e a geração dos gráficos complementares da MLP."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from Modulos.modelo_mlp import metricas_matriz
from graficos_mlp_extras import (br, classes_raras, decompor_diferenca, exemplos, f1_macro_presentes, gerar,
                                 metricas_binarias, peso, por_cento)

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


def test_raras_ordenadas_por_suporte_e_sem_classe_vazia():
    cm = np.array([[900, 0, 0], [0, 5, 0], [0, 0, 0]])
    sol = dict(validacao=dict(geral=metricas_matriz(cm, CLASSES)))
    assert classes_raras(sol, limite=1000) == [1, 0]
    assert classes_raras(sol, limite=100) == [1]


def test_decomposicao_soma_as_diferencas_totais():
    cm = np.random.default_rng(1).integers(1, 40, size=(3, 3))
    # Trocar colunas muda as previsões e preserva o suporte, como na mesma validação.
    a, b = (dict(validacao=dict(geral=metricas_matriz(m, CLASSES))) for m in (cm, cm[:, [1, 0, 2]]))
    d_f1, d_acuracia = decompor_diferenca(a, b)
    ga, gb = a['validacao']['geral'], b['validacao']['geral']
    assert d_f1.sum() == pytest.approx(ga['f1_macro']-gb['f1_macro'])
    assert d_acuracia.sum() == pytest.approx(ga['acuracia']-gb['acuracia'])


def solucao(i, k, rng):
    def validacao():
        # Todas as soluções veem a mesma validação: 150 exemplos por classe.
        cm = np.array([rng.multinomial(150, rng.dirichlet(1+5*np.eye(3)[i])) for i in range(3)])
        return metricas_matriz(cm, CLASSES)
    geral = validacao()
    return dict(id=f'S{i}', k=k, mascara=[1]*k + [0]*(4-k),
                objetivos=[1-geral['f1_macro'], k/4, 1e-6*k],
                validacao=dict(geral=geral, por_base={'A': validacao(), 'B': validacao()}),
                tempo=dict(mediana_s=1e-6*k),
                treinamento=dict(melhor_epoca=2, historico=[
                    dict(epoca=e, f1_validacao=.5+.1*e, loss=1/e) for e in (1, 2, 3)]))


def test_numeros_para_slides():
    assert br(-2.69, '+.1f') == '−2,7' and br(51.56) == '51,6'
    assert por_cento(.3) == '<1%' and por_cento(0) == '0%' and por_cento(93.4) == '93%'
    assert exemplos(1) == 'um único exemplo' and exemplos(1126) == '1.126 exemplos'
    # Uma casa decimal: as partes de cada barra de 100% somam 100.
    assert peso(100/31) == '3,2%' and peso(100*11/31) == '35,5%' and peso(2.3e-5) == '<0,1%'
    assert sum(float(peso(100*k/31)[:-1].replace(',', '.')) for k in (1, 1, 11, 18)) == pytest.approx(100)


def test_gera_oito_graficos_e_tabelas(tmp_path):
    rng = np.random.default_rng(0)
    sols = [solucao(i, k, rng) for i, k in enumerate([1, 2, 3], start=1)]
    baseline = dict(solucao(9, 4, rng), id='baseline')
    estado = dict(status='PARCIAL', geracoes_previstas=10, referencia_tempo_s=4e-6,
                  historico=[dict(geracao=3)], solucoes=sols, baseline=baseline)
    figuras = gerar(estado, tmp_path)
    assert figuras == ['g1_coordenadas_paralelas', 'g3_desempenho_por_base', 'g4_classes', 'g6_curvas_treino',
                       'g7_classes_raras', 'g8_contribuicao_classes', 'g7b_classes_raras_simples',
                       'g8b_contribuicao_simples']
    for nome in figuras:
        assert (tmp_path/f'{nome}.png').stat().st_size > 1000
    for tabela in ['metricas_solucoes.csv', 'classes_validacao.csv', 'curvas_treino.csv',
                   'classes_raras.csv', 'contribuicao_classes.csv']:
        assert len((tmp_path/tabela).read_text().splitlines()) > 1
