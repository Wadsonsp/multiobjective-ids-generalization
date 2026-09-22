"""Eu verifico isolamento do teste, três objetivos e gráficos 2D da MLP."""
import copy
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from Modulos.dados_mlp import DadosMLP, grupos_e_particoes, preparar
from Modulos.modelo_mlp import avaliar, medir_tempo, metricas_matriz, treinar
from Modulos.busca_mlp import (AvaliadorMLP, ProblemaMLP, escolher_solucoes,
                               executar_busca, normalizar_hv, testar_escolhidas as avaliar_escolhidas)
from executar_experimento_mlp import executar, validar
from relatorio_mlp import diagnosticar_curvatura, fronteira_2d, gerar


@pytest.fixture
def config_mlp(tmp_path):
    raiz = Path(__file__).resolve().parents[1]
    c = yaml.safe_load((raiz/'src/config_mlp_conjunto.yaml').read_text())
    c['datasets'] = {'pasta_local': str(tmp_path/'originais'), 'bases': {}}
    (tmp_path/'originais').mkdir()
    rng = np.random.default_rng(42)
    copia = None
    for i, nome in enumerate(['A', 'B', 'C']):
        y = np.tile(['benign', 'dos', 'scan'], 200)
        X = rng.normal(size=(600, 4)) + np.tile(np.arange(3), 200)[:, None]
        df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
        df['Attack'] = y
        if copia is None:
            copia = df.iloc[:10].copy()
        else:
            # Mesmos atributos até entre rótulos divergentes ficam no mesmo grupo.
            df = pd.concat([df, copia.assign(Attack='dos')], ignore_index=True)
        df.loc[len(df)] = [np.inf, 0., 1., 2., 'dos']
        df.to_parquet(tmp_path/'originais'/f'{nome}.parquet')
        c['datasets']['bases'][nome] = {'arquivo': f'{nome}.parquet'}
    c['resultados']['pasta'] = str(tmp_path/'resultado')
    c['recursos'].update(lote_dados=257, threads=1)
    c['mlp'].update(epocas_maximas=2, paciencia=1, batch_size=32)
    c['otimizacao'].update(n_pop=4, n_gen=2)
    c['tempo'].update(eventos=80, lote=25, aquecimentos=1, repeticoes=3)
    return c


@pytest.fixture
def dados_mlp(config_mlp, tmp_path):
    return preparar(config_mlp, tmp_path/'dados', 'teste')


def test_grupos_scaler_e_contagens(dados_mlp, config_mlp):
    d = dados_mlp
    partes = {p: d.abrir(p) for p in ['treino', 'validacao', 'teste']}
    gs = [set(v['grupo']) for v in partes.values()]
    assert all(not gs[i] & gs[j] for i in range(3) for j in range(i))
    np.testing.assert_allclose(d.media, partes['treino']['X'].mean(axis=0, dtype=np.float64), atol=1e-12)
    np.testing.assert_allclose(d.escala, partes['treino']['X'].std(axis=0, dtype=np.float64), atol=1e-12)
    assert sum(d.meta['totais'].values()) == 1820
    assert all(v['removidos'] == 1 for v in d.meta['limpeza'].values())
    for parte, arr in partes.items():
        cont = d.meta['contagens'][parte]
        for i, base in enumerate(d.meta['bases']):
            assert sum(cont[base].values()) == (arr['origem'] == i).sum()
    novamente = preparar(config_mlp, d.pasta, 'teste')
    assert novamente.meta == d.meta
    with pytest.raises(ValueError, match='outro protocolo'):
        preparar(config_mlp, d.pasta, 'outro')
    with pytest.raises(ValueError, match='não finitos'):
        d.transformar(np.full((1, 4), np.inf), [0])


def test_divisao_independe_ordem_e_rejeita_config(config_mlp, tmp_path):
    X = np.random.default_rng(8).normal(size=(1000, 3)).astype('float32')
    g, p = grupos_e_particoes(X, 42, [.7, .15, .15])
    g2, p2 = grupos_e_particoes(X[::-1], 42, [.7, .15, .15])
    np.testing.assert_array_equal(g, g2[::-1]); np.testing.assert_array_equal(p, p2[::-1])
    c = copy.deepcopy(config_mlp); c['divisao']['fracoes'] = [.8, .15, .15]
    with pytest.raises(ValueError, match='frações'):
        preparar(c, tmp_path/'invalido', 't')
    c = copy.deepcopy(config_mlp); c['divisao']['metodo'] = 'aleatorio'
    with pytest.raises(ValueError, match='Método'): validar(c)
    c = copy.deepcopy(config_mlp); c['mlp']['camadas'] = 2
    with pytest.raises(ValueError, match='três camadas'): validar(c)
    c = copy.deepcopy(config_mlp); c['tempo']['repeticoes'] = 0
    with pytest.raises(ValueError, match='positivos'): validar(c)
    c = copy.deepcopy(config_mlp); c['otimizacao']['n_pop'] = 1
    with pytest.raises(ValueError, match='dois indivíduos'): validar(c)


def test_metrica_equivale_sklearn_e_classes_sem_suporte():
    y = [0, 0, 0, 1, 1, 2]; pred = [0, 0, 1, 0, 1, 1]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    r = metricas_matriz(cm, ['a', 'b', 'c', 'd'])
    assert r['f1_macro'] == pytest.approx(f1_score(y, pred, labels=[0, 1, 2, 3], average='macro', zero_division=0))
    assert r['acuracia'] == accuracy_score(y, pred)
    assert r['por_classe'][0]['recall'] == 2/3
    assert r['por_classe'][3]['suporte'] == 0


def test_mlp_nao_abre_teste_e_tem_tres_camadas(dados_mlp, config_mlp, tmp_path, monkeypatch):
    d = dados_mlp
    abrir = d.abrir
    chamadas = []
    def seguro(parte):
        chamadas.append(parte)
        assert parte != 'teste'
        return abrir(parte)
    monkeypatch.setattr(d, 'abrir', seguro)
    m, resumo = treinar(d, [1, 0, 1, 0], config_mlp, tmp_path/'modelo')
    assert m.hidden_layer_sizes == (4, 4, 4)
    assert m.coefs_[0].shape[0] == 2
    assert resumo['epocas_executadas'] == 2
    assert 'validacao' in chamadas
    antes = len(chamadas)
    m2, r2 = treinar(d, [1, 0, 1, 0], config_mlp, tmp_path/'modelo')
    assert r2 == resumo and len(chamadas) == antes
    tempo = medir_tempo(m2, d, [0, 2], config_mlp)
    assert tempo['eventos'] == 80
    assert tempo['mediana_s'] == np.median(tempo['segundos_por_evento'])
    assert tempo['maximo_observado_s'] == max(tempo['segundos_totais'])/80
    with pytest.raises(ValueError, match='vazia'):
        treinar(d, [0]*4, config_mlp, tmp_path/'vazia')


def test_retomada_por_epoca(dados_mlp, config_mlp, tmp_path, monkeypatch):
    import Modulos.modelo_mlp as mod
    original = mod.salvar_modelo
    def interromper(caminho, valor):
        original(caminho, valor)
        if Path(caminho).name == 'treino_em_andamento.joblib':
            raise RuntimeError('interrupção')
    monkeypatch.setattr(mod, 'salvar_modelo', interromper)
    with pytest.raises(RuntimeError, match='interrupção'):
        treinar(dados_mlp, [1]*4, config_mlp, tmp_path/'retomada')
    monkeypatch.setattr(mod, 'salvar_modelo', original)
    modelo, resumo = treinar(dados_mlp, [1]*4, config_mlp, tmp_path/'retomada')
    completo, r = treinar(dados_mlp, [1]*4, config_mlp, tmp_path/'inteiro')
    for a, b in zip(modelo.coefs_, completo.coefs_): np.testing.assert_array_equal(a, b)
    assert resumo['historico'] == r['historico']


def test_objetivos_retomada_e_teste_congelado(dados_mlp, config_mlp, tmp_path, monkeypatch):
    av = AvaliadorMLP(dados_mlp, config_mlp, tmp_path/'modelos')
    problema = ProblemaMLP(av)
    out = {}; problema._evaluate(np.zeros(4), out)
    assert out['G'][0] > 0 and not av.memoria
    problema._evaluate(np.ones(4), out)
    med = av.obter([1]*4)
    assert out['F'] == [1-med['validacao']['geral']['f1_macro'], 1., med['tempo']['mediana_s']]
    assert problema.n_obj == 3
    estado = executar_busca(av, config_mlp, tmp_path/'busca')
    assert len(estado['historico']) == 2
    assert len(estado['solucoes'][0]['objetivos']) == 3
    # Reenceno com as medições salvas: não chamo treino ou relógio novamente.
    import Modulos.busca_mlp as busca
    monkeypatch.setattr(busca, 'medir_tempo', lambda *a: pytest.fail('Medição refeita no replay'))
    novo = AvaliadorMLP(dados_mlp, config_mlp, tmp_path/'modelos')
    replay = executar_busca(novo, config_mlp, tmp_path/'replay')
    assert [h['solucoes'] for h in replay['historico']] == [h['solucoes'] for h in estado['historico']]
    assert executar_busca(novo, config_mlp, tmp_path/'replay') == replay
    escolhas = escolher_solucoes(estado)
    assert any('baseline' in e['criterios'] for e in escolhas)
    teste = avaliar_escolhidas(novo, estado, tmp_path/'busca')
    assert all(sum(s['medidas']['geral']['matriz_confusao'][i][j]
                   for i in range(3) for j in range(3)) == dados_mlp.meta['totais']['teste'] for s in teste)
    assert avaliar_escolhidas(novo, estado, tmp_path/'busca') == teste
    (tmp_path/'busca/escolhas_antes_do_teste.json').write_text('[]')
    with pytest.raises(ValueError, match='divergiram'):
        avaliar_escolhidas(novo, estado, tmp_path/'busca')
    with pytest.raises(ValueError, match='positiva'):
        normalizar_hv([[.1, .5, .01]], 0)
    norm = normalizar_hv([[.1, .5, .001], [.1, .5, 10.]], .001)
    assert norm[0, 2] == .5 and norm[0, 2] < norm[1, 2] < 1
    estado['teste'] = teste
    diag = gerar(estado, tmp_path/'figuras', dados_mlp.meta)
    assert (tmp_path/'figuras/relatorio_mlp.pdf').stat().st_size > 1000
    assert len(list((tmp_path/'figuras').glob('05_matriz*.png'))) == len(teste)*4
    assert (tmp_path/'figuras/diagnostico_pareto.json').exists()


def test_pareto_nao_impoe_barriga():
    assert fronteira_2d([2, 1, 2, 3], [.7, .9, .6, .8]) == [1, 2]
    assert diagnosticar_curvatura([1, 2], [.8, .6])['joelho_indice'] is None
    assert diagnosticar_curvatura([1, 2, 3], [.9, .3, .1])['joelho_indice'] == 1
    assert diagnosticar_curvatura([1, 2, 3], [.9, .5, .1])['joelho_indice'] is None
    assert diagnosticar_curvatura([1, 2, 3], [.9, .8, .1])['joelho_indice'] is None


def test_fluxo_cli_e_identidade(config_mlp, monkeypatch):
    import relatorio_mlp
    monkeypatch.setattr(relatorio_mlp, 'gerar', lambda *a: None)
    meta = executar(config_mlp, 'preparar')
    assert meta['totais']['teste'] > 0
    base = executar(config_mlp, 'baseline')
    assert base['k'] == 4
    estado = executar(config_mlp, 'busca')
    assert 'teste' not in estado
    final = executar(config_mlp)
    assert final['status'] == 'CONCLUÍDO'
    assert executar(config_mlp) == final
    alterada = copy.deepcopy(config_mlp); alterada['mlp']['alpha'] *= 2
    with pytest.raises(ValueError, match='outro protocolo'):
        executar(alterada)
