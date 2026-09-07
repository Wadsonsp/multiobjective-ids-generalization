"""Protege a seleção da execução e a identidade das soluções do relatório."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from relatorio_orientadores import encontrar_pareto_final, carregar_fonte


def preparar(tmp_path):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/config.yaml').write_text('classificador:\n  nome: decision_tree\notimizacao:\n  n_pop: 24\n  n_gen: 15\n  seed: 42\n')
    (tmp_path / 'Resultados/pareto').mkdir(parents=True)
    return tmp_path


def test_rejeita_pareto_de_execucao_preliminar(tmp_path):
    raiz = preparar(tmp_path)
    pasta = raiz / 'Resultados/pareto'
    cfg = dict(classificador='decision_tree', n_pop=1, n_gen=1, seed=42, dados_completos=True)
    (pasta / 'pareto_antigo.json').write_text(json.dumps({'configuracao':cfg}))
    assert encontrar_pareto_final(raiz) is None
    cfg.update(n_pop=24,n_gen=15)
    final = pasta / 'pareto_final.json'
    final.write_text(json.dumps({'configuracao':cfg}))
    assert encontrar_pareto_final(raiz) == final


def test_parcial_usa_fronteira_registrada_sem_duplicar_mascaras(tmp_path):
    raiz = preparar(tmp_path)
    pasta = raiz / 'Resultados/checkpoints'
    pasta.mkdir()
    contexto = {'atributos':['A','B'],'d':2}
    registros = [{'__contexto__':contexto},
                 {'mascara':'10','criterios':{'numero_atributos':1}},
                 {'mascara':'11','criterios':{'numero_atributos':2}}]
    (pasta / 'cache_decision_tree_full_v2_s42.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in registros)+'\n{"mascara":')
    (pasta / 'progresso_decision_tree_full_v2_s42.json').write_text(
        json.dumps({'geracao':14,'mascaras_parciais':[[1,0],[1,0]]}))
    dados = carregar_fonte(raiz=raiz)
    assert dados['geracao'] == 14
    assert len(dados['cache']) == 2
    assert len(dados['solucoes']) == 1
    assert dados['solucoes'][0]['mascara'] == [1,0]
    assert 'PARCIAL' in dados['status']


def test_fase2_salva_saida_explicita_e_proveniencia(tmp_path, bases_Xy, monkeypatch):
    import algoritmo2_avaliacao as fase2
    ordem = list(next(iter(bases_Xy.values()))[0].columns)
    pareto = tmp_path / 'pareto.json'
    pareto.write_text(json.dumps({'atributos':ordem,'mascaras':[[1]*len(ordem)]}))
    saida = tmp_path / 'metricas/avaliacao.json'
    config = {'preprocessamento':{},'avaliacao':{'seed':42,'cv_folds':2},
              'classificador':{'nome':'decision_tree'},
              'resultados':{'pasta_metricas':str(tmp_path/'metricas')}}
    monkeypatch.setattr(fase2,'carregar_config',lambda _:config)
    monkeypatch.setattr(fase2,'carregar_todas_as_bases',lambda _:bases_Xy)
    monkeypatch.setattr(fase2,'preprocessar_base',lambda df, _, **kwargs:df)
    monkeypatch.setattr(sys,'argv',['algoritmo2_avaliacao.py','--pareto',str(pareto),
                                    '--saida',str(saida)])
    fase2.main()
    dados=json.loads(saida.read_text())
    assert dados['pareto_origem']==str(pareto)
    assert dados['configuracao']['cv_folds']==2
    assert len(dados['avaliacoes'])==1
    assert dados['avaliacoes'][0]['numero_atributos']==len(ordem)
    assert not Path(str(saida)+'.tmp').exists()


def test_reconstrucao_reproduz_fronteiras_sem_treinar(bases_Xy, tmp_path, monkeypatch):
    import numpy as np
    import Modulos.otimizacao as otm
    from Modulos.checkpoint import CacheAvaliacoes
    from diagnostico_nsga2 import reconstruir
    cache=CacheAvaliacoes(str(tmp_path/'cache.jsonl'),{})
    fronteiras=[]
    resultado=otm.executar_otimizacao(bases_Xy,'decision_tree',n_pop=6,n_gen=3,
        seed=42,verbose=False,cache=cache,
        registro_progresso=lambda a:fronteiras.append(a.opt.get('F').copy()))
    def proibido(*args,**kwargs):
        raise AssertionError('Replay não pode treinar modelos.')
    monkeypatch.setattr(otm,'avaliar_fase1_cross_dataset',proibido)
    dados={'configuracao':{'otimizacao':{'n_pop':6,'seed':42}},
           'contexto':{'d':8,'bases':list(bases_Xy),'classificador':'decision_tree'},
           'geracao':3,'cache':cache.memoria,
           'estado':{'mascaras':resultado['mascaras'].tolist(),'objetivos':resultado['objetivos'].tolist()}}
    diag=reconstruir(dados)
    assert diag['fronteira_validada']
    for h,F in zip(diag['historico'],fronteiras):
        np.testing.assert_allclose(h['objetivos'],F,rtol=0,atol=1e-12)
    assert len(diag['historico'])==3
    dados['estado']['objetivos'][0][0]+=.1
    import pytest
    with pytest.raises(ValueError,match='diverge'):
        reconstruir(dados)


def test_cache_incompleto_interrompe_reconstrucao():
    from diagnostico_nsga2 import CacheSomenteLeitura
    import pytest
    with pytest.raises(ValueError,match='ausente no cache'):
        CacheSomenteLeitura({}).obter([1,0])
