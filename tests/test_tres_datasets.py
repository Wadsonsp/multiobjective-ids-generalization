"""Eu verifico o protocolo de seis direções e sua separação do experimento anterior."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from Modulos.avaliacao import avaliar_fitness
from Modulos.otimizacao import executar_otimizacao, ProblemaSelecaoCaracteristicas
from Modulos.checkpoint import CacheAvaliacoes
from Modulos.experimento import identidade
from diagnostico_nsga2 import reconstruir

@pytest.fixture
def tres(bases_Xy):
    X,y=bases_Xy['BASE_B']
    return {**bases_Xy,'BASE_C':(X.copy()+.2,y.copy())}

def test_seis_direcoes_e_tres_avaliacoes_intra(tres):
    r=avaliar_fitness([1]*8,tres,'decision_tree',cv_folds=2)
    esperadas={f'{a}->{b}' for a in tres for b in tres if a!=b}
    assert set(r['f1_macro_cross_por_direcao'])==esperadas
    assert set(r['diagnostico_cross'])==esperadas
    assert set(r['f1_macro_intra'])==set(tres)
    for direcao,diag in r['diagnostico_cross'].items():
        assert np.sum(diag['matriz_confusao'])==len(tres[direcao.split('->')[1]][1])

def test_media_das_seis_direcoes(tres,monkeypatch):
    import Modulos.otimizacao as otm
    valores=dict(zip((f'{a}->{b}' for a in tres for b in tres if a!=b),[.1,.2,.3,.4,.5,.6]))
    monkeypatch.setattr(otm,'avaliar_fase1_cross_dataset',lambda *a:{'f1_macro_cross_por_direcao':valores})
    problema=ProblemaSelecaoCaracteristicas(tres,'decision_tree')
    out={};problema._evaluate(np.array([1,1,0,0,0,0,0,0]),out)
    assert out['F']==pytest.approx([.65,.25])
    valores.pop(next(iter(valores)))
    with pytest.raises(RuntimeError,match='par ordenado'):
        problema._evaluate(np.ones(8),{})

def test_identidade_muda_com_base_e_conteudo(tmp_path):
    for nome in ('a','b','c'):(tmp_path/f'{nome}.parquet').write_bytes(nome.encode())
    c={'datasets':{'pasta_local':str(tmp_path),'bases':{n:{'arquivo':f'{n}.parquet'} for n in ('a','b')}}}
    antes=identidade(c)
    c['datasets']['bases']['c']={'arquivo':'c.parquet'}
    depois=identidade(c)
    assert antes!=depois
    (tmp_path/'c.parquet').write_bytes(b'novo')
    assert depois!=identidade(c)
    caminho=str(tmp_path/'cache.jsonl')
    CacheAvaliacoes(caminho,{'experimento_id':antes})
    with pytest.raises(ValueError,match='OUTRO contexto'):
        CacheAvaliacoes(caminho,{'experimento_id':depois})

def test_retomada_tres_bases_reconstroi_pareto(tres,tmp_path,monkeypatch):
    import Modulos.otimizacao as otm
    contexto={'bases':list(tres),'d':8,'atributos':[f'F{i}' for i in range(8)],'classificador':'decision_tree'}
    cache=CacheAvaliacoes(str(tmp_path/'cache.jsonl'),contexto)
    r=executar_otimizacao(tres,'decision_tree',n_pop=6,n_gen=3,verbose=False,cache=cache)
    dados={'contexto':contexto,'configuracao':{'otimizacao':{'n_pop':6,'seed':42}},'geracao':3,
           'cache':cache.memoria,'estado':{'mascaras':r['mascaras'].tolist(),'objetivos':r['objetivos'].tolist()}}
    def proibido(*a,**k):raise AssertionError('Eu não retreino na reconstrução.')
    monkeypatch.setattr(otm,'avaliar_fase1_cross_dataset',proibido)
    diag=reconstruir(dados)
    assert diag['fronteira_validada'] and len(diag['historico'])==3


def test_relatorio_com_seis_direcoes(tres,tmp_path,monkeypatch):
    import relatorio_orientadores as rel
    raiz=tmp_path
    for d in ('src','Resultados/checkpoints','Resultados/pareto','Resultados/metricas'):(raiz/d).mkdir(parents=True)
    cfg={'classificador':{'nome':'decision_tree'},'otimizacao':{'n_pop':6,'n_gen':2,'seed':42},
         'resultados':{'pasta_checkpoints':'Resultados/checkpoints'}}
    config=raiz/'src/config.yaml';config.write_text(yaml.safe_dump(cfg))
    atributos=list(next(iter(tres.values()))[0].columns)
    contexto={'bases':list(tres),'d':8,'atributos':atributos,'classificador':'decision_tree'}
    cache=CacheAvaliacoes(str(raiz/'Resultados/checkpoints/cache_decision_tree_full_v2_s42.jsonl'),contexto)
    r=executar_otimizacao(tres,'decision_tree',n_pop=6,n_gen=2,verbose=False,cache=cache)
    pareto=raiz/'Resultados/pareto/pareto_test.json'
    pareto.write_text(json.dumps({'atributos':atributos,'mascaras':r['mascaras'].tolist(),
        'objetivos':r['objetivos'].tolist(),'configuracao':{'n_gen':2}}))
    m=r['mascaras'][0].tolist()
    avaliacao={'mascara':m,**avaliar_fitness(m,tres,'decision_tree',cv_folds=2)}
    pasta=raiz/'Resultados/metricas'
    (pasta/'avaliacao_test.json').write_text(json.dumps({'avaliacoes':[avaliacao]}))
    monkeypatch.setattr(rel,'RAIZ',raiz)
    saida=rel.gerar(pareto,raiz/'figuras',pasta,config_path=config)
    assert len(list(saida.glob('05_matriz_*.png')))==6
    assert (saida/'06_intra_vs_cross.png').exists()
    assert (saida/'00_convergencia_hipervolume.png').exists()
    assert (saida/'relatorio_graficos.pdf').stat().st_size>1000


def test_preparacao_integral_com_tres_parquets(tmp_path,base_a,base_b,config_pre):
    from Modulos.experimento import preparar_bases,ler_config
    for nome,df in [('A',base_a),('B',base_b),('C',base_a)]:
        df=df.rename(columns=dict(zip([f'FEAT_{i}' for i in range(4)],['PROTOCOL','IN_BYTES','OUT_BYTES','FLOW_DURATION_MILLISECONDS'])))
        df.to_parquet(tmp_path/f'{nome}.parquet',index=False)
    cfg={'datasets':{'pasta_local':str(tmp_path),'bases':{n:{'arquivo':f'{n}.parquet'} for n in 'ABC'}},
         'preprocessamento':{**config_pre,'dtype_features':'float32'},
         'resultados':{'pasta_checkpoints':str(tmp_path/'checkpoints')}}
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(cfg))
    bases,ordem=preparar_bases(ler_config(p))
    assert len(bases)==3 and len(ordem)==8
    for X,y in bases.values():
        assert len(y)==600
        assert all(dt==np.dtype('float32') for dt in X.dtypes)
    registro=json.loads((tmp_path/'checkpoints/dados_preprocessamento.json').read_text())
    assert all(c['originais']==c['validos']==600 for c in registro['contagens'].values())
