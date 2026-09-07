"""Reconstrói gerações exclusivamente pelo cache e mede hipervolume em escala fixa."""
from pathlib import Path
import sys

import numpy as np
from pymoo.indicators.hv import HV
import pymoo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Modulos.otimizacao import executar_otimizacao

REFERENCIA = [1.1, 1.1]


class CacheSomenteLeitura:
    def __init__(self, cache):
        self.cache = cache

    def obter(self, mascara):
        chave = ''.join(str(int(b)) for b in mascara)
        if chave not in self.cache:
            raise ValueError(f'Reconstrução interrompida: máscara ausente no cache: {chave}')
        return self.cache[chave]


def reconstruir(dados):
    """Sem Parquets, fit, gravação no cache ou aproximação de gerações por linha."""
    cfg = dados['configuracao']['otimizacao']
    contexto = dados['contexto']
    d = contexto['d']
    bases = {nome: (np.empty((0, d)), np.empty(0)) for nome in contexto['bases']}
    hv = HV(ref_point=np.array(REFERENCIA))
    historico = []

    def registrar(alg):
        F = np.atleast_2d(alg.opt.get('F'))
        mascaras = (np.atleast_2d(alg.opt.get('X')) >= .5).astype(int)
        historico.append(dict(geracao=int(alg.n_gen), avaliacoes=int(alg.evaluator.n_eval),
                              hipervolume=float(hv(F)),
                              solucoes_unicas=len(np.unique(mascaras, axis=0)),
                              objetivos=F.tolist(), mascaras=mascaras.tolist()))

    resultado = executar_otimizacao(bases, contexto['classificador'],
                                   n_pop=cfg['n_pop'], n_gen=dados['geracao'], seed=cfg['seed'],
                                   verbose=False, cache=CacheSomenteLeitura(dados['cache']),
                                   registro_progresso=registrar)
    esperado = dados['estado']
    mascaras = esperado.get('mascaras', esperado.get('mascaras_parciais'))
    objetivos = esperado.get('objetivos', esperado.get('objetivos_parciais'))
    def mapa(ms, fs):
        return {tuple(m): np.asarray(f) for m, f in zip(ms, fs)}
    real = mapa(resultado['mascaras'], resultado['objetivos'])
    salvo = mapa(mascaras, objetivos)
    if real.keys() != salvo.keys() or any(not np.allclose(real[m], salvo[m], rtol=0, atol=1e-12) for m in real):
        raise ValueError('Histórico reconstruído diverge da fronteira salva; diagnóstico não será publicado.')
    if 'avaliacoes' in esperado and historico[-1]['avaliacoes'] != esperado['avaliacoes']:
        raise ValueError('Número de avaliações reconstruído diverge do checkpoint.')
    inicio, fim = historico[0]['hipervolume'], historico[-1]['hipervolume']
    anterior = historico[-2]['hipervolume'] if len(historico)>1 else fim
    ganho = fim-anterior
    conclusao = (f"Hipervolume: {inicio:.6f} na geração 1 e {fim:.6f} na geração {dados['geracao']}. "
                 f"Variação na última geração: {ganho:+.6f}. ")
    conclusao += ('Houve melhora na última geração; a curva não sustenta afirmar estabilização ao encerrar.'
                 if ganho>1e-8 else
                 'A última transição isolada não basta para comprovar convergência; examine a trajetória e outras seeds.')
    return dict(referencia=REFERENCIA, objetivos=['1 - F1_cross_medio','k / d'],
                escala='Objetivos originais em [0,1], sem normalização por geração.',
                metodo='Replay determinístico com cache somente leitura; fronteira final conferida por máscaras e objetivos.',
                pymoo=pymoo.__version__, seed=cfg['seed'], fronteira_validada=True,
                historico=historico, conclusao=conclusao)
