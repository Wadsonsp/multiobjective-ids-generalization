"""Eu otimizo três objetivos e conservo medições reais para retomada."""
import json
from pathlib import Path

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

from Modulos.dados_mlp import salvar_json
from Modulos.modelo_mlp import avaliar, medir_tempo, treinar

NOMES_OBJETIVOS = ['1 - F1_macro_validacao', 'k / d', 'segundos_por_evento_mediana']


def normalizar_hv(F, referencia_tempo):
    """Escala fixa, monotônica e limitada; nenhum tempo é cortado."""
    F = np.array(F, dtype=float, copy=True)
    if referencia_tempo <= 0:
        raise ValueError('Referência de tempo precisa ser positiva.')
    F[:, 2] = F[:, 2] / (F[:, 2] + referencia_tempo)
    return F


class AvaliadorMLP:
    def __init__(self, dados, config, pasta):
        self.dados, self.config, self.pasta = dados, config, Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.memoria = {}

    def obter(self, mascara):
        mascara = np.asarray(mascara, dtype=int)
        chave = ''.join(map(str, mascara))
        if chave in self.memoria:
            return self.memoria[chave]
        destino = self.pasta / chave
        criterios = destino / 'criterios.json'
        if criterios.exists():
            valor = json.loads(criterios.read_text())
        else:
            modelo, resumo = treinar(self.dados, mascara, self.config, destino)
            indices = np.flatnonzero(mascara)
            validacao = avaliar(modelo, self.dados, indices, 'validacao',
                                self.config['recursos']['lote_dados'])
            tempo = medir_tempo(modelo, self.dados, indices, self.config)
            k = int(mascara.sum())
            valor = dict(mascara=mascara.tolist(), k=k, treinamento=resumo,
                         validacao=validacao, tempo=tempo,
                         objetivos=[1-validacao['geral']['f1_macro'], k/self.dados.d,
                                    tempo['mediana_s']])
            salvar_json(criterios, valor)
        self.memoria[chave] = valor
        return valor


class ProblemaMLP(ElementwiseProblem):
    def __init__(self, avaliador):
        self.avaliador = avaliador
        super().__init__(n_var=avaliador.dados.d, n_obj=3, n_ieq_constr=1, xl=0., xu=1.)

    def _evaluate(self, x, out, *args, **kwargs):
        mascara = (np.asarray(x) >= .5).astype(int)
        if not mascara.any():
            out['F'], out['G'] = [1., 1., 1.], [1.]
            return
        out['F'] = self.avaliador.obter(mascara)['objetivos']
        out['G'] = [-1.]


def executar_busca(avaliador, config, pasta, ao_registrar=None):
    pasta = Path(pasta)
    final = pasta / 'pareto_validacao.json'
    if final.exists():
        return json.loads(final.read_text())
    d = avaliador.dados.d
    baseline = avaliador.obter(np.ones(d, dtype=int))
    calibracao = pasta / 'calibracao_tempo.json'
    if calibracao.exists():
        ref_tempo = json.loads(calibracao.read_text())['referencia_s']
    else:
        ref_tempo = baseline['tempo']['mediana_s']
        salvar_json(calibracao, dict(referencia_s=ref_tempo,
                    maximo_baseline_s=baseline['tempo']['maximo_observado_s'],
                    metodo='Referência fixa: mediana por evento do baseline na validação. '
                           'HV usa t/(t+referência); NSGA-II usa segundos/evento sem transformação.'))
    historico = []
    def registrar(alg):
        opt = alg.opt
        F = np.atleast_2d(opt.get('F'))
        mascaras = (np.atleast_2d(opt.get('X')) >= .5).astype(int)
        unicas = sorted(set(tuple(m) for m in mascaras), key=lambda m: (sum(m), m))
        sols = [avaliador.obter(m) for m in unicas]
        for i, s in enumerate(sols):
            s = dict(s, id=f'S{i+1}')
            sols[i] = s
        h = dict(geracao=int(alg.n_gen), avaliacoes=int(alg.evaluator.n_eval),
                 hipervolume=float(HV(ref_point=np.array([1.1]*3))(normalizar_hv(F, ref_tempo))),
                 solucoes=sols)
        historico.append(h)
        estado = dict(status='PARCIAL', geracoes_previstas=config['otimizacao']['n_gen'],
                      objetivos=NOMES_OBJETIVOS, referencia_tempo_s=ref_tempo,
                      referencia_hv=[1.1]*3, historico=historico, solucoes=sols,
                      baseline=baseline)
        salvar_json(pasta/'geracoes'/f'{int(alg.n_gen):03d}.json', h)
        salvar_json(pasta/'progresso.json', estado)
        if ao_registrar is not None:
            ao_registrar(estado)
        print(f"[NSGA-II] geração {alg.n_gen}/{config['otimizacao']['n_gen']} | HV={h['hipervolume']:.6f}", flush=True)
    rng = np.random.default_rng(config['seed'])
    populacao = rng.random((config['otimizacao']['n_pop'], d))
    populacao[0] = 1.  # Eu incluo o baseline na população inicial.
    minimize(ProblemaMLP(avaliador),
             NSGA2(pop_size=config['otimizacao']['n_pop'], sampling=populacao),
             ('n_gen', config['otimizacao']['n_gen']), seed=config['seed'],
             callback=registrar, verbose=False)
    estado = json.loads((pasta/'progresso.json').read_text())
    estado['status'] = 'BUSCA CONCLUÍDA — TESTE AINDA NÃO AVALIADO'
    salvar_json(final, estado)
    return estado


def escolher_solucoes(estado):
    """Eu fixo representantes usando só validação, antes de abrir o teste."""
    sols = estado['solucoes']
    F = np.array([s['objetivos'] for s in sols])
    normal = normalizar_hv(F, estado['referencia_tempo_s'])
    escolhas = {'maior_f1': int(np.argmin(F[:, 0])), 'mais_compacta': int(np.argmin(F[:, 1])),
                'mais_rapida': int(np.argmin(F[:, 2])),
                'compromisso': int(np.argmin(np.linalg.norm(normal, axis=1)))}
    saida = {}
    for criterio, i in escolhas.items():
        sol = sols[i]
        chave = ''.join(map(str, sol['mascara']))
        if chave not in saida:
            saida[chave] = dict(id=sol['id'], mascara=sol['mascara'], criterios=[])
        saida[chave]['criterios'].append(criterio)
    baseline = estado['baseline']
    chave = ''.join(map(str, baseline['mascara']))
    if chave not in saida:
        saida[chave] = dict(id='Baseline', mascara=baseline['mascara'], criterios=[])
    saida[chave]['criterios'].append('baseline')
    return list(saida.values())


def testar_escolhidas(avaliador, estado, pasta):
    pasta = Path(pasta)
    escolhas_path = pasta/'escolhas_antes_do_teste.json'
    escolhas = escolher_solucoes(estado)
    if escolhas_path.exists():
        if json.loads(escolhas_path.read_text()) != escolhas:
            raise ValueError('Escolhas de validação divergiram; eu não seleciono pelo teste.')
    else:
        salvar_json(escolhas_path, escolhas)
    resultados = []
    for escolha in escolhas:
        chave = ''.join(map(str, escolha['mascara']))
        arquivo = pasta/'teste'/f'{chave}.json'
        if arquivo.exists():
            resultado = json.loads(arquivo.read_text())
        else:
            modelo, _ = treinar(avaliador.dados, escolha['mascara'], avaliador.config,
                               avaliador.pasta/chave)
            medidas = avaliar(modelo, avaliador.dados, np.flatnonzero(escolha['mascara']),
                              'teste', avaliador.config['recursos']['lote_dados'])
            resultado = dict(**escolha, medidas=medidas,
                             validacao=avaliador.obter(escolha['mascara']))
            salvar_json(arquivo, resultado)
        resultados.append(resultado)
    return resultados
