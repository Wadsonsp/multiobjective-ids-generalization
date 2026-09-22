"""Eu executo preparação, busca de três objetivos e teste reservado da MLP."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))


def validar(config):
    if config['divisao']['metodo'] != 'hash_grupos_atributos':
        raise ValueError('Método de divisão não implementado.')
    if config['mlp']['camadas'] != 3 or config['mlp']['multiplicador_atributos'] != 2:
        raise ValueError('Este protocolo exige três camadas de 2*k neurônios.')
    for secao, chaves in {'mlp': ['batch_size', 'epocas_maximas', 'paciencia', 'taxa_aprendizado'],
                         'otimizacao': ['n_pop', 'n_gen'],
                         'tempo': ['eventos', 'lote', 'repeticoes', 'aquecimentos'],
                         'recursos': ['lote_dados', 'threads']}.items():
        if any(config[secao][c] <= 0 for c in chaves):
            raise ValueError('Parâmetros devem ser positivos: ' + secao)
    if config['otimizacao']['n_pop'] < 2:
        raise ValueError('População precisa de pelo menos dois indivíduos.')


def executar(config, etapa='completo'):
    import numpy as np
    import pandas as pd
    import pymoo
    import sklearn
    from packaging.version import Version
    from threadpoolctl import threadpool_limits
    from Modulos.experimento import identidade
    from Modulos.dados_mlp import preparar, salvar_json
    from Modulos.busca_mlp import AvaliadorMLP, executar_busca, testar_escolhidas
    from relatorio_mlp import gerar

    validar(config)
    if Version(sklearn.__version__) < Version('1.7'):
        raise RuntimeError('Eu exijo scikit-learn >= 1.7 para pesos na MLP incremental.')
    pasta = Path(config['resultados']['pasta'])
    if not pasta.is_absolute():
        pasta = RAIZ/pasta
    pasta.mkdir(parents=True, exist_ok=True)
    lock = (pasta/'execucao.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit(75)
    try:
        codigos = ['Modulos/dados_mlp.py', 'Modulos/modelo_mlp.py', 'Modulos/busca_mlp.py',
                   'src/executar_experimento_mlp.py']
        hashes = {p: hashlib.sha256((RAIZ/p).read_bytes()).hexdigest() for p in codigos}
        contexto = dict(dados_config=identidade(config), codigo=hashes,
                        versoes=dict(numpy=np.__version__, pandas=pd.__version__,
                                     sklearn=sklearn.__version__, pymoo=pymoo.__version__),
                        maquina=platform.node(), processador=platform.processor(),
                        plataforma=platform.platform(), threads=config['recursos']['threads'])
        config_id = hashlib.sha256(json.dumps(contexto, sort_keys=True).encode()).hexdigest()
        manifesto = pasta/'experimento.json'
        if manifesto.exists() and json.loads(manifesto.read_text())['experimento_id'] != config_id:
            raise ValueError('Pasta de outro protocolo/código/ambiente. Eu exijo uma nova pasta.')
        salvar_json(manifesto, dict(experimento_id=config_id, configuracao=config, contexto=contexto))
        with threadpool_limits(limits=config['recursos']['threads']):
            dados = preparar(config, pasta/'dados', config_id)
            if etapa == 'preparar':
                return dados.meta
            avaliador = AvaliadorMLP(dados, config, pasta/'modelos')
            if etapa == 'baseline':
                resultado = avaliador.obter(np.ones(dados.d, dtype=int))
                salvar_json(pasta/'baseline_validacao.json', resultado)
                return resultado
            estado = executar_busca(avaliador, config, pasta,
                                    ao_registrar=lambda e: gerar(e, pasta/'figuras', dados.meta))
            gerar(estado, pasta/'figuras', dados.meta)
            if etapa == 'busca':
                return estado
            testes = testar_escolhidas(avaliador, estado, pasta)
            estado = dict(estado, status='CONCLUÍDO', teste=testes)
            salvar_json(pasta/'resultado_final.json', estado)
            gerar(estado, pasta/'figuras', dados.meta)
            salvar_json(pasta/'analise.concluida.json', dict(experimento_id=config_id, status='CONCLUÍDO'))
            return estado
    finally:
        lock.close()


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(RAIZ/'src/config_mlp_conjunto.yaml'))
    parser.add_argument('--etapa', choices=['preparar', 'baseline', 'busca', 'completo'], default='completo')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    for nome in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[nome] = str(config['recursos']['threads'])
    executar(config, args.etapa)
    print('[fim] Etapa solicitada concluída.', flush=True)


if __name__ == '__main__':
    main()
