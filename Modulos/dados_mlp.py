"""Eu preparo treino conjunto e partições por grupos, sem ajustar nada no teste."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from Modulos.carregamento import localizar_arquivo

PARTES = ('treino', 'validacao', 'teste')


def salvar_json(caminho, valor):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_suffix(caminho.suffix + '.tmp')
    tmp.write_text(json.dumps(valor, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(caminho)


def grupos_e_particoes(X, seed, fracoes):
    """Duplicatas de X, inclusive entre bases/rótulos, ficam na mesma parte.

    O hash é aplicado aos atributos float32 comuns ANTES da seleção. Colisões
    apenas agrupam registros adicionais. A divisão não é temporal nem garante
    independência de sessões; proporções são aproximadas, sem estratificação.
    """
    grupos = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()
    # Eu misturo os bits com uma semente fixa sem depender da ordem de leitura.
    with np.errstate(over='ignore'):
        h = grupos ^ np.uint64(seed)
        h = (h ^ (h >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        h ^= h >> np.uint64(31)
    u = (h >> np.uint64(11)).astype(np.float64) / (2 ** 53)
    partes = np.searchsorted(np.cumsum(fracoes)[:2], u, side='right')
    return grupos, partes


def preparar(config, pasta, identidade):
    """Eu leio todos os Parquets em lotes e gravo matrizes mapeáveis em disco."""
    pasta = Path(pasta)
    manifesto = pasta / 'manifesto.json'
    if manifesto.exists():
        meta = json.loads(manifesto.read_text())
        if meta['experimento_id'] != identidade:
            raise ValueError('Dados preparados pertencem a outro protocolo.')
        return DadosMLP(pasta)
    fracoes = config['divisao']['fracoes']
    if len(fracoes) != 3 or min(fracoes) <= 0 or not np.isclose(sum(fracoes), 1):
        raise ValueError('Eu exijo três frações positivas que somem 1.')
    pasta.mkdir(parents=True, exist_ok=True)
    trabalho = pasta / 'preparacao.tmp'
    if trabalho.exists():
        shutil.rmtree(trabalho)
    trabalho.mkdir()
    cfg_ds, pre = config['datasets'], config['preprocessamento']
    arquivos = {nome: Path(localizar_arquivo(info['arquivo'], cfg_ds['pasta_local']))
                for nome, info in cfg_ds['bases'].items()}
    schemas = [pq.ParquetFile(p).schema_arrow for p in arquivos.values()]
    ignoradas = set(pre['atributos_vazamento'] + [pre['coluna_rotulo'], pre['coluna_binaria']])
    candidatas = [[f.name for f in schema if f.name not in ignoradas
                   and (pa.types.is_integer(f.type) or pa.types.is_floating(f.type))]
                  for schema in schemas]
    atributos = [c for c in candidatas[0] if all(c in cs for cs in candidatas)]
    if not atributos:
        raise ValueError('Não há atributos numéricos comuns.')
    classes, indice = [], {}
    nomes = list(arquivos)
    contagens = {parte: {nome: {} for nome in nomes} for parte in PARTES}
    limpeza = {}
    totais = dict.fromkeys(PARTES, 0)
    scaler = StandardScaler()
    for origem, (nome, arquivo) in enumerate(arquivos.items()):
        originais = validos = 0
        for lote in pq.ParquetFile(arquivo).iter_batches(
                batch_size=config['recursos']['lote_dados'],
                columns=atributos + [pre['coluna_rotulo']]):
            df = lote.to_pandas()
            originais += len(df)
            numericos = df[atributos].to_numpy(dtype=np.float64)
            rotulos = df[pre['coluna_rotulo']].astype('string').str.strip().str.casefold()
            ok = (np.isfinite(numericos).all(axis=1)
                  & (np.abs(numericos) <= np.finfo(np.float32).max).all(axis=1)
                  & rotulos.notna().to_numpy() & rotulos.ne('').fillna(False).to_numpy())
            X = numericos[ok].astype('float32')
            X[X == 0] = 0  # Eu unifico +0 e -0 na identificação das duplicatas.
            y_texto = rotulos[ok]
            validos += len(X)
            if not len(X):
                continue
            for classe in y_texto.unique():
                if classe not in indice:
                    indice[classe] = len(classes)
                    classes.append(str(classe))
            y = y_texto.map(indice).to_numpy(dtype='int32')
            grupos, partes = grupos_e_particoes(X, config['seed'], fracoes)
            for i, parte in enumerate(PARTES):
                selecao = partes == i
                n = int(selecao.sum())
                if not n:
                    continue
                valores = {'X': X[selecao], 'y': y[selecao],
                           'origem': np.full(n, origem, dtype='uint16'),
                           'grupo': grupos[selecao]}
                for tipo, arr in valores.items():
                    with (trabalho / f'{parte}_{tipo}.bin').open('ab') as f:
                        arr.tofile(f)
                totais[parte] += n
                for classe, quantidade in y_texto[selecao].value_counts().items():
                    dest = contagens[parte][nome]
                    dest[str(classe)] = dest.get(str(classe), 0) + int(quantidade)
                if parte == 'treino':
                    scaler.partial_fit(X[selecao])
        limpeza[nome] = dict(originais=originais, validos=validos, removidos=originais-validos)
        print(f'[preparo] {nome}: {validos}/{originais} registros válidos', flush=True)
    if min(totais.values()) == 0:
        raise ValueError('Partição vazia; eu preciso de mais grupos de registros.')
    treino_classes = set().union(*(set(v) for v in contagens['treino'].values()))
    if treino_classes != set(classes):
        raise ValueError('Classes sem exemplos de treino; revisar grupos/classes raras antes da busca: '
                         + str(set(classes) - treino_classes))
    if len(classes) < 2:
        raise ValueError('Eu preciso de pelo menos duas classes.')
    meta = dict(experimento_id=identidade, atributos=atributos, classes=classes, bases=nomes,
                totais=totais, contagens=contagens, limpeza=limpeza,
                media_treino=scaler.mean_.tolist(), escala_treino=scaler.scale_.tolist(),
                grupos='hash dos atributos float32 comuns; sem origem nem rótulo',
                fracoes_planejadas=fracoes, seed=config['seed'])
    for arquivo in trabalho.iterdir():
        arquivo.replace(pasta / arquivo.name)
    trabalho.rmdir()
    salvar_json(manifesto, meta)
    return DadosMLP(pasta)


class DadosMLP:
    """Eu abro apenas a partição pedida; teste não é acessado na busca."""
    def __init__(self, pasta):
        self.pasta = Path(pasta)
        self.meta = json.loads((self.pasta / 'manifesto.json').read_text())
        self.d = len(self.meta['atributos'])
        self.media = np.array(self.meta['media_treino'])
        self.escala = np.array(self.meta['escala_treino'])

    def abrir(self, parte):
        n = self.meta['totais'][parte]
        specs = {'X': ('float32', (n, self.d)), 'y': ('int32', (n,)),
                 'origem': ('uint16', (n,)), 'grupo': ('uint64', (n,))}
        return {tipo: np.memmap(self.pasta / f'{parte}_{tipo}.bin', dtype=tipo_dado,
                               mode='r', shape=shape)
                for tipo, (tipo_dado, shape) in specs.items()}

    def transformar(self, X, indices):
        selecionado = np.array(X[:, indices], dtype='float32', copy=True)
        selecionado -= self.media[indices]
        selecionado /= self.escala[indices]
        if not np.isfinite(selecionado).all():
            raise ValueError('Transformação produziu valores não finitos.')
        return selecionado
