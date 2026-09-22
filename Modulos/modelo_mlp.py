"""Eu treino a MLP conjunta, avalio classes e meço tempo por evento em lote."""
import copy
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from Modulos.dados_mlp import salvar_json


def salvar_modelo(caminho, valor):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temp = caminho.with_suffix('.tmp')
    joblib.dump(valor, temp)
    temp.replace(caminho)


def metricas_matriz(cm, classes):
    cm = np.asarray(cm, dtype=np.int64)
    suporte = cm.sum(axis=1)
    previstos = cm.sum(axis=0)
    acertos = np.diag(cm)
    recall = np.divide(acertos, suporte, out=np.zeros(len(classes)), where=suporte > 0)
    precisao = np.divide(acertos, previstos, out=np.zeros(len(classes)), where=previstos > 0)
    f1 = np.divide(2*acertos, suporte+previstos, out=np.zeros(len(classes)),
                   where=(suporte+previstos) > 0)
    return dict(f1_macro=float(f1.mean()), acuracia=float(acertos.sum()/max(1, cm.sum())),
                matriz_confusao=cm.tolist(), classes=classes,
                por_classe=[dict(classe=c, suporte=int(n), precisao=float(p),
                                recall=float(r), f1=float(f))
                            for c, n, p, r, f in zip(classes, suporte, precisao, recall, f1)],
                convencao='Macro sobre o vocabulário fixo do treino; classe sem suporte/previsão recebe zero.')


def avaliar(modelo, dados, indices, parte, lote):
    ar = dados.abrir(parte)
    classes = dados.meta['classes']
    n_classes = len(classes)
    matrizes = np.zeros((len(dados.meta['bases']), n_classes, n_classes), dtype=np.int64)
    for inicio in range(0, len(ar['y']), lote):
        sl = slice(inicio, inicio+lote)
        pred = modelo.predict(dados.transformar(ar['X'][sl], indices))
        for i in range(len(matrizes)):
            sel = ar['origem'][sl] == i
            if sel.any():
                matrizes[i] += confusion_matrix(ar['y'][sl][sel], pred[sel], labels=np.arange(n_classes))
    return dict(geral=metricas_matriz(matrizes.sum(axis=0), classes),
                por_base={nome: metricas_matriz(cm, classes)
                          for nome, cm in zip(dados.meta['bases'], matrizes)})


def treinar(dados, mascara, config, pasta):
    """Uma época percorre TODO o treino. Retomada conserva modelo e Adam."""
    indices = np.flatnonzero(mascara)
    if not len(indices):
        raise ValueError('Máscara vazia.')
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    final, parcial = pasta/'modelo.joblib', pasta/'treino_em_andamento.joblib'
    if final.exists():
        estado = joblib.load(final)
        return estado['melhor_modelo'], estado['resumo']
    cfg = config['mlp']
    treino = dados.abrir('treino')
    n = len(treino['y'])
    classes = np.arange(len(dados.meta['classes']))
    contagens = np.bincount(treino['y'], minlength=len(classes))
    pesos = n / (len(classes) * contagens)
    if parcial.exists():
        estado = joblib.load(parcial)
    else:
        modelo = MLPClassifier(hidden_layer_sizes=(2*len(indices),)*3,
                               activation=cfg['ativacao'], solver='adam',
                               learning_rate_init=cfg['taxa_aprendizado'], alpha=cfg['alpha'],
                               batch_size=cfg['batch_size'], random_state=config['seed'],
                               shuffle=False, early_stopping=False, max_iter=1)
        estado = dict(modelo=modelo, melhor_modelo=None, melhor_f1=-1.,
                      epoca=0, sem_melhora=0, historico=[], segundos_treino=0.)
    while estado['epoca'] < cfg['epocas_maximas'] and estado['sem_melhora'] < cfg['paciencia']:
        inicio = time.perf_counter()
        epoca = estado['epoca'] + 1
        rng = np.random.default_rng(config['seed'] + epoca)
        ordem = rng.permutation(n)
        modelo = estado['modelo']
        for pos in range(0, n, config['recursos']['lote_dados']):
            sel = ordem[pos:pos+config['recursos']['lote_dados']]
            X = dados.transformar(treino['X'][sel], indices)
            y = treino['y'][sel]
            modelo.partial_fit(X, y, classes=classes,
                              sample_weight=pesos[y] if cfg['pesos_balanceados'] else None)
        medidas = avaliar(modelo, dados, indices, 'validacao', config['recursos']['lote_dados'])
        f1 = medidas['geral']['f1_macro']
        estado['segundos_treino'] += time.perf_counter()-inicio
        estado['historico'].append(dict(epoca=epoca, f1_validacao=f1, loss=float(modelo.loss_)))
        if f1 > estado['melhor_f1'] + cfg['min_delta']:
            estado.update(melhor_modelo=copy.deepcopy(modelo), melhor_f1=f1,
                          melhor_epoca=epoca, sem_melhora=0)
        else:
            estado['sem_melhora'] += 1
        estado['epoca'] = epoca
        salvar_modelo(parcial, estado)
        salvar_json(pasta/'historico_treino.json', estado['historico'])
        print(f'[MLP k={len(indices)}] época {epoca}: F1 validação={f1:.6f}', flush=True)
    resumo = dict(camadas=[2*len(indices)]*3, epocas_executadas=estado['epoca'],
                  melhor_epoca=estado['melhor_epoca'], segundos_treino=estado['segundos_treino'],
                  criterio_parada='paciência na validação ou limite de épocas',
                  historico=estado['historico'])
    salvar_modelo(final, dict(melhor_modelo=estado['melhor_modelo'], resumo=resumo))
    parcial.unlink(missing_ok=True)
    return estado['melhor_modelo'], resumo


def medir_tempo(modelo, dados, indices, config):
    """Mede transformação + predict, sem E/S, em lotes fixos da validação.

    Os índices fixos são comuns a todas as máscaras. Máximo observado é de
    médias por evento em lote, não máximo de latências individuais.
    """
    cfg = config['tempo']
    ar = dados.abrir('validacao')
    rng = np.random.default_rng(config['seed'])
    n = min(cfg['eventos'], len(ar['y']))
    sel = rng.choice(len(ar['y']), size=n, replace=False)
    X = np.array(ar['X'][sel], copy=True)
    def prever():
        for inicio in range(0, n, cfg['lote']):
            modelo.predict(dados.transformar(X[inicio:inicio+cfg['lote']], indices))
    for _ in range(cfg['aquecimentos']):
        prever()
    totais = []
    for _ in range(cfg['repeticoes']):
        inicio = time.perf_counter_ns()
        prever()
        totais.append((time.perf_counter_ns()-inicio)/1e9)
    por_evento = np.array(totais)/n
    return dict(eventos=n, lote=cfg['lote'], repeticoes=cfg['repeticoes'],
                segundos_totais=totais, segundos_por_evento=por_evento.tolist(),
                mediana_s=float(np.median(por_evento)), maximo_observado_s=float(por_evento.max()),
                minimo_s=float(por_evento.min()), desvio_s=float(por_evento.std()),
                escopo='padronização dos atributos selecionados + predict; sem leitura de disco')
