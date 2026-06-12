# -*- coding: utf-8 -*-
"""Checkpoint dos experimentos: cache de avaliações e progresso por geração.

Motivação: o Colab pode derrubar a sessão no meio de uma execução longa
do NSGA-II. Como cada avaliação de fitness é cara (CV + cross-dataset),
não posso perder o que já foi computado.

Estratégia de retomada:
1. CACHE DE AVALIAÇÕES (CacheAvaliacoes): cada critério calculado pelo
   Algoritmo 2 é gravado IMEDIATAMENTE em um arquivo .jsonl no Drive
   (append + flush, uma avaliação por linha). Como o NSGA-II com seed
   fixa é determinístico, ao reexecutar a mesma configuração a sequência
   de máscaras se repete: as avaliações já gravadas viram cache hit
   instantâneo e a execução "reencena" em segundos até o ponto da falha,
   continuando de lá. Retomar = rodar o mesmo comando de novo.
2. PROGRESSO POR GERAÇÃO (RegistroProgresso): ao fim de cada geração,
   gravo a fronteira parcial (máscaras + objetivos) em JSON no Drive.
   Mesmo que eu nunca retome, sempre existe um Pareto parcial utilizável
   do ponto exato em que a sessão caiu.

O cache só é válido para o MESMO contexto experimental (classificador,
amostra, folds, bases, seed): misturar contextos corromperia os
resultados em silêncio. Por isso o arquivo carrega um cabeçalho de
contexto e o carregamento falha alto se houver divergência.
"""

import json
import os

import numpy as np


def chave_da_mascara(mascara):
    """Chave canônica de uma máscara: string binária '0101...'."""
    return "".join(str(int(b)) for b in np.asarray(mascara).astype(int))


class CacheAvaliacoes:
    """Cache persistente (JSONL) das avaliações do Algoritmo 2.

    Formato do arquivo:
      linha 1: {"__contexto__": {...}}      <- identidade do experimento
      demais : {"mascara": "0101...", "criterios": {...}}
    """

    def __init__(self, caminho, contexto):
        self.caminho = caminho
        self.contexto = dict(contexto)
        self.memoria = {}
        os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
        if os.path.exists(caminho):
            self._carregar()
        else:
            self._iniciar_arquivo()

    def _iniciar_arquivo(self):
        with open(self.caminho, "w", encoding="utf-8") as f:
            f.write(json.dumps({"__contexto__": self.contexto},
                               ensure_ascii=False) + "\n")

    def _carregar(self):
        """Lê o cache existente, validando o contexto experimental.

        Linhas truncadas (sessão derrubada NO MEIO de uma escrita) são
        ignoradas em silêncio: a avaliação correspondente é refeita.
        """
        with open(self.caminho, "r", encoding="utf-8") as f:
            primeira = f.readline()
            try:
                meta = json.loads(primeira).get("__contexto__")
            except (json.JSONDecodeError, AttributeError):
                raise ValueError(
                    f"Arquivo de cache inválido: {self.caminho}. "
                    "Apague-o ou aponte para outro caminho."
                )
            if meta != self.contexto:
                raise ValueError(
                    "O cache em "
                    f"{self.caminho} pertence a OUTRO contexto experimental.\n"
                    f"  cache : {meta}\n  atual : {self.contexto}\n"
                    "Use um arquivo de cache por configuração (ex.: inclua "
                    "classificador e tamanho da amostra no nome do arquivo)."
                )
            for linha in f:
                try:
                    registro = json.loads(linha)
                    self.memoria[registro["mascara"]] = registro["criterios"]
                except (json.JSONDecodeError, KeyError):
                    continue  # linha truncada pela queda: será recalculada

    def obter(self, mascara):
        """Retorna os critérios já calculados para a máscara, ou None."""
        return self.memoria.get(chave_da_mascara(mascara))

    def registrar(self, mascara, criterios):
        """Grava a avaliação em memória E em disco imediatamente.

        O flush + fsync garante que a linha chegue ao Drive antes de a
        próxima avaliação começar: se a sessão cair, perde-se no máximo
        a avaliação em andamento.
        """
        chave = chave_da_mascara(mascara)
        self.memoria[chave] = criterios
        with open(self.caminho, "a", encoding="utf-8") as f:
            f.write(json.dumps({"mascara": chave, "criterios": criterios},
                               ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def __len__(self):
        return len(self.memoria)


class RegistroProgresso:
    """Grava a fronteira parcial ao fim de cada geração do NSGA-II.

    Implementa a interface de Callback do pymoo via duck typing
    (método notify), evitando herança direta para facilitar os testes.
    """

    def __init__(self, caminho):
        self.caminho = caminho
        os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)

    def __call__(self, algorithm):
        # O pymoo invoca o callback como função a cada geração
        self.notify(algorithm)

    def notify(self, algorithm):
        # opt = conjunto não-dominado corrente do algoritmo
        opt = algorithm.opt
        estado = {
            "geracao": int(algorithm.n_gen),
            "avaliacoes": int(algorithm.evaluator.n_eval),
            "mascaras_parciais": np.atleast_2d(opt.get("X")).astype(int).tolist(),
            "objetivos_parciais": np.atleast_2d(opt.get("F")).tolist(),
        }
        # Escrita atômica: gravo em arquivo temporário e renomeio, para a
        # queda nunca deixar um JSON pela metade
        temporario = self.caminho + ".tmp"
        with open(temporario, "w", encoding="utf-8") as f:
            json.dump(estado, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporario, self.caminho)
