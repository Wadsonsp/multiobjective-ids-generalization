"""Espera a Fase 1, avalia cada solução com retomada e atualiza os gráficos."""
import json
from pathlib import Path
import subprocess
import sys
import time

from relatorio_orientadores import RAIZ, encontrar_pareto_final, gerar


def executar(comando):
    print('[comando]', ' '.join(map(str, comando)), flush=True)
    subprocess.run(list(map(str, comando)), cwd=RAIZ, check=True)


def main():
    marcador = RAIZ / 'Resultados/checkpoints/otimizacao.concluida'
    print('[aguardando] Conclusão da Fase 1; resultados parciais já disponíveis.', flush=True)
    while not marcador.exists():
        time.sleep(30)
    pareto = encontrar_pareto_final()
    if pareto is None:
        raise RuntimeError('Não há Pareto final compatível com a configuração atual.')
    pasta = RAIZ / 'Resultados/metricas' / pareto.stem
    pasta.mkdir(parents=True, exist_ok=True)
    saida = RAIZ / 'Resultados/figuras/orientadores_final'
    gerar(pareto, saida, pasta)
    dados = json.loads(pareto.read_text())
    # Uma saída por solução: uma queda não obriga a repetir soluções já concluídas.
    for i, mascara in enumerate(dados['mascaras']):
        caminho = pasta / f'avaliacao_solucao_{i:02d}.json'
        if caminho.exists():
            salvo = json.loads(caminho.read_text())
            if (salvo.get('pareto_origem') != str(pareto) or
                    salvo['avaliacoes'][0]['mascara'] != mascara):
                raise ValueError(f'Avaliação existente incompatível: {caminho}')
            print('[retomada]', caminho, flush=True)
        else:
            executar([sys.executable, 'src/algoritmo2_avaliacao.py', '--pareto', pareto,
                      '--solucao', i, '--saida', caminho])
        gerar(pareto, saida, pasta)
    baseline = pasta / 'avaliacao_baseline.json'
    if not baseline.exists():
        executar([sys.executable, 'src/algoritmo2_avaliacao.py', '--mascara', 'cheia',
                  '--saida', baseline])
    gerar(pareto, saida, pasta)
    (pasta / 'analise.concluida').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    print('[fim] Avaliação de todas as soluções, baseline e gráficos concluídos.', flush=True)


if __name__ == '__main__':
    main()
