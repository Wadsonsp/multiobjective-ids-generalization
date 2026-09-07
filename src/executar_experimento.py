"""Eu executo um protocolo identificado até a avaliação e o relatório finais."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Modulos.experimento import RAIZ, ler_config, caminho_local, identidade
from relatorio_orientadores import encontrar_pareto_final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config_path=Path(args.config).resolve()
    config=ler_config(config_path)
    cp=caminho_local(config['resultados']['pasta_checkpoints']);cp.mkdir(parents=True,exist_ok=True)
    lock=(cp/'execucao.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit(75)
    config_id=identidade(config)
    manifesto=cp/'experimento.json'
    if manifesto.exists() and json.loads(manifesto.read_text())['experimento_id']!=config_id:
        raise ValueError('Pasta pertence a outro protocolo; eu exijo uma nova pasta de resultados.')
    manifesto.write_text(json.dumps({'experimento_id':config_id,'configuracao':config},indent=2))
    def executar(script,*opcoes):
        print('[etapa]',script,*map(str,opcoes),flush=True)
        subprocess.run([sys.executable,'-u',str(RAIZ/'src'/script),'--config',str(config_path),*map(str,opcoes)],cwd=RAIZ,check=True)
    def marcar(p,valor):
        temp=p.with_suffix('.tmp');temp.write_text(valor);temp.replace(p)
    concluida=cp/'analise.concluida'
    if concluida.exists():
        print('[fim] Eu já concluí este protocolo.',flush=True);return
    fase1=cp/'otimizacao.concluida'
    if not fase1.exists():
        executar('algoritmo1_otimizacao.py')
        pareto=encontrar_pareto_final(config_path=config_path)
        if pareto is None:raise RuntimeError('Eu não encontrei o Pareto da execução.')
        marcar(fase1,str(pareto))
    pareto=Path(fase1.read_text().strip())
    dados=json.loads(pareto.read_text())
    if dados['configuracao'].get('experimento_id')!=config_id:raise ValueError('Pareto incompatível.')
    metricas=caminho_local(config['resultados']['pasta_metricas'])/pareto.stem
    metricas.mkdir(parents=True,exist_ok=True)
    figuras=caminho_local(config['resultados']['pasta_figuras'])
    def relatorio():
        executar('relatorio_orientadores.py','--pareto',pareto,'--metricas',metricas,'--saida',figuras)
    relatorio()
    tarefas=[(f'avaliacao_solucao_{i:02d}.json',['--pareto',pareto,'--solucao',i],m) for i,m in enumerate(dados['mascaras'])]
    tarefas.append(('avaliacao_baseline.json',['--mascara','cheia'],[1]*len(dados['atributos'])))
    for nome,opcoes,mascara in tarefas:
        saida=metricas/nome
        if saida.exists():
            anterior=json.loads(saida.read_text())
            if anterior['configuracao'].get('experimento_id')!=config_id or anterior['avaliacoes'][0]['mascara']!=mascara:
                raise ValueError('Avaliação incompatível: '+str(saida))
            print('[retomada]',saida,flush=True)
        else:executar('algoritmo2_avaliacao.py',*opcoes,'--saida',saida)
        relatorio()
    marcar(concluida,time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    print('[fim] Eu concluí a otimização, todas as avaliações, o baseline e os gráficos.',flush=True)

if __name__=='__main__':main()
