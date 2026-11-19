"""Eu regero as figuras do E3 a partir do resultado_final.json, sem retreinar nada.

Uso, na raiz do repositório:
    .venv/bin/python src/regerar_figuras_mlp.py
Eu leio Resultados/mlp_conjunto_v1/resultado_final.json (estado concluído, com teste) e a lista de classes
de dados/manifesto.json (ou, na falta dele, de preparacao_resumo.json), e chamo relatorio_mlp.gerar.
"""
import argparse
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ), str(RAIZ/'src')]

from relatorio_mlp import gerar  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pasta', default='Resultados/mlp_conjunto_v1', help='pasta do experimento')
    args = parser.parse_args()
    pasta = RAIZ/args.pasta
    texto = (pasta/'resultado_final.json').read_text(encoding='utf-8')
    if texto.startswith('version https://git-lfs'):
        raise SystemExit('resultado_final.json é só um ponteiro do Git LFS; rode `git lfs pull` antes.')
    estado = json.loads(texto)
    manifesto = pasta/'dados/manifesto.json'
    meta = json.loads((manifesto if manifesto.exists() else pasta/'preparacao_resumo.json').read_text(encoding='utf-8'))
    diag = gerar(estado, pasta/'figuras', meta)
    print(f"[fim] Figuras em {pasta/'figuras'} | geração {estado['historico'][-1]['geracao']} | "
          f"candidato a joelho: {diag.get('joelho_id')}")


if __name__ == '__main__':
    main()
