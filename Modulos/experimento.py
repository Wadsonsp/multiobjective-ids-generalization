"""Eu identifico cada protocolo e preparo as bases sem conservar cópias brutas."""
import hashlib
import json
from pathlib import Path
import gc
import yaml
from Modulos.carregamento import carregar_dataset, localizar_arquivo
from Modulos.preprocessamento import preprocessar_base, alinhar_colunas

RAIZ=Path(__file__).resolve().parents[1]

def ler_config(caminho):
    return yaml.safe_load(Path(caminho).read_text())

def caminho_local(caminho):
    p=Path(caminho)
    return p if p.is_absolute() else RAIZ/p

def identidade(config):
    arquivos={}
    for nome,info in config['datasets']['bases'].items():
        p=Path(localizar_arquivo(info['arquivo'],config['datasets']['pasta_local']))
        h=hashlib.sha256()
        with p.open('rb') as f:
            for bloco in iter(lambda:f.read(8*1024*1024),b''):h.update(bloco)
        arquivos[nome]={'arquivo':p.name,'sha256':h.hexdigest()}
    conteudo={'configuracao':config,'arquivos':arquivos}
    return hashlib.sha256(json.dumps(conteudo,sort_keys=True).encode()).hexdigest()

def preparar_bases(config):
    bases={}; contagens={}
    for nome in config['datasets']['bases']:
        bruto=carregar_dataset(nome,config)
        total=len(bruto)
        X,y=preprocessar_base(bruto,config['preprocessamento'],nome_base=nome)
        if config['preprocessamento'].get('dtype_features')=='float32':
            X=X.astype('float32')
        bases[nome]=(X,y)
        contagens[nome]={'originais':total,'validos':len(y),'removidos':total-len(y)}
        print(f'[base] {nome}: {len(y)} registros válidos de {total}, {X.shape[1]} atributos',flush=True)
        del bruto,X,y
        gc.collect()
    bases,ordem=alinhar_colunas(bases)
    print(f'[schema] {len(ordem)} atributos comuns; {len(bases)*(len(bases)-1)} direções',flush=True)
    pasta=caminho_local(config['resultados']['pasta_checkpoints'])
    pasta.mkdir(parents=True,exist_ok=True)
    tmp=pasta/'dados_preprocessamento.json.tmp'
    tmp.write_text(json.dumps({'contagens':contagens,'atributos':ordem},indent=2))
    tmp.replace(pasta/'dados_preprocessamento.json')
    return bases,ordem
