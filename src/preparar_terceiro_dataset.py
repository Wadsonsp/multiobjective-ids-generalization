"""Eu converto integralmente o CSV oficial em Parquet, em lotes de memória limitada."""
from pathlib import Path
import csv
import hashlib
import io
import json
import zipfile
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

RAIZ=Path(__file__).resolve().parents[1]

def main():
    origem=RAIZ/'Datasets/NF-CSE-CIC-IDS2018-v2.zip'
    destino=RAIZ/'Datasets/NF-CSE-CIC-IDS2018-V2.parquet'
    if destino.exists():
        raise FileExistsError(f'Eu preservo o arquivo existente: {destino}')
    with zipfile.ZipFile(origem) as z:
        membro=next(n for n in z.namelist() if n.endswith('/data/NF-CSE-CIC-IDS2018-v2.csv'))
        with z.open(membro) as f:
            header=next(csv.reader([f.readline().decode().strip()]))
        tipos={c:pa.string() if c in ('IPV4_SRC_ADDR','IPV4_DST_ADDR','Attack') else pa.float64() for c in header}
        tmp=destino.with_suffix('.parquet.tmp')
        total=0
        with z.open(membro) as f:
            reader=pc.open_csv(f,read_options=pc.ReadOptions(block_size=8*1024*1024),
                convert_options=pc.ConvertOptions(column_types=tipos))
            with pq.ParquetWriter(tmp,reader.schema,compression='snappy') as writer:
                for batch in reader:
                    writer.write_batch(batch)
                    total+=batch.num_rows
        assert pq.ParquetFile(tmp).metadata.num_rows==total
        with z.open(membro) as f:
            linhas=sum(1 for _ in f)-1
        assert total==linhas,(total,linhas)
        tmp.replace(destino)
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    meta={'origem':str(origem),'membro':membro,'linhas_csv':linhas,'linhas_parquet':total,
          'zip_sha256':sha(origem),'parquet_sha256':sha(destino),'numericos':'float64',
          'colunas':header,'subamostragem':False}
    destino.with_suffix('.proveniencia.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()
