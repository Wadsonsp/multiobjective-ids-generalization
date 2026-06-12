# -*- coding: utf-8 -*-
"""Setup dos datasets no Google Drive (rodar UMA VEZ no Google Colab).

Este script baixa as bases NF-UNSW-NB15-v2 e NF-ToN-IoT-v2 das fontes
oficiais da University of Queensland (rdm.uq.edu.au) direto para a pasta
MyDrive/mestrado/Datasets, usando a banda do próprio Colab.

Uso (em uma célula do Colab):
    !git clone https://github.com/Wadsonsp/multiobjective-ids-generalization.git
    %cd multiobjective-ids-generalization
    !python src/setup_datasets_drive.py

Licença das bases: uso acadêmico (Sarhan, Layeghy e Portmann, 2022 -
"Towards a Standard Feature Set for Network Intrusion Detection System
Datasets", Mobile Networks and Applications). Citar conforme indicado
na página oficial.
"""

import os
import sys
import urllib.request
import zipfile

# URLs oficiais (UQ Research Data Manager). Mantenho aqui e não no
# config.yaml de propósito: este script é o único ponto de download.
DATASETS = {
    "NF-UNSW-NB15-v2": "https://rdm.uq.edu.au/files/8c6e2a00-ef9c-11ed-827d-e762de186848",
    "NF-ToN-IoT-v2": "https://rdm.uq.edu.au/files/a4ad7080-ef9c-11ed-a964-b70596e96ad5",
}

PASTA_DESTINO_COLAB = "/content/drive/MyDrive/mestrado/Datasets"


def montar_drive():
    """Monta o Drive no Colab; fora do Colab uso a pasta Datasets/ local."""
    try:
        from google.colab import drive
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
        return PASTA_DESTINO_COLAB
    except ImportError:
        print("[aviso] Fora do Colab: salvando na pasta local Datasets/")
        return "Datasets"


def baixar(url, destino):
    """Download com barra de progresso simples (arquivos de vários GB)."""
    def progresso(blocos, tam_bloco, total):
        baixado = blocos * tam_bloco
        if total > 0:
            pct = min(100.0, baixado * 100.0 / total)
            sys.stdout.write(f"\r  {pct:5.1f}% ({baixado / 1e9:.2f} GB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, destino, reporthook=progresso)
    print()


def extrair_se_zip(caminho, pasta):
    """As bases da UQ chegam zipadas; extraio e removo o zip."""
    if zipfile.is_zipfile(caminho):
        print(f"  Extraindo {os.path.basename(caminho)}...")
        with zipfile.ZipFile(caminho) as z:
            z.extractall(pasta)
        os.remove(caminho)


def main():
    pasta = montar_drive()
    os.makedirs(pasta, exist_ok=True)

    for nome, url in DATASETS.items():
        csv_final = os.path.join(pasta, f"{nome}.csv")
        if os.path.exists(csv_final):
            # Cache: não baixo de novo o que já está no Drive
            print(f"[ok] {nome}.csv já existe em {pasta} - pulando download.")
            continue

        print(f"[download] {nome} <- {url}")
        temporario = os.path.join(pasta, f"{nome}.download")
        baixar(url, temporario)
        extrair_se_zip(temporario, pasta)

        # Se não era zip, o arquivo baixado é o próprio CSV
        if os.path.exists(temporario):
            os.rename(temporario, csv_final)

        # Normalizo o nome do CSV extraído (alguns zips trazem subpastas)
        if not os.path.exists(csv_final):
            for raiz, _, arquivos in os.walk(pasta):
                for arq in arquivos:
                    if arq.lower() == f"{nome.lower()}.csv":
                        os.rename(os.path.join(raiz, arq), csv_final)

        situacao = "ok" if os.path.exists(csv_final) else "VERIFICAR MANUALMENTE"
        print(f"[{situacao}] {csv_final}")

    print("\nSetup concluído. As bases estão em:", pasta)


if __name__ == "__main__":
    main()
