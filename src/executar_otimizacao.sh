#!/usr/bin/env bash
# Execução supervisionada da Fase 1 com retomada pelo cache existente.
set -euo pipefail
raiz_projeto="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$raiz_projeto"
mkdir -p Resultados/checkpoints Resultados/logs
# Impede duas execuções simultâneas por este script.
exec 9>Resultados/checkpoints/otimizacao.lock
flock -n 9 || { echo "Já existe uma otimização iniciada por este script." >&2; exit 75; }
conclusao=Resultados/checkpoints/otimizacao.concluida
if [[ -f "$conclusao" ]]; then
    echo "Experimento já concluído. Consulte $conclusao."
    exit 0
fi
printf '\n[inicio] %s\n' "$(date -Is)"
.venv/bin/python -u src/algoritmo1_otimizacao.py
# Só marca a conclusão depois que o Python salva os resultados e termina sem erro.
date -Is > "$conclusao"
echo "[servico] Experimento concluído com sucesso."
