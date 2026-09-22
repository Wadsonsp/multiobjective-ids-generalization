#!/usr/bin/env bash
# Eu preparo o log antes de iniciar o Python, sem depender do journal.
set -euo pipefail
raiz_mlp="$(cd -- "$(dirname -- "$0")/.." && pwd)"
etapa_mlp="${1:-completo}"
case "$etapa_mlp" in
  baseline) nome_log_mlp=baseline.log ;;
  completo) nome_log_mlp=execucao.log ;;
  *) echo 'Etapa inválida: use baseline ou completo.' >&2; exit 2 ;;
esac
pasta_log_mlp="$raiz_mlp/Resultados/mlp_conjunto_v1/logs"
mkdir -p "$pasta_log_mlp"
cd "$raiz_mlp"
exec "$raiz_mlp/.venv/bin/python" -u "$raiz_mlp/src/executar_experimento_mlp.py" \
  --config "$raiz_mlp/src/config_mlp_conjunto.yaml" --etapa "$etapa_mlp" \
  >> "$pasta_log_mlp/$nome_log_mlp" 2>&1
