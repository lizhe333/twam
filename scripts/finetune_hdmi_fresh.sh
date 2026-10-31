#!/usr/bin/env bash
# Re-run both adaptation stages from expert TWAM raw weights, without EMA.
set -euo pipefail
HDMI_FRESH_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
HDMI_FRESH_COMMAND="${1:-train}"
if [ "$#" -gt 0 ]; then shift; fi
exec bash "$HDMI_FRESH_PROJECT/scripts/finetune_hdmi.sh" "$HDMI_FRESH_COMMAND" \
  --config "$HDMI_FRESH_PROJECT/configs/finetune_insert_HDMI_fresh_noema_4xa800.yaml" "$@"
