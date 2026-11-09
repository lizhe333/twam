#!/usr/bin/env bash
# Continue the completed 2k EMA for 3k additional policy updates on four A800s.
set -euo pipefail
HDMI_CONTINUE_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
HDMI_CONTINUE_COMMAND="${1:-train}"
if [ "$#" -gt 0 ]; then shift; fi
exec bash "$HDMI_CONTINUE_PROJECT/scripts/finetune_hdmi.sh" "$HDMI_CONTINUE_COMMAND" \
  --config "$HDMI_CONTINUE_PROJECT/configs/finetune_insert_HDMI_continue_4xa800.yaml" "$@"
