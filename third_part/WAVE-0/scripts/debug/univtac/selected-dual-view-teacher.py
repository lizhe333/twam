"""Resolve the completed dual-view teacher; no model loading or output mutation."""
import json
from pathlib import Path
import sys


def selected_teacher(workspace):
    stage = (Path(workspace)/'outputs/wave0-shared-8task-head-wrist-001/shared_wm/stage1').resolve()
    result = json.loads((stage/'result.json').read_text())
    if result.get('complete') is not True or result.get('formal_training') is not True:
        raise ValueError('Dual-view Stage 1 has not completed formal training')
    checkpoint = Path(result['selected_checkpoint']).resolve()
    if checkpoint.parent != stage/'checkpoints':
        raise ValueError('Selected checkpoint is outside the completed dual-view Stage 1')
    complete = json.loads((checkpoint/'complete.json').read_text())
    deployment = json.loads((checkpoint/'deployment.json').read_text())
    config = json.loads((checkpoint/'pretrained_model/config.json').read_text())
    if complete.get('stage') != 'stage1' or complete.get('identity') != result['identity']:
        raise ValueError('Selected teacher completion identity does not match Stage 1')
    if deployment.get('camera') != 'head_wrist' or config.get('world_model_stage') != 'stage1':
        raise ValueError('Expected a dual-view Stage-1 teacher')
    views = [k for k,v in config['input_features'].items()
             if v.get('type') == 'VISUAL' and '.empty_camera_' not in k]
    if views != ['observation.images.head','observation.images.wrist']:
        raise ValueError('Teacher must use ordered head and wrist real cameras')
    if config.get('world_model_future_horizons') != [9]:
        raise ValueError('Teacher future horizon changed')
    for name in ('pretrained_model/model.safetensors','wm_norm_stats.json','run_identity.json'):
        if not (checkpoint/name).is_file():
            raise FileNotFoundError(checkpoint/name)
    # The pipeline independently checks the weight digest and normalization.
    return checkpoint


if __name__ == '__main__':
    print(selected_teacher(sys.argv[1]))
