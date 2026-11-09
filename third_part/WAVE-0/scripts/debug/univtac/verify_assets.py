"""Verify the initialization assets once before a formal run; no model allocation."""
import argparse
import json
from pathlib import Path
from lerobot.configs.univtac import load_config
from lerobot.utils.univtac_runtime import atomic_json, digest, project_root

p=argparse.ArgumentParser()
p.add_argument("--config",required=True)
p.add_argument("--workspace")
p.add_argument("--output")
args,_=p.parse_known_args()
run=load_config(args.config,args.workspace,args.output)
report={"task":run["task"],"assets":{}}
for key,receipt,field in (("base","conversion.json","checkpoint_sha256"),
                          ("world_init","DOWNLOAD_SOURCE.json","output_sha256")):
    if not run.get(key):
        continue
    folder=Path(run[key])
    actual=digest(folder/"model.safetensors")
    source=json.loads((folder/receipt).read_text())
    if actual!=source[field]:
        raise ValueError(f"Initialization hash mismatch: {key}")
    report["assets"][key]={"path":str(folder),"sha256":actual,"verified":True}
report["assets"]["tokenizer"]={"sha256":digest(run["tokenizer"])}
atomic_json(project_root()/"reports"/f"assets-{run['task']}.json",report)
print(json.dumps(report,indent=2))
