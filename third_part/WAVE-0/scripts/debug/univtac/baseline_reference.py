"""Run with the existing OpenPI Python; export an independent preprocessing oracle."""
import json
import os
from pathlib import Path
import sys
import faulthandler
faulthandler.dump_traceback_later(60, repeat=True)
print("Loading original OpenPI reference environment", flush=True)

root = next(p for p in Path(__file__).resolve().parents if (p / 'UniVTAC').is_dir())
os.environ["TWAM_ROOT"] = str(root)
os.environ["OPENPI_DATA_HOME"] = str(root/"models/pi05_base")
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0,str(root/"pi05-univtac/variants/head-only-v1/src"))
sys.path.insert(0,str(root/"openpi/src"))
from pi05_univtac.transforms import UniVTACInputs
from pi05_univtac.images import decode_jpeg_rgb
from openpi import transforms as T
from openpi.shared import normalize
from openpi.models.tokenizer import PaligemmaTokenizer
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import augmax
print("Original imports ready", flush=True)

project=Path(__file__).resolve().parents[3]
out=project/"tests/univtac/fixtures"
out.mkdir(parents=True,exist_ok=True)
for task in ("insert_hole","insert_HDMI"):
    base=project/"assets/univtac/baseline"/task
    segments=json.loads((base/"segments.json").read_text())
    prompt=json.loads((base/"instruction.json").read_text())["seen"][0]
    index=[(s,t) for s in segments for t in range(s["start_frame"],s["stop_frame"]-1)]
    indices=[0,len(index)//2,len(index)-1]
    values=[]
    for i in indices:
        s,t=index[i]
        with h5py.File(root/s["source_path"],"r") as f:
            joints=np.asarray(f["embodiment/joint"][:,:8],np.float32)
            targets=joints[np.minimum(t+1+np.arange(50),s["stop_frame"]-1)]
            data=UniVTACInputs(camera_mode="head_only")({"state":joints[t],"actions":targets,
                "image":decode_jpeg_rgb(f["observation/head/rgb"][t]),"prompt":prompt})
        data=T.DeltaActions(T.make_bool_mask(7,-1))(data)
        data=T.Normalize(normalize.load(base),use_quantiles=True)(data)
        data=T.TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=True)(data)
        data=T.PadStatesAndActions(32)(data)
        values.append(data)
    np.savez(out/(task+".npz"),task=task,indices=indices,
             states=np.stack([x["state"] for x in values]),actions=np.stack([x["actions"] for x in values]),
             images=np.stack([x["image"]["base_0_rgb"] for x in values]),
             tokens=np.stack([x["tokenized_prompt"] for x in values]),
             masks=np.stack([x["tokenized_prompt_mask"] for x in values]))

images=np.random.default_rng(0).uniform(0,1,(4,224,224,3)).astype(np.float32)
parameters=[]
outputs=[]
geometry=augmax.Chain(augmax.RandomCrop(212,212),augmax.Resize(224,224),augmax.Rotate((-5,5)))
color=augmax.ColorJitter(brightness=.3,contrast=.4,saturation=.5)
for i,image in enumerate(images):
    gkey,ckey=jax.random.split(jax.random.PRNGKey(i+1))
    # Chain has one GeometricChain group; its outer key is split once.
    inner=jax.random.split(gkey,1)[0]
    crop,_,rotation=jax.random.split(inner,3)
    dy,dx=jax.random.uniform(crop,(2,),minval=-6.,maxval=6.)
    angle=jax.random.uniform(rotation,minval=-5*jnp.pi/180,maxval=5*jnp.pi/180)
    keys=jax.random.split(ckey,color.keys_needed)
    b=jax.random.uniform(keys[0],minval=-.3,maxval=.3)
    c=jax.random.uniform(keys[1],minval=-.4,maxval=.4)
    h=jax.random.uniform(keys[2],minval=-.1,maxval=.1)
    apply=jax.random.bernoulli(ckey,.5)
    parameters.append(np.asarray([dy,dx,angle,b,c,h,apply],np.float32))
    outputs.append(np.asarray(color(ckey,geometry(gkey,jnp.asarray(image)))))
np.savez(out/"augmentation.npz",images=images.transpose(0,3,1,2),parameters=np.stack(parameters),
         expected=np.stack(outputs).transpose(0,3,1,2))
print(json.dumps({"reference":str(out),"tasks":2,"samples_per_task":3,"augmentation_samples":4}))
faulthandler.cancel_dump_traceback_later()
