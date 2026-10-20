# Local execution changes

Upstream provenance remains in `SOURCE.json`. The local WAVE-0 change is restricted to
`wave0/src/lerobot/policies/wave0/modeling_wave0.py`:

- Carry the SDPA dispatch derived from `wave0-univtac-fast` directly in this
  vendored file and route four custom VLM/Expert attention calls through it.
  No sibling checkout is imported at runtime.
- Preserve eager as the default, GQA repetition, scaling, additive block masks,
  and output layout. Reject unknown backends explicitly.
- ContactWAM sets the backend per GemmaAttention module from its saved config;
  no global Transformers monkeypatch or new model parameters.

The world model already used SDPA. No empty-camera changes are required for the
single-camera Contact observation path. No EMA or DDP changes are included.
