# Third-party sources

- `vendor/wave0/` contains the locally used WAVE-0 / LeRobot source snapshot,
  including the local attention dispatch. Its Apache-2.0 license is preserved at
  `vendor/wave0/LICENSE`. Source receipt: `vendor/SOURCE.json`; local changes:
  `vendor/LOCAL_CHANGES.md`. The original receipt identifies the WAVE-0 archive
  by SHA-256 rather than an immutable Git revision.
- Transformers is fetched from the exact commit and verified archive hash in
  `vendor/SOURCE.json`. Its source and license are materialized together by
  `scripts/prepare_transformers.py`; the generated checkout is ignored by Git.
  Stock PyPI Transformers is not an equivalent dependency for this model.
- `vendor/aligned_reference/` preserves this project's WAVE-0 comparison
  implementation, task normalization, segment boundaries, camera/action
  contracts and instructions. The reference source hashes are retained.
- AnyTouch2, UniVTAC and pretrained weights are external assets. Their source
  URLs and locally observed revisions are recorded in `docs/SOURCE_SNAPSHOT.json`.
  Users obtain and use those assets under their respective licenses/terms.

This repository does not grant a new license over third-party material or model
weights. Original copyright and license notices remain with their source files.
