# Recovery collection

## Controller history

The v1 episode at seed `2000000` is retained for diagnosis and marked `excluded_from_training`: it used a slow approach and returned to the start pose during recovery. Versions v1–v4 are diagnostic collection paths and are excluded from the planned final training set.

The v2 diagnostic controller approached at native speed and introduced the disturbance near the hole. Each probe advanced 0.5 mm over 16 physics steps; the former cuRobo minimum of 104 steps made the probe unnecessarily slow. After a measured blockage, v2 retreated by a randomized 2–5 mm and corrected lateral position and angle from the current pose. It did not reset to the start pose. The retry plan had two attempts, with the first preserving 30–60% of the remaining residual. Before contact, tactile change was computed from recent-window differences after subtracting the window's baseline mean. After blockage, the controller held for 1–2 policy steps and applied an additional 0.2 mm micro-press. Its final hole alignment used the explicit object pose instead of the original expert's lateral push-and-tilt motion.

The v4 local-pose path is also excluded: post-engagement perturbation and alignment caused substantial in-hand object displacement. The v5 `insert_hole` sliding failure is excluded from training. The reviewed v5 HDMI samples remain governed by the closed collection records and quality checks documented in the current status snapshot.

## v6 oblique-hole progress estimate

The v6 `insert_hole` controller corrects false progress caused by sliding along the tilted hole. It limits estimated insertion progress to the smaller of the displacement projected onto the hole axis and the downward depth below the entrance plane converted to axis distance by dividing by the cosine of the hole tilt:

```text
progress = min(axial_projection, entrance_plane_downward_depth / cos(tilt_angle))
```

This requires evidence of both axial motion and real depth past the entrance plane. Initial 2.5–4 mm and 4–5 mm previews often self-aligned without blocking. The current hole configuration uses 5–7 mm lateral offsets, a 2° angular disturbance, and a 15 mm maximum probe distance. Because displacements in the tube's tilt plane often self-center even at that radius, `lateral_direction: transverse` draws equally from the two local Y sides with a random +/-30° spread. This is an explicit proposal-distribution bias toward measurable blockage, recorded with each episode; HDMI retains uniform directions. All four policy modes still share the same resulting data.

The diagnostic plot distinguishes object-tip motion from movement at the grasp. A long tube pivoting a few degrees can move its tip several millimetres without equivalent translation at the fingers. We track a fixed material point at the initial gripper center, with object-local coordinates `anchor = -R0.T @ p0`, and plot its later position `R @ anchor + p` alongside tip motion and relative rotation. These are geometric diagnostics, not force estimates or an additional acceptance threshold.

## Collection target

Both tasks reached 50 successful recovery episodes with at least one measured blockage and a corresponding HDF5 file. Failed and explicitly excluded attempts remain out of the mixed manifests. The final source counts, train/validation splits, feature-cache audit, and sample video/curve index are recorded in [`../IMPLEMENTATION.md`](../IMPLEMENTATION.md) and [`../reports/recovery_preview/index.json`](../reports/recovery_preview/index.json). The original AnyTouch2 cache is tracked separately from the recovery count.
