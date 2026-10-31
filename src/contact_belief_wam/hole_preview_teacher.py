"""Privileged, bounded alignment inside the native tilted tube entrance.

The normal task rotates the tube while it is still touching the entrance.
Requiring complete separation after a 6 mm retreat rejects that geometry.
This preview variant first verifies real axial unloading, then uses 1-degree
motor/IK increments towards the true target. No object pose is ever assigned.
"""
import numpy as np

from .oracle_batch_teacher import (BatchOracleTeacher, pose_array, rotation,
                                   with_rotation, ee_for_object_pose)
from .sim_takeover import TeacherStopped


class InContactHoleOracle(BatchOracleTeacher):
    revision = "hole-bounded-contact-alignment-v1"

    def unload(self):
        row = self.runner.row()
        obj = pose_array(row["prism_pose_world"])
        axis = rotation(obj).apply([0., 0., 1.])
        desired = obj.copy()
        desired[:3] += axis * self.config["retreat_mm"] / 1000
        event = self._move(ee_for_object_pose(row["ee_pose_world"], obj, desired),
            "oracle_unload", "verify_axial_unloading_before_bounded_in_contact_alignment",
            self.config["retreat_actions"], movement_axis=axis)
        if event is not None:
            event["contact_distance_m"] = self._contact_distance()
            event["teacher_revision"] = self.revision
            if not event["candidate_bc_eligible"]:
                raise TeacherStopped("oracle_unload_made_no_object_progress")

    def orient(self):
        from scipy.spatial.transform import Rotation
        while not self.runner.success:
            row = self.runner.row()
            obj = pose_array(row["prism_pose_world"])
            correction = (rotation(self.target) * rotation(obj).inv()).as_rotvec()
            error = float(np.rad2deg(np.linalg.norm(correction)))
            if error <= self.config["rotation_tolerance_deg"]:
                return
            angle = min(error, self.config["rotation_step_deg"])
            desired = with_rotation(obj, Rotation.from_rotvec(correction * (angle/error)) * rotation(obj))
            event = self._move(ee_for_object_pose(row["ee_pose_world"], obj, desired),
                "oracle_align_rotation", "bounded_rotation_towards_true_hole_axis_after_verified_unload",
                self.config["rotation_actions"], desired_object=desired, rotation_delta_deg=angle)
            if event is not None:
                event["teacher_revision"] = self.revision
                event["contact_distance_m"] = self._contact_distance()
                if not event["candidate_bc_eligible"]:
                    raise TeacherStopped("oracle_rotation_did_not_improve_orientation")
