"""Screen a saved FMVP Cartesian trajectory against a Franka kinematic model.

This checks IK, joint limits, and the speed implied by the recorded 20 ms
decision interval. It is a diagnostic, not a deployment or collision-safety
certificate: the source scene does not record robot/body or robot/cloth contact.
Pass the measured Franka base pose and calibrated grasp-frame orientation for
an embodiment-specific result. No plots or simulator edits are required.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pybullet as p


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = (ROOT.parent / "genesis-world/genesis/assets/urdf/"
                "panda_bullet/panda.urdf")


def quat_error_deg(actual, desired):
    dot = float(abs(np.dot(actual, desired)))
    return math.degrees(2.0 * math.acos(min(1.0, max(0.0, dot))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectory", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    ap.add_argument("--base-pos", type=float, nargs=3, required=True,
                    help="Measured Franka base origin in the trajectory world frame, metres.")
    ap.add_argument("--base-quat", type=float, nargs=4, default=[0, 0, 0, 1],
                    metavar=("X", "Y", "Z", "W"))
    ap.add_argument("--tool-link", default="panda_grasptarget")
    ap.add_argument("--arm-joint-prefix", default="panda_joint",
                    help="Use fr3_joint for an FR3 description, if that is the real arm.")
    ap.add_argument("--orientation-mode", choices=("none", "fixed", "source"), default="none")
    ap.add_argument("--tool-quat", type=float, nargs=4,
                    help="World-frame Franka grasp-target quaternion for fixed mode.")
    ap.add_argument("--source-to-tool-quat", type=float, nargs=4,
                    help="Calibrated source-EE to Franka grasp-target rotation for source mode.")
    ap.add_argument("--decision-dt", type=float, default=0.02)
    ap.add_argument("--max-position-error", type=float, default=0.01)
    ap.add_argument("--max-orientation-error-deg", type=float, default=5.0)
    args = ap.parse_args()
    if args.decision_dt <= 0:
        ap.error("--decision-dt must be positive")
    if args.orientation_mode == "fixed" and args.tool_quat is None:
        ap.error("--tool-quat is required for fixed orientation")
    if args.orientation_mode == "source" and args.source_to_tool_quat is None:
        ap.error("--source-to-tool-quat is required for source orientation")

    with np.load(args.trajectory, allow_pickle=False) as data:
        tcp = np.asarray(data["tcp"], dtype=np.float64)
        source_quat = (np.asarray(data["tcp_quat"], dtype=np.float64)
                       if "tcp_quat" in data else None)
        metadata = json.loads(str(data["metadata_json"]))
    if tcp.ndim != 2 or tcp.shape[1] != 3 or not np.isfinite(tcp).all():
        raise ValueError("Expected finite TCP positions [states, 3]")
    if args.orientation_mode == "source" and (source_quat is None or source_quat.shape != (len(tcp), 4)):
        raise ValueError("Source orientation mode requires aligned tcp_quat [states, 4]")

    client = p.connect(p.DIRECT)
    try:
        body = p.loadURDF(str(args.urdf), basePosition=args.base_pos,
                          baseOrientation=args.base_quat, useFixedBase=True,
                          physicsClientId=client)
        joint_info = [p.getJointInfo(body, i, physicsClientId=client)
                      for i in range(p.getNumJoints(body, physicsClientId=client))]
        tool = next((i for i, info in enumerate(joint_info)
                     if info[12].decode() == args.tool_link), None)
        if tool is None:
            raise ValueError("Tool link not found in URDF: " + args.tool_link)
        arm_names = ["%s%d" % (args.arm_joint_prefix, j) for j in range(1, 8)]
        joint_by_name = {info[1].decode(): i for i, info in enumerate(joint_info)}
        arm = [joint_by_name[name] for name in arm_names if name in joint_by_name]
        if len(arm) != 7:
            raise ValueError("Expected seven arm joints named %s1..7" % args.arm_joint_prefix)
        active = [i for i, info in enumerate(joint_info) if info[2] != p.JOINT_FIXED]
        arm_solution_indices = [active.index(i) for i in arm]
        lower = np.array([joint_info[i][8] for i in arm])
        upper = np.array([joint_info[i][9] for i in arm])
        speed_limit = np.array([joint_info[i][11] for i in arm])

        qs, pos_errors, ori_errors = [], [], []
        for k, target in enumerate(tcp):
            desired_quat = None
            if args.orientation_mode == "fixed":
                desired_quat = args.tool_quat
            elif args.orientation_mode == "source":
                desired_quat = p.multiplyTransforms(
                    [0, 0, 0], source_quat[k].tolist(), [0, 0, 0],
                    args.source_to_tool_quat)[1]
            kwargs = dict(bodyUniqueId=body, endEffectorLinkIndex=tool,
                          targetPosition=target.tolist(), maxNumIterations=300,
                          residualThreshold=1e-6, physicsClientId=client)
            if desired_quat is not None:
                kwargs["targetOrientation"] = desired_quat
            solution = p.calculateInverseKinematics(**kwargs)
            q = np.array([solution[i] for i in arm_solution_indices], dtype=np.float64)
            for i, value in zip(arm, q):
                p.resetJointState(body, i, float(value), physicsClientId=client)
            link = p.getLinkState(body, tool, computeForwardKinematics=True,
                                  physicsClientId=client)
            pos_errors.append(float(np.linalg.norm(np.asarray(link[4]) - target)))
            ori_errors.append(quat_error_deg(link[5], desired_quat)
                              if desired_quat is not None else None)
            qs.append(q)
        qs = np.asarray(qs)
        pos_errors = np.asarray(pos_errors)
        violation = (qs < lower[None] - 1e-4) | (qs > upper[None] + 1e-4)
        joint_speed = np.abs(np.diff(qs, axis=0)) / args.decision_dt
        speed_fraction = joint_speed / speed_limit[None]
        ori_array = (np.asarray(ori_errors, dtype=np.float64)
                     if args.orientation_mode != "none" else None)
        report = {
            "source": str(args.trajectory.resolve()),
            "source_robot": metadata.get("robot"),
            "franka_urdf": str(args.urdf.resolve()),
            "base_pos": args.base_pos,
            "base_quat": args.base_quat,
            "tool_link": args.tool_link,
            "arm_joint_names": arm_names,
            "orientation_mode": args.orientation_mode,
            "states": len(tcp),
            "decision_dt_s": args.decision_dt,
            "max_position_error_m": float(pos_errors.max()),
            "position_error_over_threshold_states": int((pos_errors > args.max_position_error).sum()),
            "max_orientation_error_deg": float(ori_array.max()) if ori_array is not None else None,
            "orientation_error_over_threshold_states": int((ori_array > args.max_orientation_error_deg).sum()) if ori_array is not None else None,
            "joint_limit_violation_states": int(violation.any(axis=1).sum()),
            "max_joint_speed_fraction": float(speed_fraction.max()) if len(speed_fraction) else 0.0,
            "speed_limit_violation_intervals": int((speed_fraction > 1).any(axis=1).sum()),
            "max_tcp_speed_m_s": float((np.linalg.norm(np.diff(tcp, axis=0), axis=1) /
                                         args.decision_dt).max()) if len(tcp) > 1 else 0.0,
            "kinematic_pass": bool(not (pos_errors > args.max_position_error).any()
                                   and not violation.any()
                                   and not (speed_fraction > 1).any()
                                   and (ori_array is None or not (ori_array > args.max_orientation_error_deg).any())),
            "physical_transfer_qualified": False,
            "qualification": "position-only diagnostic" if args.orientation_mode == "none"
                             else "kinematic diagnostic; excludes robot-person and robot-cloth collision",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(args.out.with_suffix(".npz"), q=qs.astype(np.float32),
                            pos_error_m=pos_errors.astype(np.float32),
                            orientation_error_deg=ori_array.astype(np.float32)
                            if ori_array is not None else np.empty(0, dtype=np.float32),
                            joint_speed_fraction=speed_fraction.astype(np.float32))
        print(json.dumps(report, indent=2))
    finally:
        p.disconnect(client)


if __name__ == "__main__":
    main()
