# -*- coding: utf-8 -*-
import os
import json
import socket
import math
import numpy as np
from argparse import ArgumentParser

from arguments import PipelineParams
from my_render_traj import (
    opk_to_R_world2cam,
    render_single_view_from_cam_info,
)


# Fixed intrinsics taken from the public traj_random.txt header.
DEFAULT_INTRINSICS = {
    "width": 5280,
    "height": 3956,
    "fx": 3480.185791,
    "fy": 3600.918701,
    "cx": 2690.748469,
    "cy": 1960.206342,
}


def is_obstacle_task(task_id: str) -> bool:
    return str(task_id).strip().lower() == "obstacle"


def scale_intrinsics(intri: dict, s: float) -> dict:
    """Downscale intrinsics before rendering to reduce memory usage and speed up inference."""
    if s <= 0 or s > 1:
        raise ValueError("internal_scale must be in (0,1].")
    out = dict(intri)
    out["width"] = max(1, int(round(intri["width"] * s)))
    out["height"] = max(1, int(round(intri["height"] * s)))
    out["fx"] = intri["fx"] * s
    out["fy"] = intri["fy"] * s
    out["cx"] = intri["cx"] * s
    out["cy"] = intri["cy"] * s
    out["FoVx"] = 2.0 * math.atan(0.5 * out["width"] / out["fx"])
    out["FoVy"] = 2.0 * math.atan(0.5 * out["height"] / out["fy"])
    return out


def cam_info_from_state(
    intrinsics: dict,
    state_xyz_a_rad,
    task_id: str,
    normal_omega_deg: float,
    normal_phi_deg: float,
    obstacle_fixed_omega_deg: float = 90.0,
    obstacle_fixed_kappa_deg: float = 180.0,
):
    """
    state = [x, y, z, a]

    - Normal tasks:
        a = kappa(rad)
        omega = normal_omega_deg
        phi   = normal_phi_deg

    - Obstacle tasks:
        a = phi(rad)
        omega = obstacle_fixed_omega_deg
        kappa = obstacle_fixed_kappa_deg
    """
    x, y, z, a_rad = [float(v) for v in np.asarray(state_xyz_a_rad, dtype=np.float32).reshape(4)]
    C_local = np.array([x, y, z], dtype=np.float32)

    if is_obstacle_task(task_id):
        omega_deg = float(obstacle_fixed_omega_deg)
        phi_deg = float(np.rad2deg(a_rad))
        kappa_deg = float(obstacle_fixed_kappa_deg)
    else:
        omega_deg = float(normal_omega_deg)
        phi_deg = float(normal_phi_deg)
        kappa_deg = float(np.rad2deg(a_rad))

    R_w2c = opk_to_R_world2cam(omega_deg, phi_deg, kappa_deg)
    t_w2c = -R_w2c @ C_local

    return {
        **intrinsics,
        "R_w2c": R_w2c.astype(np.float32),
        "T_w2c": t_w2c.astype(np.float32),
    }


def serve(
    host,
    port,
    ply_template: str,
    default_task_id: str,
    omega,
    phi,
    white_bg,
    pipeline,
    internal_scale,
    obstacle_fixed_omega,
    obstacle_fixed_kappa,
):
    intrinsics = scale_intrinsics(DEFAULT_INTRINSICS, internal_scale)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)

    print(f"[RENDER SERVER] listen {host}:{port}")
    print(f"[RENDER SERVER] ply_template         = {ply_template}")
    print(f"[RENDER SERVER] internal_scale       = {internal_scale}")
    print(
        "[RENDER SERVER] intrinsics           = "
        f"W={intrinsics['width']} H={intrinsics['height']} "
        f"fx={intrinsics['fx']:.3f} fy={intrinsics['fy']:.3f} "
        f"cx={intrinsics['cx']:.3f} cy={intrinsics['cy']:.3f}"
    )
    print(f"[RENDER SERVER] normal omega/phi     = ({omega}, {phi})")
    print(f"[RENDER SERVER] obstacle omega/kappa = ({obstacle_fixed_omega}, {obstacle_fixed_kappa})")

    conn, addr = srv.accept()
    print(f"[RENDER SERVER] client {addr}")
    f = conn.makefile("rwb")

    while True:
        line = f.readline()
        if not line:
            break

        req = json.loads(line.decode("utf-8"))
        cmd = req.get("cmd", "")

        if cmd == "quit":
            f.write((json.dumps({"ok": True}) + "\n").encode("utf-8"))
            f.flush()
            break

        if cmd != "render":
            f.write((json.dumps({"ok": False, "err": f"unknown_cmd: {cmd}"}) + "\n").encode("utf-8"))
            f.flush()
            continue

        state = req["state"]      # [x,y,z,a(rad)] ; a=kappa(rad) or phi(rad) if obstacle
        out_path = req["out_path"]

        env_id = req.get("env_id", None)
        task_id = req.get("task_id", default_task_id)

        try:
            if not env_id:
                raise ValueError("Missing env_id in request (client must send env_id).")

            ply_path = ply_template.format(env_id=env_id, task_id=task_id)

            if not os.path.exists(ply_path):
                raise FileNotFoundError(f"ply not found: {ply_path}")

            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

            cam_info = cam_info_from_state(
                intrinsics=intrinsics,
                state_xyz_a_rad=state,
                task_id=task_id,
                normal_omega_deg=omega,
                normal_phi_deg=phi,
                obstacle_fixed_omega_deg=obstacle_fixed_omega,
                obstacle_fixed_kappa_deg=obstacle_fixed_kappa,
            )

            render_single_view_from_cam_info(
                ply_path=ply_path,
                cam_info=cam_info,
                pipeline=pipeline,
                out_path=out_path,
                white_background=white_bg,
            )

            f.write((json.dumps({"ok": True, "out_path": out_path}) + "\n").encode("utf-8"))
            f.flush()

        except Exception as e:
            f.write((json.dumps({"ok": False, "err": str(e)}) + "\n").encode("utf-8"))
            f.flush()

    try:
        f.close()
        conn.close()
        srv.close()
    except Exception:
        pass

    print("[RENDER SERVER] stopped")


def main():
    parser = ArgumentParser("3DGS Render Server (env_id/task_id from client, obstacle predicts phi)")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5550)

    parser.add_argument(
        "--ply_template",
        type=str,
        default="/mnt/jingyu/ECCV_data/data_3d/{env_id}/3dgs_ply/point_cloud_utm50.ply",
    )
    parser.add_argument("--default_task_id", type=str, default="obstacle")
    # For normal tasks, state[3] is interpreted as kappa(rad).
    parser.add_argument("--omega", type=float, default=-180.0)
    parser.add_argument("--phi", type=float, default=0.0)
    # For obstacle tasks, state[3] is interpreted as phi(rad).
    parser.add_argument("--obstacle_fixed_omega", type=float, default=-90.0)
    parser.add_argument("--obstacle_fixed_kappa", type=float, default=0.0)

    parser.add_argument("--white_bg", action="store_true")
    # Downscale the render resolution before rasterization to avoid OOM.
    parser.add_argument("--internal_scale", type=float, default=0.065)

    pipeline_params = PipelineParams(parser)
    args = parser.parse_args()
    pipeline = pipeline_params.extract(args)

    serve(
        host=args.host,
        port=args.port,
        ply_template=args.ply_template,
        default_task_id=args.default_task_id,
        omega=args.omega,
        phi=args.phi,
        white_bg=args.white_bg,
        pipeline=pipeline,
        internal_scale=args.internal_scale,
        obstacle_fixed_omega=args.obstacle_fixed_omega,
        obstacle_fixed_kappa=args.obstacle_fixed_kappa,
    )


if __name__ == "__main__":
    main()

