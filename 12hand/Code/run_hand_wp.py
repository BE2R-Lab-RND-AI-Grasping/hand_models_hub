import mujoco
import mujoco_viewer
import numpy as np
from typing import Optional, Sequence


def bernstein_poly(i: int, n: int, t: float) -> float:
    from math import comb
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bezier_surface(control_points: np.ndarray, res_u: int, res_v: int) -> np.ndarray:
    m, n = control_points.shape
    u_vals = np.linspace(0, 1, res_u)
    v_vals = np.linspace(0, 1, res_v)
    surface = np.zeros((res_u, res_v))
    for iu, u in enumerate(u_vals):
        for jv, v in enumerate(v_vals):
            s = 0.0
            for i in range(m):
                for j in range(n):
                    s += (
                        control_points[i, j]
                        * bernstein_poly(i, m - 1, u)
                        * bernstein_poly(j, n - 1, v)
                    )
            surface[iu, jv] = s
    return surface


def add_finger(spec, root_pos, root_euler, finger_id, length1, length2):
    ROT = spec.worldbody.add_body(name=f"ID{finger_id}_ROT", pos=list(root_pos), euler=list(root_euler))
    ROT.add_geom(
        name=f"ID{finger_id}_ROT_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[45e-3 / 2, 25e-3 / 2, 36e-3 / 2],
        pos=[-12.5e-3, 0, 0],
        rgba=[0.5, 0.5, 0.5, 0.5],
    )

    ABD = ROT.add_body(name=f"ID{finger_id}_ABD", pos=[-27e-3 - 12.5e-3, 0, 49e-3 + 21e-3], euler=[0, 90, 0])
    ABD.add_joint(
        name=f"ID{finger_id}_ROT_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        pos=[49e-3 + 21e-3, 0, 27e-3 + 12.5e-3],
        axis=[-1, 0, 0],
        range=[-90, 0],
        stiffness=0,
    )
    ABD.add_geom(
        name=f"ID{finger_id}_ABD_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[62e-3 / 2, 25e-3 / 2, 36e-3 / 2],
        pos=[21e-3, 0, 0],
        rgba=[0.5, 0.5, 0.5, 0.5],
    )

    PIP = ABD.add_body(name=f"ID{finger_id}_PIP", pos=[0, 0, 39.5e-3], euler=[-90, 0, 0])
    PIP.add_joint(
        name=f"ID{finger_id}_ABD_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        pos=[0, 27e-3 + 12.5e-3, 0],
        axis=[0, -1, 0],
        range=[-90, 90],
        stiffness=0,
    )
    PIP.add_geom(
        name=f"ID{finger_id}_PIP_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[48e-3 / 2, 33e-3 / 2, 42e-3 / 2],
        pos=[13.6e-3, 3.5e-3, 0],
        rgba=[0.5, 0.5, 0.5, 0.5],
    )

    DIP = PIP.add_body(name=f"ID{finger_id}_DIP", pos=[-length1, 0, 0], euler=[0, 0, 0])
    DIP.add_joint(
        name=f"ID{finger_id}_PIP_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        pos=[length1, 0, 0],
        axis=[0, 0, 1],
        range=[0, 90],
        stiffness=0,
    )
    DIP.add_geom(
        name=f"ID{finger_id}_DIP_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[length1 / 2, 25e-3 / 2, 42e-3 / 2],
        pos=[length1 / 2, 0, 0],
        rgba=[1, 0.5, 0.5, 0.5],
    )

    FIN = DIP.add_body(name=f"ID{finger_id}_FIN", pos=[-length2, 0, 0], euler=[0, 0, 0])
    FIN.add_joint(
        name=f"ID{finger_id}_DIP_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        pos=[length2, 0, 0],
        axis=[0, 0, 1],
        range=[0, 90],
        stiffness=0,
    )
    FIN.add_geom(
        name=f"ID{finger_id}_FIN_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[length2 / 2, 25e-3 / 2, 42e-3 / 2],
        pos=[length2 / 2, 0, 0],
        rgba=[1, 0.5, 0.5, 0.5],
    )

    return ROT, ABD, PIP, DIP, FIN


def attach_soft_pad_on(
    spec,
    parent_body,
    finger_id: str,
    length1: Optional[float] = None,
    width_mm: float = 30.0,
    height_mm: float = 20.0,
    n_cols: int = 6,
    n_rows: int = 4,
    control_heights: Optional[Sequence[Sequence[float]]] = None,
    include_diagonals: bool = True,
    pad_offset: Optional[Sequence[float]] = None,
    pad_pin_offset: Optional[Sequence[float]] = None,
):
    if n_cols <= 0 or n_rows <= 0:
        raise ValueError("n_cols и n_rows должны быть положительными")

    W = width_mm / 1000.0
    H = height_mm / 1000.0
    sphere_radius = min(W / (2.0 * n_cols), H / (2.0 * n_rows))
    site_radius = max(1e-4, sphere_radius * 0.2)

    x_coords = np.array([0.0]) if n_cols == 1 else np.linspace(-(W / 2.0 - sphere_radius), (W / 2.0 - sphere_radius), n_cols)
    y_coords = np.array([0.0]) if n_rows == 1 else np.linspace(-(H / 2.0 - sphere_radius), (H / 2.0 - sphere_radius), n_rows)

    if control_heights is not None:
        heights_mm_arr = bezier_surface(np.array(control_heights, dtype=float), n_rows, n_cols)
    else:
        heights_mm_arr = np.full((n_rows, n_cols), 6.0, dtype=float)
    heights_m = heights_mm_arr / 1000.0

    if pad_offset is None:
        pad_forward = 1e-3
        if length1 is not None and ("DIP" in parent_body.name or "FIN" in parent_body.name or "dip" in parent_body.name.lower()):
            pad_root_pos = [length1 / 2 + pad_forward, 0.0, 0.0]
        else:
            pad_root_pos = [13.6e-3 + pad_forward, 0.0, 0.0]
    else:
        if len(pad_offset) != 3:
            raise ValueError("pad_offset должно быть [dx,dy,dz]")
        pad_root_pos = [float(pad_offset[0]), float(pad_offset[1]), float(pad_offset[2])]

    pad_root = parent_body.add_body(name=f"ID{finger_id}_pad_root_{parent_body.name}", pos=pad_root_pos)

    if pad_pin_offset is None:
        pin_default_pos = [0.0, -25e-3 / 2, 0.0]
    else:
        if len(pad_pin_offset) != 3:
            raise ValueError("pad_pin_offset должно быть [dx,dy,dz]")
        pin_default_pos = [float(pad_pin_offset[0]), float(pad_pin_offset[1]), float(pad_pin_offset[2])]

    site_names = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = float(x_coords[j])
            y = float(y_coords[i])
            z = float(heights_m[i, j])

            body_name = f"pad_{finger_id}_{parent_body.name}_{i}_{j}"
            body = pad_root.add_body(name=body_name, pos=pin_default_pos, euler=[90, 0, 0])

            body.add_joint(
                name=f"{body_name}_joint",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                pos=[0, 0, 0],
                axis=[0, 0, 1],
                range=[-z, 0.0],
                stiffness=5,
                springref=0,
                damping=1e-1,
            )

            body.add_geom(
                name=f"{body_name}_geom",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[sphere_radius, 0, 0],
                pos=[x, y, z],
                rgba=[0.5, 0.5, 0.5, 1.0],
                contype=1,
                conaffinity=1,
            )

            site = body.add_site(name=f"{body_name}_site", pos=[x, y, z], size=[site_radius, 0, 0], rgba=[1, 0, 0, 1])
            site_names.append(site.name)

    return site_names


if __name__ == "__main__":
    base_xml = "base.xml"
    mod_xml = "hand_with_pad.xml"

    spec = mujoco.MjSpec.from_file(base_xml)

    roots = [np.array([0, 45e-3, 0]), np.array([0, -45e-3, 0]), np.array([0, 0, -50e-3])]
    lengths = (70e-3, 70e-3)
    finger_bodies = {}
    for idx, root in enumerate(roots, start=1):
        euler = [180, 90, 0] if idx == 3 else [0, 0, 0]
        ROT, ABD, PIP, DIP, FIN = add_finger(spec, root, euler, str(idx), *lengths)
        finger_bodies[str(idx)] = {"ROT": ROT, "ABD": ABD, "PIP": PIP, "DIP": DIP, "FIN": FIN}

    centers = [r + np.array([-12.5e-3, 0, 0]) for r in roots]
    halfs = [np.array([45e-3 / 2, 25e-3 / 2, 36e-3 / 2])] * 3
    mins = np.vstack([c - h for c, h in zip(centers, halfs)]).min(axis=0)
    maxs = np.vstack([c + h for c, h in zip(centers, halfs)]).max(axis=0)
    ctr = (mins + maxs) / 2
    ext = (maxs - mins) / 2
    ext = np.where(ext < 1e-3, 1e-3, ext)
    spec.worldbody.add_geom(name="fingers_bbox", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(ext), pos=list(ctr), rgba=[0, 1, 0, 0.2])

    # создаём тело для сферы в worldbody и привязываем free joint
    sphere_body = spec.worldbody.add_body(name="movable_sphere_body", pos=[80e-3, 0.0, 50e-3])
    # free joint дает 6 DOF (перемещение + вращение)
    sphere_body.add_joint(name="movable_sphere_freejoint", type=mujoco.mjtJoint.mjJNT_FREE)
    # геометрия в локальных координатах тела (центр в [0,0,0])
    sphere_body.add_geom(
        name="movable_sphere_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[70e-3, 0, 0],
        pos=[0.0, 0.0, 0.0],
        rgba=[0.0, 0.0, 1.0, 1.0],
        contype=1,
        conaffinity=1,
    )

    pad_on_finger_idx_list = ["1", "2", "3"]

    control_heights_distal = np.array([
        [0.0, 2.0, 2.0, 2.0, 0.0],
        [0.0, 10.0, 10.0, 10.0, 0.0],
        [0.0, 10.0, 15.0, 10.0, 0.0],
        [0.0, 10.0, 10.0, 10.0, 0.0],
        [0.0, 10.0, 10.0, 10.0, 0.0],
    ]).T

    control_heights_proximal = np.array([
        [0.0, 10.0, 10.0, 10.0, 0.0],
        [0.0, 10.0, 10.0, 10.0, 0.0],
        [0.0, 10.0, 15.0, 10.0, 0.0],
        [0.0, 10.0, 10.0, 10.0, 0.0],
        [0.0, 10.0, 10.0, 10.0, 0.0],
    ]).T

    distal_pad_params = {
        "width_mm": 60.0,
        "height_mm": 42.0,
        "n_cols": 8,
        "n_rows": 8,
        "control_heights": control_heights_distal,
        "include_diagonals": True,
    }

    proximal_pad_params = {
        "width_mm": 40.0,
        "height_mm": 42.0,
        "n_cols": 8,
        "n_rows": 8,
        "control_heights": control_heights_proximal,
        "include_diagonals": True,
    }

    length1 = lengths[0]
    pad_forward = 1e-3
    fin_pad_offset = [length1 / 2 + pad_forward, 0.0, 0.0]
    dip_pad_offset = [length1 / 2 + pad_forward, 0.0, 0.0]

    fin_pin_offset = [0.0, -25e-3 / 2, 0.0]
    dip_pin_offset = [-5e-3, -25e-3 / 2, 0.0]

    fin_site_names_by_finger = {}
    prox_site_names_by_finger = {}

    for f in pad_on_finger_idx_list:
        fin_body = finger_bodies[f]["FIN"]
        fin_sites = attach_soft_pad_on(
            spec,
            fin_body,
            f,
            length1=length1,
            width_mm=distal_pad_params["width_mm"],
            height_mm=distal_pad_params["height_mm"],
            n_cols=distal_pad_params["n_cols"],
            n_rows=distal_pad_params["n_rows"],
            control_heights=distal_pad_params["control_heights"],
            include_diagonals=distal_pad_params["include_diagonals"],
            pad_offset=fin_pad_offset,
            pad_pin_offset=fin_pin_offset,
        )
        fin_site_names_by_finger[f] = fin_sites

        dip_body = finger_bodies[f]["DIP"]
        prox_sites = attach_soft_pad_on(
            spec,
            dip_body,
            f,
            length1=length1,
            width_mm=proximal_pad_params["width_mm"],
            height_mm=proximal_pad_params["height_mm"],
            n_cols=proximal_pad_params["n_cols"],
            n_rows=proximal_pad_params["n_rows"],
            control_heights=proximal_pad_params["control_heights"],
            include_diagonals=proximal_pad_params["include_diagonals"],
            pad_offset=dip_pad_offset,
            pad_pin_offset=dip_pin_offset,
        )
        prox_site_names_by_finger[f] = prox_sites

    _ = spec.compile()
    xml = spec.to_xml()

    def neighbors(i, j, rows, cols, include_diags=True):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if include_diags:
            dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        out = []
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                out.append((ni, nj))
        return out

    tendon_lines = ["  <tendon>"]

    fin_rows = distal_pad_params["n_rows"]
    fin_cols = distal_pad_params["n_cols"]
    for f in pad_on_finger_idx_list:
        fin_index_map = {(i, j): i * fin_cols + j for i in range(fin_rows) for j in range(fin_cols)}
        fin_sites = fin_site_names_by_finger[f]
        for i in range(fin_rows):
            for j in range(fin_cols):
                idx = fin_index_map[(i, j)]
                for ni, nj in neighbors(i, j, fin_rows, fin_cols, distal_pad_params["include_diagonals"]):
                    nidx = fin_index_map[(ni, nj)]
                    if idx < nidx:
                        name = f"pad_rope_FIN_{f}_{i}_{j}__{ni}_{nj}"
                        s1 = fin_sites[idx]
                        s2 = fin_sites[nidx]
                        tendon_lines.append(
                            f"    <spatial name='{name}' stiffness='50' damping='0.01' width='0.001' rgba='0 0 0 0'>"
                            f"<site site='{s1}'/><site site='{s2}'/></spatial>"
                        )

    prox_rows = proximal_pad_params["n_rows"]
    prox_cols = proximal_pad_params["n_cols"]
    for f in pad_on_finger_idx_list:
        prox_index_map = {(i, j): i * prox_cols + j for i in range(prox_rows) for j in range(prox_cols)}
        prox_sites = prox_site_names_by_finger[f]
        for i in range(prox_rows):
            for j in range(prox_cols):
                idx = prox_index_map[(i, j)]
                for ni, nj in neighbors(i, j, prox_rows, prox_cols, proximal_pad_params["include_diagonals"]):
                    nidx = prox_index_map[(ni, nj)]
                    if idx < nidx:
                        name = f"pad_rope_PROX_{f}_{i}_{j}__{ni}_{nj}"
                        s1 = prox_sites[idx]
                        s2 = prox_sites[nidx]
                        tendon_lines.append(
                            f"    <spatial name='{name}' stiffness='50' damping='0.01' width='0.001' rgba='0 0 0 0'>"
                            f"<site site='{s1}'/><site site='{s2}'/></spatial>"
                        )

    tendon_lines.append("  </tendon>")
    xml = xml.replace("</mujoco>", "\n" + "\n".join(tendon_lines) + "\n</mujoco>")

    actuator_lines = ["  <actuator>"]
    joint_ranges = [
        ("ROT_joint", [-90, 0]),
        ("ABD_joint", [-90, 90]),
        ("PIP_joint", [0, 90]),
        ("DIP_joint", [0, 90]),
    ]
    for f in ("1", "2", "3"):
        for suffix, rng in joint_ranges:
            jn = f"ID{f}_{suffix}"
            name = f"{jn}_pos"
            actuator_lines.append(
                f'    <position name="{name}" joint="{jn}" ctrlrange="{rng[0]} {rng[1]}" timeconst="1"/>'
            )
    actuator_lines.append("  </actuator>")
    xml = xml.replace("</mujoco>", "\n" + "\n".join(actuator_lines) + "\n</mujoco>")

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    with open(mod_xml, "w") as f:
        f.write(xml)

    ID1_ROT = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ID1_ROT_joint_pos")
    ID1_ABD = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ID1_ABD_joint_pos")
    ID1_PIP = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ID1_PIP_joint_pos")
    ID1_DIP = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ID1_DIP_joint_pos")

    ID2_ROT = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ID2_ROT_joint_pos")
    ID2_ABD = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ID2_ABD_joint_pos")
    ID2_PIP = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ID2_PIP_joint_pos")
    ID2_DIP = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ID2_DIP_joint_pos")

    ID3_ROT = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ID3_ROT_joint_pos")
    ID3_ABD = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ID3_ABD_joint_pos")
    ID3_PIP = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ID3_PIP_joint_pos")
    ID3_DIP = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ID3_DIP_joint_pos")

    viewer = mujoco_viewer.MujocoViewer(model, data, title="modular_fingers_with_pads")
    while viewer.is_alive:
        data.ctrl[ID1_ROT] = np.deg2rad(-30)
        data.ctrl[ID1_ABD] = np.deg2rad(-45)
        data.ctrl[ID1_PIP] = np.deg2rad(30)
        data.ctrl[ID1_DIP] = np.deg2rad(45)

        data.ctrl[ID2_ROT] = np.deg2rad(30)
        data.ctrl[ID2_ABD] = np.deg2rad(45)
        data.ctrl[ID2_PIP] = np.deg2rad(30)
        data.ctrl[ID2_DIP] = np.deg2rad(45)

        data.ctrl[ID3_ROT] = np.deg2rad(0)
        data.ctrl[ID3_ABD] = np.deg2rad(0)
        data.ctrl[ID3_PIP] = np.deg2rad(30)
        data.ctrl[ID3_DIP] = np.deg2rad(45)

        mujoco.mj_step(model, data)
        viewer.render()
    viewer.close()
