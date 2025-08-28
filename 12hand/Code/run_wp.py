import mujoco
import mujoco_viewer
import numpy as np
from itertools import product
from typing import Optional, Sequence


def bernstein_poly(i: int, n: int, t: float) -> float:
    """Базис многочлена Бернштейна для кривых Безье."""
    from math import comb
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bezier_surface(control_points: np.ndarray, res_u: int, res_v: int) -> np.ndarray:
    """
    Строит поверхность Безье по контрольным точкам.
    control_points : матрица (m x n)
    res_u, res_v   : число выборок по U (строки), V (столбцы)
    Возвращает матрицу (res_u x res_v)
    """
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


def create_grid_pins(
    width_mm: float = 30.0,
    height_mm: float = 40.0,
    n_cols: int = 8,
    n_rows: int = 10,
    base_xml: str = "base.xml",
    mod_xml: str = "Winkler-Pasternak_bezier.xml",
    include_diagonals: bool = True,
    run_simulation: bool = True,
    drop_radius_mm: float = 10.0,
    heights_mm: Optional[Sequence[Sequence[float]]] = None,
    control_heights: Optional[Sequence[Sequence[float]]] = None,
):
    """
    Создаёт модель сетки пинов и записывает XML.

    Приоритет параметров:
      1. control_heights -> интерполяция Bézier.
      2. heights_mm      -> напрямую.
      3. иначе           -> базовая плоская матрица (все одинаковые).
    """
    if n_cols <= 0 or n_rows <= 0:
        raise ValueError("n_cols и n_rows должны быть положительными целыми числами")
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("width_mm и height_mm должны быть положительными")
    if drop_radius_mm <= 0:
        raise ValueError("drop_radius_mm должен быть положительным")

    W = width_mm / 1000.0
    H = height_mm / 1000.0
    drop_radius = drop_radius_mm / 1000.0

    # максимально возможный радиус, чтобы сферы не пересекались и оставались внутри зоны:
    sphere_radius = min(W / (2.0 * n_cols), H / (2.0 * n_rows))
    site_radius = max(1e-4, sphere_radius * 0.2)

    # координаты центров в XY
    x_coords = (
        np.array([0.0])
        if n_cols == 1
        else np.linspace(-(W / 2.0 - sphere_radius), (W / 2.0 - sphere_radius), n_cols)
    )
    y_coords = (
        np.array([0.0])
        if n_rows == 1
        else np.linspace(-(H / 2.0 - sphere_radius), (H / 2.0 - sphere_radius), n_rows)
    )

    # генерируем матрицу высот
    if control_heights is not None:
        heights_mm_arr = bezier_surface(np.array(control_heights, dtype=float), n_rows, n_cols)
    elif heights_mm is not None:
        heights_mm_arr = np.array(heights_mm, dtype=float)
        if heights_mm_arr.shape != (n_rows, n_cols):
            raise ValueError(
                f"heights_mm должен иметь форму ({n_rows}, {n_cols}), а получил {heights_mm_arr.shape}"
            )
    else:
        heights_mm_arr = np.full((n_rows, n_cols), 40.0, dtype=float)

    heights_m = heights_mm_arr / 1000.0

    # ------------------------
    # read base scene
    # ------------------------
    spec = mujoco.MjSpec.from_file(base_xml)

    # ------------------------
    # base plate
    # ------------------------
    plate = spec.worldbody.add_body(name="base_plate", pos=[0, 0, 0])
    plate.add_geom(
        name="plate_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[50e-3, 50e-3, 1e-3],
        pos=[0, 0, 0],
        rgba=[1, 0, 0, 0.3],
        contype=1,
        conaffinity=2,
    )

    sites = []
    site_names = []

    # создаём пины
    for i in range(n_rows):
        for j in range(n_cols):
            x = float(x_coords[j])
            y = float(y_coords[i])
            z_ij = float(heights_m[i, j])
            body_name = f"pin_{i}_{j}"
            body = plate.add_body(name=body_name, pos=[0, 0, 0])
            body.add_joint(
                name=f"{body_name}_joint",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                pos=[0, 0, 0],
                axis=[0, 0, 1],
                range=[-z_ij, 0.0],
                stiffness=5,
                springref=0,
                damping=1e-1,
            )
            body.add_geom(
                name=f"{body_name}_geom",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[sphere_radius, 0, 0],
                pos=[x, y, z_ij],
                rgba=[0.5, 0.5, 0.5, 1.0],
                contype=2,
                conaffinity=1,
            )
            site = body.add_site(
                name=f"{body_name}_site",
                pos=[x, y, z_ij],
                size=[site_radius, 0, 0],
                rgba=[1, 0, 0, 1],
            )
            sites.append(site)
            site_names.append(site.name)

    # ------------------------
    # падающая сфера
    # ------------------------
    grid_max_extent = 0.0
    if len(x_coords) > 0:
        grid_max_extent = max(grid_max_extent, max(abs(x_coords[0]), abs(x_coords[-1])))
    if len(y_coords) > 0:
        grid_max_extent = max(grid_max_extent, max(abs(y_coords[0]), abs(y_coords[-1])))
    z_drop = float(np.max(heights_m)) + grid_max_extent + 0.10

    sphere_body = spec.worldbody.add_body(name="dropping_sphere", pos=[0.0, 0.0, z_drop])
    sphere_body.add_joint(
        name="dropping_sphere_free_joint", type=mujoco.mjtJoint.mjJNT_FREE
    )
    sphere_body.add_geom(
        name="dropping_sphere_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[drop_radius, 0, 0],
        pos=[0.0, 0.0, 0.0],
        rgba=[0.0, 0.0, 1.0, 1.0],
        contype=1,
        conaffinity=2,
    )

    # ------------------------
    # compile -> XML
    # ------------------------
    _ = spec.compile()
    xml = spec.to_xml()

    # ------------------------
    # создаём тендоны (spatial)
    # ------------------------
    def neighbors(i, j):
        nbs = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if include_diagonals:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < n_rows and 0 <= nj < n_cols:
                nbs.append((ni, nj))
        return nbs

    index_map = {(i, j): i * n_cols + j for i in range(n_rows) for j in range(n_cols)}

    tendon_lines = ["  <tendon>"]
    for i in range(n_rows):
        for j in range(n_cols):
            idx = index_map[(i, j)]
            for ni, nj in neighbors(i, j):
                nidx = index_map[(ni, nj)]
                if idx < nidx:
                    name = f"rope_{i}_{j}__{ni}_{nj}"
                    s1 = site_names[idx]
                    s2 = site_names[nidx]
                    tendon_lines.append(
                        f"    <spatial name='{name}' stiffness='50' damping='0.01' width='0.001' rgba='0 0 0 0'>"
                        f"<site site='{s1}'/><site site='{s2}'/>"
                        f"</spatial>"
                    )
    tendon_lines.append("  </tendon>")

    xml = xml.replace("</mujoco>", "\n" + "\n".join(tendon_lines) + "\n</mujoco>")

    # ------------------------
    # запись файла
    # ------------------------
    with open(mod_xml, "w") as f:
        f.write(xml)

    # ------------------------
    # симуляция (опционально)
    # ------------------------
    if run_simulation:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        viewer = mujoco_viewer.MujocoViewer(model, data, title="grid_pins_tendons")
        while viewer.is_alive:
            mujoco.mj_step(model, data)
            viewer.render()
        viewer.close()


if __name__ == "__main__":
    control_heights_demo = np.array([
        [40.0, 40.0, 40.0, 40.0, 40.0],
        [40.0, 43.0, 45.0, 45.0, 40.0],
        [40.0, 45.0, 55.0, 50.0, 40.0],
        [40.0, 43.0, 45.0, 45.0, 40.0],
        [40.0, 40.0, 40.0, 40.0, 40.0],
    ]).T

    create_grid_pins(
        width_mm=30.0,
        height_mm=40.0,
        n_cols=8,
        n_rows=10,
        base_xml="base.xml",
        mod_xml="Winkler-Pasternak_cage.xml",
        include_diagonals=True,
        run_simulation=True,
        drop_radius_mm=10.0,
        control_heights=control_heights_demo,
    )