import mujoco
import mujoco_viewer
import numpy as np

# Параметризация кисти состоит из:
# roots - массив положения корневых тел для каждого пальца
# lengths - длины проксимальной и дистальной фаланг. Для всех пальцев заданы одинаковые длины

def add_finger(spec, root_pos, root_euler, finger_id, length1, length2):
    """
    Добавляет цепочку фаланг пальца (ROT, ABD, PIP, DIP, FIN) в spec.worldbody.
    Возвращает ссылку на тело ROT.
    """

    # Корень вращения (ROT)
    ROT = spec.worldbody.add_body(
        name=f"ID{finger_id}_ROT",
        pos=list(root_pos),
        euler=list(root_euler)
    )
    ROT.add_geom(
        name=f"ID{finger_id}_ROT_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[45e-3/2, 25e-3/2, 36e-3/2],
        pos=[-12.5e-3, 0, 0],
        rgba=[0.5, 0.5, 0.5, 0.5]
    )

    # Абдукционно‑приводная часть (ABD)
    ABD = ROT.add_body(
        name=f"ID{finger_id}_ABD",
        pos=[-27e-3 - 12.5e-3, 0, 49e-3 + 21e-3],
        euler=[0, 90, 0]
    )
    ABD.add_joint(
        name=f"ID{finger_id}_ROT_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        pos=[49e-3 + 21e-3, 0, 27e-3 + 12.5e-3],
        axis=[-1, 0, 0],
        range=[-90, 0],
        stiffness=0
    )
    ABD.add_geom(
        name=f"ID{finger_id}_ABD_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[62e-3/2, 25e-3/2, 36e-3/2],
        pos=[21e-3, 0, 0],
        rgba=[0.5, 0.5, 0.5, 0.5]
    )

    # Средняя фаланга (PIP)
    PIP = ABD.add_body(
        name=f"ID{finger_id}_PIP",
        pos=[0, 0, 39.5e-3],
        euler=[-90, 0, 0]
    )
    PIP.add_joint(
        name=f"ID{finger_id}_ABD_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        pos=[0, 27e-3 + 12.5e-3, 0],
        axis=[0, -1, 0],
        range=[-90, 90],
        stiffness=0
    )
    PIP.add_geom(
        name=f"ID{finger_id}_PIP_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[48e-3/2, 33e-3/2, 42e-3/2],
        pos=[13.6e-3, 3.5e-3, 0],
        rgba=[0.5, 0.5, 0.5, 0.5]
    )

    # Дистальная фаланга (DIP)
    DIP = PIP.add_body(
        name=f"ID{finger_id}_DIP",
        pos=[-length1, 0, 0],
        euler=[0, 0, 0]
    )
    DIP.add_joint(
        name=f"ID{finger_id}_PIP_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        pos=[length1, 0, 0],
        axis=[0, 0, 1],
        range=[0, 90],
        stiffness=0
    )
    DIP.add_geom(
        name=f"ID{finger_id}_DIP_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[length1/2, 25e-3/2, 42e-3/2],
        pos=[length1/2, 0, 0],
        rgba=[1, 0.5, 0.5, 0.5]
    )

    # Конечная фаланга (FIN)
    FIN = DIP.add_body(
        name=f"ID{finger_id}_FIN",
        pos=[-length2, 0, 0],
        euler=[0, 0, 0]
    )
    FIN.add_joint(
        name=f"ID{finger_id}_DIP_joint",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        pos=[length2, 0, 0],
        axis=[0, 0, 1],
        range=[0, 90],
        stiffness=0
    )
    FIN.add_geom(
        name=f"ID{finger_id}_FIN_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[length2/2, 35e-3/2, 35e-3/2],
        pos=[length2/2, 0, 0],
        rgba=[1, 0.5, 0.5, 0.5]
    )

    return ROT

if __name__ == "__main__":
    # Пути к файлам
    base_xml = "base.xml"
    mod_xml  = "hand.xml"

    # 1) Загружаем спецификацию без актуаторов
    spec = mujoco.MjSpec.from_file(base_xml)

    # 2) Добавляем три пальца
    roots = [
        np.array([0,   45e-3,   0]),
        np.array([0,  -45e-3,   0]),
        np.array([0,    0,   -50e-3])
    ]
    lengths = (70e-3, 70e-3)
    for idx, root in enumerate(roots, start=1):
        euler = [180, 90, 0] if idx == 3 else [0, 0, 0]
        add_finger(spec, root, euler, str(idx), *lengths)

    # 3) (Опционально) добавляем bounding box для всех пальцев
    centers = [r + np.array([-12.5e-3, 0, 0]) for r in roots]
    halfs   = [np.array([45e-3/2, 25e-3/2, 36e-3/2])] * 3
    mins = np.vstack([c - h for c, h in zip(centers, halfs)]).min(axis=0)
    maxs = np.vstack([c + h for c, h in zip(centers, halfs)]).max(axis=0)
    ctr  = (mins + maxs) / 2
    ext  = (maxs - mins) / 2
    ext  = np.where(ext < 1e-3, 1e-3, ext)
    spec.worldbody.add_geom(
        name="fingers_bbox",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(ext),
        pos=list(ctr),
        rgba=[0, 1, 0, 0.2]
    )

    # 4) Промежуточная компиляция, чтобы spec.to_xml() сработал
    _ = spec.compile()

    # 5) Получаем строку MJCF
    xml = spec.to_xml()

    # 6) Вставляем блок <actuator> перед закрывающим тегом </mujoco>
    actuator_lines = ["  <actuator>"]
    joint_ranges = [
        ("ROT_joint", [-90,   0]),
        ("ABD_joint", [-90,  90]),
        ("PIP_joint", [  0,  90]),
        ("DIP_joint", [  0,  90]),
    ]
    for f in ("1", "2", "3"):
        for suffix, rng in joint_ranges:
            jn   = f"ID{f}_{suffix}"
            name = f"{jn}_pos"
            actuator_lines.append(
                f'    <position name="{name}" joint="{jn}" ctrlrange="{rng[0]} {rng[1]}" timeconst="1"/>'
            )
    actuator_lines.append("  </actuator>")
    xml = xml.replace("</mujoco>", "\n" + "\n".join(actuator_lines) + "\n</mujoco>")

    # 7) Компилируем финальную модель из строки MJCF
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)

    # 8) Сохраняем XML на диск (опционально)
    with open(mod_xml, "w") as f:
        f.write(xml)

    # 9) Проверяем индекс первого актуатора
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

    # 10) Запускаем визуализацию
    viewer = mujoco_viewer.MujocoViewer(model, data, title="modular_fingers_actuators")
    while viewer.is_alive:
        
        data.ctrl[ID1_ROT] = np.deg2rad(-30) # [-90,   0]
        data.ctrl[ID1_ABD] = np.deg2rad(-45) # [-90,   90]
        data.ctrl[ID1_PIP] = np.deg2rad(30) # [0,   00]
        data.ctrl[ID1_DIP] = np.deg2rad(30) # [0,   00]
        
        data.ctrl[ID2_ROT] = np.deg2rad(30) # [-90,   0]
        data.ctrl[ID2_ABD] = np.deg2rad(45) # [-90,   90]
        data.ctrl[ID2_PIP] = np.deg2rad(30) # [0,   00]
        data.ctrl[ID2_DIP] = np.deg2rad(30) # [0,   00]
        
        data.ctrl[ID3_ROT] = np.deg2rad(0) # [-90,   0]
        data.ctrl[ID3_ABD] = np.deg2rad(0) # [-90,   90]
        data.ctrl[ID3_PIP] = np.deg2rad(45) # [0,   00]
        data.ctrl[ID3_DIP] = np.deg2rad(45) # [0,   00]
        
        mujoco.mj_step(model, data)
        viewer.render()
    viewer.close()
