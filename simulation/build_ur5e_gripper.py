
import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parent.parent

UR5E_XML = (
    ROOT / "external/mujoco_menagerie/"
    "universal_robots_ur5e/ur5e.xml"
)

GRIPPER_XML = (
    ROOT / "external/mujoco_menagerie/"
    "robotiq_2f85/2f85.xml"
)

OUTPUT_XML = ROOT / "simulation/ur5e_gripper.xml"


def rename_gripper_elements(root):
    """Give gripper bodies, meshes, materials and classes unique names."""

    body_names = {}
    mesh_names = {}
    material_names = {}
    site_names = {}
    class_names = {}

    # Rename bodies
    for body in root.iter("body"):
        old = body.get("name")
        if old:
            body_names[old] = f"gripper_{old}"

    for body in root.iter("body"):
        old = body.get("name")
        if old:
            body.set("name", body_names[old])

    # Rename materials
    asset = root.find("asset")
    if asset is not None:
        for material in asset.findall("material"):
            old = material.get("name")
            if old:
                material_names[old] = f"gripper_{old}"

        for material in asset.findall("material"):
            old = material.get("name")
            if old:
                material.set("name", material_names[old])

        # Give every mesh an explicit unique name
        for mesh in asset.findall("mesh"):
            old = mesh.get("name")
            if not old:
                old = Path(mesh.get("file")).stem

            mesh_names[old] = f"gripper_{old}"

        for mesh in asset.findall("mesh"):
            old = mesh.get("name") or Path(mesh.get("file")).stem
            mesh.set("name", mesh_names[old])

    # Rename gripper sites
    for site in root.iter("site"):
        old = site.get("name")
        if old:
            site_names[old] = f"gripper_{old}"

    for site in root.iter("site"):
        old = site.get("name")
        if old:
            site.set("name", site_names[old])

    # Rename default classes
    for default in root.iter("default"):
        old = default.get("class")
        if old:
            class_names[old] = f"gripper_{old}"

    # Update references throughout the gripper model
    for element in root.iter():
        for attribute in ("body1", "body2"):
            old = element.get(attribute)
            if old in body_names:
                element.set(attribute, body_names[old])

        old = element.get("material")
        if old in material_names:
            element.set("material", material_names[old])

        old = element.get("mesh")
        if old in mesh_names:
            element.set("mesh", mesh_names[old])

        old = element.get("class")
        if old in class_names:
            element.set("class", class_names[old])

        old = element.get("childclass")
        if old in class_names:
            element.set("childclass", class_names[old])

    return body_names


def update_mesh_paths(root, source_xml):
    """Make mesh paths relative to the combined XML file."""

    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "assets") if compiler is not None else "assets"

    asset = root.find("asset")
    if asset is None:
        return

    for mesh in asset.findall("mesh"):
        original = source_xml.parent / meshdir / mesh.get("file")
        mesh.set(
            "file",
            Path(os.path.relpath(original, OUTPUT_XML.parent)).as_posix()
        )


def append_section(destination, source, section_name):
    """Merge compatible MJCF sections."""

    source_section = source.find(section_name)

    if source_section is None:
        return

    destination_section = destination.find(section_name)

    if destination_section is None:
        destination_section = ET.SubElement(
            destination, section_name
        )

    for child in source_section:
        destination_section.append(copy.deepcopy(child))


def main():
    print("\nBUILDING UR5e + ROBOTIQ 2F-85")
    print("=" * 55)

    ur5e = ET.parse(UR5E_XML).getroot()
    gripper = ET.parse(GRIPPER_XML).getroot()

    # Make gripper identifiers unique
    rename_gripper_elements(gripper)

    # Fix mesh paths for both models
    update_mesh_paths(ur5e, UR5E_XML)
    update_mesh_paths(gripper, GRIPPER_XML)

    # Use a single mesh directory
    compiler = ur5e.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", ".")

    # Merge assets
    ur5e_asset = ur5e.find("asset")
    gripper_asset = gripper.find("asset")

    if ur5e_asset is None:
        ur5e_asset = ET.SubElement(ur5e, "asset")

    if gripper_asset is not None:
        for child in gripper_asset:
            ur5e_asset.append(copy.deepcopy(child))

    # Merge default classes
    ur5e_default = ur5e.find("default")
    gripper_default = gripper.find("default")

    if ur5e_default is None:
        ur5e_default = ET.SubElement(ur5e, "default")

    if gripper_default is not None:
        for child in gripper_default:
            ur5e_default.append(copy.deepcopy(child))

    # Find UR5e attachment site
    wrist_site = ur5e.find(
        ".//body[@name='wrist_3_link']/site[@name='attachment_site']"
    )

    if wrist_site is None:
        raise RuntimeError("UR5e attachment_site not found")

    wrist = ur5e.find(".//body[@name='wrist_3_link']")

    # Find gripper root body
    gripper_world = gripper.find("worldbody")
    gripper_root_body = gripper_world.find(
        "./body[@name='gripper_base_mount']"
    )

    if gripper_root_body is None:
        raise RuntimeError("Gripper root body not found")

    # Attach gripper using the UR5e site transform
    attachment = ET.SubElement(
        wrist,
        "body",
        {
            "name": "gripper_attachment",
            "pos": wrist_site.get("pos", "0 0 0"),
            "quat": wrist_site.get("quat", "1 0 0 0")
        }
    )

    attachment.append(copy.deepcopy(gripper_root_body))

    # Merge actuator, tendon, equality and contact definitions
    for section in ("tendon", "equality", "contact", "actuator"):
        append_section(ur5e, gripper, section)

    # Write combined model
    OUTPUT_XML.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.ElementTree(ur5e)
    ET.indent(tree, space="  ")
    tree.write(
        OUTPUT_XML,
        encoding="utf-8",
        xml_declaration=True
    )

    print("Combined XML saved:", OUTPUT_XML)

    # Compile and verify the model
    print("\nCOMPILING MODEL...")

    model = mujoco.MjModel.from_xml_path(str(OUTPUT_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("\nMODEL VERIFICATION")
    print("-" * 40)
    print("Total joints:", model.njnt)
    print("Actuators:", model.nu)
    print("Position DOFs:", model.nq)
    print("Velocity DOFs:", model.nv)

    expected_actuators = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
        "fingers_actuator"
    ]

    for name in expected_actuators:
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name
        )
        if actuator_id < 0:
            raise RuntimeError(f"Missing actuator: {name}")
        print("Verified actuator:", name)

    pinch_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "gripper_pinch"
    )

    if pinch_id < 0:
        raise RuntimeError("Gripper pinch site missing")

    print("Gripper pinch position:", data.site_xpos[pinch_id])
    print("\nSUCCESS: Combined model compiled successfully!")


if __name__ == "__main__":
    main()
