import tempfile
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_urdf_loader.generator import load_urdf_into_mjcf
from mujoco_urdf_loader.hands_fcn import (
    add_hand_actuators,
    add_hand_equalities,
    add_wrist_actuators,
    set_thumb_angle,
)
from mujoco_urdf_loader.mjcf_fcn import (
    add_camera,
    add_new_worldbody,
    add_position_actuator,
    separate_left_right_collision_groups,
    set_joint_damping,
)
from mujoco_urdf_loader.urdf_fcn import (
    add_mujoco_element,
    get_mesh_path,
    get_robot_urdf,
    remove_gazebo_elements,
)

# Load the robot urdf
robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")

# find the mesh path
mesh_path = get_mesh_path(robot_urdf)

# remove the gazebo elements
robot_urdf = remove_gazebo_elements(robot_urdf)

# add the mujoco element
robot_urdf = add_mujoco_element(robot_urdf, mesh_path)

# Load the converted model to add the actuators, sensors and constraints
mjcf = load_urdf_into_mjcf(robot_urdf)

# create a new worldbody, add the robot to it and remove the old worldbody
add_new_worldbody(mjcf, freeze_root=False)

add_hand_equalities(mjcf)
set_thumb_angle(mjcf, angle=9 * 5)

# define the hand elements
hand_elements = ["thumb", "index", "middle", "ring", "pinkie"]

# add actuators
add_hand_actuators(mjcf, hand_elements=hand_elements)
add_wrist_actuators(mjcf)

for joint in mjcf.findall(".//body/joint"):
    if not any(
        joint_element in joint.attrib["name"] for joint_element in hand_elements
    ):
        ctrlrange = joint.attrib["range"]
        add_position_actuator(
            mjcf,
            joint=joint.attrib["name"],
            ctrlrange=[float(ctrlrange.split()[0]), float(ctrlrange.split()[1])],
            kp=1000,
            name=joint.attrib["name"] + "_motor",
        )

# set the damping
set_joint_damping(mjcf, damping=2)
set_joint_damping(mjcf, subset=hand_elements, damping=0.005)

# add camera to the robot
for body in mjcf.findall(".//body"):
    if "realsense" in body.attrib["name"]:
        add_camera(body, name=body.attrib["name"], r_y=-np.pi / 2, r_z=np.pi / 2)

# create collision groups and affinities
separate_left_right_collision_groups(mjcf)

# print the model
mjmodel_str = ET.tostring(mjcf, encoding="unicode", method="xml")
# print(mjmodel_str)

# save the model
with open("ergoCub.xml", "w") as f:
    f.write(mjmodel_str)

# save the model to a temporary file
path_temp_xml = tempfile.NamedTemporaryFile(mode="w+")
with open(path_temp_xml.name, "w") as f:
    f.write(mjmodel_str)

# include the model in a simple world
world_str = f"""
<mujoco model="ergoCubWorld">
    <include file="{path_temp_xml.name}"/>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
        markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>

    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <camera name="default" pos="0.846 -1.465 0.916" xyaxes="0.866 0.500 0.000 -0.171 0.296 0.940"/>
        <geom name="floor" pos="0 0 -0.78" size="0 0 0.05" type="plane" material="groundplane"/>
    </worldbody>
</mujoco>
"""
# print(world_str)


# Load the model in mujoco and visualize it
model = mujoco.MjModel.from_xml_string(world_str)
data = mujoco.MjData(model)

# Visualize the model
mujoco.viewer.launch(model=model, data=data)
