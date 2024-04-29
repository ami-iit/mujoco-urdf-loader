import copy
import xml.etree.ElementTree as ET

from generator import load_urdf_into_mjcf
from mjcf_fcn import add_new_worldbody, separate_left_right_collision_groups
from urdf_fcn import (
    add_mujoco_element,
    get_mesh_path,
    get_robot_urdf,
    remove_gazebo_elements,
    remove_links_and_joints_by_keep_list,
    remove_links_and_joints_by_remove_list,
)

# Load the robot urdf
robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")

# find the mesh path
mesh_path = get_mesh_path(robot_urdf)

# remove the gazebo elements
robot_urdf = remove_gazebo_elements(robot_urdf)

# add the mujoco element to be able to load the urdf into the mjcf
robot_urdf = add_mujoco_element(robot_urdf, mesh_path)

### Full robot, free root ###
mjcf_full_fr = load_urdf_into_mjcf(robot_urdf)

add_new_worldbody(mjcf_full_fr, freeze_root=False)
separate_left_right_collision_groups(mjcf_full_fr)

# save the full robot with free root
with open("ergoCub_full_fr.xml", "w") as f:
    mjcf_full_fr_str = ET.tostring(mjcf_full_fr, encoding="unicode", method="xml")
    f.write(mjcf_full_fr_str)

### Full robot, fixed root ###
mjcf_full_xr = load_urdf_into_mjcf(robot_urdf)

add_new_worldbody(mjcf_full_xr, freeze_root=True)
separate_left_right_collision_groups(mjcf_full_xr)

# save the full robot with fixed root
with open("ergoCub_full_xr.xml", "w") as f:
    mjcf_full_xr_str = ET.tostring(mjcf_full_xr, encoding="unicode", method="xml")
    f.write(mjcf_full_xr_str)

### Torso only ###
to_remove = ["leg", "foot", "ankle", "hip", "knee", "sole"]

robot_urdf_torso = remove_links_and_joints_by_remove_list(robot_urdf, to_remove)

mjcf_torso = load_urdf_into_mjcf(robot_urdf_torso)

add_new_worldbody(mjcf_torso, freeze_root=True)
separate_left_right_collision_groups(mjcf_torso)

# save the torso only
with open("ergoCub_torso.xml", "w") as f:
    mjcf_torso_str = ET.tostring(mjcf_torso, encoding="unicode", method="xml")
    f.write(mjcf_torso_str)

### Hand only ###

to_keep = [
    "r_hand",
    "r_wrist",
    "r_forearm",
    "r_pinkie",
    "r_ring",
    "r_middle",
    "r_index",
    "r_thumb",
]

robot_urdf_hand = remove_links_and_joints_by_keep_list(robot_urdf, to_keep)

mjcf_hand = load_urdf_into_mjcf(robot_urdf_hand)

add_new_worldbody(mjcf_hand, freeze_root=True)

# save the hand only
with open("ergoCub_hand.xml", "w") as f:
    mjcf_hand_str = ET.tostring(mjcf_hand, encoding="unicode", method="xml")
    f.write(mjcf_hand_str)
