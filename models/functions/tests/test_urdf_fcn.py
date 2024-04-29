import xml.etree.ElementTree as ET

import pytest

from models.functions.urdf_fcn import (
    add_mujoco_element,
    get_joint_limits,
    get_mesh_path,
    get_robot_urdf,
    remove_gazebo_elements,
    remove_links_and_joints_by_keep_list,
    remove_links_and_joints_by_remove_list,
)


def test_get_robot_urdf():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    assert robot_urdf is not None
    assert robot_urdf.tag == "robot"
    assert robot_urdf.attrib["name"] == "ergoCub"
    return


def test_get_mesh_path():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    mesh_path = get_mesh_path(robot_urdf)
    assert mesh_path is not None
    assert mesh_path.is_dir()
    return


def test_remove_gazebo_elements():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    robot_urdf = remove_gazebo_elements(robot_urdf)
    assert robot_urdf is not None
    assert len(robot_urdf.findall(".//gazebo")) == 0
    return


def test_remove_links_and_joints():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    to_remove = [
        "leg",
        "foot",
        "ankle",
        "hip",
        "knee",
        "sole",
    ]
    robot_urdf = remove_links_and_joints_by_remove_list(robot_urdf, to_remove)
    assert robot_urdf is not None

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
    robot_urdf = remove_links_and_joints_by_keep_list(robot_urdf, to_keep)
    assert robot_urdf is not None
    return


def test_add_mujoco_element():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    mesh_path = get_mesh_path(robot_urdf)
    robot_urdf = add_mujoco_element(robot_urdf, mesh_path)
    assert robot_urdf is not None
    assert len(robot_urdf.findall(".//mujoco")) == 1
    return


def test_get_joint_limits():
    robot_urdf = get_robot_urdf("package://ergoCub/robots/ergoCubSN001/model.urdf")
    joint_limits = get_joint_limits(robot_urdf)
    assert joint_limits is not None
    assert len(joint_limits) == 57
    return
