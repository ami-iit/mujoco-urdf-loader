import xml.etree.ElementTree as ET

import pytest

from src.models.mjcf_fcn import (
    add_camera,
    add_joint_eq,
    add_joint_pos_sensor,
    add_joint_vel_sensor,
    add_new_worldbody,
    add_position_actuator,
    separate_left_right_collision_groups,
    set_collision_groups,
    set_joint_damping,
)


def test_add_new_worldbody():
    mjcf = ET.Element("mujoco")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "test")
    mjcf = add_new_worldbody(mjcf)

    assert mjcf is not None
    # check that there is still only one worldbody element
    assert len(mjcf.findall(".//worldbody")) == 1
    # check that there is now a new body element in the worldbody
    assert len(mjcf.findall(".//worldbody/body")) == 1


def test_add_position_actuator():
    mjcf = ET.Element("mujoco")

    joint = ET.SubElement(mjcf, "joint")
    joint.set("name", "joint1")

    mjcf = add_position_actuator(mjcf, joint.attrib["name"])

    assert mjcf is not None
    # check that there is now an actuator element in the mjcf
    assert len(mjcf.findall(".//actuator")) == 1
    # check that there is a position actuator element in the actuator
    assert len(mjcf.findall(".//actuator/position")) == 1
    # check that the actuator is connected to the joint
    assert mjcf.find(".//actuator/position").attrib["joint"] == "joint1"

    # test that adding another actuator does not add another actuator element\
    joint = ET.SubElement(mjcf, "joint")
    joint.set("name", "joint2")

    mjcf = add_position_actuator(mjcf, joint.attrib["name"])

    assert len(mjcf.findall(".//actuator")) == 1
    assert len(mjcf.findall(".//actuator/position")) == 2


def test_add_joint_pos_sensor():
    mjcf = ET.Element("mujoco")

    joint = ET.SubElement(mjcf, "joint")
    joint.set("name", "joint1")

    mjcf = add_joint_pos_sensor(mjcf, joint.attrib["name"])

    assert mjcf is not None
    # check that there is now a sensor element in the mjcf
    assert len(mjcf.findall(".//sensor")) == 1
    # check that there is a joint pos sensor element in the sensor
    assert len(mjcf.findall(".//sensor/jointpos")) == 1
    # check that the sensor is connected to the joint
    assert mjcf.find(".//sensor/jointpos").attrib["joint"] == "joint1"

    # test that adding another sensor does not add another sensor element
    joint = ET.SubElement(mjcf, "joint")
    joint.set("name", "joint2")

    mjcf = add_joint_pos_sensor(mjcf, joint.attrib["name"])

    assert len(mjcf.findall(".//sensor")) == 1
    assert len(mjcf.findall(".//sensor/jointpos")) == 2


def test_add_joint_vel_sensor():
    mjcf = ET.Element("mujoco")

    joint = ET.SubElement(mjcf, "joint")
    joint.set("name", "joint1")

    mjcf = add_joint_vel_sensor(mjcf, joint.attrib["name"])

    assert mjcf is not None
    # check that there is now a sensor element in the mjcf
    assert len(mjcf.findall(".//sensor")) == 1
    # check that there is a joint vel sensor element in the sensor
    assert len(mjcf.findall(".//sensor/jointvel")) == 1
    # check that the sensor is connected to the joint
    assert mjcf.find(".//sensor/jointvel").attrib["joint"] == "joint1"

    # test that adding another sensor does not add another sensor element
    joint = ET.SubElement(mjcf, "joint")
    joint.set("name", "joint2")

    mjcf = add_joint_vel_sensor(mjcf, joint.attrib["name"])

    assert len(mjcf.findall(".//sensor")) == 1
    assert len(mjcf.findall(".//sensor/jointvel")) == 2


def test_add_joint_eq():
    mjcf = ET.Element("mujoco")

    joint1 = ET.SubElement(mjcf, "joint")
    joint1.set("name", "joint1")
    joint2 = ET.SubElement(mjcf, "joint")
    joint2.set("name", "joint2")

    mjcf = add_joint_eq(mjcf, joint1.attrib["name"], joint2.attrib["name"])

    assert mjcf is not None
    # check that there is now an equality element in the mjcf
    assert len(mjcf.findall(".//equality")) == 1
    assert len(mjcf.findall(".//equality/joint")) == 1
    # check that the equality element is connected to the joints
    assert mjcf.find(".//equality/joint").attrib["joint1"] == "joint1"
    assert mjcf.find(".//equality/joint").attrib["joint2"] == "joint2"

    # test that adding another equality does not add another equality element
    joint1 = ET.SubElement(mjcf, "joint")
    joint1.set("name", "joint3")
    joint2 = ET.SubElement(mjcf, "joint")
    joint2.set("name", "joint4")

    mjcf = add_joint_eq(mjcf, joint1.attrib["name"], joint2.attrib["name"])

    assert len(mjcf.findall(".//equality")) == 1
    assert len(mjcf.findall(".//equality/joint")) == 2


def test_add_camera():
    mjcf = ET.Element("mujoco")
    body = ET.SubElement(mjcf, "body")

    add_camera(body)

    # check that there is now a camera element in the mjcf under the body
    assert len(mjcf.findall(".//body/camera")) == 1


def test_set_collision_group():
    mjcf = ET.Element("mujoco")
    geom = ET.SubElement(mjcf, "geom")
    geom.set("mesh", "test")

    mjcf = set_collision_groups(mjcf, idx="test", group=1, affinity=1)

    # check the contype and conaffinity of the geom
    assert mjcf.find(".//geom").attrib["contype"] == "1"
    assert mjcf.find(".//geom").attrib["conaffinity"] == "1"


def test_separate_left_right_collision_groups():
    mjcf = ET.Element("mujoco")
    body = ET.SubElement(mjcf, "body")
    geom = ET.SubElement(body, "geom")
    geom.set("mesh", "_r_test")
    geom = ET.SubElement(body, "geom")
    geom.set("mesh", "_l_test")

    separate_left_right_collision_groups(mjcf, r_group=1, l_group=2, r_aff=1, l_aff=2)

    # check the contype and conaffinity of the geom
    assert mjcf.find(".//geom[@mesh='_r_test']").attrib["contype"] == "1"
    assert mjcf.find(".//geom[@mesh='_r_test']").attrib["conaffinity"] == "1"
    assert mjcf.find(".//geom[@mesh='_l_test']").attrib["contype"] == "2"
    assert mjcf.find(".//geom[@mesh='_l_test']").attrib["conaffinity"] == "2"


def test_set_joint_damping():
    mjcf = ET.Element("mujoco")
    body = ET.SubElement(mjcf, "body")
    joint = ET.SubElement(body, "joint")
    joint.set("name", "test")

    set_joint_damping(mjcf, damping=1.0)

    print(joint.attrib)

    assert joint.attrib["damping"] == "1.0"
    return
