"""Open a MuJoCo viewer for the Baby robot and the charging-station target."""

# ============================= Imports ============================= #
# Third-party libraries
from optparse import Option

import mujoco
import numpy as np
from mujoco import viewer
from pathlib import Path
import sys

# Controller imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_control.controllers import load_gait_network
from robot_control.artifacts import infer_meta_path

# ARIEL — robot bodies
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
# ARIEL — controller
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import (
    NaCPG,
    create_fully_connected_adjacency,
)
# ARIEL — simulation environments
from ariel.simulation.environments import SimpleFlatWorld, OlympicArena, SimpleTiltedWorld
# ARIEL — utilities
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
# Robot configuration
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

# ============================= Create new world =========================== #
# Always reset the control callback before building a new simulation.
mujoco.set_mjcb_control(None)

# Create the world
world = SimpleFlatWorld()

# ============================== Baby Robot ============================== #
# Create the robot body
def baby_robot() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    #core
    core = CoreModule(index=0)

    # ------ right ------ #
    # hinge at visual right (label left), top-down, connect to core
    hinge_0 = HingeModule(index=1)
    hinge_0.rotate(90)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    # hinge at visual right, left-right, connect to hinge0
    hinge_1 = HingeModule(index=3)
    hinge_1.rotate(90)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    # brick at visual right, connect to hinge1
    brick_0 = BrickModule(index=4)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    # hinge at visual right, top-down, connect to brick0
    hinge_2 = HingeModule(index=8)
    hinge_2.rotate(90)
    brick_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    # brick at visual right, connect to hinge2 (right forefoot)
    brick_1 = BrickModule(index=9)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )

    # ------ left ------ #
    # hinge at visual left (label right), top-down, connect to core
    hinge_3 = HingeModule(index=10)
    hinge_3.rotate(90)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    # brick at visual left, connect to hinge3 (left forefoot)
    brick_2 = BrickModule(index=11)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )

    # ----- back ----- #
    # hinge at visual back, left-right, connect to core
    hinge_4 = HingeModule(index=21)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    # brick at visual back, connect to hinge4
    brick_3 = BrickModule(index=22)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    # hinge at visual back, top-down, connect to brick3
    hinge_5 = HingeModule(index=23)
    brick_3.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    # brick at visual back, connect to hinge5 
    brick_4 = BrickModule(index=24)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    # hinge at visual back, relative visual right, top-down, connect to brick4
    hinge_6 = HingeModule(index=26)
    hinge_6.rotate(90)
    brick_4.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    # brick at visual back, relative visual right, connect to hinge6 (right hind foot)
    brick_5 = BrickModule(index=27)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    # hinge at visual back, relative visual left, top-down, connect to brick4
    hinge_7 = HingeModule(index=28)
    hinge_7.rotate(90)
    brick_4.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    # brick at visual back, relative visual left, connect to hinge7 (left hind foot)
    brick_6 = BrickModule(index=29)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_6.body,
        prefix="brick_6",
    )

    return core



# =============================== Sprawn robots ============================== #

# Spawn the robot at the origin
baby_core = baby_robot()
world.spawn(baby_core.spec,position=[0,0,0.1])

# # Gecko for testing
# gecko_core = gecko()
# world.spawn(gecko_core.spec, position=[0, 1, 0.1])

# # Spider for testing
# spider_core = spider()
# world.spawn(spider_core.spec, position=[0, -1, 0.1])

# =============================== Target ============================== #
# Charging station
target_pos=[1.0, -2.0, 0.01] # minus y = robot's visual front

target_body = world.spec.worldbody.add_body(name="charging_station",
                                                mocap=True,
                                                pos=target_pos)

base_half_width = 0.125
base_half_height = 0.0125
lower_collar_radius = 0.105
upper_collar_radius = 0.085
cone_bottom_z = 0.07
cone_height = 0.40
bottom_radius = 0.095
top_radius = 0.025

# Base platform
target_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[base_half_width, base_half_width, base_half_height],
    pos=[0, 0, base_half_height],  # x, y, z
    rgba=[0.02, 0.025, 0.02, 1.0], # dark gray base
)

target_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[lower_collar_radius, 0.015],
    pos=[0, 0, 0.04],
    rgba=[0.035, 0.04, 0.035, 1.0],
)

target_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[upper_collar_radius, 0.008],
    pos=[0, 0, 0.062],
    rgba=[0.015, 0.018, 0.015, 1.0],
)

# Cone body
segments = 24
segment_height = cone_height / segments

for i in range(segments):
    z = cone_bottom_z + (i + 0.5) * segment_height
    t = (z - cone_bottom_z) / cone_height
    radius = bottom_radius + (top_radius - bottom_radius) * t

    if 0.17 <= t <= 0.32 or 0.52 <= t <= 0.66:
        rgba = [0.94, 0.94, 0.88, 1.0]
    else:
        rgba = [1.0, 0.12, 0.04, 1.0]

    target_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[radius, segment_height / 2],
        pos=[0, 0, z],
        rgba=rgba,
    )

target_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[0.028, 0.003, 0.012],
    pos=[0, -0.086, 0.17],
    rgba=[0.95, 0.92, 0.86, 1.0],
)
target_body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[0.018, 0.004, 0.006],
    pos=[0, -0.089, 0.17],
    rgba=[0.45, 0.45, 0.42, 1.0],
)

# =============================== Add a camera ============================== #
world.spec.worldbody.add_camera(
        name="video_cam",
        # pos: camera's x, y, z position in world coordinates
        # (minus y = forward to the robot's travel direction)
        pos=[3.2, -0.5, 1.4],
        xyaxes=[0.1544, 0.988, 0.0, -0.3557, 0.0556, 0.9329],

        # xyaxes: camera's local X and Y axes in world coordinates
        # xyaxes = [x_axis_x, x_axis_y, x_axis_z, 
        #           y_axis_x, y_axis_y, y_axis_z]
        # x_axis: which world direction appears horizontally across the image.
        # y_axis: which world direction appears vertically across the image.

    )

# =============================== Mujoco Setup ============================== #
# Compile into a MuJoCo model and initialization data
model = world.spec.compile()
data = mujoco.MjData(model)

print(f"Number of actuators (joints): {model.nu}")
print(f"Number of dof: {model.nv}")

# =============================== Basic Controller Setup ============================== #

# Option 1: Build the CPG — one node per actuator, fully connected.
# adj_dict = create_fully_connected_adjacency(model.nu)
# cpg = NaCPG(
#     adjacency_dict=adj_dict,
#     hard_bounds=(-np.pi / 2, np.pi / 2),  # keep angles within hinge limits
# )
# print(f"CPG nodes (= actuators): {cpg.n}")
# print(f"Total learnable parameters: {cpg.num_of_parameters}")

# Option 2:Use evolved controller
weights_path = Path(
    # "results/cmaes/20260609_132427_778702_seed42/gait_best_20260609_132427_778702_seed42.npy"
    "results/gait_cpg/20260627_234215_113631_seed43_DR/gait_best_20260627_234215_113631_seed43_DR.npy"
)

try:
    meta_path = infer_meta_path(weights_path, None)
except FileNotFoundError:
    meta_path = None

gait_net, meta, weight_format = load_gait_network(weights_path, meta_path)

print(f"Loaded gait controller: {weight_format}")
print(f"num_joints={int(meta['num_joints'])}, dt={float(meta['dt']):.4f}")

# ============================== Tracker Setup ============================== #
# Set up the tracker — Controller will call tracker.update() automatically.
tracker = Tracker(
    mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
    name_to_bind="core",
    observable_attributes=["xpos"],
)
tracker.setup(world.spec, data)

# Wrap the CPG: the callback receives (model, data) and returns joint angles.
# Option 1: new CPG
# def cpg_callback(model: mujoco.MjModel, data: mujoco.MjData):
#     return cpg.forward(time=data.time)

# Option 2: evolved CPG
def cpg_callback(model: mujoco.MjModel, data: mujoco.MjData):
    return gait_net.forward(turn=0.0, speed=1.0)

# ============================== Controller Registration ============================== #
# Create the Controller.
controller = Controller(
    controller_callback_function=cpg_callback,
    time_steps_per_ctrl_step=25,   # call CPG every 50 physics steps
    time_steps_per_save=500,        # record tracker data every 500 steps
    alpha=1.0,                      # smoothing: 0 = never update, 1 = immediate update
    tracker=tracker,
)

# Register with MuJoCo — controller.set_control IS the callback.
mujoco.set_mjcb_control(controller.set_control)

print("Controller registered.")

# ============================== Viewer Setup ============================== #
# Open the interactive viewer
mujoco.set_mjcb_control(controller.set_control)
viewer.launch(model=model, data=data)
