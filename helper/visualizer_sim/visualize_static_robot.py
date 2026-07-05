"""Interactive viewer for the static Baby robot."""

from pathlib import Path
import sys

import mujoco
from mujoco import viewer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ariel.simulation.environments import SimpleFlatWorld
from blocks.baby_robot import baby_robot


mujoco.set_mjcb_control(None)

world = SimpleFlatWorld()
world.spawn(baby_robot().spec, position=[0, 0, 0])

world.spec.worldbody.add_camera(
    name="video_cam",
    pos=[3.2, -0.5, 1.4],
    xyaxes=[0.1544, 0.988, 0.0, -0.3557, 0.0556, 0.9329],
)

model = world.spec.compile()
data = mujoco.MjData(model)

print(f"Number of actuators (joints): {model.nu}")
print(f"Number of dof: {model.nv}")

viewer.launch(model=model, data=data)
