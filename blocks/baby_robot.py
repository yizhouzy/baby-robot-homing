"""The baby robot configuration."""

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

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