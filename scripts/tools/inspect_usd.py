# inspect_usd.py
import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from pxr import Usd, UsdPhysics, UsdGeom


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUR_DRONE_DIR = os.path.join(_THIS_DIR, "..", "..", "source", "isaaclab_assets", "data", "Robots", "ourdrone")

omni.usd.get_context().open_stage(
    os.path.join(_OUR_DRONE_DIR, "ourdrone.usd")
)
stage = omni.usd.get_context().get_stage()

print("=" * 60)
print("PRIM HIERARCHY:")
for prim in stage.Traverse():
    apis = []
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        apis.append("ArticulationRootAPI")
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        apis.append("RigidBodyAPI")
    if prim.IsInstance():
        apis.append("INSTANCE")
    
    type_name = prim.GetTypeName()
    api_str = f"  [{', '.join(apis)}]" if apis else ""
    print(f"  {prim.GetPath()}  ({type_name}){api_str}")

print("=" * 60)
print(f"\nDefault prim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else 'NONE'}")
print("=" * 60)

simulation_app.close()
