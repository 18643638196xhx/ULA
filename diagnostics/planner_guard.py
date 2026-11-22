"""Scene-only planner guard, importable before any ``envs.*`` import.

The scene-capture scripts must not initialise CuRobo (it JITs and is
irrelevant for a capture that executes no robot action).  Stub the planner
module before the task modules are imported.

Extracted so it can be installed *before* loading helpers that themselves
import ``envs.*`` at module scope -- that ordering is what the previous
``block_planner_import`` could not satisfy.
"""
import sys
import types


def block_planner_import():
    """Replace ``envs.robot.planner`` with a module that cannot plan.

    Raises if the real module is already imported: installing the guard after
    the fact would leave a working planner in place.
    """
    if 'envs.robot.planner' in sys.modules:
        loader = getattr(sys.modules['envs.robot.planner'], '__file__', 'stub')
        if loader and 'planner_guard' not in str(loader):
            raise RuntimeError(
                'Install scene-only planner guard before importing tasks')
    module = types.ModuleType('envs.robot.planner')

    class ForbiddenPlanner:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                'Arm planning forbidden in initial-vision benchmark')

    module.CuroboPlanner = ForbiddenPlanner
    module.MplibPlanner = ForbiddenPlanner
    sys.modules['envs.robot.planner'] = module
    return module
