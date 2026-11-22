"""Registry of tasks that share the rule-driven adaptive ULA entry.

Each spec selects a task's base runner (env + approach primitive), its
declarative rule, and the names of the hooks the shared perception and
adaptive placement replace.  The shared pieces are the unified-vision
object/support localization, the adaptive support placement and the
no-trial-lift handoff; only the env, actor names, assets and approach differ.
"""

TASKS = {
    "place_empty_cup": dict(
        env_module="envs.place_empty_cup",
        env_attr="place_empty_cup",
        base="scripts/run_place_empty_cup.py",
        object_actor="cup",
        support_actor="coaster",
        object_prompt="blue cup",
        support_prompt="round coaster",
        object_hints=dict(colors=("blue",), appearance_required=True),
        support_hints=dict(),
        object_asset=dict(model_dir="021_cup", scale=0.088),
        known_cad_worker=(
            "perception_cad_search.pillbottle_cad_search:"
            "PillbottleCadSearchWorker"
        ),
        support_asset=dict(model_dir="019_coaster", scale=0.055),
        instruction="Place the empty blue cup on the round coaster.",
        placement_function="finish_pointcloud_placement",
        trial_lift_function="verify_visual_trial_lift",
        perception_class="ExternalPerception",
        worker_class="PerceptionWorker",
        extra_args=("--approach-mode", "pointcloud", "--grasp-handoff", "visual-trial-lift", "--arm", "auto"),
        demo_approach_prior="rules/approach_priors/place_empty_cup.json",
        rule="rules/place_empty_cup.yaml",
    ),
    "put_object_cabinet": dict(
        env_module="envs.put_object_cabinet", env_attr="put_object_cabinet",
        base_env_module="envs.place_empty_cup", base_env_attr="place_empty_cup",
        base="scripts/run_place_empty_cup.py", object_actor="object", support_actor="cabinet",
        object_prompt="small movable object", support_prompt="cabinet",
        object_prompt_attr="selected_modelname",
        object_asset=dict(model_dir_attr="selected_modelname", model_id_attr="selected_model_id"),
        support_asset=dict(model_dir="036_cabinet", scale=1.0),
        instruction="Put the object in the cabinet.",
        placement_function="finish_pointcloud_placement", trial_lift_function="verify_visual_trial_lift",
        perception_class="ExternalPerception", worker_class="PerceptionWorker",
        extra_args=("--approach-mode", "pointcloud", "--grasp-handoff", "visual-trial-lift", "--arm", "auto"),
        rule="rules/put_object_cabinet.yaml",
    ),
    "move_pillbottle_pad": dict(
        # Same pipeline as place_empty_cup: its base already encapsulates
        # approach -> VLA grasp -> visual trial lift -> placement -> evaluation.
        # Only the task description differs, so reuse it instead of maintaining a
        # second, divergent implementation for this task.
        env_module="envs.move_pillbottle_pad",
        env_attr="move_pillbottle_pad",
        base="scripts/run_place_empty_cup.py",
        object_actor="pillbottle",
        support_actor="pad",
        object_prompt="orange pill bottle",
        support_prompt="blue square pad",
        object_hints=dict(colors=("orange",)),
        support_hints=dict(colors=("blue",)),
        object_asset=dict(model_dir="080_pillbottle", model_id_attr="pillbottle_id", scale=0.08),
        # The pad is a create_box primitive (8x8x1 mm) with no CAD; a generated
        # template lets the vision verify size/planarity instead of guessing.
        support_asset=dict(model_dir="pillbottle_pad", scale=1.0),
        instruction="Pick up the pill bottle and place it onto the blue pad.",
        placement_function="finish_pointcloud_placement",
        trial_lift_function="verify_visual_trial_lift",
        perception_class="ExternalPerception",
        worker_class="PerceptionWorker",
        # Exact per-episode pill-bottle CAD participates in candidate search.
        # The exported 35.7% baseline remains untouched; only the unified task
        # entry selects this successor backend.
        known_cad_worker=(
            "perception_cad_search.pillbottle_cad_search:"
            "PillbottleCadSearchWorker"
        ),
        extra_args=("--approach-mode", "pointcloud", "--grasp-handoff",
                    "visual-trial-lift", "--arm", "auto"),
        demo_approach_prior="rules/approach_priors/move_pillbottle_pad.json",
        rule="rules/move_pillbottle_pad.yaml",
    ),
    "place_fan": dict(
        env_module="envs.place_fan", env_attr="place_fan",
        base_env_module="envs.place_empty_cup", base_env_attr="place_empty_cup",
        base="scripts/run_place_empty_cup.py", object_actor="fan", support_actor="pad",
        object_prompt="electric fan", support_prompt="square pad",
        support_prompt_format="{color_name} square pad",
        object_asset=dict(model_dir="099_fan", model_id_attr="fan_id"),
        known_cad_worker=(
            "perception_cad_search.pillbottle_cad_search:"
            "PillbottleCadSearchWorker"
        ),
        support_asset=dict(model_dir="fan_pad", scale=1.0),
        instruction="Place the fan on the colored pad.",
        placement_function="finish_pointcloud_placement", trial_lift_function="verify_visual_trial_lift",
        perception_class="ExternalPerception", worker_class="PerceptionWorker",
        extra_args=("--approach-mode", "pointcloud", "--grasp-handoff", "visual-trial-lift", "--arm", "auto"),
        demo_approach_prior="rules/approach_priors/place_fan.json",
        rule="rules/place_fan.yaml",
    ),
    "place_mouse_pad": dict(
        env_module="envs.place_mouse_pad", env_attr="place_mouse_pad",
        base_env_module="envs.place_empty_cup", base_env_attr="place_empty_cup",
        base="scripts/run_place_empty_cup.py", object_actor="mouse", support_actor="target",
        object_prompt="computer mouse", support_prompt="rectangular pad",
        support_prompt_format="{color_name} rectangular pad",
        object_asset=dict(model_dir="047_mouse", model_id_attr="mouse_id"),
        known_cad_worker=(
            "perception_cad_search.pillbottle_cad_search:"
            "PillbottleCadSearchWorker"
        ),
        support_asset=dict(model_dir="mouse_pad", scale=1.0),
        instruction="Place the mouse on the colored pad.",
        placement_function="finish_pointcloud_placement", trial_lift_function="verify_visual_trial_lift",
        perception_class="ExternalPerception", worker_class="PerceptionWorker",
        extra_args=("--approach-mode", "pointcloud", "--grasp-handoff", "visual-trial-lift", "--arm", "auto"),
        demo_approach_prior="rules/approach_priors/place_mouse_pad.json",
        rule="rules/place_mouse_pad.yaml",
    ),
    "move_stapler_pad": dict(
        env_module="envs.move_stapler_pad", env_attr="move_stapler_pad",
        base_env_module="envs.place_empty_cup", base_env_attr="place_empty_cup",
        base="scripts/run_place_empty_cup.py", object_actor="stapler", support_actor="pad",
        object_prompt="stapler", support_prompt="rectangular pad",
        support_prompt_format="{color_name} rectangular pad",
        object_asset=dict(model_dir="048_stapler", model_id_attr="stapler_id"),
        known_cad_worker=(
            "perception_cad_search.pillbottle_cad_search:"
            "PillbottleCadSearchWorker"
        ),
        support_asset=dict(model_dir="stapler_pad", scale=1.0),
        instruction="Move the stapler onto the colored pad.",
        placement_function="finish_pointcloud_placement", trial_lift_function="verify_visual_trial_lift",
        perception_class="ExternalPerception", worker_class="PerceptionWorker",
        extra_args=("--approach-mode", "pointcloud", "--grasp-handoff", "visual-trial-lift", "--arm", "auto"),
        demo_approach_prior="rules/approach_priors/move_stapler_pad.json",
        rule="rules/move_stapler_pad.yaml",
    ),
    "place_object_scale": dict(
        env_module="envs.place_object_scale", env_attr="place_object_scale",
        base_env_module="envs.place_empty_cup", base_env_attr="place_empty_cup",
        base="scripts/run_place_empty_cup.py", object_actor="object", support_actor="scale",
        object_prompt="small movable object", support_prompt="kitchen scale|digital scale",
        object_prompt_attr="selected_modelname",
        object_prompt_map={"047_mouse": "computer mouse", "048_stapler": "stapler", "050_bell": "service bell"},
        object_asset=dict(model_dir_attr="selected_modelname", model_id_attr="selected_model_id"),
        support_asset=dict(model_dir="072_electronicscale", model_id_attr="scale_id"),
        instruction="Place the movable object on the electronic scale.",
        placement_function="finish_pointcloud_placement", trial_lift_function="verify_visual_trial_lift",
        perception_class="ExternalPerception", worker_class="PerceptionWorker",
        extra_args=("--approach-mode", "pointcloud", "--grasp-handoff", "visual-trial-lift", "--arm", "auto"),
        demo_approach_prior="rules/approach_priors/place_object_scale.json",
        rule="rules/place_object_scale.yaml",
    ),
    "place_object_stand": dict(
        env_module="envs.place_object_stand", env_attr="place_object_stand",
        base_env_module="envs.place_empty_cup", base_env_attr="place_empty_cup",
        base="scripts/run_place_empty_cup.py", object_actor="object", support_actor="displaystand",
        object_prompt="small movable object", support_prompt="display stand",
        object_prompt_attr="selected_modelname",
        object_prompt_map={"047_mouse": "computer mouse", "048_stapler": "stapler", "050_bell": "service bell",
                           "073_rubikscube": "rubiks cube", "057_toycar": "toy car", "079_remotecontrol": "remote control"},
        object_asset=dict(model_dir_attr="selected_modelname", model_id_attr="selected_model_id"),
        support_asset=dict(model_dir="074_displaystand", model_id_attr="displaystand_id"),
        instruction="Place the movable object on the display stand.",
        placement_function="finish_pointcloud_placement", trial_lift_function="verify_visual_trial_lift",
        perception_class="ExternalPerception", worker_class="PerceptionWorker",
        extra_args=("--approach-mode", "pointcloud", "--grasp-handoff", "visual-trial-lift", "--arm", "auto"),
        demo_approach_prior="rules/approach_priors/place_object_stand.json",
        rule="rules/place_object_stand.yaml",
    ),
    "place_shoe": dict(
        env_module="envs.place_shoe", env_attr="place_shoe",
        base_env_module="envs.place_empty_cup", base_env_attr="place_empty_cup",
        base="scripts/run_place_empty_cup.py", object_actor="shoe", support_actor="target_block",
        object_prompt="shoe", support_prompt="blue rectangular pad",
        support_hints=dict(colors=("blue",)),
        object_asset=dict(model_dir="041_shoe", model_id_attr="shoe_id"),
        known_cad_worker=(
            "perception_cad_search.pillbottle_cad_search:"
            "PillbottleCadSearchWorker"
        ),
        support_asset=dict(model_dir="shoe_pad", scale=1.0),
        instruction="Place the shoe on the blue pad.",
        placement_function="finish_pointcloud_placement", trial_lift_function="verify_visual_trial_lift",
        perception_class="ExternalPerception", worker_class="PerceptionWorker",
        extra_args=("--approach-mode", "pointcloud", "--grasp-handoff", "visual-trial-lift", "--arm", "auto"),
        demo_approach_prior="rules/approach_priors/place_shoe.json",
        rule="rules/place_shoe.yaml",
    ),
    "place_container_plate": dict(
        env_module="envs.place_container_plate", env_attr="place_container_plate",
        base_env_module="envs.place_empty_cup", base_env_attr="place_empty_cup",
        base="scripts/run_place_empty_cup.py", object_actor="container", support_actor="plate",
        object_prompt="container", support_prompt="plate", object_prompt_attr="actor_name",
        object_prompt_map={"002_bowl": "bowl", "021_cup": "cup"},
        object_asset=dict(model_dir_attr="actor_name", model_id_attr="container_id"),
        support_asset=dict(model_dir="003_plate", scale=0.025),
        instruction="Place the container on the plate.",
        placement_function="finish_pointcloud_placement", trial_lift_function="verify_visual_trial_lift",
        perception_class="ExternalPerception", worker_class="PerceptionWorker",
        extra_args=("--approach-mode", "pointcloud", "--grasp-handoff", "visual-trial-lift", "--arm", "auto"),
        demo_approach_prior="rules/approach_priors/place_container_plate.json",
        rule="rules/place_container_plate.yaml",
    ),
}
