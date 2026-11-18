"""NWaves SO-101 Sim To Real Hackathon: one task for Isaac Lab 3.0, put the cube on the tray.

Importing this package registers the gym ids:

    NWaves-SO101-PickPlace-v0            the official task (joint-position actions)
    NWaves-SO101-PickPlace-Eval-v0       the scorer's copy of it
    NWaves-SO101-PickPlace-Kit-v0        fingertip-pose actions: generator, keyboard teleop
    NWaves-SO101-PickPlace-Kit-Mimic-v0  the kit environment for Isaac Lab Mimic
"""
import gymnasium as gym

__version__ = "0.3.0"

for _suffix, _cfg in (("", "PickPlaceEnvCfg"), ("-Eval", "PickPlaceEvalEnvCfg"), ("-Kit", "PickPlaceKitEnvCfg")):
    gym.register(
        id=f"NWaves-SO101-PickPlace{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{_cfg}"},
    )

gym.register(
    id="NWaves-SO101-PickPlace-Kit-Mimic-v0",
    entry_point=f"{__name__}.mimic_env:NWavesSO101MimicEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:PickPlaceKitMimicEnvCfg"},
)
