from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("touch")
@dataclass
class TouchTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True
    device_name: str = "right"
    # Scales mm-delta to [-1, 1] action range. 0.2 means 5 mm stylus movement → action 1.0.
    position_scale: float = 0.2
    # Roll (rx) channel sensitivity multiplier.  < 1 reduces; 1.0 = 1:1 radian mapping.
    roll_scale: float = 0.3
    haptic_module_path: str = ""
    # True → movement always active, Button 1 = clutch (for record / manual control).
    # False → Button 1 = intervention trigger (for HIL-SERL training).
    clutch_mode: bool = True
