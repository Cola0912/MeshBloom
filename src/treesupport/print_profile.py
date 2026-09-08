"""Extrusion dimensions used to size solids; these are not G-code settings."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PrintProfile:
    nozzle_diameter: float = .4
    line_width: float = .4

    def __post_init__(self):
        if not math.isfinite(self.nozzle_diameter) or self.nozzle_diameter <= 0:
            raise ValueError("ノズル径は0より大きい有限の数値にしてください。")
        if not math.isfinite(self.line_width) or self.line_width < 0:
            raise ValueError("ライン幅は0以上の有限の数値にしてください。0で自動設定します。")

    @property
    def width(self):
        return self.line_width or self.nozzle_diameter * 1.125

    def dimensions(self):
        width = self.width
        return {"tip_diameter": 2*width, "contact_diameter": width,
                "branch_diameter": 5*width, "branch_diameter_max": 20*width,
                "root_diameter_min": 8*width, "contact_height": 2*self.nozzle_diameter,
                "wall_thickness": 3*width, "shell_thickness": 3*width,
                "pitch": width}
