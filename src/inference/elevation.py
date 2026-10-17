"""Stream a physical-elevation constraint on the final prediction grid."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


class ElevationConstraint:
    """Restrict complete class branches to DEM > a threshold in metres."""

    def __init__(self, config: dict[str, Any], mapping: dict[str, Any], grid):
        names = config.get("classes", [])
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
        ):
            raise ValueError("elevation_constraint.classes 必须是非空类别名称列表")
        self.class_codes: set[int] = set()
        unmatched = set(names)
        for item in mapping.get("classes", []):
            # Match an entire hierarchy node, including all descendant leaves.
            aliases = {
                *item.get("level_names", []),
                *item.get("level_names_zh", []),
                *(
                    item.get(key, "")
                    for key in ("formation", "formation_zh", "alliance", "alliance_zh")
                ),
            }
            matched = set(names) & aliases
            if matched:
                self.class_codes.add(int(item["alliance_code"]))
                unmatched -= matched
        if unmatched:
            raise ValueError(
                f"海拔约束类别未匹配 checkpoint 类别映射: {sorted(unmatched)}"
            )
        self.minimum = float(config.get("min_elevation_m", 2800))
        if not np.isfinite(self.minimum):
            raise ValueError("min_elevation_m 必须是有限数值")
        self.missing_dem = config.get("missing_dem", "nodata")
        if self.missing_dem not in {"nodata", "exclude_restricted"}:
            raise ValueError("missing_dem 只能是 nodata 或 exclude_restricted")
        method = config.get("resampling", "bilinear")
        if method not in {"nearest", "bilinear", "average"}:
            raise ValueError("DEM resampling 只能是 nearest、bilinear 或 average")
        self.resampling = Resampling[method]
        if not config.get("dem_path"):
            raise ValueError("启用海拔约束时必须配置 dem_path")
        self.path = Path(config["dem_path"]).resolve()
        self.grid = grid
        # Validate the source before expensive model inference starts. Rasterio
        # reads raw DN; apply the file's scale/offset explicitly after warping.
        with rasterio.open(self.path) as source:
            if source.count != 1 or source.crs is None:
                raise ValueError("约束 DEM 必须是带 CRS 的单波段栅格，物理单位为米")
            self.scale, self.offset = source.scales[0], source.offsets[0]
            if not np.isfinite([self.scale, self.offset]).all() or self.scale <= 0:
                raise ValueError("DEM scale 必须为有限正数，offset 必须为有限数值")
            self.source_nodata = (
                source.nodata
                if source.nodata is not None
                else config.get("dem_nodata", -9999)
            )

    @contextmanager
    def open(self):
        """Use one global VRT so row-block boundaries cannot change alignment."""
        with (
            rasterio.open(self.path) as source,
            WarpedVRT(
                source,
                crs=self.grid.crs,
                transform=self.grid.transform,
                width=self.grid.width,
                height=self.grid.height,
                src_nodata=self.source_nodata,
                nodata=np.nan,
                dtype="float32",
                resampling=self.resampling,
            ) as reader,
        ):
            yield reader

    def read(self, reader, window):
        """Return metres, allowed restricted pixels and DEM-invalid pixels."""
        elevation = reader.read(1, window=window, masked=True).filled(np.nan)
        elevation = elevation * self.scale + self.offset
        missing = ~np.isfinite(elevation)
        allowed = ~missing & (elevation > self.minimum)
        return elevation, allowed, missing

    def tags(self) -> dict[str, str]:
        return {
            "elevation_constraint": "DEM > min_elevation_m",
            "dem_path": str(self.path),
            "min_elevation_m": str(self.minimum),
            "restricted_class_codes": ",".join(map(str, sorted(self.class_codes))),
            "dem_resampling": self.resampling.name,
            "missing_dem": self.missing_dem,
            "constraint_priority": "hard_constraint_before_ground_truth_override",
        }
