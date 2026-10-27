"""Physically grounded night / headlight simulation for forward-facing rail cameras.

The model assumes a calibrated pinhole camera above a flat rail plane (as shipped
with Labels4Rail ``camera.yaml``) and headlights mounted on the train front, below
and slightly ahead of the camera. Lamps use Gaussian beam lobes with a sharp
upper cut-off, inverse-square falloff, and a Lambert cosine term on ground
(up-facing) and vertical (viewer-facing) surfaces. Occlusion shadows are not
modelled. The render is a physics *prior*: a geometry-exact reference for the
light layout that prompts and generated images can be checked against.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


PROMPT_SLOTS = (
    "task",
    "scene",
    "darkness",
    "light_source",
    "beam",
    "falloff",
    "unlit",
    "camera",
    "preserve",
)
# The rewriter may not drop the edit task or the geometry-preservation contract.
LOCKED_SLOTS = ("task", "preserve")


@dataclass(frozen=True)
class RailCamera:
    """Pinhole camera above a flat rail plane, in a train-aligned frame.

    Frame: x right, y down, z forward (metres), camera at the origin, rail plane
    at ``y = height_m``. Angles use the convention validated on L4R_NLB: the YAML
    stores pitch/yaw with the opposite sign, so ``from_opencv_yaml`` negates them.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    pitch_down_deg: float
    yaw_deg: float
    height_m: float

    @classmethod
    def from_opencv_yaml(cls, path: Path | str) -> RailCamera:
        """Parse a Labels4Rail OpenCV-style ``camera.yaml``."""
        text = Path(path).read_text(encoding="utf-8")

        def scalar(key: str) -> float:
            match = re.search(rf"^{key}:\s*([-+0-9.eE]+)", text, flags=re.MULTILINE)
            if match is None:
                raise ValueError(f"{key!r} missing in {path}")
            return float(match.group(1))

        def matrix(key: str) -> list[float]:
            match = re.search(
                rf"^{key}:.*?data:\s*\[([^\]]*)\]",
                text,
                flags=re.MULTILINE | re.DOTALL,
            )
            if match is None:
                raise ValueError(f"{key!r} missing in {path}")
            return [float(value) for value in match.group(1).split(",")]

        if abs(scalar("roll")) > 1e-9:
            raise ValueError("Non-zero camera roll is not supported")
        k = matrix("camera_matrix")
        tvec = matrix("tvec")
        return cls(
            width=int(scalar("width")),
            height=int(scalar("height")),
            fx=k[0],
            fy=k[4],
            cx=k[2],
            cy=k[5],
            pitch_down_deg=-scalar("pitch"),
            yaw_deg=-scalar("yaw"),
            height_m=abs(tvec[1]) / 1000.0,
        )

    def resized(self, width: int, height: int) -> RailCamera:
        """Return intrinsics for a resized image (pixel-centre convention)."""
        sx, sy = width / self.width, height / self.height
        return replace(
            self,
            width=int(width),
            height=int(height),
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
        )

    @property
    def horizon_row(self) -> float:
        """Image row of the rail-plane horizon."""
        return self.cy - self.fy * math.tan(math.radians(self.pitch_down_deg))

    def ray_directions(
        self, u: np.ndarray | float, v: np.ndarray | float
    ) -> np.ndarray:
        """Unnormalised ray directions (..., 3) in the train frame."""
        u_arr = np.asarray(u, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        x = (u_arr - self.cx) / self.fx
        y = (v_arr - self.cy) / self.fy
        z = np.ones(np.broadcast(x, y).shape)
        pitch = math.radians(self.pitch_down_deg)
        yaw = math.radians(self.yaw_deg)
        y1 = y * math.cos(pitch) + z * math.sin(pitch)
        z1 = -y * math.sin(pitch) + z * math.cos(pitch)
        x2 = x * math.cos(yaw) + z1 * math.sin(yaw)
        z2 = -x * math.sin(yaw) + z1 * math.cos(yaw)
        return np.stack(np.broadcast_arrays(x2, y1, z2), axis=-1)

    def pixel_to_ground(
        self, u: np.ndarray | float, v: np.ndarray | float
    ) -> np.ndarray:
        """Rail-plane intersection (..., 3); NaN at or above the horizon."""
        rays = self.ray_directions(u, v)
        down = rays[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(down > 1e-9, self.height_m / down, np.nan)
        return np.asarray(rays * scale[..., None], dtype=np.float64)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project train-frame points (..., 3) to pixels (..., 2)."""
        pts = np.asarray(points, dtype=np.float64)
        pitch = math.radians(self.pitch_down_deg)
        yaw = math.radians(self.yaw_deg)
        x2, y1, z2 = pts[..., 0], pts[..., 1], pts[..., 2]
        x = x2 * math.cos(yaw) - z2 * math.sin(yaw)
        z1 = x2 * math.sin(yaw) + z2 * math.cos(yaw)
        y = y1 * math.cos(pitch) - z1 * math.sin(pitch)
        z = y1 * math.sin(pitch) + z1 * math.cos(pitch)
        return np.stack([self.fx * x / z + self.cx, self.fy * y / z + self.cy], axis=-1)


# --- annotations -------------------------------------------------------------


def track_polylines(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return tracks as ``{"position", "left", "right"}`` with (N, 2) xy arrays."""
    out = []
    for track in (annotation.get("tracks") or {}).values():
        rails = {}
        for side in ("left", "right"):
            points = (track.get(f"{side} rail") or {}).get("points") or []
            rails[side] = np.array(
                [(float(p["x"]), float(p["y"])) for p in points], dtype=np.float64
            ).reshape(-1, 2)
        out.append(
            {
                "position": str(track.get("relative position", "unknown")),
                "left": rails["left"],
                "right": rails["right"],
            },
        )
    return out


def ego_track(annotation: dict[str, Any]) -> dict[str, Any] | None:
    """Return the annotated ego track if it has both rails."""
    for track in track_polylines(annotation):
        if (
            track["position"] == "ego"
            and len(track["left"]) >= 2
            and len(track["right"]) >= 2
        ):
            return track
    return None


def _x_at_row(polyline: np.ndarray, row: float) -> float:
    order = np.argsort(polyline[:, 1])
    ys, xs = polyline[order, 1], polyline[order, 0]
    if not ys[0] <= row <= ys[-1]:
        return float("nan")
    return float(np.interp(row, ys, xs))


def rail_gauge_check(
    annotation: dict[str, Any],
    camera: RailCamera,
    rows: tuple[int, ...] = (1050, 950, 850, 750, 680, 630),
) -> list[dict[str, float]]:
    """Measure ego-track gauge on the rail plane at several image rows.

    A correct calibration gives ~1.5 m (1435 mm gauge + rail-head width) at all
    distances; drift with distance flags a wrong pitch, height, or yaw sign.
    """
    track = ego_track(annotation)
    if track is None:
        return []
    out = []
    for row in rows:
        xl = _x_at_row(track["left"], row)
        xr = _x_at_row(track["right"], row)
        if math.isnan(xl) or math.isnan(xr):
            continue
        left = camera.pixel_to_ground(xl, row)
        right = camera.pixel_to_ground(xr, row)
        if not (np.isfinite(left).all() and np.isfinite(right).all()):
            continue
        out.append(
            {
                "row": float(row),
                "distance_m": float((left[2] + right[2]) / 2),
                "gauge_m": float(right[0] - left[0]),
                "centre_x_m": float((left[0] + right[0]) / 2),
            },
        )
    return out


def ego_centre_x(annotation: dict[str, Any], camera: RailCamera) -> float:
    """Lateral offset of the ego-track centre line near the train (metres)."""
    rows = [r for r in rail_gauge_check(annotation, camera) if r["distance_m"] < 15]
    if not rows:
        return 0.0
    return float(np.median([r["centre_x_m"] for r in rows]))


def rail_mask(
    annotation: dict[str, Any],
    camera: RailCamera,
    size: tuple[int, int],
    *,
    rail_width_m: float = 0.07,
) -> np.ndarray:
    """Perspective-correct rail-head mask (H, W) in [0, 1] from annotations.

    ``camera`` must describe the annotation's pixel grid (original resolution);
    the mask is drawn there and resized to ``size`` (width, height).
    """
    canvas = Image.new("L", (camera.width, camera.height), 0)
    draw = ImageDraw.Draw(canvas)
    for track in track_polylines(annotation):
        for side in ("left", "right"):
            pts = track[side]
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                ground = camera.pixel_to_ground((x0 + x1) / 2, (y0 + y1) / 2)
                if not np.isfinite(ground).all() or ground[2] <= 0:
                    continue
                width = max(1, min(8, round(camera.fx * rail_width_m / ground[2])))
                draw.line([(x0, y0), (x1, y1)], fill=255, width=int(width))
    resized = canvas.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


# --- geometry ----------------------------------------------------------------


@dataclass
class SceneGeometry:
    """Per-pixel 3D points and surface normals in the train frame."""

    points: np.ndarray  # (H, W, 3) metres
    normals: np.ndarray  # (H, W, 3) unit vectors
    ground: np.ndarray  # (H, W) bool
    sky: np.ndarray  # (H, W) bool
    depth_fit: dict[str, float]


def build_scene_geometry(
    camera: RailCamera,
    disparity: np.ndarray | None = None,
    track_mask: np.ndarray | None = None,
    *,
    sky_mask: np.ndarray | None = None,
    max_range_m: float = 400.0,
    far_sky_m: float = 150.0,
    object_ratio: float = 0.8,
) -> SceneGeometry:
    """Combine the calibrated rail plane with metric-aligned monocular depth.

    ``disparity`` is relative inverse depth (near = high), e.g. Depth Anything,
    at ``camera`` resolution. Its unknown scale/shift is fitted against the
    rail-plane depth on ``track_mask`` pixels, where the plane model is exact.
    Below-horizon pixels nearer than ``object_ratio`` × plane depth are treated
    as vertical objects; everything else below the horizon is ground.
    Monocular depth gives sky a finite distance, so sky comes from
    ``sky_mask`` (e.g. semantic segmentation) plus above-horizon pixels
    farther than ``far_sky_m``.
    """
    h, w = camera.height, camera.width
    v, u = np.mgrid[0:h, 0:w]
    rays = camera.ray_directions(u, v)
    ground_pts = camera.pixel_to_ground(u, v)
    plane_z = ground_pts[..., 2]
    below = np.isfinite(plane_z) & (plane_z > 0)

    points = np.where(below[..., None], ground_pts, np.nan)
    normals: np.ndarray = np.zeros((h, w, 3))
    normals[..., 1] = -1.0
    ground = below.copy()
    sky = ~below
    fit = {"a": float("nan"), "b": float("nan"), "n": 0.0, "rmse": float("nan")}

    if disparity is not None:
        if disparity.shape != (h, w):
            raise ValueError(f"disparity shape {disparity.shape} != camera {(h, w)}")
        sel = below & (plane_z > 8) & (plane_z < 80)
        if track_mask is not None:
            sel &= track_mask > 0.5
        x = 1.0 / plane_z[sel]
        y = disparity[sel].astype(np.float64)
        if x.size < 50:
            raise ValueError("Too few ground pixels to align monocular depth")
        keep = np.ones_like(x, dtype=bool)
        a = b = 0.0
        for _ in range(3):
            a, b = np.polyfit(x[keep], y[keep], 1)
            resid = np.abs(y - (a * x + b))
            keep = resid <= np.quantile(resid, 0.8)
        if a <= 0:
            raise ValueError("Depth alignment failed (non-positive scale)")
        rmse = float(np.sqrt(np.mean((y[keep] - (a * x[keep] + b)) ** 2)))
        fit = {"a": float(a), "b": float(b), "n": float(keep.sum()), "rmse": rmse}
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = (disparity.astype(np.float64) - b) / a
            mono_z = np.where(inv > 1.0 / max_range_m, 1.0 / inv, np.inf)
        obj = np.isfinite(mono_z) & (~below | (mono_z < object_ratio * plane_z))
        obj_pts = rays * (mono_z / rays[..., 2])[..., None]
        points = np.where(obj[..., None], obj_pts, points)
        horiz = np.stack([rays[..., 0], np.zeros((h, w)), rays[..., 2]], axis=-1)
        horiz /= np.linalg.norm(horiz, axis=-1, keepdims=True)
        normals = np.where(obj[..., None], -horiz, normals)
        ground = below & ~obj
        sky = ~below & ~(obj & (mono_z < far_sky_m))
    if sky_mask is not None:
        if sky_mask.shape != (h, w):
            raise ValueError(f"sky_mask shape {sky_mask.shape} != camera {(h, w)}")
        sky = sky | sky_mask.astype(bool)
    ground &= ~sky
    points = np.where(sky[..., None], np.nan, points)
    return SceneGeometry(
        points=points, normals=normals, ground=ground, sky=sky, depth_fit=fit
    )


# --- lighting ----------------------------------------------------------------


@dataclass(frozen=True)
class HeadlightRig:
    """Train-front lamps below the cab camera (defaults: typical locomotive)."""

    lamp_height_m: float = 1.0
    lamp_spacing_m: float = 1.3
    lamp_forward_m: float = 2.0
    centre_x_m: float = 0.0
    aim_down_deg: float = 0.5
    sigma_h_deg: float = 7.0
    sigma_up_deg: float = 1.2
    sigma_down_deg: float = 6.0
    spill: float = 0.03
    spill_sigma_deg: float = 25.0

    def lamp_positions(self, camera: RailCamera) -> np.ndarray:
        """Lamp centres (n, 3) in the train frame."""
        y = camera.height_m - self.lamp_height_m
        if y <= 0:
            raise ValueError("Lamps must be mounted below the camera")
        half = self.lamp_spacing_m / 2
        return np.array(
            [
                [self.centre_x_m - half, y, self.lamp_forward_m],
                [self.centre_x_m + half, y, self.lamp_forward_m],
            ],
        )

    def irradiance(
        self, points: np.ndarray, normals: np.ndarray, camera: RailCamera
    ) -> np.ndarray:
        """Irradiance from all lamps (arbitrary units) at points with normals."""
        total = np.zeros(points.shape[:-1])
        aim = math.radians(self.aim_down_deg)
        sh = math.radians(self.sigma_h_deg)
        su = math.radians(self.sigma_up_deg)
        sd = math.radians(self.sigma_down_deg)
        ss = math.radians(self.spill_sigma_deg)
        for lamp in self.lamp_positions(camera):
            d = points - lamp
            r = np.linalg.norm(d, axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                l_dir = d / r[..., None]
                az = np.arctan2(l_dir[..., 0], l_dir[..., 2])
                el = (
                    np.arctan2(l_dir[..., 1], np.hypot(l_dir[..., 0], l_dir[..., 2]))
                    - aim
                )
                sv = np.where(el >= 0, sd, su)
                lobe = np.exp(-0.5 * (az / sh) ** 2 - 0.5 * (el / sv) ** 2)
                lobe += self.spill * np.exp(-0.5 * (az**2 + el**2) / ss**2)
                lobe = np.where(l_dir[..., 2] > 0, lobe, 0.0)
                cos_inc = np.clip(-np.sum(normals * l_dir, axis=-1), 0.0, None)
                e = lobe * cos_inc / (r**2)
            total += np.nan_to_num(e, nan=0.0, posinf=0.0)
        return total

    def reference_irradiance(
        self, camera: RailCamera, distance_m: float = 15.0
    ) -> float:
        """Irradiance on the track centre ``distance_m`` ahead (normaliser)."""
        point = np.array([[self.centre_x_m, camera.height_m, distance_m]])
        normal = np.array([[0.0, -1.0, 0.0]])
        return float(self.irradiance(point, normal, camera)[0])

    def beam_reach_m(
        self, camera: RailCamera, *, threshold: float = 0.05, max_m: float = 300.0
    ) -> float:
        """Distance where track-centre irradiance drops below ``threshold`` × 15 m."""
        ref = self.reference_irradiance(camera)
        dist = np.linspace(3.0, max_m, 1200)
        pts = np.stack(
            [
                np.full_like(dist, self.centre_x_m),
                np.full_like(dist, camera.height_m),
                dist,
            ],
            axis=-1,
        )
        normals = np.tile([0.0, -1.0, 0.0], (dist.size, 1))
        rel = self.irradiance(pts, normals, camera) / ref
        beyond = np.nonzero((dist > 15) & (rel < threshold))[0]
        return float(dist[beyond[0]]) if beyond.size else max_m


def normalized_irradiance(
    geometry: SceneGeometry, rig: HeadlightRig, camera: RailCamera
) -> np.ndarray:
    """Irradiance map (H, W) relative to the track centre at 15 m (= 1.0)."""
    e = rig.irradiance(geometry.points, geometry.normals, camera)
    e[geometry.sky] = 0.0
    return (e / rig.reference_irradiance(camera)).astype(np.float32)


# --- rendering ---------------------------------------------------------------


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Decode sRGB values in [0, 1] to linear light."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Encode linear light in [0, 1] as sRGB."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def kelvin_to_rgb(kelvin: float) -> np.ndarray:
    """Approximate blackbody colour (Tanner Helland fit), max channel = 1."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = (
        99.4708025861 * math.log(t) - 161.1195681661
        if t <= 66
        else 288.1221695283 * (t - 60) ** -0.0755148492
    )
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    rgb = np.clip(np.array([r, g, b]) / 255.0, 0.0, 1.0)
    return srgb_to_linear(rgb) / max(float(srgb_to_linear(rgb).max()), 1e-6)


@dataclass(frozen=True)
class NightLook:
    """Exposure, ambient, and sensor parameters for the physics render."""

    lamp_kelvin: float = 3600.0
    exposure: float = 0.45
    ambient: float = 0.006
    ambient_tint: tuple[float, float, float] = (0.55, 0.68, 1.0)
    ambient_saturation: float = 0.3
    sky_rgb: tuple[float, float, float] = (0.0015, 0.0022, 0.0045)
    horizon_glow: float = 0.0
    rail_glint: float = 0.8
    read_noise: float = 0.004
    shot_noise: float = 0.002
    blur_px: float = 0.6
    seed: int = 0


def tone_map(x: np.ndarray, white: float = 4.0) -> np.ndarray:
    """Apply an extended Reinhard tone curve (linear → linear display)."""
    return np.asarray(x * (1 + x / white**2) / (1 + x), dtype=np.float64)


def render_night(
    image: Image.Image,
    geometry: SceneGeometry,
    irradiance: np.ndarray,
    look: NightLook | None = None,
    *,
    rails: np.ndarray | None = None,
) -> Image.Image:
    """Relight a dusk/day frame as a headlight-lit night frame.

    Linear source pixels act as an albedo proxy (diffuse dusk light), scaled so
    the 98th luminance percentile of non-sky pixels maps to 0.8.
    """
    look = look or NightLook()
    h, w = irradiance.shape
    src = np.asarray(image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS))
    lin = srgb_to_linear(src.astype(np.float64) / 255.0)
    luma = lin @ np.array([0.2126, 0.7152, 0.0722])
    ref = np.quantile(luma[~geometry.sky], 0.98) if (~geometry.sky).any() else 1.0
    albedo = np.clip(lin * (0.8 / max(ref, 1e-6)), 0.0, 1.0)

    lamp = kelvin_to_rgb(look.lamp_kelvin)
    out = albedo * irradiance[..., None] * lamp * look.exposure
    grey = (albedo @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
    ambient_albedo = grey + look.ambient_saturation * (albedo - grey)
    out += ambient_albedo * look.ambient * np.array(look.ambient_tint)
    if rails is not None:
        glint = rails * np.sqrt(np.clip(irradiance, 0.0, 4.0))
        out += glint[..., None] * lamp * look.rail_glint * look.exposure

    rows = np.arange(h, dtype=np.float64)[:, None]
    horizon = max(
        1.0, float(np.max(np.nonzero(geometry.sky.any(axis=1))[0], initial=0))
    )
    glow = look.horizon_glow * np.exp(-np.clip(horizon - rows, 0, None) / (0.08 * h))
    sky_col = np.array(look.sky_rgb) + glow[..., None] * np.array([1.0, 0.75, 0.45])
    sky_img = np.broadcast_to(sky_col, (h, w, 3))
    out = np.where(geometry.sky[..., None], sky_img, out)

    rng = np.random.default_rng(look.seed)
    sigma = np.sqrt(look.shot_noise * np.clip(out, 0, None) + look.read_noise**2)
    out = out + rng.normal(size=out.shape) * sigma
    if look.blur_px > 0:
        out = ndimage.gaussian_filter(out, sigma=(look.blur_px, look.blur_px, 0))
    disp = linear_to_srgb(tone_map(np.clip(out, 0.0, None)))
    return Image.fromarray((disp * 255 + 0.5).astype(np.uint8))


# --- metrics -----------------------------------------------------------------


def _luma(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr @ np.array([0.2126, 0.7152, 0.0722]), dtype=np.float64)


def _highpass_log(lum: np.ndarray, sigma: float) -> np.ndarray:
    log = np.log(lum + 1e-3)
    return np.asarray(log - ndimage.gaussian_filter(log, sigma), dtype=np.float64)


def _weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    if w.sum() <= 1e-9:
        return float("nan")
    w = w / w.sum()
    am, bm = np.sum(w * a), np.sum(w * b)
    cov = np.sum(w * (a - am) * (b - bm))
    var = np.sqrt(np.sum(w * (a - am) ** 2) * np.sum(w * (b - bm) ** 2))
    return float(cov / var) if var > 1e-12 else float("nan")


def night_metrics(
    original: Image.Image,
    candidate: Image.Image,
    irradiance: np.ndarray,
    geometry: SceneGeometry,
    *,
    boxes: list[dict[str, Any]] | None = None,
    box_scale: tuple[float, float] = (1.0, 1.0),
    visible_irradiance: float = 0.15,
    visible_structure: float = 0.3,
) -> dict[str, Any]:
    """Physics-referenced checks for a night candidate (analysis grid = irradiance).

    - ``mean_luma`` / ``dark_fraction``: how dark the frame is (sRGB luma).
    - ``beam_corr``: correlation of candidate log-luminance with log-irradiance
      on ground pixels — high when the light pool sits where the lamps put it.
    - ``lit_structure``: illumination-invariant detail kept from the original,
      weighted by irradiance (unlit detail may legitimately vanish).
    - ``phantom_light``: share of bright pixels where the lamps deliver almost
      no light (irradiance < 0.1) — invented light sources such as visible
      headlights, streetlights or flares. The lamps sit below the camera, so
      none should appear in frame.
    - ``clipped_lit``: share of beam-lit pixels (irradiance > 0.5) blown out to
      white; clipping destroys the ballast/sleeper detail a detector needs.
    - per switch box: mean irradiance, local structure, and a visibility flag.
    ``box_scale`` maps label coordinates onto the analysis grid.
    """
    h, w = irradiance.shape
    orig = (
        np.asarray(original.convert("RGB").resize((w, h), Image.Resampling.LANCZOS))
        / 255.0
    )
    cand = (
        np.asarray(candidate.convert("RGB").resize((w, h), Image.Resampling.LANCZOS))
        / 255.0
    )
    cand_luma = _luma(cand)
    cand_lin = _luma(srgb_to_linear(cand))
    orig_lin = _luma(srgb_to_linear(orig))
    sigma = max(2.0, w / 120)
    hp_o = _highpass_log(orig_lin, sigma)
    hp_c = _highpass_log(cand_lin, sigma)
    weight = np.clip(irradiance, 0.0, 1.0)

    ground = geometry.ground & (irradiance > 1e-4)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(ground)
    if idx.size > 20000:
        idx = rng.choice(idx, 20000, replace=False)
    beam_corr = (
        float(
            np.corrcoef(
                np.log(cand_lin.ravel()[idx] + 1e-3), np.log(irradiance.ravel()[idx])
            )[0, 1]
        )
        if idx.size > 100
        else float("nan")
    )

    lit = irradiance > 0.5
    metrics: dict[str, Any] = {
        "mean_luma": float(cand_luma.mean()),
        "dark_fraction": float((cand_luma < 0.08).mean()),
        "beam_corr": beam_corr,
        "lit_structure": _weighted_corr(hp_o, hp_c, weight),
        "phantom_light": float(((cand_luma > 0.75) & (irradiance < 0.1)).mean()),
        "clipped_lit": float((cand_luma[lit] > 0.95).mean()) if lit.any() else 0.0,
    }
    box_rows = []
    sx, sy = box_scale
    for box in boxes or []:
        x1, y1, x2, y2 = box["bbox_xyxy"]
        c1, c2 = int(max(0, math.floor(x1 * sx))), int(min(w, math.ceil(x2 * sx)))
        r1, r2 = int(max(0, math.floor(y1 * sy))), int(min(h, math.ceil(y2 * sy)))
        if c2 - c1 < 2 or r2 - r1 < 2:
            box_rows.append(
                {**box, "irradiance": 0.0, "structure": float("nan"), "visible": False}
            )
            continue
        e_box = float(irradiance[r1:r2, c1:c2].mean())
        corr = _weighted_corr(
            hp_o[r1:r2, c1:c2], hp_c[r1:r2, c1:c2], np.ones((r2 - r1, c2 - c1))
        )
        visible = bool(
            e_box >= visible_irradiance and np.nan_to_num(corr) >= visible_structure
        )
        box_rows.append(
            {**box, "irradiance": e_box, "structure": corr, "visible": visible}
        )
    metrics["boxes"] = box_rows
    return metrics


def selection_score(
    metrics: dict[str, Any],
    vlm_overall: float | None = None,
    *,
    target_luma: float = 0.12,
    phantom_tolerance: float = 5e-4,
) -> float:
    """Heuristic 0–1 ranking score (physics checks, darkness, optional VLM).

    ``phantom_tolerance`` ≈ two small lamp blobs at 768×432 zeroes the
    phantom term. Weights are a workshop choice, not a calibrated quality model.
    """
    beam = float(np.clip(np.nan_to_num(metrics["beam_corr"]), 0.0, 1.0))
    structure = float(np.clip(np.nan_to_num(metrics["lit_structure"]), 0.0, 1.0))
    phantom = 1.0 - min(1.0, float(metrics["phantom_light"]) / phantom_tolerance)
    dark = 1.0 - min(1.0, abs(float(metrics["mean_luma"]) - target_luma) / target_luma)
    physics = 0.3 * beam + 0.3 * structure + 0.2 * phantom + 0.2 * dark
    if vlm_overall is None or math.isnan(vlm_overall):
        return physics
    return 0.5 * physics + 0.5 * float(np.clip(vlm_overall / 10.0, 0.0, 1.0))


# --- prompts -----------------------------------------------------------------


def scene_facts(
    annotation: dict[str, Any],
    camera: RailCamera,
    rig: HeadlightRig,
    *,
    n_switches: int,
) -> dict[str, Any]:
    """Measured facts used to ground the prompt (tags, geometry, beam reach)."""
    tags = annotation.get("tag groups") or {}
    positions = [t["position"] for t in track_polylines(annotation)]
    return {
        "environment": [str(x) for x in tags.get("environment") or []],
        "weather": [str(x) for x in tags.get("weather") or []],
        "track_layout": [str(x) for x in tags.get("track_layout") or []],
        "n_tracks": len(positions),
        "n_switches": int(n_switches),
        "camera_height_m": round(camera.height_m, 2),
        "lamp_height_m": rig.lamp_height_m,
        "beam_reach_m": round(rig.beam_reach_m(camera)),
    }


def naive_prompt() -> str:
    """Baseline a user would type first."""
    return "Make this photo a night scene."


def physics_prompt_slots(
    facts: dict[str, Any], *, name_source: bool = False, lamp: str = "warm halogen"
) -> dict[str, str]:
    """Slot-structured, physically grounded edit instruction.

    By default the light is described only by its *effect* (where it lands).
    Naming the source ("headlights", "lamps") makes few-step editors paint those
    objects into the frame — typically an oncoming train at the vanishing point —
    even when told they are out of frame. ``name_source=True`` keeps the named
    variant for ablations. Exclusions are phrased positively for the same reason.
    """
    env = " and ".join(facts["environment"]) or "rural"
    layout = " and ".join(facts["track_layout"]) or "straight"
    weather = " and ".join(facts["weather"]) or "overcast"
    switches = (
        f" with {facts['n_switches']} railway switches" if facts["n_switches"] else ""
    )
    reach = facts["beam_reach_m"]
    if name_source:
        source = (
            f"Two {lamp} headlights are mounted low on the train front, about "
            f"{facts['lamp_height_m']:.0f} m above the rails and below the camera, so the lamps are out of frame."
        )
        darkness = "It is a moonless night under thick cloud; the only real light is the train's headlights."
    else:
        source = (
            "The only illumination is a pool of white-yellow light thrown forward from just below "
            "the bottom edge of the frame onto the track."
        )
        darkness = "Thick cloud and no moon: the scene is almost completely dark."
    return {
        "task": "Relight this forward-facing train cab photo as the same scene on a dark night.",
        "scene": (
            f"A {env} Norwegian railway, {layout} track, {facts['n_tracks']} tracks"
            f"{switches}, {weather} sky."
        ),
        "darkness": darkness,
        "light_source": source,
        "beam": (
            "The pool lights the ballast and sleepers at the bottom centre of the frame, and the "
            "steel rail heads glint where lit."
        ),
        "falloff": (
            f"The light fades quickly with distance; beyond about {reach} m the track disappears "
            "into darkness and the far end of the line is completely black."
        ),
        "unlit": (
            "Trees, poles, buildings and the sky outside the pool fall to near-black; the sky is "
            "uniformly dark."
        ),
        "camera": "Same cab camera at night: high ISO grain, slightly soft.",
        "preserve": (
            "Keep the exact same viewpoint, framing, track layout, rails, switches, sleepers, signals, poles, "
            "vegetation and buildings in the same positions and shapes; add or remove nothing."
        ),
    }


def prior_refine_slots(facts: dict[str, Any]) -> dict[str, str]:
    """Slots for refining the physics render instead of the daytime photo."""
    slots = physics_prompt_slots(facts)
    slots["task"] = (
        "This is a rough physically based night relighting of a train cab photo; make it look like a real "
        "night camera photo while keeping its light pattern and darkness."
    )
    slots["darkness"] = "Keep the overall darkness exactly as dark as the input."
    return slots


def compose_prompt(slots: dict[str, str]) -> str:
    """Join slots in canonical order."""
    unknown = set(slots) - set(PROMPT_SLOTS)
    if unknown:
        raise ValueError(f"Unknown prompt slots: {sorted(unknown)}")
    return " ".join(
        slots[key].strip() for key in PROMPT_SLOTS if slots.get(key, "").strip()
    )


def scale_boxes(
    boxes: list[dict[str, Any]], sx: float, sy: float
) -> list[dict[str, Any]]:
    """Scale ``bbox_xyxy`` labels to another resolution."""
    return [
        {
            **box,
            "bbox_xyxy": [
                box["bbox_xyxy"][0] * sx,
                box["bbox_xyxy"][1] * sy,
                box["bbox_xyxy"][2] * sx,
                box["bbox_xyxy"][3] * sy,
            ],
        }
        for box in boxes
    ]
