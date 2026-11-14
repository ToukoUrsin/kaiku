"""Classical colour and depth perception for the head camera.

The detector deliberately has no model of the object bodies.  It uses the known
camera calibration, the printed colours, and the table geometry to turn pixels
into small 3-D components.  Shape and turn come from the outline of each top
face.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

from kaiku.sim.scene import Scene


@dataclass
class PartEstimate:
    """One part found in a head-camera frame."""

    pos: np.ndarray
    yaw: float
    shape: str
    color: str
    visible: bool = True
    score: float = 1.0


def _wrap(yaw: float, symmetry: int) -> float:
    if not symmetry:
        return 0.0
    period = 2.0 * np.pi / symmetry
    return float((yaw + period / 2.0) % period - period / 2.0)


class CameraPerception:
    """Estimate parts from one RGB and depth image from ``head_rgb``.

    ``twin`` is used only for the camera calibration and the table dimensions.
    It is never inspected for object poses.  The renderer helper is convenient
    in the twin, while :meth:`estimate` also accepts real camera frames.
    """

    def __init__(self, twin=None, scene: Scene | None = None, width: int = 320, height: int = 240,
                 depth_noise_mm: float = 0.0, edge_noise_px: float = 0.0, seed: int = 0):
        self.twin = twin
        self.scene = scene or (twin.scene if twin is not None else Scene())
        self.width, self.height = int(width), int(height)
        self.depth_noise_mm = float(depth_noise_mm)
        self.edge_noise_px = float(edge_noise_px)
        self.rng = np.random.default_rng(seed)
        self.renderer = None
        self.cam_id = None
        if twin is not None:
            import mujoco
            self.cam_id = mujoco.mj_name2id(twin.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_rgb")
            self.renderer = mujoco.Renderer(twin.model, height=self.height, width=self.width)
        self._intrinsics()

    def _intrinsics(self) -> None:
        fovy = float(self.twin.model.cam_fovy[self.cam_id]) if self.twin is not None else 43.4
        fy = self.height / (2.0 * np.tan(np.radians(fovy) / 2.0))
        # MuJoCo's offscreen renderer uses the same focal length on both axes;
        # its fovy is applied to the render target rather than to its aspect.
        self.fx, self.fy = fy, fy
        self.cx, self.cy = (self.width - 1) / 2.0, (self.height - 1) / 2.0

    def frames(self, data) -> tuple[np.ndarray, np.ndarray]:
        """Render one RGB/depth pair from a twin ``MjData``."""
        if self.renderer is None:
            raise RuntimeError("CameraPerception was created without a twin")
        self._data = data
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera="head_rgb")
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera="head_rgb")
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def _camera_pose(self, camera_pos=None, camera_rot=None):
        if camera_pos is not None and camera_rot is not None:
            return np.asarray(camera_pos, float), np.asarray(camera_rot, float).reshape(3, 3)
        if self.twin is None:
            raise ValueError("camera_pos and camera_rot are required without a twin")
        # cam_xmat maps camera-frame vectors to the world frame.
        if hasattr(self, "_data"):
            d = self._data
            return d.cam_xpos[self.cam_id].copy(), d.cam_xmat[self.cam_id].reshape(3, 3).copy()
        return self.twin.model.cam_pos[self.cam_id].copy(), self.twin.model.cam_mat0[self.cam_id].reshape(3, 3).copy()

    def _world_points(self, depth: np.ndarray, camera_pos=None, camera_rot=None) -> np.ndarray:
        d = np.asarray(depth, float)
        if d.ndim == 3:
            d = d[..., 0]
        if self.depth_noise_mm:
            d = d + self.rng.normal(0.0, self.depth_noise_mm / 1000.0, d.shape)
        yy, xx = np.indices(d.shape)
        # MuJoCo's camera looks along -z; image y points down.
        c = np.stack(((xx - self.cx) * d / self.fx, (self.cy - yy) * d / self.fy, -d), -1)
        p, R = self._camera_pose(camera_pos, camera_rot)
        return p + c @ R.T

    def estimate(self, rgb: np.ndarray, depth: np.ndarray, camera_pos=None, camera_rot=None) -> list[PartEstimate]:
        """Find visible parts in an RGB/depth pair, in increasing robot x."""
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.shape[:2] != (self.height, self.width):
            raise ValueError(f"expected {(self.height, self.width)} RGB pixels, got {rgb.shape[:2]}")
        world = self._world_points(depth, camera_pos, camera_rot)
        z = world[..., 2]
        # Only the counter and the three known bin footprints can contain a part.
        mat_lo, mat_hi = self.scene.region
        mat = ((world[..., 0] >= mat_lo[0] - 0.025) & (world[..., 0] <= mat_hi[0] + 0.025) &
               (world[..., 1] >= mat_lo[1] - 0.025) & (world[..., 1] <= mat_hi[1] + 0.025))
        bins = np.zeros(world.shape[:2], bool)
        for centre in self.scene.bin_centers.values():
            half = self.scene.bin_outer / 2.0 + 0.02
            bins |= ((world[..., 0] >= centre[0] - half[0]) & (world[..., 0] <= centre[0] + half[0]) &
                     (world[..., 1] >= centre[1] - half[1]) & (world[..., 1] <= centre[1] + half[1]))
        region = mat | bins
        # Exclude the table and the parked objects, retaining the part top and sides.
        region &= (z > self.scene.counter_z - 0.01) & (z < self.scene.counter_z + 0.15)
        chrom = rgb.astype(float) / np.maximum(rgb.sum(-1, keepdims=True), 1.0)
        expected = {name: self.scene.colors[name] / self.scene.colors[name].sum()
                    for name in self.scene.colors}
        out: list[PartEstimate] = []
        for colour, ec in expected.items():
            mask = (np.linalg.norm(chrom - ec, axis=-1) < 0.09) & (rgb.max(-1) > 35) & region
            if self.edge_noise_px:
                # A real depth edge jitters by about one pixel.  Drop a few boundary
                # pixels so the outline code sees the same uncertainty.
                boundary = ndimage.binary_dilation(mask) ^ ndimage.binary_erosion(mask)
                drop = boundary & (self.rng.random(mask.shape) < min(1.0, self.edge_noise_px * 0.18))
                mask &= ~drop
            mask = ndimage.binary_closing(mask, iterations=1)
            labels, count = ndimage.label(mask, structure=np.ones((3, 3), int))
            for label in range(1, count + 1):
                pix = labels == label
                if int(pix.sum()) < 5:
                    continue
                pts = world[pix]
                # A bin is wider than any printed part; this also rejects the orange
                # bin when its chroma is close to red or yellow.
                span = np.ptp(pts[:, :2], axis=0)
                if span.max() > self.scene.size * 2.8 or span.min() < self.scene.size * 0.18:
                    continue
                top = pts[:, 2] >= np.quantile(pts[:, 2], 0.55)
                xy = pts[top, :2]
                if len(xy) < 4:
                    continue
                shape, yaw, compact = self._outline(xy)
                top_z = float(np.median(pts[top, 2]))
                pos = np.array([np.median(xy[:, 0]), np.median(xy[:, 1]), top_z - self.scene.height / 2])
                out.append(PartEstimate(pos, yaw, shape, colour, True, float(min(1.0, len(xy) / 25.0))))
        # One component can touch another part only in a malformed scene; retain the
        # higher-confidence component in that case.
        out.sort(key=lambda e: (e.pos[0], e.pos[1]))
        kept: list[PartEstimate] = []
        for e in out:
            if all(np.linalg.norm(e.pos[:2] - q.pos[:2]) > self.scene.size * 0.55 for q in kept):
                kept.append(e)
        return kept

    def _outline(self, xy: np.ndarray) -> tuple[str, float, float]:
        try:
            hull = xy[ConvexHull(xy).vertices] if len(xy) >= 3 else xy
        except Exception:
            hull = xy
        # Shoelace is invariant to the order in which pixels were found.
        area = 0.0
        if len(hull) > 2:
            area = abs(float(np.dot(hull[:, 0], np.roll(hull[:, 1], -1)) -
                           np.dot(hull[:, 1], np.roll(hull[:, 0], -1)))) / 2.0
        perimeter = float(np.linalg.norm(np.diff(np.vstack([hull, hull[:1]]), axis=0), axis=1).sum()) if len(hull) > 2 else 1.0
        compact = 4.0 * np.pi * area / max(perimeter * perimeter, 1e-9)
        shape = min(("cube", "cylinder", "triangle"), key=lambda s: abs(compact -
                    {"cube": 0.785, "cylinder": 0.92, "triangle": 0.605}[s]))
        centre = xy.mean(0)
        a = np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0])
        k = {"cube": 4, "cylinder": 0, "triangle": 3}[shape]
        if not k:
            yaw = 0.0
        else:
            harmonic = np.sum(np.exp(1j * k * a))
            yaw = _wrap(float(np.angle(harmonic) / k), k)
            # The exported meshes start a cube at a corner and a triangle at
            # the middle of an edge; convert outline phase to body yaw.
            yaw = _wrap(yaw + ({"cube": np.pi / 4.0, "triangle": np.pi / 6.0}[shape]), k)
        return shape, yaw, compact

    def observe(self, data) -> list[PartEstimate]:
        """Render and estimate one twin environment."""
        self._data = data
        return self.estimate(*self.frames(data))

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


__all__ = ["CameraPerception", "PartEstimate"]
