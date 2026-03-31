"""
tracker/byte_tracker.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ByteTrack — Multi-Object Tracking by Associating Every Detection Box
  Zhang et al., ECCV 2022.  https://arxiv.org/abs/2110.06864

Why ByteTrack over SORT?
────────────────────────
SORT discards all detections below the confidence threshold.
When an object is partially occluded the detector fires at, say, 0.25
confidence — SORT drops it entirely and the track is lost.

ByteTrack's key insight: keep those weak detections and run a
*second association stage* that matches them to existing tracks
that went unmatched in stage 1.  This prevents false track
terminations from brief occlusions or momentary detector uncertainty.

Two-stage association pipeline
  Stage 1 : high-conf dets  ↔  active tracks          (IoU, Hungarian)
  Stage 2 : low-conf dets   ↔  unmatched active tracks (IoU, Hungarian)
  Stage 3 : remaining high  ↔  Lost tracks             (re-activation)
  Stage 4 : still-unmatched high → spawn new tentative tracks

Track state machine
  Tentative ─► Confirmed  (after ≥ min_hits consecutive detections)
  Confirmed ─► Lost       (no detection for ≥ 1 frame)
  Lost      ─► Confirmed  (re-activated on matching detection)
  Lost      ─► Dead       (expired after max_age frames of coasting)
"""
from __future__ import annotations

import logging
from enum import IntEnum
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────

def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Vectorised pairwise Intersection-over-Union.

    Args:
        bb_test : (N, 4+)  rows of [x1, y1, x2, y2, ...]
        bb_gt   : (M, 4+)  rows of [x1, y1, x2, y2, ...]
    Returns:
        iou_matrix : (N, M) float64
    """
    a = bb_test[:, :4][:, None]   # (N, 1, 4)
    b = bb_gt[:, :4][None]        # (1, M, 4)
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0., ix2 - ix1) * np.maximum(0., iy2 - iy1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-6)


def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] → Kalman measurement [cx, cy, area, aspect_ratio]."""
    w  = bbox[2] - bbox[0]
    h  = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s  = max(w * h, 1.0)
    r  = w / float(max(h, 1.0))
    return np.array([cx, cy, s, r], dtype=float).reshape(4, 1)


def _z_to_bbox(x: np.ndarray) -> np.ndarray:
    """Kalman state [cx, cy, area, ratio, ...] → [x1, y1, x2, y2]."""
    s = max(float(x[2, 0]), 1e-4)
    r = max(float(x[3, 0]), 1e-4)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([
        x[0, 0] - w / 2., x[1, 0] - h / 2.,
        x[0, 0] + w / 2., x[1, 0] + h / 2.,
    ]).reshape(1, 4)


# ─────────────────────────────────────────────────────────────────
# Kalman filter for one bounding box
# ─────────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    7-state constant-velocity Kalman filter.

    State   : [cx, cy, s, r,  ċx, ċy, ṡ]
    Measured: [cx, cy, s, r]

    The global `_global_count` is intentionally never reset so track IDs
    remain unique across tracker.reset() calls in the same session.
    """

    _global_count: int = 0

    def __init__(self, bbox: np.ndarray, cls_id: int, conf: float) -> None:
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)
        kf.H = np.eye(4, 7, dtype=float)
        kf.R[2:, 2:] *= 10.    # scale + ratio measurements are noisier
        kf.P[4:, 4:] *= 1000.  # high initial uncertainty in velocity
        kf.P         *= 10.
        kf.Q[-1, -1] *= 0.01   # scale velocity drifts slowly
        kf.Q[4:, 4:] *= 0.01
        kf.x[:4]      = _bbox_to_z(bbox)
        self.kf = kf

        self.id:                int   = KalmanBoxTracker._global_count
        KalmanBoxTracker._global_count += 1
        self.cls_id:            int   = cls_id
        self.conf:              float = conf
        self.hits:              int   = 0
        self.hit_streak:        int   = 0
        self.time_since_update: int   = 0

    def update(self, bbox: np.ndarray, cls_id: int, conf: float) -> None:
        """Fuse a matched detection measurement."""
        self.time_since_update = 0
        self.hits             += 1
        self.hit_streak       += 1
        self.cls_id            = cls_id
        self.conf              = conf
        self.kf.update(_bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        """Advance the Kalman filter one step; return predicted [x1,y1,x2,y2]."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.             # prevent negative area via drift
        self.kf.predict()
        if self.time_since_update > 0:     # streak broken when coasting
            self.hit_streak = 0
        self.time_since_update += 1
        return _z_to_bbox(self.kf.x)

    def get_state(self) -> np.ndarray:
        """Current best-estimate bbox [x1, y1, x2, y2]."""
        return _z_to_bbox(self.kf.x)


# ─────────────────────────────────────────────────────────────────
# Track state machine
# ─────────────────────────────────────────────────────────────────

class TrackState(IntEnum):
    Tentative = 0   # newly spawned; awaiting min_hits confirmations
    Confirmed = 1   # actively matched; output to caller
    Lost      = 2   # temporarily unmatched; Kalman coasting
    Dead      = 3   # expired; will be pruned


class STrack:
    """
    Single tracked object with ByteTrack lifecycle semantics.

    Wraps a KalmanBoxTracker and adds the state machine and
    re-activation logic specific to ByteTrack.
    """

    __slots__ = ("kalman", "state", "frame_id", "start_frame", "tracklet_len")

    def __init__(
        self,
        bbox:     np.ndarray,
        conf:     float,
        cls_id:   int,
        frame_id: int,
    ) -> None:
        self.kalman       = KalmanBoxTracker(bbox, cls_id, conf)
        self.state        = TrackState.Tentative
        self.frame_id     = frame_id
        self.start_frame  = frame_id
        self.tracklet_len = 0

    # ── Property shortcuts ────────────────────────────────────────

    @property
    def track_id(self) -> int:     return self.kalman.id + 1  # 1-indexed display
    @property
    def cls_id(self) -> int:       return self.kalman.cls_id
    @property
    def conf(self) -> float:       return self.kalman.conf
    @property
    def hit_streak(self) -> int:   return self.kalman.hit_streak
    @property
    def time_since_update(self) -> int: return self.kalman.time_since_update

    # ── State transitions ─────────────────────────────────────────

    def activate(self, frame_id: int) -> None:
        """First confirmation — move Tentative → Confirmed."""
        self.state        = TrackState.Confirmed
        self.frame_id     = frame_id
        self.start_frame  = frame_id

    def update(self, det: np.ndarray, frame_id: int) -> None:
        """Fuse a high-confidence detection; stay/return to Confirmed."""
        self.kalman.update(det[:4], int(det[5]), float(det[4]))
        self.state        = TrackState.Confirmed
        self.frame_id     = frame_id
        self.tracklet_len += 1

    def re_activate(
        self,
        det:         np.ndarray,
        frame_id:    int,
        min_hits:    int,
    ) -> None:
        """
        Re-associate a Lost track with a new detection.

        Forces hit_streak ≥ min_hits so the track is immediately
        output again — there is no need to re-accumulate confirmation
        frames for a track that was previously stable.
        """
        self.kalman.update(det[:4], int(det[5]), float(det[4]))
        self.kalman.hit_streak = max(self.kalman.hit_streak, min_hits)
        self.state        = TrackState.Confirmed
        self.frame_id     = frame_id
        self.tracklet_len = 0

    def predict(self) -> None:
        """Advance Kalman one frame."""
        self.kalman.predict()

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def get_bbox(self) -> np.ndarray:
        """Current best-estimate [x1, y1, x2, y2]."""
        return self.kalman.get_state()[0]


# ─────────────────────────────────────────────────────────────────
# Hungarian matching helper
# ─────────────────────────────────────────────────────────────────

def _associate(
    dets:          np.ndarray,
    tracks:        list[STrack],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Match detections to tracks via Hungarian algorithm on IoU cost.

    Returns
    -------
    matched        : [(det_idx, trk_idx)]
    unmatched_dets : [det_idx]  — no track found
    unmatched_trks : [trk_idx]  — no detection found
    """
    if not tracks or len(dets) == 0:
        return [], list(range(len(dets))), list(range(len(tracks)))

    track_boxes = np.array([t.get_bbox() for t in tracks])
    iou_mat = iou_batch(dets, track_boxes)              # (N_dets, N_trks)
    row_ind, col_ind = linear_sum_assignment(-iou_mat)   # minimise cost

    matched:    list[tuple[int, int]] = []
    matched_d:  set[int] = set()
    matched_t:  set[int] = set()

    for d, t in zip(row_ind.tolist(), col_ind.tolist()):
        if iou_mat[d, t] >= iou_threshold:
            matched.append((d, t))
            matched_d.add(d)
            matched_t.add(t)

    unmatched_dets = [d for d in range(len(dets))   if d not in matched_d]
    unmatched_trks = [t for t in range(len(tracks)) if t not in matched_t]
    return matched, unmatched_dets, unmatched_trks


# ─────────────────────────────────────────────────────────────────
# ByteTracker — the public API
# ─────────────────────────────────────────────────────────────────

class ByteTracker:
    """
    Universal ByteTrack multi-object tracker.

    Accepts detections of ANY class mix in a single call.

    Input  : (N, 6) float  [x1, y1, x2, y2, conf, cls_id]
             Pass np.empty((0, 6)) for frames where YOLO was skipped.
    Output : (M, 7) float  [x1, y1, x2, y2, track_id, cls_id, conf]
             Only confirmed, long-enough tracks are returned.
    """

    def __init__(
        self,
        max_age:             int   = 40,
        min_hits:            int   = 2,
        iou_threshold:       float = 0.25,
        high_conf_threshold: float = 0.50,
        low_conf_threshold:  float = 0.20,
    ) -> None:
        self.max_age             = max_age
        self.min_hits            = min_hits
        self.iou_threshold       = iou_threshold
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold  = low_conf_threshold

        self._confirmed: list[STrack] = []
        self._lost:      list[STrack] = []
        self._frame_id:  int = 0

    # ── Main update ───────────────────────────────────────────────

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Run one full ByteTrack iteration.

        Pipeline
        ────────
          1. Split detections into high/low confidence buckets.
          2. Predict all living tracks (Kalman advance).
          3. Stage-1 : match high-conf dets ↔ confirmed tracks.
          4. Stage-2 : match low-conf  dets ↔ unmatched confirmed tracks.
          5. Stage-3 : match remaining high-conf ↔ lost tracks (re-activate).
          6. Stage-4 : spawn new tentative tracks from still-unmatched high.
          7. Promote any tentative tracks with enough hits.
          8. Expire lost tracks older than max_age.
          9. Return confirmed tracks with hit_streak ≥ min_hits.

        Args:
            detections : (N, 6) [x1,y1,x2,y2,conf,cls_id].

        Returns:
            (M, 7) [x1,y1,x2,y2,track_id,cls_id,conf]
        """
        self._frame_id += 1
        fid = self._frame_id

        # ── Step 1: Split by confidence ───────────────────────────
        if len(detections):
            high_mask = detections[:, 4] >= self.high_conf_threshold
            low_mask  = (
                (detections[:, 4] >= self.low_conf_threshold) & ~high_mask
            )
            dets_high = detections[high_mask]
            dets_low  = detections[low_mask]
        else:
            dets_high = dets_low = np.empty((0, 6), dtype=float)

        # ── Step 2: Predict all living tracks ─────────────────────
        for t in self._confirmed:
            t.predict()
        for t in self._lost:
            t.predict()

        # ── Step 3: Stage-1 — high-conf dets ↔ confirmed tracks ───
        m1, unmatched_high_idx, unmatched_conf_idx = _associate(
            dets_high, self._confirmed, self.iou_threshold
        )
        for d, t in m1:
            self._confirmed[t].update(dets_high[d], fid)

        # Collect the confirmed tracks that had no high-conf match
        unmatched_confirmed: list[STrack] = [
            self._confirmed[i] for i in unmatched_conf_idx
        ]

        # ── Step 4: Stage-2 — low-conf dets ↔ unmatched confirmed ─
        m2, _unused_low, still_unmatched_conf_idx = _associate(
            dets_low, unmatched_confirmed, self.iou_threshold
        )
        for d, t in m2:
            unmatched_confirmed[t].update(dets_low[d], fid)

        # Anything still unmatched after both stages → demote to Lost
        newly_lost: list[STrack] = [
            unmatched_confirmed[i] for i in still_unmatched_conf_idx
        ]
        for t in newly_lost:
            t.mark_lost()

        # ── Step 5: Stage-3 — remaining high-conf ↔ lost tracks ───
        rem_high = (
            dets_high[unmatched_high_idx]
            if len(unmatched_high_idx) else np.empty((0, 6), dtype=float)
        )
        m3, unmatched_rem_idx, unmatched_lost_idx = _associate(
            rem_high, self._lost, self.iou_threshold
        )
        reactivated: list[STrack] = []
        for d, t in m3:
            self._lost[t].re_activate(rem_high[d], fid, self.min_hits)
            reactivated.append(self._lost[t])

        # ── Step 6: Stage-4 — spawn new tracks ────────────────────
        new_tracks: list[STrack] = []
        for i in unmatched_rem_idx:
            det = rem_high[i]
            st  = STrack(det[:4], float(det[4]), int(det[5]), fid)
            st.activate(fid)
            new_tracks.append(st)

        # ── Step 7: Rebuild confirmed pool ────────────────────────
        # Keep: survived stage-1/2 matched (still Confirmed),
        #       re-activated from lost, newly spawned.
        self._confirmed = (
            [t for t in self._confirmed if t.state == TrackState.Confirmed]
            + reactivated
            + new_tracks
        )

        # ── Step 8: Rebuild lost pool ──────────────────────────────
        reactivated_ids = {id(t) for t in reactivated}
        self._lost = (
            [
                t for i, t in enumerate(self._lost)
                if i in unmatched_lost_idx            # not re-activated
                and id(t) not in reactivated_ids
                and t.time_since_update <= self.max_age  # not expired
            ]
            + newly_lost
        )

        # ── Step 9: Collect output ─────────────────────────────────
        results: list[list[float]] = []
        for t in self._confirmed:
            # Output once the track has accumulated min_hits consecutive hits,
            # OR we are still in the warm-up window of the first min_hits frames.
            if t.hit_streak >= self.min_hits or fid <= self.min_hits:
                box = t.get_bbox()
                results.append([*box, float(t.track_id),
                                 float(t.cls_id), t.conf])

        log.debug(
            "frame=%d  confirmed=%d  lost=%d  output=%d",
            fid, len(self._confirmed), len(self._lost), len(results),
        )

        return (
            np.array(results, dtype=float)
            if results else np.empty((0, 7), dtype=float)
        )

    # ── Utilities ─────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Clear all active and lost tracks.
        Global KalmanBoxTracker.id counter is preserved so track IDs
        remain unique across resets within the same session.
        """
        self._confirmed.clear()
        self._lost.clear()
        self._frame_id = 0
        log.info("ByteTracker reset  (global ID counter preserved at %d)",
                 KalmanBoxTracker._global_count)

    @property
    def active_count(self) -> int:
        """Number of currently confirmed tracks."""
        return len(self._confirmed)

    @property
    def lost_count(self) -> int:
        """Number of coasting (lost-but-alive) tracks."""
        return len(self._lost)
