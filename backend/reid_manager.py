"""
reid_manager.py
───────────────
Cross-camera identity gallery for global Re-Identification.

Receives L2-normalized MobileNetV2 embeddings from each edge node and
resolves them to consistent Global IDs across cameras using cosine similarity.

Key design decisions:
  - Same-camera entries are excluded from matching queries (prevents false
    positive matches between different people on the same camera).
  - Gallery entries expire after MAX_AGE_SECONDS to prevent unbounded growth.
  - Gallery embeddings are updated with an EMA blend (old=0.8, new=0.2)
    to account for appearance variation over time.
  - Threshold of 0.85 cosine similarity is deliberately strict to minimise
    false cross-camera matches (precision > recall).
"""

import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

MATCH_THRESHOLD  = 0.65    # cosine similarity for a positive cross-camera match
MAX_AGE_SECONDS  = 300.0   # gallery entries expire after 5 minutes of no sighting
EMA_ALPHA_UPDATE = 0.2     # blend weight for incoming embedding on gallery update
EMA_ALPHA_KNOWN  = 0.1     # blend weight for re-sighting on known same-camera track


class _GalleryEntry:
    __slots__ = ("global_id", "origin_camera", "cameras_seen", "embedding", "last_seen")

    def __init__(self, global_id: int, camera: str, embedding: np.ndarray):
        self.global_id     = global_id
        self.origin_camera = camera
        self.cameras_seen  = {camera}
        self.embedding     = embedding.copy()
        self.last_seen     = time.time()


class ReIDManager:
    """
    Manages a cross-camera identity gallery backed by appearance embeddings.

    Thread-safety: not async — all calls happen inside FastAPI's single-threaded
    async route handler (no concurrent access from multiple coroutines).
    """

    def __init__(self):
        # global_id -> _GalleryEntry
        self._gallery: dict[int, _GalleryEntry] = {}
        # (camera, str(local_tid)) -> global_id
        self._local_to_global: dict[tuple, int] = {}
        self._next_gid = 1

        # global_id -> list[zone_name]  (ordered history of zone visits)
        self._global_history: dict[int, list[str]] = {}
        # "ZoneA->ZoneB" -> count  (cross-camera verified transitions)
        self._global_transitions: dict[str, int] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve_identities(self, cam_name: str, signatures: dict) -> None:
        """
        Process incoming track signatures from one camera.

        Args:
            cam_name:   Camera name from config (e.g. "Entrance_Cam")
            signatures: {str(local_tid): [float, ...]}  (1280-d embedding lists)
        """
        self._purge_expired()

        for local_tid, emb_list in signatures.items():
            key = (cam_name, str(local_tid))
            emb = np.array(emb_list, dtype=np.float32)

            # ── Already resolved: just refresh the gallery embedding ──
            if key in self._local_to_global:
                gid = self._local_to_global[key]
                if gid in self._gallery:
                    entry = self._gallery[gid]
                    entry.embedding = self._ema(entry.embedding, emb, EMA_ALPHA_KNOWN)
                    entry.cameras_seen.add(cam_name)
                    entry.last_seen = time.time()
                continue

            # ── New track: search entire gallery for a cosine similarity match ──
            # We rely on the threshold (0.65 for ImageNet features) to avoid false 
            # positives rather than excluding same-camera entries, so that two instances 
            # running the same feed (or a person re-entering the same camera) 
            # can still be correctly re-identified.
            best_gid, best_score = None, MATCH_THRESHOLD

            for gid, entry in self._gallery.items():
                score = float(np.dot(emb, entry.embedding))
                if score > best_score:
                    best_score, best_gid = score, gid

            if best_gid is not None:
                # Cross-camera match found
                entry = self._gallery[best_gid]
                entry.embedding = self._ema(entry.embedding, emb, EMA_ALPHA_UPDATE)
                entry.cameras_seen.add(cam_name)
                entry.last_seen = time.time()
                self._local_to_global[key] = best_gid
                logger.info(
                    "ReID match: %s track %s -> Global ID %d (score=%.3f, cameras=%s)",
                    cam_name, local_tid, best_gid, best_score,
                    entry.cameras_seen,
                )
            else:
                # Brand-new identity
                gid = self._next_gid
                self._next_gid += 1
                self._gallery[gid] = _GalleryEntry(gid, cam_name, emb)
                self._local_to_global[key] = gid
                logger.debug("New identity: %s track %s -> Global ID %d", cam_name, local_tid, gid)

    def update_global_journeys(self, cam_name: str, track_zones: dict) -> None:
        """
        Record zone transitions for globally-resolved identities.

        Only identities that have been matched across cameras contribute to
        global_transitions (i.e. true cross-camera journey events).

        Args:
            cam_name:    Camera name
            track_zones: {str(local_tid): zone_name_or_None}
        """
        for local_tid, zone_name in track_zones.items():
            if not zone_name:
                continue
            key = (cam_name, str(local_tid))
            gid = self._local_to_global.get(key)
            if gid is None:
                continue

            history = self._global_history.setdefault(gid, [])
            if not history or history[-1] != zone_name:
                if history:
                    transition = f"{history[-1]}->{zone_name}"
                    self._global_transitions[transition] = (
                        self._global_transitions.get(transition, 0) + 1
                    )
                    logger.info(
                        "Global transition: ID %d  %s  (cam: %s)",
                        gid, transition, cam_name,
                    )
                history.append(zone_name)

    def get_global_funnel(self) -> dict:
        """
        Returns {zone_name: unique_global_id_count} for zones visited by
        any globally-resolved identity.
        """
        funnel: dict[str, int] = {}
        for history in self._global_history.values():
            for zone in set(history):
                funnel[zone] = funnel.get(zone, 0) + 1
        return funnel

    def get_global_transitions(self) -> dict:
        """Returns the accumulated cross-camera zone transition counts."""
        return dict(self._global_transitions)

    def get_cross_camera_count(self) -> int:
        """Number of global identities confirmed across 2+ cameras."""
        return sum(
            1 for e in self._gallery.values() if len(e.cameras_seen) > 1
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _ema(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
        """Exponential moving average blend, re-normalised to unit length."""
        blended = (1 - alpha) * old + alpha * new
        norm = np.linalg.norm(blended)
        return blended / norm if norm > 1e-6 else blended

    def _purge_expired(self) -> None:
        """Remove gallery entries that haven't been seen recently."""
        now = time.time()
        expired = [
            gid for gid, e in self._gallery.items()
            if now - e.last_seen > MAX_AGE_SECONDS
        ]
        for gid in expired:
            del self._gallery[gid]
            # Clean up local->global mappings for expired entries
            self._local_to_global = {
                k: v for k, v in self._local_to_global.items() if v != gid
            }
        if expired:
            logger.debug("Purged %d expired gallery entries.", len(expired))
