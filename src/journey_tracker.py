"""
journey_tracker.py
──────────────────
Tracks cross-zone transitions per camera to power the conversion funnel
and the D3 Sankey diagram on the analytics dashboard.

Design:
  - Maintains `last_zone[track_id]` — the zone each track was most recently in.
  - On a zone change, records a transition event (from → to).
  - Exposes aggregated funnel counts and a D3-ready Sankey payload.

Note: This tracker operates within a single camera's zone namespace.
      Cross-camera journey reconstruction would require a Re-ID model.
      The dashboard approximates the full funnel from aggregated zone totals.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

ZONE_ORDER = ["Entrance", "Stall", "Seating", "Shop_Entry",
              "Pots", "Cups", "Plates", "Bowls"]

class JourneyTracker:
    """
    Records zone-to-zone transitions for tracked objects within a single camera.

    Args:
        zone_names: Ordered list of zone names this camera monitors.
    """

    def __init__(self, zone_names: list):
        self.zone_names = zone_names
        self._last_zone: dict[int, str | None] = {}

        self._transitions: dict = defaultdict(lambda: defaultdict(int))

        self._zone_unique: dict[str, set] = defaultdict(set)

    def update(self, track_zone_map: dict):
        """
        Process one frame's worth of track→zone assignments.

        Args:
            track_zone_map: {track_id: zone_name_or_None} from ZoneAnalyzer.
        """
        for track_id, current_zone in track_zone_map.items():
            prev_zone = self._last_zone.get(track_id)

            if current_zone:
                self._zone_unique[current_zone].add(track_id)

            if prev_zone != current_zone:
                if prev_zone is not None and current_zone is not None:
                    self._transitions[prev_zone][current_zone] += 1
                    logger.debug("Track %d: %s → %s", track_id, prev_zone, current_zone)

            self._last_zone[track_id] = current_zone

    def get_funnel(self) -> dict:
        """
        Returns unique visitor counts per zone.

        Example: {"Entrance": 100, "Checkout": 31}
        """
        return {zone: len(visitors) for zone, visitors in self._zone_unique.items()}

    def get_transitions(self) -> dict:
        """
        Returns the full transition adjacency matrix (serializable).

        Example: {"Entrance": {"Checkout": 31}, "Aisle": {"Checkout": 27}}
        """
        return {frm: dict(tos) for frm, tos in self._transitions.items()}

    def get_sankey_data(self) -> dict:
        """
        Returns a D3-Sankey-compatible payload.

        Uses actual within-camera transition counts for links.
        Zones without any recorded transitions are still included as nodes
        if they have unique visitors (so the Sankey always shows zone sizes).

        Returns:
            {"nodes": [{"name": "Entrance"}, ...], "links": [{"source": 0, "target": 1, "value": 47}, ...]}
        """

        ordered = [z for z in ZONE_ORDER if z in self._zone_unique]

        for z in self.zone_names:
            if z not in ordered and z in self._zone_unique:
                ordered.append(z)

        if not ordered:
            return {"nodes": [], "links": []}

        node_index = {name: i for i, name in enumerate(ordered)}
        nodes = [{"name": z} for z in ordered]
        links = []

        for frm, tos in self._transitions.items():
            if frm not in node_index:
                continue
            for to, value in tos.items():
                if to in node_index and value > 0:
                    links.append({
                        "source": node_index[frm],
                        "target": node_index[to],
                        "value": value,
                    })

        return {"nodes": nodes, "links": links}
