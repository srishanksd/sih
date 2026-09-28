"""Storm lifecycle inference from consecutive detected object sets."""
import math
from scipy.optimize import linear_sum_assignment


def _dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


class LifecycleTracker:
    def __init__(self, max_distance=28.0, max_area_ratio=4.0):
        self.next_id = 1
        self.previous = []
        self.max_distance = max_distance
        self.max_area_ratio = max_area_ratio

    def _new_id(self):
        value = f"S{self.next_id:03d}"
        self.next_id += 1
        return value

    def update(self, current):
        current = [dict(x) for x in current]
        previous = self.previous
        events = []
        if not previous:
            for obj in current:
                obj["id"] = self._new_id(); events.append({"event":"birth","storm_id":obj["id"]})
            self.previous = current
            return current, events
        if not current:
            events = [{"event":"death","storm_id":p["id"]} for p in previous]
            self.previous = []
            return [], events
        cost = [[_dist(p, c) for c in current] for p in previous]
        rows, cols = linear_sum_assignment(cost)
        used_p, used_c = set(), set()
        for r, c in zip(rows, cols):
            p, obj, d = previous[r], current[c], cost[r][c]
            ratio = max(obj.get("area",1), p.get("area",1)) / max(1,min(obj.get("area",1),p.get("area",1)))
            if d <= self.max_distance and ratio <= self.max_area_ratio:
                obj["id"] = p["id"]; obj["parent_id"] = p["id"]; used_p.add(r); used_c.add(c)
                events.append({"event":"continue","storm_id":obj["id"],"distance":d})
        for c, obj in enumerate(current):
            if c not in used_c:
                obj["id"] = self._new_id(); events.append({"event":"birth","storm_id":obj["id"]})
        for r, p in enumerate(previous):
            if r not in used_p: events.append({"event":"death","storm_id":p["id"]})
        # Detect merge/split candidates from proximity after stable matching.
        for c, obj in enumerate(current):
            near = [p for p in previous if _dist(p,obj) <= self.max_distance]
            if len(near) >= 2:
                events.append({"event":"merge","storm_id":obj["id"],"parents":[p["id"] for p in near]})
        for r, p in enumerate(previous):
            near = [o for o in current if _dist(p,o) <= self.max_distance]
            if len(near) >= 2:
                events.append({"event":"split","storm_id":p["id"],"children":[o["id"] for o in near]})
        self.previous = current
        return current, events


def classify(previous, current):
    tracker = LifecycleTracker()
    tracker.previous = [dict(x) for x in previous or []]
    _, events = tracker.update(current or [])
    return events
