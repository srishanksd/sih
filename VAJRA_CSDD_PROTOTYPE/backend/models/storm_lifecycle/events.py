def classify(previous, current):
    if not previous and current:
        return [{"event": "birth", "storm_id": x["id"]} for x in current]
    if previous and not current:
        return [{"event": "death", "storm_id": x["id"]} for x in previous]
    return []
