def score(objects, field, grid_size=96):
    if not objects:
        return 0.0
    covered = 0
    h, w = field.shape
    for o in objects:
        x, y = int(o["x"] / grid_size * w), int(o["y"] / grid_size * h)
        if 0 <= x < w and 0 <= y < h and field[y, x] > 0.4:
            covered += 1
    return covered / len(objects)
