"""Storm-cell graph construction for interactions and lifecycle reasoning."""
from math import hypot


def build_graph(objects, radius=30.0):
    """Return node ids and proximity edges in grid-coordinate units."""
    nodes = [dict(o) for o in objects]
    edges = []
    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            d = hypot(a["x"] - nodes[j]["x"], a["y"] - nodes[j]["y"])
            if d <= radius:
                edges.append({"source": i, "target": j, "distance": float(d)})
    return {"nodes": nodes, "edges": edges}


def graph_features(graph):
    degree = [0] * len(graph["nodes"])
    for edge in graph["edges"]:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    return degree
