#!/usr/bin/env python3
"""
Export a custom ONNX graph for the content-based movie recommender demo.

The model takes a genre preference vector (shape [1, 10]) and returns the
top-6 most similar movies by cosine similarity against 28 pre-embedded movies.

Run from repo root:
    py scripts/export_recommender_onnx.py

Output: app/onnx_models/content_recommender.onnx
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "onnx_models"

ALL_GENRES = [
    "Action", "Animation", "Children", "Comedy", "Drama",
    "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller",
]

MOVIES = [
    {"title": "Toy Story (1995)",                   "genres": ["Animation", "Children", "Comedy", "Fantasy"]},
    {"title": "The Dark Knight (2008)",             "genres": ["Action", "Thriller"]},
    {"title": "The Matrix (1999)",                  "genres": ["Action", "Sci-Fi"]},
    {"title": "Inception (2010)",                   "genres": ["Action", "Sci-Fi", "Thriller"]},
    {"title": "Titanic (1997)",                     "genres": ["Drama", "Romance"]},
    {"title": "The Lion King (1994)",               "genres": ["Animation", "Children", "Drama", "Fantasy"]},
    {"title": "The Shawshank Redemption (1994)",    "genres": ["Drama"]},
    {"title": "Forrest Gump (1994)",                "genres": ["Comedy", "Drama", "Romance"]},
    {"title": "Jurassic Park (1993)",               "genres": ["Action", "Sci-Fi", "Thriller"]},
    {"title": "Home Alone (1990)",                  "genres": ["Children", "Comedy"]},
    {"title": "Schindler's List (1993)",            "genres": ["Drama", "Thriller"]},
    {"title": "The Silence of the Lambs (1991)",    "genres": ["Horror", "Thriller"]},
    {"title": "Speed (1994)",                       "genres": ["Action", "Thriller"]},
    {"title": "Beauty and the Beast (1991)",        "genres": ["Animation", "Children", "Fantasy", "Romance"]},
    {"title": "Pulp Fiction (1994)",                "genres": ["Drama", "Thriller"]},
    {"title": "Die Hard (1988)",                    "genres": ["Action", "Thriller"]},
    {"title": "Alien (1979)",                       "genres": ["Horror", "Sci-Fi"]},
    {"title": "When Harry Met Sally (1989)",        "genres": ["Comedy", "Romance"]},
    {"title": "The Truman Show (1998)",             "genres": ["Comedy", "Drama"]},
    {"title": "A Bug's Life (1998)",                "genres": ["Animation", "Children", "Comedy"]},
    {"title": "Braveheart (1995)",                  "genres": ["Action", "Drama"]},
    {"title": "Saving Private Ryan (1998)",         "genres": ["Action", "Drama"]},
    {"title": "Star Wars: Episode IV (1977)",       "genres": ["Action", "Fantasy", "Sci-Fi"]},
    {"title": "Interstellar (2014)",                "genres": ["Drama", "Sci-Fi"]},
    {"title": "The Notebook (2004)",                "genres": ["Drama", "Romance"]},
    {"title": "Get Out (2017)",                     "genres": ["Horror", "Thriller"]},
    {"title": "The Avengers (2012)",                "genres": ["Action", "Sci-Fi"]},
    {"title": "Finding Nemo (2003)",                "genres": ["Animation", "Children", "Comedy"]},
]


def to_vec(genres: list[str]) -> list[float]:
    return [1.0 if g in genres else 0.0 for g in ALL_GENRES]


def build_model() -> onnx.ModelProto:
    # Genre matrix: (28, 10), then L2-normalise each row
    G = np.array([to_vec(m["genres"]) for m in MOVIES], dtype=np.float32)
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    G_norm = G / norms          # (28, 10) — each movie row is a unit vector
    G_norm_T = G_norm.T         # (10, 28) — transposed for MatMul

    # Initializers (constants baked into the graph)
    G_init = numpy_helper.from_array(G_norm_T, name="G_norm_T")
    k_init = numpy_helper.from_array(np.array([6], dtype=np.int64), name="k_val")

    # Graph (opset 12 — ReduceSum still takes axes as an attribute here)
    #
    # user_vec  (1,10)
    #   ├─ MatMul(G_norm_T)  →  dot        (1,28)  raw dot products
    #   ├─ Mul(self)         →  user_sq    (1,10)
    #       └─ ReduceSum axes=[1] keepdims →  user_sum  (1,1)
    #           └─ Sqrt      →  user_norm  (1,1)
    # Div(dot, user_norm)    →  cos_sim    (1,28)
    # TopK(cos_sim, k=6)     →  (top_scores, top_indices)  each (1,6)

    nodes = [
        make_node("MatMul",    ["user_vec", "G_norm_T"], ["dot"]),
        make_node("Mul",       ["user_vec", "user_vec"], ["user_sq"]),
        make_node("ReduceSum", ["user_sq"], ["user_sum"], axes=[1], keepdims=1),
        make_node("Sqrt",      ["user_sum"], ["user_norm"]),
        make_node("Div",       ["dot", "user_norm"], ["cos_sim"]),
        make_node("TopK",      ["cos_sim", "k_val"], ["top_scores", "top_indices"],
                  largest=1, sorted=1),
    ]

    inputs  = [make_tensor_value_info("user_vec",    TensorProto.FLOAT, [1, 10])]
    outputs = [
        make_tensor_value_info("top_scores",  TensorProto.FLOAT, [1, 6]),
        make_tensor_value_info("top_indices", TensorProto.INT64,  [1, 6]),
    ]

    graph = make_graph(nodes, "content_recommender", inputs, outputs,
                       initializer=[G_init, k_init])
    model = make_model(graph, opset_imports=[make_opsetid("", 12)])
    model.doc_string = (
        "Content-based movie recommender. "
        "Input: user genre vector [1,10] float32. "
        "Outputs: top_scores [1,6] float32, top_indices [1,6] int64."
    )
    return model


def main() -> None:
    model = build_model()
    onnx.checker.check_model(model)
    out = OUT / "content_recommender.onnx"
    out.write_bytes(model.SerializeToString())
    print(f"Wrote {out}  ({out.stat().st_size:,} bytes)")
    print("Movie order (index -> title):")
    for i, m in enumerate(MOVIES):
        print(f"  {i:2d}  {m['title']}")


if __name__ == "__main__":
    main()
