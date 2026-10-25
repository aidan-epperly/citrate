# Lightning Laplace solver

Laplace's and Poisson's equations on polygons by rational functions (Gopal and Trefethen).

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `laplace`

```luau
laplace(vertices: { { number } }, conditions: { Condition }, options: LightningOptions?): LightningSolution
```

Solves Laplace's equation (Poisson's, with options.source) on the polygon with the given vertices ({ x, y } pairs, counterclockwise); conditions[i] holds on the side from vertex i to vertex i + 1 (the last side closing the polygon).
