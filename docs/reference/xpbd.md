# Position-based dynamics

Particles and compliant constraints by extended position-based dynamics (XPBD) with substeps.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `new`

```luau
new(positions: { vector }, options: { mass: (number | { number })?, gravity: vector?, drag: number? }?): World
```

A world of particles at the given positions (index i is particle i). options: mass (a number, or one per particle; default 1), gravity (default 0), drag (per second; default 0). Its methods: world:distance(a, b, compliance, damping?, length?)  keeps particles a and b `length` apart (default: as far apart as they are now); compliance 0 makes it rigid world:offset(a, b, d, compliance, damping?)  keeps x_b - x_a = d (a vector): a spring toward a rest shape, along each axis world:pin(i)  makes particle i kinematic: constraints do not move it, world:move does world:move(i, p)  where kinematic particle i is to be at the end of the next step (it moves there in a straight line over the substeps) world:step(dt, substeps?)  advances dt in `substeps` substeps (default 1), each with one sweep over the constraints in the order they were added world:position(i), world:velocity(i)  as vectors and fields x, v (positions and velocities, 3 per particle, f64), t and stats (steps, substeps).
