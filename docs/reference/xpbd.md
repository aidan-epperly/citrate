# Position-based dynamics

Particles and compliant constraints by extended position-based dynamics (XPBD) with substeps.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `new`

```luau
new(positions: { vector } | buffer, options: { space: any?, mass: (number | { number })?, radius: (number | { number })?, gravity: vector?, drag: number?, wind: vector?, friction: number? }?): any
```

A world of particles at the given positions (index i is particle i): a list of vectors, or a buffer (x, y, z per particle) of options.space's precision. options: space (f64.space or f32.space: keep the positions in a buffer of that precision; default: as given), mass (a number, or one per particle; default 1), radius (for collisions; a number or one per particle; default 0), gravity, drag (per second), wind (the air's velocity, the drag's reference; default 0), friction (the fraction of a particle's motion along a collider's surface removed while touching it; default 0). Its methods: world:distance(a, b, compliance, damping?, length?)  keeps particles a and b `length` apart (default: as far apart as they are now); compliance 0 makes it rigid world:offset(a, b, d, compliance, damping?)  keeps x_b - x_a = d (a vector): a spring toward a rest shape, along each axis (both return the constraint's number k, for world:set_offset(k, d), world:set_length(k, length) and world:set_compliance(k, compliance, damping?), which change it between steps: a rest shape that turns with an animated parent) world:pin(i)  makes particle i kinematic: constraints do not move it, world:move does world:move(i, p)  where kinematic particle i is to be at the end of the next step (it moves there in a straight line over the substeps) world:sphere(center, r), world:capsule(a, b, r), world:plane(point, normal)  add a collider and return it; change its fields (a, b, radius) to move it world:step(dt, substeps?)  advances dt in `substeps` substeps (default 1), each with one sweep over the constraints in the order they were added world:position(i), world:velocity(i)  as vectors and fields x, v (positions and velocities in the world's storage), gravity, drag, wind, friction (settable between steps), colliders, t and stats (steps, substeps).
