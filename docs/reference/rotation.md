# Rotations

Rotations in any representation (quaternions, frames, CFrames, your own) through rotation spaces: Euler angles, slerp, rotation splines, means, point-set fits, integration of angular velocity.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `quaternion`

```luau
quaternion: RotationSpace<Quat>
```

### `frame`

```luau
frame: RotationSpace<Frame>
```

### `cframe`

```luau
cframe -- (a value)
```

### `quat`

```luau
quat(w: number, x: number, y: number, z: number): Quat
```

A unit quaternion from components (normalized).

### `identity`

```luau
identity(space: any?): any
```

### `exp`

```luau
exp(v: any, space: any?): any
```

The rotation by the rotation vector v (angle |v| about v): the inverse of log.

### `axis_angle`

```luau
axis_angle(axis: any, angle: number, space: any?): any
```

The rotation by angle about axis (any nonzero length).

### `between`

```luau
between(a: any, b: any, space: any?): any
```

The shortest rotation taking direction a to direction b (about a x b; for opposite directions, a half turn about an axis perpendicular to a).

### `from_matrix`

```luau
from_matrix(R: any, space: any?): any
```

From a rotation matrix: rows { { r00, r01, r02 }, ... } or a 3 by 3 f64 Matrix.

### `matrix`

```luau
matrix(r: any, space: any?): { { number } }
```

The rotation matrix, as rows.

### `from_euler`

```luau
from_euler(rx: number, ry: number, rz: number, order: string?, space: any?): any
```

From Euler angles about X, Y and Z (radians), composed as the order says: "XYZ" (the default) is Rx(rx) Ry(ry) Rz(rz), as CFrame.fromEulerAngles(rx, ry, rz, order).

### `to_euler`

```luau
to_euler(r: any, order: string?, space: any?): (number, number, number)
```

The Euler angles (rx, ry, rz) for the order (see from_euler); the middle angle is in [-pi/2, pi/2], and at its ends (gimbal lock) the last is taken as 0.

### `compose`

```luau
compose(a: any, b: any, space: any?): any
```

a b: b, then a (a * b for CFrames; for frames, positions compose too).

### `inverse`

```luau
inverse(r: any, space: any?): any
```

### `apply`

```luau
apply(r: any, p: any, space: any?): vector
```

The point p rotated (and, for a frame or CFrame, moved): a vector.

### `angle`

```luau
angle(a: any, b: any, space: any?): number
```

The angle of the rotation from a to b, in [0, pi].

### `log`

```luau
log(r: any, space: any?): vector
```

The rotation vector (angle times axis; angle in [0, pi]).

### `slerp`

```luau
slerp(a: any, b: any, t: number, space: any?): any
```

From a toward b at fraction t (0: a, 1: b), along the shorter arc at constant angular speed (frames' and CFrames' positions move linearly, as CFrame:Lerp).

### `angular_velocity`

```luau
angular_velocity(a: any, b: any, dt: number, space: any?): vector
```

The angular velocity (world coordinates) that turns a into b in time dt at a constant rate: log(b a^-1) / dt.

### `integrate`

```luau
integrate(r: any, omega: any, dt: number, omega_end: any?, space: any?): any
```

r advanced by dt under angular velocity omega (world coordinates; dq/dt = omega q / 2): exp(omega dt) r, exact for a constant omega. With omega_end (omega at the step's end, changing linearly between), the fourth-order Magnus step: the rotation vector dt (omega + omega_end) / 2 + dt^2 / 12 (omega_end x omega). Positions stay.

### `mean`

```luau
mean(rotations: { any }, weights: { number }?, space: any?): any
```

The weighted geodesic (Karcher) mean: the rotation m with sum w_i log(m^-1 r_i) = 0, found by m <- m exp(the weighted mean of the log(m^-1 r_i)) until that step stops shrinking (a few iterations; it needs the rotations within a half turn of each other). Frames' positions are averaged too.

### `fit`

```luau
fit(a: { any }, b: { any }, weights: { number }?, space: any?): (any, any, number?)
```

The rigid motion best taking points a_i to b_i (least squares, weighted): Horn's method, the rotation from the top eigenvector of a 4 by 4 matrix of the centered points' cross covariances, built in the space by exp. Returns the motion (with its translation, in a space with positions: rotation.frame by default) and the root-mean-square distance remaining; in a space without positions, the rotation, the translation and the distance.

### `convert`

```luau
convert(r: any, space: any, from: any?): any
```

A value in another space (through its rotation as a quaternion and its position; both spaces need components and from_components).

### `place`

```luau
place(r: any, position: any, space: any?): any
```

r placed at a position (keeping its rotation): in r's space if that has positions, else (for a quaternion) a frame.

### `spline`

```luau
spline(keys: { any }, options: SplineOptions?): RotationSpline
```

A spline through (or near) rotations at times (default 0, 1, 2, ...), in the keys' space (options.space for your own). options.kind: "smooth" (the default)  through every key, with continuous angular velocity: on each interval a Bezier curve on the sphere (Shoemake's, evaluated by slerps), its end angular velocities those of the neighbors (Catmull-Rom, weighted for uneven times); "slerp"                 through every key at constant angular speed between keys (the angular velocity jumps at each); "bspline"               a cumulative cubic B-spline (Kim, Kim and Shin, 1995): angular acceleration continuous too, but it passes near the keys, not through them; even times only. closed = true joins the last key to the first. S:at(t), S:angular_velocity(t) (world coordinates), S.angle (the total angle turned) and S:at_angle(s) (the rotation after turning through angle s: constant angular speed along any spline). Positions of frames and CFrames are carried along by the space's slerp (between keys' positions); for a path, interpolate positions with a curve.

### `frames`

```luau
frames(n: number): buffer
```

A buffer of n frames, each at the origin with no rotation.

### `get`

```luau
get(b: buffer, i: number, space: any?): any
```

Frame i of a buffer, in a space (default rotation.frame).

### `set`

```luau
set(b: buffer, i: number, r: any, space: any?)
```

Store a value as frame i.

### `compose_into`

```luau
compose_into(out: buffer, i: number, A: buffer, j: number, B: buffer, k: number)
```

Frame i of out := (frame j of A)(frame k of B), without tables (out may be A or B).
