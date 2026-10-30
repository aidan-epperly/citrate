# Rotations

Quaternions, frames and CFrames: conversions, Euler angles, slerp, rotation splines, means, point-set fits, integration of angular velocity.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `quat`

```luau
quat(w: number, x: number, y: number, z: number): Quat
```

A unit quaternion from components (normalized).

### `identity`

```luau
identity(kind: string?): any
```

### `frame`

```luau
frame(position: any, r: any, kind: string?): any
```

A frame from a position and a rotation (any kind; its own position is replaced).

### `axis_angle`

```luau
axis_angle(axis: any, angle: number, kind: string?): any
```

The rotation by angle about axis (any nonzero length).

### `exp`

```luau
exp(v: any, kind: string?): any
```

The rotation by the rotation vector v (angle |v| about v): the inverse of log.

### `between`

```luau
between(a: any, b: any, kind: string?): any
```

The shortest rotation taking direction a to direction b (about a x b; for opposite directions, a half turn about some axis perpendicular to a).

### `from_matrix`

```luau
from_matrix(R: any, kind: string?): any
```

From a rotation matrix: rows { { r00, r01, r02 }, ... } or a 3 by 3 f64 Matrix.

### `matrix`

```luau
matrix(r: any): { { number } }
```

The rotation matrix, as rows.

### `from_euler`

```luau
from_euler(rx: number, ry: number, rz: number, order: string?, kind: string?): any
```

From Euler angles about X, Y and Z (radians), applied as the order says: "XYZ" (the default) is Rx(rx) Ry(ry) Rz(rz), as CFrame.fromEulerAngles(rx, ry, rz, order).

### `to_euler`

```luau
to_euler(r: any, order: string?): (number, number, number)
```

The Euler angles (rx, ry, rz) of a rotation for the order (see from_euler); the middle angle is in [-pi/2, pi/2], and at its ends (gimbal lock) the last is taken as 0.

### `mul`

```luau
mul(a: any, b: any): any
```

a b: b, then a (as a * b for CFrames; for frames, positions compose too).

### `inverse`

```luau
inverse(r: any): any
```

### `apply`

```luau
apply(r: any, p: any): vector
```

The point p rotated (and, for a frame or CFrame, moved): a vector.

### `angle`

```luau
angle(a: any, b: any): number
```

The angle of the rotation from a to b, in [0, pi].

### `log`

```luau
log(r: any): vector
```

The rotation vector (angle times axis; angle in [0, pi]).

### `slerp`

```luau
slerp(a: any, b: any, t: number): any
```

From a toward b at fraction t (0: a, 1: b), along the shorter arc at constant angular speed; frames' positions move linearly.

### `angular_velocity`

```luau
angular_velocity(a: any, b: any, dt: number): vector
```

The angular velocity (world coordinates) that turns a into b in time dt at a constant rate: log(b a*) / dt.

### `integrate`

```luau
integrate(r: any, omega: any, dt: number, omega_end: any?): any
```

r advanced by dt under angular velocity omega (world coordinates; dq/dt = omega q / 2): exactly for a constant omega. With omega_end (omega at the end of the step, omega changing linearly between), by the fourth-order Magnus step: the rotation vector dt (omega + omega_end) / 2 + dt^2 / 12 (omega_end x omega). Frames keep their position.

### `mean`

```luau
mean(rotations: { any }, weights: { number }?): any
```

The weighted mean rotation (Markley et al., 2007): the unit quaternion maximizing sum w_i (q . q_i)^2, the top eigenvector of sum w_i q_i q_i^T, which does not depend on the signs of the q_i. Of the first rotation's kind.

### `fit`

```luau
fit(a: { any }, b: { any }, weights: { number }?, kind: string?): (any, number)
```

The rigid motion best taking points a_i to b_i (least squares, weighted): Horn's method, the rotation from the top eigenvector of a 4 by 4 matrix of the centered points' cross covariances, and the translation matching the centroids. Points: vectors or { x, y, z }. Returns a frame (or kind), and the root-mean-square distance remaining.

### `spline`

```luau
spline(keys: { any }, options: SplineOptions?): RotationSpline
```

A spline through (or near) rotations at times (default 0, 1, 2, ...); the keys may be of any kind, and the spline returns the first one's kind (frames and CFrames at the first key's position: it interpolates orientation only). options.kind: "smooth" (the default)  through every key, with continuous angular velocity: on each interval a Bezier curve on the sphere (Shoemake's, evaluated by slerps), its end angular velocities those of the neighbors (Catmull-Rom, weighted for uneven times); "slerp"                 through every key at constant angular speed between keys (the angular velocity jumps at each); "bspline"               a cumulative cubic B-spline (Kim, Kim and Shin, 1995): angular acceleration continuous too, but it passes near the keys, not through them; even times only. closed = true joins the last key to the first. S:at(t), S:angular_velocity(t) (world coordinates), S.angle (the total angle turned) and S:at_angle(s) (the rotation after turning through angle s: constant angular speed along any spline).

### `frames`

```luau
frames(n: number): buffer
```

A buffer of n frames, each at the origin with no rotation.

### `get`

```luau
get(b: buffer, i: number, kind: string?): any
```

Frame i of a buffer, as a frame (or kind).

### `set`

```luau
set(b: buffer, i: number, r: any)
```

Store a rotation, frame or CFrame as frame i.

### `compose_into`

```luau
compose_into(out: buffer, i: number, A: buffer, j: number, B: buffer, k: number)
```

Frame i of out := (frame j of A)(frame k of B), without tables (out may be A or B).

### `components`

```luau
components(r: any): (number, number, number, number, number, number, number)
```

A value's quaternion and position as numbers (w, x, y, z, px, py, pz), for code that composes many rotations without tables.
