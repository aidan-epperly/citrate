# Rotations

Reference: [rotation](reference/rotation.md).

```luau
local rotation = citrate.rotation
local q = rotation.axis_angle(vector.create(0, 1, 0), math.pi / 2)   -- a quaternion { w, x, y, z }
local c = rotation.axis_angle(vector.create(0, 1, 0), math.pi / 2, rotation.cframe)
local r = rotation.slerp(c, CFrame.identity, 0.25)                   -- CFrames in, a CFrame out
local S = rotation.spline({ c1, c2, c3, c4 })                         -- smooth, through the keys
local mean = rotation.mean({ q1, q2, q3 })
q = rotation.integrate(q, omega, dt)                                  -- by an angular velocity
```

## Representations: rotation spaces

The module is written against a rotation space, as the rest of the library is written
against a `Space`: a representation supplies five primitives (compose, inverse, apply to
a point, and the exponential and logarithm maps between rotations and rotation vectors),
and slerp, Euler angles, splines, means, fits and integration are built from those alone.
A value is never converted to some internal form: CFrames are worked on by CFrame
methods, quaternions by quaternion arithmetic. The spaces provided:

| space | values | primitives |
| --- | --- | --- |
| `rotation.quaternion` | `{ w, x, y, z }` | quaternion arithmetic on numbers, double precision |
| `rotation.frame` | `{ px, py, pz, w, x, y, z }`: a position and a rotation | the same, with positions |
| `rotation.cframe` (Roblox only) | CFrames | `a * b`, `:Inverse()`, `a * v`, `CFrame.fromAxisAngle`, `:Lerp`; single precision, as CFrames are |

Functions find the space from their arguments; constructors (`identity`, `axis_angle`,
`exp`, `between`, `from_euler`, `from_matrix`) take one (default `rotation.quaternion`); and
any function takes a space of one's own as its last argument (`options.space` for
splines), such as rotation matrices, with only the five primitives required. The tests
build a spline in such a space. `rotation.convert(r, space)` changes representation when
you want it to.

The one place a CFrame primitive is not Roblox's own is the logarithm: `ToAxisAngle`
computes the angle by an arccosine, which loses it near zero (spline angles came out
3e-3 too small), so the CFrame space takes it from the matrix by Shepperd's method.

Cost in Studio, against Roblox's own operations:

| | Roblox | `rotation` | `rotation.bind(space)` |
| --- | --- | --- | --- |
| compose two CFrames | 32 ns (`a * b`) | 61 ns | 46 ns |
| slerp | 87 to 103 ns (`:Lerp`) | 122 ns | as `:Lerp` (it is `:Lerp`: 102 ns against 103 in one run) |

The difference is finding the space (which `bind` skips: `local R =
rotation.bind(rotation.cframe)`, then `R.slerp(a, b, t)`) and calling through the space's
table. For many frames as plain numbers, a buffer of 7 numbers a frame
(`rotation.frames(n)`, `get`, `set`, `compose_into`) avoids tables altogether.

A space may also supply `compose_exp3(a, v1, v2, v3)`, a exp(v1) exp(v2) exp(v3) in one
call: that is a whole spline evaluation (below), which otherwise takes six calls. The
CFrame space does it as one quaternion product in numbers, one CFrame and one CFrame
product; the quaternion space with one table instead of six. Spline evaluation (`S:at`,
four keys):

| | before (Shoemake's, 6 slerps) | cumulative form, generic | with `compose_exp3` |
| --- | --- | --- | --- |
| smooth, CFrames (Studio) | 708 ns | 420 ns | 218 ns |
| smooth, quaternions (natively) | 874 ns | 371 ns | 173 ns |
| B-spline, CFrames (Studio) | | 422 ns | 253 ns |
| B-spline, quaternions (natively) | 406 ns | 407 ns | 212 ns |

Conventions match Roblox's: `compose(a, b)` is `a * b` (b first), `slerp` is `:Lerp`,
`from_euler(rx, ry, rz, order)` is `CFrame.fromEulerAngles`, and angular velocities are in
world coordinates, as `AssemblyAngularVelocity`.

## Interpolating through orientations

`rotation.spline(keys, { kind, times, closed })` interpolates orientations (frames' and
CFrames' positions are carried along between keys by the space's slerp; for a path, use a
[curve](curves.md)):

| `kind` | through the keys | angular velocity | angular acceleration |
| --- | --- | --- | --- |
| `"smooth"` (default) | yes | continuous | jumps at keys |
| `"slerp"` | yes | constant between keys, jumps at them | |
| `"bspline"` | near them | continuous | continuous (even times only) |

`"smooth"` is a cubic Bezier curve of rotations on each interval, in the cumulative form of
Kim, Kim and Shin (q exp(c1 w1) exp(c2 w2) exp(c3 w3), the w_j between consecutive control
points, fixed per interval), with the end angular velocities of a Catmull-Rom spline
(weighted for uneven times), so it turns smoothly through every key. Shoemake's form (the
same control points, evaluated by six slerps) is a slightly different curve, within 0.03
radians of this one on random keys, and took two to four times as long. `"bspline"` is the cumulative B-spline of Kim, Kim and Shin.
`S:angular_velocity(t)` gives the angular velocity, `S.angle` the total angle turned, and
`S:at_angle(s)` the orientation after turning through angle s: a constant angular speed
along any spline (Chebyshev interpolants of the angular speed on each interval, inverted by
Newton's method; checked constant to 1e-7 for quaternions).

## Integrating angular velocity

`rotation.integrate(q, omega, dt)` advances q under a constant angular velocity exactly:
q becomes exp(omega dt) q. The common update q + dt omega q / 2, normalized, is first order
and drifts with the frame rate. When omega changes during the step, pass its end value too
(`integrate(q, omega, dt, omega_end)`): the fourth-order Magnus step, whose error fell 16
times when the step halved (tested).

## Several rotations

- `rotation.mean(list, weights)`: the average orientation, the geodesic (Karcher) mean,
  found from the primitives by m <- m exp(mean of log(m^-1 r_i)) in a few iterations (it
  needs the rotations within a half turn of each other). Averaging components or Euler
  angles is wrong when the rotations are far apart. Frames' positions are averaged too.
- `rotation.fit(a, b, weights)`: the rigid motion best taking points a_i to b_i (Horn's
  method), with the root-mean-square distance left: for aligning shapes, or recovering a
  body's frame from tracked points.

## Euler angles

`from_euler(rx, ry, rz, order)` and `to_euler(q, order)` for the six orders ("XYZ", the
default, is Rx Ry Rz, as `CFrame.Angles`). At gimbal lock (the middle angle at +-pi/2) the
last angle is set to 0 and the rotation is still reproduced exactly.
