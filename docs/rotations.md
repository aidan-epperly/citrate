# Rotations

Reference: [rotation](reference/rotation.md).

```luau
local rotation = citrate.rotation
local q = rotation.axis_angle(vector.create(0, 1, 0), math.pi / 2)   -- { w, x, y, z }
local v = rotation.apply(q, vector.create(1, 0, 0))                   -- (0, 0, -1)
local r = rotation.slerp(q, rotation.identity(), 0.25)
local S = rotation.spline({ q1, q2, q3, q4 })                         -- smooth, through the keys
local mean = rotation.mean({ q1, q2, q3 })
q = rotation.integrate(q, omega, dt)                                  -- by an angular velocity
```

## Quaternions, frames and CFrames

Every function takes quaternions (`{ w, x, y, z }`), frames (`{ px, py, pz, w, x, y, z }`:
a position and a rotation) and, in Roblox, CFrames, in any mix, and returns the kind of its
first argument. Constructors (`identity`, `axis_angle`, `exp`, `between`, `from_euler`,
`from_matrix`) take the kind wanted as an optional last argument: `"quat"` (the default),
`"frame"` or `"cframe"`.

The arithmetic is done on plain numbers in double precision. Measured outside Roblox,
natively, composing two rotations stored as tables took 61 ns (most of it creating the
result table), as 7-number buffers 16 ns, and on numbers 16 ns; `vector`s would be as fast
but hold single precision. So for many frames, a buffer of 7 numbers a frame avoids the
tables: `rotation.frames(n)`, `get`, `set` and `compose_into` (46 ns in Studio).

For a CFrame, Roblox's own operations are faster: in Studio `a * b` took 61 ns against 286
for `rotation.mul(a, b)`, which converts to a quaternion and back. Use CFrame arithmetic
for single products, and this module for what CFrames do not have: splines, means, fits,
integration, Euler angles in every order. Its conventions match Roblox's: `mul(a, b)` is
`a * b`, `slerp` is `:Lerp` (to 2e-7, the CFrame's own precision), `from_euler(rx, ry, rz,
order)` is `CFrame.fromEulerAngles`, and angular velocities are in world coordinates, as
`AssemblyAngularVelocity`.

## Interpolating through orientations

`rotation.spline(keys, { kind, times, closed })` interpolates orientation only (for
frames and CFrames, positions belong to a [curve](curves.md)):

| `kind` | through the keys | angular velocity | angular acceleration |
| --- | --- | --- | --- |
| `"smooth"` (default) | yes | continuous | jumps at keys |
| `"slerp"` | yes | constant between keys, jumps at them | |
| `"bspline"` | near them | continuous | continuous (even times only) |

`"smooth"` is Shoemake's spherical Bezier curve on each interval, evaluated by slerps, with
the end angular velocities of a Catmull-Rom spline (weighted for uneven times), so it turns
smoothly through every key. `"bspline"` is the cumulative B-spline of Kim, Kim and Shin.
`S:angular_velocity(t)` gives the angular velocity, `S.angle` the total angle turned, and
`S:at_angle(s)` the orientation after turning through angle s: a constant angular speed
along any spline (Chebyshev interpolants of the angular speed on each interval, inverted by
Newton's method; checked constant to 1e-7).

## Integrating angular velocity

`rotation.integrate(q, omega, dt)` advances q under a constant angular velocity exactly:
q becomes exp(omega dt) q. The common update q + dt omega q / 2, normalized, is first order
and drifts with the frame rate. When omega changes during the step, pass its end value too
(`integrate(q, omega, dt, omega_end)`): the fourth-order Magnus step, whose error fell 16
times when the step halved (tested).

## Several rotations

- `rotation.mean(list, weights)`: the average orientation (Markley's method, the top
  eigenvector of the sum of q q^T). Averaging components or Euler angles is wrong when the
  rotations are far apart, and depends on each quaternion's sign; this does not.
- `rotation.fit(a, b, weights)`: the rigid motion best taking points a_i to b_i (Horn's
  method), with the root-mean-square distance left: for aligning shapes, or recovering a
  body's frame from tracked points.

## Euler angles

`from_euler(rx, ry, rz, order)` and `to_euler(q, order)` for the six orders ("XYZ", the
default, is Rx Ry Rz, as `CFrame.Angles`). At gimbal lock (the middle angle at +-pi/2) the
last angle is set to 0 and the rotation is still reproduced exactly.
