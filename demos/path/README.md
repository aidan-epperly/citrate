# Path: a roller coaster on the path component

A roller coaster with a vertical loop, built the way one would build it in Studio: a Folder of
Parts named P1, P2, ... (with a `Closed` attribute), which citrate's path component
(`roblox/path.luau`) turns into a smooth closed track. There are two copies of it, each with
a train of four cars. Both trains are placed by the same arclength; they differ only in how
the cars are turned:

- Left: `CFrame.lookAt(p, p + direction)`, which keeps the world's up, as is common. Through
  the loop the cars stay upright instead of rolling with the track: on its vertical parts
  they spin about the track, and at the top they ride on the outside, upright.
- Right: `path:follow`, which turns the cars with twist-free (rotation-minimizing) frames:
  they roll through the loop, upside down at its top.

The buttons ride along with either train.

The component:

```luau
local Path = require(ReplicatedStorage.Citrate.roblox.path)
local path = Path.new(workspace.Track)   -- through its Attachment or Part children
local mover = path:follow(workspace.Cart, { speed = 28, offset = CFrame.new(0, 0.9, 0) })
```

- `Path.new(source, options)`: `source` is an Instance whose Attachment and BasePart children
  are the points, in the order of their `Order` attribute or else by name (P2 before P10),
  or a list of Vector3s, CFrames, Attachments and Parts. The curve is a centripetal
  Catmull-Rom spline (`curve.catmull_rom`), which passes through the points without
  overshooting or looping between them. `path:rebuild()` rereads the points after they move.
- `path:position(s)` and `path:cframe(s)`: the point and the frame `s` studs along (wrapping
  on a closed path). The frame's LookVector is along the path and its UpVector the
  twist-free normal (`curve.framing`).
- `path:closest(position, near?)`: the distance along the path of the point closest to
  `position`, the point and the distance to it; with `near` (such as last frame's answer)
  it searches locally, which is much faster.
- `path:follow(target, options)`: moves a Model or a part by its pivot at `speed` studs per
  second, each Heartbeat (or call `mover:step(dt)` yourself with `manual = true`). At the end
  of an open path it loops, bounces or stops (`mode`).

In Studio the four cars took 0.026 ms per frame with `path:follow` and 0.04 ms with
`lookAt` (which evaluates two points per car). Outside Roblox a frame (`framing:at`) costs
0.64 us natively and 1.8 us interpreted, as players' devices run scripts.

Files:

- `client.client.luau`: the Roblox client script.
- `path.project.json`: builds a playable place with `rojo build demos/path/path.project.json
  -o path.rbxl` (open it in Roblox Studio and press Play).
