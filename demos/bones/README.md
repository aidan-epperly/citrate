# Bones: dynamic bones for Roblox

Three creatures walk in circles, each with a tail of 8 bones and a stalk that forks into two
antennae, all real `Bone` instances moved by citrate's dynamic bones (`roblox/bones.luau`).
The left one is updated every frame, the middle one every second frame and the right one
every third, as the component throttles rigs farther from the camera. Each has a ball on its
path that its tail drags over (a collider), and the button turns `workspace.GlobalWind` on.

The component (`roblox/bones.luau`):

```luau
local Bones = require(ReplicatedStorage.Citrate.roblox.bones)
local rig = Bones.new(tail_root_bone, { colliders = { ball } })
RunService.Heartbeat:Connect(function(dt) rig:step(dt) end)
-- or: tag parts "CitrateBones" and call Bones.start() once on the client
```

- Each bone is a particle; the root follows its animated pose, every other bone hangs from
  its parent by a rigid rod and is pulled toward where the animation puts it, with drag
  toward the wind and gravity, by citrate's XPBD solver (`citrate.xpbd`).
- The simulation runs in substeps of a fixed length (1/120 s by default), so a rig moves
  the same at any frame rate, and updating a distant rig less often does not change how it
  moves (SmartBone's damping and droop change with its update rate; see `demos/tails`).
- The defaults were fitted to move like SmartBone's tails at 60 Hz (`demos/tails`,
  `look.luau`, by citrate's least squares): bones much lighter toward the tips (each 0.07
  of its parent: a whip), a spring toward the animated pose of about the same strength
  along the tail, drag 6.8 per second and little other damping. The fit was on a tail of 6
  bones; a longer chain spreads the same tip mass over its length (at 0.07 a bone, the 8th
  bone would weigh 1e-8 of the first and flail: its joint bent up to 109 degrees while
  walking, against at most 4 with the mass spread).
- Colliders: balls are spheres, cylinders capsules, other parts the sphere of their
  smallest half size; with friction 0.3 by default.
- Writing back: each bone is placed at its particle and turned from its animated pose by
  the rotation taking the direction to its first child onto the simulated one. The rest
  poses are kept, so `rig:reset()` (and `destroy`) restore them.

In Studio a creature (two rigs, 18 bones) took about 0.04 to 0.05 ms per update natively.
Outside Roblox the tuned 6-bone tail takes 5.4 us per 60 Hz update interpreted (as
players' devices run scripts) and 1.2 us natively (`demos/tails`).

Files:

- `client.client.luau`: the Roblox client script (the creatures, segments drawn along the
  bones, the wind button).
- `bones.project.json`: builds a playable place with `rojo build
  demos/bones/bones.project.json -o bones.rbxl` (open it in Roblox Studio and press Play).
