--!strict
--!native
-- Dynamic bones for Roblox: Bones (in a MeshPart's skeleton, or under any part) that swing,
-- droop and whip as their part moves, simulated by citrate's XPBD solver. Unlike updating
-- once per frame with per-frame damping (as SmartBone does), the simulation runs in
-- substeps of a fixed length, so a rig moves the same at any frame rate, and throttling a
-- distant rig only updates its bones less often, without changing how it moves.
--
-- Each bone is a particle at its tip (its position). The root bone follows its animated
-- pose; every other bone hangs from its parent by a rigid rod and is pulled toward where
-- the animation puts it relative to its parent (a spring), with drag toward the wind and
-- gravity. Bones get lighter toward the tips (taper), which makes a tail whip. The defaults were fitted to move like SmartBone's tails at 60 Hz
-- (demos/tails, look.luau).
--
--   local Bones = require(Citrate.roblox.bones)
--   local rig = Bones.new(tail_root_bone)        -- simulate every bone under it
--   RunService.Heartbeat:Connect(function(dt) rig:step(dt) end)
--
-- or tag BaseParts "CitrateBones" and call Bones.start() once on the client: each tagged
-- part's top-level bones (or the bones named in its "Roots" attribute, comma separated)
-- are simulated, nearer the camera more often; attributes override the options below.

local CollectionService = game:GetService("CollectionService")
local RunService = game:GetService("RunService")

local citrate = require("../src")
local xpbd = citrate.xpbd

export type Options = {
	stiffness: number?, -- the spring toward the animated pose at the root's children (1 / compliance); default 208
	stiffness_falloff: number?, -- its factor from each bone to the next (default 1 / 0.9952)
	spring_damping: number?, -- default 3.8e-6 (drag does the damping)
	drag: number?, -- per second, toward the wind (default 6.78)
	taper: number?, -- each bone's mass relative to its parent's (default 0.070), as fitted on a
	-- 6-bone tail; a longer chain spreads the same tip mass over its length (at 0.070 per
	-- bone an 8-bone tail's tip would weigh 1e-8 of its root and flail)
	gravity: Vector3?, -- default (0, -24.8, 0), as SmartBone's
	wind: Vector3?, -- the air's velocity; default workspace.GlobalWind if global_wind, else 0
	global_wind: boolean?, -- follow workspace.GlobalWind each step (default true)
	substep_rate: number?, -- substeps per second (default 120)
	radius: number?, -- the particles' radius for collisions (default 0.1 studs)
	colliders: { BasePart }?, -- balls are spheres, cylinders capsules (along their X axis), other parts the sphere of their smallest half size
	friction: number?, -- default 0.3
}

local DEFAULTS = {
	stiffness = 1 / 0.004811,
	stiffness_falloff = 1 / 0.9952,
	spring_damping = 3.844e-6,
	drag = 6.776,
	taper = 0.07022,
	gravity = Vector3.new(0, -24.8, 0),
	global_wind = true,
	substep_rate = 120,
	radius = 0.1,
	friction = 0.3,
}

export type Rig = {
	root: Bone,
	bones: { Bone },
	world: any,
	step: (self: Rig, dt: number) -> (),
	reset: (self: Rig) -> (),
	destroy: (self: Rig) -> (),
	stats: { steps: number, substeps: number },
}

-- the defaults were fitted on a tail of 6 bones: its tip is taper^5 of its first bone
local TAPER_BONES = 5

local function opt(o: Options, key: string): any
	local v = (o :: any)[key]
	if v == nil then
		return (DEFAULTS :: any)[key]
	end
	return v
end

-- The animated world pose of a bone: its parent's (bone or part) times its rest CFrame times
-- its animation Transform.
local function parent_cframe(b: Instance, world: { [Instance]: CFrame }): CFrame
	local p = b.Parent
	if p and world[p] then
		return world[p]
	elseif p and p:IsA("BasePart") then
		return (p :: BasePart).CFrame
	end
	return CFrame.identity
end

local Bones = {}

-- A rig simulating `root` and every Bone under it.
function Bones.new(root: Bone, options: Options?): Rig
	local o: Options = options or {}
	-- the bones, parents first, their rest CFrames (kept: writing WorldCFrame changes CFrame),
	-- depths and parents' indices
	local bones, rest, parent, depth = { root }, { [root :: Instance] = root.CFrame }, { 0 }, { 0 }
	local index = { [root :: Instance] = 1 }
	local k = 1
	while k <= #bones do
		for _, c in bones[k]:GetChildren() do
			if c:IsA("Bone") then
				table.insert(bones, c)
				index[c] = #bones
				rest[c] = c.CFrame
				parent[#bones] = k
				depth[#bones] = depth[k] + 1
			end
		end
		k += 1
	end
	local n = #bones

	-- the animated pose now
	local anim: { [Instance]: CFrame } = {}
	local function animate()
		for i = 1, n do
			local b = bones[i]
			anim[b] = parent_cframe(b, anim) * rest[b] * b.Transform
		end
	end
	animate()

	local positions, masses = {}, {}
	local deepest = 0
	for i = 1, n do
		deepest = math.max(deepest, depth[i])
	end
	-- the taper's exponent, scaled so no bone is lighter than a fitted 6-bone tail's tip
	local spread = math.min(1, TAPER_BONES / math.max(deepest - 1, 1))
	for i = 1, n do
		local p = anim[bones[i]].Position
		positions[i] = vector.create(p.X, p.Y, p.Z)
		masses[i] = (opt(o, "taper") :: number) ^ (math.max(depth[i] - 1, 0) * spread)
	end
	local gravity: Vector3 = opt(o, "gravity")
	local world = xpbd.new(positions, {
		mass = masses,
		radius = opt(o, "radius"),
		gravity = vector.create(gravity.X, gravity.Y, gravity.Z),
		drag = opt(o, "drag"),
		friction = opt(o, "friction"),
	})
	world:pin(1)
	local springs = {}
	for i = 2, n do
		local a = parent[i]
		world:distance(a, i, 0)
		local stiffness = (opt(o, "stiffness") :: number) * (opt(o, "stiffness_falloff") :: number) ^ (depth[i] - 1)
		local d = anim[bones[i]].Position - anim[bones[a]].Position
		springs[i] = world:offset(a, i, vector.create(d.X, d.Y, d.Z), 1 / (stiffness * masses[i]), opt(o, "spring_damping"))
	end
	-- colliders
	local shapes = {}
	for _, part in (o.colliders or {}) :: { BasePart } do
		table.insert(shapes, { part = part, c = world:sphere(vector.zero, 0) })
	end
	local function place_colliders()
		for _, s in shapes do
			local part, c = s.part, s.c
			local cf, size = part.CFrame, part.Size
			if part:IsA("Part") and (part :: Part).Shape == Enum.PartType.Cylinder then
				local half = math.max(size.X / 2 - size.Y / 2, 0)
				local a, b = cf * Vector3.new(-half, 0, 0), cf * Vector3.new(half, 0, 0)
				c.kind, c.a, c.b, c.radius = 2, vector.create(a.X, a.Y, a.Z), vector.create(b.X, b.Y, b.Z), size.Y / 2
			else
				local p = cf.Position
				c.kind, c.a, c.b = 1, vector.create(p.X, p.Y, p.Z), vector.create(p.X, p.Y, p.Z)
				c.radius = math.min(size.X, size.Y, size.Z) / 2
			end
		end
	end

	local rig = { root = root, bones = bones, world = world, stats = world.stats }
	local carry = 0
	function rig.step(self: Rig, dt: number)
		animate()
		local rp = anim[root].Position
		world:move(1, vector.create(rp.X, rp.Y, rp.Z))
		for i = 2, n do
			local d = anim[bones[i]].Position - anim[bones[parent[i]]].Position
			world:set_offset(springs[i], vector.create(d.X, d.Y, d.Z))
		end
		place_colliders()
		if opt(o, "global_wind") and o.wind == nil then
			local w = workspace.GlobalWind
			world.wind = vector.create(w.X, w.Y, w.Z)
		elseif o.wind then
			world.wind = vector.create(o.wind.X, o.wind.Y, o.wind.Z)
		end
		-- whole substeps of the fixed length; the remainder carries to the next step
		local h = 1 / (opt(o, "substep_rate") :: number)
		carry += dt
		local steps = math.floor(carry / h + 1e-9)
		if steps < 1 then
			return
		end
		steps = math.min(steps, 8 * math.ceil(1 / (60 * h))) -- after a hitch, drop the backlog
		carry = math.max(carry - steps * h, 0) % h
		world:step(steps * h, steps)
		-- write back: each bone at its particle, turned from its animated pose by the rotation
		-- that takes its animated direction to its first child onto the simulated one
		local turned: { [number]: CFrame } = {}
		for i = 1, n do
			local b = bones[i]
			local pv = world:position(i)
			local pos = Vector3.new(pv.x, pv.y, pv.z)
			local rot = anim[b].Rotation
			local child = 0
			for j = i + 1, n do
				if parent[j] == i then
					child = j
					break
				end
			end
			if child > 0 then
				local cv = world:position(child)
				local from = anim[bones[child]].Position - anim[b].Position
				local to = Vector3.new(cv.x, cv.y, cv.z) - pos
				if from.Magnitude > 1e-6 and to.Magnitude > 1e-6 then
					local u, w = from.Unit, to.Unit
					local axis = u:Cross(w)
					local s, c = axis.Magnitude, u:Dot(w)
					local turn = if s > 1e-9 then CFrame.fromAxisAngle(axis / s, math.atan2(s, c)) else CFrame.identity
					rot = turn * rot
				end
			elseif i > 1 and turned[parent[i]] then
				rot = turned[parent[i]] * rot
			end
			turned[i] = rot * anim[b].Rotation:Inverse()
			if i > 1 then
				b.WorldCFrame = CFrame.new(pos) * rot
			end
		end
	end

	-- Puts the bones back in their animated pose (and the simulation at rest there).
	function rig.reset(self: Rig)
		for b, cf in rest do
			(b :: Bone).CFrame = cf
		end
	end

	function rig.destroy(self: Rig)
		self:reset()
	end

	return rig :: any
end

-- Simulates every BasePart tagged `tag` (default "CitrateBones"), now and as tags are added,
-- each frame (on the client). Rigs farther than options.activation_distance (default 60
-- studs) from the camera stop and return to their animated pose; between
-- options.throttle_distance (default 20) and that, their bones are updated less often (the
-- simulation's substeps stay the same). Part attributes override the options. Returns a
-- function that stops it all.
function Bones.start(tag: string?, options: (Options & { activation_distance: number?, throttle_distance: number? })?): () -> ()
	local name = tag or "CitrateBones"
	local base = options or {}
	local rigs: { [BasePart]: { Rig } } = {}
	local function attach(part: Instance)
		if not part:IsA("BasePart") or rigs[part :: BasePart] then
			return
		end
		local o = table.clone(base) :: any
		for key, value in part:GetAttributes() do
			o[string.lower(key)] = value
		end
		local list = {}
		local roots = part:GetAttribute("Roots")
		for _, b in part:GetDescendants() do
			if b:IsA("Bone") and not (b.Parent and b.Parent:IsA("Bone")) then
				if type(roots) ~= "string" or string.find("," .. roots .. ",", "," .. b.Name .. ",", 1, true) then
					table.insert(list, Bones.new(b :: Bone, o))
				end
			end
		end
		rigs[part :: BasePart] = list
	end
	local function detach(part: Instance)
		local list = rigs[part :: BasePart]
		if list then
			for _, r in list do
				r:destroy()
			end
			rigs[part :: BasePart] = nil
		end
	end
	for _, part in CollectionService:GetTagged(name) do
		attach(part)
	end
	local added = CollectionService:GetInstanceAddedSignal(name):Connect(attach)
	local removed = CollectionService:GetInstanceRemovedSignal(name):Connect(detach)
	local near, far = base.throttle_distance or 20, base.activation_distance or 60
	local waiting: { [BasePart]: number } = {}
	local beat = RunService.Heartbeat:Connect(function(dt: number)
		local camera = workspace.CurrentCamera
		local eye = if camera then camera.CFrame.Position else Vector3.zero
		for part, list in rigs do
			local d = (part.Position - eye).Magnitude
			if d > far then
				continue
			end
			-- update every frame up close, down to every fourth frame at the activation edge
			local every = 1 + math.floor(3 * math.clamp((d - near) / (far - near), 0, 1))
			waiting[part] = (waiting[part] or 0) + dt
			local frames = (waiting[part] :: number) * 60
			if frames + 1e-6 >= every then
				for _, r in list do
					r:step(waiting[part])
				end
				waiting[part] = 0
			end
		end
	end)
	return function()
		beat:Disconnect()
		added:Disconnect()
		removed:Disconnect()
		for part in rigs do
			detach(part)
		end
	end
end

return Bones
