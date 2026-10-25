--!strict
--!native
-- Paths for Roblox: a smooth curve through Attachments or Parts (or positions), and things
-- moved along it by distance, at a set speed, turned with twist-free frames.
--
--   local Path = require(Citrate.roblox.path)
--   local path = Path.new(workspace.Track)          -- through its Attachment/Part children
--   local cf = path:cframe(10)                      -- 10 studs along: facing along the path
--   local mover = path:follow(workspace.Cart, { speed = 20 })
--
-- The curve is a C^2 cubic spline through the points (citrate.curve.cubic_spline, periodic
-- on a closed path, knots spaced centripetally): its curvature is continuous, so a mover's
-- turning rate and sideways acceleration change smoothly through the points. (A Catmull-Rom
-- spline, options.spline = "catmull_rom", matches only direction there: its curvature jumps
-- at each point, by up to 83% on the demo's coaster, which a rider feels as a jolt; it is
-- local, though: moving a point only changes the curve next to it.) Distances are along the curve,
-- so a mover's speed is the set speed exactly. Frames are rotation-minimizing: the "up" of
-- a mover only turns as much as the path bends, never twisting about its direction, and,
-- unlike CFrame.lookAt with a fixed up vector, does not flip where the path runs vertically
-- (a loop). On a closed path the frames join after a lap (citrate.curve.framing).

local RunService = game:GetService("RunService")

local citrate = require("../src")
local curve = citrate.curve

export type Options = {
	closed: boolean?, -- default: the source's "Closed" attribute, else false
	spline: ("cubic" | "catmull_rom")?, -- default "cubic" (C^2); "catmull_rom" is C^1 but local
	alpha: number?, -- knot spacing: 0.5 centripetal (default), 0 uniform, 1 chordal
	up: Vector3?, -- the up direction at the start (default the Y axis)
	tol: number?, -- the arclength tolerance (default 1e-8: what a Vector3 holds)
}

export type FollowOptions = {
	speed: number?, -- studs per second (default 16); negative goes backwards
	start: number?, -- the distance along the path to start at (default 0)
	mode: ("loop" | "bounce" | "stop")?, -- at an end of an open path (default "loop": jump to the other end)
	offset: CFrame?, -- the target's pivot relative to the path's frame (default identity)
	finished: (() -> ())?, -- called when a "stop" mover reaches the end
	manual: boolean?, -- if true, no Heartbeat connection: call mover:step(dt) yourself
}

export type Mover = {
	distance: number,
	speed: number,
	step: (self: Mover, dt: number) -> (),
	stop: (self: Mover) -> (),
}

export type Path = {
	length: number,
	closed: boolean,
	curve: any,
	position: (self: Path, s: number) -> Vector3,
	cframe: (self: Path, s: number) -> CFrame,
	closest: (self: Path, position: Vector3, near: number?) -> (number, Vector3, number),
	follow: (self: Path, target: PVInstance, options: FollowOptions?) -> Mover,
	rebuild: (self: Path) -> (),
}

-- The points of a source: an array of Vector3s, CFrames, Attachments and BaseParts, or an
-- Instance whose Attachment and BasePart children are the points, ordered by their "Order"
-- attribute if they have one, else by name with numbers compared as numbers (P2 < P10).
local function natural_less(a: string, b: string): boolean
	local i, j = 1, 1
	while i <= #a and j <= #b do
		local na, nb = string.match(a, "^%d+", i), string.match(b, "^%d+", j)
		if na and nb then
			if tonumber(na) ~= tonumber(nb) then
				return (tonumber(na) :: number) < (tonumber(nb) :: number)
			end
			i += #na
			j += #nb
		else
			local ca, cb = string.sub(a, i, i), string.sub(b, j, j)
			if ca ~= cb then
				return ca < cb
			end
			i += 1
			j += 1
		end
	end
	return #a - i < #b - j
end

local function point_of(p: any): Vector3
	if typeof(p) == "Vector3" then
		return p
	elseif typeof(p) == "CFrame" then
		return p.Position
	elseif p:IsA("Attachment") then
		return p.WorldPosition
	end
	return (p :: BasePart).Position
end

local function sources(source: any): { any }
	if typeof(source) ~= "Instance" then
		return source
	end
	local list = {}
	for _, c in (source :: Instance):GetChildren() do
		if c:IsA("Attachment") or c:IsA("BasePart") then
			table.insert(list, c)
		end
	end
	table.sort(list, function(a: Instance, b: Instance): boolean
		local oa, ob = a:GetAttribute("Order"), b:GetAttribute("Order")
		if type(oa) == "number" and type(ob) == "number" and oa ~= ob then
			return oa < ob
		end
		return natural_less(a.Name, b.Name)
	end)
	return list
end

local Path = {}

-- A path through source's points (see sources above).
function Path.new(source: Instance | { any }, options: Options?): Path
	local o: Options = options or {}
	local closed = o.closed
	if closed == nil and typeof(source) == "Instance" then
		closed = (source :: Instance):GetAttribute("Closed") == true
	end
	local up = o.up or Vector3.yAxis
	local self = { closed = closed == true, length = 0, curve = nil :: any }
	local framing: any

	-- Rebuilds the curve from the points' current positions (call after moving them).
	function self.rebuild(me: any)
		local pts = {}
		for i, p in sources(source) do
			local v = point_of(p)
			pts[i] = vector.create(v.X, v.Y, v.Z)
		end
		local make = if o.spline == "catmull_rom" then curve.catmull_rom else curve.cubic_spline
		local C = make(pts, { closed = me.closed, alpha = o.alpha })
		curve.arclength(C, { tol = o.tol or 1e-8 })
		framing = curve.framing(C, { up = vector.create(up.X, up.Y, up.Z) })
		me.curve, me.length = C, framing.length
	end
	self:rebuild()

	-- The point s studs along the path (wrapping on a closed path, clamped on an open one).
	function self.position(_me: any, s: number): Vector3
		local x = curve.at_length(self.curve, if self.closed then s % self.length else s)
		return Vector3.new(x.x, x.y, x.z)
	end

	-- The frame s studs along: at the point, LookVector along the path, UpVector the
	-- twist-free normal.
	function self.cframe(_me: any, s: number): CFrame
		local x, d, n = framing:at(s)
		local look, upv = Vector3.new(d.x, d.y, d.z), Vector3.new(n.x, n.y, n.z)
		return CFrame.fromMatrix(Vector3.new(x.x, x.y, x.z), look:Cross(upv), upv, -look)
	end

	-- The distance along the path of its point closest to position, that point and the
	-- distance to it. With near (a distance along the path, such as last frame's answer),
	-- searches locally from there, which is much faster; else globally.
	function self.closest(_me: any, position: Vector3, near: number?): (number, Vector3, number)
		local q = vector.create(position.X, position.Y, position.Z)
		local t, x, dist
		if near then
			t, x, dist = curve.closest_near(self.curve, q, curve.param_at(self.curve, if self.closed then near % self.length else near))
		else
			t, x, dist = curve.closest(self.curve, q)
		end
		return curve.length_at(self.curve, t), Vector3.new(x.x, x.y, x.z), dist
	end

	-- Moves target (a Model or a part, by its pivot) along the path at options.speed, each
	-- Heartbeat unless options.manual.
	function self.follow(me: any, target: PVInstance, fo: FollowOptions?): Mover
		local f: FollowOptions = fo or {}
		local mode = f.mode or "loop"
		local offset = f.offset or CFrame.identity
		local mover = { distance = f.start or 0, speed = f.speed or 16 }
		local connection: RBXScriptConnection? = nil
		local done = false
		function mover.step(m: any, dt: number)
			if done then
				return
			end
			local s = m.distance + m.speed * dt
			local L = me.length
			if not me.closed and (s < 0 or s > L) then
				if mode == "bounce" then
					-- reflect off the end
					s = if s > L then 2 * L - s else -s
					m.speed = -m.speed
				elseif mode == "stop" then
					s = math.clamp(s, 0, L)
					done = true
				else
					s %= L
				end
			end
			m.distance = s
			local cf = me:cframe(s)
			if m.speed < 0 and mode == "bounce" then
				cf *= CFrame.Angles(0, math.pi, 0) -- face the way it moves
			end
			target:PivotTo(cf * offset)
			if done then
				m:stop()
				if f.finished then
					f.finished()
				end
			end
		end
		function mover.stop(_m: any)
			done = true
			if connection then
				connection:Disconnect()
				connection = nil
			end
		end
		if not f.manual then
			connection = RunService.Heartbeat:Connect(function(dt: number)
				mover:step(dt)
			end)
		end
		mover:step(0)
		return mover :: any
	end

	return self :: any
end

return Path
