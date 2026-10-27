--!strict
-- Flock demo (LocalScript): spatial queries in two scenes (flock.luau).
-- Flock: boids steering by their neighbors within a radius, found each frame by citrate's
-- grid, its k-d tree, by testing every pair, or by Roblox's own workspace:GetPartBoundsInRadius
-- (one query per boid); the panel shows the time each takes per frame.
-- Navigation: waypoints among obstacles, joined where the way is clear; click the floor to
-- send the walkers there along A* routes.

local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local Citrate = ReplicatedStorage:WaitForChild("Citrate")
local flock = require(Citrate:WaitForChild("demos"):WaitForChild("flock"):WaitForChild("flock")) :: any

local FLOCK_CENTER = Vector3.new(170, 40, 0)
local HALF = vector.create(40, 15, 40)
local RADIUS = 4
local NAV_CENTER = Vector3.new(-80, 0, 0)
local NAV_SIZE, NAV_BOXES, NAV_POINTS, NAV_REACH = 160, 18, 400, 18

local folder = Instance.new("Folder")
folder.Name = "FlockDemo"
folder.Parent = workspace
local boid_folder = Instance.new("Folder")
boid_folder.Name = "Boids"
boid_folder.Parent = folder
local nav_folder = Instance.new("Folder")
nav_folder.Name = "Navigation"
nav_folder.Parent = folder

local function part(parent: Instance, size: Vector3, color: Color3, shape: Enum.PartType?): Part
	local p = Instance.new("Part")
	p.Anchored, p.CanCollide, p.CanTouch, p.CanQuery, p.CastShadow = true, false, false, false, false
	p.Size, p.Color, p.Material = size, color, Enum.Material.SmoothPlastic
	if shape then
		p.Shape = shape
	end
	p.Parent = parent
	return p
end

-- Flock

local count = 1000
local method = "grid"
local F: any
local boids: { BasePart } = {}
local index_of: { [Instance]: number } = {}
local params = OverlapParams.new()
params.FilterType = Enum.RaycastFilterType.Include
params.FilterDescendantsInstances = { boid_folder }

local function make_flock(n: number)
	for _, b in boids do
		b:Destroy()
	end
	table.clear(boids)
	table.clear(index_of)
	F = flock.new(n, HALF, RADIUS)
	for i = 1, n do
		local b = part(boid_folder, Vector3.new(0.5, 0.35, 1.3), Color3.fromHSV((i % 12) / 12 * 0.15 + 0.55, 0.6, 0.95))
		b.CanQuery = true -- for GetPartBoundsInRadius
		boids[i] = b
		index_of[b] = i
	end
end

-- Roblox's query: the boid parts within the radius (bounding boxes, so then by distance)
local function roblox_query(i: number, out: { number }): number
	table.clear(out)
	local p = F.pos[i]
	local center = FLOCK_CENTER + Vector3.new(p.x, p.y, p.z)
	local c = 0
	for _, hit in workspace:GetPartBoundsInRadius(center, RADIUS, params) do
		local j = index_of[hit]
		if j then
			local d = F.pos[j] - p
			if vector.dot(d, d) <= RADIUS * RADIUS then
				c += 1
				out[c] = j
			end
		end
	end
	return c
end

local cframes: { CFrame } = {}
local function draw_flock()
	for i = 1, F.n do
		local p, v = F.pos[i], F.vel[i]
		local at = FLOCK_CENTER + Vector3.new(p.x, p.y, p.z)
		cframes[i] = CFrame.lookAt(at, at + Vector3.new(v.x, v.y, v.z))
	end
	for i = F.n + 1, #cframes do
		cframes[i] = nil
	end
	workspace:BulkMoveTo(boids, cframes, Enum.BulkMoveMode.FireCFrameChanged)
end

-- the box's edges
do
	local h = Vector3.new(HALF.x, HALF.y, HALF.z)
	local color = Color3.fromRGB(120, 130, 150)
	for _, axis in { Vector3.xAxis, Vector3.yAxis, Vector3.zAxis } do
		local others = if axis == Vector3.xAxis then { Vector3.yAxis, Vector3.zAxis } elseif axis == Vector3.yAxis then { Vector3.xAxis, Vector3.zAxis } else { Vector3.xAxis, Vector3.yAxis }
		for _, s1 in { -1, 1 } do
			for _, s2 in { -1, 1 } do
				local len = 2 * h:Dot(axis)
				local e = part(folder, Vector3.new(0.15, 0.15, len), color)
				local mid = FLOCK_CENTER + others[1] * s1 * h:Dot(others[1]) + others[2] * s2 * h:Dot(others[2])
				e.CFrame = CFrame.lookAt(mid, mid + axis)
			end
		end
	end
end

-- Navigation

local N = flock.nav(NAV_SIZE, NAV_BOXES, NAV_POINTS, NAV_REACH)
local floor = part(nav_folder, Vector3.new(NAV_SIZE, 1, NAV_SIZE), Color3.fromRGB(70, 82, 70))
floor.Position = NAV_CENTER - Vector3.new(0, 0.45, 0)
floor.CanQuery = true
for _, B in N.boxes do
	local b = part(nav_folder, Vector3.new(2 * B.half.x, 2 * B.half.y, 2 * B.half.z), Color3.fromRGB(150, 120, 100))
	b.Position = NAV_CENTER + Vector3.new(B.center.x, B.center.y, B.center.z)
end
for _, p in N.points do
	local d = part(nav_folder, Vector3.new(0.3, 1.6, 1.6), Color3.fromRGB(200, 210, 120), Enum.PartType.Cylinder)
	d.CFrame = CFrame.new(NAV_CENTER + Vector3.new(p.x, 2, p.z)) * CFrame.Angles(0, 0, math.pi / 2)
	d.Material = Enum.Material.Neon
end

type Walker = { part: BasePart, at: vector, route: { vector }?, leg: number }
local walkers: { Walker } = {}
for k = 1, 6 do
	local p = N.points[(k * 53) % #N.points + 1]
	local w = part(nav_folder, Vector3.new(3, 3, 3), Color3.fromHSV(k / 6, 0.7, 1), Enum.PartType.Ball)
	table.insert(walkers, { part = w, at = p, route = nil, leg = 1 })
end
local path_parts: { BasePart } = {}
local function draw_route(route: { vector })
	for _, p in path_parts do
		p:Destroy()
	end
	table.clear(path_parts)
	for k = 1, #route - 1 do
		local a = NAV_CENTER + Vector3.new(route[k].x, 2.3, route[k].z)
		local b = NAV_CENTER + Vector3.new(route[k + 1].x, 2.3, route[k + 1].z)
		local seg = part(nav_folder, Vector3.new(0.8, 0.3, (b - a).Magnitude), walkers[1].part.Color)
		seg.Material = Enum.Material.Neon
		seg.CFrame = CFrame.lookAt((a + b) / 2, b)
		table.insert(path_parts, seg)
	end
end

-- Panel

local gui = Instance.new("ScreenGui")
gui.Name, gui.ResetOnSpawn, gui.IgnoreGuiInset = "FlockDemo", false, true
local panel = Instance.new("Frame")
panel.AnchorPoint, panel.Position, panel.Size = Vector2.new(0, 1), UDim2.new(0, 16, 1, -16), UDim2.fromOffset(560, 150)
panel.BackgroundColor3, panel.BackgroundTransparency = Color3.new(0, 0, 0), 0.45
panel.Parent = gui
local info = Instance.new("TextLabel")
info.BackgroundTransparency = 1
info.Position, info.Size = UDim2.fromOffset(12, 8), UDim2.new(1, -24, 0, 70)
info.TextColor3, info.Font, info.TextSize = Color3.new(1, 1, 1), Enum.Font.GothamMedium, 15
info.TextWrapped, info.TextXAlignment, info.TextYAlignment = true, Enum.TextXAlignment.Left, Enum.TextYAlignment.Top
info.Parent = panel
local buttons: { [string]: TextButton } = {}
local function button(key: string, text: string, x: number, y: number, w: number, on: () -> ())
	local b = Instance.new("TextButton")
	b.Position, b.Size = UDim2.fromOffset(x, y), UDim2.fromOffset(w, 26)
	b.BackgroundColor3, b.TextColor3 = Color3.fromRGB(60, 62, 70), Color3.new(1, 1, 1)
	b.Font, b.TextSize, b.Text = Enum.Font.GothamMedium, 14, text
	b.Parent = panel
	b.Activated:Connect(on)
	buttons[key] = b
end

local scene = "flock"
local camera = workspace.CurrentCamera
camera.CameraType = Enum.CameraType.Scriptable
local function highlight()
	for key, b in buttons do
		local on = key == scene or key == method or key == "n" .. count
		b.BackgroundColor3 = if on then Color3.fromRGB(70, 120, 200) else Color3.fromRGB(60, 62, 70)
	end
	if scene == "flock" then
		camera.CFrame = CFrame.lookAt(FLOCK_CENTER + Vector3.new(0, 30, 105), FLOCK_CENTER)
	else
		camera.CFrame = CFrame.lookAt(NAV_CENTER + Vector3.new(0, 125, 75), NAV_CENTER + Vector3.new(0, 0, 5))
	end
end
local samples: { number } = {}
local function choose_method(m: string)
	method = m
	table.clear(samples)
	highlight()
end
button("flock", "Flock", 12, 80, 80, function() scene = "flock"; highlight() end)
button("nav", "Navigation", 96, 80, 100, function() scene = "nav"; highlight() end)
button("grid", "grid", 12, 114, 70, function() choose_method("grid") end)
button("kdtree", "k-d tree", 86, 114, 80, function() choose_method("kdtree") end)
button("brute", "every pair", 170, 114, 90, function() choose_method("brute") end)
button("roblox", "GetPartBoundsInRadius", 264, 114, 170, function() choose_method("roblox") end)
for k, n in { 500, 1000, 2000 } do
	button("n" .. n, tostring(n), 438 + (k - 1) * 40, 114, 38, function()
		count = n
		make_flock(n)
		table.clear(samples)
		highlight()
	end)
end
game:GetService("StarterGui"):SetCoreGuiEnabled(Enum.CoreGuiType.All, false)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")

make_flock(count)
highlight()

local last_route: any = nil
local mouse = Players.LocalPlayer:GetMouse()
mouse.TargetFilter = folder
local function send(hit: Vector3)
	for k, w in walkers do
		-- each to a spot near the click
		local goal = vector.create(hit.X + 3 * math.cos(k), 0, hit.Z + 3 * math.sin(k))
		local route, rinfo = flock.route(N, w.at, goal)
		w.route, w.leg = route, 1
		if k == 1 then
			last_route = rinfo
			if route then
				draw_route(route)
			end
		end
	end
end
mouse.Button1Down:Connect(function()
	if scene ~= "nav" then
		return
	end
	local ray = camera:ScreenPointToRay(mouse.X, mouse.Y)
	if math.abs(ray.Direction.Y) < 1e-6 then
		return
	end
	local t = (NAV_CENTER.Y - ray.Origin.Y) / ray.Direction.Y
	local hit = ray.Origin + ray.Direction * t - NAV_CENTER
	if math.abs(hit.X) <= NAV_SIZE / 2 and math.abs(hit.Z) <= NAV_SIZE / 2 then
		send(hit)
	end
end)
-- The same by attributes of the FlockDemo folder (for driving the demo from a script):
-- Method, Scene, Count, and Goal (a Vector3 on the floor).
folder:SetAttribute("Method", method)
folder:SetAttribute("Scene", scene)
folder:SetAttribute("Count", count)
folder.AttributeChanged:Connect(function(name: string)
	local v = folder:GetAttribute(name)
	if name == "Method" and v ~= method then
		choose_method(v :: string)
	elseif name == "Scene" and v ~= scene then
		scene = v :: string
		highlight()
	elseif name == "Count" and v ~= count then
		count = v :: number
		make_flock(count)
		table.clear(samples)
		highlight()
	elseif name == "Goal" then
		send(v :: Vector3)
	end
end)

local WALK = 14
RunService.RenderStepped:Connect(function(dt: number)
	dt = math.min(dt, 1 / 20)
	local query: any = if method == "roblox" then roblox_query else method
	local ms, mean = flock.step(F, dt, query)
	table.insert(samples, ms)
	if #samples > 30 then
		table.remove(samples, 1)
	end
	local avg = 0
	for _, s in samples do
		avg += s
	end
	avg /= #samples
	draw_flock()
	-- walkers
	local parts, cfs = {}, {}
	for _, w in walkers do
		local route = w.route
		if route then
			local step = WALK * dt
			while step > 0 and w.leg < #route do
				local target = route[w.leg + 1]
				local d = target - w.at
				local len = vector.magnitude(d)
				if len <= step then
					w.at = target
					w.leg += 1
					step -= len
				else
					w.at += d * (step / len)
					step = 0
				end
			end
		end
		table.insert(parts, w.part)
		table.insert(cfs, CFrame.new(NAV_CENTER + Vector3.new(w.at.x, 2.5, w.at.z)))
	end
	workspace:BulkMoveTo(parts, cfs, Enum.BulkMoveMode.FireCFrameChanged)
	local names = { grid = "citrate's grid", kdtree = "citrate's k-d tree", brute = "testing every pair", roblox = "GetPartBoundsInRadius, per boid" }
	if scene == "flock" then
		info.Text = string.format("%d boids, each steering by its neighbors within %d studs (%.1f on average).\nFinding them by %s: %.2f ms per frame.", F.n, RADIUS, mean, names[method], avg)
	else
		local r = last_route
		info.Text = string.format("%d waypoints, %d clear edges (of %d within %d studs). Click the floor to send the walkers.", #N.points, N.edges, N.candidates, NAV_REACH)
			.. (if r then string.format("\nLast route: A* %.3f ms, %d waypoints expanded; Dijkstra's search %.3f ms.", r.ms, r.expanded, r.dijkstra_ms) else "")
	end
end)
