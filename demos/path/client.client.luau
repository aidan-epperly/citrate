--!strict
-- Path demo (LocalScript): a roller coaster with a vertical loop, built the way one would in
-- Studio: a Folder of Parts (P1, P2, ...) that citrate's path component (roblox/path.luau)
-- turns into a smooth closed track. Two copies, a train of cars on each, placed by the same
-- arclength and differing only in how the cars are turned. Left: CFrame.lookAt(p, p +
-- direction), with the world's up, as is common; the cars flip where the track runs
-- vertically. Right: path:follow, with twist-free frames; the cars roll through the loop.
-- The buttons ride along on either side.

local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local Citrate = ReplicatedStorage:WaitForChild("Citrate")
local Path = require(Citrate:WaitForChild("roblox"):WaitForChild("path")) :: any

local SPEED = 28 -- studs per second
local CARS = 4
local GAP = 3.6 -- studs between cars
local RIDE = CFrame.new(0, 0.9, 0) -- a car's center above the track

local folder = Instance.new("Folder")
folder.Name = "PathDemo"
folder.Parent = workspace
local function part(size: Vector3, color: Color3, parent: Instance?): Part
	local p = Instance.new("Part")
	p.Anchored, p.CanCollide, p.CastShadow = true, false, false
	p.Size, p.Color, p.Material = size, color, Enum.Material.SmoothPlastic
	p.Parent = parent or folder
	return p
end

-- The track's points: a straight, a vertical loop (shifted sideways by 6 studs over the
-- loop so that it does not meet itself), a climbing turn and a dip back to the start.
local POINTS = { Vector3.new(0, 6, -50), Vector3.new(0, 6, -25) }
for k = 0, 8 do
	local phi = k * math.pi / 4
	table.insert(POINTS, Vector3.new(6 * k / 8, 22 - 16 * math.cos(phi), 16 * math.sin(phi)))
end
for _, p in { { 6, 6, 25 }, { 14, 8, 45 }, { 30, 12, 50 }, { 44, 12, 38 }, { 48, 9, 15 }, { 48, 16, -10 }, { 44, 10, -35 }, { 32, 7, -55 }, { 16, 6, -62 } } do
	table.insert(POINTS, Vector3.new(p[1], p[2], p[3]))
end

-- A track: its Folder of point Parts (as a builder would place them), the path through
-- them, rails and ties drawn from the path's frames.
local function build(origin: Vector3, color: Color3): any
	local points = Instance.new("Folder")
	points.Name = "Track"
	points:SetAttribute("Closed", true)
	for i, p in POINTS do
		local marker = part(Vector3.new(0.8, 0.8, 0.8), Color3.fromRGB(230, 230, 240), points)
		marker.Shape = Enum.PartType.Ball
		marker.Transparency = 0.4
		marker.Name = "P" .. i
		marker.Position = origin + p
	end
	points.Parent = folder
	local path = Path.new(points)
	local n = math.ceil(path.length / 1.5)
	local prev = path:cframe(0)
	for k = 1, n do
		local cf = path:cframe(k / n * path.length)
		for side = -1, 1, 2 do
			local a, b = (prev * CFrame.new(side * 0.9, 0, 0)).Position, (cf * CFrame.new(side * 0.9, 0, 0)).Position
			local rail = part(Vector3.new(0.3, 0.3, (b - a).Magnitude + 0.05), color)
			rail.CFrame = CFrame.lookAt((a + b) / 2, b, cf.UpVector)
		end
		if k % 2 == 0 then
			local tie = part(Vector3.new(2.4, 0.15, 0.4), Color3.fromRGB(90, 80, 70))
			tie.CFrame = cf * CFrame.new(0, -0.2, 0)
		end
		prev = cf
	end
	return path
end

-- A car: a body and a fin on its top, so which way is up is plain.
local function car(color: Color3): Model
	local m = Instance.new("Model")
	local body = part(Vector3.new(2.2, 1.1, 3.2), color, m)
	local fin = part(Vector3.new(0.3, 1, 1.6), Color3.new(1, 1, 1), m)
	fin.CFrame = body.CFrame * CFrame.new(0, 1, 0.4)
	m.PrimaryPart = body
	m.Parent = folder
	return m
end

local LEFT, RIGHT = Vector3.new(0, 0, 62), Vector3.new(0, 0, -50)
local left_path = build(LEFT, Color3.fromRGB(200, 120, 60))
local right_path = build(RIGHT, Color3.fromRGB(60, 140, 210))

local left_cars, right_movers = {}, {}
for i = 1, CARS do
	left_cars[i] = car(Color3.fromRGB(230, 140, 70))
	right_movers[i] = right_path:follow(car(Color3.fromRGB(80, 160, 240)), {
		speed = SPEED,
		start = -(i - 1) * GAP,
		offset = RIDE,
		manual = true, -- stepped below, to time it
	})
end

-- Panel and buttons
local gui = Instance.new("ScreenGui")
gui.Name, gui.ResetOnSpawn, gui.IgnoreGuiInset = "PathDemo", false, true
local function label(text: string, anchor: Vector2, position: UDim2, size: Vector2): TextLabel
	local l = Instance.new("TextLabel")
	l.AnchorPoint, l.Position, l.Size = anchor, position, UDim2.fromOffset(size.X, size.Y)
	l.BackgroundColor3, l.BackgroundTransparency = Color3.new(0, 0, 0), 0.45
	l.TextColor3, l.Font, l.TextSize = Color3.new(1, 1, 1), Enum.Font.GothamMedium, 15
	l.TextWrapped = true
	l.Text = text
	l.Parent = gui
	return l
end
label("CFrame.lookAt(p, p + direction): the world's up", Vector2.new(0, 0), UDim2.fromOffset(16, 16), Vector2.new(330, 30))
label("citrate path:follow: twist-free frames", Vector2.new(1, 0), UDim2.new(1, -16, 0, 16), Vector2.new(330, 30))
local status = label("", Vector2.new(0.5, 1), UDim2.new(0.5, 0, 1, -16), Vector2.new(640, 30))
local camera = workspace.CurrentCamera
camera.CameraType = Enum.CameraType.Scriptable
local OVERVIEW = CFrame.lookAt(Vector3.new(122, 40, 6), Vector3.new(16, 14, 6))
camera.CFrame = OVERVIEW
local view = "overview"
local VIEWS = { { "overview", "Overview" }, { "left", "Ride left" }, { "right", "Ride right" } }
for k, v in VIEWS do
	local b = Instance.new("TextButton")
	b.Name = v[1]
	b.AnchorPoint = Vector2.new(0.5, 1)
	b.Position = UDim2.new(0.5, (k - 2) * 130, 1, -58)
	b.Size = UDim2.fromOffset(120, 32)
	b.BackgroundColor3 = Color3.fromRGB(55, 55, 70)
	b.TextColor3, b.Font, b.TextSize = Color3.new(1, 1, 1), Enum.Font.GothamBold, 15
	b.Text = v[2]
	b.Parent = gui
	b.Activated:Connect(function()
		view = v[1]
		if view == "overview" then
			camera.CFrame = OVERVIEW
		end
	end)
end
game:GetService("StarterGui"):SetCoreGuiEnabled(Enum.CoreGuiType.All, false)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")

local distance = 0
local ms_left, ms_right = 0, 0
RunService.Heartbeat:Connect(function(dt: number)
	distance += SPEED * dt
	-- left: the common way, the point at the distance and lookAt along the direction
	local t0 = os.clock()
	for i, m in left_cars do
		local s = distance - (i - 1) * GAP
		local p = left_path:position(s)
		local ahead = left_path:position(s + 0.05)
		m:PivotTo(CFrame.lookAt(p, ahead) * RIDE)
	end
	local t1 = os.clock()
	for _, mover in right_movers do
		mover:step(dt)
	end
	local t2 = os.clock()
	ms_left = 0.95 * ms_left + 0.05 * (t1 - t0) * 1000
	ms_right = 0.95 * ms_right + 0.05 * (t2 - t1) * 1000
	status.Text = string.format("%d cars per side; time per frame: left %.3f ms, right %.3f ms", CARS, ms_left, ms_right)
	if view ~= "overview" then
		local lead = if view == "left" then left_cars[1] else nil
		local cf = if lead then lead:GetPivot() else right_path:cframe(right_movers[1].distance) * RIDE
		camera.CFrame = cf * CFrame.new(0, 2.2, 2.5) * CFrame.Angles(-0.12, 0, 0)
	end
end)
