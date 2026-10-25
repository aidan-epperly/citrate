--!strict
--!native
-- Track demo (LocalScript): carts moving at a constant speed along the same track, placed
-- by arclength in two ways. Left: a lookup table of chord lengths, as popular Roblox Bézier
-- modules do. Right: citrate's curve.at_length. The plots show the lead cart's speed
-- relative to the set speed; the panel shows each side's time per frame for all its carts.

local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local Citrate = ReplicatedStorage:WaitForChild("Citrate")
local track = require(Citrate:WaitForChild("demos"):WaitForChild("track"):WaitForChild("track")) :: any

local OFFSET = 100 -- studs between the two tracks' centers
local PLOT = 300 -- frames shown in each speed plot
local RANGE = 0.006 -- the plots span 1 - RANGE .. 1 + RANGE of the set speed
local SAMPLES = { 101, 1001, 4701 }
local COUNTS = { 1, 50, 300, 1000 }
local SPEEDS = { 20, 40, 80 }

local bez = track.bezier_points(track.POINTS)
local cit = track.citrate(bez)
local tables = {}
for _, n in SAMPLES do
	tables[n] = track.lookup(bez, n)
end
local samples, count, speed = 101, 300, 40
local length = cit.length

-- Scenery: a rail along each copy of the track, from citrate's curve (sampled densely).
local folder = Instance.new("Folder")
folder.Name = "TrackDemo"
folder.Parent = workspace
local function rail(origin: Vector3, color: Color3)
	local n = 400
	local prev = track.citrate_at(cit, 0) + origin
	for k = 1, n do
		local p = track.citrate_at(cit, k / n * length) + origin
		local part = Instance.new("Part")
		part.Anchored, part.CanCollide, part.CastShadow = true, false, false
		part.Color, part.Material = color, Enum.Material.SmoothPlastic
		part.Size = Vector3.new(1.2, 0.4, (p - prev).Magnitude + 0.1)
		part.CFrame = CFrame.lookAt((p + prev) / 2, p)
		part.Parent = folder
		if k % 8 == 0 then -- a post down to the ground
			local post = Instance.new("Part")
			post.Anchored, post.CanCollide, post.CastShadow = true, false, false
			post.Color, post.Material = Color3.fromRGB(120, 120, 130), Enum.Material.Metal
			post.Size = Vector3.new(0.4, p.Y, 0.4)
			post.Position = Vector3.new(p.X, p.Y / 2, p.Z)
			post.Parent = folder
		end
		prev = p
	end
end
local LEFT, RIGHT = Vector3.new(-OFFSET, 0, 0), Vector3.new(OFFSET, 0, 0)
rail(LEFT, Color3.fromRGB(200, 120, 60))
rail(RIGHT, Color3.fromRGB(60, 140, 210))

local function cart(size: Vector3, color: Color3): Part
	local p = Instance.new("Part")
	p.Anchored, p.CanCollide, p.CastShadow = true, false, false
	p.Size, p.Color, p.Material = size, color, Enum.Material.SmoothPlastic
	p.Parent = folder
	return p
end
type Side = { origin: Vector3, at: (number) -> Vector3, lead: Part, carts: { Part }, ms: number, plot: { Frame }, history: { number }, label: TextLabel }
local function side(origin: Vector3, color: Color3, at: (number) -> Vector3): Side
	return { origin = origin, at = at, lead = cart(Vector3.new(5, 3.5, 8), color), carts = {}, ms = 0, plot = {}, history = table.create(PLOT, 1), label = (nil :: any) }
end
local sides = {
	side(LEFT, Color3.fromRGB(240, 150, 70), function(s)
		return track.lookup_at(tables[samples], s)
	end),
	side(RIGHT, Color3.fromRGB(80, 170, 255), function(s)
		return track.citrate_at(cit, s)
	end),
}
local function set_count(n: number)
	count = n
	for _, sd in sides do
		while #sd.carts < n - 1 do
			table.insert(sd.carts, cart(Vector3.new(2.2, 2.2, 2.2), sd.lead.Color:Lerp(Color3.new(1, 1, 1), 0.35)))
		end
		while #sd.carts > n - 1 do
			(table.remove(sd.carts) :: Part):Destroy()
		end
	end
end
set_count(count)

-- Panel
local gui = Instance.new("ScreenGui")
gui.Name, gui.ResetOnSpawn, gui.IgnoreGuiInset = "TrackDemo", false, true
local function label(parent: Instance, text: string, pos: UDim2, size: UDim2, color: Color3?): TextLabel
	local l = Instance.new("TextLabel")
	l.BackgroundTransparency = 1
	l.TextColor3 = color or Color3.new(1, 1, 1)
	l.Font, l.TextSize = Enum.Font.GothamMedium, 15
	l.TextXAlignment = Enum.TextXAlignment.Left
	l.Text, l.Position, l.Size = text, pos, size
	l.Parent = parent
	return l
end
local W, H = 400, 110
for i, sd in sides do
	local box = Instance.new("Frame")
	box.Size = UDim2.fromOffset(W, H + 46)
	box.AnchorPoint = Vector2.new(i - 1, 1)
	box.Position = UDim2.new(i - 1, if i == 1 then 16 else -16, 1, -16)
	box.BackgroundColor3, box.BackgroundTransparency = Color3.new(0, 0, 0), 0.45
	box.Parent = gui
	sd.label = label(box, "", UDim2.fromOffset(8, 4), UDim2.new(1, -16, 0, 36), sd.lead.Color)
	sd.label.TextWrapped = true
	sd.label.TextYAlignment = Enum.TextYAlignment.Top
	local area = Instance.new("Frame")
	area.Position, area.Size = UDim2.fromOffset(8, 40), UDim2.fromOffset(W - 16, H)
	area.BackgroundTransparency, area.ClipsDescendants = 1, true
	area.Parent = box
	for _, v in { -RANGE, -RANGE / 2, 0, RANGE / 2, RANGE } do -- grid lines and labels
		local line = Instance.new("Frame")
		line.BorderSizePixel = 0
		line.BackgroundColor3 = Color3.new(1, 1, 1)
		line.BackgroundTransparency = if v == 0 then 0.5 else 0.85
		line.Size = UDim2.new(1, 0, 0, 1)
		line.Position = UDim2.new(0, 0, 0.5 - v / (2 * RANGE), 0)
		line.Parent = area
		if v ~= 0 then
			label(area, string.format("%+.1f%%", v * 100), UDim2.new(1, -44, 0.5 - v / (2 * RANGE), if v > 0 then 1 else -14), UDim2.fromOffset(44, 14), Color3.new(0.8, 0.8, 0.8)).TextSize = 11
		end
	end
	for k = 1, PLOT do
		local dot = Instance.new("Frame")
		dot.BorderSizePixel = 0
		dot.BackgroundColor3 = sd.lead.Color
		dot.Size = UDim2.fromOffset(2, 2)
		dot.Position = UDim2.new((k - 1) / PLOT, 0, 0.5, 0)
		dot.Parent = area
		sd.plot[k] = dot
	end
end

local controls = Instance.new("Frame")
controls.AnchorPoint = Vector2.new(0.5, 1)
controls.Position = UDim2.new(0.5, 0, 1, -16)
controls.Size = UDim2.fromOffset(330, 110)
controls.BackgroundColor3, controls.BackgroundTransparency = Color3.new(0, 0, 0), 0.45
controls.Parent = gui
local function row(y: number, title: string, options: { number }, current: number, fmt: string, on: (number) -> ())
	label(controls, title, UDim2.fromOffset(8, y), UDim2.fromOffset(90, 26))
	local buttons = {}
	local function paint(v: number)
		for opt, b in buttons do
			b.BackgroundColor3 = if opt == v then Color3.fromRGB(90, 110, 160) else Color3.fromRGB(55, 55, 70)
		end
	end
	for k, opt in options do
		local b = Instance.new("TextButton")
		b.Name = title .. opt
		b.Size, b.Position = UDim2.fromOffset(52, 24), UDim2.fromOffset(96 + 56 * (k - 1), y + 1)
		b.TextColor3, b.Font, b.TextSize = Color3.new(1, 1, 1), Enum.Font.GothamBold, 13
		b.Text = string.format(fmt, opt)
		b.Parent = controls
		buttons[opt] = b
		b.Activated:Connect(function()
			on(opt)
			paint(opt)
		end)
	end
	paint(current)
end
row(6, "Table size", SAMPLES, samples, "%d", function(v)
	samples = v
end)
row(40, "Carts", COUNTS, count, "%d", set_count)
row(74, "Speed", SPEEDS, speed, "%d", function(v)
	speed = v
end)

game:GetService("StarterGui"):SetCoreGuiEnabled(Enum.CoreGuiType.All, false)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")
local camera = workspace.CurrentCamera
camera.CameraType = Enum.CameraType.Scriptable
camera.CFrame = CFrame.lookAt(Vector3.new(0, 290, 150), Vector3.new(0, 0, 18))

local distance = 0
local frame = 0
RunService.Heartbeat:Connect(function(dt: number)
	dt = math.min(dt, 1 / 20)
	distance = (distance + speed * dt) % length
	frame += 1
	for _, sd in sides do
		-- Place every cart, timed: the lead cart at `distance`, the rest evenly spaced behind.
		local positions = table.create(count)
		local t0 = os.clock()
		for k = 0, count - 1 do
			positions[k + 1] = sd.at(distance - k * length / math.max(count, 1))
		end
		local ms = (os.clock() - t0) * 1000
		sd.ms = if sd.ms == 0 then ms else 0.95 * sd.ms + 0.05 * ms
		-- The lead cart's speed relative to the set speed: the distance between the points
		-- placed at `distance` and 0.25 studs before, over 0.25 (a frame's whole chord would
		-- fall short of the arc on tight curves by about 0.07%).
		local p = positions[1]
		local v = (p - sd.at(distance - 0.25)).Magnitude / 0.25
		table.remove(sd.history, 1)
		table.insert(sd.history, v)
		local ahead = sd.at(distance + 1)
		local cfs, parts = table.create(count), table.create(count)
		cfs[1], parts[1] = CFrame.lookAt(p + sd.origin + Vector3.new(0, 1.4, 0), ahead + sd.origin + Vector3.new(0, 1.4, 0)), sd.lead
		for k = 2, count do
			cfs[k], parts[k] = CFrame.new(positions[k] + sd.origin + Vector3.new(0, 1, 0)), sd.carts[k - 1]
		end
		workspace:BulkMoveTo(parts, cfs, Enum.BulkMoveMode.FireCFrameChanged)
		for k = 1, PLOT do
			local y = math.clamp(0.5 - (sd.history[k] - 1) / (2 * RANGE), 0, 1)
			sd.plot[k].Position = UDim2.new((k - 1) / PLOT, 0, y, -1)
		end
	end
	for i, sd in sides do
		local lo, hi = math.huge, -math.huge
		for _, v in sd.history do
			lo = math.min(lo, v)
			hi = math.max(hi, v)
		end
		local name = if i == 1 then `Lookup table, {samples} samples` else "citrate curve.at_length"
		sd.label.Text = string.format("%s: %.2f ms for %d carts\nspeed %+.2f%% to %+.2f%% of the set speed", name, sd.ms, count, (lo - 1) * 100, (hi - 1) * 100)
	end
end)
