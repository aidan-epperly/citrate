--!strict
--!native
-- Tails demo (LocalScript): nine tails on roots that move, rest and move again.
-- Top row: SmartBone 2's rule. Middle row: citrate's XPBD solver with options fitted so that
-- its tail moves like SmartBone's at 60 Hz (which is drawn over it, translucent). Bottom row: the tail as a physical system advanced
-- by citrate's implicit mechanics integrator. The columns step at 60, 30 and 15 Hz, as
-- SmartBone does for objects 15, 30 and 37.5 studs from the camera.

local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local Citrate = ReplicatedStorage:WaitForChild("Citrate")
local tails = require(Citrate:WaitForChild("demos"):WaitForChild("tails"):WaitForChild("tails")) :: any

local RATES = { 60, 30, 15 }
local SPACING = 18 -- studs between columns
local ROWS = {
	{ name = "SmartBone rule", make = tails.smartbone, y = 32, color = Color3.fromRGB(240, 150, 70) },
	{ name = "XPBD tuned", ghost = true, make = function(rate: number)
		return tails.xpbd(rate, tails.SMARTBONE_LOOK)
	end, y = 18, color = Color3.fromRGB(120, 220, 140) },
	{ name = "citrate implicit", make = tails.physical, y = 4, color = Color3.fromRGB(80, 170, 255) },
}
-- The roots cycle through three motions, 6 s each: a vertical shake, a sideways shake and a
-- wobble in all three axes, each ending at rest.
local function envelope(t: number, len: number): number
	return if t < len then math.sin(math.pi * t / len) else 0
end
local MOTIONS = {
	function(t: number): vector
		return vector.create(0, 0.8 * math.sin(2 * math.pi * 1.5 * t) * envelope(t, 4 / 3), 0)
	end,
	function(t: number): vector
		return vector.create(0, 0, math.sin(2 * math.pi * t) * envelope(t, 2))
	end,
	function(t: number): vector
		return vector.create(0.5 * math.sin(5.1 * t), 0.6 * math.sin(7.3 * t + 1), 0.7 * math.sin(3.7 * t + 2)) * envelope(t, 3)
	end,
}
local function motion(t: number): vector
	local k = math.floor(t / 6)
	return MOTIONS[k % #MOTIONS + 1](t - 6 * k)
end

local folder = Instance.new("Folder")
folder.Name = "TailsDemo"
folder.Parent = workspace
local function part(size: Vector3, color: Color3, shape: Enum.PartType?): Part
	local p = Instance.new("Part")
	p.Anchored, p.CanCollide, p.CastShadow = true, false, false
	p.Size, p.Color, p.Material = size, color, Enum.Material.SmoothPlastic
	if shape then
		p.Shape = shape
	end
	p.Parent = folder
	return p
end
type Entry = { tail: any, origin: Vector3, root: Part, bones: { Part }, joints: { Part }, row: number, title: string, color: Color3, ghost: any?, ghost_bones: { Part } }
local entries: { Entry } = {}
for r, row in ROWS do
	for c, rate in RATES do
		local origin = Vector3.new((c - 2) * SPACING - 4, row.y, 0)
		local e: Entry = { tail = row.make(rate), origin = origin, root = part(Vector3.new(2, 2, 2), Color3.fromRGB(90, 90, 100)), bones = {}, joints = {}, row = r, title = `{row.name}, {rate} Hz`, color = row.color, ghost = nil, ghost_bones = {} }
		for i = 1, tails.BONES do
			e.bones[i] = part(Vector3.new(0.5, 0.5, tails.LENGTH), row.color)
			e.joints[i] = part(Vector3.new(0.7, 0.7, 0.7), row.color:Lerp(Color3.new(1, 1, 1), 0.3), Enum.PartType.Ball)
		end
		if (row :: any).ghost then
			-- SmartBone's 60 Hz tail drawn over it, translucent, for comparison
			e.ghost = tails.smartbone(60)
			for i = 1, tails.BONES do
				local g = part(Vector3.new(0.8, 0.8, tails.LENGTH), ROWS[1].color)
				g.Transparency = 0.6
				e.ghost_bones[i] = g
			end
		end
		table.insert(entries, e)
	end
end

-- Panel: the time each row spends per frame.
local gui = Instance.new("ScreenGui")
gui.Name, gui.ResetOnSpawn, gui.IgnoreGuiInset = "TailsDemo", false, true
local status = Instance.new("TextLabel")
status.AnchorPoint = Vector2.new(0.5, 1)
status.Position = UDim2.new(0.5, 0, 1, -16)
status.Size = UDim2.fromOffset(760, 64)
status.BackgroundColor3, status.BackgroundTransparency = Color3.new(0, 0, 0), 0.45
status.TextColor3 = Color3.new(1, 1, 1)
status.Font, status.TextSize = Enum.Font.GothamMedium, 15
status.Parent = gui

game:GetService("StarterGui"):SetCoreGuiEnabled(Enum.CoreGuiType.All, false)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")
local camera = workspace.CurrentCamera
camera.CameraType = Enum.CameraType.Scriptable
camera.CFrame = CFrame.lookAt(Vector3.new(4, 44, 44), Vector3.new(4, 15, 0))
-- A label above each tail, placed each frame (the viewport has no size at first).
local labels: { [Entry]: TextLabel } = {}
for _, e in entries do
	local l = Instance.new("TextLabel")
	l.AnchorPoint = Vector2.new(0.5, 1)
	labels[e] = l
	l.Size = UDim2.fromOffset(280, 22)
	l.BackgroundTransparency = 1
	l.TextColor3, l.TextStrokeTransparency = e.color, 0.4
	l.Font, l.TextSize = Enum.Font.GothamBold, 16
	l.Text = e.title
	l.Parent = gui
end

local clock = 0
local cost = { 0, 0, 0 } -- ms per frame, per row
RunService.Heartbeat:Connect(function(dt: number)
	clock += math.min(dt, 1 / 20)
	local spent = { 0, 0, 0 }
	local cfs, parts = {}, {}
	for _, e in entries do
		local t0 = os.clock()
		local rate = e.tail.rate
		while e.tail.t + 1 / rate <= clock + 1e-9 do
			e.tail:step(motion)
		end
		spent[e.row] += os.clock() - t0
		local nodes = e.tail:nodes()
		local function world(v: vector): Vector3
			return Vector3.new(v.x, v.y, v.z) + e.origin
		end
		if e.ghost then
			while e.ghost.t + 1 / 60 <= clock + 1e-9 do
				e.ghost:step(motion)
			end
			local g = e.ghost:nodes()
			for i = 1, tails.BONES do
				local a, b = world(g[i]), world(g[i + 1])
				table.insert(parts, e.ghost_bones[i])
				table.insert(cfs, CFrame.lookAt((a + b) / 2, b))
			end
		end
		table.insert(parts, e.root)
		table.insert(cfs, CFrame.new(world(nodes[1])))
		for i = 1, tails.BONES do
			local a, b = world(nodes[i]), world(nodes[i + 1])
			table.insert(parts, e.bones[i])
			table.insert(cfs, CFrame.lookAt((a + b) / 2, b))
			table.insert(parts, e.joints[i])
			table.insert(cfs, CFrame.new(b))
		end
	end
	workspace:BulkMoveTo(parts, cfs, Enum.BulkMoveMode.FireCFrameChanged)
	for e, l in labels do
		local p = camera:WorldToViewportPoint(e.origin + Vector3.new(4, 3.5, 0))
		l.Position = UDim2.fromOffset(p.X, p.Y)
	end
	for r = 1, 3 do
		cost[r] = 0.95 * cost[r] + 0.05 * spent[r] * 1000
	end
	status.Text = string.format("Time per frame for each row of three tails: SmartBone rule %.3f ms, XPBD %.3f ms, implicit %.3f ms", cost[1], cost[2], cost[3])
		.. "\nThe middle row is tuned to move like SmartBone's 60 Hz tail; citrate's tails move the same at every rate."
end)
