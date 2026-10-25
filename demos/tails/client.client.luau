--!strict
--!native
-- Tails demo (LocalScript): six identical tails on roots that shake, rest and shake again.
-- Top row: SmartBone 2's rule; bottom row: the same tail as a physical system advanced by
-- citrate's implicit mechanics integrator. The columns step at 60, 30 and 15 Hz, as
-- SmartBone does for objects 15, 30 and 37.5 studs from the camera.

local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local Citrate = ReplicatedStorage:WaitForChild("Citrate")
local tails = require(Citrate:WaitForChild("demos"):WaitForChild("tails"):WaitForChild("tails")) :: any

local RATES = { 60, 30, 15 }
local SPACING = 18 -- studs between columns
local ROWS = { { name = "SmartBone rule", make = tails.smartbone, y = 26, color = Color3.fromRGB(240, 150, 70) }, { name = "citrate", make = tails.physical, y = 8, color = Color3.fromRGB(80, 170, 255) } }
local motion = tails.motion(0.8, 4 / 3, 6) -- shake for 4/3 s every 6 s

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
type Entry = { tail: any, origin: Vector3, root: Part, bones: { Part }, joints: { Part }, row: number, title: string, color: Color3 }
local entries: { Entry } = {}
for r, row in ROWS do
	for c, rate in RATES do
		local origin = Vector3.new((c - 2) * SPACING - 4, row.y, 0)
		local e: Entry = { tail = row.make(rate), origin = origin, root = part(Vector3.new(2, 2, 2), Color3.fromRGB(90, 90, 100)), bones = {}, joints = {}, row = r, title = `{row.name}, {rate} Hz`, color = row.color }
		for i = 1, tails.BONES do
			e.bones[i] = part(Vector3.new(0.5, 0.5, tails.LENGTH), row.color)
			e.joints[i] = part(Vector3.new(0.7, 0.7, 0.7), row.color:Lerp(Color3.new(1, 1, 1), 0.3), Enum.PartType.Ball)
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
status.Size = UDim2.fromOffset(640, 64)
status.BackgroundColor3, status.BackgroundTransparency = Color3.new(0, 0, 0), 0.45
status.TextColor3 = Color3.new(1, 1, 1)
status.Font, status.TextSize = Enum.Font.GothamMedium, 15
status.Parent = gui

game:GetService("StarterGui"):SetCoreGuiEnabled(Enum.CoreGuiType.All, false)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")
local camera = workspace.CurrentCamera
camera.CameraType = Enum.CameraType.Scriptable
camera.CFrame = CFrame.lookAt(Vector3.new(4, 18, 40), Vector3.new(4, 16, 0))
-- A label above each tail, placed each frame (the viewport has no size at first).
local labels: { [Entry]: TextLabel } = {}
for _, e in entries do
	local l = Instance.new("TextLabel")
	l.AnchorPoint = Vector2.new(0.5, 1)
	labels[e] = l
	l.Size = UDim2.fromOffset(220, 22)
	l.BackgroundTransparency = 1
	l.TextColor3, l.TextStrokeTransparency = e.color, 0.4
	l.Font, l.TextSize = Enum.Font.GothamBold, 16
	l.Text = e.title
	l.Parent = gui
end

local clock = 0
local cost = { 0, 0 } -- ms per frame, per row
RunService.Heartbeat:Connect(function(dt: number)
	clock += math.min(dt, 1 / 20)
	local spent = { 0, 0 }
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
	for r = 1, 2 do
		cost[r] = 0.95 * cost[r] + 0.05 * spent[r] * 1000
	end
	status.Text = string.format("Time per frame for three tails: SmartBone rule %.3f ms, citrate %.3f ms\nSmartBone's tail droops further and swings longer at lower rates; citrate's is the same at every rate.", cost[1], cost[2])
end)
