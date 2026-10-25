--!strict
-- Bones demo (LocalScript): creatures walking in circles, each with a tail and a pair of
-- antennae made of Bones, simulated by citrate's dynamic bones (roblox/bones.luau). The
-- left one is updated every frame, the others every second and third frame, as the
-- component throttles distant rigs; their simulation runs in substeps of the same length,
-- so they all move alike. The balls are colliders; the button blows wind.

local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local Citrate = ReplicatedStorage:WaitForChild("Citrate")
local Bones = require(Citrate:WaitForChild("roblox"):WaitForChild("bones")) :: any

local folder = Instance.new("Folder")
folder.Name = "BonesDemo"
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


type Creature = { body: Part, center: Vector3, radius: number, speed: number, rig_tail: any, rig_antennae: any, every: number, waited: number, segments: { { a: Bone, b: Bone, p: Part } }, ms: number }
local creatures: { Creature } = {}

-- A chain of `count` bones of length `len` under `parent`, starting at `offset` and going
-- along `dir` (in the parent's space).
local function chain(parent: Instance, name: string, offset: Vector3, dir: Vector3, count: number, len: number): Bone
	local root = Instance.new("Bone")
	root.Name = name
	root.CFrame = CFrame.new(offset)
	root.Parent = parent
	local prev = root
	for i = 1, count do
		local b = Instance.new("Bone")
		b.Name = name .. i
		b.CFrame = CFrame.new(dir * len)
		b.Parent = prev
		prev = b
	end
	return root
end

local COLORS = { Color3.fromRGB(240, 150, 70), Color3.fromRGB(120, 220, 140), Color3.fromRGB(80, 170, 255) }
local RADIUS = 6
for k = 1, 3 do
	local center = Vector3.new(16 * (k - 2), 0, 0)
	-- a ball on its path, for the tail to drag over
	local ball = part(Vector3.new(3, 3, 3), Color3.fromRGB(110, 110, 125), Enum.PartType.Ball)
	ball.Position = center + Vector3.new(0, 1.5, RADIUS)
	local body = part(Vector3.new(2.4, 2, 3.2), COLORS[k])
	local tail = chain(body, "Tail", Vector3.new(0, 0.3, 1.6), Vector3.new(0, 0, 1), 8, 1.1)
	-- antennae: a stalk that forks in two
	local antenna = chain(body, "Antenna", Vector3.new(0, 1, -1.2), Vector3.new(0, 1, -0.3).Unit, 2, 0.8)
	local tip = antenna:FindFirstChild("Antenna1"):FindFirstChild("Antenna2") :: Bone
	for side = -1, 1, 2 do
		local b = tip
		for i = 1, 3 do
			local nb = Instance.new("Bone")
			nb.Name = "Fork" .. side .. i
			nb.CFrame = CFrame.new(Vector3.new(side * 0.4, 0.5, -0.2).Unit * 0.7)
			nb.Parent = b
			b = nb
		end
	end
	local c: Creature = {
		body = body,
		center = center,
		radius = RADIUS,
		speed = 0.7,
		rig_tail = Bones.new(tail, { colliders = { ball }, radius = 0.25 }),
		rig_antennae = Bones.new(antenna, { stiffness = 2000 }),
		every = k,
		waited = 0,
		segments = {},
		ms = 0,
	}
	-- a segment for each bone, drawn from its parent
	for _, b in body:GetDescendants() do
		if b:IsA("Bone") and b.Parent and b.Parent:IsA("Bone") then
			table.insert(c.segments, { a = b.Parent :: Bone, b = b :: Bone, p = part(Vector3.new(0.35, 0.35, 1), COLORS[k]:Lerp(Color3.new(1, 1, 1), 0.25)) })
		end
	end
	table.insert(creatures, c)
end

-- Panel
local gui = Instance.new("ScreenGui")
gui.Name, gui.ResetOnSpawn, gui.IgnoreGuiInset = "BonesDemo", false, true
local status = Instance.new("TextLabel")
status.AnchorPoint = Vector2.new(0.5, 1)
status.Position = UDim2.new(0.5, 0, 1, -16)
status.Size = UDim2.fromOffset(720, 64)
status.BackgroundColor3, status.BackgroundTransparency = Color3.new(0, 0, 0), 0.45
status.TextColor3 = Color3.new(1, 1, 1)
status.Font, status.TextSize = Enum.Font.GothamMedium, 15
status.Parent = gui
local wind = Instance.new("TextButton")
wind.Name = "Wind"
wind.AnchorPoint = Vector2.new(1, 0)
wind.Position = UDim2.new(1, -16, 0, 16)
wind.Size = UDim2.fromOffset(110, 32)
wind.BackgroundColor3 = Color3.fromRGB(55, 55, 70)
wind.TextColor3, wind.Font, wind.TextSize = Color3.new(1, 1, 1), Enum.Font.GothamBold, 15
wind.Text = "Wind: off"
wind.Parent = gui
wind.Activated:Connect(function()
	local on = workspace.GlobalWind.Magnitude == 0
	workspace.GlobalWind = if on then Vector3.new(12, 0, 4) else Vector3.zero
	wind.Text = if on then "Wind: on" else "Wind: off"
end)
game:GetService("StarterGui"):SetCoreGuiEnabled(Enum.CoreGuiType.All, false)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")
local camera = workspace.CurrentCamera
camera.CameraType = Enum.CameraType.Scriptable
camera.CFrame = CFrame.lookAt(Vector3.new(0, 22, 30), Vector3.new(0, 1, 0))

local clock = 0
RunService.Heartbeat:Connect(function(dt: number)
	clock += dt
	local parts, cfs = {}, {}
	local text = {}
	for k, c in creatures do
		-- walk in a circle, bobbing
		local a = clock * c.speed
		local pos = c.center + Vector3.new(c.radius * math.cos(a), 3.4 + 0.4 * math.sin(6 * a), c.radius * math.sin(a))
		local ahead = Vector3.new(-math.sin(a), 0, math.cos(a))
		c.body.CFrame = CFrame.lookAt(pos, pos + ahead)
		-- update the bones every c.every frames, with the time since the last update
		c.waited += dt
		if (clock * 60) % c.every < 1 then
			local t0 = os.clock()
			c.rig_tail:step(c.waited)
			c.rig_antennae:step(c.waited)
			c.ms = 0.9 * c.ms + 0.1 * (os.clock() - t0) * 1000
			c.waited = 0
		end
		for _, s in c.segments do
			local p, q = s.a.WorldPosition, s.b.WorldPosition
			s.p.Size = Vector3.new(0.35, 0.35, math.max((q - p).Magnitude, 0.05))
			table.insert(parts, s.p)
			table.insert(cfs, CFrame.lookAt((p + q) / 2, q))
		end
		table.insert(text, string.format("updated every %d frame%s: %.3f ms", c.every, if c.every > 1 then "s" else "", c.ms))
	end
	workspace:BulkMoveTo(parts, cfs, Enum.BulkMoveMode.FireCFrameChanged)
	status.Text = "Bones per creature: a tail of 8 and two forked antennae (14), by citrate's dynamic bones\n" .. table.concat(text, "    ")
end)
