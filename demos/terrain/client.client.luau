--!strict
--!native
-- Terrain demo (LocalScript): marbles rolling over two copies of a heightfield known only
-- at grid points (the dark dots). Left: the surface between the points interpolated
-- bilinearly, the usual approach; its slope jumps at every grid line. Right: citrate's
-- interp.grid (cubic); its slope is continuous. The marbles start at the same places and
-- are stepped the same way, with no friction, so ideally they would roll forever.

local AssetService = game:GetService("AssetService")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local Citrate = ReplicatedStorage:WaitForChild("Citrate")
local terrain = require(Citrate:WaitForChild("demos"):WaitForChild("terrain"):WaitForChild("terrain")) :: any

local SIZE, N, SPACING = terrain.SIZE, terrain.N, terrain.SPACING
local SUB = 4 -- mesh vertices per grid spacing (each mesh samples its own surface)
local GAP = 24 -- studs between the two terrains
local LIFT = 12 -- studs above the baseplate
local COUNTS = { 50, 200, 1000 }
local SUBSTEPS = { 1, 2, 4, 8 }
local TRAILS = 12 -- marbles with trails
local MOVING = 3 -- studs/s: slower counts as stopped

local heights = terrain.heights()
local linear, cubic = terrain.surfaces(heights)
local low = math.huge
for i = 1, N do
	for j = 1, N do
		low = math.min(low, heights[i][j])
	end
end

local folder = Instance.new("Folder")
folder.Name = "TerrainDemo"
folder.Parent = workspace

type Side = {
	name: string,
	S: any,
	origin: Vector3,
	color: Color3,
	marbles: any,
	balls: { Part },
	e0: number,
	avail: number,
	ms: number,
	label: TextLabel,
}
local sides: { Side } = {
	{ name = "Bilinear (slope jumps at grid lines)", S = linear, origin = Vector3.new(-SIZE - GAP / 2, LIFT, -SIZE / 2), color = Color3.fromRGB(196, 150, 104), ms = 0 } :: any,
	{ name = "citrate interp.grid (cubic)", S = cubic, origin = Vector3.new(GAP / 2, LIFT, -SIZE / 2), color = Color3.fromRGB(120, 160, 200), ms = 0 } :: any,
}

-- The terrain meshes: SUB vertices per grid spacing, each height from the side's own
-- surface, so the marbles roll on what is drawn.
for _, sd in sides do
	local mesh = AssetService:CreateEditableMesh()
	assert(mesh, "EditableMesh unavailable (enable Mesh / Image APIs in the experience settings)")
	local m = (N - 1) * SUB + 1
	local ys = table.create(m * m, 0)
	local lo, hi = math.huge, -math.huge
	for a = 1, m do
		for b = 1, m do
			local y = sd.S:at((a - 1) * SPACING / SUB, (b - 1) * SPACING / SUB)
			ys[(a - 1) * m + b] = y
			lo, hi = math.min(lo, y), math.max(hi, y)
		end
	end
	-- vertices relative to the mesh's bounding box center, so the part sits where it should
	local center = Vector3.new(SIZE / 2, (lo + hi) / 2, SIZE / 2)
	local ids = table.create(m * m, 0)
	for a = 1, m do
		for b = 1, m do
			local p = Vector3.new((a - 1) * SPACING / SUB, ys[(a - 1) * m + b], (b - 1) * SPACING / SUB)
			ids[(a - 1) * m + b] = mesh:AddVertex(p - center)
		end
	end
	for a = 1, m - 1 do
		for b = 1, m - 1 do
			local p, q = ids[(a - 1) * m + b], ids[a * m + b]
			local r, s = ids[(a - 1) * m + b + 1], ids[a * m + b + 1]
			mesh:AddTriangle(p, r, q)
			mesh:AddTriangle(q, r, s)
		end
	end
	local part = AssetService:CreateMeshPartAsync(Content.fromObject(mesh))
	part.Anchored, part.CanCollide, part.CanQuery, part.CanTouch = true, false, false, false
	part.DoubleSided = true
	part.Color, part.Material = sd.color, Enum.Material.SmoothPlastic
	part.CFrame = CFrame.new(sd.origin + center)
	part.Name = sd.name
	part.Parent = folder
	-- the grid points: the only heights either surface is given
	for i = 1, N do
		for j = 1, N do
			local dot = Instance.new("Part")
			dot.Anchored, dot.CanCollide, dot.CanQuery, dot.CanTouch, dot.CastShadow = true, false, false, false, false
			dot.Shape = Enum.PartType.Ball
			dot.Size = Vector3.new(0.5, 0.5, 0.5)
			dot.Color, dot.Material = Color3.fromRGB(40, 40, 45), Enum.Material.SmoothPlastic
			dot.Position = sd.origin + Vector3.new((i - 1) * SPACING, heights[i][j] + 0.1, (j - 1) * SPACING)
			dot.Parent = folder
		end
	end
end

local count, substeps = 200, 4
local function marble_color(k: number): Color3
	return Color3.fromHSV((k * 0.618034) % 1, 0.75, 1)
end
local function reset()
	local start = terrain.marbles(count)
	for _, sd in sides do
		sd.marbles = terrain.copy(start)
		sd.e0 = terrain.energy(sd.S, sd.marbles)
		sd.avail = sd.e0 - count * terrain.GRAVITY * low
		sd.balls = sd.balls or {}
		while #sd.balls < count do
			local k = #sd.balls + 1
			local ball = Instance.new("Part")
			ball.Anchored, ball.CanCollide, ball.CanQuery, ball.CanTouch, ball.CastShadow = true, false, false, false, false
			ball.Shape = Enum.PartType.Ball
			ball.Size = Vector3.one * 2 * terrain.RADIUS
			ball.Color, ball.Material = marble_color(k), Enum.Material.SmoothPlastic
			ball.Parent = folder
			if k <= TRAILS then
				local a0, a1 = Instance.new("Attachment"), Instance.new("Attachment")
				a0.Position, a1.Position = Vector3.new(0, 0.3, 0), Vector3.new(0, -0.3, 0)
				a0.Parent, a1.Parent = ball, ball
				local trail = Instance.new("Trail")
				trail.Attachment0, trail.Attachment1 = a0, a1
				trail.Lifetime = 4
				trail.Color = ColorSequence.new(ball.Color)
				trail.Transparency = NumberSequence.new(0.2, 1)
				trail.FaceCamera = true
				trail.Parent = ball
			end
			table.insert(sd.balls, ball)
		end
		while #sd.balls > count do
			(table.remove(sd.balls) :: Part):Destroy()
		end
		for _, ball in sd.balls do
			for _, t in ball:GetChildren() do
				if t:IsA("Trail") then
					t:Clear()
				end
			end
		end
	end
end
reset()

-- Interface
local gui = Instance.new("ScreenGui")
gui.Name, gui.ResetOnSpawn = "TerrainDemo", false
local panel = Instance.new("Frame")
panel.Size = UDim2.fromOffset(560, 168)
panel.Position = UDim2.new(0, 16, 1, -184)
panel.BackgroundColor3, panel.BackgroundTransparency = Color3.fromRGB(25, 25, 30), 0.15
panel.Parent = gui
local layout = Instance.new("UIListLayout")
layout.SortOrder = Enum.SortOrder.LayoutOrder
layout.Padding = UDim.new(0, 5)
layout.Parent = panel
local padding = Instance.new("UIPadding")
padding.PaddingLeft, padding.PaddingTop = UDim.new(0, 10), UDim.new(0, 10)
padding.Parent = panel
local order = 0
local function label(text: string, size: number?, color: Color3?): TextLabel
	local l = Instance.new("TextLabel")
	order += 1
	l.LayoutOrder = order
	l.Size = UDim2.new(1, -20, 0, (size or 15) + 4)
	l.BackgroundTransparency = 1
	l.TextColor3 = color or Color3.new(1, 1, 1)
	l.TextXAlignment = Enum.TextXAlignment.Left
	l.Font, l.TextSize = Enum.Font.GothamMedium, size or 15
	l.Text = text
	l.Parent = panel
	return l
end
label(string.format("Heights known every %d studs (dots). No friction: ideally the marbles never stop.", SPACING), 13)
for _, sd in sides do
	sd.label = label("", 14, sd.color:Lerp(Color3.new(1, 1, 1), 0.4))
end
local note = label("", 13)
local function choices(title: string, options: { number }, current: number, on: (number) -> ())
	local row = Instance.new("Frame")
	order += 1
	row.LayoutOrder = order
	row.Size = UDim2.new(1, -20, 0, 26)
	row.BackgroundTransparency = 1
	row.Parent = panel
	local t = Instance.new("TextLabel")
	t.BackgroundTransparency = 1
	t.Size = UDim2.fromOffset(90, 26)
	t.TextColor3, t.Font, t.TextSize = Color3.new(1, 1, 1), Enum.Font.GothamMedium, 13
	t.TextXAlignment = Enum.TextXAlignment.Left
	t.Text = title
	t.Parent = row
	local buttons: { [number]: TextButton } = {}
	local function paint(v: number)
		for opt, b in buttons do
			b.BackgroundColor3 = if opt == v then Color3.fromRGB(90, 110, 160) else Color3.fromRGB(55, 55, 70)
		end
	end
	for k, opt in options do
		local b = Instance.new("TextButton")
		b.Name = title .. opt
		b.Size, b.Position = UDim2.fromOffset(52, 24), UDim2.fromOffset(92 + 56 * (k - 1), 1)
		b.TextColor3, b.Font, b.TextSize = Color3.new(1, 1, 1), Enum.Font.GothamBold, 13
		b.Text = tostring(opt)
		b.Parent = row
		buttons[opt] = b
		b.Activated:Connect(function()
			on(opt)
			paint(opt)
		end)
	end
	paint(current)
	return row
end
choices("Marbles", COUNTS, count, function(v)
	count = v
	reset()
end)
local sub_row = choices("Substeps", SUBSTEPS, substeps, function(v)
	substeps = v
	reset()
end)
local again = Instance.new("TextButton")
again.Name = "Reset"
again.Size, again.Position = UDim2.fromOffset(72, 24), UDim2.fromOffset(92 + 56 * #SUBSTEPS + 12, 1)
again.BackgroundColor3 = Color3.fromRGB(60, 60, 75)
again.TextColor3, again.Font, again.TextSize = Color3.new(1, 1, 1), Enum.Font.GothamBold, 13
again.Text = "Reset"
again.Parent = sub_row
again.Activated:Connect(reset)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")

game:GetService("StarterGui"):SetCoreGuiEnabled(Enum.CoreGuiType.All, false)
local camera = workspace.CurrentCamera
camera.CameraType = Enum.CameraType.Scriptable
camera.FieldOfView = 50
camera.CFrame = CFrame.lookAt(Vector3.new(0, 175, 175), Vector3.new(0, LIFT, -6))

local H = 1 / 60
local accumulator = 0
local elapsed = 0
RunService.Heartbeat:Connect(function(dt: number)
	accumulator = math.min(accumulator + dt, 4 * H)
	while accumulator >= H do
		accumulator -= H
		elapsed += H
		for _, sd in sides do
			local t0 = os.clock()
			for _ = 1, substeps do
				terrain.step(sd.S, sd.marbles, H / substeps, 0)
			end
			sd.ms = if sd.ms == 0 then (os.clock() - t0) * 1000 else 0.95 * sd.ms + 0.05 * (os.clock() - t0) * 1000
		end
	end
	for _, sd in sides do
		local m = sd.marbles
		local cfs = table.create(count)
		local moving = 0
		for k = 1, count do
			local p = terrain.position(sd.S, m, k)
			cfs[k] = CFrame.new(sd.origin + Vector3.new(p.x, p.y, p.z))
			if m.vx[k] ^ 2 + m.vy[k] ^ 2 + m.vz[k] ^ 2 > MOVING * MOVING then
				moving += 1
			end
		end
		workspace:BulkMoveTo(sd.balls, cfs, Enum.BulkMoveMode.FireCFrameChanged)
		local lost = (sd.e0 - terrain.energy(sd.S, m)) / sd.avail
		sd.label.Text = string.format("%s: energy lost %3.0f%%, %d of %d rolling, %.2f ms/frame", sd.name, 100 * lost, moving, count, sd.ms)
	end
	note.Text = string.format("%d substeps per frame. More substeps help only the cubic surface: the bilinear one's losses are its creases.", substeps)
end)
