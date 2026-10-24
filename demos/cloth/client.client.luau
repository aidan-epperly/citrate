--!strict
-- Cloth demo (LocalScript): two 30 by 30 cloths, pinned along their top edge, each
-- holding a heavy steel bar along the bottom, with the same CPU time per frame.
--
-- The implicit cloth takes one linearized backward Euler step per frame
-- (ode.implicit_mechanics) with stiff springs. The explicit cloth (symplectic Euler) gets
-- as many substeps as fit in the same time, measured at startup, and therefore the
-- stiffest springs that stay stable at that many substeps; that is all an explicit solver
-- can afford, and under the bar's weight such a cloth stretches and bounces like rubber.

local AssetService = game:GetService("AssetService")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local Citrate = ReplicatedStorage:WaitForChild("Citrate")
local root = require(Citrate:WaitForChild("src")) :: any
local cloth = require(Citrate:WaitForChild("demos"):WaitForChild("cloth"):WaitForChild("cloth")) :: any
local f64, ode = root.f64, root.ode

local N = 30
local H = 1 / 60
local SCALE = 14 -- studs per meter
-- Workspace attributes: Stiffness (the implicit cloth's springs), Bar (kg), Wind (start
-- with wind), Record (a clean view: fixed camera, no buttons or core UI), Preconditioner
-- ("auto", the default: the faster of the library's multigrid and block Jacobi, timed at
-- startup; "multigrid" or "jacobi").
local stiffness = workspace:GetAttribute("Stiffness") or 1e6
local bar_mass = workspace:GetAttribute("Bar") or 1000
local wind = workspace:GetAttribute("Wind") == true
local record = workspace:GetAttribute("Record") == true
-- Match = "time" (default): the explicit cloth gets the implicit step's CPU time, and the
-- springs that allows. Match = "stiffness": the same springs, and the substeps they need.
local match_stiffness = workspace:GetAttribute("Match") == "stiffness"
local requested: string = workspace:GetAttribute("Preconditioner") or "auto"
local preconditioner, best = "multigrid", "multigrid"
-- The library's multigrid, refilled every frame; plain aggregation refills fastest.
local AMG = { smooth = false }
-- The stiffest explicit springs stable at s substeps per 60 Hz frame are about
-- 26 to 30 s^2 for this hanging cloth (measured over 15 s with wind; the light particles
-- set the limit, not the bar): use 22 s^2 for a margin.
local STABLE = 22

type Sim = {
	name: string,
	origin: Vector3,
	implicit: boolean,
	k: number,
	substeps: number,
	model: any,
	q: buffer,
	v: buffer,
	it: any,
	mesh: any,
	vertices: { number },
	part: MeshPart?,
	bar: Part?,
	ms: number,
	exploded: number,
}

local function build(sim: Sim)
	local k = sim.k
	sim.model = cloth.new({
		nx = N,
		ny = N,
		spacing = 1 / (N - 1),
		k_struct = k,
		k_shear = k / 5,
		k_bend = k / 25,
		bar = bar_mass,
		pin_top = true,
		hanging = true,
	})
	sim.model.wind = if wind then { 0, 0, 6 } else { 0, 0, 0 }
	sim.q = f64.copy(nil, sim.model.q0)
	sim.v = f64.copy(nil, sim.model.v0)
	if sim.implicit then
		sim.it = ode.implicit_mechanics(f64.space, sim.model.force, 0, sim.q, sim.v, {
			jacobian = sim.model.block_jacobian,
			amg = if preconditioner == "multigrid" then AMG else nil,
			mass = sim.model.mass,
			fixed = sim.model.fixed,
			tol = 3e-3,
		})
	end
	sim.exploded = 0
end

local function step(sim: Sim)
	if sim.implicit then
		sim.it:step(H)
		sim.q, sim.v = sim.it.q, sim.it.v
	else
		sim.q, sim.v = sim.model.explicit_step(sim.q, sim.v, H, sim.substeps)
	end
end

local function make_mesh(sim: Sim)
	local mesh = AssetService:CreateEditableMesh()
	assert(mesh, "EditableMesh unavailable (enable Mesh / Image APIs in the experience settings)")
	local ids = table.create(N * N, 0)
	for p = 1, N * N do
		ids[p] = mesh:AddVertex(Vector3.zero)
	end
	for j = 1, N - 1 do
		for i = 1, N - 1 do
			local a, b = (j - 1) * N + i, (j - 1) * N + i + 1
			local c, d = j * N + i, j * N + i + 1
			mesh:AddTriangle(ids[a], ids[c], ids[b])
			mesh:AddTriangle(ids[b], ids[c], ids[d])
		end
	end
	sim.mesh, sim.vertices = mesh, ids
end

local function point(sim: Sim, p: number): Vector3
	local o = 24 * (p - 1)
	local q = sim.q
	return Vector3.new(buffer.readf64(q, o), buffer.readf64(q, o + 8), buffer.readf64(q, o + 16)) * SCALE
end

local function update_mesh(sim: Sim)
	local mesh, ids = sim.mesh, sim.vertices
	for p = 1, N * N do
		mesh:SetPosition(ids[p], point(sim, p))
	end
	-- The bar runs between the bottom row's end particles.
	local a = sim.origin + point(sim, (N - 1) * N + 1)
	local b = sim.origin + point(sim, N * N)
	local bar = sim.bar :: Part
	bar.Size = Vector3.new((b - a).Magnitude + 1, 0.9, 0.9)
	bar.CFrame = CFrame.lookAt((a + b) / 2, b) * CFrame.Angles(0, math.pi / 2, 0)
end

-- The cloth's length: the bar's drop below the pinned edge, in meters.
local function length(sim: Sim): number
	local total = 0
	for i = 1, N do
		total += buffer.readf64(sim.q, 24 * ((N - 1) * N + i - 1) + 8)
	end
	return -total / N
end

local function show(sim: Sim, color: Color3)
	-- Create the part from a mesh spanning a full cube (the cloth's range of motion), so
	-- the part's size matches the mesh's and vertex coordinates map to studs one to one.
	local mesh, ids = sim.mesh, sim.vertices
	for p = 1, N * N do
		mesh:SetPosition(ids[p], point(sim, p))
	end
	mesh:SetPosition(ids[N * N], Vector3.new(SCALE, -SCALE, SCALE))
	local part = AssetService:CreateMeshPartAsync(Content.fromObject(mesh))
	part.Anchored = true
	part.CanCollide = false
	part.DoubleSided = true
	part.Color = color
	part.Material = Enum.Material.Fabric
	part.CFrame = CFrame.new(sim.origin)
	part.Parent = workspace
	sim.part = part
	local bar = Instance.new("Part")
	bar.Shape = Enum.PartType.Cylinder
	bar.Anchored = true
	bar.CanCollide = false
	bar.Material = Enum.Material.DiamondPlate
	bar.Color = Color3.fromRGB(90, 95, 105)
	bar.Parent = workspace
	sim.bar = bar
end

local sims: { Sim } = {
	{ name = "Implicit", origin = Vector3.new(-12, 24, 0), implicit = true, k = stiffness, substeps = 1 } :: any,
	{ name = "Explicit", origin = Vector3.new(12, 24, 0), implicit = false, k = 1000, substeps = 1 } :: any,
}
local implicit, explicit = sims[1], sims[2]

-- Calibration: the implicit step's cost and an explicit substep's cost, so the explicit
-- cloth gets the same time per frame. Each runs 3 s to settle (the implicit step is
-- dearer while the bar first drops), then is timed over 90 frames.
local function timed(sim: Sim): number
	build(sim)
	for frame = 1, 180 do
		step(sim)
		if frame % 5 == 0 then
			task.wait()
		end
	end
	local total = 0
	for frame = 1, 90 do
		local t0 = os.clock()
		step(sim)
		total += os.clock() - t0
		if frame % 5 == 0 then
			task.wait()
		end
	end
	return total / 90 * 1000
end
-- The preconditioner: the attribute's, or (auto) whichever of the library's multigrid and
-- block Jacobi runs faster here.
local implicit_ms = math.huge
for _, choice in (if requested == "auto" then { "multigrid", "jacobi" } else { requested }) :: { string } do
	preconditioner = choice
	local ms = timed(implicit)
	if ms < implicit_ms then
		implicit_ms, best = ms, choice
	end
end
preconditioner = best
local substep_ms = 0
if match_stiffness then
	explicit.k = stiffness
	explicit.substeps = math.ceil(math.sqrt(stiffness / STABLE))
else
	explicit.substeps = 40
	substep_ms = timed(explicit) / 40
	explicit.substeps = math.max(1, math.floor(implicit_ms / substep_ms))
	explicit.k = math.min(STABLE * explicit.substeps ^ 2, stiffness)
end

for k, sim in sims do
	sim.ms = if sim.implicit then implicit_ms else 0
	build(sim)
	make_mesh(sim)
	show(sim, if k == 1 then Color3.fromRGB(70, 140, 220) else Color3.fromRGB(220, 110, 70))
end

-- Interface
local gui = Instance.new("ScreenGui")
gui.Name = "ClothDemo"
gui.ResetOnSpawn = false
local panel = Instance.new("Frame")
panel.Name = "Panel"
panel.Size = UDim2.fromOffset(520, 162)
panel.Position = UDim2.new(0, 16, 1, -178)
panel.BackgroundColor3 = Color3.fromRGB(25, 25, 30)
panel.BackgroundTransparency = 0.15
panel.Parent = gui
local layout = Instance.new("UIListLayout")
layout.SortOrder = Enum.SortOrder.LayoutOrder
layout.Padding = UDim.new(0, 6)
layout.Parent = panel
local padding = Instance.new("UIPadding")
padding.PaddingLeft = UDim.new(0, 10)
padding.PaddingTop = UDim.new(0, 10)
padding.Parent = panel

local order = 0
local function label(text: string, size: number?): TextLabel
	local l = Instance.new("TextLabel")
	order += 1
	l.LayoutOrder = order
	l.Size = UDim2.new(1, -20, 0, (size or 15) + 5)
	l.BackgroundTransparency = 1
	l.TextColor3 = Color3.new(1, 1, 1)
	l.TextXAlignment = Enum.TextXAlignment.Left
	l.Font = Enum.Font.GothamMedium
	l.TextSize = size or 15
	l.Text = text
	l.Parent = panel
	return l
end
label(string.format("%s. Steel bar: %d kg; 30 x 30 cloth: 9 kg.", if match_stiffness then "Same springs" else "Same CPU time per frame", bar_mass), 13)
local rows = {}
rows[1] = label("")
label(string.format("    1 step/frame, spring stiffness %.0f, %s preconditioner", implicit.k, if preconditioner == "jacobi" then "block Jacobi" else preconditioner), 13)
rows[2] = label("")
label(if match_stiffness
	then string.format("    spring stiffness %.0f, so %d substeps/frame to stay stable", explicit.k, explicit.substeps)
	else string.format("    %d substeps/frame, so the stiffest stable springs: %.0f", explicit.substeps, explicit.k), 13)

local buttons = Instance.new("Frame")
buttons.Name = "Buttons"
order += 1
buttons.LayoutOrder = order
buttons.Size = UDim2.new(1, -20, 0, 30)
buttons.BackgroundTransparency = 1
buttons.Parent = panel
local row = Instance.new("UIListLayout")
row.SortOrder = Enum.SortOrder.LayoutOrder
row.FillDirection = Enum.FillDirection.Horizontal
row.Padding = UDim.new(0, 6)
row.Parent = buttons
local function button(text: string, action: () -> ()): TextButton
	local b = Instance.new("TextButton")
	b.Name = text
	order += 1
	b.LayoutOrder = order
	b.Size = UDim2.fromOffset(72, 28)
	b.BackgroundColor3 = Color3.fromRGB(60, 60, 75)
	b.TextColor3 = Color3.new(1, 1, 1)
	b.Font = Enum.Font.GothamBold
	b.TextSize = 14
	b.Text = text
	b.Parent = buttons
	b.Activated:Connect(action)
	return b
end
local wind_button: TextButton
wind_button = button("Wind", function()
	wind = not wind
	for _, sim in sims do
		sim.model.wind = if wind then { 0, 0, 6 } else { 0, 0, 0 }
	end
	wind_button.BackgroundColor3 = if wind then Color3.fromRGB(60, 120, 80) else Color3.fromRGB(60, 60, 75)
end)
button("Reset", function()
	for _, sim in sims do
		build(sim)
	end
end)
if record then
	buttons.Visible = false
	panel.Size = UDim2.fromOffset(520, 126)
	panel.Position = UDim2.new(0, 16, 1, -142)
	game:GetService("StarterGui"):SetCoreGuiEnabled(Enum.CoreGuiType.All, false)
	local camera = workspace.CurrentCamera
	camera.CameraType = Enum.CameraType.Scriptable
	camera.FieldOfView = 50
	camera.CFrame = CFrame.lookAt(Vector3.new(0, 15, 44), Vector3.new(0, 14, 0))
end
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")

-- Exploded: a coordinate more than 2 m out (the cloth spans 1 m), or not finite. Checked
-- before each frame is drawn, so a cloth coming apart is never shown.
local function finite(q: buffer): boolean
	return f64.amax(q) < 2
end

local accumulator = 0
RunService.Heartbeat:Connect(function(dt: number)
	accumulator = math.min(accumulator + dt, 4 * H)
	while accumulator >= H do
		accumulator -= H
		for _, sim in sims do
			if sim.exploded > 0 then
				sim.exploded -= H
				if sim.exploded <= 0 then
					build(sim)
				end
				continue
			end
			local t0 = os.clock()
			step(sim)
			sim.ms = 0.95 * sim.ms + 0.05 * (os.clock() - t0) * 1000
			if not finite(sim.q) then
				sim.exploded = 1.5
			end
		end
	end
	for k, sim in sims do
		local part, bar = sim.part :: MeshPart, sim.bar :: Part
		if sim.exploded <= 0 then
			update_mesh(sim)
			part.Transparency = 0
			bar.Transparency = 0
			rows[k].Text = string.format("%s: %4.1f ms/frame, length %.2f m (rest 1.00 m)", sim.name, sim.ms, length(sim))
		else
			part.Transparency = 1
			bar.Transparency = 1
			rows[k].Text = `{sim.name}: exploded, resetting`
		end
	end
end)
