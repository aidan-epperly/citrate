--!strict
-- Cloth demo (LocalScript): two 30 by 30 cloths, one stepped implicitly (one linearized
-- backward Euler step per frame, ode.implicit_mechanics) and one explicitly (symplectic
-- Euler with substeps), at the same spring stiffness. Raise the stiffness and the
-- explicit cloth explodes while the implicit one stays put.

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
local SUBSTEPS = 10
local stiffness = 2000
local wind = false

-- One cloth: its model, state, mesh and timing.
type Sim = {
	name: string,
	origin: Vector3,
	model: any,
	q: buffer,
	v: buffer,
	it: any,
	mesh: any,
	vertices: { number },
	part: MeshPart?,
	ms: number,
	exploded: number,
	implicit: boolean,
}

local function build(sim: Sim)
	sim.model = cloth.new({ nx = N, ny = N, spacing = 1 / (N - 1), k_struct = stiffness, k_shear = stiffness / 5, k_bend = stiffness / 25 })
	sim.model.wind = if wind then { 0, 0, 6 } else { 0, 0, 0 }
	sim.q = f64.copy(nil, sim.model.q0)
	sim.v = f64.copy(nil, sim.model.v0)
	if sim.implicit then
		sim.it = ode.implicit_mechanics(f64.space, sim.model.force, 0, sim.q, sim.v, {
			system = sim.model.system,
			mass = sim.model.mass,
			fixed = sim.model.fixed,
			tol = 1e-3,
		})
	end
	sim.exploded = 0
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

local function update_mesh(sim: Sim)
	local q, mesh, ids = sim.q, sim.mesh, sim.vertices
	for p = 1, N * N do
		local o = 24 * (p - 1)
		mesh:SetPosition(ids[p], Vector3.new(buffer.readf64(q, o), buffer.readf64(q, o + 8), buffer.readf64(q, o + 16)) * SCALE)
	end
end

local function show(sim: Sim, color: Color3)
	-- Create the part from a mesh spanning a full cube (the cloth's range of motion), so
	-- its size matches the mesh's and vertex coordinates map to studs one to one; a flat
	-- starting mesh would have zero height, and the part would scale heights wildly.
	update_mesh(sim)
	sim.mesh:SetPosition(sim.vertices[N * N], Vector3.new(SCALE, -SCALE, SCALE))
	local part = AssetService:CreateMeshPartAsync(Content.fromObject(sim.mesh))
	part.Anchored = true
	part.CanCollide = false
	part.DoubleSided = true
	part.Color = color
	part.Material = Enum.Material.Fabric
	-- The mesh is centered on the part: keep the part at the pinned edge's height and
	-- let the vertices move in its local frame.
	part.CFrame = CFrame.new(sim.origin)
	part.Parent = workspace
	sim.part = part
end

local sims: { Sim } = {
	{ name = "Implicit (1 step/frame)", origin = Vector3.new(-12, 22, 0), implicit = true } :: any,
	{ name = `Explicit ({SUBSTEPS} substeps/frame)`, origin = Vector3.new(12, 22, 0), implicit = false } :: any,
}
for k, sim in sims do
	sim.ms = 0
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
panel.Size = UDim2.fromOffset(330, 150)
panel.Position = UDim2.new(0, 16, 1, -166)
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
local function label(text: string): TextLabel
	local l = Instance.new("TextLabel")
	order += 1
	l.LayoutOrder = order
	l.Size = UDim2.new(1, -20, 0, 20)
	l.BackgroundTransparency = 1
	l.TextColor3 = Color3.new(1, 1, 1)
	l.TextXAlignment = Enum.TextXAlignment.Left
	l.Font = Enum.Font.GothamMedium
	l.TextSize = 15
	l.Text = text
	l.Parent = panel
	return l
end
local stiffness_label = label("")
local rows = { label(""), label("") }

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
local function rebuild()
	for _, sim in sims do
		build(sim)
	end
end
button("Stiffer", function()
	stiffness = math.min(stiffness * 3, 1e6)
	rebuild()
end)
button("Softer", function()
	stiffness = math.max(stiffness / 3, 50)
	rebuild()
end)
local wind_button: TextButton
wind_button = button("Wind", function()
	wind = not wind
	for _, sim in sims do
		sim.model.wind = if wind then { 0, 0, 6 } else { 0, 0, 0 }
	end
	wind_button.BackgroundColor3 = if wind then Color3.fromRGB(60, 120, 80) else Color3.fromRGB(60, 60, 75)
end)
button("Reset", rebuild)
local hint = label("Pinned at two corners. Stiffness applies to both.")
hint.TextSize = 12
hint.TextColor3 = Color3.fromRGB(180, 180, 190)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")

local function finite(q: buffer): boolean
	local n = f64.norm(q)
	return n < 1e4
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
			local start = os.clock()
			if sim.implicit then
				sim.it:step(H)
				sim.q, sim.v = sim.it.q, sim.it.v
			else
				sim.q, sim.v = sim.model.explicit_step(sim.q, sim.v, H, SUBSTEPS)
			end
			sim.ms = 0.9 * sim.ms + 0.1 * (os.clock() - start) * 1000
			if not finite(sim.q) then
				sim.exploded = 1.5
			end
		end
	end
	for k, sim in sims do
		local part = sim.part :: MeshPart
		if sim.exploded <= 0 then
			update_mesh(sim)
			part.Transparency = 0
			rows[k].Text = string.format("%s: %.1f ms", sim.name, sim.ms)
		else
			part.Transparency = 1
			rows[k].Text = `{sim.name}: EXPLODED, resetting`
		end
	end
	stiffness_label.Text = string.format("Spring stiffness: %d", stiffness)
end)
