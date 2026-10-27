--!strict
-- Drum demo (LocalScript): the Gordon-Webb-Wolpert drums, two different shapes with the
-- same frequencies. Each drum's lowest modes come from the library's Lanczos eigensolver
-- (krylov.eigsh, shift-invert: solves by banded Cholesky after a reverse Cuthill-McKee
-- ordering) on a finite-element Laplacian (drum.luau). Click a drum to strike it
-- there: it rings as the sum of its modes, each at its own frequency. The buttons show one
-- mode at a time on both drums, which then oscillate in step although their shapes differ.

local AssetService = game:GetService("AssetService")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local UserInputService = game:GetService("UserInputService")

local Citrate = ReplicatedStorage:WaitForChild("Citrate")
local drum = require(Citrate:WaitForChild("demos"):WaitForChild("drum"):WaitForChild("drum")) :: any
local f64 = require(Citrate:WaitForChild("src")).f64 :: any

local M = 16 -- subdivisions of each grid square
local MODES = 12
local SCALE = 8 -- studs per grid unit
local HEIGHT = 3 -- studs of displacement at a mode's peak
local DECAY = 0.25 -- per second
-- Driscoll's eigenvalues (SIAM Review 1997, accurate to 12 digits) for drums twice this
-- size: ours are four times these.
local DRISCOLL = { 2.53794399980, 3.65550971352, 5.17555935622, 6.53755744376, 7.24807786256, 9.20929499840, 10.5969856913, 11.5413953956, 12.3370055014, 13.0536540557, 14.3138624643, 15.8713026200 }

type Drum = {
	name: string,
	origin: Vector3,
	model: any,
	lambdas: { number },
	omegas: { number },
	-- each mode at every node (0 on the boundary), and its largest magnitude
	nodal: { { number } },
	peak: { number },
	-- center of the drum's bounding box (the part's origin), in grid units
	cx: number,
	cy: number,
	-- modal state: displacement and velocity of each mode
	q: { number },
	v: { number },
	mesh: any,
	ids: { number },
	part: MeshPart?,
}

local drums: { Drum } = {
	{ name = "Drum A", origin = Vector3.new(-2, 7, 4) } :: any,
	{ name = "Drum B", origin = Vector3.new(28, 7, 4) } :: any,
}

-- Interface: built first, so the progress shows while the modes are computed.
local gui = Instance.new("ScreenGui")
gui.Name = "DrumDemo"
gui.ResetOnSpawn = false
local panel = Instance.new("Frame")
panel.Name = "Panel"
panel.Size = UDim2.fromOffset(570, 380)
panel.Position = UDim2.fromOffset(16, 16)
panel.BackgroundColor3 = Color3.fromRGB(25, 25, 30)
panel.BackgroundTransparency = 0.15
panel.Parent = gui
local layout = Instance.new("UIListLayout")
layout.SortOrder = Enum.SortOrder.LayoutOrder
layout.Padding = UDim.new(0, 3)
layout.Parent = panel
local padding = Instance.new("UIPadding")
padding.PaddingLeft = UDim.new(0, 10)
padding.PaddingTop = UDim.new(0, 8)
padding.Parent = panel
local order = 0
local function label(text: string, size: number?, mono: boolean?): TextLabel
	local l = Instance.new("TextLabel")
	order += 1
	l.LayoutOrder = order
	l.Size = UDim2.new(1, -20, 0, (size or 14) + 4)
	l.BackgroundTransparency = 1
	l.TextColor3 = Color3.new(1, 1, 1)
	l.TextXAlignment = Enum.TextXAlignment.Left
	l.Font = if mono then Enum.Font.RobotoMono else Enum.Font.GothamMedium
	l.TextSize = size or 14
	l.Text = text
	l.Parent = panel
	return l
end
local title = label("Can you hear the shape of a drum? Computing the modes...", 15)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")

local camera = workspace.CurrentCamera
camera.CameraType = Enum.CameraType.Scriptable
camera.FieldOfView = 50
camera.CFrame = CFrame.lookAt(Vector3.new(6, 44, 38), Vector3.new(6, 0, -16))

for _, d in drums do
	local poly = if d.name == "Drum A" then drum.gww[1] else drum.gww[2]
	d.model = drum.new(poly, M)
	local lambdas, modes = drum.modes(d.model, MODES, { tol = 1e-10, yield = function()
		task.wait()
	end })
	d.lambdas = lambdas
	d.omegas, d.nodal, d.peak, d.q, d.v = {}, {}, {}, {}, {}
	local unknown = d.model.unknown
	for j = 1, MODES do
		-- shown frequency: the fundamental at 0.7 Hz
		d.omegas[j] = 2 * math.pi * 0.7 * math.sqrt(lambdas[j] / (4 * DRISCOLL[1]))
		local values = table.create(#unknown, 0)
		local peak = 0
		for node, u in unknown do
			if u > 0 then
				values[node] = f64.get(modes[j], u)
				peak = math.max(peak, math.abs(values[node]))
			end
		end
		d.nodal[j], d.peak[j], d.q[j], d.v[j] = values, peak, 0, 0
	end
end

-- Meshes: one vertex per node, at (SCALE (x - cx), height, -SCALE (y - cy)) from the
-- drum's origin. A mesh part is centered on its mesh's bounding box, so the mesh is built
-- centered: the drum's box around the origin, heights symmetric about it.
local function height(d: Drum, node: number): number
	local h = 0
	for j = 1, MODES do
		local q = d.q[j]
		if q ~= 0 then
			h += q * d.nodal[j][node]
		end
	end
	return h
end
for k, d in drums do
	local mesh = AssetService:CreateEditableMesh()
	assert(mesh, "EditableMesh unavailable (enable Mesh / Image APIs in the experience settings)")
	local x, y = d.model.x, d.model.y
	local xmin: number, xmax: number, ymin: number, ymax: number = math.huge, -math.huge, math.huge, -math.huge
	for node = 1, #x do
		xmin, xmax = math.min(xmin, x[node]), math.max(xmax, x[node])
		ymin, ymax = math.min(ymin, y[node]), math.max(ymax, y[node])
	end
	d.cx, d.cy = (xmin + xmax) / 2, (ymin + ymax) / 2
	local ids = table.create(#x, 0)
	for node = 1, #x do
		ids[node] = mesh:AddVertex(Vector3.new(SCALE * (x[node] - d.cx), 0, -SCALE * (y[node] - d.cy)))
	end
	local T = d.model.triangles
	for t = 0, #T // 3 - 1 do
		mesh:AddTriangle(ids[T[3 * t + 1]], ids[T[3 * t + 3]], ids[T[3 * t + 2]])
	end
	-- a part made from a mesh with extent in every axis, so mesh units are studs (two
	-- boundary nodes lifted for the moment, symmetrically)
	local last = #x
	mesh:SetPosition(ids[1], Vector3.new(SCALE * (x[1] - d.cx), 2 * HEIGHT, -SCALE * (y[1] - d.cy)))
	mesh:SetPosition(ids[last], Vector3.new(SCALE * (x[last] - d.cx), -2 * HEIGHT, -SCALE * (y[last] - d.cy)))
	local part = AssetService:CreateMeshPartAsync(Content.fromObject(mesh))
	mesh:SetPosition(ids[1], Vector3.new(SCALE * (x[1] - d.cx), 0, -SCALE * (y[1] - d.cy)))
	mesh:SetPosition(ids[last], Vector3.new(SCALE * (x[last] - d.cx), 0, -SCALE * (y[last] - d.cy)))
	part.Anchored = true
	part.DoubleSided = true
	part.Color = if k == 1 then Color3.fromRGB(70, 140, 220) else Color3.fromRGB(220, 150, 60)
	part.Material = Enum.Material.SmoothPlastic
	part.CFrame = CFrame.new(d.origin)
	part.Name = d.name
	part.Parent = workspace
	d.mesh, d.ids, d.part = mesh, ids, part
end

game:GetService("StarterGui"):SetCoreGuiEnabled(Enum.CoreGuiType.All, false)

-- Readout: both drums' eigenvalues and Driscoll's (scaled to this size).
title.Text = "Can you hear the shape of a drum? No: these two have the same frequencies."
label("Click a drum to strike it, or play one mode on both.", 13)
label(string.format("%-5s %-11s %-11s %-11s", "mode", "drum A", "drum B", "exact"), 13, true)
local agree = 0
for j = 1, MODES do
	local a, b = drums[1].lambdas[j], drums[2].lambdas[j]
	agree = math.max(agree, math.abs(a - b) / a)
	label(string.format("%-5d %-11.6f %-11.6f %-11.6f", j, a, b, 4 * DRISCOLL[j]), 13, true)
end
label(string.format("Eigenvalues; the drums agree to %.0e (finite elements, %d unknowns each).", agree, drums[1].model.n), 12)

local buttons = Instance.new("Frame")
buttons.Name = "Buttons"
order += 1
buttons.LayoutOrder = order
buttons.Size = UDim2.new(1, -20, 0, 26)
buttons.BackgroundTransparency = 1
buttons.Parent = panel
local row = Instance.new("UIListLayout")
row.SortOrder = Enum.SortOrder.LayoutOrder
row.FillDirection = Enum.FillDirection.Horizontal
row.Padding = UDim.new(0, 4)
row.Parent = buttons
local function button(text: string, width: number, action: () -> ())
	local b = Instance.new("TextButton")
	b.Name = text
	order += 1
	b.LayoutOrder = order
	b.Size = UDim2.fromOffset(width, 24)
	b.BackgroundColor3 = Color3.fromRGB(60, 60, 75)
	b.TextColor3 = Color3.new(1, 1, 1)
	b.Font = Enum.Font.GothamBold
	b.TextSize = 13
	b.Text = text
	b.Parent = buttons
	b.Activated:Connect(action)
end
local function calm()
	for _, d in drums do
		for j = 1, MODES do
			d.q[j], d.v[j] = 0, 0
		end
	end
end
for j = 1, MODES do
	button(tostring(j), 28, function()
		calm()
		for _, d in drums do
			d.q[j] = HEIGHT / d.peak[j]
		end
	end)
end
button("Calm", 50, calm)

-- A strike at drum point (px, py): a unit velocity impulse there, so mode j gains
-- velocity proportional to its value at the point. Scaled so the modes' peaks would add to
-- twice HEIGHT if they all lined up (they rarely do).
local function strike(d: Drum, px: number, py: number)
	local total = 0
	local dv = {}
	for j = 1, MODES do
		dv[j] = drum.interpolate(d.model, d.nodal[j], px, py)
		total += math.abs(dv[j]) / d.omegas[j] * d.peak[j]
	end
	if total == 0 then
		return
	end
	for j = 1, MODES do
		d.v[j] += 2 * HEIGHT * dv[j] / total
	end
end
UserInputService.InputBegan:Connect(function(input: InputObject, processed: boolean)
	if processed then
		return
	end
	if input.UserInputType ~= Enum.UserInputType.MouseButton1 and input.UserInputType ~= Enum.UserInputType.Touch then
		return
	end
	local ray = camera:ViewportPointToRay(input.Position.X, input.Position.Y)
	local hit = workspace:Raycast(ray.Origin, ray.Direction * 500)
	if not hit then
		return
	end
	for _, d in drums do
		if hit.Instance == d.part then
			local p = hit.Position - d.origin
			strike(d, d.cx + p.X / SCALE, d.cy - p.Z / SCALE)
		end
	end
end)

-- Each mode evolves exactly: a damped rotation of (q, v / omega) by omega dt.
RunService.Heartbeat:Connect(function(dt: number)
	dt = math.min(dt, 0.1)
	local damp = math.exp(-DECAY * dt)
	for _, d in drums do
		for j = 1, MODES do
			local w = d.omegas[j]
			local c, s = math.cos(w * dt), math.sin(w * dt)
			local q, v = d.q[j], d.v[j]
			d.q[j] = damp * (c * q + s * v / w)
			d.v[j] = damp * (-w * s * q + c * v)
		end
		local mesh, ids, x, y = d.mesh, d.ids, d.model.x, d.model.y
		for node = 1, #ids do
			mesh:SetPosition(ids[node], Vector3.new(SCALE * (x[node] - d.cx), height(d, node), -SCALE * (y[node] - d.cy)))
		end
	end
end)
