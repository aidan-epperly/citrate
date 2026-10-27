--!strict
-- Reach demo (LocalScript): two arms reaching for the same moving target (each its own
-- copy). Left: CCD, the usual approach, 16 sweeps a frame. Right: citrate's least squares
-- (Levenberg-Marquardt) on the hand's distance plus a pull toward a rest pose, from last
-- frame's angles. The panels show each one's time per frame, its hand's distance from the
-- target and its largest angle change in a frame (twitching).

local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local Citrate = ReplicatedStorage:WaitForChild("Citrate")
local reach = require(Citrate:WaitForChild("demos"):WaitForChild("reach"):WaitForChild("reach")) :: any

local SWEEPS = 16
local folder = Instance.new("Folder")
folder.Name = "ReachDemo"
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

type Arm = { origin: Vector3, segments: { Part }, joints: { Part }, target: Part, ms: number, dist: number, jump: number, prev: { number }? }
local function arm(origin: Vector3, color: Color3): Arm
	local base = part(Vector3.new(2.4, 0.6, 2.4), Color3.fromRGB(70, 70, 80), Enum.PartType.Cylinder)
	base.CFrame = CFrame.new(origin + Vector3.new(0, 0.3, 0)) * CFrame.Angles(0, 0, math.pi / 2)
	local a: Arm = { origin = origin, segments = {}, joints = {}, target = part(Vector3.new(0.9, 0.9, 0.9), Color3.fromRGB(255, 220, 80), Enum.PartType.Ball), ms = 0, dist = 0, jump = 0, prev = nil }
	for j = 1, #reach.LENGTHS do
		local w = 0.55 - 0.06 * j
		a.segments[j] = part(Vector3.new(w, w, reach.LENGTHS[j]), color)
		a.joints[j] = part(Vector3.new(w + 0.2, w + 0.2, w + 0.2), Color3.fromRGB(230, 230, 235), Enum.PartType.Ball)
	end
	return a
end
local LEFT, RIGHT = Vector3.new(-8, 0, 0), Vector3.new(8, 0, 0)
local ccd_arm = arm(LEFT, Color3.fromRGB(230, 140, 70))
local lm_arm = arm(RIGHT, Color3.fromRGB(80, 160, 240))

local theta_ccd = table.clone(reach.REST)
local solve = reach.citrate_solver("lm")
local phi = reach.phi_of(reach.REST)

-- Panel
local gui = Instance.new("ScreenGui")
gui.Name, gui.ResetOnSpawn, gui.IgnoreGuiInset = "ReachDemo", false, true
local function label(anchor: Vector2, position: UDim2, size: Vector2): TextLabel
	local l = Instance.new("TextLabel")
	l.AnchorPoint, l.Position, l.Size = anchor, position, UDim2.fromOffset(size.X, size.Y)
	l.BackgroundColor3, l.BackgroundTransparency = Color3.new(0, 0, 0), 0.45
	l.TextColor3, l.Font, l.TextSize = Color3.new(1, 1, 1), Enum.Font.GothamMedium, 15
	l.TextWrapped = true
	l.Parent = gui
	return l
end
local left_label = label(Vector2.new(0, 1), UDim2.new(0, 16, 1, -16), Vector2.new(400, 74))
local right_label = label(Vector2.new(1, 1), UDim2.new(1, -16, 1, -16), Vector2.new(400, 74))
game:GetService("StarterGui"):SetCoreGuiEnabled(Enum.CoreGuiType.All, false)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")
local camera = workspace.CurrentCamera
camera.CameraType = Enum.CameraType.Scriptable
camera.CFrame = CFrame.lookAt(Vector3.new(0, 9, 19), Vector3.new(0, 4.5, 2))

local function draw(a: Arm, theta: { number }, target: Vector3, parts: { BasePart }, cfs: { CFrame })
	local pts = reach.forward(theta)
	for j = 1, #reach.LENGTHS do
		local p, q = a.origin + Vector3.new(pts[j].x, pts[j].y, pts[j].z), a.origin + Vector3.new(pts[j + 1].x, pts[j + 1].y, pts[j + 1].z)
		table.insert(parts, a.segments[j])
		table.insert(cfs, CFrame.lookAt((p + q) / 2, q))
		table.insert(parts, a.joints[j])
		table.insert(cfs, CFrame.new(p))
	end
	table.insert(parts, a.target)
	table.insert(cfs, CFrame.new(a.origin + target))
	local hand = pts[#pts]
	local d = (Vector3.new(hand.x, hand.y, hand.z) - target).Magnitude
	a.dist = 0.9 * a.dist + 0.1 * d
	local jump = 0
	if a.prev then
		for i = 1, reach.N do
			jump = math.max(jump, math.abs(theta[i] - (a.prev :: { number })[i]))
		end
	end
	a.jump = math.max(0.98 * a.jump, jump)
	a.prev = table.clone(theta)
end

local clock = 0
RunService.Heartbeat:Connect(function(dt: number)
	clock += dt
	local tv = reach.target(clock)
	local target = Vector3.new(tv.x, tv.y, tv.z)
	local t0 = os.clock()
	reach.ccd(theta_ccd, tv, SWEEPS)
	local t1 = os.clock()
	phi = solve(tv, phi)
	local t2 = os.clock()
	ccd_arm.ms = 0.95 * ccd_arm.ms + 0.05 * (t1 - t0) * 1000
	lm_arm.ms = 0.95 * lm_arm.ms + 0.05 * (t2 - t1) * 1000
	local parts, cfs = {}, {}
	draw(ccd_arm, theta_ccd, target, parts, cfs)
	draw(lm_arm, reach.theta_of(phi), target, parts, cfs)
	workspace:BulkMoveTo(parts, cfs, Enum.BulkMoveMode.FireCFrameChanged)
	left_label.Text = string.format("CCD, %d sweeps a frame\n%.3f ms a frame   distance %.3f studs\nlargest angle change a frame: %.2f rad", SWEEPS, ccd_arm.ms, ccd_arm.dist, ccd_arm.jump)
	right_label.Text = string.format("citrate least squares (Levenberg-Marquardt)\n%.3f ms a frame   distance %.3f studs\nlargest angle change a frame: %.2f rad", lm_arm.ms, lm_arm.dist, lm_arm.jump)
end)
