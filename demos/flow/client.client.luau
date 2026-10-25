--!strict
--!native
-- Flow demo (LocalScript): incompressible Navier-Stokes past an obstacle, solved every
-- frame (flow.luau: MacCormack advection, and diffusion and projection by the library's
-- FFT). Drag the obstacle; past a Reynolds number of about 47 a cylinder sheds a Karman
-- vortex street.

local AssetService = game:GetService("AssetService")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local UserInputService = game:GetService("UserInputService")

local Citrate = ReplicatedStorage:WaitForChild("Citrate")
local flow = require(Citrate:WaitForChild("demos"):WaitForChild("flow"):WaitForChild("flow")) :: any

local NX, NY = 192, 96
local PIXEL = 5 -- screen pixels per cell
local SPEED = 67.5 -- cells per second
local DT = 1 / 60
local REYNOLDS = { 20, 50, 100, 200, 500 }
local SIZES: { [string]: number } = { cylinder = 12, plate = 12, airfoil = 30 }

local re_index = 3
local shape = "cylinder"
local ox, oy = NX / 4, NY / 2
local view = "dye"

local sim = flow.new({ nx = NX, ny = NY, speed = SPEED, reynolds = REYNOLDS[re_index] })
sim:set_obstacle(shape, ox, oy, SIZES[shape])
-- The fringe (the last columns, where the flow is forced back to the inflow) is a
-- numerical device, so it is not shown.
local W = NX - sim.fringe

-- Display
local image = AssetService:CreateEditableImage({ Size = Vector2.new(W, NY) })
assert(image, "EditableImage unavailable (enable Mesh / Image APIs in the experience settings)")
local pixels = buffer.create(4 * W * NY)

local gui = Instance.new("ScreenGui")
gui.Name = "FlowDemo"
gui.ResetOnSpawn = false
gui.IgnoreGuiInset = true
local frame = Instance.new("ImageLabel")
frame.Name = "Flow"
frame.Size = UDim2.fromOffset(W * PIXEL, NY * PIXEL)
frame.AnchorPoint = Vector2.new(0.5, 0)
frame.Position = UDim2.new(0.5, 0, 0, 24)
frame.BackgroundColor3 = Color3.new(0, 0, 0)
frame.ImageContent = Content.fromObject(image)
frame.Parent = gui

local panel = Instance.new("Frame")
panel.Name = "Panel"
panel.AnchorPoint = Vector2.new(0.5, 0)
panel.Size = UDim2.fromOffset(W * PIXEL, 70)
panel.Position = UDim2.new(0.5, 0, 0, 24 + NY * PIXEL + 8)
panel.BackgroundTransparency = 1
panel.Parent = gui
local layout = Instance.new("UIListLayout")
layout.SortOrder = Enum.SortOrder.LayoutOrder
layout.Padding = UDim.new(0, 6)
layout.Parent = panel
local status = Instance.new("TextLabel")
status.LayoutOrder = 1
status.Size = UDim2.new(1, 0, 0, 20)
status.BackgroundTransparency = 1
status.TextColor3 = Color3.new(1, 1, 1)
status.TextXAlignment = Enum.TextXAlignment.Left
status.Font = Enum.Font.GothamMedium
status.TextSize = 15
status.Parent = panel
local buttons = Instance.new("Frame")
buttons.Name = "Buttons"
buttons.LayoutOrder = 2
buttons.Size = UDim2.new(1, 0, 0, 30)
buttons.BackgroundTransparency = 1
buttons.Parent = panel
local row = Instance.new("UIListLayout")
row.SortOrder = Enum.SortOrder.LayoutOrder
row.FillDirection = Enum.FillDirection.Horizontal
row.Padding = UDim.new(0, 6)
row.Parent = buttons
local order = 0
local function button(name: string, text: string, width: number, action: (TextButton) -> ()): TextButton
	local b = Instance.new("TextButton")
	b.Name = name
	order += 1
	b.LayoutOrder = order
	b.Size = UDim2.fromOffset(width, 28)
	b.BackgroundColor3 = Color3.fromRGB(60, 60, 75)
	b.TextColor3 = Color3.new(1, 1, 1)
	b.Font = Enum.Font.GothamBold
	b.TextSize = 14
	b.Text = text
	b.Parent = buttons
	b.Activated:Connect(function()
		action(b)
	end)
	return b
end
for _, s in { "cylinder", "plate", "airfoil" } do
	button(s, s:sub(1, 1):upper() .. s:sub(2), 80, function()
		shape = s
		sim:set_obstacle(shape, ox, oy, SIZES[shape])
	end)
end
button("View", "Vorticity", 90, function(b: TextButton)
	view = if view == "dye" then "vorticity" else "dye"
	b.Text = if view == "dye" then "Vorticity" else "Dye"
end)
button("Slower", "Re -", 56, function()
	re_index = math.max(1, re_index - 1)
	sim:set_reynolds(REYNOLDS[re_index])
end)
button("Faster", "Re +", 56, function()
	re_index = math.min(#REYNOLDS, re_index + 1)
	sim:set_reynolds(REYNOLDS[re_index])
end)
button("Reset", "Reset", 70, function()
	sim = flow.new({ nx = NX, ny = NY, speed = SPEED, reynolds = REYNOLDS[re_index] })
	sim:set_obstacle(shape, ox, oy, SIZES[shape])
end)

game:GetService("StarterGui"):SetCoreGuiEnabled(Enum.CoreGuiType.All, false)
gui.Parent = Players.LocalPlayer:WaitForChild("PlayerGui")

-- Dragging: while the button is held on the image, the obstacle follows the pointer.
local dragging = false
local function move_to(position: Vector3 | Vector2)
	local rel = Vector2.new(position.X, position.Y) - frame.AbsolutePosition
	local x, y = rel.X / PIXEL, NY - rel.Y / PIXEL
	local half = SIZES[shape] / 2 + 2
	ox = math.clamp(x, half, W - half)
	oy = math.clamp(y, half, NY - half)
	sim:set_obstacle(shape, ox, oy, SIZES[shape])
end
frame.InputBegan:Connect(function(input: InputObject)
	if input.UserInputType == Enum.UserInputType.MouseButton1 or input.UserInputType == Enum.UserInputType.Touch then
		dragging = true
		move_to(input.Position)
	end
end)
UserInputService.InputChanged:Connect(function(input: InputObject)
	if dragging and (input.UserInputType == Enum.UserInputType.MouseMovement or input.UserInputType == Enum.UserInputType.Touch) then
		move_to(input.Position)
	end
end)
UserInputService.InputEnded:Connect(function(input: InputObject)
	if input.UserInputType == Enum.UserInputType.MouseButton1 or input.UserInputType == Enum.UserInputType.Touch then
		dragging = false
	end
end)

-- Colors: dye white on deep blue; vorticity red (counterclockwise) and blue (clockwise)
-- on white; solid gray.
local vort = buffer.create(8 * NX * NY)
local function draw()
	local solid = sim.solid
	local field: buffer
	local scale: number
	if view == "dye" then
		field, scale = sim.dye, 1
	else
		field = sim:vorticity(vort)
		scale = 1 / (0.25 * SPEED / SIZES.cylinder) -- color saturates at |w| = 4 U / D
	end
	for j = 0, W - 1 do
		for i = 0, NY - 1 do
			local k = j * NY + i
			local p = 4 * ((NY - 1 - i) * W + j)
			local r, g, b
			if buffer.readu8(solid, k) == 1 then
				r, g, b = 150, 150, 160
			elseif view == "dye" then
				local d = math.clamp(buffer.readf64(field, 8 * k), 0, 1)
				r, g, b = 20 + 235 * d, 40 + 205 * d, 90 + 140 * d
			else
				local w = math.clamp(buffer.readf64(field, 8 * k) * scale, -1, 1)
				if w > 0 then
					r, g, b = 255, 255 * (1 - w), 255 * (1 - w)
				else
					r, g, b = 255 * (1 + w), 255 * (1 + w), 255
				end
			end
			buffer.writeu8(pixels, p, r)
			buffer.writeu8(pixels, p + 1, g)
			buffer.writeu8(pixels, p + 2, b)
			buffer.writeu8(pixels, p + 3, 255)
		end
	end
	image:WritePixelsBuffer(Vector2.zero, Vector2.new(W, NY), pixels)
end

local step_ms, draw_ms = 0, 0
RunService.Heartbeat:Connect(function()
	local t0 = os.clock()
	sim:step(DT)
	local t1 = os.clock()
	draw()
	local t2 = os.clock()
	step_ms = 0.95 * step_ms + 0.05 * (t1 - t0) * 1000
	draw_ms = 0.95 * draw_ms + 0.05 * (t2 - t1) * 1000
	status.Text = string.format("Re %d (%s, %dx%d)    solver %.1f ms/frame, drawing %.1f ms    max |div u| %.0e    drag the %s",
		REYNOLDS[re_index], shape, NX, NY, step_ms, draw_ms, sim:divergence(), shape)
end)
