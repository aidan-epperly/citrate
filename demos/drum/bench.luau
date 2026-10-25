--!strict
-- The Gordon-Webb-Wolpert drums outside Roblox (luau -O2 --codegen demos/drum/bench.luau):
-- the two discrete drums' eigenvalues agree to rounding at every resolution, while
-- against Driscoll's 12-digit values (SIAM Review 1997, for drums twice this size) the
-- error falls with the mesh.
local drum = require("./drum")
local driscoll = { 2.53794399980, 3.65550971352, 5.17555935622, 6.53755744376, 7.24807786256, 9.20929499840, 10.5969856913, 11.5413953956, 12.3370055014, 13.0536540557 }
for _, m in { 2, 4, 8, 16, 32 } do
	local A, B = drum.new(drum.gww[1], m), drum.new(drum.gww[2], m)
	local t0 = os.clock()
	local la = drum.modes(A, 10, { tol = 1e-12 })
	local ms = (os.clock() - t0) * 1000
	local lb = drum.modes(B, 10, { tol = 1e-12 })
	local diff, worst = 0, 0
	for i = 1, 10 do
		diff = math.max(diff, math.abs(la[i] - lb[i]) / la[i])
		worst = math.max(worst, math.abs(la[i] / 4 - driscoll[i]) / driscoll[i])
	end
	print(string.format("m = %2d: %5d unknowns, %6.0f ms; drums A and B differ by %.0e; largest error against Driscoll %.1e", m, A.n, ms, diff, worst))
end
