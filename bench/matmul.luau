--!strict
-- Matrix products: matmul for every array module (luau -O2 --codegen bench/matmul.luau).
local lib = require("../src")
for _, entry in { { "f64", lib.f64 }, { "f32", lib.f32 }, { "c64", lib.c64 }, { "c32", lib.c32 } } :: { { any } } do
	local name, M = entry[1], entry[2]
	local line = string.format("%-4s", name)
	for _, n in { 64, 128, 256 } do
		local A, B = M.matrix(n, n), M.matrix(n, n)
		local reps = math.max(1, math.floor(2e7 / n ^ 3))
		local best = math.huge
		for _ = 1, 3 do
			local t0 = os.clock()
			for _ = 1, reps do
				M.matmul(nil, A, B)
			end
			best = math.min(best, (os.clock() - t0) / reps * 1000)
		end
		line ..= string.format("   n %3d: %8.2f ms", n, best)
	end
	print(line)
end
