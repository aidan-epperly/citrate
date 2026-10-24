--!strict

-- Minimal test harness. `finish` raises an error when any check failed, so the
-- process exits with a nonzero status.

local current = "?"
local passed, failed = 0, 0

local function test(name: string, body: () -> ())
	current = name
	local ok, err = pcall(body)
	if not ok then
		failed += 1
		print(`FAIL {name}: error: {err}`)
	end
end

local function check(cond: boolean, message: string?)
	if cond then
		passed += 1
	else
		failed += 1
		print(`FAIL {current}: {message or "check failed"}`)
	end
end

-- |actual - expected| <= tol * max(1, |expected|)
local function close(actual: number, expected: number, tol: number, message: string?)
	local ok = math.abs(actual - expected) <= tol * math.max(1, math.abs(expected))
	check(ok, `{message or "value"}: got {actual}, expected {expected} (tol {tol})`)
end

-- True if every value stored in b (f32 values if single, else f64) is NaN.
local function all_nan(b: buffer, single: boolean): boolean
	local w = if single then 4 else 8
	for k = 0, buffer.len(b) - w, w do
		local v = if single then buffer.readf32(b, k) else buffer.readf64(b, k)
		if v == v then
			return false
		end
	end
	return true
end

local function finish()
	print(`{passed} checks passed, {failed} failed`)
	if failed > 0 then
		error(`{failed} test failures`, 0)
	end
end

-- Whether the platform keeps subnormal numbers in double and single precision (buffers
-- and vectors): Roblox servers flush both to zero, and Roblox clients flush the single
-- precision ones, so checks of behavior at subnormal magnitudes only apply elsewhere.
local subnormals = (function(): boolean
	local b = buffer.create(4)
	buffer.writef32(b, 0, math.ldexp(1, -140))
	return math.ldexp(1, -1060) > 0 and buffer.readf32(b, 0) > 0 and vector.create(math.ldexp(1, -140), 0, 0).x > 0
end)()

return {
	subnormals = subnormals,
	test = test,
	check = check,
	close = close,
	all_nan = all_nan,
	finish = finish,
	eps32 = 2 ^ -23,
	eps64 = 2 ^ -52,
}
