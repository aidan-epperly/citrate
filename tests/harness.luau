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

local function finish()
	print(`{passed} checks passed, {failed} failed`)
	if failed > 0 then
		error(`{failed} test failures`, 0)
	end
end

return {
	test = test,
	check = check,
	close = close,
	finish = finish,
	eps32 = 2 ^ -23,
	eps64 = 2 ^ -52,
}
