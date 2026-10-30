--!strict
--!nolint DeprecatedApi

-- What the host provides that the library can use when present: Roblox's CFrame and
-- Vector3 types (nil elsewhere). Kept in its own module because reading the environment (getfenv) turns off
-- some of the compiler's optimizations for the script that does it.

local env: any = getfenv(0)

return {
	CFrame = env.CFrame :: any,
	Vector3 = env.Vector3 :: any,
}
