--!strict
--!native
--!optimize 2

-- Double-precision complex scalars, passed and returned as (re, im) pairs of numbers.

-- True for negative numbers and -0, which decides the side of a branch cut.
local function is_negative(v: number): boolean
	return v < 0 or (v == 0 and 1 / v < 0)
end

-- a + b.
local function add(ar: number, ai: number, br: number, bi: number): (number, number)
	return ar + br, ai + bi
end

-- a - b.
local function sub(ar: number, ai: number, br: number, bi: number): (number, number)
	return ar - br, ai - bi
end

-- a b.
local function mul(ar: number, ai: number, br: number, bi: number): (number, number)
	return ar * br - ai * bi, ar * bi + ai * br
end

-- s a for a real s.
local function scale(ar: number, ai: number, s: number): (number, number)
	return ar * s, ai * s
end

-- The conjugate.
local function conj(re: number, im: number): (number, number)
	return re, -im
end

-- Smith's algorithm: avoids overflow in br^2 + bi^2.
local function div(ar: number, ai: number, br: number, bi: number): (number, number)
	if math.abs(br) >= math.abs(bi) then
		local r = bi / br
		local d = br + bi * r
		return (ar + ai * r) / d, (ai - ar * r) / d
	else
		local r = br / bi
		local d = br * r + bi
		return (ar * r + ai) / d, (ai * r - ar) / d
	end
end

-- 1 / z.
local function inv(re: number, im: number): (number, number)
	return div(1, 0, re, im)
end

-- |re + i im| without overflow or underflow in the squares.
local function abs(re: number, im: number): number
	local a, b = math.abs(re), math.abs(im)
	if a < b then
		a, b = b, a
	end
	if a == 0 or a == math.huge then
		return a
	end
	local r = b / a
	return a * math.sqrt(1 + r * r)
end

-- The argument (angle) of z, in (-pi, pi].
local function arg(re: number, im: number): number
	return math.atan2(im, re)
end

-- The complex number with modulus r and argument theta.
local function polar(r: number, theta: number): (number, number)
	return r * math.cos(theta), r * math.sin(theta)
end

-- e^z.
local function exp(re: number, im: number): (number, number)
	local e = math.exp(re)
	return e * math.cos(im), e * math.sin(im)
end

-- Principal branch.
local function log(re: number, im: number): (number, number)
	return math.log(abs(re, im)), math.atan2(im, re)
end

-- Principal branch, without cancellation in either component.
local function sqrt(re: number, im: number): (number, number)
	if re == 0 and im == 0 then
		return 0, im
	end
	local t = math.sqrt((math.abs(re) + abs(re, im)) / 2)
	if re >= 0 then
		return t, im / (2 * t)
	end
	return math.abs(im) / (2 * t), if is_negative(im) then -t else t
end

-- Principal branch of z^w = exp(w log z).
local function pow(zr: number, zi: number, wr: number, wi: number): (number, number)
	if zr == 0 and zi == 0 then
		if wr == 0 and wi == 0 then
			return 1, 0
		elseif wr > 0 then
			return 0, 0
		end
		return 0 / 0, 0 / 0
	end
	local lr, li = log(zr, zi)
	return exp(mul(wr, wi, lr, li))
end

-- z as a string, such as "1 - 2i".
local function tostring_(re: number, im: number): string
	if is_negative(im) then
		return `{re} - {-im}i`
	end
	return `{re} + {im}i`
end

return {
	add = add,
	sub = sub,
	mul = mul,
	div = div,
	scale = scale,
	inv = inv,
	conj = conj,

	abs = abs,
	arg = arg,
	polar = polar,

	exp = exp,
	log = log,
	sqrt = sqrt,
	pow = pow,

	tostring = tostring_,
}
