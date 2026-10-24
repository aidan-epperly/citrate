--!strict
--!optimize 2

-- Single-precision complex scalars stored in a vector (x = re, y = im, z = 0).
-- Addition, subtraction and real scaling are native single-precision vector operations.
-- Everything else is computed in double precision and rounded once. Squares of
-- single-precision values cannot overflow or underflow a double, so the plain formulas
-- are safe here (complex64 needs scaled versions).

local function is_negative(v: number): boolean
	return v < 0 or (v == 0 and 1 / v < 0)
end

-- The complex number re + i im as a vector.
local function create(re: number, im: number): vector
	return vector.create(re, im, 0)
end

-- The real part.
local function real(z: vector): number
	return z.x
end

-- The imaginary part.
local function imag(z: vector): number
	return z.y
end

-- a + b.
local function add(a: vector, b: vector): vector
	return a + b
end

-- a - b.
local function sub(a: vector, b: vector): vector
	return a - b
end

-- s a for a real s.
local function scale(a: vector, s: number): vector
	return a * s
end

-- a b.
local function mul(a: vector, b: vector): vector
	local ar, ai, br, bi = a.x, a.y, b.x, b.y
	return vector.create(ar * br - ai * bi, ar * bi + ai * br, 0)
end

-- a / b.
local function div(a: vector, b: vector): vector
	local ar, ai, br, bi = a.x, a.y, b.x, b.y
	local d = br * br + bi * bi
	return vector.create((ar * br + ai * bi) / d, (ai * br - ar * bi) / d, 0)
end

-- 1 / z.
local function inv(z: vector): vector
	local re, im = z.x, z.y
	local d = re * re + im * im
	return vector.create(re / d, -im / d, 0)
end

-- The conjugate.
local function conj(z: vector): vector
	return vector.create(z.x, -z.y, 0)
end

-- The modulus |z|, without overflow.
local function abs(z: vector): number
	local re, im = z.x, z.y
	return math.sqrt(re * re + im * im)
end

-- The argument (angle) of z, in (-pi, pi].
local function arg(z: vector): number
	return math.atan2(z.y, z.x)
end

-- The complex number with modulus r and argument theta.
local function polar(r: number, theta: number): vector
	return vector.create(r * math.cos(theta), r * math.sin(theta), 0)
end

-- e^z.
local function exp(z: vector): vector
	local e, im = math.exp(z.x), z.y
	return vector.create(e * math.cos(im), e * math.sin(im), 0)
end

-- Principal branch.
local function log(z: vector): vector
	local re, im = z.x, z.y
	return vector.create(0.5 * math.log(re * re + im * im), math.atan2(im, re), 0)
end

-- Principal branch, without cancellation in either component.
local function sqrt(z: vector): vector
	local re, im = z.x, z.y
	if re == 0 and im == 0 then
		return vector.create(0, im, 0)
	end
	local t = math.sqrt((math.abs(re) + math.sqrt(re * re + im * im)) / 2)
	if re >= 0 then
		return vector.create(t, im / (2 * t), 0)
	end
	return vector.create(math.abs(im) / (2 * t), if is_negative(im) then -t else t, 0)
end

-- Principal branch of z^w = exp(w log z), with the intermediate values in double precision.
local function pow(z: vector, w: vector): vector
	local zr, zi, wr, wi = z.x, z.y, w.x, w.y
	if zr == 0 and zi == 0 then
		if wr == 0 and wi == 0 then
			return vector.create(1, 0, 0)
		elseif wr > 0 then
			return vector.create(0, 0, 0)
		end
		return vector.create(0 / 0, 0 / 0, 0)
	end
	local lr, li = 0.5 * math.log(zr * zr + zi * zi), math.atan2(zi, zr)
	local er, ei = wr * lr - wi * li, wr * li + wi * lr
	local e = math.exp(er)
	return vector.create(e * math.cos(ei), e * math.sin(ei), 0)
end

-- z as a string, such as "1 - 2i".
local function tostring_(z: vector): string
	local re, im = z.x, z.y
	if is_negative(im) then
		return `{re} - {-im}i`
	end
	return `{re} + {im}i`
end

return {
	create = create,
	real = real,
	imag = imag,

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
