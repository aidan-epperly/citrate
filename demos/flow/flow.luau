--!strict
--!native
--!optimize 2
-- Incompressible 2D Navier-Stokes past an obstacle, in real time: a projection method on a
-- periodic nx by ny staggered (MAC) grid with unit spacing.
--
-- Each step (dt):
-- 1. Advect the velocity and the dye with MacCormack's scheme (a semi-Lagrangian step back,
--    one forward, and half the round trip's error corrected, clamped to the values it
--    interpolated; Selle et al. 2008), or plain first-order semi-Lagrangian.
-- 2. Fringe: in the last columns, relax the flow to the uniform inflow and reset the dye
--    to streaks, so what wraps around the periodic box enters as fresh inflow.
-- 3. Obstacle: zero the velocity on faces inside the solid (Brinkman penalization in the
--    limit), so the boundary is accurate to first order in the grid spacing.
-- 4. Diffuse and project in Fourier space, with one FFT pair: u and v are real, so
--    z = u + i v carries both; diffusion multiplies each mode by exp(nu lambda dt) (exact
--    for the 5-point Laplacian, any time step), and the projection subtracts the gradient
--    of p solving L p = div u, using the exact symbols of the staggered differences, so
--    the discrete divergence is zero to rounding.
--
-- u[j, i] lives at (x, y) = (j, i + 1/2), v[j, i] at (j + 1/2, i), the dye and p at cell
-- centers (j + 1/2, i + 1/2), for 0 <= j < nx, 0 <= i < ny; arrays are stored with y
-- contiguous (index j ny + i), as the library's column-major ny by nx matrices are.

local citrate = require("../../src")
local c64 = citrate.c64

local flow = {}

export type Params = {
	nx: number,
	ny: number,
	speed: number, -- inflow speed, cells per second
	reynolds: number, -- U D / nu, D the obstacle's height
	advection: string?, -- "maccormack" (default) or "semilagrangian"
	fringe: number?, -- columns (default nx / 10)
}

export type Flow = {
	nx: number,
	ny: number,
	fringe: number, -- the last columns, where the flow is forced back to the inflow
	speed: number,
	nu: number,
	u: buffer,
	v: buffer,
	dye: buffer,
	solid: buffer, -- 1 in solid cells (u8)
	diameter: number,
	t: number,
	step: (self: Flow, dt: number) -> (),
	set_obstacle: (self: Flow, shape: string, cx: number, cy: number, size: number) -> (),
	set_reynolds: (self: Flow, re: number) -> (),
	vorticity: (self: Flow, out: buffer?) -> buffer,
	divergence: (self: Flow) -> number,
}

function flow.new(p: Params): Flow
	local nx, ny = p.nx, p.ny
	local n = nx * ny
	local maccormack = (p.advection or "maccormack") == "maccormack"
	local F = p.fringe or math.floor(nx / 10)
	local self = {
		nx = nx,
		ny = ny,
		fringe = F,
		speed = p.speed,
		nu = 0,
		u = buffer.create(8 * n),
		v = buffer.create(8 * n),
		dye = buffer.create(8 * n),
		solid = buffer.create(n),
		diameter = 1,
		t = 0,
	} :: any
	local reynolds = p.reynolds

	-- scratch
	local u1, v1, d1 = buffer.create(8 * n), buffer.create(8 * n), buffer.create(8 * n)
	local u2, v2, d2 = buffer.create(8 * n), buffer.create(8 * n), buffer.create(8 * n)
	local Z = c64.matrix(ny, nx)

	-- Fringe weights per column (smooth ramp up to 1) and the dye's inflow streaks.
	local fringe = table.create(nx, 0)
	for j = nx - F, nx - 1 do
		local s = (j - (nx - F) + 0.5) / F
		fringe[j + 1] = math.sin(math.pi / 2 * math.min(1, 2 * s)) ^ 2
	end
	local function streak(i: number): number
		return if (i // 4) % 4 == 0 then 1 else 0
	end

	-- Uniform flow, plus a smooth transverse perturbation v = 0.02 U sin(2 pi x / nx), which
	-- is divergence-free: symmetric obstacles start shedding only from an asymmetry, which
	-- the discretization would otherwise keep at the rounding level.
	for j = 0, nx - 1 do
		local w = 0.02 * p.speed * math.sin(2 * math.pi * (j + 0.5) / nx)
		for i = 0, ny - 1 do
			buffer.writef64(self.u, 8 * (j * ny + i), p.speed)
			buffer.writef64(self.v, 8 * (j * ny + i), w)
		end
	end
	for j = 0, nx - 1 do
		for i = 0, ny - 1 do
			buffer.writef64(self.dye, 8 * (j * ny + i), streak(i))
		end
	end

	-- One semi-Lagrangian pass over a field at offset (ox, oy): out(x) = f(x - s dt vel(x)),
	-- with the departure point found by the midpoint rule. s = 1 traces back, s = -1
	-- forward. If lo and hi are given, record the clamp bounds there (the min and max of
	-- the four values interpolated). The bilinear interpolations are written out inline:
	-- as calls they cost three times as much (a call is about 6 ns, the work about 3 ns). A
	-- departure point is never more than one period outside the box, so one branch wraps it.
	local row = 8 * ny
	local function pass(out: buffer, f: buffer, u: buffer, v: buffer, ox: number, oy: number, s: number, dt: number, lo: buffer?, hi: buffer?)
		local h = s * dt
		local kind = if ox == 0 then 0 elseif oy == 0 then 1 else 2
		for j = 0, nx - 1 do
			for i = 0, ny - 1 do
				local x, y = j + ox, i + oy
				local k = 8 * (j * ny + i)
				-- the velocity at the grid point itself: exact for the field's own component,
				-- the average of its two or four nearest values for the other
				local jm = if j == 0 then nx - 1 else j - 1
				local jp = if j == nx - 1 then 0 else j + 1
				local im = if i == 0 then ny - 1 else i - 1
				local ip = if i == ny - 1 then 0 else i + 1
				local ux, vy
				if kind == 0 then -- u-face (j, i + 1/2): v from the cells (j - 1, j) and faces (i, i + 1)
					ux = buffer.readf64(u, k)
					vy = 0.25 * (buffer.readf64(v, row * j + 8 * i) + buffer.readf64(v, row * j + 8 * ip)
						+ buffer.readf64(v, row * jm + 8 * i) + buffer.readf64(v, row * jm + 8 * ip))
				elseif kind == 1 then -- v-face (j + 1/2, i)
					vy = buffer.readf64(v, k)
					ux = 0.25 * (buffer.readf64(u, row * j + 8 * i) + buffer.readf64(u, row * jp + 8 * i)
						+ buffer.readf64(u, row * j + 8 * im) + buffer.readf64(u, row * jp + 8 * im))
				else -- cell center (j + 1/2, i + 1/2)
					ux = 0.5 * (buffer.readf64(u, row * j + 8 * i) + buffer.readf64(u, row * jp + 8 * i))
					vy = 0.5 * (buffer.readf64(v, row * j + 8 * i) + buffer.readf64(v, row * j + 8 * ip))
				end
				local xm, ym = x - 0.5 * h * ux, y - 0.5 * h * vy
				local um
				do
					local px, py = xm, ym - 0.5
					local j0, i0 = math.floor(px), math.floor(py)
					local a, b = px - j0, py - i0
					if j0 < 0 then
						j0 += nx
					elseif j0 >= nx then
						j0 -= nx
					end
					if i0 < 0 then
						i0 += ny
					elseif i0 >= ny then
						i0 -= ny
					end
					local c0 = row * j0
					local c1 = if j0 == nx - 1 then 0 else c0 + row
					local r0 = 8 * i0
					local r1 = if i0 == ny - 1 then 0 else r0 + 8
					local f00, f01 = buffer.readf64(u, c0 + r0), buffer.readf64(u, c0 + r1)
					local f10, f11 = buffer.readf64(u, c1 + r0), buffer.readf64(u, c1 + r1)
					um = (1 - a) * ((1 - b) * f00 + b * f01) + a * ((1 - b) * f10 + b * f11)
				end
				local vm
				do
					local px, py = xm - 0.5, ym
					local j0, i0 = math.floor(px), math.floor(py)
					local a, b = px - j0, py - i0
					if j0 < 0 then
						j0 += nx
					elseif j0 >= nx then
						j0 -= nx
					end
					if i0 < 0 then
						i0 += ny
					elseif i0 >= ny then
						i0 -= ny
					end
					local c0 = row * j0
					local c1 = if j0 == nx - 1 then 0 else c0 + row
					local r0 = 8 * i0
					local r1 = if i0 == ny - 1 then 0 else r0 + 8
					local f00, f01 = buffer.readf64(v, c0 + r0), buffer.readf64(v, c0 + r1)
					local f10, f11 = buffer.readf64(v, c1 + r0), buffer.readf64(v, c1 + r1)
					vm = (1 - a) * ((1 - b) * f00 + b * f01) + a * ((1 - b) * f10 + b * f11)
				end
				local xd, yd = x - h * um - ox, y - h * vm - oy
				if lo then
					local value
					do
						local px, py = xd, yd
						local j0, i0 = math.floor(px), math.floor(py)
						local a, b = px - j0, py - i0
						if j0 < 0 then
							j0 += nx
						elseif j0 >= nx then
							j0 -= nx
						end
						if i0 < 0 then
							i0 += ny
						elseif i0 >= ny then
							i0 -= ny
						end
						local c0 = row * j0
						local c1 = if j0 == nx - 1 then 0 else c0 + row
						local r0 = 8 * i0
						local r1 = if i0 == ny - 1 then 0 else r0 + 8
						local f00, f01 = buffer.readf64(f, c0 + r0), buffer.readf64(f, c0 + r1)
						local f10, f11 = buffer.readf64(f, c1 + r0), buffer.readf64(f, c1 + r1)
						buffer.writef64(lo :: buffer, k, math.min(math.min(f00, f01), math.min(f10, f11)))
						buffer.writef64(hi :: buffer, k, math.max(math.max(f00, f01), math.max(f10, f11)))
						value = (1 - a) * ((1 - b) * f00 + b * f01) + a * ((1 - b) * f10 + b * f11)
					end
	buffer.writef64(out, k, value)
				else
					local value
					do
						local px, py = xd, yd
						local j0, i0 = math.floor(px), math.floor(py)
						local a, b = px - j0, py - i0
						if j0 < 0 then
							j0 += nx
						elseif j0 >= nx then
							j0 -= nx
						end
						if i0 < 0 then
							i0 += ny
						elseif i0 >= ny then
							i0 -= ny
						end
						local c0 = row * j0
						local c1 = if j0 == nx - 1 then 0 else c0 + row
						local r0 = 8 * i0
						local r1 = if i0 == ny - 1 then 0 else r0 + 8
						local f00, f01 = buffer.readf64(f, c0 + r0), buffer.readf64(f, c0 + r1)
						local f10, f11 = buffer.readf64(f, c1 + r0), buffer.readf64(f, c1 + r1)
						value = (1 - a) * ((1 - b) * f00 + b * f01) + a * ((1 - b) * f10 + b * f11)
					end
	buffer.writef64(out, k, value)
				end
			end
		end
	end

	local lo, hi = buffer.create(8 * n), buffer.create(8 * n)
	-- Cells within two of the solid (u8), where MacCormack falls back to first order: at
	-- the solid's edge the flow nearly stagnates, and the correction (dispersive, like
	-- Lax-Wendroff's at small Courant numbers) would leave grid-scale wiggles there.
	local near = buffer.create(n)
	-- Advects f (offset ox, oy) by the velocity (u, v) into out.
	local function advect(out: buffer, f: buffer, u: buffer, v: buffer, ox: number, oy: number, dt: number, tmp: buffer)
		if not maccormack then
			pass(out, f, u, v, ox, oy, 1, dt)
			return
		end
		pass(out, f, u, v, ox, oy, 1, dt, lo, hi) -- forward in time: phi^
		pass(tmp, out, u, v, ox, oy, -1, dt) -- back again: phi~
		for c = 0, n - 1 do
			if buffer.readu8(near, c) == 0 then
				local k = 8 * c
				local value = buffer.readf64(out, k) + 0.5 * (buffer.readf64(f, k) - buffer.readf64(tmp, k))
				buffer.writef64(out, k, math.clamp(value, buffer.readf64(lo, k), buffer.readf64(hi, k)))
			end
		end
	end

	-- Fourier symbols: for the staggered differences, (e^{i theta} - 1) for the divergence
	-- and (1 - e^{-i theta}) for the gradient, and lambda = -4 sin^2(theta / 2) per axis.
	local cx, sx, lx = table.create(nx, 0), table.create(nx, 0), table.create(nx, 0)
	local cy, sy, ly = table.create(ny, 0), table.create(ny, 0), table.create(ny, 0)
	for k = 0, nx - 1 do
		local th = 2 * math.pi * k / nx
		cx[k + 1], sx[k + 1], lx[k + 1] = math.cos(th), math.sin(th), -4 * math.sin(th / 2) ^ 2
	end
	for k = 0, ny - 1 do
		local th = 2 * math.pi * k / ny
		cy[k + 1], sy[k + 1], ly[k + 1] = math.cos(th), math.sin(th), -4 * math.sin(th / 2) ^ 2
	end

	local function diffuse_project(dt: number)
		local u, v, z = self.u, self.v, Z.data
		for k = 0, n - 1 do
			buffer.writef64(z, 16 * k, buffer.readf64(u, 8 * k))
			buffer.writef64(z, 16 * k + 8, buffer.readf64(v, 8 * k))
		end
		c64.fft2(Z)
		local nu = self.nu
		-- Each pair of modes (k, -k) is handled together: U(k) = (Z(k) + conj Z(-k)) / 2,
		-- V(k) = (Z(k) - conj Z(-k)) / (2i), and after the update Z(k) = U(k) + i V(k).
		for a = 0, nx - 1 do
			local na = (nx - a) % nx
			for b = 0, ny - 1 do
				local nb = (ny - b) % ny
				local k, m = 16 * (a * ny + b), 16 * (na * ny + nb)
				if m < k then
					continue
				end
				local zr, zi = buffer.readf64(z, k), buffer.readf64(z, k + 8)
				local wr, wi = buffer.readf64(z, m), buffer.readf64(z, m + 8)
				-- at k: U = (z + conj w) / 2, V = (z - conj w) / (2i); at -k: U' = conj U, V' = conj V
				local Ur, Ui = 0.5 * (zr + wr), 0.5 * (zi - wi)
				local Vr, Vi = 0.5 * (zi + wi), -0.5 * (zr - wr)
				local lam = lx[a + 1] + ly[b + 1]
				local decay = math.exp(nu * lam * dt)
				Ur, Ui, Vr, Vi = decay * Ur, decay * Ui, decay * Vr, decay * Vi
				if lam ~= 0 then
					-- div = (e^{i tx} - 1) U + (e^{i ty} - 1) V; P = div / lam
					local axr, axi = cx[a + 1] - 1, sx[a + 1]
					local ayr, ayi = cy[b + 1] - 1, sy[b + 1]
					local dr = axr * Ur - axi * Ui + ayr * Vr - ayi * Vi
					local di = axr * Ui + axi * Ur + ayr * Vi + ayi * Vr
					local pr, pi = dr / lam, di / lam
					-- U -= (1 - e^{-i tx}) P, V -= (1 - e^{-i ty}) P; 1 - e^{-i t} = (1 - cos t) + i sin t
					local gxr, gxi = 1 - cx[a + 1], sx[a + 1]
					local gyr, gyi = 1 - cy[b + 1], sy[b + 1]
					Ur, Ui = Ur - (gxr * pr - gxi * pi), Ui - (gxr * pi + gxi * pr)
					Vr, Vi = Vr - (gyr * pr - gyi * pi), Vi - (gyr * pi + gyi * pr)
				end
				-- Z(k) = U + i V; Z(-k) = conj U + i conj V
				buffer.writef64(z, k, Ur - Vi)
				buffer.writef64(z, k + 8, Ui + Vr)
				if m ~= k then
					buffer.writef64(z, m, Ur + Vi)
					buffer.writef64(z, m + 8, -Ui + Vr)
				end
			end
		end
		c64.ifft2(Z)
		for k = 0, n - 1 do
			buffer.writef64(u, 8 * k, buffer.readf64(z, 16 * k))
			buffer.writef64(v, 8 * k, buffer.readf64(z, 16 * k + 8))
		end
	end

	function self.set_reynolds(s: Flow, re: number)
		reynolds = re
		s.nu = s.speed * s.diameter / re
	end

	-- Obstacles: "cylinder" (size = diameter), "plate" (a vertical plate, size = height) or
	-- "airfoil" (a NACA 0012 at 10 degrees angle of attack, size = chord), centered at (cx, cy).
	function self.set_obstacle(s: Flow, shape: string, ox: number, oy: number, size: number)
		local solid = s.solid
		local height = size
		for j = 0, nx - 1 do
			for i = 0, ny - 1 do
				local x, y = j + 0.5 - ox, i + 0.5 - oy
				local inside = false
				if shape == "cylinder" then
					inside = x * x + y * y <= size * size / 4
				elseif shape == "plate" then
					inside = math.abs(x) <= 1.5 and math.abs(y) <= size / 2
				else
					local a = math.rad(10)
					local xr, yr = math.cos(a) * x - math.sin(a) * y + size / 2, math.sin(a) * x + math.cos(a) * y
					local t = xr / size
					if t >= 0 and t <= 1 then
						local half = 5 * 0.12 * size * (0.2969 * math.sqrt(t) - 0.126 * t - 0.3516 * t * t + 0.2843 * t ^ 3 - 0.1015 * t ^ 4)
						inside = math.abs(yr) <= half
					end
				end
				buffer.writeu8(solid, j * ny + i, if inside then 1 else 0)
			end
		end
		if shape == "airfoil" then
			height = size * math.sin(math.rad(10)) + 0.12 * size
		end
		buffer.fill(near, 0, 0)
		for j = 0, nx - 1 do
			for i = 0, ny - 1 do
				if buffer.readu8(solid, j * ny + i) == 1 then
					for dj = -2, 2 do
						for di = -2, 2 do
							buffer.writeu8(near, ((j + dj) % nx) * ny + (i + di) % ny, 1)
						end
					end
				end
			end
		end
		s.diameter = height
		s.set_reynolds(s, reynolds)
	end

	-- Zero the velocity on faces touching the solid, and the dye inside it (dye left there
	-- would not move and would bleed into the wake through MacCormack's correction).
	local function penalize()
		local u, v, solid, dye = self.u, self.v, self.solid, self.dye
		for j = 0, nx - 1 do
			local jm = (j - 1) % nx
			for i = 0, ny - 1 do
				local im = (i - 1) % ny
				local c = buffer.readu8(solid, j * ny + i)
				if c == 1 then
					buffer.writef64(dye, 8 * (j * ny + i), 0)
				end
				if c == 1 or buffer.readu8(solid, jm * ny + i) == 1 then
					buffer.writef64(u, 8 * (j * ny + i), 0)
				end
				if c == 1 or buffer.readu8(solid, j * ny + im) == 1 then
					buffer.writef64(v, 8 * (j * ny + i), 0)
				end
			end
		end
	end

	local function force_fringe(dt: number)
		local u, v, dye = self.u, self.v, self.dye
		local U = self.speed
		local rate = 1 - math.exp(-dt / 0.05)
		for j = nx - F, nx - 1 do
			local w = fringe[j + 1] * rate
			for i = 0, ny - 1 do
				local k = 8 * (j * ny + i)
				buffer.writef64(u, k, buffer.readf64(u, k) + w * (U - buffer.readf64(u, k)))
				buffer.writef64(v, k, buffer.readf64(v, k) * (1 - w))
				buffer.writef64(dye, k, buffer.readf64(dye, k) + w * (streak(i) - buffer.readf64(dye, k)))
			end
		end
	end

	function self.step(s: Flow, dt: number)
		advect(u1, s.u, s.u, s.v, 0, 0.5, dt, u2)
		advect(v1, s.v, s.u, s.v, 0.5, 0, dt, v2)
		advect(d1, s.dye, s.u, s.v, 0.5, 0.5, dt, d2)
		s.u, u1 = u1, s.u
		s.v, v1 = v1, s.v
		s.dye, d1 = d1, s.dye
		force_fringe(dt)
		penalize()
		diffuse_project(dt)
		s.t += dt
	end

	-- Vorticity dv/dx - du/dy at the cell corners (j, i).
	function self.vorticity(s: Flow, out: buffer?): buffer
		local w = out or buffer.create(8 * n)
		local u, v = s.u, s.v
		for j = 0, nx - 1 do
			local jm = (j - 1) % nx
			for i = 0, ny - 1 do
				local im = (i - 1) % ny
				local val = buffer.readf64(v, 8 * (j * ny + i)) - buffer.readf64(v, 8 * (jm * ny + i))
					- buffer.readf64(u, 8 * (j * ny + i)) + buffer.readf64(u, 8 * (j * ny + im))
				buffer.writef64(w, 8 * (j * ny + i), val)
			end
		end
		return w
	end

	-- Largest |div u| over the cells.
	function self.divergence(s: Flow): number
		local u, v = s.u, s.v
		local worst = 0
		for j = 0, nx - 1 do
			local jp = (j + 1) % nx
			for i = 0, ny - 1 do
				local ip = (i + 1) % ny
				local d = buffer.readf64(u, 8 * (jp * ny + i)) - buffer.readf64(u, 8 * (j * ny + i))
					+ buffer.readf64(v, 8 * (j * ny + ip)) - buffer.readf64(v, 8 * (j * ny + i))
				worst = math.max(worst, math.abs(d))
			end
		end
		return worst
	end

	return self :: Flow
end

return flow
