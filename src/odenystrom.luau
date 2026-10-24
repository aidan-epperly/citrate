--!strict
--!optimize 2

-- Runge-Kutta-Nystrom methods for y'' = f(t, y): the explicit embedded pairs of Dormand,
-- El-Mikkawy and Prince, which use the structure of a second-order equation whose force
-- does not depend on the velocity to reach a given order with fewer stages than a
-- Runge-Kutta method applied to the equivalent first-order system:
--   "dprkn6"   order 6 with an order 4 error estimate, 6 stages (one reused from the last
--              step), and a free sixth-order continuous extension
--   "dprkn8"   order 8(6), 9 stages (one reused); its continuous extension interpolates
--              y, v and y'' at the step's ends and the neighboring steps' far ends
--   "dprkn12"  order 12(10), 17 stages, for very tight tolerances; its continuous
--              extension takes a fresh step to the requested time (16 evaluations)
-- A stage at y evaluates f(t, y, v) with v the velocity at the start of the step: f must
-- not depend on v (use a Runge-Kutta method through ode.solve_second_order otherwise).
-- These are used through ode.solve_second_order; the state there is the pair (y, v) in
-- the pair space defined here.
--
-- The coefficients are copied from OrdinaryDiffEq.jl's rkn_tableaus.jl (MIT; see the
-- third-party coefficients in docs/development.md), evaluated in high precision from its exact
-- rational (and, for dprkn6, sqrt(8581)) forms. Stage i of a step of size h takes
--   u_i = y + c_i h v + h^2 sum_{j<i} a_ij k_j,   k_i = f(t + c_i h, u_i),
-- and y_new = y + h v + h^2 sum b_i k_i, v_new = v + h sum bp_i k_i, with error estimates
-- h^2 sum btilde_i k_i and h sum bptilde_i k_i.

local types = require("./types")
local core = require("./odecore")

type Space<T> = types.Space<T>
type Piece<T> = core.Piece<T>

local TABLES: { [string]: any } = {
	dprkn6 = {
		stages = 6,
		a = {
			{  },
			{ 0.008358715283968025 },
			{ 0.011144953711957367, 0.022289907423914734 },
			{ 0.1454747428010918, -0.22986064052264749, 0.3090349872029675 },
			{ -0.20766826295078997, 0.6863667842925143, -0.19954927787234925, 0.12585075653062489 },
			{ 0.07811016144349478, 0.0, 0.2882917411897668, 0.12242553717457041, 0.011172560192168035 },
		},
		c = { 0.0, 0.12929590313670442, 0.25859180627340883, 0.67029708261548, 0.9, 1.0 },
		b = { 0.07811016144349478, 0.0, 0.2882917411897668, 0.12242553717457041, 0.011172560192168035, 0.0 },
		bp = { 0.07811016144349478, 0.0, 0.3888434787059826, 0.3713207579288423, 0.11172560192168035, 0.05 },
		btilde = { -0.9807490989269235, 2.406751371924452, -1.559600370364267, 0.12242553717457041, 0.011172560192168035, 0.0 },
		bptilde = { 0.023504273504273504, 0.0, -0.07242330719764424, 0.17543989844952962, -0.2765208647561589, 0.15 },
		r = {
			{ 0.5, -1.245014245014245, 1.5161443494776827, -0.9066951566951567, 0.21367521367521367 },
			{ 0.0, 0.0, 0.0, 0.0, 0.0 },
			{ 0.0, 1.9859371988705032, -3.577652872294921, 2.5384011164109506, -0.6583937017967658 },
			{ 0.0, -1.8559358276926614, 5.547586751052329, -5.164133553908094, 1.5949081677229964 },
			{ 0.0, 2.178649237472767, -6.926987319144182, 7.273336685101391, -2.513826043237808 },
			{ 0.0, -1.0636363636363637, 3.440909090909091, -3.7409090909090907, 1.3636363636363635 },
		},
		rp = {
			{ 1.0, -3.735042735042735, 6.064577397910731, -4.533475783475783, 1.2820512820512822 },
			{ 0.0, 0.0, 0.0, 0.0, 0.0 },
			{ 0.0, 5.95781159661151, -14.310611489179683, 12.692005582054751, -3.950362210780595 },
			{ 0.0, -5.567807483077984, 22.190347004209315, -25.820667769540467, 9.56944900633798 },
			{ 0.0, 6.5359477124183005, -27.707949276576727, 36.366683425506956, -15.082956259426847 },
			{ 0.0, -3.190909090909091, 13.763636363636364, -18.704545454545453, 8.181818181818182 },
		},
	},
	dprkn8 = {
		stages = 9,
		a = {
			{  },
			{ 0.00125 },
			{ 0.0016666666666666668, 0.0033333333333333335 },
			{ 0.045, -0.09, 0.09 },
			{ -0.3379798532571243, 0.8611516478170984, -0.48613644655235316, 0.0879646519923791 },
			{ 0.7461462641043086, -1.804347430246737, 1.2761518285888953, -0.031220932716737166, 0.05827027027027027 },
			{ -1.2551623461807584, 3.257463187064823, -2.068866619575089, 0.4206478455603251, -0.018382978723404254, 0.06930091185410335 },
			{ 3.772901830095941, -9.669613121677195, 6.7991544547851674, -0.8490343907823893, 0.4671734865470852, -0.04939742152466368, 0.028815162556053812 },
			{ 0.028092718568909044, 0.0, 0.1457093253968254, 0.1529431216931217, 0.09151785714285714, 0.06554705215419501, 0.01618992504409171, 0.0 },
		},
		c = { 0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.0 },
		b = { 0.028092718568909044, 0.0, 0.1457093253968254, 0.1529431216931217, 0.09151785714285714, 0.06554705215419501, 0.01618992504409171, 0.0, 0.0 },
		bp = { 0.028092718568909044, 0.0, 0.1618992504409171, 0.2184901738473167, 0.18303571428571427, 0.2184901738473167, 0.1618992504409171, 0.028092718568909044, 0.0 },
		btilde = { -0.044557986852984274, 0.0, 0.10965442077101599, -0.14620589436135464, 0.1315853049252192, -0.06265966901200913, 0.012183824530112887, 0.0, 0.0 },
		bptilde = { -0.044557986852984274, 0.0, 0.12183824530112887, -0.2088655633733638, 0.2631706098504384, -0.2088655633733638, 0.12183824530112887, 0.10544201314701572, -0.15 },
	},
	dprkn12 = {
		stages = 17,
		a = {
			{  },
			{ 0.0002 },
			{ 0.0002666666666666667, 0.0005333333333333334 },
			{ 0.002916666666666667, -0.004166666666666667, 0.00625 },
			{ 0.0016460905349794238, 0.0, 0.0054869684499314125, 0.0017558299039780521 },
			{ 0.0019456, 0.0, 0.007151746031746032, 0.0029127111111111113, 0.0007899428571428571 },
			{ 0.00056640625, 0.0, 0.0008809730489417989, -0.0004369212962962963, 0.00033900669642857143, -9.946469907407407e-05 },
			{ 0.0030833333333333333, 0.0, 0.0, 0.0017777777777777779, 0.0027, 0.0015782828282828283, 0.01086060606060606 },
			{ 0.00365183937480113, 0.0, 0.003965171714072343, 0.0031972582629306284, 0.008221467306855435, -0.0013130926959572379, 0.009771586968064868, 0.003755769069232834 },
			{ 0.003707241068718501, 0.0, 0.005082045854555286, 0.001174708002175412, -0.021147629915126993, 0.06010463698107881, 0.02010573476850619, -0.02835075012293358, 0.014879568918581932 },
			{ 0.03512537656073344, 0.0, -0.008615749195138479, -0.005791448051007917, 1.945554823782616, -3.4351238674565137, -0.10930701107475221, 2.3496383118995166, -0.7560094086870229, 0.10952897222156927 },
			{ 0.020527792537482496, 0.0, -0.00728644676448018, -0.0021153556079618403, 0.9275807968723522, -1.6522824844257367, -0.02107956300568657, 1.2065364326207872, -0.4137144770010661, 0.09079873982809654, 0.005355552600533985 },
			{ -0.14324078875545515, 0.0, 0.012528703773091817, 0.006826019163969827, -4.799555395574387, 5.6986250439519415, 0.7553430369523645, -0.12755487858281084, -1.9605926051117384, 0.9185609056635262, -0.23880085505284432, 0.15911081357234216 },
			{ 0.8045019205520489, 0.0, -0.016658527067011247, -0.021415834042629735, 16.827235928962466, -11.172835357176098, -3.377159297226324, -15.243326655360846, 17.179835738215417, -5.437719239823995, 1.3878671618364655, -0.5925827732652812, 0.029603873171297354 },
			{ -0.913296766697358, 0.0, 0.0024112725757805178, 0.01765812269386174, -14.851649779720384, 2.1589708670045757, 3.99791558311788, 28.434151800232232, -25.2593643549416, 7.733878542362238, -1.8913028948478674, 1.0014845070224718, 0.004641199599109052, 0.011218755022148957 },
			{ -0.27519629720559396, 0.0, 0.03661188877915492, 0.009789519688231562, -12.293062345886211, 14.207226453937903, 1.5866476906789537, 2.4577735327595946, -8.935193694403273, 4.373672731613407, -1.8347181765449492, 1.1592085289061491, -0.017290253165383924, 0.019325977904460768, 0.005204442937554993 },
			{ 1.3076391847404059, 0.0, 0.017364109189745843, -0.018544456454265796, 14.811522032867726, 9.38317630848247, -5.2284261999445425, -48.95128052584765, 38.297096034337926, -10.58738133697598, 2.4332304376226275, -1.0453406042575444, 0.0717732095086726, 0.0021622109708082783, 0.007009595759602514, 0.0 },
		},
		c = { 0.0, 0.02, 0.04, 0.1, 0.13333333333333333, 0.16, 0.05, 0.2, 0.25, 0.3333333333333333, 0.5, 0.5555555555555556, 0.75, 0.8571428571428571, 0.9452162222720143, 1.0, 1.0 },
		b = { 0.012127868517185414, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08629746251568875, 0.2525469581187147, -0.1974186799326823, 0.2031869190789726, -0.020775808077714918, 0.10967804874502014, 0.038065132526466504, 0.01163406880432423, 0.0046580297040248785, 0.0, 0.0 },
		bp = { 0.012127868517185414, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09083943422704079, 0.3156836976483934, -0.2632249065769097, 0.3047803786184589, -0.041551616155429835, 0.2467756096762953, 0.15226053010586602, 0.08143848163026961, 0.08502571193890811, -0.009155189630077963, 0.025 },
		btilde = { -0.004880833389821577, 0.0, 0.0, 0.0, 0.0, 0.0, 0.014038126584857338, -0.11947921920803832, 0.20440246507662121, -0.1322681492223791, 0.11053069299761689, -0.07975385787102851, 0.011224330486437457, -0.004671596801593694, 0.0008580413473282841, 0.0, 0.0 },
		bptilde = { -0.004880833389821577, 0.0, 0.0, 0.0, 0.0, 0.0, 0.014776975352481408, -0.1493490240100479, 0.2725366201021616, -0.19840222383356862, 0.22106138599523378, -0.17944618020981415, 0.04489732194574983, -0.03270117761115586, 0.015662325288859434, -0.029155189630077964, 0.025 },
	},
}
TABLES.dprkn6.order, TABLES.dprkn6.eorder, TABLES.dprkn6.fsal = 6, 4, true
TABLES.dprkn8.order, TABLES.dprkn8.eorder, TABLES.dprkn8.fsal = 8, 6, true
TABLES.dprkn12.order, TABLES.dprkn12.eorder, TABLES.dprkn12.fsal = 12, 10, false
TABLES.dprkn12.restart = true

-- The pair space: states { y, v } of two values of S, with the operations componentwise,
-- the norm of the concatenation and the error norm the RMS of the two halves'.
local function pair_space<T>(S: Space<T>): Space<{ T }>
	local function part(out: { T }?, i: number, like: T): T
		if out ~= nil and out[i] ~= nil then
			return out[i]
		end
		return S.similar(like)
	end
	local P: any = { eps = S.eps, real = S.real }
	function P.similar(x: { T }): { T }
		return { S.similar(x[1]), S.similar(x[2]) }
	end
	function P.copy(out: { T }?, x: { T }): { T }
		return { S.copy(part(out, 1, x[1]), x[1]), S.copy(part(out, 2, x[2]), x[2]) }
	end
	function P.add(out: { T }?, x: { T }, y: { T }): { T }
		return { S.add(part(out, 1, x[1]), x[1], y[1]), S.add(part(out, 2, x[2]), x[2], y[2]) }
	end
	function P.sub(out: { T }?, x: { T }, y: { T }): { T }
		return { S.sub(part(out, 1, x[1]), x[1], y[1]), S.sub(part(out, 2, x[2]), x[2], y[2]) }
	end
	function P.scale(out: { T }?, a: number, x: { T }): { T }
		return { S.scale(part(out, 1, x[1]), a, x[1]), S.scale(part(out, 2, x[2]), a, x[2]) }
	end
	function P.axpy(out: { T }?, a: number, x: { T }, y: { T }): { T }
		return { S.axpy(part(out, 1, x[1]), a, x[1], y[1]), S.axpy(part(out, 2, x[2]), a, x[2], y[2]) }
	end
	function P.dot(x: { T }, y: { T }): number
		return S.dot(x[1], y[1]) + S.dot(x[2], y[2])
	end
	function P.norm(x: { T }): number
		local a, b = S.norm(x[1]), S.norm(x[2])
		return math.sqrt(a * a + b * b)
	end
	local errnorm = S.errnorm
	if errnorm then
		function P.errnorm(e: { T }, y: { T }, yn: { T }, atol: number, rtol: number): number
			local a = (errnorm :: any)(e[1], y[1], yn[1], atol, rtol)
			local b = (errnorm :: any)(e[2], y[2], yn[2], atol, rtol)
			return math.sqrt((a * a + b * b) / 2)
		end
	end
	return P
end

-- y + sum_i w[i] K[i] over the nonzero weights (w scaled by s), into a new value.
local function accumulate<T>(S: Space<T>, base: T, s: number, w: { number }, K: { T }, count: number): T
	local o = S.copy(S.similar(base), base)
	for i = 1, count do
		local c = w[i]
		if c and c ~= 0 then
			o = S.axpy(o, s * c, K[i], o)
		end
	end
	return o
end

-- For Hermite interpolation with value, first and second derivative at each node: the
-- matrix W with W[r][k] the coefficient of x^(k-1) in the basis polynomial of datum r
-- (data ordered node by node: value, derivative, second derivative). It is the inverse
-- transpose of the confluent Vandermonde matrix, by Gauss-Jordan with partial pivoting.
local function hermite_inverse(nodes: { number }): { { number } }
	local m = 3 * #nodes
	-- V[r][k] = the datum r of x^(k-1); solve V^T W^T = I, i.e. W = V^-1 transposed.
	local A: { { number } } = {}
	for r = 1, m do
		local x = nodes[(r - 1) // 3 + 1]
		local kind = (r - 1) % 3
		local row = table.create(2 * m, 0)
		for k = 1, m do
			local e = k - 1
			if kind == 0 then
				row[k] = x ^ e
			elseif kind == 1 then
				row[k] = if e >= 1 then e * x ^ (e - 1) else 0
			else
				row[k] = if e >= 2 then e * (e - 1) * x ^ (e - 2) else 0
			end
		end
		row[m + r] = 1
		A[r] = row
	end
	for col = 1, m do
		local piv = col
		for r = col + 1, m do
			if math.abs(A[r][col]) > math.abs(A[piv][col]) then
				piv = r
			end
		end
		A[col], A[piv] = A[piv], A[col]
		local d = A[col][col]
		for k = 1, 2 * m do
			A[col][k] /= d
		end
		for r = 1, m do
			if r ~= col then
				local factor = A[r][col]
				if factor ~= 0 then
					for k = 1, 2 * m do
						A[r][k] -= factor * A[col][k]
					end
				end
			end
		end
	end
	-- A's right half is V^-1: column r of V^-1 holds the coefficients of basis r.
	local W: { { number } } = {}
	for r = 1, m do
		local w = table.create(m, 0)
		for k = 1, m do
			w[k] = A[k][m + r]
		end
		W[r] = w
	end
	return W
end

-- An adaptive RKN stepper over pair states (see ode.luau's steppers), for
-- f(t, y, v, out) -> the acceleration.
local function stepper<T>(S: Space<T>, f: (number, T, T, T) -> T, t0: number, y0: T, v0: T, o: any): any
	local tab = TABLES[o.method]
	assert(tab ~= nil, `ode: unknown Nystrom method {o.method}`)
	local P = pair_space(S)
	local rtol, atol = o.rtol or 1e-3, o.atol or 1e-6
	local maxstep = o.max_step or math.huge
	local s = tab.stages
	local alpha = 1 / (tab.eorder + 1)
	local y, v = S.copy(S.similar(y0), y0), S.copy(S.similar(v0), v0)
	local acc = f(t0, y, v, S.similar(y0))
	local stats = { nfev = 1, steps = 0, rejected = 0 }
	local st: any = { t = t0, y = { y, v }, piece = nil, stats = stats }
	local h: number? = if o.h then math.abs(o.h) else nil
	local rejected_last = false
	-- The start of the previous step, for the dense output of the Hermite methods.
	local prev: { t: number, y: T, v: T, a: T }? = nil
	-- The last Hermite piece, which learns the end of the following step.
	local last_link: any = nil

	-- One step of size h from (t, y, v) with y'' = a there: (y_new, v_new, stages).
	local function raw_step(t: number, y: T, v: T, a: T, h: number): (T, T, { T })
		local K: { T } = { a }
		for i = 2, s do
			local u = accumulate(S, y, h * h, tab.a[i], K, i - 1)
			u = S.axpy(u, tab.c[i] * h, v, u)
			K[i] = f(t + tab.c[i] * h, u, v, S.similar(y))
			stats.nfev += 1
		end
		local y1 = accumulate(S, y, h * h, tab.b, K, s)
		y1 = S.axpy(y1, h, v, y1)
		return y1, accumulate(S, v, h, tab.bp, K, s), K
	end

	function st.step(self: any, t_bound: number, want: boolean): (boolean, string?)
		local t = self.t
		local direction = if t_bound >= t then 1 else -1
		if h == nil then
			local F = function(tt: number, x: { T }, _: { T }): { T }
				return { x[2], f(tt, x[1], x[2], S.similar(x[1])) }
			end
			local hs, used = core.initial_step(P, F, t, { y, v }, { v, acc }, direction, math.abs(t_bound - t), atol, rtol, maxstep, tab.eorder)
			h = hs
			stats.nfev += used
		end
		while true do
			local hcur = h :: number
			local remaining = math.abs(t_bound - t)
			local hs = math.min(hcur, remaining)
			if remaining - hs <= 4 * core.EPS * math.abs(t_bound) then
				hs = remaining
			end
			if hs <= 16 * core.EPS * math.max(math.abs(t), 1) then
				return false, "the step size underflowed"
			end
			local hd = direction * hs
			local y1, v1, K = raw_step(t, y, v, acc, hd)
			local zero = S.scale(S.similar(y), 0, y)
			local ey = accumulate(S, zero, hd * hd, tab.btilde, K, s)
			local ev = accumulate(S, zero, hd, tab.bptilde, K, s)
			local err = core.weighted(P, { ey, ev }, { y, v }, { y1, v1 }, atol, rtol)
			if err ~= err then
				return false, "the solution became NaN"
			elseif err > 1 then
				h = hs * math.max(0.2, 0.9 * err ^ -alpha)
				rejected_last = true
				stats.rejected += 1
				continue
			end
			local factor = if err == 0 then 10 else math.clamp(0.9 * err ^ -alpha, 0.2, if rejected_last then 1 else 10)
			rejected_last = false
			-- A step cut short at t_bound keeps the controller's step for next time.
			if hs >= hcur then
				h = math.min(hs * factor, maxstep)
			end
			local tnew = if hs == remaining then t_bound else t + hd
			-- The last stage of the FSAL pairs is f at (t + h, y_new).
			local acc1 = if tab.fsal then K[s] else f(tnew, y1, v1, S.similar(y))
			if not tab.fsal then
				stats.nfev += 1
			end
			if want then
				local yo, vo = y, v
				local piece: Piece<{ T }> = { t0 = t, h = hd, y0 = { y, v }, kind = "eval", q = {} }
				if tab.restart then
					-- Dense output of dprkn12 by a fresh step from the step's start to the
					-- requested time: as accurate as the steps themselves (no interpolant of
					-- moderate degree keeps up with its long steps), at s - 1 evaluations of
					-- f per point.
					local a0 = acc
					piece.eval = function(tq: number, _: { T }?): { T }
						if tq == t then
							return { S.copy(S.similar(yo), yo), S.copy(S.similar(vo), vo) }
						end
						local yq, vq = raw_step(t, yo, vo, a0, tq - t)
						return { yq, vq }
					end
				elseif tab.r then
					-- Dense output of dprkn6: y(theta) = y + h theta (v + h theta sum_i
					-- b_i(theta) k_i), v(theta) = v + h theta sum_i bp_i(theta) k_i, with
					-- b_i, bp_i the polynomials of degree 4 in r and rp.
					local Ks = K
					piece.eval = function(tq: number, _: { T }?): { T }
						local th = (tq - t) / hd
						local wy, wv = {}, {}
						for i = 1, 6 do
							local ri, rpi = tab.r[i], tab.rp[i]
							local by, bv = 0, 0
							for j = 5, 1, -1 do
								by = by * th + ri[j]
								bv = bv * th + rpi[j]
							end
							wy[i], wv[i] = by, bv
						end
						local yq = accumulate(S, zero, hd * th, wy, Ks, 6)
						yq = S.add(yq, yq, vo)
						yq = S.scale(yq, hd * th, yq)
						yq = S.add(yq, yq, yo)
						local vq = accumulate(S, vo, hd * th, wv, Ks, 6)
						return { yq, vq }
					end
				else
					-- Hermite interpolation of y from y, y' = v and y'' = f at the step's
					-- ends, the previous step's start and, once it is taken, the next step's
					-- end (degree up to 11; at least 5 on a lone step), and its derivative
					-- for v. No further evaluations of f.
					local before = prev
					local link: any = {}
					local cache: { [number]: { { number } } } = {}
					local ends: { { T } } = { { y, v, acc }, { y1, v1, acc1 } }
					piece.eval = function(tq: number, _: { T }?): { T }
						local nodes: { number } = { 0, 1 }
						local data: { { T } } = { ends[1], ends[2] }
						if before ~= nil then
							local b0 = before :: any
							table.insert(nodes, 1, (b0.t - t) / hd)
							table.insert(data, 1, { b0.y, b0.v, b0.a })
						end
						if link.next ~= nil then
							table.insert(nodes, (link.next.t - t) / hd)
							table.insert(data, { link.next.y, link.next.v, link.next.a })
						end
						local W = cache[#nodes + (if link.next then 10 else 0)]
						if W == nil then
							W = hermite_inverse(nodes)
							cache[#nodes + (if link.next then 10 else 0)] = W
						end
						local m = 3 * #nodes
						local x = (tq - t) / hd
						-- Monomials and their derivatives at x, then the basis weights.
						local mono, dmono = table.create(m, 0), table.create(m, 0)
						local pw = 1
						for k = 1, m do
							mono[k] = pw
							dmono[k] = if k >= 2 then (k - 1) * x ^ (k - 2) else 0
							pw *= x
						end
						local yq = S.scale(S.similar(y1), 0, y1)
						local vq = S.scale(S.similar(v1), 0, v1)
						for r = 1, m do
							local wy, wv = 0, 0
							local Wr = W[r]
							for k = 1, m do
								wy += Wr[k] * mono[k]
								wv += Wr[k] * dmono[k]
							end
							local node = data[(r - 1) // 3 + 1]
							local kind = (r - 1) % 3
							-- Datum r is y, h v or h^2 a at its node (derivatives in x).
							local scale = if kind == 0 then 1 elseif kind == 1 then hd else hd * hd
							yq = S.axpy(yq, wy * scale, node[kind + 1], yq)
							vq = S.axpy(vq, wv * scale / hd, node[kind + 1], vq)
						end
						return { yq, vq }
					end
					if last_link ~= nil then
						(last_link :: any).next = { t = tnew, y = y1, v = v1, a = acc1 }
					end
					last_link = link
				end
				self.piece = piece
			else
				self.piece = nil
			end
			prev = { t = t, y = y, v = v, a = acc }
			y, v, acc = y1, v1, acc1
			self.t, self.y = tnew, { y, v }
			stats.steps += 1
			return true, nil
		end
	end
	return st
end

local methods: { [string]: boolean } = { dprkn6 = true, dprkn8 = true, dprkn12 = true }

return {
	pair_space = pair_space,
	stepper = stepper,
	methods = methods,
}
