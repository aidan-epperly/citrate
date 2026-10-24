--!strict
--!optimize 2

-- Ordinary differential equations y' = f(t, y), generic over how the state is stored: the
-- solvers touch y only through a Space, so a state can be a `vector` (and stays one), an
-- f32/f64/c32/c64 array, or anything with a Space. Times, step sizes and tolerances are
-- plain numbers.
--
-- The right-hand side is f(t, y, out) -> T: it may write into out (buffers, to avoid
-- allocating) or return a new value (vectors). Explicit Runge-Kutta methods:
--   fixed step  "euler" (order 1), "midpoint", "heun" (2), "rk4" (4)
--   adaptive    "bs3" (Bogacki-Shampine 3(2)), "dp5" (Dormand-Prince 5(4), the default),
--               "dop853" (Hairer's 8(5,3), for tight tolerances)
-- and, for stiff problems and DAEs M y' = f(t, y) (see odestiff.luau):
--   "radau"     Radau IIA of order 5 (Hairer and Wanner's RADAU5)
--   "bdf"       numerical differentiation formulas of orders 1 to 5 (as in ode15s),
--               also for complex states and with sparse or matrix-free linear algebra
-- Adaptive methods control the local error with the weighted RMS norm
-- |e / (atol + rtol max(|y|, |ynew|))| (Space.errnorm; spaces without it use whole-vector
-- norms) and a PI step-size controller, and each has its own continuous extension (dense
-- output) of matching accuracy; fixed-step methods use cubic Hermite interpolation.
--
-- ode.solve integrates over an interval and returns a solution with dense output, events
-- and optional output times; ode.integrator steps a state forward under your control, for
-- simulations that advance by a frame time. ode.mechanics and ode.solve_mechanics are
-- symplectic integrators for q'' = a(t, q) (see below). ode.jacobian estimates df/dy by
-- finite differences (dense, banded or sparse), as the stiff solvers do.

local types = require("./types")
local core = require("./odecore")
local stiff = require("./odestiff")

type Space<T> = types.Space<T>
export type OdeFunction<T> = types.OdeFunction<T>
export type OdeOptions<T> = types.OdeOptions<T>
export type OdeEvent<T> = types.OdeEvent<T>
export type OdeSolution<T> = types.OdeSolution<T>

local EPS = core.EPS
local similar = core.similar
local combine = core.combine
local weighted = core.weighted
local piece_eval = core.piece_eval
local bracket_root = core.bracket_root
type Piece<T> = core.Piece<T>

-- Butcher tableaux. a[s] holds the coefficients of stages 1 .. s-1 for stage s. e (on
-- the stages and, for FSAL methods, the stage f(t + h, y_new) at the end) gives the
-- error estimate h sum e_i k_i. p gives the dense output y_old + h sum_j (sum_i k_i
-- p[i][j]) theta^j (j = 1 ..), as in Hairer's and Shampine's interpolants.
type Tableau = {
	name: string,
	order: number,
	eorder: number?,
	c: { number },
	a: { { number } },
	b: { number },
	e: { number }?,
	p: { { number } }?,
	beta: number,
}

local TABLEAUX: { [string]: Tableau } = {}

TABLEAUX.euler = { name = "euler", order = 1, c = { 0 }, a = { {} }, b = { 1 }, beta = 0 }
TABLEAUX.midpoint = { name = "midpoint", order = 2, c = { 0, 1 / 2 }, a = { {}, { 1 / 2 } }, b = { 0, 1 }, beta = 0 }
TABLEAUX.heun = { name = "heun", order = 2, c = { 0, 1 }, a = { {}, { 1 } }, b = { 1 / 2, 1 / 2 }, beta = 0 }
TABLEAUX.rk4 = {
	name = "rk4",
	order = 4,
	c = { 0, 1 / 2, 1 / 2, 1 },
	a = { {}, { 1 / 2 }, { 0, 1 / 2 }, { 0, 0, 1 } },
	b = { 1 / 6, 1 / 3, 1 / 3, 1 / 6 },
	beta = 0,
}

-- Bogacki and Shampine (1989), with its free cubic interpolant. The interpolant's
-- coefficients p are copied from SciPy's rk.py (BSD-3-Clause; see the README's note on
-- third-party coefficients).
TABLEAUX.bs3 = {
	name = "bs3",
	order = 3,
	eorder = 2,
	c = { 0, 1 / 2, 3 / 4 },
	a = { {}, { 1 / 2 }, { 0, 3 / 4 } },
	b = { 2 / 9, 1 / 3, 4 / 9 },
	e = { 5 / 72, -1 / 12, -1 / 9, 1 / 8 },
	p = { { 1, -4 / 3, 5 / 9 }, { 0, 1, -2 / 3 }, { 0, 4 / 3, -8 / 9 }, { 0, -1, 1 } },
	beta = 0,
}

-- Dormand and Prince (1980), with Shampine's fourth-order interpolant (as in MATLAB's ode45
-- and SciPy's RK45). The interpolant's coefficients p are copied from SciPy's rk.py
-- (BSD-3-Clause; see the README's note on third-party coefficients).
TABLEAUX.dp5 = {
	name = "dp5",
	order = 5,
	eorder = 4,
	c = { 0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1 },
	a = {
		{},
		{ 1 / 5 },
		{ 3 / 40, 9 / 40 },
		{ 44 / 45, -56 / 15, 32 / 9 },
		{ 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729 },
		{ 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656 },
	},
	b = { 35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84 },
	e = { -71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40 },
	p = {
		{ 1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432 },
		{ 0, 0, 0, 0 },
		{ 0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799 },
		{ 0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072 },
		{ 0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632 },
		{ 0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844 },
		{ 0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423 },
	},
	beta = 0.04,
}

-- Hairer's DOP853 (Hairer, Norsett and Wanner, Solving ODEs I): 12 stages, an error
-- estimate combining embedded third- and fifth-order formulas, and a seventh-order
-- interpolant from 3 more stages. Coefficients as in Hairer's code, copied from SciPy's
-- dop853_coefficients.py (BSD-3-Clause; see the README's note on third-party
-- coefficients).
local D8 = {
	c = {
		0.0,
		0.526001519587677318785587544488e-01,
		0.789002279381515978178381316732e-01,
		0.118350341907227396726757197510,
		0.281649658092772603273242802490,
		0.333333333333333333333333333333,
		0.25,
		0.307692307692307692307692307692,
		0.651282051282051282051282051282,
		0.6,
		0.857142857142857142857142857142,
		1.0,
		1.0,
		0.1,
		0.2,
		0.777777777777777777777777777778,
	},
	-- Rows 2 .. 16 (row 13 is the weights b, row 13's stage is f(t + h, y_new)), as sparse
	-- { column, value } lists (columns 1-based).
	rows = {
		[2] = { { 1, 5.26001519587677318785587544488e-2 } },
		[3] = { { 1, 1.97250569845378994544595329183e-2 }, { 2, 5.91751709536136983633785987549e-2 } },
		[4] = { { 1, 2.95875854768068491816892993775e-2 }, { 3, 8.87627564304205475450678981324e-2 } },
		[5] = {
			{ 1, 2.41365134159266685502369798665e-1 },
			{ 3, -8.84549479328286085344864962717e-1 },
			{ 4, 9.24834003261792003115737966543e-1 },
		},
		[6] = {
			{ 1, 3.7037037037037037037037037037e-2 },
			{ 4, 1.70828608729473871279604482173e-1 },
			{ 5, 1.25467687566822425016691814123e-1 },
		},
		[7] = {
			{ 1, 3.7109375e-2 },
			{ 4, 1.70252211019544039314978060272e-1 },
			{ 5, 6.02165389804559606850219397283e-2 },
			{ 6, -1.7578125e-2 },
		},
		[8] = {
			{ 1, 3.70920001185047927108779319836e-2 },
			{ 4, 1.70383925712239993810214054705e-1 },
			{ 5, 1.07262030446373284651809199168e-1 },
			{ 6, -1.53194377486244017527936158236e-2 },
			{ 7, 8.27378916381402288758473766002e-3 },
		},
		[9] = {
			{ 1, 6.24110958716075717114429577812e-1 },
			{ 4, -3.36089262944694129406857109825 },
			{ 5, -8.68219346841726006818189891453e-1 },
			{ 6, 2.75920996994467083049415600797e1 },
			{ 7, 2.01540675504778934086186788979e1 },
			{ 8, -4.34898841810699588477366255144e1 },
		},
		[10] = {
			{ 1, 4.77662536438264365890433908527e-1 },
			{ 4, -2.48811461997166764192642586468 },
			{ 5, -5.90290826836842996371446475743e-1 },
			{ 6, 2.12300514481811942347288949897e1 },
			{ 7, 1.52792336328824235832596922938e1 },
			{ 8, -3.32882109689848629194453265587e1 },
			{ 9, -2.03312017085086261358222928593e-2 },
		},
		[11] = {
			{ 1, -9.3714243008598732571704021658e-1 },
			{ 4, 5.18637242884406370830023853209 },
			{ 5, 1.09143734899672957818500254654 },
			{ 6, -8.14978701074692612513997267357 },
			{ 7, -1.85200656599969598641566180701e1 },
			{ 8, 2.27394870993505042818970056734e1 },
			{ 9, 2.49360555267965238987089396762 },
			{ 10, -3.0467644718982195003823669022 },
		},
		[12] = {
			{ 1, 2.27331014751653820792359768449 },
			{ 4, -1.05344954667372501984066689879e1 },
			{ 5, -2.00087205822486249909675718444 },
			{ 6, -1.79589318631187989172765950534e1 },
			{ 7, 2.79488845294199600508499808837e1 },
			{ 8, -2.85899827713502369474065508674 },
			{ 9, -8.87285693353062954433549289258 },
			{ 10, 1.23605671757943030647266201528e1 },
			{ 11, 6.43392746015763530355970484046e-1 },
		},
		[13] = {
			{ 1, 5.42937341165687622380535766363e-2 },
			{ 6, 4.45031289275240888144113950566 },
			{ 7, 1.89151789931450038304281599044 },
			{ 8, -5.8012039600105847814672114227 },
			{ 9, 3.1116436695781989440891606237e-1 },
			{ 10, -1.52160949662516078556178806805e-1 },
			{ 11, 2.01365400804030348374776537501e-1 },
			{ 12, 4.47106157277725905176885569043e-2 },
		},
		[14] = {
			{ 1, 5.61675022830479523392909219681e-2 },
			{ 7, 2.53500210216624811088794765333e-1 },
			{ 8, -2.46239037470802489917441475441e-1 },
			{ 9, -1.24191423263816360469010140626e-1 },
			{ 10, 1.5329179827876569731206322685e-1 },
			{ 11, 8.20105229563468988491666602057e-3 },
			{ 12, 7.56789766054569976138603589584e-3 },
			{ 13, -8.298e-3 },
		},
		[15] = {
			{ 1, 3.18346481635021405060768473261e-2 },
			{ 6, 2.83009096723667755288322961402e-2 },
			{ 7, 5.35419883074385676223797384372e-2 },
			{ 8, -5.49237485713909884646569340306e-2 },
			{ 11, -1.08347328697249322858509316994e-4 },
			{ 12, 3.82571090835658412954920192323e-4 },
			{ 13, -3.40465008687404560802977114492e-4 },
			{ 14, 1.41312443674632500278074618366e-1 },
		},
		[16] = {
			{ 1, -4.28896301583791923408573538692e-1 },
			{ 6, -4.69762141536116384314449447206 },
			{ 7, 7.68342119606259904184240953878 },
			{ 8, 4.06898981839711007970213554331 },
			{ 9, 3.56727187455281109270669543021e-1 },
			{ 13, -1.39902416515901462129418009734e-3 },
			{ 14, 2.9475147891527723389556272149 },
			{ 15, -9.15095847217987001081870187138 },
		},
	},
	-- Error weights on the 13 stages (stage 13 is f(t + h, y_new)).
	e5 = {
		[1] = 0.1312004499419488073250102996e-1,
		[6] = -0.1225156446376204440720569753e+1,
		[7] = -0.4957589496572501915214079952,
		[8] = 0.1664377182454986536961530415e+1,
		[9] = -0.3503288487499736816886487290,
		[10] = 0.3341791187130174790297318841,
		[11] = 0.8192320648511571246570742613e-1,
		[12] = -0.2235530786388629525884427845e-1,
	},
	-- E3 = b - (0.2440..., at stage 1; 0.7338..., at stage 9; 0.02205..., at stage 12).
	e3_corrections = {
		[1] = 0.244094488188976377952755905512,
		[9] = 0.733846688281611857341361741547,
		[12] = 0.220588235294117647058823529412e-1,
	},
	-- Interpolant rows 4 .. 7 on the 16 stages.
	d = {
		{
			[1] = -0.84289382761090128651353491142e+1,
			[6] = 0.56671495351937776962531783590,
			[7] = -0.30689499459498916912797304727e+1,
			[8] = 0.23846676565120698287728149680e+1,
			[9] = 0.21170345824450282767155149946e+1,
			[10] = -0.87139158377797299206789907490,
			[11] = 0.22404374302607882758541771650e+1,
			[12] = 0.63157877876946881815570249290,
			[13] = -0.88990336451333310820698117400e-1,
			[14] = 0.18148505520854727256656404962e+2,
			[15] = -0.91946323924783554000451984436e+1,
			[16] = -0.44360363875948939664310572000e+1,
		},
		{
			[1] = 0.10427508642579134603413151009e+2,
			[6] = 0.24228349177525818288430175319e+3,
			[7] = 0.16520045171727028198505394887e+3,
			[8] = -0.37454675472269020279518312152e+3,
			[9] = -0.22113666853125306036270938578e+2,
			[10] = 0.77334326684722638389603898808e+1,
			[11] = -0.30674084731089398182061213626e+2,
			[12] = -0.93321305264302278729567221706e+1,
			[13] = 0.15697238121770843886131091075e+2,
			[14] = -0.31139403219565177677282850411e+2,
			[15] = -0.93529243588444783865713862664e+1,
			[16] = 0.35816841486394083752465898540e+2,
		},
		{
			[1] = 0.19985053242002433820987653617e+2,
			[6] = -0.38703730874935176555105901742e+3,
			[7] = -0.18917813819516756882830838328e+3,
			[8] = 0.52780815920542364900561016686e+3,
			[9] = -0.11573902539959630126141871134e+2,
			[10] = 0.68812326946963000169666922661e+1,
			[11] = -0.10006050966910838403183860980e+1,
			[12] = 0.77771377980534432092869265740,
			[13] = -0.27782057523535084065932004339e+1,
			[14] = -0.60196695231264120758267380846e+2,
			[15] = 0.84320405506677161018159903784e+2,
			[16] = 0.11992291136182789328035130030e+2,
		},
		{
			[1] = -0.25693933462703749003312586129e+2,
			[6] = -0.15418974869023643374053993627e+3,
			[7] = -0.23152937917604549567536039109e+3,
			[8] = 0.35763911791061412378285349910e+3,
			[9] = 0.93405324183624310003907691704e+2,
			[10] = -0.37458323136451633156875139351e+2,
			[11] = 0.10409964950896230045147246184e+3,
			[12] = 0.29840293426660503123344363579e+2,
			[13] = -0.43533456590011143754432175058e+2,
			[14] = 0.96324553959188282948394950600e+2,
			[15] = -0.39177261675615439165231486172e+2,
			[16] = -0.14972683625798562581422125276e+3,
		},
	},
}

do
	-- The DOP853 tableau in the same form as the others (stages 1 .. 12).
	local a: { { number } } = { {} }
	for s = 2, 12 do
		local row = table.create(s - 1, 0) :: { number }
		for _, entry in D8.rows[s] do
			row[entry[1]] = entry[2]
		end
		a[s] = row
	end
	local b = table.create(12, 0) :: { number }
	for _, entry in D8.rows[13] do
		b[entry[1]] = entry[2]
	end
	local c = table.create(12, 0) :: { number }
	for s = 1, 12 do
		c[s] = D8.c[s]
	end
	TABLEAUX.dop853 = { name = "dop853", order = 8, eorder = 7, c = c, a = a, b = b, beta = 0 }
end

-- A Runge-Kutta engine: state, stages and one step of a tableau.
type Engine<T> = {
	S: Space<T>,
	f: OdeFunction<T>,
	tab: Tableau,
	t: number,
	y: T,
	k1: T,
	K: { T },
	work: T,
	nfev: number,
}

local function new_engine<T>(S: Space<T>, f: OdeFunction<T>, method: string, t0: number, y0: T): Engine<T>
	local tab = TABLEAUX[method]
	assert(tab ~= nil, `ode: unknown method {method}`)
	local y = S.copy(S.similar(y0), y0)
	local k1 = f(t0, y, S.similar(y0))
	return { S = S, f = f, tab = tab, t = t0, y = y, k1 = k1, K = {}, work = S.similar(y0), nfev = 1 }
end

-- The stages of a step of size h from (E.t, E.y); returns y_new (a new value) and, for
-- adaptive methods, f(t + h, y_new) in E.K[stages + 1]. Does not change E.t or E.y.
local function rk_stages<T>(E: Engine<T>, h: number): T
	local S, f, tab = E.S, E.f, E.tab
	local K = E.K
	local ns = #tab.c
	K[1] = E.k1
	for s = 2, ns do
		E.work = combine(S, E.work, E.y, h, K, tab.a[s], s - 1)
		K[s] = f(E.t + tab.c[s] * h, E.work, similar(S, E.y, if K[s] ~= E.k1 then K[s] else nil))
		E.nfev += 1
	end
	return combine(S, S.similar(E.y), E.y, h, K, tab.b, ns)
end

-- The new slope f(t + h, y_new) into K[ns + 1] (never the buffer holding k1).
local function end_slope<T>(E: Engine<T>, h: number, ynew: T): T
	local ns = #E.tab.c
	local slot = E.K[ns + 1]
	if slot == E.k1 then
		slot = nil
	end
	local fn = E.f(E.t + h, ynew, similar(E.S, ynew, slot))
	E.K[ns + 1] = fn
	E.nfev += 1
	return fn
end

-- The error norm of the step just computed (adaptive methods; stages 1 .. ns + 1 set).
local function step_error<T>(E: Engine<T>, h: number, ynew: T, atol: number, rtol: number): number
	local S, tab, K = E.S, E.tab, E.K
	if tab.name == "dop853" then
		local e5 = combine(S, S.similar(ynew), S.scale(S.similar(ynew), 0, ynew), h, K, D8.e5, 13)
		local e3c: { [number]: number } = {}
		for i = 1, 12 do
			e3c[i] = tab.b[i] - (D8.e3_corrections[i] or 0)
		end
		local e3 = combine(S, S.similar(ynew), S.scale(S.similar(ynew), 0, ynew), h, K, e3c, 12)
		local n5 = weighted(S, e5, E.y, ynew, atol, rtol)
		local n3 = weighted(S, e3, E.y, ynew, atol, rtol)
		if n5 == 0 and n3 == 0 then
			return 0
		end
		return n5 * n5 / math.sqrt(n5 * n5 + 0.01 * n3 * n3)
	end
	local e = combine(S, S.similar(ynew), S.scale(S.similar(ynew), 0, ynew), h, K, tab.e :: { number }, #tab.c + 1)
	return weighted(S, e, E.y, ynew, atol, rtol)
end

-- The continuous extension of the step just taken from (E.t, E.y) to (t + h, ynew), with
-- f(t + h, ynew) = fnew.
local function make_piece<T>(E: Engine<T>, h: number, ynew: T, fnew: T): Piece<T>
	local S, tab, K = E.S, E.tab, E.K
	local y0 = S.copy(S.similar(E.y), E.y)
	if tab.p then
		local p = tab.p
		local q: { T } = {}
		for j = 1, #p[1] do
			local col: { [number]: number } = {}
			for i = 1, #p do
				col[i] = p[i][j]
			end
			q[j] = combine(S, S.similar(ynew), S.scale(S.similar(ynew), 0, ynew), 1, K, col, #p)
		end
		return { t0 = E.t, h = h, y0 = y0, kind = "poly", q = q }
	elseif tab.name == "dop853" then
		-- Three more stages, then F1 .. F7.
		for s = 14, 16 do
			E.work = combine(S, E.work, E.y, h, K, (function(): { [number]: number }
				local row: { [number]: number } = {}
				for _, entry in D8.rows[s] do
					row[entry[1]] = entry[2]
				end
				return row
			end)(), s - 1)
			K[s] = E.f(E.t + D8.c[s] * h, E.work, similar(S, E.y, if K[s] ~= E.k1 then K[s] else nil))
			E.nfev += 1
		end
		local dy = S.sub(S.similar(ynew), ynew, E.y)
		local F2 = S.scale(S.similar(ynew), h, E.k1)
		F2 = S.sub(F2, F2, dy)
		local F3 = S.add(S.similar(ynew), fnew, E.k1)
		F3 = S.scale(F3, -h, F3)
		F3 = S.axpy(F3, 2, dy, F3)
		local q: { T } = { dy, F2, F3 }
		for r = 1, 4 do
			q[3 + r] = combine(S, S.similar(ynew), S.scale(S.similar(ynew), 0, ynew), h, K, D8.d[r], 16)
		end
		return { t0 = E.t, h = h, y0 = y0, kind = "dop853", q = q }
	end
	return {
		t0 = E.t,
		h = h,
		y0 = y0,
		kind = "hermite",
		q = { S.copy(S.similar(ynew), ynew), S.copy(S.similar(E.k1), E.k1), S.copy(S.similar(fnew), fnew) },
	}
end

-- Accept the step: move to (t + h, ynew) with slope fnew as the next k1. The buffer of
-- the old k1 is recycled for the next end slope.
local function accept<T>(E: Engine<T>, h: number, ynew: T, fnew: T)
	local ns = #E.tab.c
	local old = E.k1
	E.t += h
	E.y = ynew
	E.k1 = fnew
	E.K[ns + 1] = old
end

-- Hairer's initial step size (odecore.initial_step) from the engine's state.
local function initial_step<T>(E: Engine<T>, direction: number, span: number, atol: number, rtol: number, maxstep: number): number
	local h, used = core.initial_step(E.S, E.f, E.t, E.y, E.k1, direction, span, atol, rtol, maxstep, E.tab.eorder or E.tab.order)
	E.nfev += used
	return h
end

-- Steppers: { t, y, piece, stats, step(self, t_bound, want_piece) -> (ok, message?) }.
-- step takes one accepted step toward t_bound (never past it, and landing on it exactly
-- when it gets there), leaving the step's continuous extension in piece when want_piece
-- is set; a failure returns false and a message. The explicit stepper runs a tableau with
-- the PI controller; the stiff ones are in odestiff.
local function explicit_stepper<T>(S: Space<T>, f: OdeFunction<T>, t0: number, y0: T, o: OdeOptions<T>): any
	local method = o.method or "dp5"
	local E = new_engine(S, f, method, t0, y0)
	local tab = E.tab
	local adaptive = tab.eorder ~= nil
	local rtol, atol = o.rtol or 1e-3, o.atol or 1e-6
	local maxstep = o.max_step or math.huge
	local h: number? = if o.h then math.abs(o.h) else nil
	assert(adaptive or (h ~= nil and h > 0), `ode: the fixed-step method {method} needs options.h`)
	local errold = 1e-4
	local rejected_last = false
	local stats = { nfev = E.nfev, steps = 0, rejected = 0 }
	local st: any = { t = t0, y = E.y, piece = nil, stats = stats }
	function st.step(self: any, t_bound: number, want: boolean): (boolean, string?)
		local direction = if t_bound >= E.t then 1 else -1
		if h == nil then
			h = initial_step(E, direction, math.abs(t_bound - E.t), atol, rtol, maxstep)
		end
		while true do
			local hcur = h :: number
			local remaining = math.abs(t_bound - E.t)
			local hs = math.min(hcur, remaining)
			if remaining - hs <= 4 * EPS * math.abs(t_bound) then
				hs = remaining
			end
			if hs <= 16 * EPS * math.max(math.abs(E.t), 1) then
				stats.nfev = E.nfev
				return false, "the step size underflowed (a singularity, or very stiff?)"
			end
			local hd = direction * hs
			local ynew = rk_stages(E, hd)
			local fnew = end_slope(E, hd, ynew)
			if adaptive then
				local err = step_error(E, hd, ynew, atol, rtol)
				local alpha = 1 / ((tab.eorder :: number) + 1) - 0.75 * tab.beta
				if err ~= err then
					stats.nfev = E.nfev
					return false, "the solution became NaN"
				elseif err > 1 then
					h = hs * math.max(0.2, 0.9 * err ^ -alpha)
					rejected_last = true
					stats.rejected += 1
					continue
				end
				local factor = if err == 0 then 10 else 0.9 * err ^ -alpha * errold ^ tab.beta
				factor = math.clamp(factor, 0.2, if rejected_last then 1 else 10)
				errold = math.max(err, 1e-4)
				rejected_last = false
				-- A step cut short at t_bound keeps the controller's step for next time.
				if hs >= hcur then
					h = math.min(hs * factor, maxstep)
				end
			end
			local piece = if want then make_piece(E, hd, ynew, fnew) else nil
			accept(E, hd, ynew, fnew)
			if hs == remaining then
				E.t = t_bound
			end
			stats.steps += 1
			stats.nfev = E.nfev
			self.t, self.y, self.piece = E.t, E.y, piece
			return true, nil
		end
	end
	return st
end

local function make_stepper<T>(S: Space<T>, f: OdeFunction<T>, t0: number, y0: T, o: OdeOptions<T>): any
	local method = o.method or "dp5"
	if TABLEAUX[method] then
		return explicit_stepper(S, f, t0, y0, o)
	end
	assert(stiff.methods[method], `ode: unknown method {method}`)
	return stiff.stepper(S, f, t0, y0, o)
end

-- Solves y' = f(t, y), y(t0) = y0 over tspan = { t0, t1 } (t1 < t0 integrates
-- backward). options (all optional):
--   method   "dp5" (default), "bs3", "dop853", or a fixed-step method with options.h
--   rtol, atol  local error tolerances (default 1e-3 and 1e-6), adaptive methods
--   h        the step size for fixed-step methods (and the first step for adaptive ones)
--   max_step, max_steps (default 100000)
--   dense    keep the continuous extension for sol.at (default true)
--   t_eval   times (between t0 and t1, in order) at which to report the solution
--   events   a list of { g = function(t, y) -> number, terminal = boolean?,
--            direction = number? }: zeros of g are located on the dense output; direction
--            1 (or -1) counts only increasing (decreasing) crossings; a terminal event
--            stops the integration there.
--   jac, jac_band, jac_sparsity, linear, precond, krylov, mass  for the stiff methods:
--            the Jacobian df/dy (dense, banded or sparse; by finite differences without
--            jac), direct or Newton-Krylov linear algebra, and a constant mass matrix M
--            in M y' = f; see odestiff.luau
-- Returns { t, y (the steps), status ("success", "event" or "failed"), message, stats
-- { nfev, steps, rejected, and for stiff methods njev, nlu (Jacobians and
-- factorizations) and nlinear (GMRES iterations) }, events (a list of { t, y, index }), t_eval, y_eval, at }:
-- sol.at(t) evaluates the dense output at any t in the integrated range.
local function solve<T>(S: Space<T>, f: OdeFunction<T>, tspan: { number }, y0: T, options: OdeOptions<T>?): OdeSolution<T>
	local o: OdeOptions<T> = options or {}
	local t0, t1 = tspan[1], tspan[2]
	local direction = if t1 >= t0 then 1 else -1
	local st = make_stepper(S, f, t0, y0, o)
	local maxsteps = o.max_steps or 100000
	local dense = o.dense ~= false
	local events = o.events or {}
	local t_eval = o.t_eval
	local needpieces = dense or #events > 0 or t_eval ~= nil
	local ycur: T = S.copy(S.similar(y0), y0)
	local ts: { number } = { t0 }
	local ys: { T } = { ycur }
	local pieces: { Piece<T> } = {}
	local found: { { t: number, y: T, index: number } } = {}
	local yeval: { T } = {}
	local teval_next = 1
	local status, message = "success", "reached the end of the interval"
	-- Event function values at the current point.
	local gv: { number } = {}
	for i, ev in events do
		gv[i] = ev.g(t0, ycur)
	end
	if t_eval then
		-- Output times at t0 exactly.
		while t_eval[teval_next] and t_eval[teval_next] == t0 do
			yeval[teval_next] = S.copy(S.similar(ycur), ycur)
			teval_next += 1
		end
	end
	while direction * (t1 - st.t) > 0 do
		if st.stats.steps >= maxsteps then
			status, message = "failed", "too many steps (options.max_steps)"
			break
		end
		local told = st.t
		local ok, why = st:step(t1, needpieces)
		if not ok then
			status, message = "failed", why or "the step failed"
			break
		end
		local piece: Piece<T>? = st.piece
		local tnow: number, ynow: T = st.t, st.y
		-- Events in (told, tnow].
		local hit: { index: number, t: number }? = nil
		for i, ev in events do
			local gnew = ev.g(tnow, ynow)
			local gold = gv[i]
			local crosses = (gold < 0 and gnew >= 0) or (gold > 0 and gnew <= 0)
			local dir = ev.direction or 0
			if crosses and (dir == 0 or (dir > 0 and gnew > gold) or (dir < 0 and gnew < gold)) then
				local te = if gnew == 0
					then tnow
					else bracket_root(function(t: number): number
						return ev.g(t, piece_eval(S, piece :: Piece<T>, t))
					end, told, tnow, gold, gnew)
				table.insert(found, { t = te, y = piece_eval(S, piece :: Piece<T>, te), index = i })
				if ev.terminal and (hit == nil or direction * (te - (hit :: any).t) < 0) then
					hit = { index = i, t = te }
				end
			end
			gv[i] = gnew
		end
		if hit then
			-- Stop at the earliest terminal event: drop later events, cut the step there.
			local te = hit.t
			local kept = {}
			for _, ev in found do
				if direction * (ev.t - te) <= 0 then
					table.insert(kept, ev)
				end
			end
			found = kept
			ynow = piece_eval(S, piece :: Piece<T>, te)
			tnow = te
			status, message = "event", `terminal event {hit.index}`
		end
		if t_eval then
			while t_eval[teval_next] and direction * (t_eval[teval_next] - tnow) <= 0 do
				local te = t_eval[teval_next]
				yeval[teval_next] = if te == tnow then S.copy(S.similar(ynow), ynow) else piece_eval(S, piece :: Piece<T>, te)
				teval_next += 1
			end
		end
		if dense and piece then
			table.insert(pieces, piece)
		end
		table.insert(ts, tnow)
		table.insert(ys, S.copy(S.similar(ynow), ynow))
		if hit then
			break
		end
	end
	local stats = table.clone(st.stats)
	local sol: OdeSolution<T> = {
		t = ts,
		y = ys,
		status = status,
		message = message,
		stats = stats,
		events = found,
		t_eval = t_eval,
		y_eval = if t_eval then yeval else nil,
		at = function(t: number, out: T?): T
			assert(dense, "sol.at needs options.dense")
			local lo, hi = ts[1], ts[#ts]
			assert(direction * (t - lo) >= -4 * EPS * math.abs(lo) and direction * (hi - t) >= -4 * EPS * math.abs(hi), "sol.at: t is outside the integrated range")
			if #pieces == 0 then
				return S.copy(similar(S, ys[1], out), ys[1])
			end
			-- The step containing t.
			local a, b = 1, #pieces
			while b > a do
				local mid = (a + b) // 2
				if direction * (t - (pieces[mid].t0 + pieces[mid].h)) > 0 then
					a = mid + 1
				else
					b = mid
				end
			end
			return piece_eval(S, pieces[a], t, out)
		end,
	}
	return sol
end

-- An integrator: a state (it.t, it.y) you advance yourself, for simulations that step
-- by a frame time. options as for solve (method, rtol, atol, max_step; h is the step
-- for fixed-step methods' advance). it:step(h) takes exactly one step of size h (for an
-- adaptive method, its main formula without error control); it:advance(dt) moves exactly
-- dt ahead: adaptive methods take error-controlled steps, remembering the step size for
-- next time; fixed-step methods take steps of options.h (or one step of dt). Both return
-- it.y. Stiff methods only advance.
local function integrator<T>(S: Space<T>, f: OdeFunction<T>, t0: number, y0: T, options: OdeOptions<T>?): any
	local o: OdeOptions<T> = options or {}
	local method = o.method or "dp5"
	if not TABLEAUX[method] then
		local st = make_stepper(S, f, t0, y0, o)
		local it: any = { t = t0, y = st.y, stats = st.stats }
		function it.step(self: any, _: number): T
			error(`ode integrator: step(h) needs an explicit method; use advance with {method}`)
		end
		function it.advance(self: any, dt: number): T
			local target = st.t + dt
			while st.t ~= target do
				local ok, why = st:step(target, false)
				assert(ok, `ode integrator: {why}`)
			end
			it.t, it.y = st.t, st.y
			return st.y
		end
		return it
	end
	local E = new_engine(S, f, method, t0, y0)
	local adaptive = E.tab.eorder ~= nil
	local rtol, atol = o.rtol or 1e-3, o.atol or 1e-6
	local maxstep = o.max_step or math.huge
	local it: any = { t = t0, y = E.y, h = o.h, stats = { nfev = 1, steps = 0, rejected = 0 } }
	local errold = 1e-4
	local function sync()
		it.t, it.y = E.t, E.y
		it.stats.nfev = E.nfev
	end
	function it.step(self: any, h: number): T
		local ynew = rk_stages(E, h)
		local fnew = end_slope(E, h, ynew)
		accept(E, h, ynew, fnew)
		it.stats.steps += 1
		sync()
		return E.y
	end
	function it.advance(self: any, dt: number): T
		if dt == 0 then
			return E.y
		end
		local direction = if dt > 0 then 1 else -1
		local target = E.t + dt
		if not adaptive then
			local hfix = if o.h then math.abs(o.h) else math.abs(dt)
			while direction * (target - E.t) > 4 * EPS * math.abs(target) do
				local hs = math.min(hfix, math.abs(target - E.t))
				self:step(direction * hs)
			end
			E.t = target
			sync()
			return E.y
		end
		local h = it.h or initial_step(E, direction, math.abs(dt), atol, rtol, maxstep)
		local rejected_last = false
		local tab = E.tab
		local q = (tab.eorder :: number) + 1
		local alpha = 1 / q - 0.75 * tab.beta
		while direction * (target - E.t) > 4 * EPS * math.abs(target) do
			local hs = math.min(h, math.abs(target - E.t))
			assert(hs > 16 * EPS * math.max(math.abs(E.t), 1), "ode integrator: the step size underflowed")
			local hd = direction * hs
			local ynew = rk_stages(E, hd)
			local fnew = end_slope(E, hd, ynew)
			local err = step_error(E, hd, ynew, atol, rtol)
			assert(err == err, "ode integrator: the solution became NaN")
			if err <= 1 then
				local factor = if err == 0 then 10 else 0.9 * err ^ -alpha * errold ^ tab.beta
				factor = math.clamp(factor, 0.2, if rejected_last then 1 else 10)
				errold = math.max(err, 1e-4)
				accept(E, hd, ynew, fnew)
				it.stats.steps += 1
				-- Keep the controller's step for next time (unless this one was cut short).
				if hs >= h then
					h = math.min(hs * factor, maxstep)
				end
				rejected_last = false
			else
				h = hs * math.max(0.2, 0.9 * err ^ -alpha)
				rejected_last = true
				it.stats.rejected += 1
			end
		end
		E.t = target
		it.h = h
		sync()
		return E.y
	end
	return it
end

-- Mechanics: symplectic integrators for q'' = a(t, q)
--
-- Splitting methods for separable Hamiltonians (kinetic energy in v, potential in q):
-- each step is a sequence of drifts q += c h v and kicks v += c h a(t, q). They are
-- symplectic and time-reversible, so on conservative problems the energy error stays
-- bounded (it oscillates) over arbitrarily long runs instead of drifting as it does with
-- Runge-Kutta methods: orbits, springs, rigid chains, game physics.
--   "verlet"    velocity Verlet / leapfrog, order 2, one force evaluation per step
--   "yoshida4"  Yoshida's triple-jump composition of Verlet, order 4, 3 per step
--   "pefrl"     Omelyan, Mryglod and Folk's position-extended Forest-Ruth-like method,
--               order 4 with an error constant about 100 times smaller, 4 per step
--   "yoshida6"  Yoshida's composition (solution A), order 6, 7 per step
-- The acceleration may depend on t but not on v (velocity-dependent forces such as drag
-- break the structure; use ode.solve for those).

type Op = { drift: boolean, c: number }

-- The operations of a composition of velocity Verlet steps (kick-drift-kick) with
-- weights w, adjacent kicks merged.
local function compose(w: { number }): { Op }
	local ops: { Op } = {}
	local pending = 0
	for _, wi in w do
		pending += wi / 2
		table.insert(ops, { drift = false, c = pending })
		table.insert(ops, { drift = true, c = wi })
		pending = wi / 2
	end
	table.insert(ops, { drift = false, c = pending })
	return ops
end

local SPLITTINGS: { [string]: { order: number, ops: { Op } } } = {}
do
	SPLITTINGS.verlet = { order = 2, ops = compose({ 1 }) }
	local cr = 2 ^ (1 / 3)
	local w1 = 1 / (2 - cr)
	SPLITTINGS.yoshida4 = { order = 4, ops = compose({ w1, -cr * w1, w1 }) }
	-- Yoshida (1990), sixth order, solution A.
	local y1, y2, y3 = -1.17767998417887, 0.235573213359357, 0.784513610477560
	local y0 = 1 - 2 * (y1 + y2 + y3)
	SPLITTINGS.yoshida6 = { order = 6, ops = compose({ y3, y2, y1, y0, y1, y2, y3 }) }
	-- Omelyan, Mryglod and Folk (2002), PEFRL.
	local xi, lambda, chi = 0.1786178958448091, -0.2123418310626054, -0.06626458266981849
	SPLITTINGS.pefrl = {
		order = 4,
		ops = {
			{ drift = true, c = xi },
			{ drift = false, c = (1 - 2 * lambda) / 2 },
			{ drift = true, c = chi },
			{ drift = false, c = lambda },
			{ drift = true, c = 1 - 2 * (chi + xi) },
			{ drift = false, c = lambda },
			{ drift = true, c = chi },
			{ drift = false, c = (1 - 2 * lambda) / 2 },
			{ drift = true, c = xi },
		},
	}
end

type Mech<T> = {
	S: Space<T>,
	accel: OdeFunction<T>,
	ops: { Op },
	t: number,
	q: T,
	v: T,
	a: T,
	valid: boolean,
	nfev: number,
}

local function new_mech<T>(S: Space<T>, accel: OdeFunction<T>, method: string, t0: number, q0: T, v0: T): Mech<T>
	local sp = SPLITTINGS[method]
	assert(sp ~= nil, `ode: unknown mechanics method {method}`)
	return {
		S = S,
		accel = accel,
		ops = sp.ops,
		t = t0,
		q = S.copy(S.similar(q0), q0),
		v = S.copy(S.similar(v0), v0),
		a = S.similar(q0),
		valid = false,
		nfev = 0,
	}
end

-- One step of size h (negative h runs backward: the methods are time-reversible).
local function mech_step<T>(M: Mech<T>, h: number)
	local S = M.S
	for _, op in M.ops do
		if op.drift then
			M.q = S.axpy(M.q, op.c * h, M.v, M.q)
			M.t += op.c * h
			M.valid = false
		else
			if not M.valid then
				M.a = M.accel(M.t, M.q, M.a)
				M.nfev += 1
				M.valid = true
			end
			M.v = S.axpy(M.v, op.c * h, M.a, M.v)
		end
	end
end

-- The acceleration at the current state (cached).
local function mech_accel<T>(M: Mech<T>): T
	if not M.valid then
		M.a = M.accel(M.t, M.q, M.a)
		M.nfev += 1
		M.valid = true
	end
	return M.a
end

-- A mechanics integrator: positions it.q and velocities it.v (in the same space) at
-- time it.t, advanced by it:step(h) (one step) or it:advance(dt) (steps of options.h, or
-- one step of dt). options.method is "verlet" (default), "yoshida4", "pefrl" or
-- "yoshida6"; accel(t, q, out) returns the acceleration.
local function mechanics<T>(S: Space<T>, accel: OdeFunction<T>, t0: number, q0: T, v0: T, options: { method: string?, h: number? }?): any
	local o: { method: string?, h: number? } = options or {}
	local M = new_mech(S, accel, o.method or "verlet", t0, q0, v0)
	local it: any = { t = t0, q = M.q, v = M.v, stats = { nfev = 0, steps = 0 } }
	local function sync()
		it.t, it.q, it.v = M.t, M.q, M.v
		it.stats.nfev = M.nfev
	end
	function it.step(self: any, h: number)
		mech_step(M, h)
		it.stats.steps += 1
		sync()
		return M.q, M.v
	end
	function it.advance(self: any, dt: number)
		local target = M.t + dt
		local direction = if dt >= 0 then 1 else -1
		local hfix = if o.h then math.abs(o.h :: number) else math.abs(dt)
		while direction * (target - M.t) > 4 * EPS * math.max(math.abs(target), 1) do
			self:step(direction * math.min(hfix, math.abs(target - M.t)))
		end
		M.t = target
		sync()
		return M.q, M.v
	end
	sync()
	return it
end

-- Integrates q'' = accel(t, q) over tspan with fixed steps options.h (required), by a
-- symplectic method (see mechanics). Returns { t, q, v (the steps), stats, at(t) ->
-- (q, v) (cubic Hermite interpolation, from the velocities and accelerations at the step
-- ends), and q_eval, v_eval at options.t_eval }.
local function solve_mechanics<T>(S: Space<T>, accel: OdeFunction<T>, tspan: { number }, q0: T, v0: T, options: { method: string?, h: number?, t_eval: { number }? }): any
	local o = options
	assert(o and o.h and o.h > 0, "ode.solve_mechanics needs options.h")
	local t0, t1 = tspan[1], tspan[2]
	local direction = if t1 >= t0 then 1 else -1
	local M = new_mech(S, accel, o.method or "verlet", t0, q0, v0)
	local ts: { number } = { t0 }
	local qs: { T } = { S.copy(S.similar(M.q), M.q) }
	local vs: { T } = { S.copy(S.similar(M.v), M.v) }
	local as: { T } = { S.copy(S.similar(M.q), mech_accel(M)) }
	local steps = 0
	while direction * (t1 - M.t) > 4 * EPS * math.max(math.abs(t1), 1) do
		local hs = math.min(o.h :: number, math.abs(t1 - M.t))
		mech_step(M, direction * hs)
		steps += 1
		table.insert(ts, M.t)
		table.insert(qs, S.copy(S.similar(M.q), M.q))
		table.insert(vs, S.copy(S.similar(M.v), M.v))
		table.insert(as, S.copy(S.similar(M.q), mech_accel(M)))
	end
	local function at(t: number): (T, T)
		local n = #ts
		if n == 1 then
			return S.copy(S.similar(qs[1]), qs[1]), S.copy(S.similar(vs[1]), vs[1])
		end
		local a, b = 1, n - 1
		while b > a do
			local mid = (a + b) // 2
			if direction * (t - ts[mid + 1]) > 0 then
				a = mid + 1
			else
				b = mid
			end
		end
		local hh = ts[a + 1] - ts[a]
		local Pq: Piece<T> = { t0 = ts[a], h = hh, y0 = qs[a], kind = "hermite", q = { qs[a + 1], vs[a], vs[a + 1] } }
		local Pv: Piece<T> = { t0 = ts[a], h = hh, y0 = vs[a], kind = "hermite", q = { vs[a + 1], as[a], as[a + 1] } }
		return piece_eval(S, Pq, t), piece_eval(S, Pv, t)
	end
	local qe: { T }?, ve: { T }? = nil, nil
	if o.t_eval then
		local qq, vv = {}, {}
		for i, te in o.t_eval do
			qq[i], vv[i] = at(te)
		end
		qe, ve = qq, vv
	end
	return { t = ts, q = qs, v = vs, at = at, stats = { nfev = M.nfev, steps = steps }, t_eval = o.t_eval, q_eval = qe, v_eval = ve }
end

-- The names of the available methods, and whether each is adaptive.
local function methods(): { [string]: boolean }
	local out = {}
	for name, tab in TABLEAUX do
		out[name] = tab.eorder ~= nil
	end
	for name in stiff.methods do
		out[name] = true
	end
	return out
end

return {
	solve = solve,
	integrator = integrator,
	methods = methods,
	mechanics = mechanics,
	solve_mechanics = solve_mechanics,
	jacobian = stiff.jacobian,
}
