--!strict
--!native
--!optimize 2

-- Linear algebra made simple: solve equations, find eigenvalues and so on, without choosing
-- methods or storage. Real numbers only.
--
-- Matrices can be plain tables of rows, { { 1, 2 }, { 3, 4 } }, or the library's own:
-- an f64 Matrix (citrate.f64.matrix) or a sparse matrix (linalg.sparse, citrate.f64.sparse).
-- Vectors can be plain tables { 1, 2 } or f64 arrays. Answers come back in the form the
-- question came in (tables for tables).
--
-- Each function checks its input (shapes, numbers that are not finite, symmetry within a
-- tolerance), picks a method, and returns an info table saying which method it used and how
-- accurate the answer is. Nothing here is a new algorithm: each function calls the
-- library's own (named in its comment), which are faster to call directly when the
-- choices are known in advance. These ones favor safety over speed: they copy their input,
-- check their answer, and refine it when that helps.

local types = require("./types")
local f64 = require("./f64")
local krylov = require("./krylov")

type Matrix = types.Matrix
type Sparse = types.Sparse

export type Info = {
	method: string, -- the method used
	residual: number?, -- how far off the answer is, relative (0 is exact)
	warning: string?, -- set when the answer may be inaccurate, and why
	[string]: any,
}

-- the largest dense matrix built from a sparse one (entries): 4 million, 32 MB
local MAX_DENSE = 4e6

-- Checks and conversions

local function fail(who: string, message: string): never
	error("linalg." .. who .. ": " .. message, 3)
end

local function finite(x: number): boolean
	return x == x and x ~= math.huge and x ~= -math.huge
end

local function is_sparse(A: any): boolean
	return type(A) == "table" and A.colptr ~= nil
end

local function is_matrix(A: any): boolean
	return type(A) == "table" and A.data ~= nil and A.rows ~= nil
end

-- A table of rows as an f64 Matrix, checked.
local function from_rows(rows: any, who: string): Matrix
	if type(rows) ~= "table" or #rows == 0 or type(rows[1]) ~= "table" then
		fail(who, "expected a matrix: a table of rows, such as { { 1, 2 }, { 3, 4 } }")
	end
	local m, n = #rows, #rows[1]
	if n == 0 then
		fail(who, "the matrix has an empty row")
	end
	local A = f64.matrix(m, n)
	for i = 1, m do
		local row = rows[i]
		if type(row) ~= "table" or #row ~= n then
			fail(who, string.format("row %d has %d entries where row 1 has %d", i, if type(row) == "table" then #row else 0, n))
		end
		for j = 1, n do
			local v = row[j]
			if type(v) ~= "number" or not finite(v) then
				fail(who, string.format("entry (%d, %d) is %s, not a finite number", i, j, tostring(v)))
			end
			buffer.writef64(A.data, 8 * ((j - 1) * m + i - 1), v)
		end
	end
	return A
end

local function check_values(values: buffer, count: number, who: string)
	for k = 0, count - 1 do
		if not finite(buffer.readf64(values, 8 * k)) then
			fail(who, "the matrix holds a NaN or an infinity")
		end
	end
end

-- The matrix A as "dense" (an f64 Matrix, never one the caller still uses: a copy) or
-- "sparse" (A itself), with its sizes.
local function matrix_of(A: any, who: string): (string, any, number, number)
	if is_sparse(A) then
		local S = A :: Sparse
		assert(buffer.len(S.values) == 8 * buffer.readu32(S.colptr, 4 * S.cols), "")
		check_values(S.values, buffer.readu32(S.colptr, 4 * S.cols), who)
		return "sparse", S, S.rows, S.cols
	elseif is_matrix(A) then
		local M = A :: Matrix
		if buffer.len(M.data) ~= 8 * M.rows * M.cols then
			fail(who, "expected an f64 matrix (this one holds another number type)")
		end
		check_values(M.data, M.rows * M.cols, who)
		local C = f64.matrix(M.rows, M.cols)
		buffer.copy(C.data, 0, M.data)
		return "dense", C, M.rows, M.cols
	end
	local D = from_rows(A, who)
	return "dense", D, D.rows, D.cols
end

-- A sparse matrix as a dense one, if that is not too large.
local function densify(S: Sparse, who: string): Matrix
	if S.rows * S.cols > MAX_DENSE then
		fail(who, string.format("this needs a dense %d by %d matrix, too large (over %d entries)", S.rows, S.cols, MAX_DENSE))
	end
	return f64.sparse_matrix(S)
end

-- The vector b (length n) as an f64 array (a copy) and whether it was a table.
local function vector_of(b: any, n: number, who: string): (buffer, boolean)
	if type(b) == "buffer" then
		if buffer.len(b) ~= 8 * n then
			fail(who, string.format("the vector has %d entries where %d are needed", buffer.len(b) // 8, n))
		end
		check_values(b, n, who)
		local c = buffer.create(8 * n)
		buffer.copy(c, 0, b)
		return c, false
	end
	if type(b) ~= "table" or #b ~= n then
		fail(who, string.format("expected a vector of %d numbers", n))
	end
	local c = buffer.create(8 * n)
	for i = 1, n do
		local v = b[i]
		if type(v) ~= "number" or not finite(v) then
			fail(who, string.format("entry %d is %s, not a finite number", i, tostring(v)))
		end
		buffer.writef64(c, 8 * (i - 1), v)
	end
	return c, true
end

local function vector_out(x: buffer, as_table: boolean): any
	return if as_table then f64.totable(x) else x
end

-- A Matrix as a table of rows.
local function rows_of(M: Matrix): { { number } }
	local t = table.create(M.rows) :: { { number } }
	for i = 1, M.rows do
		local row = table.create(M.cols) :: { number }
		for j = 1, M.cols do
			row[j] = buffer.readf64(M.data, 8 * ((j - 1) * M.rows + i - 1))
		end
		t[i] = row
	end
	return t
end

-- Whether A equals its transpose to within tol times its largest entry.
local function symmetric(kind: string, A: any, tol: number): boolean
	if kind == "dense" then
		local M = A :: Matrix
		local n = M.rows
		if M.cols ~= n then
			return false
		end
		local big, diff = 0, 0
		for j = 0, n - 1 do
			for i = 0, n - 1 do
				local a = buffer.readf64(M.data, 8 * (j * n + i))
				big = math.max(big, math.abs(a))
				if i > j then
					diff = math.max(diff, math.abs(a - buffer.readf64(M.data, 8 * (i * n + j))))
				end
			end
		end
		return diff <= tol * big
	end
	local S = A :: Sparse
	if S.rows ~= S.cols then
		return false
	end
	local T = f64.sparse_transpose(S)
	local negT: Sparse = { rows = T.rows, cols = T.cols, colptr = T.colptr, rowind = T.rowind, values = f64.scale(nil, -1, T.values) }
	local D = f64.sparse_add(S, negT)
	local big = f64.amax(S.values)
	local diff = if f64.length(D.values) > 0 then f64.amax(D.values) else 0
	return diff <= tol * big
end

local function times(kind: string, A: any, x: buffer): buffer
	return if kind == "sparse" then f64.sparse_matvec(nil, A, x) else f64.matvec(nil, A, x)
end

local function relative_residual(kind: string, A: any, x: buffer, b: buffer): number
	local r = f64.sub(nil, times(kind, A, x), b)
	local nb = f64.norm(b)
	return if nb > 0 then f64.norm(r) / nb else f64.norm(r)
end

-- Building and reading matrices

-- A matrix from a table of rows, checked: linalg.matrix({ { 1, 2 }, { 3, 4 } }). The other
-- functions take tables directly; this is for building a matrix once and using it often.
local function matrix(rows: { { number } }): Matrix
	return from_rows(rows, "matrix")
end

-- A sparse matrix (one storing only its nonzero entries) of the given size from a list of
-- entries { row, column, value }: linalg.sparse(3, 3, { { 1, 1, 4 }, { 2, 1, -1 }, ... }).
-- Repeated positions are added together.
local function sparse(rows: number, cols: number, entries: { { number } }): Sparse
	if type(rows) ~= "number" or type(cols) ~= "number" or rows < 1 or cols < 1 or rows % 1 ~= 0 or cols % 1 ~= 0 then
		fail("sparse", "the sizes must be positive whole numbers")
	end
	local ii = table.create(#entries) :: { number }
	local jj = table.create(#entries) :: { number }
	local vv = table.create(#entries) :: { number }
	for k, e in entries do
		local i, j, v = e[1], e[2], e[3]
		if type(i) ~= "number" or type(j) ~= "number" or i % 1 ~= 0 or j % 1 ~= 0 or i < 1 or i > rows or j < 1 or j > cols then
			fail("sparse", string.format("entry %d: position (%s, %s) is outside the %d by %d matrix", k, tostring(i), tostring(j), rows, cols))
		end
		if type(v) ~= "number" or not finite(v) then
			fail("sparse", string.format("entry %d: the value %s is not a finite number", k, tostring(v)))
		end
		ii[k], jj[k], vv[k] = i, j, v
	end
	return f64.sparse(rows, cols, ii, jj, f64.from(vv))
end

-- A matrix as a table of rows (a sparse one as its list of entries { row, column, value }),
-- or a vector as a table of numbers.
local function totable(A: any): any
	if type(A) == "buffer" then
		return f64.totable(A)
	elseif is_sparse(A) then
		local S = A :: Sparse
		local out = {}
		for j = 0, S.cols - 1 do
			for p = buffer.readu32(S.colptr, 4 * j), buffer.readu32(S.colptr, 4 * j + 4) - 1 do
				table.insert(out, { buffer.readu32(S.rowind, 4 * p) + 1, j + 1, buffer.readf64(S.values, 8 * p) })
			end
		end
		return out
	elseif is_matrix(A) then
		return rows_of(A :: Matrix)
	end
	return A
end

-- Solving equations

-- The solution x of A x = b for a square matrix A. Picks the method: for a symmetric
-- matrix, Cholesky factorization (f64.chol, f64.sparse_chol), falling back to LU if it is
-- not positive definite; otherwise LU with pivoting (f64.lu, f64.sparse_lu). The answer is
-- checked (info.residual: |A x - b| / |b|) and refined by up to two steps of iterative
-- refinement when it can be improved. For a dense matrix info.rcond estimates how close A
-- is to singular (1: far, 0: singular), and info.warning is set when the answer may be
-- inaccurate. Errors if A is singular.
local function solve(A: any, b: any): (any, Info)
	local kind, M, m, n = matrix_of(A, "solve")
	if m ~= n then
		fail("solve", string.format("the matrix is %d by %d; solve needs a square one (see linalg.lstsq)", m, n))
	end
	local rhs, as_table = vector_of(b, n, "solve")
	local info: Info = { method = "" }
	local solver: (buffer) -> buffer
	local sym = symmetric(kind, M, 1e-12)
	if kind == "sparse" then
		local F: any = nil
		if sym then
			local C = f64.sparse_chol(M)
			if C.definite then
				info.method = "sparse Cholesky"
				F = C
				solver = function(r: buffer): buffer
					return f64.sparse_chol_solve(nil, C, r)
				end
			end
		end
		if F == nil then
			local LU = f64.sparse_lu(M)
			if LU.singular then
				fail("solve", "the matrix is singular (some column has no usable pivot)")
			end
			info.method = if sym then "sparse LU (symmetric, not positive definite)" else "sparse LU"
			solver = function(r: buffer): buffer
				return f64.sparse_lu_solve(nil, LU, r)
			end
		end
	else
		local done = false
		if sym then
			local C = f64.chol_copy(M)
			if C.definite then
				info.method = "Cholesky"
				done = true
				solver = function(r: buffer): buffer
					return f64.chol_solve(nil, C, r)
				end
			end
		end
		if not done then
			local LU = f64.lu_copy(M)
			if LU.singular then
				fail("solve", "the matrix is singular")
			end
			info.method = if sym then "LU (symmetric, not positive definite)" else "LU"
			solver = function(r: buffer): buffer
				return f64.lu_solve(nil, LU, r)
			end
		end
		info.rcond = f64.rcond(M)
	end
	local x = solver(rhs)
	local res = relative_residual(kind, M, x, rhs)
	info.refinements = 0
	-- iterative refinement: solve for the correction from the residual
	while res > 1e-14 and info.refinements < 2 do
		local r = f64.sub(nil, rhs, times(kind, M, x))
		local x2 = f64.add(nil, x, solver(r))
		local res2 = relative_residual(kind, M, x2, rhs)
		if not (res2 < res) then
			break
		end
		x, res = x2, res2
		info.refinements += 1
	end
	info.residual = res
	if info.rcond and info.rcond < 1e-12 then
		info.warning = string.format("the matrix is nearly singular (rcond %.1e): the answer may be inaccurate", info.rcond)
	elseif res > 1e-8 then
		info.warning = string.format("the residual is large (%.1e): the matrix may be nearly singular", res)
	end
	return vector_out(x, as_table), info
end

-- The least-squares solution of A x = b for any shape of A: the x making |A x - b| smallest
-- (for a tall A, the best fit; for a wide or rank-deficient one, a solution with as many
-- zeros as the rank requires). Column-pivoted QR (f64.qrp, f64.qrp_solve). info.rank is the
-- numerical rank and info.residual |A x - b| / |b|. A sparse A is made dense.
local function lstsq(A: any, b: any): (any, Info)
	local kind, M, m, n = matrix_of(A, "lstsq")
	if kind == "sparse" then
		M = densify(M, "lstsq")
	end
	local rhs, as_table = vector_of(b, m, "lstsq")
	local F = f64.qrp_copy(M)
	local x, _, rank = f64.qrp_solve(nil, F, rhs)
	local info: Info = { method = "pivoted QR", rank = rank, residual = relative_residual("dense", M, x, rhs) }
	if rank < math.min(m, n) then
		info.warning = string.format("the matrix has rank %d, less than its %d columns or rows: the solution is not unique", rank, math.min(m, n))
	end
	return vector_out(x, as_table), info
end

-- Eigenvalues

export type EigOptions = { count: number?, largest: boolean? }

-- Gershgorin bounds: every eigenvalue of S lies in [lo, hi].
local function gershgorin(S: Sparse): (number, number)
	local n = S.rows
	local radius, diag = table.create(n, 0), table.create(n, 0)
	for j = 0, S.cols - 1 do
		for p = buffer.readu32(S.colptr, 4 * j), buffer.readu32(S.colptr, 4 * j + 4) - 1 do
			local i = buffer.readu32(S.rowind, 4 * p)
			local v = buffer.readf64(S.values, 8 * p)
			if i == j then
				diag[i + 1] += v
			else
				radius[i + 1] += math.abs(v)
			end
		end
	end
	local lo, hi = math.huge, -math.huge
	for i = 1, n do
		lo = math.min(lo, diag[i] - radius[i])
		hi = math.max(hi, diag[i] + radius[i])
	end
	return lo, hi
end

-- The eigenvalues (ascending) and eigenvectors of a symmetric matrix. options.count asks for
-- only the smallest few (options.largest: the largest few). A dense matrix: all of them by
-- f64.eigh. A sparse one: count of them (default 6) by Lanczos (krylov.eigsh) on the inverse
-- of A shifted past its eigenvalues (sparse Cholesky of A - s I, s from Gershgorin's bound),
-- which finds the end of the spectrum quickly; info.residual is the largest |A v - lambda v|
-- relative to |A|'s scale. A sparse matrix of 100 rows or fewer, or one asked for more than
-- a quarter of its eigenvalues, is made dense instead (f64.eigh costs n^3: 46 ms at 200 rows,
-- where Lanczos found 4 eigenvalues of 1521 rows in 22). Errors if A is not symmetric: a nonsymmetric matrix can have
-- complex eigenvalues (see f64.eig). Returns the values (a table) and the vectors (a table
-- of vectors, each in the form A came in).
local function eig(A: any, options: EigOptions?): ({ number }, { any }, Info)
	local o: EigOptions = options or {}
	local kind, M, m, n = matrix_of(A, "eig")
	if m ~= n then
		fail("eig", string.format("the matrix is %d by %d; eigenvalues need a square one", m, n))
	end
	if not symmetric(kind, M, 1e-10) then
		fail("eig", "the matrix is not symmetric: its eigenvalues may be complex (use citrate.f64.eig)")
	end
	local as_table = not (is_sparse(A) or is_matrix(A))
	local values: { number } = {}
	local vectors: { any } = {}
	local info: Info = { method = "" }
	if kind == "sparse" and (n <= 100 or 4 * (o.count or 6) > n) then
		kind, M = "dense", densify(M, "eig")
	end
	if kind == "dense" then
		local E = f64.eigh_copy(M, true)
		local V = E.vectors :: Matrix
		local count = math.min(o.count or n, n)
		local first = if o.largest then n - count + 1 else 1
		for k = first, first + count - 1 do
			table.insert(values, f64.get(E.values, k))
			local v = f64.create(n)
			buffer.copy(v, 0, V.data, 8 * (k - 1) * n, 8 * n)
			table.insert(vectors, vector_out(v, as_table))
		end
		info.method = "dense symmetric eigensolver"
	else
		local S = M :: Sparse
		local count = math.min(o.count or 6, n)
		local lo, hi = gershgorin(S)
		local scale = math.max(math.abs(lo), math.abs(hi), 1e-300)
		-- shifted past the spectrum (by a little more than its width's millionth), so the
		-- shifted matrix is positive definite: B = A - s I (smallest) or s I - A (largest)
		local s = if o.largest then hi + 1e-6 * scale else lo - 1e-6 * scale
		local sign = if o.largest then -1 else 1
		local ii, jj, vv = {}, {}, {}
		for j = 0, S.cols - 1 do
			for p = buffer.readu32(S.colptr, 4 * j), buffer.readu32(S.colptr, 4 * j + 4) - 1 do
				table.insert(ii, buffer.readu32(S.rowind, 4 * p) + 1)
				table.insert(jj, j + 1)
				table.insert(vv, sign * buffer.readf64(S.values, 8 * p))
			end
		end
		for i = 1, n do
			table.insert(ii, i)
			table.insert(jj, i)
			table.insert(vv, -sign * s)
		end
		local B = f64.sparse(n, n, ii, jj, f64.from(vv))
		local C = f64.sparse_chol(B)
		if not C.definite then
			fail("eig", "the shifted matrix was not positive definite (is the matrix symmetric?)")
		end
		local v0 = f64.create(n)
		for i = 1, n do
			f64.set(v0, i, 1 + 0.1 * math.sin(i))
		end
		local mu, vecs = krylov.eigsh(f64.space, function(out: buffer, v: buffer): buffer
			return f64.sparse_chol_solve(out, C, v)
		end, v0, count, { which = "LA", tol = 1e-10, ncv = math.max(2 * count + 10, 30), maxiter = 2000 })
		-- B's eigenvalues are 1 / mu: A's are s + 1 / mu (smallest) or s - 1 / mu (largest)
		local order: { any } = {}
		for k = 1, f64.length(mu) do
			order[k] = { s + sign / f64.get(mu, k), vecs[k] }
		end
		table.sort(order, function(a: any, b: any): boolean
			return a[1] < b[1]
		end)
		local worst = 0
		for _, e: any in order do
			table.insert(values, e[1])
			local r = f64.axpy(nil, -e[1], e[2], f64.sparse_matvec(nil, S, e[2]))
			worst = math.max(worst, f64.norm(r) / f64.norm(e[2]))
			table.insert(vectors, vector_out(e[2], false))
		end
		info.method = "Lanczos, shift-invert (sparse Cholesky)"
		info.residual = worst / scale
		if #values < count then
			info.warning = string.format("only %d of the %d eigenvalues asked for converged", #values, count)
		end
	end
	return values, vectors, info
end

-- Singular values and other properties

-- The singular values (descending) and the singular vectors: A = U diag(s) V^T, as tables
-- of columns (u_k, v_k: vectors in the form A came in). f64.svd; a sparse A is made dense.
local function svd(A: any): ({ number }, { any }, { any }, Info)
	local kind, M = matrix_of(A, "svd")
	if kind == "sparse" then
		M = densify(M, "svd")
	end
	local as_table = not (is_sparse(A) or is_matrix(A))
	local F = f64.svd_copy(M, true)
	local U, V = F.u :: Matrix, F.v :: Matrix
	local s, us, vs = {}, {}, {}
	for k = 1, f64.length(F.values) do
		s[k] = f64.get(F.values, k)
		local u, v = f64.create(U.rows), f64.create(V.rows)
		buffer.copy(u, 0, U.data, 8 * (k - 1) * U.rows, 8 * U.rows)
		buffer.copy(v, 0, V.data, 8 * (k - 1) * V.rows, 8 * V.rows)
		us[k], vs[k] = vector_out(u, as_table), vector_out(v, as_table)
	end
	return s, us, vs, { method = "singular value decomposition" }
end

-- The determinant of a square matrix (f64.det; a sparse one from its sparse LU factors: the
-- product of the pivots, signed by the row and column permutations). May overflow to
-- infinity for large matrices.
local function det(A: any): number
	local kind, M, m, n = matrix_of(A, "det")
	if m ~= n then
		fail("det", string.format("the matrix is %d by %d; a determinant needs a square one", m, n))
	end
	if kind == "dense" then
		return f64.det(M)
	end
	local F = f64.sparse_lu(M, { tol = 1 }) :: any
	if F.singular then
		return 0
	end
	-- the sign of a permutation: (-1)^(n - number of cycles)
	local function sign(perm: buffer, onebased: boolean): number
		local seen = table.create(n, false)
		local cycles = 0
		for i = 1, n do
			if not seen[i] then
				cycles += 1
				local j = i
				while not seen[j] do
					seen[j] = true
					j = buffer.readi32(perm, 4 * (j - 1)) + (if onebased then 0 else 1)
				end
			end
		end
		return if (n - cycles) % 2 == 0 then 1 else -1
	end
	local d = sign(F.q, true) * sign(F.pinv, false)
	for j = 0, n - 1 do
		d *= buffer.readf64(F.Ux, 8 * (buffer.readu32(F.Up, 4 * (j + 1)) - 1))
	end
	return d
end

-- The inverse of a square matrix (f64.inv). Solving with linalg.solve is faster and more
-- accurate than multiplying by the inverse; use this when the inverse itself is wanted.
-- A sparse A is made dense (its inverse is dense). Errors if A is singular.
local function inv(A: any): any
	local kind, M, m, n = matrix_of(A, "inv")
	if m ~= n then
		fail("inv", string.format("the matrix is %d by %d; an inverse needs a square one", m, n))
	end
	if kind == "sparse" then
		M = densify(M, "inv")
	end
	if f64.rcond(M) == 0 then
		fail("inv", "the matrix is singular")
	end
	local X = f64.inv(M)
	return if is_sparse(A) or is_matrix(A) then X else rows_of(X)
end

-- The length (2-norm) of a vector, or the 2-norm of a matrix (its largest singular value;
-- a sparse one made dense).
local function norm(x: any): number
	if type(x) == "buffer" or (type(x) == "table" and type(x[1]) == "number") then
		local v = vector_of(x, if type(x) == "buffer" then buffer.len(x) // 8 else #x, "norm")
		return f64.norm(v)
	end
	local kind, M = matrix_of(x, "norm")
	if kind == "sparse" then
		M = densify(M, "norm")
	end
	return f64.matnorm(M, 2)
end

-- The condition number of a matrix: the ratio of its largest to smallest singular value
-- (f64.cond). Large means nearly singular: solving with it loses about log10(cond) digits.
-- A sparse A is made dense.
local function cond(A: any): number
	local kind, M = matrix_of(A, "cond")
	if kind == "sparse" then
		M = densify(M, "cond")
	end
	return f64.cond(M)
end

return {
	matrix = matrix,
	sparse = sparse,
	totable = totable,
	solve = solve,
	lstsq = lstsq,
	eig = eig,
	svd = svd,
	det = det,
	inv = inv,
	norm = norm,
	cond = cond,
}
