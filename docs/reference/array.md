# Arrays

Creation and access, elementwise arithmetic, reductions and norms, and each module's `space`. Parts `array_f32`, `array_f64`, `array_c32` and `array_c64`.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `create`

```luau
create(n: number): buffer
```

An array of n zeros.

### `length`

```luau
length(x: buffer): number
```

The number of elements of x.

### `similar`

```luau
similar(x: buffer): buffer
```

A new array of zeros as long as x.

### `matrix`

```luau
matrix(rows: number, cols: number): Matrix
```

A rows by cols matrix of zeros.

### `from`

```luau
from(t: { number }): buffer
-- complex modules:
from(re: { number }, im: { number }?): buffer
```

An array holding the numbers of t.

### `totable`

```luau
totable(x: buffer): { number }
-- complex modules:
totable(x: buffer): ({ number }, { number })
```

The elements of x as a table of numbers.

### `get`

```luau
get(x: buffer, i: number): number
-- complex modules:
get(x: buffer, i: number): (number, number)
```

Element i.

### `set`

```luau
set(x: buffer, i: number, v: number)
-- complex modules:
set(x: buffer, i: number, re: number, im: number)
```

Sets element i to v.

### `fill`

```luau
fill(out: buffer, v: number): buffer
-- complex modules:
fill(out: buffer, re: number, im: number): buffer
```

Sets every element of out to v and returns out.

### `copy`

```luau
copy(out: buffer?, x: buffer): buffer
```

out := x (a new array if out is nil).

### `add`

```luau
add(out: buffer?, x: buffer, y: buffer): buffer
```

out := x + y, elementwise.

### `sub`

```luau
sub(out: buffer?, x: buffer, y: buffer): buffer
```

out := x - y, elementwise.

### `mul`

```luau
mul(out: buffer?, x: buffer, y: buffer): buffer
```

Elementwise (Hadamard) product.

### `scale`

```luau
scale(out: buffer?, a: number, x: buffer): buffer
```

out := a x.

### `axpy`

```luau
axpy(out: buffer?, a: number, x: buffer, y: buffer): buffer
```

a * x + y

### `dot`

```luau
dot(x: buffer, y: buffer): number
-- complex modules:
dot(x: buffer, y: buffer): (number, number)
```

The dot product, the sum of x_i y_i.

### `sum`

```luau
sum(x: buffer): number
-- complex modules:
sum(x: buffer): (number, number)
```

The sum of the elements.

### `amax`

```luau
amax(x: buffer): number
```

Largest absolute value (NaN if any element is NaN, 0 for an empty array).

### `norm`

```luau
norm(x: buffer): number
```

Euclidean norm, without overflow or underflow.

### `space`

```luau
space: types.ComplexSpace<buffer>
```

The module's arrays as a ComplexSpace (see types.ComplexSpace), for the iterative and ODE solvers.

### `complexify`

```luau
complexify(re: buffer, im: buffer?): buffer
```

Builds a complex array from real arrays of the same precision.

### `real`

```luau
real(out: buffer?, x: buffer): buffer
```

The real parts of x, as a real array of the same precision.

### `imag`

```luau
imag(out: buffer?, x: buffer): buffer
```

The imaginary parts of x, as a real array of the same precision.

### `abs`

```luau
abs(out: buffer?, x: buffer): buffer
```

Elementwise modulus, as a real array.

### `conj`

```luau
conj(out: buffer?, x: buffer): buffer
```

out := conj(x), elementwise.

### `cscale`

```luau
cscale(out: buffer?, ar: number, ai: number, x: buffer): buffer
```

out := a x for the complex a = ar + i ai.

### `caxpy`

```luau
caxpy(out: buffer?, ar: number, ai: number, x: buffer, y: buffer): buffer
```

out := a x + y for the complex a = ar + i ai.

### `realdot`

```luau
realdot(x: buffer, y: buffer): number
```

Re(sum conj(x_i) y_i): the real inner product of the underlying real vector space (the dot product of the 2n real components).

### `dotc`

```luau
dotc(x: buffer, y: buffer): (number, number)
```

sum conj(x_i) y_i
