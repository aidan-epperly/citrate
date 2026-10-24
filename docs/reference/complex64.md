# complex64

Double-precision complex scalars as `(re, im)` pairs.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `add`

```luau
add(ar: number, ai: number, br: number, bi: number): (number, number)
```

a + b.

### `sub`

```luau
sub(ar: number, ai: number, br: number, bi: number): (number, number)
```

a - b.

### `mul`

```luau
mul(ar: number, ai: number, br: number, bi: number): (number, number)
```

a b.

### `div`

```luau
div(ar: number, ai: number, br: number, bi: number): (number, number)
```

Smith's algorithm: avoids overflow in br^2 + bi^2.

### `scale`

```luau
scale(ar: number, ai: number, s: number): (number, number)
```

s a for a real s.

### `inv`

```luau
inv(re: number, im: number): (number, number)
```

1 / z.

### `conj`

```luau
conj(re: number, im: number): (number, number)
```

The conjugate.

### `abs`

```luau
abs(re: number, im: number): number
```

|re + i im| without overflow or underflow in the squares.

### `arg`

```luau
arg(re: number, im: number): number
```

The argument (angle) of z, in (-pi, pi].

### `polar`

```luau
polar(r: number, theta: number): (number, number)
```

The complex number with modulus r and argument theta.

### `exp`

```luau
exp(re: number, im: number): (number, number)
```

e^z.

### `log`

```luau
log(re: number, im: number): (number, number)
```

Principal branch.

### `sqrt`

```luau
sqrt(re: number, im: number): (number, number)
```

Principal branch, without cancellation in either component.

### `pow`

```luau
pow(zr: number, zi: number, wr: number, wi: number): (number, number)
```

Principal branch of z^w = exp(w log z).

### `tostring`

```luau
tostring(re: number, im: number): string
```

z as a string, such as "1 - 2i".
