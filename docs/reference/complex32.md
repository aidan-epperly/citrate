# complex32

Single-precision complex scalars stored in a `vector`.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `create`

```luau
create(re: number, im: number): vector
```

The complex number re + i im as a vector.

### `real`

```luau
real(z: vector): number
```

The real part.

### `imag`

```luau
imag(z: vector): number
```

The imaginary part.

### `add`

```luau
add(a: vector, b: vector): vector
```

a + b.

### `sub`

```luau
sub(a: vector, b: vector): vector
```

a - b.

### `mul`

```luau
mul(a: vector, b: vector): vector
```

a b.

### `div`

```luau
div(a: vector, b: vector): vector
```

a / b.

### `scale`

```luau
scale(a: vector, s: number): vector
```

s a for a real s.

### `inv`

```luau
inv(z: vector): vector
```

1 / z.

### `conj`

```luau
conj(z: vector): vector
```

The conjugate.

### `abs`

```luau
abs(z: vector): number
```

The modulus |z|, without overflow.

### `arg`

```luau
arg(z: vector): number
```

The argument (angle) of z, in (-pi, pi].

### `polar`

```luau
polar(r: number, theta: number): vector
```

The complex number with modulus r and argument theta.

### `exp`

```luau
exp(z: vector): vector
```

e^z.

### `log`

```luau
log(z: vector): vector
```

Principal branch.

### `sqrt`

```luau
sqrt(z: vector): vector
```

Principal branch, without cancellation in either component.

### `pow`

```luau
pow(z: vector, w: vector): vector
```

Principal branch of z^w = exp(w log z), with the intermediate values in double precision.

### `tostring`

```luau
tostring(z: vector): string
```

z as a string, such as "1 - 2i".
