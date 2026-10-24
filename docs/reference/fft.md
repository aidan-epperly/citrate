# Fourier and cosine transforms

FFTs of any length (complex modules), real FFTs and discrete cosine transforms (real modules). Parts `fft_f32`, `fft_f64`, `fft_c32` and `fft_c64`.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `rfft`

```luau
rfft(out: buffer?, x: buffer): buffer
```

The n // 2 + 1 coefficients X_0 .. X_(n/2) of the DFT of the real array x (the rest follow from X_(n-k) = conj(X_k)), as a complex array (c64). Even lengths use one complex FFT of half the length.

### `irfft`

```luau
irfft(out: buffer?, X: buffer, n: number): buffer
```

The real array of length n whose rfft is X (n // 2 + 1 complex coefficients, c64). The imaginary parts of X_0 and, for even n, X_(n/2) are ignored.

### `dct`

```luau
dct(x: buffer, kind: number?, norm: DctNorm?): buffer
```

The discrete cosine transform of the given type (1 to 4, default 2), with SciPy's definitions: unnormalized, or orthonormal with norm = "ortho". In place; returns x.

### `idct`

```luau
idct(x: buffer, kind: number?, norm: DctNorm?): buffer
```

The inverse of dct with the same type and norm. In place; returns x.

### `dct_copy`

```luau
dct_copy(x: buffer, kind: number?, norm: DctNorm?): buffer
```

dct of a copy of x; x is not modified.

### `idct_copy`

```luau
idct_copy(x: buffer, kind: number?, norm: DctNorm?): buffer
```

idct of a copy of x; x is not modified.

### `fft`

```luau
fft(x: buffer, plan: FFTPlan?): buffer
```

Discrete Fourier transform X_k = sum_j x_j exp(-2 pi i j k / n), in place. Returns x.

### `ifft`

```luau
ifft(x: buffer, plan: FFTPlan?): buffer
```

Inverse transform x_j = (1/n) sum_k X_k exp(2 pi i j k / n), in place. Returns x.

### `fft_copy`

```luau
fft_copy(x: buffer, plan: FFTPlan?): buffer
```

fft of a copy of x; x is not modified.

### `ifft_copy`

```luau
ifft_copy(x: buffer, plan: FFTPlan?): buffer
```

ifft of a copy of x; x is not modified.

### `fft2`

```luau
fft2(A: Matrix): Matrix
```

Two-dimensional DFT of a matrix, in place. Returns A.

### `ifft2`

```luau
ifft2(A: Matrix): Matrix
```

Inverse two-dimensional DFT of a matrix, in place. Returns A.
