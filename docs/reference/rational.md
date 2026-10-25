# Rational approximation

The AAA algorithm: rational approximation from samples anywhere in the complex plane, with poles, residues and zeros.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `aaa`

```luau
aaa(Z: buffer, F: buffer, options: RationalOptions?): Rational
```

The AAA rational approximation of the values F (a c64 array) at the sample points Z (a c64 array of distinct points; for real data, c64.complexify(x)). options.tol is the relative tolerance on the samples (default 1e-13), options.mmax the most support points (default 100), options.cleanup (default true) removes Froissart doublets.
