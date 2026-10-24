# vec

`vector` as a real `Space` of dimension at most 3.

_Generated from the source comments by `tools/gen_docs.py`; do not edit._

### `space`

```luau
space: types.Space<vector>
```

vector as a Space (see types.Space), for the solvers that take one.

### `similar`

```luau
similar(x: vector): vector
```

A zero vector (vectors are values, so there is nothing to allocate).

### `copy`

```luau
copy(_: vector?, x: vector): vector
```

x (vectors are values; the out argument is ignored).

### `add`

```luau
add(_: vector?, x: vector, y: vector): vector
```

x + y.

### `sub`

```luau
sub(_: vector?, x: vector, y: vector): vector
```

x - y.

### `scale`

```luau
scale(_: vector?, a: number, x: vector): vector
```

a x.

### `axpy`

```luau
axpy(_: vector?, a: number, x: vector, y: vector): vector
```

a x + y.

### `dot`

```luau
dot(x: vector, y: vector): number
```

The dot product.

### `norm`

```luau
norm(x: vector): number
```

The Euclidean norm.
