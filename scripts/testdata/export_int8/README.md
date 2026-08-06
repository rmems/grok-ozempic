# Synthetic pickle shards for `export_grok1_int8` tests

`scripts/test_export_grok1_int8_npy.py` builds every shard it needs **in memory**
at run time via `shard_factory`, so the suite has no dependency on the `.bin`
files here and nothing needs regenerating before running tests:

```bash
python3 -m unittest scripts.test_export_grok1_int8_npy
python3 scripts/dev_generate_int8_export_fixtures.py   # prints this guidance
```

The `.bin` files are a committed reference corpus of real numpy pickle framing,
useful for inspecting layout by hand:

```bash
python3 scripts/export_grok1_int8_npy.py --inspect scripts/testdata/export_int8/dequant_grouped.bin
```

The test module never imports `pickle` itself; `shard_factory` owns that, and
this directory is excluded from Codacy analysis.

## numpy-version coupling

`shard_factory.write_quantized` emits bfloat16 scales by pickling a `<u2` array
and rewriting the dtype descriptor bytes `\x8c\x02u2\x94` → `\x8c\x08bfloat16\x94`,
because numpy cannot name a `bfloat16` dtype without `ml_dtypes`.

That byte substitution is pinned to how numpy frames a short dtype string in
pickle protocol 4 (`SHORT_BINUNICODE` + memoize). It was written against the
numpy that produced the committed `.bin` files — **numpy 2.5.1, CPython 3.14,
pickle protocol 4**. A different framing (a byte-order prefix such as `<u2`, a
different memo layout, or a newer default protocol) makes the substitution
either raise at generation time — the helper asserts exactly one match — or,
in principle, emit a malformed shard.

If you add fixtures on a newer numpy and the assertion fires, derive the
descriptor through `pickletools` instead of raw byte replacement rather than
loosening the match.
