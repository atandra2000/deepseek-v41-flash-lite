# CSA2 indexer oracle (Task 5)

Upstream port: tests/oracle_upstream.py (model.py:527-610 verbatim, fp4/fp8
quantization removed per Lite deviation D6). Fixed input: seed 42, B=2, S=16,
ratio-1 decoder indexer (toy dims), candidate source on.

Result: 32/32 query rows selected identical index sets; 0 divergences.
