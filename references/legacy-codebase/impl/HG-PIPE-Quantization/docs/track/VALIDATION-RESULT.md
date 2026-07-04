# Validation Result

## Commands

```bash
python3 -m unittest discover -s tests
python3 -m hgpipe_quantization.cli list --limit 12
python3 -m hgpipe_quantization.cli verify
```

## Results

- Unit/smoke tests: 5 tests passed.
- Discovered quantization cases: 97.
- Full refs verification: 97/97 passed.
- Elements checked: 5,899,008.
- Total mismatches: 0.

## Reports

- `reports/verification.md`
- `reports/verification.json`
