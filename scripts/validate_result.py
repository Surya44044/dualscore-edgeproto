#!/usr/bin/env python3
"""Validate a results JSON: parses, is a dict, and has numeric metrics."""
import sys, json
path = sys.argv[1]
try:
    with open(path) as f:
        j = json.load(f)
    assert isinstance(j, dict), "not a dict"
    assert len(j) > 0, "empty dict"
    assert any(isinstance(v, (int, float)) for v in j.values()), "no numeric values"
    print("VALID", path)
    sys.exit(0)
except Exception as e:
    print("INVALID:", path, "->", e)
    sys.exit(1)
