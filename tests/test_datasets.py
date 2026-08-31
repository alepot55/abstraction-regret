"""Checksum/verification logic for dataset acquisition (no network)."""

from __future__ import annotations

import hashlib

import pytest

from gpufsm.io.datasets import Dataset, ensure, sha256_file, verify


def test_sha256_and_verify(tmp_path):
    p = tmp_path / "blob.bin"
    payload = b"gpufsm" * 1000
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert sha256_file(p) == expected
    assert verify(p, expected) is True
    assert verify(p, "0" * 64) is False
    assert verify(tmp_path / "missing", expected) is False


def test_ensure_uses_verified_cache(tmp_path):
    payload = b"cached-data"
    sha = hashlib.sha256(payload).hexdigest()
    (tmp_path / "ds.bin").write_bytes(payload)
    ds = Dataset(name="ds.bin", url="http://invalid.invalid/never", sha256=sha)
    # Cached file already matches -> no download attempted.
    assert ensure(ds, tmp_path).read_bytes() == payload


def test_ensure_refuses_unpinned_dataset(tmp_path):
    ds = Dataset(name="x", url="http://example/x", sha256="")
    with pytest.raises(ValueError):
        ensure(ds, tmp_path)
