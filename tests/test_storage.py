from neural_kv.storage import default_storage_roots, format_bytes, parse_size


def test_parse_size_supports_decimal_and_binary_units() -> None:
    assert parse_size("10TB") == 10_000_000_000_000
    assert parse_size("1GiB") == 2**30


def test_format_bytes() -> None:
    assert format_bytes(999) == "999B"
    assert format_bytes(1_500_000).endswith("MB")


def test_default_storage_roots_include_uv_cache() -> None:
    roots = {str(path) for path in default_storage_roots()}
    assert "data" in roots
    assert any(".cache" in root and "uv" in root for root in roots)
