from neural_kv.storage import format_bytes, parse_size


def test_parse_size_supports_decimal_and_binary_units() -> None:
    assert parse_size("10TB") == 10_000_000_000_000
    assert parse_size("1GiB") == 2**30


def test_format_bytes() -> None:
    assert format_bytes(999) == "999B"
    assert format_bytes(1_500_000).endswith("MB")
