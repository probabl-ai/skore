from skore._utils import b64_str_to_bytes, bytes_to_b64_str


def test_base64_round_trip():
    """Bytes helpers encode and decode a payload without changing it."""
    payload = b"skore"

    encoded = bytes_to_b64_str(payload)

    assert isinstance(encoded, str)
    assert b64_str_to_bytes(encoded) == payload
