"""Adversarial checks across the Python/native trust boundary."""

import tempfile
import unittest
from pathlib import Path

from remoteworker import _native
from remoteworker.core.result_cache import ByteResultCache
from remoteworker.integration.discovery_routes import discard_request_inputs


class NativeBoundaryTests(unittest.TestCase):
    def test_boolean_timeout_does_not_coerce_to_one(self):
        for value in (True, False, 5.0, "5", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _native.proxy_config({"server_url": "http://peer", "timeout_seconds": value})
        self.assertEqual(
            _native.proxy_config({"server_url": "http://peer", "timeout_seconds": 5}),
            ("http://peer", 5),
        )

    def test_scoped_cleanup_preserves_symlink_destinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            other = root / "unrelated"
            other.mkdir()
            saved = other / "valuable.txt"
            saved.write_text("keep")
            base = root / "temp" / "remote-conditioning"
            base.mkdir(parents=True)
            (base / "req1").symlink_to(other, target_is_directory=True)
            with self.assertRaises(ValueError):
                discard_request_inputs("req1", root / "temp")
            self.assertEqual(saved.read_text(), "keep")
            (base / "req1").unlink()
            base.rmdir()
            base.symlink_to(other, target_is_directory=True)
            with self.assertRaises(ValueError):
                discard_request_inputs("req1", root / "temp")
            self.assertTrue(saved.exists())

    def test_lru_byte_budget_replacement_and_oversize_are_atomic(self):
        cache = ByteResultCache(max_entries=2, max_bytes=5)
        cache.put(("A",), (b"aa",))
        cache.put(("B",), (b"bb",))
        self.assertEqual(cache.get(("A",)), (b"aa",))
        cache.put(("C",), (b"cc",))
        self.assertIsNone(cache.get(("B",)))
        cache.put(("A",), (b"123456",))
        self.assertIsNone(cache.get(("A",)))
        self.assertEqual(cache.get(("C",)), (b"cc",))

    def test_native_module_has_no_python_fallback(self):
        self.assertEqual(_native.ARCHITECTURE, "modular-monolith")
        self.assertTrue(_native.__file__.endswith(".so"))

    def test_upload_rejects_header_injection_and_nonbytes(self):
        for filename, data, mime in (
            ("../x", b"x", "image/png"),
            ("x", b"x", "image/png\r\nX: bad"),
            ("x", bytearray(b"x"), "image/png"),
            ("x", b"", "image/png"),
        ):
            with (
                self.subTest(filename=filename, mime=mime),
                self.assertRaises(ValueError),
            ):
                _native.validate_upload("req1", filename, data, mime, 1024)
