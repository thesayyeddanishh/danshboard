import unittest
from unittest.mock import patch
import sql_loader as loader

class TextModeRecoveryTests(unittest.TestCase):
    def test_normalizer_reverses_likely_text_mode_conversion(self):
        original = b"PGDMP" + (b"binary\n" * 100)
        converted = original.replace(b"\n", b"\r\n")
        self.assertEqual(loader._normalise_text_mode_dump(converted), original)
        self.assertIsNone(loader._normalise_text_mode_dump(b"plain text\r\n" * 100))
        self.assertIsNone(loader._normalise_text_mode_dump(b"PGDMP" + b"x\n" * 100))

    def test_read_tables_retries_candidate_and_accepts_only_success(self):
        original = b"PGDMP" + (b"payload\n" * 100)
        converted = original.replace(b"\n", b"\r\n")
        calls = []
        def fake_read(data):
            calls.append(data)
            if data == converted:
                raise loader.DumpReadError("original bytes fail")
            if data == original:
                return {"test_table": "loaded"}
            raise AssertionError("unexpected byte representation")
        with patch.object(loader, "_read_tables_once", side_effect=fake_read):
            self.assertEqual(loader.read_tables(converted), {"test_table": "loaded"})
        self.assertEqual(calls, [converted, original])

    def test_failed_recovery_reports_corruption(self):
        converted = (b"PGDMP" + b"payload\n" * 100).replace(b"\n", b"\r\n")
        with patch.object(loader, "_read_tables_once", side_effect=loader.DumpReadError("bad compressed chunk")):
            with self.assertRaisesRegex(loader.DumpReadError, "cannot safely reconstruct missing binary data"):
                loader.read_tables(converted)

if __name__ == "__main__":
    unittest.main()
