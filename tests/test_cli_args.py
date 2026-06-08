import contextlib
import io
import unittest

import lightsaber_mvp


class CliArgsTest(unittest.TestCase):
    def test_default_camera_and_mirror(self):
        args = lightsaber_mvp.parse_args([])

        self.assertEqual(args.camera_index, 0)
        self.assertFalse(args.no_mirror)
        self.assertEqual(args.max_hands, lightsaber_mvp.MAX_HANDS)

    def test_custom_camera_and_no_mirror(self):
        args = lightsaber_mvp.parse_args([
            "--camera-index", "1",
            "--no-mirror",
            "--max-hands", "2",
        ])

        self.assertEqual(args.camera_index, 1)
        self.assertTrue(args.no_mirror)
        self.assertEqual(args.max_hands, 2)

    def test_max_hands_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                lightsaber_mvp.parse_args(["--max-hands", "0"])


if __name__ == "__main__":
    unittest.main()
