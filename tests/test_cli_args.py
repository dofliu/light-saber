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
        self.assertEqual(args.display_size, (
            lightsaber_mvp.DISPLAY_WIDTH,
            lightsaber_mvp.DISPLAY_HEIGHT,
        ))
        self.assertEqual(args.process_scale, lightsaber_mvp.MP_PROCESS_SCALE)

    def test_custom_camera_and_no_mirror(self):
        args = lightsaber_mvp.parse_args([
            "--camera-index", "1",
            "--no-mirror",
            "--max-hands", "2",
            "--display-size", "1280x720",
            "--process-scale", "0.75",
        ])

        self.assertEqual(args.camera_index, 1)
        self.assertTrue(args.no_mirror)
        self.assertEqual(args.max_hands, 2)
        self.assertEqual(args.display_size, (1280, 720))
        self.assertEqual(args.process_scale, 0.75)

    def test_max_hands_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                lightsaber_mvp.parse_args(["--max-hands", "0"])

    def test_display_size_must_use_width_x_height(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                lightsaber_mvp.parse_args(["--display-size", "1280-720"])

    def test_display_size_dimensions_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                lightsaber_mvp.parse_args(["--display-size", "1280x0"])

    def test_process_scale_must_be_positive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                lightsaber_mvp.parse_args(["--process-scale", "0"])

    def test_process_scale_must_not_exceed_one(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                lightsaber_mvp.parse_args(["--process-scale", "1.1"])


if __name__ == "__main__":
    unittest.main()
