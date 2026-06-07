import unittest

import lightsaber_mvp


class CliArgsTest(unittest.TestCase):
    def test_default_camera_and_mirror(self):
        args = lightsaber_mvp.parse_args([])

        self.assertEqual(args.camera_index, 0)
        self.assertFalse(args.no_mirror)

    def test_custom_camera_and_no_mirror(self):
        args = lightsaber_mvp.parse_args(["--camera-index", "1", "--no-mirror"])

        self.assertEqual(args.camera_index, 1)
        self.assertTrue(args.no_mirror)


if __name__ == "__main__":
    unittest.main()
