import unittest
import os
import tempfile

import numpy as np

import lightsaber_mvp


class SegmentCircleHitTest(unittest.TestCase):
    def test_segment_through_center_is_perfect_hit(self):
        accuracy = lightsaber_mvp.segment_circle_hit(
            np.array([0.0, 50.0]),
            np.array([100.0, 50.0]),
            np.array([50.0, 50.0]),
            20,
        )

        self.assertEqual(accuracy, 1.0)

    def test_segment_outside_circle_misses(self):
        accuracy = lightsaber_mvp.segment_circle_hit(
            np.array([0.0, 0.0]),
            np.array([100.0, 0.0]),
            np.array([50.0, 30.0]),
            20,
        )

        self.assertIsNone(accuracy)


class ArcadeGameTest(unittest.TestCase):
    def setUp(self):
        self.game = lightsaber_mvp.ArcadeGame(
            enabled=True,
            round_seconds=10.0,
            seed=7,
        )
        self.game.reset(10.0)

    def start_playing(self):
        self.assertTrue(self.game.start(10.0))
        self.game.update(
            10.0 + lightsaber_mvp.GAME_COUNTDOWN_SECONDS,
            1280,
            720,
        )
        self.assertEqual(self.game.state, "playing")
        self.assertGreaterEqual(len(self.game.targets), 1)

    def test_round_moves_from_ready_to_countdown_to_playing(self):
        self.assertEqual(self.game.state, "ready")

        self.assertTrue(self.game.start(10.0))
        self.assertEqual(self.game.state, "countdown")
        self.game.update(12.9, 1280, 720)
        self.assertEqual(self.game.state, "countdown")
        self.game.update(13.0, 1280, 720)

        self.assertEqual(self.game.state, "playing")

    def test_blade_hit_adds_score_and_combo(self):
        self.start_playing()
        target = self.game.targets[0]
        start = target.center + np.array([-target.radius * 2, 0.0])
        end = target.center + np.array([target.radius * 2, 0.0])

        hit = self.game.register_blade(
            start, end, speed=15.0, now=13.2,
            swing_vector=target.direction * 15.0,
        )

        self.assertIsNotNone(hit)
        self.assertGreater(self.game.score, 0)
        self.assertEqual(self.game.combo, 1)
        self.assertEqual(self.game.hits, 1)
        self.assertNotIn(target, self.game.targets)

    def test_stationary_blade_does_not_score(self):
        self.start_playing()
        target = self.game.targets[0]
        start = target.center + np.array([-target.radius * 2, 0.0])
        end = target.center + np.array([target.radius * 2, 0.0])

        hit = self.game.register_blade(
            start, end, speed=0.0, now=13.2,
            swing_vector=target.direction,
        )

        self.assertIsNone(hit)
        self.assertEqual(self.game.score, 0)
        self.assertIn(target, self.game.targets)

    def test_second_of_multiple_targets_can_be_removed(self):
        self.start_playing()
        first = self.game.targets[0]
        second = self.game._spawn_target(13.1, 1280, 720)
        self.game.targets.append(second)
        start = second.center + np.array([-second.radius * 2, 0.0])
        end = second.center + np.array([second.radius * 2, 0.0])

        hit = self.game.register_blade(
            start, end, speed=15.0, now=13.2,
            swing_vector=second.direction * 15.0,
        )

        self.assertIsNotNone(hit)
        self.assertIn(first, self.game.targets)
        self.assertNotIn(second, self.game.targets)

    def test_wrong_swing_direction_does_not_score(self):
        self.start_playing()
        target = self.game.targets[0]
        start = target.center + np.array([-target.radius * 2, 0.0])
        end = target.center + np.array([target.radius * 2, 0.0])

        hit = self.game.register_blade(
            start, end, speed=15.0, now=13.2,
            swing_vector=-target.direction * 15.0,
        )

        self.assertIsNone(hit)
        self.assertEqual(self.game.score, 0)
        self.assertIn(target, self.game.targets)
        self.assertEqual(self.game.hit_flashes[-1][3], "WRONG WAY")

    def test_hard_difficulty_uses_shorter_target_lifetime(self):
        hard_game = lightsaber_mvp.ArcadeGame(
            enabled=True,
            round_seconds=10.0,
            difficulty="hard",
            seed=7,
        )
        hard_game.reset(10.0)
        hard_game.start(10.0)
        hard_game.update(13.0, 1280, 720)

        target = hard_game.targets[0]
        self.assertEqual(hard_game.difficulty.name, "hard")
        self.assertAlmostEqual(
            target.expires_at - target.spawned_at,
            lightsaber_mvp.DIFFICULTY_PRESETS["hard"].target_lifetime,
        )

    def spawn_laser(self, now=13.2):
        self.game.next_laser_at = now
        self.game.update(now, 1280, 720)
        self.assertEqual(len(self.game.laser_bolts), 1)
        return self.game.laser_bolts[0]

    def test_laser_moves_toward_destination(self):
        self.start_playing()
        bolt = self.spawn_laser()
        initial_distance = np.linalg.norm(bolt.destination - bolt.position)

        self.game.update(13.7, 1280, 720)

        moved_distance = np.linalg.norm(bolt.destination - bolt.position)
        self.assertLess(moved_distance, initial_distance)

    def test_blade_can_parry_laser(self):
        self.start_playing()
        bolt = self.spawn_laser()
        start = bolt.position + np.array([-bolt.radius * 2, 0.0])
        end = bolt.position + np.array([bolt.radius * 2, 0.0])

        parry = self.game.register_laser_parry(
            start, end, speed=15.0, now=13.21)

        self.assertIsNotNone(parry)
        self.assertEqual(self.game.parries, 1)
        self.assertGreater(self.game.score, 0)
        self.assertNotIn(bolt, self.game.laser_bolts)

    def test_unblocked_laser_removes_life(self):
        self.start_playing()
        bolt = self.spawn_laser()

        self.game.update(bolt.danger_at + 0.01, 1280, 720)

        self.assertEqual(self.game.lives, lightsaber_mvp.GAME_STARTING_LIVES - 1)
        self.assertEqual(self.game.misses, 1)
        self.assertNotIn(bolt, self.game.laser_bolts)

    def test_last_unblocked_laser_ends_round(self):
        self.start_playing()
        self.game.lives = 1
        bolt = self.spawn_laser()

        self.game.update(bolt.danger_at + 0.01, 1280, 720)

        self.assertEqual(self.game.state, "results")
        self.assertEqual(self.game.lives, 0)
        self.assertEqual(self.game.result_reason, "game_over")

    def test_pause_freezes_round_and_target_clocks(self):
        self.start_playing()
        target = self.game.targets[0]
        original_expiry = target.expires_at
        remaining_before = self.game.remaining(14.0)

        self.assertTrue(self.game.toggle_pause(14.0))
        self.game.update(30.0, 1280, 720)
        self.assertEqual(self.game.state, "paused")
        self.assertIn(target, self.game.targets)
        self.assertTrue(self.game.toggle_pause(19.0))

        self.assertEqual(self.game.state, "playing")
        self.assertAlmostEqual(self.game.remaining(19.0), remaining_before)
        self.assertAlmostEqual(target.expires_at, original_expiry + 5.0)

    def test_pause_freezes_laser_damage(self):
        self.start_playing()
        bolt = self.spawn_laser()
        self.game.toggle_pause(13.3)

        self.game.update(bolt.danger_at + 5.0, 1280, 720)

        self.assertEqual(self.game.lives, lightsaber_mvp.GAME_STARTING_LIVES)
        self.assertIn(bolt, self.game.laser_bolts)

    def test_finishing_above_high_score_marks_record(self):
        game = lightsaber_mvp.ArcadeGame(
            enabled=True,
            round_seconds=1.0,
            difficulty="normal",
            seed=7,
            high_score=100,
        )
        game.reset(10.0)
        game.start(10.0)
        game.update(13.0, 1280, 720)
        game.score = 250

        game.update(14.01, 1280, 720)

        self.assertEqual(game.high_score, 250)
        self.assertTrue(game.new_high_score)

    def test_combo_expires_without_followup_hit(self):
        self.start_playing()
        self.game.combo = 3
        self.game.last_hit_at = 13.1

        self.game.update(
            13.1 + lightsaber_mvp.GAME_COMBO_WINDOW + 0.01,
            1280,
            720,
        )

        self.assertEqual(self.game.combo, 0)

    def test_expired_target_counts_as_miss_and_breaks_combo(self):
        self.start_playing()
        self.game.combo = 4
        target = self.game.targets[0]

        self.game.update(target.expires_at + 0.01, 1280, 720)

        self.assertEqual(self.game.misses, 1)
        self.assertEqual(self.game.combo, 0)

    def test_round_finishes_and_clears_targets(self):
        self.start_playing()

        self.game.update(23.01, 1280, 720)

        self.assertEqual(self.game.state, "results")
        self.assertEqual(self.game.targets, [])

    def test_free_play_does_not_start_round(self):
        self.game.set_enabled(False, 20.0)

        self.assertFalse(self.game.start(20.0))
        self.assertEqual(self.game.state, "free")


class HighScoreStoreTest(unittest.TestCase):
    def test_record_persists_only_best_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "scores.json")
            store = lightsaber_mvp.HighScoreStore(path)

            self.assertEqual(store.load(), {})
            self.assertEqual(store.record("normal", 500), 500)
            self.assertEqual(store.record("normal", 300), 500)
            self.assertEqual(store.load(), {"normal": 500})

    def test_invalid_score_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "scores.json")
            with open(path, "w", encoding="utf-8") as score_file:
                score_file.write("not-json")

            store = lightsaber_mvp.HighScoreStore(path)

            self.assertEqual(store.load(), {})

if __name__ == "__main__":
    unittest.main()
