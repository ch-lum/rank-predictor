import unittest
import json
import os
import numpy as np
from GameData import GameToData


class TestGameData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_case_dir = "test_cases"
        cls.replay_dir = "test_replays"
        cls.replay_map = {
            "general_case_0.json": "general_case.json",
            "general_case_1.json": "general_case.json",
            "general_case_2.json": "general_case.json",
            "general_case_3.json": "general_case.json",
            "general_case_4.json": "general_case.json",
            "general_case_5.json": "general_case.json",
            "general_case_6.json": "general_case.json",
            "general_case_7.json": "general_case.json",
            "general_case_8.json": "general_case.json",
            "general_case_9.json": "general_case.json",
            "general_case_10.json": "general_case.json",
            "demoed_during_goal_0.json": "demoed_during_goal.json",
        }

        cls.game_data_cache = {}

    def get_game_data(self, replay_file: str) -> GameToData:
        if replay_file not in self.game_data_cache:
            replay_path = os.path.join(self.replay_dir, replay_file)
            self.game_data_cache[replay_file] = GameToData(replay_path)
        return self.game_data_cache[replay_file]

    def load_test_case(self, test_case_file: str) -> tuple[dict, GameToData]:
        """Load a test case and return both the test data and corresponding GameToData instance"""
        file_path = os.path.join(self.test_case_dir, test_case_file)
        with open(file_path, "r") as file:
            test_case = json.load(file)

        replay_file = self.replay_map[test_case_file]
        game_data = self.get_game_data(replay_file)

        return test_case, game_data

    def test_get_relevant_keyframes(self):
        """Test that relevant keyframes match saved test cases"""
        for test_case_file in self.replay_map.keys():
            test_case, game_data = self.load_test_case(test_case_file)

            calculated_keyframes = game_data.get_relevant_keyframes(
                test_case["goal_data"]
            )
            self.assertEqual(
                calculated_keyframes,
                test_case["relevant_keyframes"],
                f"Keyframes mismatch for {test_case_file}",
            )

    def test_get_actors(self):
        """Test that actor IDs match saved test cases"""
        for test_case_file in self.replay_map.keys():
            test_case, game_data = self.load_test_case(test_case_file)

            keyframes = test_case["relevant_keyframes"]
            calculated_actors = game_data.get_actors(
                keyframes[-1], test_case["goal_data"]
            )
            self.assertEqual(
                calculated_actors,
                test_case["actors"],
                f"Actor IDs mismatch for {test_case_file}",
            )

    def test_build_dataset_goal(self):
        """Test that raw snippet data matches saved test cases"""
        for test_case_file in self.replay_map.keys():
            test_case, game_data = self.load_test_case(test_case_file)
            test_case["raw_snippet"] = {
                int(k): v for k, v in test_case["raw_snippet"].items()
            }

            calculated_snippet = game_data.build_dataset_goal(test_case["goal_data"])
            self.assertEqual(
                calculated_snippet,
                test_case["raw_snippet"],
                f"Raw snippet mismatch for {test_case_file}",
            )

    def test_interpolate_missing_data(self):
        """Test that interpolated data matches saved test cases"""
        for test_case_file in self.replay_map.keys():
            test_case, game_data = self.load_test_case(test_case_file)
            test_case["raw_snippet"] = {
                int(k): v for k, v in test_case["raw_snippet"].items()
            }
            test_case["interpolated_data"] = {
                int(k): v for k, v in test_case["interpolated_data"].items()
            }

            calculated_interpolation = game_data.interpolate_missing_data(
                test_case["raw_snippet"]
            )

            # Compare each frame's data
            for frame in test_case["interpolated_data"]:
                self.assertIn(frame, calculated_interpolation)
                for entity in ["ball", "car"]:
                    if entity in test_case["interpolated_data"][frame]:
                        self.assertIn(entity, calculated_interpolation[frame])
                        # Compare numerical values with small tolerance
                        for attr in [
                            "location",
                            "rotation",
                            "linear_velocity",
                            "angular_velocity",
                        ]:
                            for coord in ["x", "y", "z"]:
                                if (
                                    test_case["interpolated_data"][frame][entity][attr]
                                    is None
                                ):
                                    self.assertEqual(
                                        calculated_interpolation[frame][entity][attr],
                                        None,
                                        f"Interpolated data mismatch for {entity} {attr} at frame {frame} for {test_case_file}",
                                    )
                                elif (
                                    coord
                                    in test_case["interpolated_data"][frame][entity][
                                        attr
                                    ]
                                ):
                                    self.assertAlmostEqual(
                                        calculated_interpolation[frame][entity][attr][
                                            coord
                                        ],
                                        test_case["interpolated_data"][frame][entity][
                                            attr
                                        ][coord],
                                        places=5,
                                        msg=f"Mismatch in {entity} {attr} {coord} at frame {frame} for {test_case_file}",
                                    )


if __name__ == "__main__":
    unittest.main()
