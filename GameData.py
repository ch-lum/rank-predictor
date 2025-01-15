import json
import os
from pathlib import Path
import numpy as np


class GameToData:
    def __init__(self, fp: str, seconds: int = 5, frame_skip: int = 5):
        with open(fp, "r") as f:
            self.replay_data = json.load(f)

        self.max_frames = int(self.replay_data["properties"]["RecordFPS"] * seconds)
        self.snippet_size = self.max_frames + 5  # buffer
        self.frame_skip = frame_skip

        self.keyframes = [
            keyframe_data["frame"] for keyframe_data in self.replay_data["keyframes"]
        ]
        self.goals = [
            highlight
            for highlight in self.replay_data["properties"]["HighLights"]
            if highlight["GoalActorName"] != "None"
        ]

        self.ball_idxs = set(
            i for i, name in enumerate(self.replay_data["names"]) if "Ball" in name
        )

        self._name_id_cache = {
            name: idx for idx, name in enumerate(self.replay_data["names"])
        }

    def name_to_id(self, name: str) -> int:
        if name == "None":
            raise ValueError("Name is None")
        return self._name_id_cache[name]

    def to_recent_keyframe(self, frame: int) -> int:
        return min(
            self.keyframes,
            key=lambda keyframe: (
                frame - keyframe if keyframe <= frame else float("inf")
            ),
        )

    def get_relevant_keyframes(self, goal_data: dict) -> list[int]:
        return [
            keyframe
            for keyframe in self.keyframes
            if self.to_recent_keyframe(goal_data["frame"] - self.snippet_size)
            <= keyframe
            <= goal_data["frame"]
        ]

    def handle_ball_error(self, keyframe: int, goal_data: dict) -> int:
        possible_ball_idxs = []
        for actor in self.replay_data["network_frames"]["frames"][keyframe][
            "new_actors"
        ]:
            if actor["name_id"] in self.ball_idxs:
                possible_ball_idxs.append(actor["name_id"])
        if len(possible_ball_idxs) == 1:
            return possible_ball_idxs[0]
        else:
            raise ValueError("Multiple Balls Found")

    def get_actors(self, keyframe: int, goal_data: dict) -> dict:
        """Get the actors for a given keyframe for the ball and scorer name

        Args:
            keyframe (int): A keyframe
            goal_data (dict): The goal data
        Returns:
            dict: ball and car actor idds
        """
        if keyframe not in self.keyframes:
            raise ValueError("Keyframe not in keyframes")
        actors = {}

        # Check actor_id when keyframe regenerates actor_ids
        new_actors = self.replay_data["network_frames"]["frames"][keyframe][
            "new_actors"
        ]
        for actor in new_actors:
            try:
                ball_id = self.name_to_id(goal_data["BallName"])
            except ValueError:
                ball_id = self.handle_ball_error(keyframe, goal_data)
            car_id = self.name_to_id(goal_data["CarName"])
            if actor["name_id"] == ball_id:
                actors["ball"] = actor["actor_id"]
            elif actor["name_id"] == car_id:
                actors["car"] = actor["actor_id"]

        # handle ball error
        if "ball" not in actors:
            ball_id = self.handle_ball_error(keyframe, goal_data)
            for actor in new_actors:
                if actor["name_id"] == ball_id:
                    actors["ball"] = actor["actor_id"]

        # final error check
        if "ball" not in actors:
            raise ValueError("Ball not found")
        elif "car" not in actors:
            raise ValueError("Car not found")
        return actors

    def get_carball_data(self, actor_data: list[dict], actors: dict) -> dict:
        output = {}
        for actor in actor_data:
            if actor["actor_id"] == actors["ball"]:
                try:
                    output["ball"] = actor["attribute"]["RigidBody"]
                except (
                    KeyError
                ):  # Sometimes the attribute being updated isn't the rigid body
                    continue
            elif actor["actor_id"] == actors["car"]:
                try:
                    output["car"] = actor["attribute"]["RigidBody"]
                except KeyError:
                    continue

        return output

    def build_dataset_goal(self, goal_data: dict) -> dict:
        """Builds the dataset for a given goal

        Args:
            goal_data (dict): Data from `goals`

        Returns:
            dict: _description_
        """
        relevant_keyframes = self.get_relevant_keyframes(goal_data)
        goal_snippet_data = {}
        straddles = len(relevant_keyframes) != 1
        actors = self.get_actors(relevant_keyframes[-1], goal_data)

        for distance in range(1, self.snippet_size + 1):
            frame = goal_data["frame"] - distance
            if straddles:
                try:
                    actors = self.get_actors(self.to_recent_keyframe(frame), goal_data)
                except ValueError:
                    continue
                    # raise ValueError('Failed to get actors')
            actor_data = self.replay_data["network_frames"]["frames"][frame][
                "updated_actors"
            ]
            goal_snippet_data[distance] = self.get_carball_data(actor_data, actors)
        return goal_snippet_data

    def interpolate_missing_data(self, goal_snippet_data: dict) -> dict:
        """Interpolate missing car and ball data between known points.
        Handles:
        - Kickoff positions for ball and car at start
        - Normal interpolation between known points
        - Demolished car state for large gaps at end
        """
        interpolated_data = goal_snippet_data.copy()
        max_frame = self.snippet_size

        # Default values
        KICKOFF_BALL_DATA = {
            "sleeping": False,
            "location": {"x": 0, "y": 0, "z": 110},
            "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "linear_velocity": {"x": 0, "y": 0, "z": 0},
            "angular_velocity": {"x": 0, "y": 0, "z": 0},
        }

        KICKOFF_CAR_DATA = {
            "sleeping": False,
            "location": {
                "x": 2160.300000000003,
                "y": 2672.3000000000175,
                "z": 17.00999999999999,
            },
            "rotation": {
                "x": -0.00538131,
                "y": -0.0018153489999999939,
                "z": 0.9238633999999948,
                "w": -0.38268401999999924,
            },
            "linear_velocity": {"x": 2624.95, "y": 2624.95, "z": 3.7300000000000004},
            "angular_velocity": {
                "x": 0.3400000000000001,
                "y": 0.050000000000000044,
                "z": 0.0,
            },
        }

        DEMOLISHED_CAR_DATA = KICKOFF_CAR_DATA.copy()
        DEMOLISHED_CAR_DATA["sleeping"] = True

        # Normal interpolation
        for entity in ["car", "ball"]:
            known_frames = [
                frame for frame, data in goal_snippet_data.items() if entity in data
            ]

            if not known_frames:
                continue

            # Interpolate between known frames
            for i in range(len(known_frames) - 1):
                start_frame = known_frames[i]
                end_frame = known_frames[i + 1]
                frame_gap = end_frame - start_frame

                if frame_gap <= 1:
                    continue

                if entity == "car" and frame_gap > 10:
                    # Use the next known position (post-respawn) with sleeping=True
                    demolished_car_data = goal_snippet_data[end_frame]["car"].copy()
                    demolished_car_data["sleeping"] = True

                    # Fill in all frames in the gap
                    for frame in range(start_frame + 1, end_frame):
                        if frame not in interpolated_data:
                            interpolated_data[frame] = {}
                        interpolated_data[frame]["car"] = demolished_car_data.copy()
                else:
                    start_data = goal_snippet_data[start_frame][entity]
                    end_data = goal_snippet_data[end_frame][entity]

                    for frame in range(start_frame + 1, end_frame):
                        t = (frame - start_frame) / frame_gap
                        if frame not in interpolated_data:
                            interpolated_data[frame] = {}

                        interpolated_data[frame][entity] = self.interpolate_attributes(
                            start_data, end_data, t
                        )

            # Handle end frames
            min_frame = min(goal_snippet_data.keys())
            if entity == "car":
                # Check remaining frames after interpolation
                remaining_frames = [
                    f
                    for f in range(min_frame, max_frame)
                    if f not in interpolated_data or "car" not in interpolated_data[f]
                ]
                if len(remaining_frames) >= 10:  # Car was likely demolished
                    # Find the first frame where we have car data (closest to goal)
                    first_car_frame = known_frames[0]
                    demolished_car_data = goal_snippet_data[first_car_frame][
                        "car"
                    ].copy()
                    demolished_car_data["sleeping"] = True  # Mark as demolished

                    # Copy this data to all remaining frames
                    for frame in remaining_frames:
                        if frame not in interpolated_data:
                            interpolated_data[frame] = {}
                        interpolated_data[frame]["car"] = demolished_car_data.copy()
                elif len(known_frames) >= 2:  # Normal extrapolation for short gaps
                    last_frame = known_frames[-1]
                    second_last_frame = known_frames[-2]
                    frame_gap = last_frame - second_last_frame

                    last_data = goal_snippet_data[last_frame]["car"]
                    second_last_data = goal_snippet_data[second_last_frame]["car"]

                    for frame in remaining_frames:
                        t = (frame - last_frame) / frame_gap
                        if frame not in interpolated_data:
                            interpolated_data[frame] = {}
                        interpolated_data[frame]["car"] = self.interpolate_attributes(
                            second_last_data, last_data, 1 + t
                        )

            else:  # Normal extrapolation for ball
                remaining_frames = [
                    f
                    for f in range(min_frame, max_frame)
                    if f not in interpolated_data or "ball" not in interpolated_data[f]
                ]

                if remaining_frames and len(known_frames) >= 2:
                    last_frame = known_frames[-1]
                    second_last_frame = known_frames[-2]
                    frame_gap = last_frame - second_last_frame

                    last_data = goal_snippet_data[last_frame]["ball"]
                    second_last_data = goal_snippet_data[second_last_frame]["ball"]

                    for frame in remaining_frames:
                        t = (frame - last_frame) / frame_gap
                        if frame not in interpolated_data:
                            interpolated_data[frame] = {}
                        interpolated_data[frame]["ball"] = self.interpolate_attributes(
                            second_last_data, last_data, 1 + t
                        )

        # Handle kickoff positions
        early_frames = [f for f in goal_snippet_data.keys() if f <= 150]
        for entity, default_data in [
            ("ball", KICKOFF_BALL_DATA),
            ("car", KICKOFF_CAR_DATA),
        ]:
            empty_frames = [
                f
                for f in early_frames
                if f not in goal_snippet_data or entity not in goal_snippet_data[f]
            ]

            if len(empty_frames) >= 10:
                for frame in empty_frames:
                    if frame not in interpolated_data:
                        interpolated_data[frame] = {}
                    interpolated_data[frame][entity] = default_data.copy()

        return interpolated_data

    def interpolate_attributes(
        self, start_data: dict, end_data: dict, t: float
    ) -> dict:
        """Helper function to interpolate/extrapolate between two data points.

        Args:
            start_data (dict): Starting point data
            end_data (dict): Ending point data
            t (float): Interpolation factor (0-1 for interpolation, >1 for extrapolation)
        """
        if start_data["sleeping"]:
            return end_data
        return {
            "sleeping": False,
            "location": {
                "x": start_data["location"]["x"] * (1 - t)
                + end_data["location"]["x"] * t,
                "y": start_data["location"]["y"] * (1 - t)
                + end_data["location"]["y"] * t,
                "z": start_data["location"]["z"] * (1 - t)
                + end_data["location"]["z"] * t,
            },
            "rotation": {
                "x": start_data["rotation"]["x"] * (1 - t)
                + end_data["rotation"]["x"] * t,
                "y": start_data["rotation"]["y"] * (1 - t)
                + end_data["rotation"]["y"] * t,
                "z": start_data["rotation"]["z"] * (1 - t)
                + end_data["rotation"]["z"] * t,
                "w": start_data["rotation"]["w"] * (1 - t)
                + end_data["rotation"]["w"] * t,
            },
            "linear_velocity": {
                "x": start_data["linear_velocity"]["x"] * (1 - t)
                + end_data["linear_velocity"]["x"] * t,
                "y": start_data["linear_velocity"]["y"] * (1 - t)
                + end_data["linear_velocity"]["y"] * t,
                "z": start_data["linear_velocity"]["z"] * (1 - t)
                + end_data["linear_velocity"]["z"] * t,
            },
            "angular_velocity": {
                "x": start_data["angular_velocity"]["x"] * (1 - t)
                + end_data["angular_velocity"]["x"] * t,
                "y": start_data["angular_velocity"]["y"] * (1 - t)
                + end_data["angular_velocity"]["y"] * t,
                "z": start_data["angular_velocity"]["z"] * (1 - t)
                + end_data["angular_velocity"]["z"] * t,
            },
        }

    def flatten_game_state(self, game_state: dict) -> np.ndarray:
        """
        Flatten the game state dictionary into a numpy array.
        For each frame, extracts:
        - Ball: location (3), rotation (4), linear_velocity (3), angular_velocity (3)
        - Car: location (3), rotation (4), linear_velocity (3), angular_velocity (3)
        - Ball sleeping (1)
        - Car sleeping (1)
        Total features per frame: 25
        """
        # Sort frames by frame number and get total count
        sorted_frames = sorted(game_state.items())
        n_frames = len(sorted_frames)
        n_features = 28  # hard coded number of features

        # Initialize array with zeros
        flattened = np.zeros(n_frames * n_features, dtype=np.float32)

        INDICES = {
            "ball_sleeping": 0,
            "ball_loc": slice(1, 4),
            "ball_rot": slice(4, 8),
            "ball_lin_vel": slice(8, 11),
            "ball_ang_vel": slice(11, 14),
            "car_sleeping": 14,
            "car_loc": slice(15, 18),
            "car_rot": slice(18, 22),
            "car_lin_vel": slice(22, 25),
            "car_ang_vel": slice(25, 28),
        }

        for frame_idx, (_, frame_data) in enumerate(sorted_frames):
            start_idx = frame_idx * n_features
            ball_data = frame_data["ball"]
            car_data = frame_data["car"]

            flattened[start_idx + INDICES["ball_sleeping"]] = float(
                ball_data["sleeping"]
            )
            flattened[start_idx + INDICES["ball_loc"]] = [
                ball_data["location"][k] for k in "xyz"
            ]
            flattened[start_idx + INDICES["ball_rot"]] = [
                ball_data["rotation"][k] for k in "xyzw"
            ]
            flattened[start_idx + INDICES["ball_lin_vel"]] = [
                ball_data["linear_velocity"][k] for k in "xyz"
            ]
            flattened[start_idx + INDICES["ball_ang_vel"]] = [
                ball_data["angular_velocity"][k] for k in "xyz"
            ]

            flattened[start_idx + INDICES["car_sleeping"]] = float(car_data["sleeping"])
            flattened[start_idx + INDICES["car_loc"]] = [
                car_data["location"][k] for k in "xyz"
            ]
            flattened[start_idx + INDICES["car_rot"]] = [
                car_data["rotation"][k] for k in "xyzw"
            ]
            flattened[start_idx + INDICES["car_lin_vel"]] = [
                car_data["linear_velocity"].get(k, 0.0) for k in "xyz"
            ]
            flattened[start_idx + INDICES["car_ang_vel"]] = [
                car_data["angular_velocity"].get(k, 0.0) for k in "xyz"
            ]

        return flattened

    def build_dataset(self) -> np.ndarray:
        n_goals = len(self.goals)
        n_frames = self.max_frames // self.frame_skip
        n_features = 28
        output = np.zeros((n_goals, n_frames * n_features), dtype=np.float32)

        for i, goal in enumerate(self.goals):
            raw_data = self.build_dataset_goal(goal)
            interpolated_data = self.interpolate_missing_data(raw_data)
            shaved_data = {
                frame: data
                for frame, data in interpolated_data.items()
                if frame <= self.max_frames and frame % self.frame_skip == 0
            }
            output[i] = self.flatten_game_state(shaved_data)

        return output

    def save_test_case(self, goal_index: int, output_dir: str = "test_cases") -> None:
        """Save intermediate outputs for a specific goal as test cases.

        Args:
            goal_index (int): Index of the goal to save test case data for
            output_dir (str): Directory to save test cases in
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get the goal data
        goal_data = self.goals[goal_index]

        # Collect test case data
        test_case = {
            "goal_data": goal_data,
            "raw_snippet": self.build_dataset_goal(goal_data),
            "interpolated_data": self.interpolate_missing_data(
                self.build_dataset_goal(goal_data)
            ),
            "relevant_keyframes": self.get_relevant_keyframes(goal_data),
            "actors": self.get_actors(
                self.get_relevant_keyframes(goal_data)[-1], goal_data
            ),
        }

        # Save to file
        output_path = os.path.join(output_dir, f"test_case_goal_{goal_index}.json")

        with open(output_path, "w") as f:
            json.dump(test_case, f, indent=2)

    def save_all_test_cases(self, output_dir: str = "test_cases") -> None:
        """Save test cases for all goals in the replay.

        Args:
            output_dir (str): Directory to save test cases in
        """
        for i in range(len(self.goals)):
            self.save_test_case(i, output_dir)
