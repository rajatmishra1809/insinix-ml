import math
import time
import requests

from generate_dataset import SimulatedEnvironment


NUM_MISSIONS = 5
MAX_STEPS = 150


def run_acceptance_tests():
    print("==========================================================")
    print("     INSINIX ML ROVER CONTROL ACCEPTANCE TEST SUITE       ")
    print("==========================================================")

    url = "http://localhost:8000"
    session = requests.Session()

    try:
        # ======================================================
        # TEST 1: Check FastAPI server health
        # ======================================================
        try:
            r = session.get(
                f"{url}/",
                timeout=5
            )
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            print("\nERROR: FastAPI server is not running.")
            print("Start it with:")
            print("python -m uvicorn ml.server:app --host 127.0.0.1 --port 8000")
            return

        data = r.json()

        print(
            f"\n[TEST 1] GET / Endpoint: "
            f"{data['status']} "
            f"(Model Loaded: {data['model_loaded']})"
        )

        assert r.status_code == 200
        assert data["status"] == "online"
        assert data["model_loaded"] is True

        # ======================================================
        # TEST 2: Check Safety Layer EMERGENCY_STOP
        # ======================================================
        close_sample = {
            "lidar": [0.04] * 16,
            "goal_angle": 0.0,
            "goal_distance": 0.5,
            "speed": 0.3,
            "heading": 0.0,
            "payload_weight": 0.2,
            "friction": 0.8
        }

        r_stop = session.post(
            f"{url}/predict",
            json=close_sample,
            timeout=5
        )

        r_stop.raise_for_status()

        stop_data = r_stop.json()

        print(
            f"[TEST 2] Safety Layer Emergency Stop: "
            f"Recommendation={stop_data['recommendation']}, "
            f"Target Speed={stop_data['target_speed']}"
        )

        assert stop_data["recommendation"] == "EMERGENCY_STOP"
        assert stop_data["target_speed"] == 0.0

        # ======================================================
        # TEST 3: Run 5 Complete Missions
        # ======================================================
        print(
            f"\n--- Running {NUM_MISSIONS} Complete "
            f"INSINIX ML Missions ---"
        )

        sim = SimulatedEnvironment()

        successes = 0
        collisions = 0

        total_time = 0.0
        total_path = 0.0

        latencies = []

        # Timing diagnostics & model metrics
        total_lidar_time = 0.0
        total_api_time = 0.0
        total_movement_time = 0.0

        total_steps = 0

        all_target_speeds = []
        all_steerings = []
        all_recommendations = []

        # ======================================================
        # MISSIONS
        # ======================================================
        for m in range(1, NUM_MISSIONS + 1):

            obstacles = sim.generate_obstacles(
                num_obstacles=20
            )

            # Start position
            rx = 80.0
            ry = 80.0

            # Goal position
            gx = 520.0
            gy = 520.0

            # Rover state
            heading = 0.0
            speed = 0.0

            path_len = 0.0

            start_time = time.time()

            reached = False
            mission_collision = False

            mission_lidar_time = 0.0
            mission_api_time = 0.0

            if m == 1:
                print("\n--- Diagnostic Steps for Mission 01 (First 20 Steps) ---")

            # ==================================================
            # SIMULATION LOOP
            # ==================================================
            for step in range(MAX_STEPS):

                total_steps += 1

                # ------------------------------------------------
                # LiDAR
                # ------------------------------------------------
                lidar_start = time.perf_counter()

                lidar = sim.cast_lidar(
                    rx,
                    ry,
                    heading,
                    obstacles
                )

                lidar_elapsed = (
                    time.perf_counter()
                    - lidar_start
                )

                total_lidar_time += lidar_elapsed
                mission_lidar_time += lidar_elapsed

                min_dist = min(lidar)

                # ------------------------------------------------
                # Collision detection
                # ------------------------------------------------
                if min_dist < 0.05:
                    collisions += 1
                    mission_collision = True
                    break

                # ------------------------------------------------
                # Goal distance
                # ------------------------------------------------
                dx = gx - rx
                dy = gy - ry

                goal_dist_raw = math.sqrt(
                    dx * dx + dy * dy
                )

                # ------------------------------------------------
                # Goal reached
                # ------------------------------------------------
                if goal_dist_raw < 25.0:
                    reached = True
                    successes += 1
                    break

                # ------------------------------------------------
                # CORRECT GOAL ANGLE
                # ------------------------------------------------
                target_angle = math.atan2(
                    dy,
                    dx
                )

                angle_error = (
                    target_angle - heading
                )

                # Normalize angle to [-pi, pi]
                angle_error = math.atan2(
                    math.sin(angle_error),
                    math.cos(angle_error)
                )

                # Normalize to [-1, 1]
                goal_angle = (
                    angle_error / math.pi
                )

                goal_angle = max(
                    -1.0,
                    min(
                        1.0,
                        goal_angle
                    )
                )

                # ------------------------------------------------
                # Normalize goal distance
                # ------------------------------------------------
                goal_dist = min(
                    1.0,
                    goal_dist_raw / 600.0
                )

                # ------------------------------------------------
                # Other state variables
                # ------------------------------------------------
                payload_weight = 0.3
                friction = 0.8

                state_req = {
                    "lidar": [
                        float(x)
                        for x in lidar
                    ],

                    "goal_angle": float(
                        goal_angle
                    ),

                    "goal_distance": float(
                        goal_dist
                    ),

                    "speed": float(
                        min(
                            1.0,
                            speed / 4.0
                        )
                    ),

                    "heading": float(
                        max(
                            -1.0,
                            min(
                                1.0,
                                heading / math.pi
                            )
                        )
                    ),

                    "payload_weight": float(
                        payload_weight
                    ),

                    "friction": float(
                        friction
                    )
                }

                # ------------------------------------------------
                # ML API prediction
                # ------------------------------------------------
                api_start = time.perf_counter()

                response = session.post(
                    f"{url}/predict",
                    json=state_req,
                    timeout=5
                )

                response.raise_for_status()

                resp = response.json()

                api_elapsed = (
                    time.perf_counter()
                    - api_start
                )

                total_api_time += api_elapsed
                mission_api_time += api_elapsed

                latency = (
                    api_elapsed * 1000.0
                )

                latencies.append(latency)

                # ------------------------------------------------
                # ML outputs
                # ------------------------------------------------
                steering = float(
                    resp["steering"]
                )

                target_speed = float(
                    resp["target_speed"]
                )

                recommendation = resp.get("recommendation", "NORMAL")

                all_target_speeds.append(target_speed)
                all_steerings.append(steering)
                all_recommendations.append(recommendation)

                if m == 1 and step < 20:
                    print(
                        f"Step: {step + 1}\n"
                        f"goal_distance: {goal_dist_raw:.2f}\n"
                        f"goal_angle: {goal_angle:.4f}\n"
                        f"current_speed: {speed:.4f}\n"
                        f"predicted_steering: {steering:.4f}\n"
                        f"predicted_target_speed: {target_speed:.4f}\n"
                        f"recommendation: {recommendation}\n"
                        f"nearest_lidar: {min_dist:.4f}\n"
                        f"rx: {rx:.2f}\n"
                        f"ry: {ry:.2f}\n"
                        f"heading: {heading:.4f}\n"
                        f"----------------------------------------"
                    )

                # ------------------------------------------------
                # Rover dynamics
                # ------------------------------------------------
                movement_start = (
                    time.perf_counter()
                )

                # Steering changes heading
                turn_rate = (
                    steering
                    * 0.20
                    * friction
                )

                heading += turn_rate

                # Keep heading in [-pi, pi]
                heading = math.atan2(
                    math.sin(heading),
                    math.cos(heading)
                )

                # Smooth acceleration
                desired_speed = (
                    target_speed * 4.0
                )

                speed += (
                    desired_speed - speed
                ) * 0.30

                # ------------------------------------------------
                # ACTUAL HEADING-BASED MOVEMENT
                # ------------------------------------------------
                step_distance = speed

                old_x = rx
                old_y = ry

                rx += (
                    math.cos(heading)
                    * step_distance
                )

                ry += (
                    math.sin(heading)
                    * step_distance
                )

                actual_step_distance = math.sqrt(
                    (rx - old_x) ** 2
                    + (ry - old_y) ** 2
                )

                path_len += (
                    actual_step_distance
                )

                movement_elapsed = (
                    time.perf_counter()
                    - movement_start
                )

                total_movement_time += (
                    movement_elapsed
                )

            # ==================================================
            # MISSION RESULT
            # ==================================================
            mission_time = (
                time.time() - start_time
            )

            total_time += mission_time
            total_path += path_len

            if reached:
                result = "SUCCESS"
            elif mission_collision:
                result = "COLLISION"
            else:
                result = "INCOMPLETE"

            print(
                f"  Mission {m:02d}: {result} | "
                f"Path: {path_len:.1f}px | "
                f"Time: {mission_time:.2f}s | "
                f"Steps: {step + 1}"
            )

            print(
                f"             LiDAR: "
                f"{mission_lidar_time:.3f}s | "
                f"API: {mission_api_time:.3f}s"
            )

        # ======================================================
        # FINAL REPORT
        # ======================================================
        avg_latency = (
            sum(latencies) / len(latencies)
            if latencies
            else 0.0
        )

        avg_lidar_ms = (
            total_lidar_time
            / total_steps
            * 1000.0
            if total_steps
            else 0.0
        )

        avg_api_ms = (
            total_api_time
            / total_steps
            * 1000.0
            if total_steps
            else 0.0
        )

        avg_movement_ms = (
            total_movement_time
            / total_steps
            * 1000.0
            if total_steps
            else 0.0
        )

        avg_target_speed = (
            sum(all_target_speeds) / len(all_target_speeds)
            if all_target_speeds
            else 0.0
        )
        min_target_speed = (
            min(all_target_speeds)
            if all_target_speeds
            else 0.0
        )
        max_target_speed = (
            max(all_target_speeds)
            if all_target_speeds
            else 0.0
        )
        avg_abs_steering = (
            sum(abs(s) for s in all_steerings) / len(all_steerings)
            if all_steerings
            else 0.0
        )
        pct_low_speed = (
            sum(1 for s in all_target_speeds if s < 0.05) / len(all_target_speeds) * 100.0
            if all_target_speeds
            else 0.0
        )
        pct_emergency = (
            sum(1 for r in all_recommendations if r == "EMERGENCY_STOP") / len(all_recommendations) * 100.0
            if all_recommendations
            else 0.0
        )
        pct_slow_down = (
            sum(1 for r in all_recommendations if r == "SLOW_DOWN") / len(all_recommendations) * 100.0
            if all_recommendations
            else 0.0
        )

        print("\n==========================================================")
        print("                    FINAL REPORT METRICS")
        print("==========================================================")

        print(
            f"Success Rate:          "
            f"{successes / NUM_MISSIONS * 100:.1f}% "
            f"({successes}/{NUM_MISSIONS})"
        )

        print(
            f"Collision Count:       "
            f"{collisions}"
        )

        print(
            f"Total Mission Runtime: "
            f"{total_time:.2f} s"
        )

        print(
            f"Avg Mission Time:      "
            f"{total_time / NUM_MISSIONS:.2f} s"
        )

        print(
            f"Avg Path Length:       "
            f"{total_path / NUM_MISSIONS:.1f} px"
        )

        print(
            f"Avg Inference Latency: "
            f"{avg_latency:.2f} ms"
        )

        print(
            f"Avg LiDAR Time/Step:   "
            f"{avg_lidar_ms:.2f} ms"
        )

        print(
            f"Avg API Time/Step:     "
            f"{avg_api_ms:.2f} ms"
        )

        print(
            f"Avg Movement Time:     "
            f"{avg_movement_ms:.4f} ms"
        )

        print(
            f"Total Simulation Steps: "
            f"{total_steps}"
        )

        print("\n--- MODEL & SAFETY DIAGNOSTICS ---")
        print(f"Average Predicted Target Speed:   {avg_target_speed:.4f}")
        print(f"Minimum Predicted Target Speed:   {min_target_speed:.4f}")
        print(f"Maximum Predicted Target Speed:   {max_target_speed:.4f}")
        print(f"Average Absolute Steering:        {avg_abs_steering:.4f}")
        print(f"Steps with target_speed < 0.05:   {pct_low_speed:.1f}%")
        print(f"Steps with EMERGENCY_STOP:       {pct_emergency:.1f}%")
        print(f"Steps with SLOW_DOWN:            {pct_slow_down:.1f}%")

        print("==========================================================")
    finally:
        session.close()


if __name__ == "__main__":
    run_acceptance_tests()