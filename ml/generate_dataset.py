from pathlib import Path
import random
import math
import time
import heapq

import pandas as pd
import numpy as np


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TELEMETRY_CSV = DATA_DIR / "telemetry.csv"


# ============================================================
# A* PLANNER
# ============================================================

class AStarPlanner:
    """
    Grid-based A* path planner with obstacle inflation for safe navigation.
    """
    def __init__(self, map_size=600.0, cell_size=5.0, rover_radius=18.0):
        self.map_size = map_size
        self.cell_size = cell_size
        self.grid_dim = int(map_size / cell_size)
        self.rover_radius = rover_radius

    def create_grid(self, obstacles):
        grid = np.zeros((self.grid_dim, self.grid_dim), dtype=bool)
        if len(obstacles) == 0:
            return grid

        x_coords = np.arange(self.grid_dim) * self.cell_size + self.cell_size / 2.0
        y_coords = np.arange(self.grid_dim) * self.cell_size + self.cell_size / 2.0
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')

        for ox, oy, r in obstacles:
            safe_r = r + self.rover_radius
            dist_sq = (xx - ox) ** 2 + (yy - oy) ** 2
            grid |= (dist_sq <= safe_r ** 2)

        return grid

    def plan(self, start, goal, grid):
        sx = int(start[0] / self.cell_size)
        sy = int(start[1] / self.cell_size)
        gx = int(goal[0] / self.cell_size)
        gy = int(goal[1] / self.cell_size)

        sx = max(0, min(self.grid_dim - 1, sx))
        sy = max(0, min(self.grid_dim - 1, sy))
        gx = max(0, min(self.grid_dim - 1, gx))
        gy = max(0, min(self.grid_dim - 1, gy))

        if grid[sx, sy] or grid[gx, gy]:
            return None

        open_set = []
        heapq.heappush(open_set, (0.0, (sx, sy)))
        came_from = {}
        g_score = {(sx, sy): 0.0}

        neighbors = [
            (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]

        found = False
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == (gx, gy):
                found = True
                break

            curr_g = g_score[current]
            cx, cy = current

            for dx, dy, cost in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_dim and 0 <= ny < self.grid_dim:
                    if grid[nx, ny]:
                        continue
                    tentative_g = curr_g + cost
                    neighbor = (nx, ny)
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        h = math.hypot(gx - nx, gy - ny)
                        heapq.heappush(open_set, (tentative_g + h, neighbor))

        if not found:
            return None

        curr = (gx, gy)
        path = []
        while curr in came_from:
            world_x = curr[0] * self.cell_size + self.cell_size / 2.0
            world_y = curr[1] * self.cell_size + self.cell_size / 2.0
            path.append((world_x, world_y))
            curr = came_from[curr]
        path.append((start[0], start[1]))
        path.reverse()
        return path


# ============================================================
# SIMULATED ENVIRONMENT
# ============================================================

class SimulatedEnvironment:
    """
    Python-based INSINIX telemetry dataset generator with A* expert planning.
    """

    def __init__(self, map_size=600.0):
        self.map_size = map_size
        self.max_lidar_range = 250.0
        self.num_rays = 16
        self.fov = math.radians(150.0)

        self.ray_offsets = np.linspace(
            -self.fov / 2.0,
            self.fov / 2.0,
            self.num_rays,
            dtype=np.float32
        )

        self.planner = AStarPlanner(map_size=map_size, cell_size=5.0, rover_radius=18.0)

    # ========================================================
    # OBSTACLES
    # ========================================================

    def generate_obstacles(self, num_obstacles=25, map_type="random"):
        obstacles = []

        if map_type == "corridor":
            for y in range(50, 550, 40):
                obstacles.append((200.0, float(y), 15.0))
                obstacles.append((380.0, float(y), 15.0))

        elif map_type == "obstacle_field":
            for _ in range(35):
                x = random.uniform(60, 540)
                y = random.uniform(60, 540)
                r = random.uniform(12, 30)
                obstacles.append((x, y, r))

        else:
            for _ in range(num_obstacles):
                x = random.uniform(50, 550)
                y = random.uniform(50, 550)
                r = random.uniform(10, 25)
                obstacles.append((x, y, r))

        return np.asarray(obstacles, dtype=np.float32)

    # ========================================================
    # FAST VECTORIZED LIDAR
    # ========================================================

    def cast_lidar(self, rx, ry, heading, obstacles):
        angles = heading + self.ray_offsets
        ray_dx = np.cos(angles)
        ray_dy = np.sin(angles)

        min_dist = np.full(self.num_rays, self.max_lidar_range, dtype=np.float32)

        with np.errstate(divide="ignore", invalid="ignore"):
            dist_x_positive = np.where(ray_dx > 0, (self.map_size - rx) / ray_dx, self.max_lidar_range)
            dist_x_negative = np.where(ray_dx < 0, -rx / ray_dx, self.max_lidar_range)
            dist_y_positive = np.where(ray_dy > 0, (self.map_size - ry) / ray_dy, self.max_lidar_range)
            dist_y_negative = np.where(ray_dy < 0, -ry / ray_dy, self.max_lidar_range)

        boundary_x = np.minimum(dist_x_positive, dist_x_negative)
        boundary_y = np.minimum(dist_y_positive, dist_y_negative)

        min_dist = np.minimum(min_dist, np.maximum(0.0, boundary_x))
        min_dist = np.minimum(min_dist, np.maximum(0.0, boundary_y))

        if len(obstacles) == 0:
            return np.clip(min_dist / self.max_lidar_range, 0.0, 1.0).tolist()

        ox = obstacles[:, 0]
        oy = obstacles[:, 1]
        radius = obstacles[:, 2]

        dx = ox - rx
        dy = oy - ry

        proj = dx[:, None] * ray_dx[None, :] + dy[:, None] * ray_dy[None, :]
        center_dist_sq = dx[:, None] ** 2 + dy[:, None] ** 2
        perp_sq = center_dist_sq - proj ** 2
        radius_sq = radius[:, None] ** 2

        valid = (proj > 0) & (perp_sq < radius_sq)
        discriminant = np.maximum(0.0, radius_sq - perp_sq)
        hit_dist = proj - np.sqrt(discriminant)
        hit_dist = np.where(valid & (hit_dist > 0), hit_dist, self.max_lidar_range)

        obstacle_min = np.min(hit_dist, axis=0)
        min_dist = np.minimum(min_dist, obstacle_min)

        return np.clip(min_dist / self.max_lidar_range, 0.0, 1.0).tolist()

    # ========================================================
    # SINGLE MISSION WITH A* PLANNER
    # ========================================================

    def run_simulation_mission(self, map_type="random"):
        # Generate environment with valid A* path
        for _ in range(50):
            obstacles = self.generate_obstacles(map_type=map_type)

            rx = random.uniform(50, 150)
            ry = random.uniform(50, 150)
            gx = random.uniform(450, 550)
            gy = random.uniform(450, 550)

            # Check start and goal clearance
            if len(obstacles) > 0:
                dist_start = np.hypot(obstacles[:, 0] - rx, obstacles[:, 1] - ry) - obstacles[:, 2]
                dist_goal = np.hypot(obstacles[:, 0] - gx, obstacles[:, 1] - gy) - obstacles[:, 2]
                if np.min(dist_start) < 20.0 or np.min(dist_goal) < 20.0:
                    continue

            if math.hypot(gx - rx, gy - ry) < 300.0:
                continue

            grid = self.planner.create_grid(obstacles)
            path = self.planner.plan((rx, ry), (gx, gy), grid)
            if path is not None and len(path) > 3:
                break
        else:
            return []

        heading = random.uniform(-math.pi, math.pi)
        payload_weight = float(np.clip(random.uniform(0.0, 5.0) / 5.0, 0.0, 1.0))
        friction = float(np.clip(random.uniform(0.2, 1.0), 0.0, 1.0))

        current_speed = 0.0
        max_speed = 4.0 * (1.0 - 0.3 * payload_weight)

        mission_data = []
        path_idx = 0

        for step in range(160):
            dx_final = gx - rx
            dy_final = gy - ry
            goal_dist_raw = math.hypot(dx_final, dy_final)

            if goal_dist_raw < 20.0:
                break

            # Find lookahead waypoint on A* path
            lookahead_dist = 40.0
            target_pt = (gx, gy)
            for i in range(path_idx, len(path)):
                dist_to_pt = math.hypot(path[i][0] - rx, path[i][1] - ry)
                if dist_to_pt >= lookahead_dist:
                    target_pt = path[i]
                    path_idx = i
                    break
            else:
                target_pt = (gx, gy)

            # Waypoint angle & relative angle error
            dx_way = target_pt[0] - rx
            dy_way = target_pt[1] - ry
            target_angle = math.atan2(dy_way, dx_way)

            angle_err = math.atan2(
                math.sin(target_angle - heading),
                math.cos(target_angle - heading)
            )

            # Steering target based on waypoint angle error
            steering_desired = angle_err / (math.pi / 2.5)

            # LiDAR features
            lidar_16 = self.cast_lidar(rx, ry, heading, obstacles)

            left_sector = min(lidar_16[0:5])
            right_sector = min(lidar_16[11:16])
            front_sector = min(lidar_16[5:11])
            min_lidar = min(lidar_16)

            # Obstacle avoidance adjustment
            if front_sector < 0.25:
                avoid_strength = (0.25 - front_sector) / 0.25
                if left_sector < right_sector:
                    steering_desired += 0.6 * avoid_strength
                else:
                    steering_desired -= 0.6 * avoid_strength

            expert_steering = float(np.clip(steering_desired, -1.0, 1.0))

            # Target speed logic
            if min_lidar < 0.05:
                target_speed_norm = 0.0
            elif front_sector < 0.15:
                target_speed_norm = 0.05
            elif front_sector < 0.30:
                target_speed_norm = 0.30
            else:
                base_speed = 0.85 - 0.40 * abs(expert_steering)
                target_speed_norm = float(np.clip(base_speed, 0.15, 1.0))

            # Goal angle & distance features
            goal_angle_raw = math.atan2(dy_final, dx_final) - heading
            goal_angle_raw = math.atan2(math.sin(goal_angle_raw), math.cos(goal_angle_raw))
            goal_angle_norm = float(np.clip(goal_angle_raw / math.pi, -1.0, 1.0))

            goal_dist_norm = float(np.clip(goal_dist_raw / self.map_size, 0.0, 1.0))
            heading_norm = float(np.clip(heading / math.pi, -1.0, 1.0))
            speed_norm = float(np.clip(current_speed / 4.0, 0.0, 1.0))

            # Downsample excessive straight-line samples
            if abs(expert_steering) < 0.05 and front_sector > 0.6 and random.random() < 0.5:
                pass
            else:
                row = {}
                for i in range(16):
                    row[f"lidar_{i}"] = round(lidar_16[i], 4)
                row["goal_angle"] = round(goal_angle_norm, 4)
                row["goal_distance"] = round(goal_dist_norm, 4)
                row["speed"] = round(speed_norm, 4)
                row["heading"] = round(heading_norm, 4)
                row["payload_weight"] = round(payload_weight, 4)
                row["friction"] = round(friction, 4)
                row["steering"] = round(expert_steering, 4)
                row["target_speed"] = round(target_speed_norm, 4)

                mission_data.append(row)

            # Kinematic state update
            turn_rate = expert_steering * 0.20 * friction
            heading += turn_rate
            heading = math.atan2(math.sin(heading), math.cos(heading))

            desired_speed = target_speed_norm * max_speed
            current_speed += (desired_speed - current_speed) * 0.30

            rx += current_speed * math.cos(heading)
            ry += current_speed * math.sin(heading)

            rx = max(10.0, min(self.map_size - 10.0, rx))
            ry = max(10.0, min(self.map_size - 10.0, ry))

        return mission_data


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_full_dataset(target_samples=40000):
    print(f"Generating expert A* telemetry dataset (~{target_samples:,} samples)...")

    sim = SimulatedEnvironment()
    all_samples = []
    map_types = ["random", "corridor", "obstacle_field"]

    start_time = time.perf_counter()
    mission_count = 0

    while len(all_samples) < target_samples:
        mission_count += 1
        map_type = map_types[mission_count % len(map_types)]
        samples = sim.run_simulation_mission(map_type=map_type)
        all_samples.extend(samples)

        if mission_count % 25 == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"Missions: {mission_count:03d} | "
                f"Samples: {len(all_samples):,} / {target_samples:,} | "
                f"Time: {elapsed:.2f}s"
            )

    df = pd.DataFrame(all_samples)
    df.to_csv(TELEMETRY_CSV, index=False)

    elapsed = time.perf_counter() - start_time

    steer = df["steering"].values
    speed = df["target_speed"].values

    pct_straight = (np.abs(steer) < 0.05).mean() * 100.0
    pct_sharp = (np.abs(steer) > 0.30).mean() * 100.0

    print("\n================================================")
    print("      A* DATASET GENERATION COMPLETE            ")
    print("================================================")
    print(f"Number of missions:                   {mission_count}")
    print(f"Number of samples:                    {len(df):,}")
    print(f"Steering mean/std/min/max:            {steer.mean():.4f} / {steer.std():.4f} / {steer.min():.4f} / {steer.max():.4f}")
    print(f"Target Speed mean/std/min/max:        {speed.mean():.4f} / {speed.std():.4f} / {speed.min():.4f} / {speed.max():.4f}")
    print(f"Percentage of samples with |steering| < 0.05: {pct_straight:.1f}%")
    print(f"Percentage of samples with |steering| > 0.30: {pct_sharp:.1f}%")
    print(f"Generation time:                      {elapsed:.2f}s")
    print(f"Saved to:                             {TELEMETRY_CSV}")
    print("================================================")


if __name__ == "__main__":
    generate_full_dataset()