"""Synthetic GPS ETA training data and a diamond-network traffic model."""

import numpy as np

from gaussian_solver import gaussian_elimination


def train_eta_model():
    """Fit a linear ETA model using normal equations and custom elimination."""
    rng = np.random.default_rng(42)
    volume = rng.uniform(0.0, 100.0, 25)
    weather = rng.uniform(0.0, 1.0, 25)
    accident = rng.integers(0, 2, 25).astype(float)
    features = np.column_stack((np.ones(25), volume, weather, accident))
    expected_weights = np.array([5.0, 0.1, 7.5, 16.5])
    noise = rng.uniform(-0.15, 0.15, 25)
    travel_time = features @ expected_weights + noise

    normal_matrix = features.T @ features
    right_hand_side = features.T @ travel_time
    weights, snapshots = gaussian_elimination(normal_matrix, right_hand_side)
    return features, travel_time, weights, snapshots


def predict_eta(weights, volume, weather, accident):
    """Predict travel time in minutes for a live route's conditions."""
    live_features = np.array([1.0, volume, weather, accident], dtype=float)
    return float(live_features @ np.asarray(weights, dtype=float))


def solve_traffic_split(total_cars, accident_active):
    """Solve route-edge flows and return ``(route1, route2, augmented_matrix)``.

    The unknowns are A→B, B→D, A→C, and C→D flows. The matrix contains
    conservation at A, B, and C plus the route-choice equilibrium/active
    constraint. Conservation at D follows from these equations and total
    demand. In normal traffic the equilibrium equalizes capacity-adjusted
    travel-time pressure; when B→D is blocked, the active constraint sets its
    flow to zero (the alternate route is then the only feasible choice).
    """
    if not np.isfinite(total_cars) or total_cars < 0:
        raise ValueError("total_cars must be a finite non-negative number")

    # Columns: A→B, B→D, A→C, C→D. Rows: A, B, C, route choice.
    if accident_active:
        matrix = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        rhs = np.array([total_cars, 0.0, 0.0, 0.0])
    else:
        # Effective service rates 36/12=3 and 16/16=1 give a 75/25 split.
        matrix = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
                [1.0 / 3.0, 0.0, -1.0, 0.0],
            ]
        )
        rhs = np.array([total_cars, 0.0, 0.0, 0.0])

    edge_flows, _ = gaussian_elimination(matrix, rhs)
    augmented = np.column_stack((matrix, rhs))
    route1_flow = float(edge_flows[0])
    route2_flow = float(edge_flows[2])
    return route1_flow, route2_flow, augmented
