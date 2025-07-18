import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import plotly.express as px
import pandas as pd

# Constants from Flux Calculus v1.1
WINDOW_MIN = 5
WINDOW_MAX = 120
K = 0.3  # window sensitivity
LAMBDA = 0.94  # volatility decay
PROX_EPSILON0 = 1e-6
DRIFT_DELTA = 1e-9

# Simulation parameters
TICK_MINUTES = 1
SIMULATION_HOURS = 2  # shorter run for example; adjust as needed

# Simple coordinates for Bangalore intersections for visualisation
COORDS = {
    "Hebbal": (-0.5, 0.8),
    "Silk_Board": (0.1, -0.9),
    "KR_Puram": (0.8, 0.3),
    "Electronic_City": (0.6, -1.2),
}

# Choke points with approximate capacities and congestion factors
CHOKE_POINTS = {
    "Silk_Board": {"capacity": 300, "factor": 0.5},
    "KR_Puram": {"capacity": 350, "factor": 0.4},
    "Hebbal": {"capacity": 400, "factor": 0.3},
    "Electronic_City": {"capacity": 320, "factor": 0.4},
}

@dataclass
class Fluxon:
    v: float                  # latest numeric value (e.g., vehicle count)
    u: float = 0.0            # smoothed velocity
    s: float = 0.0            # volatility
    W: float = WINDOW_MIN     # adaptive window length
    anchor_until: int = 0     # if > current tick, velocity frozen

    def update_window(self):
        self.W = WINDOW_MIN + (WINDOW_MAX - WINDOW_MIN) * math.exp(-K * math.sqrt(self.s))

    def update_velocity(self, prev_v: float):
        alpha = 2 / (self.W + 1)
        if self.anchor_until > 0:
            self.u = 0.0
        else:
            self.u = (1 - alpha) * self.u + alpha * (self.v - prev_v)

    def update_volatility(self, prev_v: float):
        diff = self.v - prev_v
        self.s = LAMBDA * self.s + (1 - LAMBDA) * diff * diff

@dataclass
class Anchor:
    id: str
    delta_u: float
    delta_tau: int
    precedence: int

class TrafficSimulation:
    def __init__(self, intersections: List[str], coords: Dict[str, Tuple[float, float]]):
        self.time = 0
        self.fluxons: Dict[str, Fluxon] = {
            name: Fluxon(v=random.randint(200, 500)) for name in intersections
        }
        self.coords = coords
        self.anchors: List[Tuple[int, str, Anchor]] = []  # queue of (time, name, anchor)


    def run_visual(self, hours: int = SIMULATION_HOURS):
        """Run the simulation and show an interactive Plotly animation."""
        ticks = int((60 / TICK_MINUTES) * hours)
        self.time = 0
        frames: List[Dict[str, float]] = []
        for _ in range(ticks):
            self.step()
            self.time += 1
            for name, f in self.fluxons.items():
                x, y = self.coords.get(name, (0, 0))
                frames.append({
                    "time": self.time,
                    "x": x,
                    "y": y,
                    "intersection": name,
                    "volume": f.v,
                })
        df = pd.DataFrame(frames)
        fig = px.scatter(
            df,
            x="x",
            y="y",
            animation_frame="time",
            animation_group="intersection",
            size="volume",
            color="intersection",
            range_x=[-1.5, 1.5],
            range_y=[-1.5, 1.5],
            hover_name="intersection",
        )
        fig.update_layout(
            title="Bangalore Traffic Simulation",
            showlegend=False,
            xaxis_visible=False,
            yaxis_visible=False,
        )
        fig.show()

    def schedule_anchor(self, when: int, name: str, anchor: Anchor):
        self.anchors.append((when, name, anchor))
        # sort by Lamport timestamp (when) and precedence
        self.anchors.sort(key=lambda x: (x[0], x[2].precedence))

    def step(self):
        # process anchors
        while self.anchors and self.anchors[0][0] == self.time:
            _, name, anchor = self.anchors.pop(0)
            f = self.fluxons[name]
            f.u += anchor.delta_u
            f.anchor_until = self.time + anchor.delta_tau

        # update fluxons
        for name, f in self.fluxons.items():
            prev_v = f.v
            # simple random walk for vehicle count
            f.v = max(0, f.v + random.randint(-20, 20))
            # apply congestion effects if intersection is a choke point
            cp = CHOKE_POINTS.get(name)
            if cp and f.v > cp["capacity"]:
                excess = f.v - cp["capacity"]
                f.v -= cp["factor"] * excess
            f.update_volatility(prev_v)
            f.update_window()
            if f.anchor_until <= self.time:
                f.anchor_until = 0
            f.update_velocity(prev_v)

    def proximity(self, a: Fluxon, b: Fluxon, eps: float) -> bool:
        return abs(a.v - b.v) <= eps * max(a.s, b.s, PROX_EPSILON0)

    def drift(self, a: Fluxon, b: Fluxon) -> float:
        dv = abs(a.v - b.v)
        ddv_dt = -(dv - abs((a.v - a.u) - (b.v - b.u)))
        denom = abs(a.u) + abs(b.u) + DRIFT_DELTA
        return ddv_dt / denom

    def run(self, hours: int = SIMULATION_HOURS):
        ticks = int((60 / TICK_MINUTES) * hours)
        for _ in range(ticks):
            self.step()
            self.time += 1
            self.report()

    def report(self):
        print(f"Time {self.time} min")
        for name, f in self.fluxons.items():
            print(f"  {name}: v={f.v:.1f}, u={f.u:.2f}, s={f.s:.2f}, W={f.W:.1f}")
        # example drift between first two intersections
        names = list(self.fluxons.keys())
        if len(names) >= 2:
            d = self.drift(self.fluxons[names[0]], self.fluxons[names[1]])
            print(f"  Drift {names[0]} ↔ {names[1]}: {d:.3f}")
        print()

if __name__ == "__main__":
    import argparse

    intersections = [
        "Hebbal",
        "Silk_Board",
        "KR_Puram",
        "Electronic_City",
    ]
    parser = argparse.ArgumentParser(description="Bangalore traffic simulation")
    parser.add_argument("--visual", action="store_true", help="show Plotly animation")
    parser.add_argument("--hours", type=int, default=1, help="simulation duration in hours")
    args = parser.parse_args()

    sim = TrafficSimulation(intersections, coords=COORDS)

    # Example anchor: roadwork near Silk Board at t=30 minutes
    sim.schedule_anchor(
        when=30,
        name="Silk_Board",
        anchor=Anchor(id="roadwork", delta_u=-5, delta_tau=10, precedence=1),
    )

    if args.visual:
        sim.run_visual(hours=args.hours)
    else:
        sim.run(hours=args.hours)
