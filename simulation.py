import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from copy import deepcopy

from plotly.subplots import make_subplots

import plotly.graph_objects as go

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

# Simple routes connecting major intersections
ROUTES = [
    ("Hebbal", "KR_Puram"),
    ("KR_Puram", "Silk_Board"),
    ("Silk_Board", "Electronic_City"),
    ("Electronic_City", "Hebbal"),
]

# Alternative destinations used when traffic is diverted away from a route
DIVERSION_MAP = {
    "Hebbal": "Silk_Board",
    "KR_Puram": "Electronic_City",
    "Silk_Board": "Hebbal",
    "Electronic_City": "KR_Puram",
}

@dataclass
class Fluxon:
    v: float                  # latest numeric value (e.g., vehicle count)
    u: float = 0.0            # smoothed velocity
    s: float = 0.0            # volatility
    W: float = WINDOW_MIN     # adaptive window length
    anchor_until: int = 0     # if > current tick, velocity frozen
    active_anchor: Optional[str] = None  # id of active anchor
    target: Optional[str] = None         # diversion target if any

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
    target: Optional[str] = None

class TrafficSimulation:
    def __init__(self, intersections: List[str], coords: Dict[str, Tuple[float, float]], routes: List[Tuple[str, str]] = ROUTES):
        self.time = 0
        self.fluxons: Dict[str, Fluxon] = {
            name: Fluxon(v=random.randint(200, 500)) for name in intersections
        }
        self.coords = coords
        self.routes = routes
        self.anchors: List[Tuple[int, str, Anchor]] = []  # queue of (time, name, anchor)
        # cooldown tracker so mitigation anchors are not reissued every tick
        self.last_decision_tick: Dict[str, int] = {name: -1000 for name in intersections}
        self.active_diversion: Dict[str, str] = {}


    def run_visual(self, hours: int = SIMULATION_HOURS):
        """Run the simulation and generate an interactive Plotly animation with
        time-series charts and color legend."""
        ticks = int((60 / TICK_MINUTES) * hours)
        self.time = 0

        history: Dict[str, List[float]] = {name: [] for name in self.fluxons}
        states: List[Dict[str, Fluxon]] = []
        anchor_events: List[Tuple[int, str, Anchor]] = []
        max_v = 0.0

        for _ in range(ticks):
            triggered = self.step()
            if triggered:
                anchor_events.extend([(self.time, n, a) for n, a in triggered])

            snapshot = {name: deepcopy(f) for name, f in self.fluxons.items()}
            states.append(snapshot)

            for name, f in self.fluxons.items():
                history[name].append(f.v)
                max_v = max(max_v, f.v)

            self.time += 1

        if not states:
            return

        time_axis = list(range(1, ticks + 1))

        frames: List[go.Frame] = []
        for t_idx, state in enumerate(states, start=1):
            node_x, node_y, node_text = [], [], []
            for name, f in state.items():
                x, y = self.coords.get(name, (0, 0))
                node_x.append(x)
                node_y.append(y)
                anchor_active = f.anchor_until > t_idx
                info = f"{name}<br>v={int(f.v)} u={f.u:.1f} W={f.W:.1f}"
                if anchor_active and f.active_anchor:
                    if f.target:
                        info += f"<br>{f.active_anchor}→{f.target}"
                    else:
                        info += f"<br>{f.active_anchor}"
                node_text.append(info)

            edge_traces = []
            for start, end in self.routes:
                x0, y0 = self.coords[start]
                x1, y1 = self.coords[end]
                f_start = state[start]
                f_end = state[end]
                load = (f_start.v + f_end.v) / 2
                width = max(1, load / 150)
                d = self.drift(f_start, f_end)
                if d >= 0.85:
                    color = "blue"
                elif d >= 0.45:
                    color = "green"
                elif d >= 0.05:
                    color = "orange"
                else:
                    color = "red"
                edge_traces.append(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(width=width, color=color),
                        hovertext=f"drift {d:.2f}",
                        hoverinfo="text",
                        showlegend=False,
                    )
                )

            node_colors = [
                "orange" if state[name].anchor_until > t_idx else "blue"
                for name in state
            ]
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=node_text,
                textposition="top center",
                marker=dict(size=12, color=node_colors),
                showlegend=False,
            )

            vert_line = go.Scatter(
                x=[t_idx, t_idx],
                y=[0, max_v],
                mode="lines",
                line=dict(color="black", dash="dot"),
                showlegend=False,
                xaxis="x2",
                yaxis="y2",
            )

            past_events = [e for e in anchor_events if e[0] < t_idx]
            anchor_trace = go.Scatter(
                x=[ev_time + 1 for ev_time, n, _ in past_events],
                y=[history[n][ev_time] for ev_time, n, _ in past_events],
                text=[f"⚑ {anc.id} @ {n}" + (f"→{anc.target}" if anc.target else "") for ev_time, n, anc in past_events],
                mode="markers",
                marker=dict(color="purple", size=8, symbol="diamond"),
                hoverinfo="text",
                name="anchor",
                xaxis="x2",
                yaxis="y2",
                showlegend=False,
            )

            frames.append(go.Frame(data=[node_trace] + edge_traces + [anchor_trace, vert_line], name=str(t_idx)))

        # build static traces for time series
        line_traces: List[go.Scatter] = []
        for name, series in history.items():
            line_traces.append(
                go.Scatter(
                    x=time_axis,
                    y=series,
                    mode="lines",
                    name=name,
                    xaxis="x2",
                    yaxis="y2",
                )
            )


        legend_traces = [
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="blue"), name="stable drift"),
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="green"), name="plausible drift"),
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="orange"), name="possible drift"),
            go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="red"), name="void drift"),
            go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="purple", size=8, symbol="diamond"), name="anchor"),
        ]

        fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], vertical_spacing=0.08)

        for tr in frames[0].data:
            fig.add_trace(tr, row=1, col=1)

        for tr in line_traces:
            fig.add_trace(tr, row=2, col=1)

        for tr in legend_traces:
            fig.add_trace(tr, row=1, col=1)

        fig.frames = frames

        fig.update_layout(
            xaxis=dict(range=[-1.5, 1.5], visible=False),
            yaxis=dict(range=[-1.5, 1.5], visible=False),
            xaxis2=dict(title="Time (min)", range=[0, ticks + 1]),
            yaxis2=dict(title="Vehicle count", range=[0, max_v * 1.1]),
            title="Bangalore Traffic Simulation",
            showlegend=True,
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        }
                    ],
                }
            ],
        )

        import os
        import plotly.io as pio
        output_path = os.path.abspath("traffic_simulation.html")
        pio.write_html(fig, file=output_path, auto_open=True)

    def schedule_anchor(self, when: int, name: str, anchor: Anchor):
        self.anchors.append((when, name, anchor))
        # sort by Lamport timestamp (when) and precedence
        self.anchors.sort(key=lambda x: (x[0], x[2].precedence))

    def manage_traffic(self):
        """Issue mitigation anchors based on congestion and drift."""
        for name, f in self.fluxons.items():
            cp = CHOKE_POINTS.get(name)
            if cp and f.v > cp["capacity"] * 1.2 and self.time - self.last_decision_tick[name] > 10:
                # extend green signal to relieve congestion
                anc = Anchor(id="extend_green", delta_u=5, delta_tau=0, precedence=2)
                self.schedule_anchor(self.time, name, anc)
                self.last_decision_tick[name] = self.time

        for start, end in self.routes:
            d = self.drift(self.fluxons[start], self.fluxons[end])
            last = self.last_decision_tick[start]
            if d < 0.05 and start not in self.active_diversion and self.time - last > 10:
                # divert traffic away from unstable link
                target = DIVERSION_MAP.get(start, end)
                anc = Anchor(id="divert", delta_u=-5, delta_tau=0, precedence=2, target=target)
                self.schedule_anchor(self.time, start, anc)
                self.active_diversion[start] = target
                self.last_decision_tick[start] = self.time
            elif d >= 0.45 and start in self.active_diversion and self.time - last > 10:
                # clear previous diversion when drift improves
                target = self.active_diversion[start]
                anc = Anchor(id="clear", delta_u=5, delta_tau=0, precedence=2, target=target)
                self.schedule_anchor(self.time, start, anc)
                del self.active_diversion[start]
                self.last_decision_tick[start] = self.time

    def step(self) -> List[Tuple[str, Anchor]]:
        """Advance simulation by one tick and return triggered anchors."""
        # issue any mitigation anchors based on current state
        self.manage_traffic()

        triggered: List[Tuple[str, Anchor]] = []
        # process anchors
        while self.anchors and self.anchors[0][0] == self.time:
            _, name, anchor = self.anchors.pop(0)
            triggered.append((name, anchor))
            f = self.fluxons[name]
            f.u += anchor.delta_u
            f.anchor_until = self.time + anchor.delta_tau
            f.active_anchor = anchor.id
            f.target = anchor.target
            if anchor.id == "divert" and anchor.target and anchor.target in self.fluxons:
                target_f = self.fluxons[anchor.target]
                shift = 10
                f.v = max(0, f.v - shift)
                target_f.v += shift

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
                f.active_anchor = None
                f.target = None
            f.update_velocity(prev_v)

        return triggered

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
