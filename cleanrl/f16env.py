"""
F‑16 Interception Gymnasium Environment
======================================

*   Interceptor  : agent‑controlled F‑16
*   Target       : scripted F‑16 (constant throttle, neutral controls)
*   Observation  : 10‑step rolling buffer, 26 features / step
*   Action space : throttle, aileron, elevator, rudder  ∈ [-1, 1]
*   Reward       : –‖relative_position‖  (with finish bonus)
"""

from __future__ import annotations
import math
from collections import deque
from typing import Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
matplotlib.use("Agg")  # head‑less backend
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused
import io

NAN_SAFE = lambda x: float(np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6))
import jsbsim                              # pip install jsbsim
# Try to silence JSBSim console output; older bindings may not expose this API
try:
    jsbsim.set_logging_level(jsbsim.LogLevel.LOG_WARNING)
except AttributeError:
    # Fallback: disable debug printing if available, otherwise ignore
    try:
        jsbsim.enable_debug_level(0)
    except AttributeError:
        pass

# ---------- constants ----------
OBS_WINDOW          = 10                  # number of timesteps in history buffer
INTCPT_MODEL        = "f16"
TARGET_MODEL        = "f16"
SIM_DT              = 0.02                # 50 Hz (JSBSim default is 0.0083333 s)
MAX_STEPS           = 6_000               # 2 min episode @ 50 Hz
INTERCEPT_DIST_FT   = 300.0               # success radius
CRASH_ALT_FT        =   0.0               # ground collision
FT_PER_DEG_LAT      = 60.0 * 6076.12      # ≈ 364 566 ft
R_EARTH_FT          = 20_925_524.9

# ---------- helper functions ----------
def _deg2rad(deg: float) -> float: return deg * math.pi / 180.0

def geodetic_to_ned(lat_deg: float, lon_deg: float, alt_ft: float,
                    ref_lat_deg: float, ref_lon_deg: float, ref_alt_ft: float) -> np.ndarray:
    """
    Very small‑angle flat‑Earth NED approximation (enough for a few nm).
    """
    d_north = (lat_deg - ref_lat_deg) * FT_PER_DEG_LAT
    mean_lat = _deg2rad(0.5 * (lat_deg + ref_lat_deg))
    d_east  = (lon_deg - ref_lon_deg) * FT_PER_DEG_LAT * math.cos(mean_lat)
    d_down  = ref_alt_ft - alt_ft
    return np.array([d_north, d_east, d_down], dtype=np.float32)

# ---------- environment ----------
class F16InterceptEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # 26 scalars / step  →  (10,26) observation
    _N_FEATURES = 26
    observation_space = spaces.Box(
        low  = -np.inf, high = np.inf, shape=(OBS_WINDOW, _N_FEATURES), dtype=np.float32
    )
    # throttle, aileron, elevator, rudder
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    # -------------------------------
    def __init__(self, e_ref: float | None = None, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self._make_sims()
        self.history: deque[np.ndarray] = deque(maxlen=OBS_WINDOW)
        self.steps  = 0
        # Reference energy: 500 ft/s (~300 kt) at 15 000 ft → ~1.9×10^5 ft²/s²
        self.E_REF = e_ref if e_ref is not None else (
            0.5 * 500.0**2 + 32.174 * 15_000
        )

        # Action memory for smoothness penalty
        self.action_prev = np.zeros(4, dtype=np.float32)
        self.action_curr = np.zeros(4, dtype=np.float32)

    # -------------------------------
    def _make_sims(self):
        # interceptor
        self.sim_i = jsbsim.FGFDMExec(None)
        self.sim_i.set_dt(SIM_DT)
        self.sim_i.load_model(INTCPT_MODEL)

        # target
        self.sim_t = jsbsim.FGFDMExec(None)
        self.sim_t.set_dt(SIM_DT)
        self.sim_t.load_model(TARGET_MODEL)

    # -------------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any]|None = None):
        super().reset(seed=seed)
        # Initial conditions – target straight & level @ 15 000 ft, 350 kt
        tgt_ic = {
            "ic/lat-gc-deg":   0.0,
            "ic/long-gc-deg":  0.0,
            "ic/h-sl-ft":  15_000,
            "ic/psi-true-deg": 90.0,      # east
            "ic/vt-fps":   350 * 1.68781, # kts→fps
        }
        # Interceptor spawns 3 nm behind & 1 000 ft below
        int_ic = tgt_ic.copy()
        int_ic["ic/lat-gc-deg"] -= 3 / 6076.12  # 3 nm south (approx)
        int_ic["ic/h-sl-ft"]     = 14_000

        self._apply_ic(self.sim_t, tgt_ic)
        self._apply_ic(self.sim_i, int_ic)

        self.action_prev = np.zeros(4, dtype=np.float32)
        self.action_curr = np.zeros(4, dtype=np.float32)

        # one JSBSim step to load IC
        self.sim_t.run_ic()
        self.sim_i.run_ic()

        # scripted target controls
        self.sim_t["fcs/throttle-cmd-norm"] = 0.85

        self.history.clear()
        self._push_obs()

        self.steps = 0
        return self._get_obs(), {}

    # -------------------------------
    def _apply_ic(self, sim: jsbsim.FGFDMExec, ic_dict: Dict[str, float]):
        for k, v in ic_dict.items():
            sim[k] = v

    # -------------------------------------------------
    def _heading_vector(self, sim):
        """
        Returns a unit vector (North‑East‑Down frame) that points
        straight out of the jet’s nose, using Euler angles.
        """
        psi   = sim["attitude/psi-rad"]    # yaw   (0 = North, +East)
        theta = sim["attitude/theta-rad"]  # pitch (+nose‑up)
        # roll φ not needed for forward direction
        cth, sth = math.cos(theta), math.sin(theta)
        cps, sps = math.cos(psi),   math.sin(psi)
        # body‑x axis rotated into NED:
        #   x_N =  cosθ cosψ
        #   y_E =  cosθ sinψ
        #   z_D = ‑sinθ         (Down positive)
        vec = np.array([cth * cps, cth * sps, -sth], dtype=np.float32)
        # normalise just in case
        return vec / (np.linalg.norm(vec) + 1e-8)
    # -------------------------------------------------
    
    # -------------------------------
    def _compute_reward(self):
        # -- distance & closure --------------------------------------------------
        dist_now  = np.linalg.norm(self._rel_ned())
        dist_prev = hasattr(self, "_prev_dist") and self._prev_dist or dist_now
        self._prev_dist = dist_now
        delta_d   = dist_prev - dist_now
        r_dist = 0.5 * np.clip(delta_d / 30_000.0, -1.0, 1.0)  # 30 kft normaliser, bounded

        # LOS unit vector
        los_u = self._rel_ned() / (dist_now + 1e-6)
        v_rel = self._extract_state(self.sim_t)["vel_be"] - \
                self._extract_state(self.sim_i)["vel_be"]
        los_rate = np.dot(v_rel, los_u)               # ft/s
        r_closure = 0.3 * np.tanh(los_rate / 1000.0)           # bounded via tanh

        # boresight alignment
        heading_vec = self._heading_vector(self.sim_i)
        r_boresight = 0.2 * np.dot(heading_vec, los_u)

        # ---------- new guidance terms -----------------------------------
        # Line‑of‑sight (LOS) angular velocity
        tgt_state = self._extract_state(self.sim_t)
        if not hasattr(self, "_prev_los"):
            self._prev_los = los_u.copy()
        omega_los = np.linalg.norm(np.cross(self._prev_los, los_u)) / SIM_DT
        self._prev_los = los_u
        r_los = 400.0 * np.exp(-abs(omega_los) * 57.29578 / 2.0)  # rad→deg

        # Lead‑angle bonus (nose vs target velocity direction)
        u_vel_tgt = tgt_state["vel_be"] / (np.linalg.norm(tgt_state["vel_be"]) + 1e-6)
        theta_lead = np.arccos(np.clip(np.dot(heading_vec, u_vel_tgt), -1.0, 1.0))
        r_lead = 300.0 * np.cos(theta_lead)

        # Time‑to‑go estimate (encourages fast closure)
        v_cl = abs(los_rate)
        r_tgo = 200.0 / max(dist_now / v_cl, 1.0)

        # Energy safety penalty
        spd  = self.sim_i["velocities/vt-fps"]
        aoa  = abs(self.sim_i["aero/alpha-deg"])
        pen_E = -500.0 if (spd < 350.0 or aoa > 18.0) else 0.0

        # total specific energy (TSE)
        v   = self.sim_i["velocities/vt-fps"]
        h   = self.sim_i["position/h-sl-ft"]
        m   = 1.0                                     # cancels in normalisation
        g   = 32.174
        tse = 0.5*v*v + g*h
        energy_err = (tse - self.E_REF) / self.E_REF
        r_energy   = 0.1 * np.tanh(energy_err)                 # bounded via tanh

        # control smoothness
        smooth_pen = 0.05 * np.sum(np.abs(self.action_prev - self.action_curr))

        # envelope penalties
        nz   = self.sim_i["accelerations/Nz"]
        pen_env = 0.0
        if abs(nz) > 7:  pen_env -= 50.0 * (abs(nz) - 7)**2
        if aoa > 20:     pen_env -=  5.0 * (aoa   - 20)**2
        if spd < 300:    pen_env -=  2.0 * (300   - spd)**2

        # cap envelope penalty so it cannot explode
        pen_env = np.clip(pen_env, -500.0, 0.0)

        # terminal bonuses / penalties
        bonus = 0.0
        if dist_now <= 300.0:       # successful intercept
            bonus += 1_000.0
        if self.sim_i["position/h-sl-ft"] <= 0 or abs(nz) > 9 or aoa > 25 or spd < 200:
            bonus -= 1_000.0

        total_reward = (r_dist + r_closure + r_boresight + r_energy
                        + r_los + r_lead + r_tgo
                        - smooth_pen + pen_env + pen_E + bonus)

        # Restore small per‑step scale: clip then down‑scale so values are ~‑10…+10
        return float(np.clip(total_reward, -1000.0, 1000.0) * 0.01)
    
    # -------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.action_prev = self.action_curr.copy()
        # --- apply agent action ---
        throttle = float( (action[0] + 1.0) * 0.5 )           # [0,1]
        sim = self.sim_i
        sim["fcs/throttle-cmd-norm"] = np.clip(throttle, 0.0, 1.0)
        sim["fcs/aileron-cmd-norm"]  = float(action[1])
        sim["fcs/elevator-cmd-norm"] = float(action[2])
        sim["fcs/rudder-cmd-norm"]   = float(action[3])

        # --- store *current* command vector -----------------------
        self.action_curr = np.array(
            [throttle, action[1], action[2], action[3]], dtype=np.float32
        )
        
        # --- advance both jets ---
        self.sim_t.run()
        self.sim_i.run()
        self._push_obs()
        self.steps += 1

        # --- termination / safety check BEFORE computing reward -----------
        dist = np.linalg.norm(self._rel_ned())

        # distance explosion or latency wrap → immediate fail
        hard_fail = dist > 1e5

        # envelope checks (height, speed, G, Mach)
        h   = self.sim_i["position/h-sl-ft"]
        v_t = self.sim_i["velocities/vt-fps"]
        mach= self.sim_i["velocities/mach"]
        nz  = abs(self.sim_i["accelerations/Nz"])

        out_of_bounds = (
            h < -100 or h > 60000 or
            v_t < 50  or v_t > 2000 or
            mach > 2.5 or
            nz   > 9.0
        )

        terminated = False
        reward = -1000.0 if (hard_fail or out_of_bounds) else 0.0

        if not (hard_fail or out_of_bounds):
            # safe to compute shaped reward
            reward = self._compute_reward()
        else:
            # skip pushing the corrupted observation
            obs = np.nan_to_num(self._get_obs(), nan=0.0, posinf=1e6, neginf=-1e6)

        # intercept / crash / time limit
        if dist <= INTERCEPT_DIST_FT:
            reward += 1000.0
            terminated = True
        if self.sim_i["position/h-sl-ft"] <= CRASH_ALT_FT:
            reward -= 1000.0
            terminated = True
        if self.steps >= MAX_STEPS:
            terminated = True
        if hard_fail or out_of_bounds:
            terminated = True

        # ensure reward finite
        reward = float(np.nan_to_num(reward, nan=-1000.0, posinf=-1000.0, neginf=-1000.0))
        if not (hard_fail or out_of_bounds):
            # observation has already been pushed safely
            obs = np.nan_to_num(self._get_obs(), nan=0.0, posinf=1e6, neginf=-1e6)
        return obs, reward, terminated, False, {}

    # -------------------------------
    # internal helpers
    def _push_obs(self):
        int_state = self._extract_state(self.sim_i)
        tgt_state = self._extract_state(self.sim_t)

        rel_pos = tgt_state["pos_ned"] - int_state["pos_ned"]
        rel_vel = tgt_state["vel_be"]  - int_state["vel_be"]

        feat = np.concatenate([
            int_state["features"],
            rel_pos,
            rel_vel
        ], dtype=np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
        self.history.append(feat)

    def _get_obs(self) -> np.ndarray:
        while len(self.history) < OBS_WINDOW:
            self.history.appendleft(np.zeros(self._N_FEATURES, dtype=np.float32))
        return np.stack(self.history, axis=0)

    def _rel_ned(self) -> np.ndarray:
        int_state = self._extract_state(self.sim_i)
        tgt_state = self._extract_state(self.sim_t)
        return tgt_state["pos_ned"] - int_state["pos_ned"]

    # -------------------------------
    def _extract_state(self, sim: jsbsim.FGFDMExec) -> Dict[str, np.ndarray]:
        # position (geodetic) → flat‑Earth NED
        lat = NAN_SAFE(sim["position/lat-gc-deg"])
        lon = NAN_SAFE(sim["position/long-gc-deg"])
        h   = NAN_SAFE(sim["position/h-sl-ft"])
        pos_ned = geodetic_to_ned(lat, lon, h,
                                  ref_lat_deg   = 0.0,
                                  ref_lon_deg   = 0.0,
                                  ref_alt_ft    = 0.0)

        # body‑axis velocities (u,v,w)  & inertial (North/East/Down)
        vel_be  = np.array([
            NAN_SAFE(sim["velocities/u-fps"]),
            NAN_SAFE(sim["velocities/v-fps"]),
            NAN_SAFE(sim["velocities/w-fps"])
        ], dtype=np.float32)

        inertial = np.array([
            NAN_SAFE(sim["velocities/v-north-fps"]),
            NAN_SAFE(sim["velocities/v-east-fps"]),
            NAN_SAFE(sim["velocities/v-down-fps"])
        ], dtype=np.float32)

        ang_rates = np.array([
            NAN_SAFE(sim["velocities/p-rad_sec"]),
            NAN_SAFE(sim["velocities/q-rad_sec"]),
            NAN_SAFE(sim["velocities/r-rad_sec"])
        ], dtype=np.float32)

        flight_dyn = np.array([
            NAN_SAFE(sim["aero/alpha-deg"]),
            NAN_SAFE(sim["aero/beta-deg"]),
            NAN_SAFE(sim["velocities/mach"]),
            NAN_SAFE(sim["velocities/vt-fps"])
        ], dtype=np.float32)

        controls = np.array([
            NAN_SAFE(sim["fcs/aileron-pos-rad"]),
            NAN_SAFE(sim["fcs/elevator-pos-rad"]),
            NAN_SAFE(sim["fcs/rudder-pos-rad"]),
            NAN_SAFE(sim["fcs/throttle-pos-norm"])
        ], dtype=np.float32)

        features = np.concatenate([
            pos_ned,                # 3
            vel_be,                 # 3 
            inertial,               # 3
            ang_rates,              # 3
            flight_dyn,             # 4
            controls                # 4
        ], dtype=np.float32)        # = 26 total
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        # hard‑clip extreme magnitudes before returning
        features = np.clip(features, -1e5, 1e5)

        return {
            "pos_ned":  pos_ned,
            "vel_be":   vel_be,
            "features": features
        }

    # -------------------------------
    # -------------------------------------------------
    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """
        Simple head‑less 3‑D scatter of interceptor (red) and target (blue).
        Returns an RGB image (HxWx3 uint8) so Gym RecordVideo can build MP4.
        """
        assert mode == "rgb_array", "Only rgb_array supported"
        int_state = self._extract_state(self.sim_i)
        tgt_state = self._extract_state(self.sim_t)

        p_i = int_state["pos_ned"]
        p_t = tgt_state["pos_ned"]

        fig = plt.figure(figsize=(4, 4), dpi=128)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(0, 0, 0, c="red", s=40, label="Interceptor")        # origin
        ax.scatter(p_t[1]-p_i[1], p_t[0]-p_i[0], -p_t[2]+p_i[2],
                   c="blue", s=40, label="Target")

        ax.set_xlim(-20000, 20000)
        ax.set_ylim(-20000, 20000)
        ax.set_zlim(-5000, 5000)
        ax.set_xlabel("East ft"); ax.set_ylabel("North ft"); ax.set_zlabel("Up ft")
        ax.view_init(elev=20, azim=135)
        ax.legend(loc="upper left", fontsize=6)
        ax.set_title(f"t = {self.steps*SIM_DT:6.1f}s")

        # draw to buffer
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
        plt.close(fig)
        return img

    def close(self):
        del self.sim_i, self.sim_t