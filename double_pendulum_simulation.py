import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

# ==========================================
# 1. USER CONFIGURATION (EDIT THESE)
# ==========================================
params = {
    "M1": 10.0,   # mass 1 (kg)
    "M2": 10.0,   # mass 2 (kg)
    "L1": 1.0,   # rod 1 length (m)
    "L2": 1.0,   # rod 2 length (m)
    "g": 9.81,   # gravity (m/s^2)
}

# Initial conditions at t = 0: [theta1, theta2, dtheta1/dt, dtheta2/dt]
# Angles in radians, angular velocities in rad/s
y0 = [2.4, 2.7, 0.0, 0.0]

t_span = (0.0, 12.0)  # start and end time (s)
fps = 30              # animation frames per second
trail_len = 120       # number of past frames in Bob 2's trail (set to None for full trail)

# ==========================================
# 2. LAGRANGIAN & EULER-LAGRANGE IN SYMPY
# ==========================================
t = sp.symbols("t")
M1, M2, L1, L2, g = sp.symbols("M1 M2 L1 L2 g", positive=True)

theta1 = sp.Function("theta1")(t)
theta2 = sp.Function("theta2")(t)
dtheta1 = theta1.diff(t)
dtheta2 = theta2.diff(t)
ddtheta1 = theta1.diff(t, 2)
ddtheta2 = theta2.diff(t, 2)

y1 = -L1 * sp.cos(theta1)
x1 = L1 * sp.sin(theta1)
y2 = y1 - L2 * sp.cos(theta2)
x2 = x1 + L2 * sp.sin(theta2)

x1_dot, y1_dot = x1.diff(t), y1.diff(t)
x2_dot, y2_dot = x2.diff(t), y2.diff(t)

T = sp.Rational(1, 2) * (M1 * (x1_dot**2 + y1_dot**2) + M2 * (x2_dot**2 + y2_dot**2))
V = -M1 * g * L1 * sp.cos(theta1) - M2 * g * (L1 * sp.cos(theta1) + L2 * sp.cos(theta2))
Lg = T - V


def euler_lagrange(L_expr, q, dq):
    q_dot_sym = sp.Dummy("q_dot")
    dL_ddq = sp.diff(L_expr.subs(dq, q_dot_sym), q_dot_sym).subs(q_dot_sym, dq)
    dL_dq = sp.diff(L_expr, q)
    return sp.Eq(sp.diff(dL_ddq, t) - dL_dq, 0)


eqn1 = euler_lagrange(Lg, theta1, dtheta1)
eqn2 = euler_lagrange(Lg, theta2, dtheta2)

sol = sp.solve([eqn1, eqn2], (ddtheta1, ddtheta2), simplify=True)
subs_map = {
    M1: params["M1"],
    M2: params["M2"],
    L1: params["L1"],
    L2: params["L2"],
    g: params["g"],
}

acc1_fn = sp.lambdify((theta1, theta2, dtheta1, dtheta2), sol[ddtheta1].subs(subs_map), "numpy")
acc2_fn = sp.lambdify((theta1, theta2, dtheta1, dtheta2), sol[ddtheta2].subs(subs_map), "numpy")

# ==========================================
# 3. NUMERICAL INTEGRATION
# ==========================================
def rhs(t_val, state):
    th1, th2, om1, om2 = state
    return [
        om1,
        om2,
        float(acc1_fn(th1, th2, om1, om2)),
        float(acc2_fn(th1, th2, om1, om2)),
    ]


num_frames = int((t_span[1] - t_span[0]) * fps)
t_eval = np.linspace(t_span[0], t_span[1], num_frames)
sol_ivp = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10)

th1_arr, th2_arr = sol_ivp.y[0], sol_ivp.y[1]
L1_val, L2_val = params["L1"], params["L2"]

x1_arr = L1_val * np.sin(th1_arr)
y1_arr = -L1_val * np.cos(th1_arr)
x2_arr = x1_arr + L2_val * np.sin(th2_arr)
y2_arr = y1_arr - L2_val * np.cos(th2_arr)

# ==========================================
# 4. MATPLOTLIB ANIMATION
# ==========================================
fig, ax = plt.subplots(figsize=(6.5, 6.5))
lim = L1_val + L2_val + 0.25
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.grid(True, alpha=0.35)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Double Pendulum Simulation")

# Pivot marker
ax.plot(0, 0, "ko", markersize=6)

# Rods, bobs, trail, and timer text
(rod_line,) = ax.plot([], [], "o-", lw=2.5, color="#1f2937", markersize=8, markerfacecolor="#ef4444")
(trail_line,) = ax.plot([], [], "-", lw=1.2, color="#3b82f6", alpha=0.65)
time_text = ax.text(0.03, 0.95, "", transform=ax.transAxes, fontsize=11, verticalalignment="top")


def init():
    rod_line.set_data([], [])
    trail_line.set_data([], [])
    time_text.set_text("")
    return rod_line, trail_line, time_text


def update(frame):
    xs = [0.0, x1_arr[frame], x2_arr[frame]]
    ys = [0.0, y1_arr[frame], y2_arr[frame]]
    rod_line.set_data(xs, ys)

    start_idx = 0 if trail_len is None else max(0, frame - trail_len)
    trail_line.set_data(x2_arr[start_idx : frame + 1], y2_arr[start_idx : frame + 1])

    time_text.set_text(f"t = {sol_ivp.t[frame]:.2f} s")
    return rod_line, trail_line, time_text


ani = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    init_func=init,
    interval=1000 / fps,
    blit=True,
)

# Optional: save as MP4 (requires ffmpeg) or GIF (requires pillow)
# ani.save("double_pendulum.gif", writer="pillow", fps=fps)

plt.tight_layout()
plt.show()