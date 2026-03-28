import numpy as np
import matplotlib.pyplot as plt

# constants
g0_earth = 9.80665  # m/s^2
mu_earth = 3.986e14  # m^3/s^2
R_earth = 6378e3  # m
h0 = 7194.0  # m
rho0 = 1.225  # kg/m^3
p0 = 101325.0  # Pa
gamma_air = 1.4

# SSTO specs
m0 = 3.0e6  # kg
mf = 1.0e5  # kg
T_max = 3.3e7  # N
Isp = 450.0  # s
CD = 0.2
D_ref = 10.0  # m
S_ref = np.pi * (D_ref / 2.0)**2

# launch site speed
v_eq = 465.0  # m/s
lat_deg = 28.45
v_LS = v_eq * np.cos(np.deg2rad(lat_deg))

# shutdown targets
h_target = 200e3
h_min = 190e3
h_max = 210e3
gamma_tol_deg = 1.0

r_target = R_earth + h_target
v_circ_target = np.sqrt(mu_earth / r_target)
v_req_target = v_circ_target - v_LS

print(f"Launch site speed = {v_LS:.1f} m/s")
print(f"Circular orbit speed = {v_circ_target:.1f} m/s")
print(f"Required shutdown speed = {v_req_target:.1f} m/s")

# integration settings
dt = 0.1
t_max = 1200.0
coast_extra = 20.0

kick_altitude = 900.0  # m
kick_deg = 0.080  # deg

use_throttle = True
throttle_start_1 = 100e3  # m
throttle_frac_1 = 0.9
throttle_start_2 = 130e3  # m
throttle_frac_2 = 0.349

# initial conditions
t = 0.0
h = 0.0
x = 0.0
v = 0.0
gamma = np.deg2rad(90.0)
m = m0

dv_grav = 0.0
dv_drag = 0.0

engine_on = True
kicked = False
conditions_met = False
propellant_out = False
shutdown_done = False
shutdown_time = None
shutdown_index = None

# history
t_hist = []
h_hist = []
x_hist = []
v_hist = []
gamma_hist = []
a_hist = []
rho_hist = []
D_hist = []
W_hist = []
m_hist = []
mp_hist = []
q_hist = []
throttle_hist = []
vreq_hist = []

# integration loop
for i in range(int(t_max / dt)):

    r = R_earth + h
    g = mu_earth / r**2

    rho = rho0 * np.exp(-h / h0)
    p = p0 * np.exp(-h / h0)

    throttle = 1.0
    if use_throttle:
        if h >= throttle_start_1:
            throttle = throttle_frac_1
        if h >= throttle_start_2:
            throttle = throttle_frac_2

    thrust = T_max * throttle if engine_on else 0.0

    if (not kicked) and (h >= kick_altitude):
        gamma -= np.deg2rad(kick_deg)
        kicked = True

    q = 0.5 * rho * v**2
    D = CD * S_ref * q
    W = m * g

    if not kicked:
        gamma_dot = 0.0
        gamma = np.deg2rad(90.0)
    else:
        gamma_dot = (v / r - g / v) * np.cos(gamma) if v > 1e-6 else 0.0

    v_dot = (thrust - D) / m - g * np.sin(gamma)
    h_dot = v * np.sin(gamma)
    x_dot = (R_earth / r) * v * np.cos(gamma)
    m_dot = -thrust / (Isp * g0_earth) if engine_on else 0.0

    if engine_on:
        dv_grav += g * np.sin(gamma) * dt
        dv_drag += (D / m) * dt

    v_req_now = np.sqrt(mu_earth / (R_earth + max(h, 0.0))) - v_LS

    t_hist.append(t)
    h_hist.append(h)
    x_hist.append(x)
    v_hist.append(v)
    gamma_hist.append(np.rad2deg(gamma))
    a_hist.append(v_dot)
    rho_hist.append(rho)
    D_hist.append(D)
    W_hist.append(W)
    m_hist.append(m)
    mp_hist.append(m0 - m)
    q_hist.append(q)
    throttle_hist.append(throttle if engine_on else 0.0)
    vreq_hist.append(v_req_now)

    t += dt
    h += h_dot * dt
    x += x_dot * dt
    v += v_dot * dt
    gamma += gamma_dot * dt
    m += m_dot * dt

    if engine_on and m <= mf:
        m = mf
        engine_on = False
        propellant_out = True
        shutdown_done = True
        shutdown_time = t
        shutdown_index = len(t_hist) - 1

    if engine_on:
        gamma_deg_now = np.rad2deg(gamma)
        v_req_now = np.sqrt(mu_earth / (R_earth + h)) - v_LS
        v_min_now = v_req_now
        v_max_now = 1.01 * v_req_now

        cond_gamma = abs(gamma_deg_now) <= gamma_tol_deg
        cond_v = v_min_now <= v <= v_max_now
        cond_h = h_min <= h <= h_max

        if cond_gamma and cond_v and cond_h:
            engine_on = False
            conditions_met = True
            shutdown_done = True
            shutdown_time = t
            shutdown_index = len(t_hist) - 1

    if shutdown_done and t >= shutdown_time + coast_extra:
        break

# results
print(f"\nkick_deg = {kick_deg:.3f}")
print(f"kick altitude = {kick_altitude/1e3:.1f} km")
print(f"dt = {dt:.2f} s")

if conditions_met:
    print("\nSUCCESS")
elif propellant_out:
    print("\nPROPELLANT DEPLETED")
else:
    print("\nNO SHUTDOWN")

if shutdown_done:
    si = shutdown_index
    v_req_shutdown = np.sqrt(mu_earth / (R_earth + h_hist[si])) - v_LS
    v_max_shutdown = 1.01 * v_req_shutdown

    print(f"shutdown time = {shutdown_time:.1f} s")
    print(f"shutdown altitude = {h_hist[si]/1e3:.2f} km")
    print(f"shutdown speed = {v_hist[si]:.2f} m/s")
    print(f"shutdown gamma = {gamma_hist[si]:.3f} deg")
    print(f"remaining mass = {m_hist[si]:.0f} kg")
    print(f"propellant used = {mp_hist[si]:.0f} kg")
    print(f"throttle at shutdown = {throttle_hist[si]:.2f}")

# max-q
q_arr = np.array(q_hist)
t_arr = np.array(t_hist)
h_arr = np.array(h_hist)
v_arr = np.array(v_hist)
rho_arr = np.array(rho_hist)
x_arr = np.array(x_hist)
gamma_arr = np.array(gamma_hist)
a_arr = np.array(a_hist)

iq = np.argmax(q_arr)
q_max = q_arr[iq]
t_qmax = t_arr[iq]
h_qmax = h_arr[iq]
v_qmax = v_arr[iq]

a_sound = np.sqrt(gamma_air * p0 * np.exp(-h_qmax / h0) / rho_arr[iq])
M_qmax = v_qmax / a_sound

print(f"\nMax-Q:")
print(f"t_q-max = {t_qmax:.1f} s")
print(f"q_max = {q_max/1e3:.3f} kPa")
print(f"h_q-max = {h_qmax/1e3:.2f} km")
print(f"v_q-max = {v_qmax:.1f} m/s")
print(f"M_q-max = {M_qmax:.2f}")

# losses
if shutdown_done:
    dv_grav_est = g0_earth * shutdown_time
else:
    dv_grav_est = np.nan

print(f"\nGravity loss (exact) = {dv_grav:.1f} m/s")
print(f"Gravity loss (estimated) = {dv_grav_est:.1f} m/s")
print(f"Drag loss (exact) = {dv_drag:.1f} m/s")

# Plot 1: v, h, gamma, dv/dt, q vs time
fig1, axs = plt.subplots(5, 1, figsize=(8, 8), sharex=True)

axs[0].plot(t_arr, v_arr)
axs[0].set_ylabel("Speed (m/s)")
axs[0].grid(True)

axs[1].plot(t_arr, h_arr / 1e3)
axs[1].set_ylabel("Altitude (km)")
axs[1].grid(True)

axs[2].plot(t_arr, gamma_arr)
axs[2].axhline(gamma_tol_deg, color="gray", linestyle="--", linewidth=0.8)
axs[2].axhline(-gamma_tol_deg, color="gray", linestyle="--", linewidth=0.8)
axs[2].set_ylabel("Flight path angle (deg)")
axs[2].grid(True)

axs[3].plot(t_arr, a_arr)
axs[3].set_ylabel("dv/dt (m/s²)")
axs[3].grid(True)

axs[4].plot(t_arr, q_arr / 1e3)
axs[4].set_ylabel("Dynamic pressure (kPa)")
axs[4].set_xlabel("Time (s)")
axs[4].grid(True)

axs[4].annotate(
    f"q-max\n{q_max/1e3:.2f} kPa\nt={t_qmax:.0f}s\nh={h_qmax/1e3:.1f}km",
    xy=(t_qmax, q_max / 1e3),
    xytext=(t_qmax + 20, q_max / 1e3 * 0.8),
    arrowprops=dict(arrowstyle="->"),
    fontsize=8
)

if shutdown_done:
    for ax in axs:
        ax.axvline(shutdown_time, color="red", linestyle="--", linewidth=1.2, label="shutdown")
    axs[0].legend()

plt.suptitle(f"SSTO Earth Ascent", fontsize=13)
plt.tight_layout()
plt.show()

# Plot 2: altitude vs downrange
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(x_arr / 1e3, h_arr / 1e3)
ax2.set_xlabel("Downrange distance (km)")
ax2.set_ylabel("Altitude (km)")
ax2.grid(True)

if shutdown_done:
    sd_x = x_hist[shutdown_index] / 1e3
    sd_h = h_hist[shutdown_index] / 1e3
    ax2.plot(sd_x, sd_h, 'ro')
    ax2.axvline(sd_x, color="red", linestyle="--", linewidth=1.2, label="shutdown")
    ax2.legend()

plt.suptitle(f"SSTO Earth Ascent — Altitude vs Downrange", fontsize=13)
plt.tight_layout()
plt.show()