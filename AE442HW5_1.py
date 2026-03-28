import numpy as np
import matplotlib.pyplot as plt

# constants
g0_earth = 9.80665  # m/s^2
mu_moon = 4902.8e9  # m^3/s^2
R_moon = 1740e3  # m

# LM ascent stage
m0 = 4780.0  # kg
mp = 2375.0  # kg
mdry = m0 - mp  # kg
T = 15.57e3  # N
Isp = 311.0  # s

# shutdown targets
vp = 1712.31  # m/s
v_min = vp
v_max = 1.01 * vp
h_min = 25e3  # m
h_max = 35e3  # m
gamma_tol_deg = 3.0

# integration settings
dt = 0.1
t_max = 800.0
coast_extra = 20.0

kick_deg = 7.9

# initial conditions
t = 0.0
h = 0.0
x = 0.0
v = 0.0
gamma = np.deg2rad(90.0)
m = m0
dv_grav = 0.0

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

# integration loop
for i in range(int(t_max / dt)):

    r = R_moon + h
    g = mu_moon / r**2

    thrust = T if engine_on else 0.0

    if (not kicked) and (h >= 100.0):
        gamma -= np.deg2rad(kick_deg)
        kicked = True

    if not kicked:
        gamma_dot = 0.0
        gamma = np.deg2rad(90.0)
    else:
        gamma_dot = -(g / v - v / r) * np.cos(gamma) if v > 1e-6 else 0.0

    v_dot = thrust / m - g * np.sin(gamma)
    h_dot = v * np.sin(gamma)
    x_dot = (R_moon / r) * v * np.cos(gamma)
    m_dot = -thrust / (Isp * g0_earth) if engine_on else 0.0

    t += dt
    h += h_dot * dt
    x += x_dot * dt
    v += v_dot * dt
    gamma += gamma_dot * dt
    m += m_dot * dt

    if engine_on:
        dv_grav += g * np.sin(gamma) * dt

    t_hist.append(t)
    h_hist.append(h)
    x_hist.append(x)
    v_hist.append(v)
    gamma_hist.append(np.rad2deg(gamma))
    a_hist.append(v_dot)

    if engine_on and m <= mdry:
        m = mdry
        engine_on = False
        propellant_out = True
        shutdown_done = True
        shutdown_time = t
        shutdown_index = len(t_hist) - 1

    gamma_deg = np.rad2deg(gamma)
    if engine_on:
        cond_gamma = abs(gamma_deg) <= gamma_tol_deg
        cond_v = v_min <= v <= v_max
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
print(f"\nkick_deg = {kick_deg:.1f}")

if conditions_met:
    print("SUCCESS")
elif propellant_out:
    print("PROPELLANT DEPLETED")
else:
    print("NO SHUTDOWN")

if shutdown_done:
    i = shutdown_index
    print(f"shutdown time = {shutdown_time:.1f} s")
    print(f"shutdown altitude = {h_hist[i]/1e3:.2f} km")
    print(f"shutdown speed = {v_hist[i]:.2f} m/s")
    print(f"shutdown gamma = {gamma_hist[i]:.2f} deg")

dv_grav_estimated = 1.625 * shutdown_time if shutdown_done else np.nan

print(f"\nGravity loss (exact) = {dv_grav:.1f} m/s")
print(f"Gravity loss (estimated) = {dv_grav_estimated:.1f} m/s")
print(f"Difference = {abs(dv_grav_estimated - dv_grav):.1f} m/s")

# plots
t_arr = np.array(t_hist)
h_arr = np.array(h_hist)
x_arr = np.array(x_hist)
v_arr = np.array(v_hist)
gamma_arr = np.array(gamma_hist)
a_arr = np.array(a_hist)

fig, axs = plt.subplots(5, 1, figsize=(8, 8), sharex=True)

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

axs[4].plot(t_arr, x_arr / 1e3)
axs[4].set_ylabel("Downrange (km)")
axs[4].set_xlabel("Time (s)")
axs[4].grid(True)

if shutdown_done:
    for ax in axs:
        ax.axvline(shutdown_time, color="red", linestyle="--", linewidth=1.2, label="shutdown")
    axs[0].legend()

plt.suptitle(f"LM Ascent Simulation", fontsize=13)
plt.tight_layout()
plt.show()