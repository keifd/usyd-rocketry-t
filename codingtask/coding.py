import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

# CONSTANTS
GRAVITY = 9.8           # gravity m/s**2
R = 287.058             # specific gas constant for air J/(kg*K)
T = 24 + 273.15         # temperature in Kelvin
SPEED_OF_SOUND = 343    # m/s
PI = 3.14               # number pi
OUTER_DIAMETER = 0.016  # m
INNER_DIAMETER = 0.008  # m
THICKNESS = 0.011       # m

# LOAD DATA
cd_df = pd.read_csv("cd_mach.csv")                           
mass_df = pd.read_csv("mass_time.csv")                       
flight_df = pd.read_csv("flightdata.csv", low_memory=False)  

# Remove measurements before liftoff
flight_df_after_liftoff = flight_df[flight_df["time"] >= 0].reset_index(drop=True)

# Merge flight and mass data by nearest time
df = pd.merge_asof(flight_df_after_liftoff.sort_values("time"),
                   mass_df.sort_values("time"), on="time", direction="nearest")

# Velocity by integrating acceleration
df["velocity"] = cumulative_trapezoid(df["acceleration"], df["time"], initial=0)

# Mach and Cd
df["mach"] = df["velocity"] / SPEED_OF_SOUND
df["cd"] = np.interp(df["mach"], cd_df["mach"], cd_df["cd"])

# Air density
df["rho"] = df["pressure"] / (R * T)

# Reference area
A = PI * (OUTER_DIAMETER / 2) ** 2

# Drag
df["drag"] = 0.5 * df["rho"] * df["velocity"]**2 * df["cd"] * A

# Thrust including drag
df["thrust"] = df["mass"] * df["acceleration"] + df["mass"] * GRAVITY + df["drag"]

# Filter for time 0-10 seconds
df_plot = df[(df["time"] >= 0) & (df["time"] <= 10)]

# PLOTTING
plt.figure(figsize=(12,8))

# Thrust vs time
plt.subplot(2,2,1)
plt.plot(df_plot["time"], df_plot["thrust"], label="Thrust", color="r")
plt.xlabel("Time (s)")
plt.ylabel("Thrust (N)")
plt.title("Thrust vs Time")
plt.grid(True)

# Acceleration vs time
plt.subplot(2,2,2)
plt.plot(df_plot["time"], df_plot["acceleration"], label="Acceleration", color="b")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Acceleration vs Time")
plt.grid(True)

# Velocity vs time
plt.subplot(2,2,3)
plt.plot(df_plot["time"], df_plot["velocity"], label="Velocity", color="g")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity vs Time")
plt.grid(True)

# Altitude vs time
plt.subplot(2,2,4)
plt.plot(df_plot["time"], df_plot["altitude"], label="Altitude", color="m")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.title("Altitude vs Time")
plt.grid(True)

plt.tight_layout()
plt.show()
