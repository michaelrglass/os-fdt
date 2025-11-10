import numpy as np
import matplotlib.pyplot as plt

# Endowment
E = 60
v_R = np.linspace(0, E, 300)
v_D = E - v_R

# Logarithmic utilities
u_R = np.log1p(v_R)      # ln(1 + v_R)
u_D = np.log1p(v_D)      # ln(1 + v_D)
u_C = u_R + u_D

# Plot
plt.figure(figsize=(5, 3))
plt.plot(v_R, u_D, label=r"Dictator: $\ln(1 + v_D)$", lw=2)
plt.plot(v_R, u_R, label=r"Recipient: $\ln(1 + v_R)$", lw=2, linestyle='--')
# plt.plot(v_R, u_C, label=r"Total Utility", lw=2, linestyle='--')
plt.xlabel("Recipient’s share ($v_R$)")
plt.ylabel("Utility")
# plt.title("Logarithmic Utilities")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("log_utility_curve.png", dpi=300)
plt.show()