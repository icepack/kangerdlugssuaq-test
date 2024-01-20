import maptlotlib.pyplot as plt
import firedrake

filename = "kangerdlugssuaq-initial.h5"
with firedrake.CheckpointFile(filename, "r") as chk:
    mesh = chk.load_mesh()
    q = chk.load_function(mesh, name="log_friction")

fig, axes = plt.subplots()
axes.set_aspect("equal")
colors = firedrake.tripcolor(q, vmin=-8, vmax=+8, cmap="RdBu", axes=axes)
fig.colorbar(colors)
fig.savefig("kangerdlugssuaq.png", bbox_inches="tight")
