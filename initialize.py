import argparse
import subprocess
import numpy as np
import geojson
import rasterio
import xarray
import firedrake
import firedrake.adjoint
from firedrake import exp, ln, sqrt, assemble, Constant, inner, grad, dx
import icepack
from icepack.constants import (
    ice_density as ρ_I, gravity as g, weertman_sliding_law as m
)
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--outline")
parser.add_argument("--num-levels", type=int, default=1)
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--regularization", type=float, default=2.5e3)
parser.add_argument("--output", default="kangerdlugssuaq-initial.h5")
args = parser.parse_args()

# Fetch the glacier outline, generate a mesh, and create function spaces
outline_filename = "kangerdlugssuaq.geojson"
if args.outline:
    outline_filename = args.outline
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)

geometry = icepack.meshing.collection_to_geo(outline)
with open("kangerdlugssuaq.geo", "w") as geometry_file:
    geometry_file.write(geometry.get_code())

command = "gmsh -2 -v 0 -o kangerdlugssuaq.msh kangerdlugssuaq.geo"
subprocess.run(command.split())

coarse_mesh = firedrake.Mesh("kangerdlugssuaq.msh")
mesh_hierarchy = firedrake.MeshHierarchy(coarse_mesh, args.num_levels)
mesh = mesh_hierarchy[-1]
Q = firedrake.FunctionSpace(mesh, "CG", args.degree)
V = firedrake.VectorFunctionSpace(mesh, "CG", args.degree)

print("done generating mesh")

# Read the thickness and surface data
bedmachine = xarray.open_dataset(icepack.datasets.fetch_bedmachine_greenland())
h_obs = icepack.interpolate(bedmachine["thickness"], Q)
s_obs = icepack.interpolate(bedmachine["surface"], Q)

def smoothe(q_obs, λ):
    q = q_obs.copy(deepcopy=True)
    J = 0.5 * ((q - q_obs)**2 + Constant(λ)**2 * inner(grad(q), grad(q))) * dx
    F = firedrake.derivative(J, q)
    firedrake.solve(F == 0, q)
    return q

λ = 2e3
h = smoothe(h_obs, λ)
s = smoothe(s_obs, λ)

# Read all the raw velocity data
coords = np.array(list(geojson.utils.coords(outline)))
delta = 2.5e3
extent = {
    "left": coords[:, 0].min() - delta,
    "right": coords[:, 0].max() + delta,
    "bottom": coords[:, 1].min() - delta,
    "top": coords[:, 1].max() + delta,
}
measures_filenames = icepack.datasets.fetch_measures_greenland()

velocity_data = {}
for key in ["vx", "vy", "ex", "ey"]:
    filename = [f for f in measures_filenames if key in f][0]
    with rasterio.open(filename, "r") as source:
        window = rasterio.windows.from_bounds(
            **extent, transform=source.transform
        ).round_lengths().round_offsets()
        transform = source.window_transform(window)
        velocity_data[key] = source.read(indexes=1, window=window)

print("done reading observational data")

# Find the points that are inside the domain and create a point cloud
indices = np.array(
    [
        (i, j)
        for i in range(window.width)
        for j in range(window.height)
        if (
            mesh.locate_cell(transform * (i, j), tolerance=1e-8) and
            velocity_data["ex"][j, i] >= 0.0
        )
    ]
)
xs = np.array([transform * idx for idx in indices])
point_set = firedrake.VertexOnlyMesh(
    mesh, xs, missing_points_behaviour="error"
)

Δ = firedrake.FunctionSpace(point_set, "DG", 0)
if hasattr(point_set, "input_ordering"):
    print("Using vertex re-numbering!")
    Δ_input = firedrake.FunctionSpace(point_set.input_ordering, "DG", 0)
    def gridded_to_point_set(field):
        f_input = firedrake.Function(Δ_input)
        f_input.dat.data[:] = field[indices[:, 1], indices[:, 0]]
        f_output = firedrake.Function(Δ)
        f_output.interpolate(f_input)
        return f_output
else:
    print("No vertex renumbering available!")
    def gridded_to_point_set(field):
        f = firedrake.Function(Δ)
        f.dat.data[:] = field[indices[:, 1], indices[:, 0]]
        return f

u_o = gridded_to_point_set(velocity_data["vx"])
v_o = gridded_to_point_set(velocity_data["vy"])
σ_x = gridded_to_point_set(velocity_data["ex"])
σ_y = gridded_to_point_set(velocity_data["ey"])

# Set up and solve a data assimilation problem to interpolate the sparse
# velocity data to the velocity space. This step is necessary because the
# gridded data are missing some points near the terminus
u = firedrake.Function(V)
N = Constant(len(indices))
area = assemble(Constant(1) * dx(mesh))

def loss_functional(u):
    u_int = firedrake.interpolate(u[0], Δ)
    v_int = firedrake.interpolate(u[1], Δ)
    δu = u_int - u_o
    δv = v_int - v_o
    return 0.5 / N * ((δu / σ_x)**2 + (δv / σ_y)**2) * dx

def regularization(u):
    Ω = Constant(area)
    α = Constant(80.0)
    return 0.5 * α**2 / Ω * inner(grad(u), grad(u)) * dx

problem = icepack.statistics.StatisticsProblem(
    simulation=lambda u: u.copy(deepcopy=True),
    loss_functional=loss_functional,
    regularization=regularization,
    controls=u,
)

estimator = icepack.statistics.MaximumProbabilityEstimator(
    problem,
    gradient_tolerance=5e-5,
    step_tolerance=1e-8,
    max_iterations=50,
)

u = estimator.solve()

print("done interpolating velocity data")

# Make an initial estimate for the basal friction by assuming it supports some
# fraction of the driving stress
τ = firedrake.project(-ρ_I * g * h * grad(s), V)

area = assemble(Constant(1) * dx(mesh))
u_avg = assemble(sqrt(inner(u, u)) * dx) / area
τ_avg = assemble(sqrt(inner(τ, τ)) * dx) / area

frac = Constant(0.5)
C = frac * sqrt(inner(τ, τ)) / sqrt(inner(u, u)) ** (1 / m)
q = firedrake.interpolate(-ln(u_avg ** (1 / m) * C / τ_avg), Q)

def bed_friction(**kwargs):
    u, q = map(kwargs.get, ("velocity", "log_friction"))
    C = Constant(τ_avg) / Constant(u_avg) ** (1 / m) * exp(-q)
    return icepack.models.friction.bed_friction(velocity=u, friction=C)

# Compute an initial estimate for the ice velocity
T = firedrake.Constant(260.0)
A = icepack.rate_factor(T)

flow_model = icepack.models.IceStream(friction=bed_friction)
opts = {
    "dirichlet_ids": [1, 2, 3, 4],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "snes_type": "newtontr",
        "snes_max_it": 100,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}
flow_solver = icepack.solvers.FlowSolver(flow_model, **opts)
u_init = u.copy(deepcopy=True)
u = flow_solver.diagnostic_solve(
    velocity=u_init,
    thickness=h,
    surface=s,
    fluidity=A,
    log_friction=q,
)

print("done solving for initial velocity")

def simulation(q):
    fields = {"velocity": u_init, "thickness": h, "surface": s, "fluidity": A}
    return flow_solver.diagnostic_solve(log_friction=q, **fields)

def regularization(q):
    Ω = Constant(area)
    α = Constant(args.regularization)
    return 0.5 * α**2 / Ω * inner(grad(q), grad(q)) * dx

# Solve a statistical estimation problem for the log-friction
problem = icepack.statistics.StatisticsProblem(
    simulation=simulation,
    loss_functional=loss_functional,
    regularization=regularization,
    controls=q,
)

estimator = icepack.statistics.MaximumProbabilityEstimator(
    problem,
    gradient_tolerance=5e-5,
    step_tolerance=1e-8,
    max_iterations=50,
)

q = estimator.solve()

# Save the result to disk
with firedrake.CheckpointFile(args.output, "w") as chk:
    chk.save_function(q, name="log_friction")
    chk.save_function(u, name="velocity")
    chk.h5pyfile.attrs["mean_stress"] = τ_avg
    chk.h5pyfile.attrs["mean_speed"] = u_avg
