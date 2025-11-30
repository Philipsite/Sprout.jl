using sb21_surrogate
using JLD2
using Flux
using CairoMakie

# Load model
m_classfier = Chain(
    Dense(8 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 22-2, sigmoid)
)

model = Parallel(connection_reduced, m_classfier, Chain(
    Dense(8 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 73, relu)
))

model_state = JLD2.load("../model_runs/sb21_sm_2025Nov14_1644_prelimREG/saved_model.jld2", "model_state");
Flux.loadmodel!(model, model_state)

x_norm = Norm([204.95631, 1499.9882, 0.43890578, 0.073391415, 0.054225676, 0.06534755, 0.3575318, 0.010903176], [112.81744, 577.22375, 0.04530368, 0.03775132, 0.027514301, 0.0031478975, 0.120003514, 0.006286985])




# plot_mineral_assemblage_diagram(asm_grid::Matrix, var_vec_grid::Matrix, P_bounds::Tuple, T_bounds::Tuple, color::Symbol)


P_bounds = Tuple{Float32, Float32}((10., 400.))
T_bounds = Tuple{Float32, Float32}((500, 2500))
n = 1000                    # resolution

P = range(P_bounds[1], P_bounds[2], length=n)
T = range(T_bounds[1], T_bounds[2], length=n)

# Reverse P for desired orientation
P_rev = reverse(P)

# Create 2D grid
P_grid = Matrix{Float32}(repeat(P_rev, 1, n))
T_grid = Matrix{Float32}(repeat(T', n, 1))

P_flat = vec(P_grid)
T_flat = vec(T_grid)

PYR_Xu_mol = Vector{Float32}([38.71, 2.94, 2.22, 6.17, 49.85, 0.11])
PYR_Xu_mol ./= sum(PYR_Xu_mol)
BULK = PYR_Xu_mol

# Concatenate P-T with bulk rock composition vectors
input_vecs = hcat([(vcat(p, t, BULK)) for (p, t) in zip(P_flat, T_flat)]...)
input_vecs_n = x_norm(input_vecs)


y = model(input_vecs_n)


OUTPUT_IDX = 9           # Olivine molar fraction
OUTPUT_IDX = 20 + 8      # MgO molar fraction in Olivine
PHASE_IDX     = 16
COMPONENT_IDX = 1
idx = sum(sb21_surrogate.N_variable_components_in_SS[1:PHASE_IDX-6-1]) + 20 + COMPONENT_IDX

y_mat = reshape(y[idx, :], n, n)

fig = Figure(; size=(500, 500))
ax = Axis(fig[1, 1],  xlabel=L"Temperature\ [°C]", ylabel=L"Pressure\ [kbar]", aspect = 1)
hm = heatmap!(ax, T, P_rev, y_mat'; colormap=:batlow, colorrange=(0.0, 1.0), interpolate=false)

Colorbar(fig[1, 2], hm, height=Relative(0.85))

fig
