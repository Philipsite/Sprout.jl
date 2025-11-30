using sb21_surrogate
using JLD2
using Flux

# Load model
m_classifier = Chain(
    Dense(8 => 100, relu),
    Dense(100 => 100, relu),
    Dense(100 => 100, relu),
    Dense(100 => 100, relu),
    Dense(100 => 100, relu),
    Dense(100 => 19, sigmoid)
)

model_state = JLD2.load("../model_runs/sb21_sm_2025Oct02_1644/saved_model.jld2", "model_state");
Flux.loadmodel!(m_classifier, model_state)

x_norm = Norm([204.95631, 1499.9882, 0.43890578, 0.073391415, 0.054225676, 0.06534755, 0.3575318, 0.010903176], [112.81744, 577.22375, 0.04530368, 0.03775132, 0.027514301, 0.0031478975, 0.120003514, 0.006286985])

P_bounds = Tuple{Float32, Float32}((10., 400.))
T_bounds = Tuple{Float32, Float32}((500, 2500))
n = 1000                    # resolution

PYR_Xu_mol = Vector{Float32}([38.71, 2.94, 2.22, 6.17, 0, 0.11])
PYR_Xu_mol ./= sum(PYR_Xu_mol)
BULK = PYR_Xu_mol

# AOC_mol = Vector{Float32}([0.53443, 0.15154, 0.077081, 0.11207, 0.10066, 0.024203])
# BULK = AOC_mol

# PSUM_mol = Vector{Float32}([0.4627867, 0.04347390, 0.04888310, 0.08895121, 0.35259941, 0.0033056195])
# BULK = PSUM_mol

# Modified Harzburgite
# HAR_Xu_mol = Vector{Float32}([36.04, 0.79, 0.65, 5.97, 56.54, 0.00])
# HAR_Xu_mol ./= sum(HAR_Xu_mol)
# BULK = HAR_Xu_mol

asm_grid, var_vec_grid = generate_mineral_assemblage_diagram(P_bounds, T_bounds, BULK, 1000, m_classifier, x_norm)
fig = plot_mineral_assemblage_diagram(asm_grid, var_vec_grid, P_bounds, T_bounds, :acton)
unique(var_vec_grid)