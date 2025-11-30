using sb21_surrogate
using CSV, DataFrames, JLD2
using Flux
using CairoMakie

# Load model
m_classifier = Chain(
    Dense(8 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 22-2, sigmoid)
)


create_composite_model(4, 250, 8, 73, connection_reduced, m_classifier)


model = Parallel(connection_reduced, m_classifier, Chain(
    Dense(8 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 73, relu)
))

model_state = JLD2.load("../model_runs/sb21_sm_2025Nov14_1644_prelimREG/saved_model.jld2", "model_state");
Flux.loadmodel!(model, model_state)

x_norm = Norm([204.95631, 1499.9882, 0.43890578, 0.073391415, 0.054225676, 0.06534755, 0.3575318, 0.010903176], [112.81744, 577.22375, 0.04530368, 0.03775132, 0.027514301, 0.0031478975, 0.120003514, 0.006286985])

x_val = CSV.read("data/sb21_02Oct25_val_x.csv", DataFrame);
y_val = CSV.read("data/sb21_02Oct25_val_y.csv", DataFrame);
x_val, y_val = preprocess_for_regressor(x_val, y_val);

cols_no_nan = [!any(isnan, y_val[:, j]) for j in 1:size(y_val, 2)]
x_val = x_val[:, cols_no_nan]
y_val = y_val[:, cols_no_nan]

# remove the physical properties
y_val = y_val[1:end-3, :]

x_val = x_norm(x_val);

y = model(x_val)

mae_trivial_zeros(y, y_val, agg=identity)
mre_trivial_zeros(y, y_val)

mre_trivial_zeros(y, y_val, ϵ=0.01)


mae_no_zeros(y, y_val)
mre_no_zeros(y, y_val)

IDX = 16

SSₜ = sum((y_val[IDX, :] .- mean(y_val[IDX, :])).^2)
SSᵣ = sum((y[IDX, :] .- (y_val[IDX, :])).^2)
R² = 1 - SSᵣ / SSₜ

fig = Figure(; size=(800, 400))

ax = Axis(fig[1:4, 1],  xlabel=L"True\ [molmol^{-1}]", ylabel=L"Predicted\ [molmol^{-1}]", aspect = 1)
ln = lines!([0., 1.], [0., 1.], color="red")
sc = scatter!(ax, y_val[IDX, :], y[IDX, :], markersize=1)
text!(ax, 0.01, 0.9, text = "R² = $(round(R², digits=4))", align = (:left, :bottom))

ax12 = Axis(fig[1, 2], xgridvisible=false, xlabel=L"mae (trivial\ zeros)")
hideydecorations!(ax12)
hidespines!(ax12)
xlims!(ax12, (0., 0.5))
density!(ax12, mae_trivial_zeros(y[IDX, :], y_val[IDX, :], agg=identity))

ax22 = Axis(fig[2, 2], xgridvisible=false, xlabel=L"mre (trivial\ zeros)")
hideydecorations!(ax22)
hidespines!(ax22)
xlims!(ax22, (0.1, 50))
density!(ax22, mre_trivial_zeros(y[IDX, :], y_val[IDX, :], ϵ=0.001, agg=identity))

ax32 = Axis(fig[3, 2], xgridvisible=false, xlabel=L"mae (no\ zeros)")
hideydecorations!(ax32)
hidespines!(ax32)
xlims!(ax32, (0., 0.5))
density!(ax32, mae_no_zeros(y[IDX, :], y_val[IDX, :], agg=identity))

ax42 = Axis(fig[4, 2], xgridvisible=false, xlabel=L"mre (no\ zeros)")
hideydecorations!(ax42)
hidespines!(ax42)
xlims!(ax42, (0.1, 50))
density!(ax42, mre_no_zeros(y[IDX, :], y_val[IDX, :], agg=identity))


fig

mae_no_zeros(y, y_val, agg=identity)