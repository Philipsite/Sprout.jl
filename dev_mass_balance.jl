using sb21_surrogate
using CSV, DataFrames, JLD2
using Flux, ChainRulesCore
using CairoMakie


# LOAD DATA
#-----------------------------------------------------------------------
# Reduce the y_data to only include the phase fractions of the 22 phases
# Convert phase fractions into one-hot vec for phase stability
x_train = CSV.read("data/sb21_02Oct25_train_x.csv", DataFrame);
y_train = CSV.read("data/sb21_02Oct25_train_y.csv", DataFrame);
x_train, y_train = preprocess_for_regressor(x_train, y_train);

# filter out data points that are corrupted by a NaN (failed minimisations?)
cols_no_nan = [!any(isnan, y_train[:, j]) for j in 1:size(y_train, 2)]
x_train = x_train[:, cols_no_nan]
y_train = y_train[:, cols_no_nan]

# remove the physical properties
y_train = y_train[1:end-3, :]

x_val = CSV.read("data/sb21_02Oct25_val_x.csv", DataFrame);
y_val = CSV.read("data/sb21_02Oct25_val_y.csv", DataFrame);
x_val, y_val = preprocess_for_regressor(x_val, y_val);

cols_no_nan = [!any(isnan, y_val[:, j]) for j in 1:size(y_val, 2)]
x_val = x_val[:, cols_no_nan]
y_val = y_val[:, cols_no_nan]

# remove the physical properties
y_val = y_val[1:end-3, :]

# Normalise input data
x_norm = Norm(x_train);

x_train = x_norm(x_train);
x_val = x_norm(x_val);

# Scale output data
y_scale = MinMaxScaler(y_train);

y_train = y_scale(y_train)
y_val = y_scale(y_val)

loader = Flux.DataLoader((x_train, y_train), batchsize=100000, shuffle=true);


# MODEL SETUP
#-----------------------------------------------------------------------
INPUT_DIM = size(x_train)[1];
OUTPUT_DIM = size(y_train)[1];

# CLASSIFIER
m_classfier = Chain(
    Dense(INPUT_DIM => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 22-2, sigmoid)
);

model_state = JLD2.load("../model_runs/sb21_sm_2025Nov14_1439_prelimCLAS/saved_model.jld2", "model_state");
Flux.loadmodel!(m_classfier, model_state)
# freeze CLASSIFIER
m_tree_classifier = Flux.setup(Flux.Adam(), m_classfier)
Flux.freeze!(m_tree_classifier)

# test prediction
ŷ_asm = m_classfier(x_train[:, 1:5])
println("Predicted phases:   ", ŷ_asm[:,3] .>= 1e-5)
println("Goundtrouth phases: ", y_train[1:20,3] .!= 0.0)


REGRESSOR
model = Chain(Parallel(connection_reduced, m_classfier, Chain(
    Dense(INPUT_DIM => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => OUTPUT_DIM, relu)
)), Out())

# model_preOut = Chain(Parallel(connection_reduced, m_classfier, Chain(
#     Dense(INPUT_DIM => 250, relu),
#     Dense(250 => 250, relu),
#     Dense(250 => 250, relu),
#     Dense(250 => 250, relu),
#     Dense(250 => OUTPUT_DIM, relu)
# )))

# model = Chain(model_preOut, Out())

opt_state = Flux.setup(Flux.Adam(0.001), model);


# x_train[:, 1:1]
# χ̂, v̂ = model(x_train)

# χ̂_, v̂_ = Out()(model_preOut(x_train))

# model(x_train) == Out()(model_preOut(x_train))

# χ̂[:, :, 1]
# χ̂_[:, :, 1]

# y_train_ = y_train
# y_train = Out()(y_train)
# y_val_ = y_val
# y_val = Out()(y_val)

# TRAIN
#-----------------------------------------------------------------------
# loss(ŷ, y) = Flux.Losses.mae(ŷ, y, agg=sum)
# mae_descaled(ŷ, y) = Flux.Losses.mae(inv_scaling(y_scale, ŷ), inv_scaling(y_scale, y), agg=mean)

# define a loss to work with the output of Out
function loss_(ŷ, y)
        χ̂, v̂ = ŷ
        χ, v = y
    return Flux.Losses.mae(χ̂, χ, agg=sum) + Flux.Losses.mae(v̂, v, agg=sum)
end

loss_(model(x_val), y_val)

gradient((m, x, y) -> loss_(m(x), y), model, x_val, y_val)

model, opt_state, logs, log_dir_path = train_loop(model, loader, opt_state, (x_val, y_val), loss_, 10; metrics=[loss_])

fig = post_training_plots(logs, log_dir_path)

χ, v = model(x_val)
ȳ = model(x_val)
ȳ_ = model_preOut(x_val)



ȳχ, ȳv = ȳ

batch_size = size(x_val, 2)

gχ_unreshaped = reshape(ȳχ, size(χ,1)*size(χ,2), batch_size)
gv_unreshaped = reshape(ȳv, size(v,1), batch_size)

gx = similar(y_val_)
gx .= zero(eltype(gx))
gx[1:20, :] .= gv_unreshaped

# pick rows from gχ corresponding to variable SS components (indices are in full_comp)
gx[21:end, :] .= gχ_unreshaped[sb21_surrogate.IDX_of_variable_components_in_SS_adj .+ 36, :]

ȳ_
gx

ȳ_ == gx



x_norm = Norm(Float32[204.95631, 1499.9882, 0.43890578, 0.073391415, 0.054225676, 0.06534755, 0.3575318, 0.010903176], Float32[112.81744, 577.22375, 0.04530368, 0.03775132, 0.027514301, 0.0031478975, 0.120003514, 0.006286985])

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


y



m = Chain(Out())

o1.indices_var_components_SS_in_χ
 y1 = y[:,1]
@btime m($y1)






sb21_surrogate.IDX_of_variable_components_in_SS_adj
PP_COMP_adj
SS_COMP_adj

PP_COMP_adj_         = repeat(PP_COMP_adj, 1, size(y, 2))
SS_COMP_adj_injected = repeat(SS_COMP_adj, 1, size(y, 2))
# inject predictions into SS_COMP_adj
SS_COMP_adj_injected[sb21_surrogate.IDX_of_variable_components_in_SS_adj, :] .= y[20+1:end, :]

χ = vcat(PP_COMP_adj_, SS_COMP_adj_injected)
χ = reshape(χ, 6, Int(size(χ, 1) / 6), size(y, 2))

v = reshape(y[1:20, :], 20, 1, size(y, 2))

χ
v






x̂ = χ ⊠ v
⊠
x̂ = reshape(x̂, 6, size(y, 2))

x = denorm(x_norm, x_val)[3:end,:]

mean(abs.(x .- x̂) ./ x)