
using Flux
using CSV, DataFrames
using JLD2
using Sprout
using CairoMakie

n_layers = [3, 6, 9];
n_neurons = [32, 64, 128];
fraction_backbone_layers = 2//3;
batch_size = 100000;

masking_f = (clas_out, reg_out) -> (mask_𝑣(clas_out, reg_out[1]), mask_𝐗(clas_out, reg_out[2]));
# Load and freeze CLASSIFIER
m_classifier = create_classifier_model(3, 250, 8, 20);
model_state = JLD2.load("examples/data/saved_classifier.jld2", "model_state");
Flux.loadmodel!(m_classifier, model_state);
m_tree_classifier = Flux.setup(Flux.Adam(), m_classifier);
Flux.freeze!(m_tree_classifier);

# LOAD DATA
#-----------------------------------------------------------------------
x_train = CSV.read("data/sb21_02Oct25_train_x.csv", DataFrame);
y_train = CSV.read("data/sb21_02Oct25_train_y.csv", DataFrame);
x_val = CSV.read("data/sb21_02Oct25_val_x.csv", DataFrame);
y_val = CSV.read("data/sb21_02Oct25_val_y.csv", DataFrame);

x_train, 𝑣_train, 𝐗_ss_train, ρ_train, Κ_train, μ_train = preprocess_data(x_train, y_train);
x_val, 𝑣_val, 𝐗_ss_val, ρ_val, Κ_val, μ_val = preprocess_data(x_val, y_val);

# Normalise inputs
xNorm = Norm(x_train);
x_train = xNorm(x_train);
x_val = xNorm(x_val);

# Scale outputs
𝐗Scale = MinMaxScaler(𝐗_ss_train);
𝐗_ss_train = 𝐗Scale(𝐗_ss_train);
𝐗_ss_val = 𝐗Scale(𝐗_ss_val);

𝑣Scale = MinMaxScaler(𝑣_train);
𝑣_train = 𝑣Scale(𝑣_train);
𝑣_val = 𝑣Scale(𝑣_val);

# SETUP LOSS & METRICS
#----------------------------------------------------------------------
# Normalisation/scaling structures must live on the same device as the model is trained on
# for training on GPU move normalisers/scalers/pure_phase_comp to GPU; e.g. xNorm_gpu = xNorm |> gpu
xNorm_gpu = xNorm |> gpu;
𝑣Scale_gpu = 𝑣Scale |> gpu;
𝐗Scale_gpu = 𝐗Scale |> gpu;
pp_mat_gpu = reshape(PP_COMP_adj, 6, :) |> gpu;

function loss((𝑣_ŷ, 𝐗_ŷ), (𝑣, 𝐗), x)
    return sum(abs2, 𝑣_ŷ .- 𝑣) + sum(abs2, 𝐗_ŷ .- 𝐗) + misfit.mass_balance_abs_misfit((descale(𝑣Scale_gpu, 𝑣_ŷ), descale(𝐗Scale_gpu, 𝐗_ŷ)), denorm(xNorm_gpu, x)[3:end,:,:], agg=sum, pure_phase_comp=pp_mat_gpu)
end
# Metrics (for validation only, must follow signature (ŷ, y) -> Real)
function mass_balance_metric((𝑣_ŷ, 𝐗_ŷ), (_, _))
    return misfit.mass_balance_abs_misfit((descale(𝑣Scale, 𝑣_ŷ), descale(𝐗Scale, 𝐗_ŷ)), denorm(xNorm, x_val)[3:end,:,:], agg=mean)
end
function mae_𝑣(ŷ, y)
    return misfit.mae_no_zeros(descale(𝑣Scale, ŷ[1]), descale(𝑣Scale, y[1]))
end
function mae_𝐗(ŷ, y)
    return misfit.mae_no_zeros(descale(𝐗Scale, ŷ[2]), descale(𝐗Scale, y[2]))
end

metrics = [mass_balance_metric, mae_𝑣, mae_𝐗];

# TUNE IT
#-----------------------------------------------------------------------
hpt_regressor_pretrained_classifier(n_layers, n_neurons, fraction_backbone_layers, batch_size, loss,
               (x_train, (𝑣_train, 𝐗_ss_train)), (x_val, (𝑣_val, 𝐗_ss_val)),
               m_classifier, masking_f,
               1000, metrics,
               lr_schedule=false)

# Alternative: shared backbone model
# hpt_regressor_common_backbone(n_layers, n_neurons, fraction_backbone_layers, batch_size, loss,
#                               (x_train, (𝑣_train, 𝐗_ss_train)), (x_val, (𝑣_val, 𝐗_ss_val)),
#                               masking_f,
#                               10, metrics,
#                               lr_schedule=false)

# VISUALISE RESULTS
#-----------------------------------------------------------------------
log_matrix = load_hyperparam_tuning_results("hyperparam_tuning2025Dec19_1611", n_layers, n_neurons);

min_val_loss = minimum.(getfield.(log_matrix, :mean_loss));
min_mae_𝑣 = minimum.(getfield.(log_matrix, :mae_𝑣));
min_mae_𝐗 = minimum.(getfield.(log_matrix, :mae_𝐗));

fig = Figure(size = (1200, 400));
ax = Axis(fig[1, 1], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons in hidden layers");
ax.xticks = (n_layers, string.(n_layers));
ax.yticks = (n_neurons, string.(n_neurons));

hm = heatmap!(n_layers, n_neurons, min_val_loss);
Colorbar(fig[1, 2], hm; label = "min. validation loss");

ax = Axis(fig[1, 3], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons in hidden layers");
ax.xticks = (n_layers, string.(n_layers));
ax.yticks = (n_neurons, string.(n_neurons));

hm = heatmap!(n_layers, n_neurons, min_mae_𝑣);
Colorbar(fig[1, 4], hm; label = "min. mae 𝑣");

ax = Axis(fig[1, 5], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons in hidden layers");
ax.xticks = (n_layers, string.(n_layers));
ax.yticks = (n_neurons, string.(n_neurons));

hm = heatmap!(n_layers, n_neurons, min_mae_𝐗);
Colorbar(fig[1, 6], hm; label = "min. mae 𝐗");

fig