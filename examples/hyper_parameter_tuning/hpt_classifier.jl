
using Flux
using CSV, DataFrames
using sb21_surrogate
using CairoMakie

n_layers = [2, 3, 4];
n_neurons = [32, 64, 128];
batch_size = 100000;

# LOAD DATA
#-----------------------------------------------------------------------
x_train = CSV.read("data/sb21_02Oct25_train_x.csv", DataFrame);
y_train = CSV.read("data/sb21_02Oct25_train_y.csv", DataFrame);
x_val = CSV.read("data/sb21_02Oct25_val_x.csv", DataFrame);
y_val = CSV.read("data/sb21_02Oct25_val_y.csv", DataFrame);

x_train, 𝑣_train, _, _, _, _ = preprocess_data(x_train, y_train);
y_train = one_hot_phase_stability(𝑣_train);
x_val, 𝑣_val, _, _, _, _ = preprocess_data(x_val, y_val);
y_val = one_hot_phase_stability(𝑣_val);

# Normalise inputs
xNorm = Norm(x_train);
x_train = xNorm(x_train);
x_val = xNorm(x_val);


# TUNE IT
#-----------------------------------------------------------------------
hpt_classifier(n_layers, n_neurons, batch_size, Flux.Losses.binarycrossentropy,
               (x_train, y_train), (x_val, y_val),
               1000, [misfit.loss_asm, misfit.fraction_mismatched_asm, misfit.fraction_mismatched_phases],
               lr_schedule=true)


# VISUALISE RESULTS
#-----------------------------------------------------------------------
log_matrix = load_hyperparam_tuning_results("hyperparam_tuning2025Dec17_1521", n_layers, n_neurons);

min_val_loss = minimum.(getfield.(log_matrix, :mean_loss));
min_qasm_loss = minimum.(getfield.(log_matrix, :loss_asm));

fig = Figure(size = (800, 400));
ax = Axis(fig[1, 1], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons in hidden layers");
ax.xticks = (n_layers, string.(n_layers));
ax.yticks = (n_neurons, string.(n_neurons));

hm = heatmap!(n_layers, n_neurons, min_val_loss);
Colorbar(fig[1, 2], hm; label = "min. validation loss");

ax = Axis(fig[1, 3], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons in hidden layers");
ax.xticks = (n_layers, string.(n_layers));
ax.yticks = (n_neurons, string.(n_neurons));

hm = heatmap!(n_layers, n_neurons, min_qasm_loss);
Colorbar(fig[1, 4], hm; label = "min. Qasm loss");

fig