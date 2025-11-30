using CSV, DataFrames, JLD2
using sb21_surrogate
using CairoMakie

n_layers  = [3, 4, 5, 6, 7, 8, 9]
n_neurons = [100, 150, 200, 250]


# LOAD DATA
#-----------------------------------------------------------------------
# Reduce the y_data to only include the phase fractions of the 22 phases
# Convert phase fractions into one-hot vec for phase stability
x_train = CSV.read("data/sb21_02Oct25_train_x.csv", DataFrame);
y_train = CSV.read("data/sb21_02Oct25_train_y.csv", DataFrame);
x_train, y_train = preprocess_for_classifier(x_train, y_train);

# filter out data points that are corrupted by a NaN (failed minimisations?)
cols_no_nan = [!any(isnan, y_train[:, j]) for j in 1:size(y_train, 2)]
x_train = x_train[:, cols_no_nan]
y_train = y_train[:, cols_no_nan]

x_val = CSV.read("data/sb21_02Oct25_val_x.csv", DataFrame);
y_val = CSV.read("data/sb21_02Oct25_val_y.csv", DataFrame);
x_val, y_val = preprocess_for_classifier(x_val, y_val);

# Normalise data
x_norm = Norm(x_train);

x_train = x_norm(x_train);
x_val = x_norm(x_val);


# READ BACK TUNING RESULTS
#-----------------------------------------------------------------------


log_matrix = load_hyperparam_tuning_results("/Users/philip/Research/PhDProjectUNIL/0X1_SurrogateGEM/01_SB21/model_runs/hyperparam_tuning2025Oct16_2253", n_layers, n_neurons)

min_val_loss = minimum.(getfield.(log_matrix, :mean_loss))
min_qasm_loss = minimum.(getfield.(log_matrix, :loss_asm))

inference_time_ms = estimate_inference_time("/Users/philip/Research/PhDProjectUNIL/0X1_SurrogateGEM/01_SB21/model_runs/hyperparam_tuning2025Oct16_2253", n_layers, n_neurons, x_val, y_val)


fig = Figure(size = (1200, 400))
ax = Axis(fig[1, 1], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons in hidden layers")
ax.xticks = (n_layers, string.(n_layers))
ax.yticks = (n_neurons, string.(n_neurons))

hm = heatmap!(n_layers, n_neurons, min_val_loss)
Colorbar(fig[1, 2], hm; label = "min. validation loss")

ax = Axis(fig[1, 3], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons in hidden layers")
ax.xticks = (n_layers, string.(n_layers))
ax.yticks = (n_neurons, string.(n_neurons))

hm = heatmap!(n_layers, n_neurons, min_qasm_loss)
Colorbar(fig[1, 4], hm; label = "min. Qasm loss")

ax = Axis(fig[1, 5], aspect=1.0, xlabel="n.o. hidden layers", ylabel="n.o. neurons in hidden layers")
ax.xticks = (n_layers, string.(n_layers))
ax.yticks = (n_neurons, string.(n_neurons))

hm = heatmap!(n_layers, n_neurons, inference_time_ms)
Colorbar(fig[1, 6], hm; label = "Inference time [ms]")


fig
