using Flux, Statistics, ProgressMeter
using CSV, DataFrames
using CUDA
using Plots
using sb21_surrogate
using JLD2

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
) |> gpu_device();

model_state = JLD2.load("../model_runs/sb21_sm_2025Nov14_1439_prelimCLAS/saved_model.jld2", "model_state");
Flux.loadmodel!(m_classfier, model_state)
# freeze CLASSIFIER
m_tree_classifier = Flux.setup(Flux.Adam(), m_classfier)
Flux.freeze!(m_tree_classifier)


# test prediction
ŷ_asm = m_classfier(x_train[:, 1:5])
println("Predicted phases:   ", ŷ_asm[:,3] .>= 1e-5)
println("Goundtrouth phases: ", y_train[1:20,3] .!= 0.0)


# REGRESSOR
model = Parallel(connection_reduced, m_classfier, Chain(
    Dense(INPUT_DIM => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => 250, relu),
    Dense(250 => OUTPUT_DIM, relu)
))

opt_state = Flux.setup(Flux.Adam(0.001), model);

# TRAIN
#-----------------------------------------------------------------------
loss(ŷ, y) = Flux.Losses.mae(ŷ, y, agg=sum)
mae_descaled(ŷ, y) = Flux.Losses.mae(inv_scaling(y_scale, ŷ), inv_scaling(y_scale, y), agg=mean)
model, opt_state, logs, log_dir_path = train_loop(model, loader, opt_state, (x_val, y_val), loss, 100; metrics=[Flux.Losses.mae, mae_descaled])

# POST TRAIN
#-----------------------------------------------------------------------
post_training_plots(logs, log_dir_path)


using CairoMakie
    metrics = [k for k in keys(logs) if !endswith(String(k),"_loss")]

    loss_color = :lightskyblue
    val_loss_color = :royalblue
    metric_color = cgrad(:sun)[1:length(metrics)]

    fig = Figure(size = (800, 600))

    ax1 = Axis(fig[1, 1], xscale = log10,
               ygridvisible=false, xgridvisible=false,
               yticklabelcolor = loss_color,
               leftspinecolor = loss_color,
               ytickcolor = loss_color,
               ylabelcolor = loss_color,
               rightspinevisible = false,
               xlabel="Epochs (iteration over training set)",
               ylabel="Loss value")

    ax2 = Axis(fig[1, 1], xscale = log10,
               ygridvisible=false, xgridvisible=false,
               yaxisposition = :right,
               yticklabelcolor = metric_color[1],
               rightspinecolor = metric_color[1],
               ytickcolor = metric_color[1],
               ylabelcolor = metric_color[1],
               ylabel="Metric score")

    loss_line = lines!(ax1, 1:length(logs.mean_loss), logs.mean_loss; color = loss_color)
    val_loss_line = lines!(ax1, 1:length(logs.mean_loss), logs.val_loss; color = val_loss_color)
    for (i, m) in enumerate(metrics)
        m_line = lines!(ax2, 1:length(logs.mean_loss), logs[m]; color = metric_color[i], label=String(m))
    end
    hidespines!(ax2, :l, :b, :t)
    hidexdecorations!(ax2)

    axislegend(ax1, [loss_line, val_loss_line], ["Loss", "Loss (val)"], position = :lb, framevisible=false)
    axislegend(ax2, position = :rt, framevisible=false)

    ax3 = Axis(fig[2, 1], xscale = log10,
               ygridvisible=false, xgridvisible=false,
               yaxisposition = :right,
               yticklabelcolor = metric_color[1],
               rightspinecolor = metric_color[1],
               ytickcolor = metric_color[1],
               ylabelcolor = metric_color[1],
               ylabel="Metric score")

    for (i, m) in enumerate(metrics)
        m_line = lines!(ax3, 1:length(logs.mean_loss), logs[m]; color = metric_color[i], label=String(m))
    end

    # Makie.xlims!(ax3, 0.9 * length(logs.mean_loss), length(logs.mean_loss)+0.2*length(logs.mean_loss))
    Makie.ylims!(ax3, 0.0, 0.015)

    axislegend(ax3, position = :rt, framevisible=false)

    fig