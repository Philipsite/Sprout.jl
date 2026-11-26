using CSV, DataFrames
using JLD2
using Flux
using Sprout
using CairoMakie

# LOAD DATA
#-----------------------------------------------------------------------
x_train = CSV.read("examples/data/sb21_02Oct25_train_x.csv", DataFrame);
y_train = CSV.read("examples/data/sb21_02Oct25_train_y.csv", DataFrame);
x_val = CSV.read("examples/data/sb21_02Oct25_val_x.csv", DataFrame);
y_val = CSV.read("examples/data/sb21_02Oct25_val_y.csv", DataFrame);

x_train, 𝑣_train, 𝐗_ss_train, _ = preprocess_data(x_train, y_train, bulk_props=String[]);
x_val, 𝑣_val, 𝐗_ss_val, _       = preprocess_data(x_val, y_val, bulk_props=String[]);

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

# Setup DataLoader
loader = Flux.DataLoader((x_train, (𝑣_train, 𝐗_ss_train)), batchsize=100000, shuffle=true);

# SETUP MODEL
#----------------------------------------------------------------------
fraction_backbone_layers = 2//3;
n_layers = 3;
n_neurons = 256;

masking_f = (clas_out, reg_out) -> (mask_𝑣(clas_out, reg_out[1]), mask_𝐗(clas_out, reg_out[2]));

m = create_model_shared_backbone(fraction_backbone_layers, n_layers, n_neurons, masking_f) |> gpu;

opt_state = Flux.setup(Flux.Adam(0.001), m);

# SETUP LOSS & METRICS
#----------------------------------------------------------------------
# Normalisation/scaling structures must live on the same device as the model is trained on
# for training on GPU move normalisers/scalers/pure_phase_comp to GPU; e.g. xNorm_gpu = xNorm |> gpu
xNorm_gpu = xNorm
𝑣Scale_gpu = 𝑣Scale
𝐗Scale_gpu = 𝐗Scale
pp_mat_gpu = reshape(PP_COMP_adj, 6, :)

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

# TRAIN MODEL
#-----------------------------------------------------------------------
model, opt_state, logs_t, dir = train_loop(
    m,
    loader,
    opt_state,
    (x_val, (𝑣_val, 𝐗_ss_val)),
    loss,
    500,
    metrics = [mae_𝑣, mae_𝐗, mass_balance_metric],
    gpu_device = nothing,
    save_to_subdir = joinpath("examples", "reg_model_simoultaneous")
);

# POST-TRAINING PLOTS
fig = post_training_plots(logs_t, dir)
