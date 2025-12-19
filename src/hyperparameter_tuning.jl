
"""
Load the logs(::NamedTuples) of a hyperparameter tuning run into a matrix.

The logged values can then be accessed from that `log_matrix` using getfield.(log_matrix, :KEY)
"""
function load_hyperparam_tuning_results(dir::String, n_layers::AbstractVector, n_neurons::AbstractVector)
    all_models_dir = [d for d in readdir(dir, join=true) if isdir(d)]

    log_matrix = Matrix{NamedTuple}(undef, length(n_layers), length(n_neurons))

    logs = [load(all_models_dir[i] * "/log.jld2", "logs_t") for i in eachindex(all_models_dir)]
    log_matrix = reshape(logs, (length(n_layers), length(n_neurons)))

    return log_matrix
end


"""
Perform hyperparameter tuning for the CLASSIFIER model.
"""
function hpt_classifier(n_layers::Vector{<:Integer}, n_neurons::Vector{<:Integer}, batch_size::Integer, loss::Function,
                            train_data::Tuple{AbstractArray{Float32, 3}, BitArray{3}}, val_data::Tuple{AbstractArray{Float32, 3}, BitArray{3}},
                            max_epochs::Integer, metrics::Vector{<:Function};
                            lr_schedule::Bool = false)
    subdir = "hyperparam_tuning" * Dates.format(now(),"yyyyudd_HHMM")

    loader = Flux.DataLoader((train_data[1], train_data[2]), batchsize=batch_size, shuffle=true)
    INPUT_DIM = size(train_data[1])[1]
    OUTPUT_DIM = size(train_data[2])[1]

    for (n_l, n_n) in ProgressBar(Iterators.product(n_layers, n_neurons))
        model = create_classifier_model(n_l, n_n, INPUT_DIM, OUTPUT_DIM) |> gpu_device()
        opt_state = Flux.setup(Flux.Adam(0.001), model)

        # setup early_stopping_condition
        early_stopping = Flux.early_stopping((val_loss) -> val_loss, 10, init_score=Inf32)

        # set-up optional learning rate schedule (Cosine Annealing)
        if lr_schedule
            lrs = CosAnneal(λ0 = 1e-5, λ1 = 1e-3, period = 50, restart=true)
        else
            lrs = nothing
        end

        model, opt_state, logs, log_dir_path = train_loop(model, loader, opt_state, (val_data[1], val_data[2]), loss, max_epochs; metrics=metrics, early_stopping_condition=early_stopping, lr_schedule=lrs, gpu_device=gpu_device(), save_to_subdir=subdir, show_progressbar=false)
        sb21_surrogate.post_training_plots_asm(logs, log_dir_path)

        # write hyperparam configuration into saved model dir
        open(log_dir_path*"/hp_config.txt", "w") do file
            println(file, "Hyperparameter configuration:")
            println(file, "-----------------------------")
            println(file, "Number of layers:\t$n_l")
            println(file, "Number of neurons:\t$n_n")
        end
    end
    println("---TUNING COMPLETE---")

    return nothing
end


function hpt_regressor_pretrained_classifier(n_layers::Vector{<:Integer}, n_neurons::Vector{<:Integer}, fraction_backbone_layers::Rational{Int}, batch_size::Integer, loss::Function,
                                             train_data::Tuple{AbstractArray{Float32, 3}, AbstractArray{Float32, 3}}, val_data::Tuple{AbstractArray{Float32, 3}, Tuple{AbstractArray{Float32, 3}, AbstractArray{Float32, 3}}},
                                             classifier::Chain, masking_f::Function,
                                             max_epochs::Integer, metrics::Vector{<:Function};
                                             lr_schedule::Bool = false)
    subdir = "hyperparam_tuning" * Dates.format(now(),"yyyyudd_HHMM")

    loader = Flux.DataLoader((train_data[1], train_data[2]), batchsize=batch_size, shuffle=true)
    x_val, (𝑣_val, 𝐗_ss_val) = val_data
    for (n_l, n_n) in Iterators.product(n_layers, n_neurons)
        model = create_model_pretrained_classifier(fraction_backbone_layers, n_l, n_n,
                                                   masking_f, classifier;
                                                   out_dim_𝑣 = 20, out_dim_𝐗 = (6, 14)) |> gpu_device()
        opt_state = Flux.setup(Flux.Adam(0.001), model)

        # setup early_stopping_condition
        early_stopping = Flux.early_stopping((val_loss) -> val_loss, 10, init_score=Inf32)

        # set-up optional learning rate schedule (Cosine Annealing)
        if lr_schedule
            lrs = CosAnneal(λ0 = 1e-5, λ1 = 1e-3, period = 50, restart=true)
        else
            lrs = nothing
        end

        model, opt_state, logs, log_dir_path = train_loop(model, loader, opt_state, (x_val, (𝑣_val, 𝐗_ss_val)), loss, max_epochs; metrics = metrics, early_stopping_condition=early_stopping, lr_schedule=lrs, gpu_device=gpu_device(), save_to_subdir=subdir, show_progressbar=false)
        sb21_surrogate.post_training_plots_asm(logs, log_dir_path)

        # write hyperparam configuration into saved model dir
        open(log_dir_path*"/hp_config.txt", "w") do file
            println(file, "Hyperparameter configuration:")
            println(file, "-----------------------------")
            println(file, "Number of layers:\t$n_l")
            println(file, "Number of neurons:\t$n_n")
            println(file, "Fraction of backbone layers:\t$fb")
        end
    end
    println("---TUNING COMPLETE---")

    return nothing
end


function hpt_regressor_common_backbone(n_layers::Vector{<:Integer}, n_neurons::Vector{<:Integer}, fraction_backbone_layers::Rational{Int}, batch_size::Integer, loss::Function,
                                       train_data::Tuple{AbstractArray{Float32, 3}, AbstractArray{Float32, 3}}, val_data::Tuple{AbstractArray{Float32, 3}, AbstractArray{Float32, 3}},
                                       masking_f::Function,
                                       max_epochs::Integer, metrics::Vector{<:Function};
                                       lr_schedule::Bool = false)
    return nothing
end
