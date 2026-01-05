
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
                            lr_schedule::Bool = false,
                            subdir_appendix::String = "")
    subdir = "hyperparam_tuning" * subdir_appendix * "_"* Dates.format(now(),"yyyyudd_HHMM")

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
        Sprout.post_training_plots_asm(logs, log_dir_path)

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


"""
Perform hyperparameter tuning for the REGRESSOR model with a pretrained CLASSIFIER as backbone.
"""
function hpt_regressor_pretrained_classifier(n_layers::Vector{<:Integer}, n_neurons::Vector{<:Integer}, fraction_backbone_layers::Rational{Int}, batch_size::Integer, loss::Function,
                                             train_data::Tuple{AbstractArray{Float32, 3}, Tuple{AbstractArray{Float32, 3}, AbstractArray{Float32, 3}}}, val_data::Tuple{AbstractArray{Float32, 3}, Tuple{AbstractArray{Float32, 3}, AbstractArray{Float32, 3}}},
                                             classifier::Chain, masking_f::Function,
                                             max_epochs::Integer, metrics::Vector{<:Function};
                                             lr_schedule::Bool = false,
                                             freeze_classifier::Bool = false,
                                             subdir_appendix::String = "")
    subdir = "hyperparam_tuning" * subdir_appendix * "_"* Dates.format(now(),"yyyyudd_HHMM")

    loader = Flux.DataLoader((train_data[1], train_data[2]), batchsize=batch_size, shuffle=true)
    x_val, (𝑣_val, 𝐗_ss_val) = val_data
    for (n_l, n_n) in ProgressBar(Iterators.product(n_layers, n_neurons))
        model = create_model_pretrained_classifier(fraction_backbone_layers, n_l, n_n,
                                                   masking_f, classifier;
                                                   out_dim_𝑣 = 20, out_dim_𝐗 = (6, 14)) |> gpu_device()
        opt_state = Flux.setup(Flux.Adam(0.001), model)

        if freeze_classifier
            Flux.freeze!(opt_state.layers[1])  # freeze the classifier part
        end

        # setup early_stopping_condition
        early_stopping = Flux.early_stopping((val_loss) -> val_loss, 10, init_score=Inf32)

        # set-up optional learning rate schedule (Cosine Annealing)
        if lr_schedule
            lrs = CosAnneal(λ0 = 1e-5, λ1 = 1e-3, period = 50, restart=true)
        else
            lrs = nothing
        end

        model, opt_state, logs, log_dir_path = train_loop(model, loader, opt_state, (x_val, (𝑣_val, 𝐗_ss_val)), loss, max_epochs; metrics = metrics, early_stopping_condition=early_stopping, lr_schedule=lrs, gpu_device=gpu_device(), save_to_subdir=subdir, show_progressbar=false)
        post_training_plots(logs, log_dir_path)

        # write hyperparam configuration into saved model dir
        open(log_dir_path*"/hp_config.txt", "w") do file
            println(file, "Hyperparameter configuration:")
            println(file, "-----------------------------")
            println(file, "Number of layers:\t$n_l")
            println(file, "Number of neurons:\t$n_n")
            println(file, "Fraction of backbone layers:\t$fraction_backbone_layers")
        end
    end
    println("---TUNING COMPLETE---")

    return nothing
end


"""
Perform hyperparameter tuning for the REGRESSOR model with a shared backbone.
"""
function hpt_regressor_common_backbone(n_layers::Vector{<:Integer}, n_neurons::Vector{<:Integer}, fraction_backbone_layers::Rational{Int}, batch_size::Integer, loss::Function,
                                       train_data::Tuple{AbstractArray{Float32, 3}, Tuple{AbstractArray{Float32, 3}, AbstractArray{Float32, 3}}}, val_data::Tuple{AbstractArray{Float32, 3}, Tuple{AbstractArray{Float32, 3}, AbstractArray{Float32, 3}}},
                                       masking_f::Function,
                                       max_epochs::Integer, metrics::Vector{<:Function};
                                       lr_schedule::Bool = false,
                                       subdir_appendix::String = "")
    subdir = "hyperparam_tuning" * subdir_appendix * "_"* Dates.format(now(),"yyyyudd_HHMM")

    loader = Flux.DataLoader((train_data[1], train_data[2]), batchsize=batch_size, shuffle=true)
    x_val, (𝑣_val, 𝐗_ss_val) = val_data
    for (n_l, n_n) in ProgressBar(Iterators.product(n_layers, n_neurons))
        model = create_model_shared_backbone(fraction_backbone_layers, n_l, n_n,
                                             masking_f;
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
        post_training_plots(logs, log_dir_path)

        # write hyperparam configuration into saved model dir
        open(log_dir_path*"/hp_config.txt", "w") do file
            println(file, "Hyperparameter configuration:")
            println(file, "-----------------------------")
            println(file, "Number of layers:\t$n_l")
            println(file, "Number of neurons:\t$n_n")
            println(file, "Fraction of backbone layers:\t$fraction_backbone_layers")
        end
    end
    println("---TUNING COMPLETE---")

    return nothing
end


"""
Benchmark the inference time of the models generated in a hyperparameter tuning run.
Returns a matrix of inference times in milliseconds with dimensions (length(n_layers), length(n_neurons)).
"""
function estimate_inference_time(dir::String, n_layers::AbstractVector, n_neurons::AbstractVector, val_data::T) where T <: Tuple{AbstractArray{Float32, 3}, BitArray{3}}
    x_val, y_val = val_data
    INPUT_DIM = size(x_val)[1]
    OUTPUT_DIM = size(y_val)[1]

    all_models_dir = readdir(dir, join=true)

    inference_time_ms = []
    hyperparams_setup = collect(Iterators.product(n_layers, n_neurons))

    for i in eachindex(all_models_dir)
        m = create_classifier_model(hyperparams_setup[i][1], hyperparams_setup[i][2], INPUT_DIM, OUTPUT_DIM)

        model_state = JLD2.load(all_models_dir[i] * "/saved_model.jld2", "model_state")
        Flux.loadmodel!(m, model_state)

        # closure to measure inference time
        infer_time = () -> begin
            _ = m(x_val)
        end
        # warm-up execution (trigger JIT)
        infer_time()

        res = @benchmark $infer_time()
        # convert to milliseconds (BenchmarkTools output in ns per default)
        res_ms = median(res.times) / 1_000_000

        push!(inference_time_ms, res_ms)
    end

    inference_time_ms = reshape(inference_time_ms, (length(n_layers), length(n_neurons)))
    return inference_time_ms
end

function estimate_inference_time(dir::String, n_layers::AbstractVector, n_neurons::AbstractVector, fraction_backbone_layers::Rational{Int}, val_data::T) where T <: Tuple{AbstractArray{Float32, 3}, Tuple{AbstractArray{Float32, 3}, AbstractArray{Float32, 3}}}
    println("INFERENCE TIME ESTIMATION FOR REGRESSOR WITH SHARED BACKBONE - NOT YET IMPLEMENTED")
    return nothing
end

function estimate_inference_time(dir::String, n_layers::AbstractVector, n_neurons::AbstractVector, fraction_backbone_layers::Rational{Int}, masking_f::Function, classifier::Chain, val_data::T) where T <: Tuple{AbstractArray{Float32, 3}, Tuple{AbstractArray{Float32, 3}, AbstractArray{Float32, 3}}}
    x_val, (𝑣_val, 𝐗_ss_val) = val_data
    𝑣_DIM = size(𝑣_val)[1]
    𝐗_DIM = size(𝐗_ss_val)[1:2]

    all_models_dir = readdir(dir, join=true)

    inference_time_ms = []
    hyperparams_setup = collect(Iterators.product(n_layers, n_neurons))

    for i in eachindex(all_models_dir)
        m = create_model_pretrained_classifier(fraction_backbone_layers, hyperparams_setup[i][1], hyperparams_setup[i][2],
                                              masking_f, classifier;
                                              out_dim_𝑣=𝑣_DIM, out_dim_𝐗=𝐗_DIM)

        model_state = JLD2.load(all_models_dir[i] * "/saved_model.jld2", "model_state")
        Flux.loadmodel!(m, model_state)

        # closure to measure inference time
        infer_time = () -> begin
            _, _ = m(x_val)
        end
        # warm-up execution (trigger JIT)
        infer_time()

        res = @benchmark $infer_time()
        # convert to milliseconds (BenchmarkTools output in ns per default)
        res_ms = median(res.times) / 1_000_000

        push!(inference_time_ms, res_ms)
    end

    inference_time_ms = reshape(inference_time_ms, (length(fraction_backbone_layers), length(n_layers), length(n_neurons)))
    return inference_time_ms
end
