"""
Create a flux model with a given number of (hidden) layers, and number of neurons in these hidden layers.
"""
function create_model(n_layers::Integer, n_neurons::Integer, input_dim::Integer, output_dim::Integer)
    layers = []

    # First layer (input to first hidden)
    push!(layers, Dense(input_dim => n_neurons, relu))

    # Hidden layers
    for i in 2:n_layers
        push!(layers, Dense(n_neurons => n_neurons, relu))
    end

    # Output layer
    push!(layers, Dense(n_neurons => output_dim, sigmoid))

    return Chain(layers...)
end


"""
Test all possible combination of n_layers * n_neurons.
"""
function run_hyperparam_tuning(n_layers::Vector{<:Integer}, n_neurons::Vector{<:Integer}, train_data::Tuple{<:AbstractArray, <:AbstractArray}, val_data::Tuple{<:AbstractArray, <:AbstractArray}, loss::Function, max_epochs::Integer, metrics::Vector{<:Function})
    subdir = "hyperparam_tuning" * Dates.format(now(),"yyyyudd_HHMM")

    loader = Flux.DataLoader((train_data[1], train_data[2]), batchsize=100000, shuffle=true);
    INPUT_DIM = size(train_data[1])[1];
    OUTPUT_DIM = size(train_data[2])[1];

    for (n_l, n_n) in ProgressBar(Iterators.product(n_layers, n_neurons))
        model = create_model(n_l, n_n, INPUT_DIM, OUTPUT_DIM) |> gpu_device()
        opt_state = Flux.setup(Flux.Adam(0.001), model)

        # setup early_stopping_condition
        early_stopping = Flux.early_stopping((val_loss) -> val_loss, 10, init_score=Inf32)

        model, opt_state, logs, log_dir_path = train_loop(model, loader, opt_state, (val_data[1], val_data[2]), loss, max_epochs; metrics=metrics, early_stopping_condition=early_stopping, gpu_device=gpu_device(), save_to_subdir=subdir, show_progressbar=false)
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
end

"""
Load the logs(::NamedTuples) of a hyperparameter tuning run into a matrix.

The logged values can then be accessed from that `log_matrix` using getfield.(log_matrix, :KEY)
"""
function load_hyperparam_tuning_results(dir::String, n_layers::AbstractVector, n_neurons::AbstractVector)
    all_models_dir = readdir(dir, join=true)

    log_matrix = Matrix{NamedTuple}(undef, length(n_layers), length(n_neurons))

    logs = [load(all_models_dir[i] * "/log.jld2", "logs_t") for i in eachindex(all_models_dir)]
    log_matrix = reshape(logs, (length(n_layers), length(n_neurons)))

    return log_matrix
end


"""
Benchmark the inference time of the models generated in a hyperparameter tuning run.
"""
function estimate_inference_time(dir, n_layers, n_neurons, x_val)
    INPUT_DIM = size(x_val)[1]
    OUTPUT_DIM = size(y_val)[1]

    all_models_dir = readdir(dir, join=true)

    inference_time_ms = []
    hyperparams_setup = collect(Iterators.product(n_layers, n_neurons))

    for i in eachindex(all_models_dir)
        m = create_model(hyperparams_setup[i][1], hyperparams_setup[i][2], INPUT_DIM, OUTPUT_DIM)

        model_state = JLD2.load(all_models_dir[i] * "/saved_model.jld2", "model_state")
        Flux.loadmodel!(m, model_state)
        # warm-up execution (trigger JIT)
        _ = m(x_val)

        res = @benchmark $m($x_val)
        # convert to milliseconds (BenchmarkTools output in ns per default)
        res_ms = median(res.times) / 1_000_000

        push!(inference_time_ms, res_ms)
    end

    inference_time_ms = reshape(inference_time_ms, (length(n_layers), length(n_neurons)))
    return inference_time_ms
end
