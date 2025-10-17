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
function run_hyperparam_tuning(n_layers::Integer, n_neurons::Integer, train_data::Tuple{<:AbstractArray, <:AbstractArray}, val_data::Tuple{<:AbstractArray, <:AbstractArray}, loss::Function, max_epochs::Integer, metrics::Vector{<:Function})
    subdir = "hyperparam_tuning" * Dates.format(now(),"yyyyudd_HHMM")

    min_val_loss = []
    min_qasm_loss = []

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

        # push best val_loss/q_asm into array
        push!(min_val_loss, minimum(logs.val_loss))
        push!(min_qasm_loss, minimum(logs.loss_asm))
    end

    min_val_loss = reshape(min_val_loss, (length(n_layers), length(n_neurons)))
    min_qasm_loss = reshape(min_qasm_loss, (length(n_layers), length(n_neurons)))

    return min_val_loss, min_qasm_loss
end
