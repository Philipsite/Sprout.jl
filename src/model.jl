

const FC_SS = reshape(SS_COMP_adj, 6, Int(length(SS_COMP_adj) / 6))
const FC_SS_MASK = sb21_surrogate.SS_COMP_VARIABLE

"""
Reshape layer for regression of solid solution composition:
Reshapes vector-output of a fully-connected layer into the form Matrix(N_COMPONENTS, N_PHASES).
"""
struct ReshapeLayer
    n :: Int
    m :: Int
end
Flux.@layer ReshapeLayer
Flux.trainable(rl::ReshapeLayer) = (;)
(rl::ReshapeLayer)(x::Union{AbstractArray{Float32,3}, CuArray{Float32, 3}}) = reshape(x, rl.n, rl.m, :)


"""
Masking layer that injects fixed components into the predicted solid solution compositions.
E.g., Si in Olivine is always 1/3 molmol⁻¹.

This layer uses the global constants FC_SS_MASK (boolean mask of fixed components in solid solutions) and FC_SS (fixed components values).
"""
struct InjectLayer
    var_mask :: AbstractArray
    fc_vals  :: AbstractArray
end
function InjectLayer()
    return InjectLayer(FC_SS_MASK, FC_SS)
end
Flux.@layer InjectLayer
Flux.trainable(il::InjectLayer) = (;)
(il::InjectLayer)(x::Union{AbstractArray{Float32,3}, CuArray{Float32, 3}}) = x .* il.var_mask .+ il.fc_vals


"""
Mask solid solution predictions with classifier output (phase stability).
Use this function as connection in a Flux.Parallel layer:
```
Parallel(mask_𝐗,
         m_classfier,
         Chain(...)
        )
```
"""
function mask_𝐗(classifier_out, regressor_out)
    ss_stable_view = @view classifier_out[7:20, :, :]           #//NOTE - Hard-coded indices a bit hacky; classifier_out[N_PP+1, N_TOTAL,:,:] > would need to be global constants
    ss_stable_view = reshape(ss_stable_view, 1, 14, :)          #//NOTE - Same as line above
    return regressor_out .* ss_stable_view
end


"""
Mask phase fraction predictions with classifier output (phase stability).
Use this function as connection in a Flux.Parallel layer:
```
Parallel(mask_𝑣,
         m_classfier,
         Chain(...)
        )
```
"""
function mask_𝑣(classifier_out, regressor_out)
    return regressor_out .* classifier_out
end


"""
Create a flux model with a given number of (hidden) layers, and number of neurons in these hidden layers.

model = Chain(
    Dense(INPUT_DIM => N_NEURONS, relu),
    ...
    N_LAYERS
    ...
    Dense(N_NEURONS => OUTPUT_DIM, sigmoid)
)
"""
function create_classifier_model(n_layers::Integer, n_neurons::Integer, input_dim::Integer, output_dim::Integer)
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
Create a flux model with the general structure:

```
model = Parallel(MASKING_FUNCTION,
                 CLASSIFIER_MODEL,
                 Chain(Dense(INPUT_DIM => N_NEURONS, relu),
                       ...
                       FRACTION_COMMON * N_LAYERS
                       ...
                       Dense(N_NEURONS => N_NEURONS, relu),
                       Parallel((𝑣, 𝐗) -> (𝑣, 𝐗),
                                Chain(Dense(N_NEURONS => N_NEURONS, relu),
                                            ...
                                            (1-FRACTION_COMMON) * N_LAYERS
                                            ...
                                            Dense(N_NEURONS => OUTPUT_DIM_𝑣)),
                                Chain(Dense(N_NEURONS => N_NEURONS, relu),
                                            ...
                                            (1-FRACTION_COMMON) * N_LAYERS
                                            ...
                                            Dense(N_NEURONS => *(OUTPUT_DIM_𝐗...)),
                                            ReshapeLayer(OUTPUT_DIM_REG...),
                                            InjectLayer())
                       )
                )
```

with a given number of (hidden) layers, and number of neurons in these hidden layers.
"""
function create_model_pretrained_classifier(fraction_common_layers::Rational{Int}, n_layers::Integer, n_neurons::Integer,
                                       masking_f::Function, m_classifier::Chain;
                                       out_dim_𝑣::Integer = 20, out_dim_𝐗::Tuple = (6, 20))
    # check if fraction_common_layers is valid
    isinteger(n_layers * fraction_common_layers) || error("n_layers * fraction_common_layers must be an integer.")

    input_dim = size(m_classifier[1].weight, 2)
    output_dim_class = size(m_classifier[end].weight, 1)
    output_dim_class == out_dim_𝑣 || error("Classifier output dimension does not match out_dim_𝑣.")
    output_dim_reg = *(out_dim_𝐗...)

    # set-up regressor model
    common_layers = []
    for i in 1:(fraction_common_layers * n_layers)
        if i == 1
            push!(common_layers, Dense(input_dim => n_neurons, relu))
        else
            push!(common_layers, Dense(n_neurons => n_neurons, relu))
        end
    end
    layers_reg_𝑣 = []
    layers_reg_𝐗 = []
    for i in 1:(1 - fraction_common_layers) * n_layers
        push!(layers_reg_𝑣, Dense(n_neurons => n_neurons, relu))
        push!(layers_reg_𝐗, Dense(n_neurons => n_neurons, relu))
    end

    push!(layers_reg_𝑣, Dense(n_neurons => out_dim_𝑣))
    push!(layers_reg_𝐗, Dense(n_neurons => output_dim_reg))
    push!(layers_reg_𝐗, ReshapeLayer(out_dim_𝐗...))
    push!(layers_reg_𝐗, InjectLayer())

    m_regressor = Chain(vcat(common_layers,
                             [Parallel((𝑣, 𝐗) -> (𝑣, 𝐗),
                                       Chain(layers_reg_𝑣...),
                                       Chain(layers_reg_𝐗...)
                                      )
                             ]...
                            )...
                       )

    # create full model
    m = Parallel(masking_f,
                 m_classifier,
                 m_regressor)
    return m
end



function create_model_common_base()
    return nothing
end
