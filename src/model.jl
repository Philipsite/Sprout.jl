

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
