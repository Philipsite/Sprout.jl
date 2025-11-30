# THIS NOW DONE WITHIN phases_sb21
# # Remove solid-solution phases that are never predicted to be stable from N_variable_components_in_SS
# idx_ss_never_stable = IDX_OF_PHASES_NEVER_STABLE[IDX_OF_PHASES_NEVER_STABLE .> 7]        # sb21 contains 7 pure phases!
# idx_ss_never_stable .-= 7                                                                # reset idx
# N_variable_components_in_SS_adjusted = [v for (i,v) in enumerate(N_variable_components_in_SS) if i ∉ idx_ss_never_stable]

# # Set-up the indices of different outputs in the REG output vector
# IDX_phase_frac = 1:(length(PP) + length(SS) - length(IDX_OF_PHASES_NEVER_STABLE))

"""
connection function used within compound model's Parallel layer to reduce ŷ_classifier with ŷ_regressor.

This function is used for models that only predict phases that are stable in at least one assemblage of the training generate_dataset,
and only predicts the variable compositional variables for SS-phases.

Including bulk rock physical params.
"""
function connection_reduced_phys_params(y_clas::T, y_reg::T) where {T <: Union{Matrix, CuArray}}
    # using @view to get a StridedArray and avoid Scalar indexing
    y_phase_frac = @view(y_reg[IDX_phase_frac, :]) .* y_clas

    ss_comp_asm = vcat([repeat(@view(y_clas[5+i:5+i, :]), n, 1) for (n, i) in zip(N_variable_components_in_SS_adj, 1:length(y_clas[6:end, 1]))]...)
    y_ss_comp = @view(y_reg[IDX_phase_frac[end]+1:end-3, :]) .* ss_comp_asm

    y_phys_prop = @view(y_reg[end-2:end, :])

    return vcat(y_phase_frac, y_ss_comp, y_phys_prop)
end


"""
connection function used within compound model's Parallel layer to reduce ŷ_classifier with ŷ_regressor.

This function is used for models that only predict phases that are stable in at least one assemblage of the training generate_dataset,
and only predicts the variable compositional variables for SS-phases.

Excluding bulk rock physical params.
"""
function connection_reduced(y_clas::T, y_reg::T) where {T <: Union{Matrix, CuArray}}
    # using @view to get a StridedArray and avoid Scalar indexing
    y_phase_frac = @view(y_reg[IDX_phase_frac, :]) .* y_clas

    ss_comp_asm = vcat([repeat(@view(y_clas[5+i:5+i, :]), n, 1) for (n, i) in zip(N_variable_components_in_SS_adj, 1:length(y_clas[6:end, 1]))]...)
    y_ss_comp = @view(y_reg[IDX_phase_frac[end]+1:end, :]) .* ss_comp_asm
    return vcat(y_phase_frac, y_ss_comp)
end


#=
Custom Flux.jl compatible output layer: Out

This section defines a custom layers and the corresponding forward-pass + reverse-mode AD rules (rrule) for gradient-based optimisation
=#
struct Out{T}
    comp_PP ::T
    comp_SS ::T
    indices_var_components_in_SS::Vector{Int}
end
# Constructor:
function Out(; comp_PP::VecOrMat{Float32} = PP_COMP_adj, comp_SS::VecOrMat{Float32} = SS_COMP_adj)
    comp_PP = reshape(comp_PP, :, 1)
    comp_SS = reshape(comp_SS, :, 1)

    indices_var_components_in_SS = sb21_surrogate.IDX_of_variable_components_in_SS_adj

    return Out(comp_PP, comp_SS, indices_var_components_in_SS)
end
Flux.@layer Out
Flux.trainable(o::Out) = (;)


"""
Forward-call
"""
function Out_f(o::Out, x)
    x = ndims(x) == 1 ? reshape(x, :, 1) : x
    batch_size = size(x, 2)

    comp_PP_mat = repeat(o.comp_PP, 1, batch_size)
    comp_SS_mat = repeat(o.comp_SS, 1, batch_size)

    comp_SS_injected = copy(comp_SS_mat)
    comp_SS_injected[o.indices_var_components_in_SS, :] = x[21:end, :]

    full_comp = vcat(comp_PP_mat, comp_SS_injected)
    χ = reshape(full_comp, 6, Int(size(full_comp, 1) / 6), batch_size)
    v = reshape(x[1:20, :], 20, 1, batch_size)
    return χ, v
end
(o::Out)(x) = Out_f(o, x)

"""
Reverse-mode AD rule for (o::Out)(x)
"""
# //TODO - Make sure to double-check this....
function ChainRulesCore.rrule(::typeof(Out_f), o::Out, x)
    # ----- Forward pass -----
    χ, v = Out_f(o, x)

    function pullback(ȳ)
        ȳχ, ȳv = ȳ

        batch_size = size(x, 2)

        gχ_unreshaped = reshape(ȳχ, size(χ,1)*size(χ,2), batch_size)
        gv_unreshaped = reshape(ȳv, size(v,1), batch_size)

        gx = similar(x)
        gx .= zero(eltype(gx))
        gx[1:20, :] .= gv_unreshaped

        # pick rows from gχ corresponding to variable SS components (indices are in full_comp)
        gx[21:end, :] .= gχ_unreshaped[sb21_surrogate.IDX_of_variable_components_in_SS_adj .+ 36, :]

        ∂o = NoTangent()   # layer has no trainable params
        return NoTangent(), ∂o, gx
    end

    return (χ, v), pullback
end

#=
ARCHIVED CONNECTION FUNCTIONS

archived: 24 Sept 2025
=#
"""
connection function used within compound model's Parallel layer to reduce ŷ_classifier with ŷ_regressor.
"""
function zz_connection(y_clas::T, y_reg::T) where {T <: Union{Matrix, CuArray}}
    # using @view to get a StridedArray and avoid Scalar indexing
    y_phase_frac = @view(y_reg[1:22, :]) .* y_clas

    reg_indices  = [22 + (6*(i-1)+1):22 + (6*i) for i in 1:15]
    clas_indices = [7+i:7+i for i in 1:15]

    y_ss_comp = mapreduce((reg_indices, clas_indices) -> @view(y_reg[reg_indices, :]) .* repeat(@view(y_clas[clas_indices, :]), 6, 1),
                          vcat, reg_indices, clas_indices)

    return vcat(y_phase_frac, y_ss_comp)
end


"""
connection function used within compound model's Parallel layer to reduce ŷ_classifier with ŷ_regressor.

This function is used for models that only predict variable variables for SS-phases.
"""
function zz_connection_reduced_ss_comp(y_clas::T, y_reg::T) where {T <: Union{Matrix, CuArray}}
    # using @view to get a StridedArray and avoid Scalar indexing
    y_phase_frac = @view(y_reg[1:22, :]) .* y_clas

    ss_comp_asm = vcat([repeat(@view(y_clas[7+i:7+i, :]), n, 1) for (n, i) in zip(N_variable_components_in_SS, 1:length(y_clas[8:end, 1]))]...)

    y_ss_comp = @view(y_reg[23:end, :]) .* ss_comp_asm

    return vcat(y_phase_frac, y_ss_comp)
end
