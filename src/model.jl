# Remove the entry for post-perovskite, as this phase is never predicted to be stable
N_variable_components_in_SS_adjusted = N_variable_components_in_SS[[1:11...,13:end...]]

"""
connection function used within compound model's Parallel layer to reduce ŷ_classifier with ŷ_regressor.

This function is used for models that only predict phases that are stale in at least one asemblage of the training generate_dataset,
and only predicts the variable compositional variables for SS-phases.

Including bulk rock physical params.
"""
function connection_reduced_phys_params(y_clas::T, y_reg::T) where {T <: Union{Matrix, CuArray}}
    # using @view to get a StridedArray and avoid Scalar indexing
    y_phase_frac = @view(y_reg[1:19, :]) .* y_clas

    ss_comp_asm = vcat([repeat(@view(y_clas[5+i:5+i, :]), n, 1) for (n, i) in zip(N_variable_components_in_SS_adjusted, 1:length(y_clas[6:end, 1]))]...)
    y_ss_comp = @view(y_reg[20:end-3, :]) .* ss_comp_asm

    y_phys_prop = @view(y_reg[end-2:end, :])

    return vcat(y_phase_frac, y_ss_comp, y_phys_prop)
end


"""
connection function used within compound model's Parallel layer to reduce ŷ_classifier with ŷ_regressor.

This function is used for models that only predict phases that are stale in at least one asemblage of the training generate_dataset,
and only predicts the variable compositional variables for SS-phases.

Excluding bulk rock physical params.
"""
function connection_reduced(y_clas::T, y_reg::T) where {T <: Union{Matrix, CuArray}}
    # using @view to get a StridedArray and avoid Scalar indexing
    y_phase_frac = @view(y_reg[1:19, :]) .* y_clas

    ss_comp_asm = vcat([repeat(@view(y_clas[5+i:5+i, :]), n, 1) for (n, i) in zip(N_variable_components_in_SS_adjusted, 1:length(y_clas[6:end, 1]))]...)
    y_ss_comp = @view(y_reg[20:end, :]) .* ss_comp_asm

    return vcat(y_phase_frac, y_ss_comp)
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