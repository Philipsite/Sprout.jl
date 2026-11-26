
"""
Returns indices of data points that do not contain NaN values.
"""
function filter_NaN(data::Matrix)
    return [!any(isnan, data[:, j]) for j in 1:size(data, 2)]
end


"""
Used within preprocess-functions.
Set indices for the filtering.
"""
function indices_of_stable_phases()
    # set up indices of stable phases > to extract phase fractions
    n_phases = length(PP) + length(SS)
    idx_stable_phases = [i for i in 1:n_phases if i ∉ IDX_OF_PHASES_NEVER_STABLE]

    # setup indices of stable solid solution components > to extract solid solution compositions
    # offset by number of phases
    idx_stable_ss = 1:(length(SS)*6)
    idx_stable_ss = [i for i in idx_stable_ss if i ∉ [6 * (k-1) + j for k in Sprout.IDX_SS_NEVER_STABLE for j in 1:6]] .+ n_phases

    return idx_stable_phases, idx_stable_ss
end


"""
Takes DataFrame of Training/Validation/Test data, returns:
- x:: Array{Float32, 3} (Vec, 1, N)        - Input features P [GPa], T [°C], bulk composition [molmol⁻¹]
- 𝑣:: Array{Float32, 3} (Vec, 1, N)        - Phase fraction [molmol⁻¹]
- 𝐗_ss:: Array{Float32, 3} (Matrix, N)     - Solid solution phase compositions [molmol⁻¹]
- 𝐩_sys:: Array{Float32, 3} (n_props, 1, N) - Bulk-system properties, see `BULK_PROPS` for order and units

Applies the following filters:
- filter observation containing NaN
- only extract phases that are predicted as part of the stable assemblage at least once in the dataset.

"""
function preprocess_data(x_data::DataFrame, y_data::DataFrame;
                         bulk_props::AbstractVector{<:AbstractString} = BULK_PROPS)

    x = Matrix(Matrix{Float32}(x_data)')
    y = Matrix((Matrix{Float32}(y_data))')

    # filter data points with NaNs (failed minimisations? > failed volume computation!)
    cols_no_nan = filter_NaN(x) .& filter_NaN(y)

    # (1) INPUTS
    x = x[:, cols_no_nan]
    x = reshape(x, size(x, 1), 1, size(x, 2))

    # (2) OUTPUTS
    y = y[:, cols_no_nan]
    # filter the stable phases only
    idx_stable_phases, idx_stable_ss = indices_of_stable_phases()

    𝑣 = y[idx_stable_phases, :]
    𝑣 = reshape(𝑣, size(𝑣, 1), 1, size(𝑣, 2))
    vec_ss = y[idx_stable_ss, :]
    𝐗_ss = reshape(vec_ss, 6, Int(size(vec_ss, 1) / 6), :)

    # bulk-system properties, selected by column name
    𝐩_sys = Matrix{Float32}(Matrix(y_data[cols_no_nan, bulk_props])')
    𝐩_sys = reshape(𝐩_sys, size(𝐩_sys, 1), 1, :)

    return x::Array{Float32, 3}, 𝑣::Array{Float32, 3}, 𝐗_ss::Array{Float32, 3}, 𝐩_sys::Array{Float32, 3}
end


"""
Converts phase fraction matrix to one-hot encoded phase stability matrix.
"""
function one_hot_phase_stability(𝑣::Array{Float32})
    return 𝑣 .!= 0.0
end
