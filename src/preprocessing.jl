
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
    idx_stable_ss = [i for i in idx_stable_ss if i ∉ [6 * k + j for k in sb21_surrogate.IDX_SS_NEVER_STABLE for j in 1:6]] .+ n_phases

    return idx_stable_phases, idx_stable_ss
end


"""
Takes DataFrame of Training/Validation/Test data, returns:
- x    :: Matrix{Float32}     - Input features P [GPa], T [°C], bulk composition [molmol⁻¹]
- 𝑣    :: Matrix{Float32}     - Phase fraction [molmol⁻¹]
- 𝐗_ss :: Array{Float32, 3}   - Solid solution phase compositions [molmol⁻¹]
— ρ    :: Vector{Float32}     - System densities
- Κ    :: Vector{Float32}     - Bulk moduli
- μ    :: Vector{Float32}     - Shear moduli

Applies the foloowing filters:
- filter observation containing NaN
- only extract phases that are predicted as part of the stable assemblage at least once in the dataset.

"""
function preprocess_data(x_data::DataFrame, y_data::DataFrame)
    x = Matrix(Matrix{Float32}(x_data)')
    y = Matrix((Matrix{Float32}(y_data))')

    # filter data points with NaNs (failed minimisations? > failed volume computation!)
    cols_no_nan = filter_NaN(x) .& filter_NaN(y)

    x = x[:, cols_no_nan]
    y = y[:, cols_no_nan]

    # filter the stable phases only
    idx_stable_phases, idx_stable_ss = indices_of_stable_phases()

    𝑣 = y[idx_stable_phases, :]
    vec_ss = y[idx_stable_ss, :]
    𝐗_ss = reshape(vec_ss, 6, Int(size(vec_ss, 1) / 6), :)
    ρ = y[end - 2, :]
    Κ = y[end - 1, :]
    μ = y[end, :]
    return x::Matrix{Float32}, 𝑣::Matrix{Float32}, 𝐗_ss::Array{Float32,3}, ρ::Vector{Float32}, Κ::Vector{Float32}, μ::Vector{Float32}
end


"""
Converts phase fraction matrix to one-hot encoded phase stability matrix.
"""
function one_hot_phase_stability(𝑣::Matrix{Float32})
    return 𝑣 .!= 0.0
end
