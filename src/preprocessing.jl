"""
Takes DataFrame of Training/Validation/Test data, returns a Matrix with each vector (vector of features) being an independent datapoint.
Filters to only extract phases that are predicted as part of the stable assemblage at least once in the dataset.

This method (for_classifier) generate a boolean y-matrix with assemblage vectors as entries.
"""
function preprocess_for_classifier(x_data::DataFrame, y_data::DataFrame)
    x = Matrix(Matrix{Float32}(x_data)')

    # set up indices of stable phases
    idx_stable_phases = [i for i in 1:22 if i ∉ IDX_OF_PHASES_NEVER_STABLE]

    y = Matrix(Matrix{Float32}(y_data)')[idx_stable_phases,:]
    # transform modes-submatrix into boolean (one-hot) matrix
    y = y .!= 0.0

    return x::Matrix, y::Union{Matrix, BitMatrix}
end


"""
Takes DataFrame of Training/Validation/Test data, returns a Matrix with each vector (vector of features) being an independent datapoint.
Filters to only extract phases that are predicted as part of the stable assemblage at least once in the dataset.
For the SS composition only variable components are extracted, e.g. no Si in Olivine as this is constant.
"""
function preprocess_for_regressor(x_data::DataFrame, y_data::DataFrame)
    x = Matrix(Matrix{Float32}(x_data)')

    idx_stable_phases, idx_SS_variable_and_stable = indices_of_stable_phases()

    n_features = size(y_data, 2)
    y = Matrix(Matrix{Float32}(y_data)')[[idx_stable_phases..., idx_SS_variable_and_stable..., n_features-2, n_features-1, n_features],:]

    return x::Matrix, y::Matrix
end


"""
Takes DataFrame of Training/Validation/Test data, returns a Matrix with each vector (vector of features) being an independent datapoint.
Filters to only extract phases that are predicted as part of the stable assemblage at least once in the dataset.
For the SS composition only variable components are extracted, e.g. no Si in Olivine as this is constant.

Extract modes + ss-composition only (no bulk rock physical params)
"""
function preprocess_for_regressor_modes_sscomp(x_data::DataFrame, y_data::DataFrame)
    x = Matrix(Matrix{Float32}(x_data)')

    idx_stable_phases, idx_SS_variable_and_stable = indices_of_stable_phases()

    y = Matrix(Matrix{Float32}(y_data)')[[idx_stable_phases..., idx_SS_variable_and_stable...],:]

    return x::Matrix, y::Matrix
end


"""
Used within preprocess-functions.
Set indices for the filtering.
"""
function indices_of_stable_phases()
    # set up indices of stable phases
    idx_stable_phases = [i for i in 1:22 if i ∉ IDX_OF_PHASES_NEVER_STABLE]
    # set up indices of compositional entries in SS that are variable, that belong to stable phases
    IDX_of_SS_never_stable = [i for i in IDX_OF_PHASES_NEVER_STABLE if i > 7] .- 7
    idx_SS_variable_and_stable = [i for i in IDX_of_variable_components_in_SS if i ∉ [k for j in IDX_of_SS_never_stable for k in (j-1)*6+1:j*6]]

    # correct to start at index 23
    idx_SS_variable_and_stable .+= 22

    return idx_stable_phases, idx_SS_variable_and_stable
end
