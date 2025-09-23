# Some phases are never stable for the training
# Nepheline, Corundum, Post-Perovskite
IDX_of_phases_never_stable = [5, 7, 19]

"""
Takes DataFrame of Training/Validation/Test data, returns a Matrix with each vector (vector of features) being an independent datapoint.
Filters to only extract phases that are predicted as part of the stable assemblage at least once in the dataset.

This method (for_classifier) generate a boolean y-matrix with assemblage vectors as entries.
"""
function preprocess_for_classifier(x_data::DataFrame, y_data::DataFrame)
    x = Matrix(Matrix{Float32}(x_data)')

    # set up indices of stable phases
    idx_stable_phases = [i for i in 1:22 if i ∉ IDX_of_phases_never_stable]

    y = Matrix(Matrix{Float32}(y_data)')[idx_stable_phases,:]
    # transform modes-submatrix into boolen (one-hot) matrix
    y = y .!= 0.0

    return x::Matrix, y::Union{Matrix, BitMatrix}
end

function preprocess_for_regressor(x_data::DataFrame, y_data::DataFrame)

end
