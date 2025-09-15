"""
connection fucntion used within compund models Parallel layer to reduce ŷ_classifier with ŷ_regressor.
"""
function connection(y_clas::T, y_reg::T) where {T <: Union{Matrix, CuArray}}
    # using @view to get a StridedArray and avoid Scalar indexing
    y_phase_frac = @view(y_reg[1:22, :]) .* y_clas

    reg_indices  = [22 + (6*(i-1)+1):22 + (6*i) for i in 1:15]
    clas_indices = [7+i:7+i for i in 1:15]

    y_ss_comp = mapreduce((reg_indices, clas_indices) -> @view(y_reg[reg_indices, :]) .* repeat(@view(y_clas[clas_indices, :]), 6, 1),
                          vcat, reg_indices, clas_indices)

    return vcat(y_phase_frac, y_ss_comp)
end
