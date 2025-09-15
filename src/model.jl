"""
connection fucntion used within compund models Parallel layer to reduce ŷ_classifier with ŷ_regressor.
"""
function connection(y_clas::Matrix, y_reg::Matrix)
    y_phase_frac = y_reg[1:22, :] .* y_clas

    y_ss_vecs = [y_reg[22+(6*(i-1)+1):22+(6*i), :] .* repeat(y_clas[7+i:7+i, :], inner=(6,1)) for i = 1:15]
    y_ss_comp = vcat(y_ss_vecs...)

    return vcat(y_phase_frac, y_ss_comp)
end
