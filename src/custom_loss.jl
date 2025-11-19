function loss_asm(ŷ::VecOrMat{Float32}, y::VecOrMat{Float32}; agg = mean, ϵ = 1e-3)
    p̂ = ŷ .> ϵ
    p = y .> 0.
    # total number of present phases
    k = sum(p̂ .| p, dims=1)
    # number of matching phases
    l = sum(p̂ .& p, dims=1)

    return agg(1 .- l ./ k)
end
function loss_asm(ŷ::VecOrMat{Float32}, y::BitMatrix; agg = mean, ϵ = 0.5)
    p̂ = ŷ .> ϵ
    p = y
    # total number of present phases
    k = sum(p̂ .| p, dims=1)
    # number of matching phases
    l = sum(p̂ .& p, dims=1)

    return agg(1 .- l ./ k)
end

#=====================================================================
Additional metrics to evaluate the performance of the classifier model
=====================================================================#
"""
Returns the fraction of a batch of data, for which the assemblage is off by
one or more phase(s).
"""
function fraction_mismatched_asm(ŷ::VecOrMat{Float32}, y::BitMatrix; ϵ = 0.5)
    p̂ = ŷ .> ϵ
    p = y
    mismatch = p̂ .!= p

    return sum(sum(mismatch, dims=1) .!= 0) / size(p)[2]
end
"""
Returns the fraction of phases of a batch of data that are not predicted correctly.
"""
function fraction_mismatched_phases(ŷ::VecOrMat{Float32}, y::BitMatrix; ϵ = 0.5)
    p̂ = ŷ .> ϵ
    p = y
    mismatch = p̂ .!= p

    return sum(mismatch) / prod(size(p))
end


function loss_vol(ŷ, y; agg = mean, ϵ = 1e-3)
    # check if a phase can be considered present (v > 0 or ϵ)
    p̂ = ŷ .> ϵ
    p = y .> 0
    # phase considered if present in either prediction or reference
    p = p .| p̂

    return agg(abs.(ŷ[p] .- y[p]))
end


#=====================================================================
Additional metrics to evaluate the performance of the regressor model
=====================================================================#
"""
Mean absolute error for non-zero y-values
"""
function mae_no_zeros(ŷ, y; agg = mean)
    non_zero_idx = y .!= 0.0

    return agg(abs.(y[non_zero_idx] .- ŷ[non_zero_idx]))
end

"""
Mean relative error for non-zero y-values
"""
function mre_no_zeros(ŷ, y; agg = mean)
    non_zero_idx = y .!= 0.0

    return agg(abs.(y[non_zero_idx] .- ŷ[non_zero_idx]) ./ y[non_zero_idx])
end

"""
Mean absolute error treating correctly predicted zero y-values as trivial
"""
function mae_trivial_zeros(ŷ, y; agg = mean)
    non_zero_idx_y = y .!= 0.0
    non_zero_idx_ŷ = ŷ .!= 0.0

    non_zero_idx = non_zero_idx_y .| non_zero_idx_ŷ

    return agg(abs.(y[non_zero_idx] .- ŷ[non_zero_idx]))
end

"""
Mean relative error treating correctly predicted zero y-values as trivial
"""
function mre_trivial_zeros(ŷ, y; agg = mean, ϵ=eps(Float32))
    non_zero_idx_y = y .!= 0.0
    non_zero_idx_ŷ = ŷ .!= 0.0

    non_zero_idx = non_zero_idx_y .| non_zero_idx_ŷ

    return agg(abs.(y[non_zero_idx] .- ŷ[non_zero_idx]) ./ max.(y[non_zero_idx], ϵ))
end
