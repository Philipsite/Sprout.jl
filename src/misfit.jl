"""
Custom loss and metric functions for misfit evaluation.

These function should follow the convention:
```
ϕ = f(ŷ, y; agg = mean, kwargs...)
```
where `ŷ` is the model prediction, `y` is the ground truth, `agg` is an aggregation function (mean, sum, etc.) and `kwargs` are any additional keyword arguments.
"""
module misfit
using Statistics
#=====================================================================
(1) Misfit metrics to evaluate the performance of the classifier model
=====================================================================#
#//TODO - test this function
"""
Q_asm loss function defined as (1 - Q_asm), where Q_asm is the assemblage quality factor defined in [Duesterhoeft, E. & Lanari, P. (2020)](https://doi.org/10.1111/jmg.12538).
"""
function loss_asm(ŷ::T, y::T; agg = mean, ϵ = 0.5) where T <: AbstractArray{Float32}
    p̂ = ŷ .> ϵ
    p = y .> 0
    # total number of present phases
    k = sum(p̂ .| p, dims=1)
    # number of matching phases
    l = sum(p̂ .& p, dims=1)

    return agg(1 .- l ./ k)
end

#//TODO - test this function
"""
Binary focal loss following the implementation [Flux.jl](https://github.com/FluxML/Flux.jl/blob/461a1b670f15279f9251c6d627554abeac44a906/src/losses/functions.jl#L275-L318)

Fixed error of focal loss returning NaN when ŷ -> 1.0 within the range of ϵ for Float32. This method uses clamp() instead of adding ϵ to ŷ (Flux.jl implemenation)
"""
function binary_focal_loss(ŷ::T, y::T; agg=mean, gamma=2, eps::Real=Flux.epseltype(ŷ)) where T <: AbstractArray{Float32}
    γ = gamma isa Integer ? gamma : ofeltype(ŷ, gamma)
    Flux.Losses._check_sizes(ŷ, y)

    # Clamp to avoid log(0), negative values, or >1 values
    ŷϵ = clamp.(ŷ, eps, 1 - eps)

    # Standard p_t definition
    p_t = y .* ŷϵ .+ (1 .- y) .* (1 .- ŷϵ)

    ce = .-log.(p_t)
    weight = (1 .- p_t) .^ γ
    loss = weight .* ce

    return agg(loss)
end

#//TODO - test this function
"""
Fraction of a batch of data, for which the assemblage is off by one or more phase(s).
"""
function fraction_mismatched_asm(ŷ::T, y::T; ϵ = 0.5) where T <: AbstractArray{Float32}
    p̂ = ŷ .> ϵ
    p = y
    mismatch = p̂ .!= p

    return sum(sum(mismatch, dims=1) .!= 0) / size(p)[end]
end

#//TODO - test this function
"""
Fraction of phases of a batch of data that are not predicted correctly.
"""
function fraction_mismatched_phases(ŷ::T, y::T; ϵ = 0.5) where T <: AbstractArray{Float32}
    p̂ = ŷ .> ϵ
    p = y
    mismatch = p̂ .!= p

    return sum(mismatch) / prod(size(p))
end


#=====================================================================
(2) Misfit metrics to evaluate the performance of the regressor model
=====================================================================#
#//TODO - Adapt + test this function + write doc string
function loss_vol(ŷ, y; agg = mean, ϵ = 1e-3)
    # check if a phase can be considered present (v > 0 or ϵ)
    p̂ = ŷ .> ϵ
    p = y .> 0
    # phase considered if present in either prediction or reference
    p = p .| p̂

    return agg(abs.(ŷ[p] .- y[p]))
end


"""
Mean absolute error for non-zero y-values
"""
function mae_no_zeros(ŷ, y; agg = mean)
    non_zero_idx = y .!= 0.0

    return agg(abs.(y[non_zero_idx] .- ŷ[non_zero_idx]))
end


"""
Mean relative error for non-zero y-values
"""
function mre_no_zeros(ŷ, y; agg = mean)
    non_zero_idx = y .!= 0.0

    return agg(abs.(y[non_zero_idx] .- ŷ[non_zero_idx]) ./ y[non_zero_idx])
end


"""
Mean absolute error treating correctly predicted zero y-values as trivial
"""
function mae_trivial_zeros(ŷ, y; agg = mean)
    non_zero_idx_y = y .!= 0.0
    non_zero_idx_ŷ = ŷ .!= 0.0

    non_zero_idx = non_zero_idx_y .| non_zero_idx_ŷ

    return agg(abs.(y[non_zero_idx] .- ŷ[non_zero_idx]))
end


"""
Mean relative error treating correctly predicted zero y-values as trivial
"""
function mre_trivial_zeros(ŷ, y; agg = mean, ϵ=eps(Float32))
    non_zero_idx_y = y .!= 0.0
    non_zero_idx_ŷ = ŷ .!= 0.0

    non_zero_idx = non_zero_idx_y .| non_zero_idx_ŷ

    return agg(abs.(y[non_zero_idx] .- ŷ[non_zero_idx]) ./ max.(y[non_zero_idx], ϵ))
end

end