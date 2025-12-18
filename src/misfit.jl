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
using Flux
#=====================================================================
(1) Misfit metrics to evaluate the performance of the classifier model
=====================================================================#

"""
Q_asm loss function defined as (1 - Q_asm), where Q_asm is the assemblage quality factor defined in [Duesterhoeft, E. & Lanari, P. (2020)](https://doi.org/10.1111/jmg.12538).
"""
function loss_asm(ŷ::T1, y::T2; agg = mean, ϵ = 0.5) where {T1 <: AbstractArray{Float32}, T2 <: AbstractArray{Bool}}
    p̂ = ŷ .> ϵ
    p = y
    # total number of present phases
    k = sum(p̂ .| p, dims=1)
    # number of matching phases
    l = sum(p̂ .& p, dims=1)

    return agg(1 .- l ./ k)
end


"""
Binary focal loss following the implementation [Flux.jl](https://github.com/FluxML/Flux.jl/blob/461a1b670f15279f9251c6d627554abeac44a906/src/losses/functions.jl#L275-L318)

Fixed error of focal loss returning NaN when ŷ -> 1.0 within the range of ϵ for Float32. This method uses clamp() instead of adding ϵ to ŷ (Flux.jl implemenation)
"""
function binary_focal_loss(ŷ::T1, y::T2; agg=mean, gamma=2, eps::Real=Flux.epseltype(ŷ)) where {T1 <: AbstractArray{Float32}, T2 <: Union{AbstractArray{Bool}, AbstractArray{Float32}}}
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


"""
Fraction of a batch of data, for which the assemblage is off by one or more phase(s).
"""
function fraction_mismatched_asm(ŷ::T1, y::T2; ϵ = 0.5) where {T1 <: AbstractArray{Float32}, T2 <: AbstractArray{Bool}}
    p̂ = ŷ .> ϵ
    p = y
    mismatch = p̂ .!= p

    return sum(sum(mismatch, dims=1) .!= 0) / size(p)[end]
end


"""
Fraction of phases of a batch of data that are not predicted correctly.
"""
function fraction_mismatched_phases(ŷ::T1, y::T2; ϵ = 0.5) where {T1 <: AbstractArray{Float32}, T2 <: AbstractArray{Bool}}
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


#=====================================================================
(2b) Misfit metrics considering mass-balance
=====================================================================#

#//TODO - Hacky... Some indices are hard-coded. Should be generalized in the future.
function recalculate_bulk((𝑣_ŷ, 𝐗_ŷ); pure_phase_comp = reshape(PP_COMP_adj, 6, :))
    return pure_phase_comp ⊠ 𝑣_ŷ[1:6, :, :] .+ 𝐗_ŷ[:, :, :] ⊠ 𝑣_ŷ[7:end, :, :]
end

#//TODO - Hacky... Some indices are hard-coded. Should be generalized in the future.
"""
Mass-balance misfit: Absolute deviation between input bulk rock composition and reconstructed bulk rock composition from predicted phase proportions and compositions.
"""
function mass_balance_abs_misfit((𝑣_ŷ, 𝐗_ŷ), x_bulk; agg = mean, pure_phase_comp = reshape(PP_COMP_adj, 6, :))
    bulk_reconstructed = recalculate_bulk((𝑣_ŷ, 𝐗_ŷ), pure_phase_comp = pure_phase_comp)
    return agg(abs.(x_bulk .- bulk_reconstructed))
end

#//TODO - Hacky... Some indices are hard-coded. Should be generalized in the future.
"""
Mass-balance misfit: Relative deviation with respect to input bulk rock composition of the reconstructed bulk rock composition from predicted phase proportions and compositions.
"""
function mass_balance_rel_misfit((𝑣_ŷ, 𝐗_ŷ), x_bulk; agg = mean, pure_phase_comp = reshape(PP_COMP_adj, 6, :))
    bulk_reconstructed = recalculate_bulk((𝑣_ŷ, 𝐗_ŷ), pure_phase_comp = pure_phase_comp)
    return agg(abs.(x_bulk .- bulk_reconstructed) ./ (x_bulk .+ eps(Float32)))
end

#//TODO - Hacky... Some indices are hard-coded. Should be generalized in the future.
"""
Mass-balance misfit: Relative deviation with respect to input bulk rock composition of the reconstructed bulk rock composition from predicted phase proportions and compositions.
"""
function mass_residual((𝑣_ŷ, 𝐗_ŷ); agg = mean, pure_phase_comp = reshape(PP_COMP_adj, 6, :))
    bulk_reconstructed = recalculate_bulk((𝑣_ŷ, 𝐗_ŷ), pure_phase_comp = pure_phase_comp)
    residual = sum(bulk_reconstructed, dims=1) .- 1.0
    return agg(abs.(residual))
end

end # module misfit