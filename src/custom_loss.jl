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

function loss_vol(ŷ, y; agg = mean, ϵ = 1e-3)
    # check if a phase can be considered present (v > 0 or ϵ)
    p̂ = ŷ .> ϵ
    p = y .> 0
    # phase consideed if present in either prediction or reference
    p = p .| p̂

    return agg(abs.(ŷ[p] .- y[p]))
end

# UNTESTED //TODO
function abs_rel_non_zero_deviation(ŷ, y; agg = mean)
    rel_abs_dev = []
    for (y_vec, ŷ_vec) = zip(eachcol(y), eachcol(ŷ))
        non_zero_idx = y_vec .!= 0.0
    
        y_f = y_vec[non_zero_idx]
        ŷ_f = ŷ_vec[non_zero_idx]

        push!(rel_abs_dev, agg(abs.(y_f .- ŷ_f) ./ y_f))
    end
    return agg(rel_abs_dev)
end
