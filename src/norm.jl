"""
Normalise data feature-wise
"""
struct Norm
    mean::VecOrMat
    std ::VecOrMat
end
function (n1::Norm)(x::AbstractMatrix)
    x_n = (x .- n1.mean) ./ n1.std
    x_n[isnan.(x_n)] .= 0

    return x_n
end

function Norm(x::AbstractMatrix)
    mean = Statistics.mean(x, dims=2)
    std = Statistics.std(x, dims=2)
    return Norm(mean, std)
end

"""
Min-Max scale data feature-wise
"""
struct MinMaxScaler
    min::Vector{Float32}
    max::Vector{Float32}
end
function (mms::MinMaxScaler)(x::AbstractMatrix)
    x_n = (x .- mms.min) ./ (mms.max .- mms.min)
    x_n[isnan.(x_n)] .= 0

    return x_n
end

function MinMaxScaler(x::AbstractMatrix)
    min = Statistics.minimum(x, dims=2)
    max = Statistics.maximum(x, dims=2)
    return MinMaxScaler(min, max)
end
