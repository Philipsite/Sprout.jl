
"""
Normalise data feature-wise
"""
struct Norm
    mean::Array{Float32, 3}
    std ::Array{Float32, 3}
end
function Norm(x::AbstractArray)
    mean = Statistics.mean(x, dims=ndims(x))
    std = Statistics.std(x, dims=ndims(x))
    return Norm(mean, std)
end

function (n1::Norm)(x::AbstractArray)
    x_n = (x .- n1.mean) ./ n1.std
    x_n[isnan.(x_n)] .= 0

    return x_n
end

# de-normalise call
function denorm(n::Norm, x_n::AbstractArray)
    x = x_n .* n.std .+ n.mean
    return x
end


"""
Min-Max scale data feature-wise
"""
struct MinMaxScaler
    min::Array{Float32, 3}
    max::Array{Float32, 3}
end
function MinMaxScaler(x::AbstractArray)
    min = Statistics.minimum(x, dims=ndims(x))
    max = Statistics.maximum(x, dims=ndims(x))
    return MinMaxScaler(min, max)
end

function (mms::MinMaxScaler)(x::AbstractArray)
    x_s = (x .- mms.min) ./ (mms.max .- mms.min)
    x_s = replace(x_s, NaN32 => 0)
    return x_s
end

# invert scaling
function descale(mms::MinMaxScaler, x_s::AbstractArray)
    x = x_s .* (mms.max .- mms.min) .+ mms.min
    return x
end
