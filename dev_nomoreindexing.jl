using Flux
using BenchmarkTools

x = rand(Float32, 8, 1, 5000)

y = zeros(Float32, 3, 5, 5000)
y[1, 1, :] = sum(sin.(x), dims=1)
y[1, 2, :] .= 10
y[2, 3, :] = sum(cos.(x.^2), dims=1)
y[2, 4, :] .= 3
y[3, 5, :] = sum(2 .*x .+ exp.(x), dims=1)

data = Flux.DataLoader((x, y))

const var_mask = Float32[1 0 0 0 0;
                         0 0 1 0 0;
                         0 0 0 0 1]

const fc_vals = Float32[0 10 0 0 0;
                        0  0 0 3 0;
                        0  0 0 0 0]

stability_mask = [0, 1, 0, 0, 0]

y .* var_mask.+ fc_vals

y .* stability_mask'


struct ReshapeLayer
    n :: Int
    m :: Int
end
Flux.@layer ReshapeLayer
Flux.trainable(rl::ReshapeLayer) = (;)
(rl::ReshapeLayer)(x::Array{Float32,3}) = reshape(x, rl.n, rl.m, :)

struct InjectLayer
    var_mask :: Matrix{Float32}
    fc_vals  :: Matrix{Float32}
end
function InjectLayer()
    return InjectLayer(var_mask, fc_vals)
end
Flux.@layer InjectLayer
Flux.trainable(il::InjectLayer) = (;)
(il::InjectLayer)(x::Array{Float32,3}) = x .* il.var_mask.+ il.fc_vals
# (il::InjectLayer)(x::AbstractArray) = x .* il.var_mask.+ il.fc_vals

m = Chain(
    Dense(size(x, 1), 64, relu),
    Dense(64, *(size(y)[1:2]...)),
    ReshapeLayer(size(y)[1:2]...),
    InjectLayer())


ŷ_prior = m(x[:, :, 1:1])

opt_state = Flux.setup(Flux.Adam(0.001), m)

loss_f = (ŷ, y) -> sum((ŷ .- y).^2)

gradient((m, x, y) -> loss_f(m(x), y), m, x, y)

losses = []
for d in data
    x, y = d
    # compute loss, let Zygote watch the gradient
    loss, grads = Flux.withgradient(m) do m
        ŷ = m(x)
        loss_f(ŷ, y)
    end

    # update model params and opt_state
    Flux.update!(opt_state, m, grads[1])

    # push loss of batch onto loss_batches
    push!(losses, loss)

end

ŷ_post = m(x[:, :, 1:1])

ŷ_prior
ŷ_post
y[:,:,1]