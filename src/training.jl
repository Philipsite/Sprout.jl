"""
Training loop
"""
function train_loop(model, loader, opt_state, val_data::Tuple, loss_f::Function, n_epochs::Int; metrics::Vector{<:Function})
    # init NamedTuple for logged loss and metrics
    log_names = vcat([:batch_loss, :mean_loss, :val_loss], [nameof(m) for m in metrics])
    log_vecs = vcat([Matrix{Float32}(undef, n_epochs, Int(size(loader.data[1])[2] / loader.batchsize)), Vector{Float32}(undef, n_epochs), Vector{Float32}(undef, n_epochs)],
                    [Vector{Float32}(undef, n_epochs) for _ in metrics])
    logs = NamedTuple{Tuple(log_names)}(log_vecs)

    # TRAINING
    for epoch in ProgressBar(1:n_epochs)
        loss_batches = Float32[]
        for xy_cpu in loader
            x, y = xy_cpu

            # compute loss, let Zygote watch the gradient
            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x)
                loss_f(ŷ, y)
            end

            # update model params and opt_state
            Flux.update!(opt_state, model, grads[1])

            # push loss of batch onto loss_batches
            push!(loss_batches, loss)
        end

        # compute metrics to log
        mean_epoch_loss = mean(loss_batches)
        logs.batch_loss[epoch,:] .= loss_batches
        logs.mean_loss[epoch] = mean_epoch_loss

        # predict on validation data
        x_val = val_data[1]
        y_val = val_data[2]

        ŷ_val = model(x_val)

        logs.val_loss[epoch] = loss_f(ŷ_val, y_val)

        for m in metrics
            logs[nameof(m)][epoch] = m(ŷ_val, y_val)
        end
    end

    # All results of training should be save into one directory
    dir = pwd() * "/sb21_sm_" * Dates.format(now(),"yyyyudd_HHMM")
    mkdir(dir)

    # save model
    model_state = Flux.state(model)
    jldsave(dir * "/saved_model.jld2"; model_state)

    # save opt_state
    jldsave(dir * "/saved_opt.jld2"; opt_state)

    # save log
    jldsave(dir * "/log.jld2"; log)

    return model, opt_state, logs
end
