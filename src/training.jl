"""
Training loop
"""
function train_loop(model, loader, opt_state, val_data::Tuple, loss_f::Function, n_epochs::Int; metrics::Vector{<:Function}, gpu_device::Union{Nothing, CUDADevice} = nothing)

    # init NamedTuple for logged loss and metrics
    log_names = vcat([:batch_loss, :mean_loss, :val_loss], [nameof(m) for m in metrics])
    log_vecs = vcat([Matrix{Float32}(undef, n_epochs, Int(size(loader.data[1])[2] / loader.batchsize)), Vector{Float32}(undef, n_epochs), Vector{Float32}(undef, n_epochs)],
                    [Vector{Float32}(undef, n_epochs) for _ in metrics])
    logs = NamedTuple{Tuple(log_names)}(log_vecs)

    # TRAINING
    for epoch in ProgressBar(1:n_epochs)
        loss_batches = Float32[]
        for xy_cpu in loader
            if !isnothing(gpu_device)
                x, y = xy_cpu |> gpu_device
            else
                x, y = xy_cpu
            end

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

        if !isnothing(gpu_device)
            ŷ_val = model(x_val |> gpu_device) |> cpu
        else
            ŷ_val = model(x_val)
        end

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
    jldsave(dir * "/log.jld2"; logs)

    return model, opt_state, logs, dir
end


function post_training_plots(logs::NamedTuple, log_dir_path::String)
    loss_color = :lightskyblue
    val_loss_color = :royalblue
    asm_color = :crimson

    fig = Figure(size = (800, 600))

    ax1 = Axis(fig[1, 1], xscale = log10,
               ygridvisible=false, xgridvisible=false,
               yticklabelcolor = loss_color, 
               leftspinecolor = loss_color,
               ytickcolor = loss_color,
               ylabelcolor = loss_color,
               rightspinevisible = false,
               xlabel="Epochs (iteration over training set)",
               ylabel="Loss value")

    ax2 = Axis(fig[1, 1], xscale = log10,
               ygridvisible=false, xgridvisible=false,
               yaxisposition = :right,
               yticklabelcolor = asm_color,
               rightspinecolor = asm_color,
               ytickcolor = asm_color,
               ylabelcolor = asm_color,
               ylabel="Q(assemblage)")

    loss_line = lines!(ax1, 1:length(logs.mean_loss), logs.mean_loss; color = loss_color)
    val_loss_line = lines!(ax1, 1:length(logs.mean_loss), logs.val_loss; color = val_loss_color)
    qasm_line = lines!(ax2, 1:length(logs.mean_loss), logs.loss_asm; color = asm_color)
    hidespines!(ax2, :l, :b, :t)
    hidexdecorations!(ax2)

    axislegend(ax1, [loss_line, val_loss_line], ["Loss", "Loss (val)"], position = :lb, framevisible=false)

    ax3 = Axis(fig[2, 1], xscale = log10,
               ygridvisible=false, xgridvisible=false,
               yaxisposition = :right,
               yticklabelcolor = asm_color,
               rightspinecolor = asm_color,
               ytickcolor = asm_color,
               ylabelcolor = asm_color,
               ylabel="Q(assemblage)")

    qasm_line = lines!(ax3, 1:length(logs.mean_loss), logs.loss_asm; color = asm_color)

    xlims!(ax3, (8, length(logs.mean_loss)+0.2*length(logs.mean_loss)))           
    ylims!(ax3, (-0.01,0.101))

    save(log_dir_path * "/train_log.pdf", fig)
    return fig
end
