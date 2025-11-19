"""
Training loop
"""
function train_loop(model, loader, opt_state, val_data::Tuple, loss_f::Function, max_epochs::Int;
                    metrics::Vector{<:Function}, early_stopping_condition::Function = (val_loss) -> false, lr_schedule = nothing,
                    gpu_device::Union{Nothing, CUDADevice} = nothing, save_to_subdir::Union{Nothing, AbstractString} = nothing, show_progressbar::Bool = true)

    # init NamedTuple for logged loss and metrics
    log_names = vcat([:batch_loss, :mean_loss, :val_loss], [nameof(m) for m in metrics])
    log_vecs = vcat([Matrix{Float32}(undef, max_epochs, ceil(Int, size(loader.data[1])[2] / loader.batchsize)), Vector{Float32}(undef, max_epochs), Vector{Float32}(undef, max_epochs)],
                    [Vector{Float32}(undef, max_epochs) for _ in metrics])
    logs = NamedTuple{Tuple(log_names)}(log_vecs)
    epoch_trained = 0

    if show_progressbar
        iter = ProgressBar(1:max_epochs)
    else
        iter = 1:max_epochs
    end

    # TRAINING
    for epoch in iter
        # Learning-rate scheduling
        if !isnothing(lr_schedule)
            eta = lr_schedule(epoch)
            Flux.adjust!(opt_state, eta)
        end

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

        val_loss = loss_f(ŷ_val, y_val)
        logs.val_loss[epoch] = val_loss

        for m in metrics
            logs[nameof(m)][epoch] = m(ŷ_val, y_val)
        end

        epoch_trained += 1

        # if an early_stopping condition is given as kwarg, check
        # default for early_stopping condition returns false > right-side of && never evaluated
        early_stopping_condition(val_loss) && break
    end

    # All results of training should be save into one directory
    if isnothing(save_to_subdir)
        dir = pwd() * "/sb21_sm_" * Dates.format(now(),"yyyyudd_HHMM")
        mkdir(dir)
    else
        subdir = pwd() * "/" * save_to_subdir
        if !isdir(subdir)
            mkdir(subdir)
        end
        dir = pwd() * "/" * save_to_subdir * "/sb21_sm_" * Dates.format(now(),"yyyyudd_HHMM")
        mkdir(dir)
    end

    # save model
    # move to cpu
    m_cpu = cpu(model)
    model_state = Flux.state(m_cpu)
    jldsave(dir * "/saved_model.jld2"; model_state)

    # save opt_state
    jldsave(dir * "/saved_opt.jld2"; opt_state)

    # trim the logs to the length of epoch_trained
    logs_t = map(log_vector -> typeof(log_vector) <: Matrix ? log_vector[1:epoch_trained, :] : log_vector[1:epoch_trained], logs)
    # save log
    jldsave(dir * "/log.jld2"; logs_t)

    return model, opt_state, logs_t, dir
end


function post_training_plots(logs::NamedTuple, log_dir_path::String)
    # retrieve all metrics
    metrics = [k for k in keys(logs) if !endswith(String(k),"_loss")]

    loss_color = :lightskyblue
    val_loss_color = :royalblue
    metric_color = cgrad(:sun)[1:length(metrics)]

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
               yticklabelcolor = metric_color[1],
               rightspinecolor = metric_color[1],
               ytickcolor = metric_color[1],
               ylabelcolor = metric_color[1],
               ylabel="Metric score")

    loss_line = lines!(ax1, 1:length(logs.mean_loss), logs.mean_loss; color = loss_color)
    val_loss_line = lines!(ax1, 1:length(logs.mean_loss), logs.val_loss; color = val_loss_color)
    for (i, m) in enumerate(metrics)
        m_line = lines!(ax2, 1:length(logs.mean_loss), logs[m]; color = metric_color[i], label=String(m))
    end
    hidespines!(ax2, :l, :b, :t)
    hidexdecorations!(ax2)

    axislegend(ax1, [loss_line, val_loss_line], ["Loss", "Loss (val)"], position = :lb, framevisible=false)
    axislegend(ax2, position = :rt, framevisible=false)

    ax3 = Axis(fig[2, 1], xscale = log10,
               ygridvisible=false, xgridvisible=false,
               yaxisposition = :right,
               yticklabelcolor = metric_color[1],
               rightspinecolor = metric_color[1],
               ytickcolor = metric_color[1],
               ylabelcolor = metric_color[1],
               ylabel="Metric score")

    for (i, m) in enumerate(metrics)
        m_line = lines!(ax3, 1:length(logs.mean_loss), logs[m]; color = metric_color[i], label=String(m))
    end

    xlims!(ax3, (0.9 * length(logs.mean_loss), length(logs.mean_loss)+0.2*length(logs.mean_loss)))

    axislegend(ax3, position = :rt, framevisible=false)

    save(log_dir_path * "/train_log.pdf", fig)
    return fig
end


function post_training_plots_asm(logs::NamedTuple, log_dir_path::String)
    loss_color = :lightskyblue
    val_loss_color = :royalblue
    asm_color = :crimson
    frac_mismatch_asm_color = :red
    frac_mismatch_phases_color = :orange

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
               ylabel="Metric value")

    loss_line = lines!(ax1, 1:length(logs.mean_loss), logs.mean_loss; color = loss_color)
    val_loss_line = lines!(ax1, 1:length(logs.mean_loss), logs.val_loss; color = val_loss_color)

    qasm_line = lines!(ax2, 1:length(logs.mean_loss), logs.loss_asm; color = asm_color)
    frac_mismatch_asm_line = lines!(ax2, 1:length(logs.fraction_mismatched_asm), logs.fraction_mismatched_asm; color = frac_mismatch_asm_color)
    frac_mismatch_phases_line = lines!(ax2, 1:length(logs.fraction_mismatched_phases), logs.fraction_mismatched_phases; color = frac_mismatch_phases_color)

    hidespines!(ax2, :l, :b, :t)
    hidexdecorations!(ax2)

    axislegend(ax1, [loss_line, val_loss_line], ["Loss", "Loss (val)"], position = :lb, framevisible=false)
    axislegend(ax2, [qasm_line, frac_mismatch_asm_line, frac_mismatch_phases_line], ["Q_asm", "Mismatched asm", "Mismatched phases"], position = :rt, framevisible=false)

    ax3 = Axis(fig[2, 1], xscale = log10,
               ygridvisible=false, xgridvisible=false,
               yaxisposition = :right,
               yticklabelcolor = asm_color,
               rightspinecolor = asm_color,
               ytickcolor = asm_color,
               ylabelcolor = asm_color,
               ylabel="Q(assemblage)")

    qasm_line = lines!(ax3, 1:length(logs.mean_loss), logs.loss_asm; color = asm_color)
    frac_mismatch_asm_line = lines!(ax3, 1:length(logs.fraction_mismatched_asm), logs.fraction_mismatched_asm; color = frac_mismatch_asm_color)
    frac_mismatch_phases_line = lines!(ax3, 1:length(logs.fraction_mismatched_phases), logs.fraction_mismatched_phases; color = frac_mismatch_phases_color)


    xlims!(ax3, (8, length(logs.mean_loss)+0.2*length(logs.mean_loss)))
    ylims!(ax3, (-0.01,0.11))

    save(log_dir_path * "/train_log.pdf", fig)
    return fig
end
