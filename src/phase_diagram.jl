

function generate_mineral_assemblage_diagram(P_bounds::Tuple, T_bounds::Tuple, X::Vector{Float32}, resolution::Int, classifier_model::Any, x_norm::Any)
    # Create a P-T grid
    n = resolution
    P = range(P_bounds[1], P_bounds[2], length=n)
    T = range(T_bounds[1], T_bounds[2], length=n)

    # Reverse P for desired orientation
    P_rev = reverse(P)

    # Create 2D grid
    P_grid = Matrix{Float32}(repeat(P_rev, 1, n))
    T_grid = Matrix{Float32}(repeat(T', n, 1))

    P_flat = vec(P_grid)
    T_flat = vec(T_grid)

    # Concatenate P-T with bulk rock composition vectors
    input_vecs = hcat([(vcat(p, t, X)) for (p, t) in zip(P_flat, T_flat)]...)
    input_vecs_n = x_norm(input_vecs)

    # predict assemblages, convert prediction into BitMatrix with 0.5 activation treshold
    ŷ = classifier_model(input_vecs_n) .> 0.5

    # filter phase_names for phases that are never stable
    phase_names = [phase for phase in [PP..., SS...] if phase ∉ ["neph", "co", "ppv"]]

    # tidy up predictions into a grid of assemblages & variance
    asm_vec = []
    var_vec = []
    for y_i in eachcol(ŷ)
        asm = phase_names[y_i]
        var = 6 - length(asm) + 2
        push!(asm_vec, join(asm, "-"))
        push!(var_vec, var)
    end

    asm_grid = reshape(asm_vec, n, n)
    var_vec_grid = reshape(var_vec, n, n)

    return asm_grid, var_vec_grid
end


function plot_mineral_assemblage_diagram(asm_grid::Matrix, var_vec_grid::Matrix, P_bounds::Tuple, T_bounds::Tuple, color::Symbol)
    palette = cgrad(color)

    n = size(asm_grid)[1]

    P = range(P_bounds[1], P_bounds[2], length=n)
    T = range(T_bounds[1], T_bounds[2], length=n)

    # Reverse P for desired orientation
    P_rev = reverse(P)

    # Create 2D grid
    P_grid = Matrix{Float32}(repeat(P_rev, 1, n))
    T_grid = Matrix{Float32}(repeat(T', n, 1))

    fig = Figure(; size=(500, 500))
    ax = Axis(fig[1, 1],  xlabel=L"Temperature\ [°C]", ylabel=L"Pressure\ [kbar]", aspect = 1)
    hm = heatmap!(ax, T, P_rev, var_vec_grid'; colormap=palette, colorrange=(2, 7), interpolate=false)

    # Find unique values and their centroids
    unique_values = unique(vec(asm_grid))
    centroids_x = Float64[]
    centroids_y = Float64[]
    labels = String[]

    for value in unique_values
        # Find all indices where this value occurs
        indices = findall(x -> x == value, asm_grid)

        # Calculate centroid coordinates
        mean_i = mean([idx[1] for idx in indices])  # row indices
        mean_j = mean([idx[2] for idx in indices])  # column indices

        # Convert to actual coordinates
        centroid_x = T_grid[1, Int(round(mean_j))]  # T coordinate
        centroid_y = P_grid[Int(round(mean_i)), 1]  # P coordinate

        push!(centroids_x, centroid_x)
        push!(centroids_y, centroid_y)
        push!(labels, value)
    end

    # Add text labels only at centroids
    text!(ax, centroids_x, centroids_y, text=labels,
          align=(:center, :center), fontsize=8, color=:white)

    return fig
end
