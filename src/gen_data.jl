# Define mantle composition end-member after Kerswell et al. 2024
# following "Xoxides = ["SiO2"; "CaO";"Al2O3"; "FeO"; "MgO"; "Na2O"]"
DSUM_wt = [44.1, 0.22, 0.261, 7.96, 47.4, 0.042];
PSUM_wt = [46.2, 4.34, 4.88, 8.88, 35.2, 0.33];

function generate_dataset(n::Int, filename_base::String;
                          database              ::String            = "sb21",
                          Xoxides               ::Vector{String}    = ["SiO2"; "CaO"; "Al2O3";  "FeO"; "MgO"; "Na2O"],
                          sys_in                ::String            = "wt",
                          pressure_range_kbar   ::Tuple             = (10., 400.),
                          temperature_range_C   ::Tuple             = (700., 1800.),
                          bulk_em_1             ::Vector{Float64}   = DSUM_wt,
                          bulk_em_2             ::Vector{Float64}   = PSUM_wt,
                          noisy_bulk            ::Bool              = false,
                          λ_dirichlet           ::Real              = 100,
                          phase_list            ::Vector{String}    = [PP..., SS...],
                          save_to_csv           ::Bool              = true)

    # init random generator
    rng = Xoshiro(filename_base)

    # init MAGEMin
    MAGEMin_db = Initialize_MAGEMin(database, solver=0, verbose=false)

    # generate P-T-X_bulk
    pressure_kbar = rand(rng, Uniform(pressure_range_kbar[1], pressure_range_kbar[2]), n)
    temperature_C = rand(rng, Uniform(temperature_range_C[1], temperature_range_C[2]), n)

    if noisy_bulk
        X_bulk = generate_noisy_bulk_array(rng, n; bulk_em_1=bulk_em_1, bulk_em_2=bulk_em_2, λ=λ_dirichlet)
    else
        X_bulk = generate_bulk_array(rng, n; bulk_em_1=bulk_em_1, bulk_em_2=bulk_em_2)
    end

    # GEM
    out = multi_point_minimization(pressure_kbar, temperature_C, MAGEMin_db, X=X_bulk, Xoxides=Xoxides, sys_in=sys_in)

    # extract data
    # use bulk_S as bulk composition, test with "molar conservence" showed less deviation.
    # Is this because the mass residual in MAGEMin is not included into bulk_S? > Check with nico.
    # //ANCHOR - This only works for SB21 which has no liquid phases, not given that this goes trough with other databases
    bulks = reduce(hcat, [out_i.bulk_S for out_i in out])

    oxides_in_out = Matrix{String}(undef, 6, n)
    ph_mode  = zeros(22, n)
    # ph_ρ    = zeros(22, n)
    ss_comp = zeros(90, n)
    phys_prop = zeros(3, n)

    @threads for i in ProgressBar(eachindex(out))
        oxides_in_out[:, i] .= out[i].oxides

        ph_i = out[i].ph
        indices_phaselist = [findfirst(.==(p), phase_list) for p in ph_i]
        is_ss = Bool.(out[i].ph_type)

        # add density, bulk- and shear-modulus
        phys_prop_i = vcat(out[i].rho, out[i].bulkMod, out[i].shearMod)
        phys_prop[:, i] .= phys_prop_i

        # add ph_mode
        ph_mode_i = out[i].ph_frac
        ph_mode[indices_phaselist, i] .= ph_mode_i

        # # add ph_ρ
        # ph_ρ_i = []
        # SS_idx = 0
        # PP_idx = 0
        # for j = eachindex(ph_i)
        #     if is_ss[j]
        #         SS_idx += 1
        #         ρ = out[i].SS_vec[SS_idx].rho
        #     else
        #         PP_idx += 1
        #         ρ = out[i].PP_vec[PP_idx].rho
        #     end
        #     push!(ph_ρ_i, ρ)
        # end
        # ph_ρ[indices_phaselist, i] .= ph_ρ_i

        # add ph_comp (only ss phases considered)
        # update indices_phaselist to only consider the solid solutions (ss)
        indices_phaselist = indices_phaselist[is_ss]
        # substract 7 for the 7 pure phases in the PHASE_LIST
        indices_phaselist .-= 7
        # adjust indices to ranges, e.g. idx 1 > 1:6, idx 5 > 25:30, etc.
        indices_phaselist = vcat([vcat((idx-1)*6+1:(idx-1)*6+6) for idx in indices_phaselist]...)

        ph_comp_i = reduce(vcat, [ss.Comp for ss in out[i].SS_vec])
        ss_comp[indices_phaselist,i] .= ph_comp_i
    end

    Finalize_MAGEMin(MAGEMin_db)

    # check that all oxides in oxides_in_out have the same order
    if !([all(x -> x == row[1], row) for row in eachrow(oxides_in_out)] == [true, true, true, true, true, true])
        @error "Not all out.oxides are indentical."
    end
    oxides_in_out = oxides_in_out[:, 1]

    # write a CSV
    x_names = ["p_kbar", "t_c", oxides_in_out...]
    y_names = [(phase_list .* "_mol_frac")...,
               # (phase_list .* "_rho")...,
               vcat([repeat([ss], 6) .* ("_" .* oxides_in_out)  for ss in SS]...)...,
               "bulk density", "bulk_modulus", "shear_modulus"]

    x_data = vcat(pressure_kbar', temperature_C', bulks)
    y_data = vcat(ph_mode,
                  # ph_ρ,
                  ss_comp,
                  phys_prop)

    x_data = DataFrame(x_data', Symbol.(x_names))
    y_data = DataFrame(y_data', Symbol.(y_names))

    if save_to_csv
        CSV.write(filename_base * "x.csv", x_data)
        CSV.write(filename_base * "y.csv", y_data)
    end
    return x_data, y_data
end


function generate_bulk_array(rng::Xoshiro, n::Int;
                             bulk_em_1::AbstractVector{Float64} = DSUM_wt,
                             bulk_em_2::AbstractVector{Float64} = PSUM_wt
                             )::AbstractVector{<:AbstractVector{Float64}}
    X = Vector{Vector{Float64}}()
    for _ in eachindex(1:n)
        x_em1 = rand(rng, Float64)
        x_em2 = 1 - x_em1
        x = x_em1 .* bulk_em_1 .+ x_em2 .* bulk_em_2

        x ./= sum(x)
        push!(X, x)
    end

    return X
end


function generate_noisy_bulk_array(rng::Xoshiro, n::Int;
                                   bulk_em_1::AbstractVector{Float64} = DSUM_wt,
                                   bulk_em_2::AbstractVector{Float64} = PSUM_wt,
                                   λ        ::Real                    = 100
                                   )::AbstractVector{<:AbstractVector{Float64}}
    X = Vector{Vector{Float64}}()
    for _ in eachindex(1:n)
        x_em1 = rand(rng, Float64)
        x_em2 = 1 - x_em1
        x = x_em1 .* bulk_em_1 .+ x_em2 .* bulk_em_2

        x ./= sum(x)

        dirichlet_x = Dirichlet(x .* λ)
        x_noisy = rand(dirichlet_x)
        push!(X, x_noisy)
    end

    return X
end
