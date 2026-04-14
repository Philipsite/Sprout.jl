

"""
Calculate W_G values from W, P and T as (WG = WH - T*WS + P*WV),
where P is in kbar and T in K.
"""
function calculate_w_g(W::Matrix{Float64}, pressure_kbar::Float64, temperature_K::Float64)
    # calculate the g values from WG = WH - T*WS + P*WV, using P[kbar] and T[K] to match MAGEMin's calculation
    return W[:,1] .- temperature_K .* W[:,2] .+ pressure_kbar .* W[:,3]
end

"""
A custom version of 'multi_point_minimization()' that allows modification
of the thermodynamic parameters (gbase and W) for specified phases.
"""
function mpm_custom(pressure_kbar   ::T,
                    temperature_C   ::T,
                    MAGEMin_db      ::MAGEMin_Data,
                    X               ::Vector{T},
                    Xoxides         ::Vector{String},
                    sys_in          ::String;
                    mod_phases      ::Union{Vector{String}, Nothing}                    = nothing,
                    W               ::Union{Vector{<:Vector{<:Matrix{<:AbstractFloat}}}, Nothing}  = nothing,
                    ∆G°             ::Union{Vector{<:Vector{<:Vector{<:AbstractFloat}}}, Nothing}  = nothing,
                    name_solvus     ::Bool                                                    = false,
                    progressbar     ::Bool                                                    = true
                    ) where {T  <: Vector{<:AbstractFloat}}

    outs = Vector{MAGEMin_C.gmin_struct{Float64, Int64}}(undef, length(pressure_kbar))
    # main loop
    if progressbar
        progr = Progress(length(pressure_kbar), desc="Computing $(length(pressure_kbar)) points...") # progress meter
    end

    @threads :static for i in eachindex(pressure_kbar)
        id          = Threads.threadid()
        gv          = MAGEMin_db.gv[id]
        z_b         = MAGEMin_db.z_b[id]
        DB          = MAGEMin_db.DB[id]
        splx_data   = MAGEMin_db.splx_data[id]

        gv = define_bulk_rock(gv, X[i], Xoxides, sys_in, MAGEMin_db.db)
        gv, z_b, DB, splx_data = pwm_init(pressure_kbar[i], temperature_C[i], gv, z_b, DB, splx_data)

        ss_names  = unsafe_string.(unsafe_wrap(Vector{Ptr{Int8}}, gv.SS_list, gv.len_ss))
        ss_struct = unsafe_wrap(Vector{LibMAGEMin.SS_ref},DB.SS_ref_db,gv.len_ss)

        # modify the gbase and W values for each phase in mod_phases
        if !isnothing(mod_phases)
            for j in eachindex(mod_phases)
                phase = mod_phases[j]
                ss_idx = findfirst(x->x==phase, ss_names)

                ss_gbase = unsafe_wrap(Vector{Float64}, ss_struct[ss_idx].gbase, ss_struct[ss_idx].n_em)
                ss_gbase_mod = copy(ss_gbase)
                if !isnothing(∆G°)
                    ss_gbase_mod += ∆G°[i][j]
                end
                if !isnothing(W)
                    w_g = calculate_w_g(W[i][j], pressure_kbar[i], temperature_C[i] + 273.15)
                end
                unsafe_copyto!(ss_struct[ss_idx].W, pointer(w_g), ss_struct[ss_idx].n_w)
                unsafe_copyto!(ss_struct[ss_idx].gbase, pointer(ss_gbase_mod), ss_struct[ss_idx].n_em)
            end
        else
            @info "mpm_custom called without modified parameters. Use multi_point_minimization instead!"
        end

        out = pwm_run(gv, z_b, DB, splx_data, name_solvus=name_solvus)
        outs[i]   = deepcopy(out)

        if progressbar
            next!(progr)
        end
    end

    if progressbar
        finish!(progr)
    end

    return outs
end
function mpm_custom(pressure_kbar   ::T,
                    temperature_C   ::T,
                    MAGEMin_db      ::MAGEMin_Data,
                    X               ::Vector{T},
                    Xoxides         ::Vector{String},
                    sys_in          ::String;
                    mod_phases      ::Union{Vector{String}, Nothing}                = nothing,
                    W               ::Union{Vector{<:Matrix{<:AbstractFloat}}, Nothing}  = nothing,
                    ∆G°             ::Union{Vector{<:Vector{<:AbstractFloat}}, Nothing}  = nothing,
                    name_solvus     ::Bool                                             = false,
                    progressbar     ::Bool                                             = true
                    ) where {T  <: AbstractFloat}

    pressure_kbar = [pressure_kbar]
    temperature_C = [temperature_C]
    X = [X]
    if !isnothing(W)
        W = [W]
    end
    if !isnothing(∆G°)
        ∆G° = [∆G°]
    end

    return mpm_custom(pressure_kbar, temperature_C, MAGEMin_db, X, Xoxides, sys_in; mod_phases=mod_phases, W=W, ∆G°=∆G°, name_solvus=name_solvus, progressbar=progressbar)[1]
end


function generate_data(
        n                     ::Int,
        db_info               ::DatabaseInfo,
        pressure_range_kbar   ::Tuple,
        temperature_range_C   ::Tuple,
        X_bulk                ::AbstractVector{<:AbstractVector{Float64}},
        X_oxides              ::Vector{String},
        sys_in                ::String;
        name_solvus           ::Bool                                                          = false,
        modified_phases       ::Union{Vector{String}, Nothing}                               = nothing,
        W                     ::Union{Vector{<:Vector{<:Matrix{<:AbstractFloat}}}, Nothing}  = nothing,
        ∆G°                   ::Union{Vector{<:Vector{<:Vector{<:AbstractFloat}}}, Nothing}  = nothing,
        seed                  ::Int                                                          = 42
    ) ::AbstractArray{<:MAGEMin_C.out_struct}

    db          = db_info.db_MAGEMin
    oxides      = db_info.oxides

    #NOTE - this is commented, need an idea of how to deal with Fe2O3 / O that can be both present.
    # @assert Set(oxides) == Set(X_oxides) "Oxides in db_info and X_bulk do not match."

    # generate P-T
    rng = Xoshiro(seed)
    pressure_kbar = rand(rng, Uniform(pressure_range_kbar[1], pressure_range_kbar[2]), n)
    temperature_C = rand(rng, Uniform(temperature_range_C[1], temperature_range_C[2]), n)

    # init MAGEMin
    MAGEMin_db = Initialize_MAGEMin(db, solver=0, verbose=false)

    if !isnothing(modified_phases)
        @info "Generating data with altered thermodynamic properties for phases: $(modified_phases)."
        @assert !isnothing(W) || !isnothing(∆G°) "If modified_phases is provided, W or ∆G° must also be provided."
        out = mpm_custom(
            pressure_kbar,
            temperature_C,
            MAGEMin_db,
            X_bulk,
            X_oxides,
            sys_in;
            name_solvus=name_solvus,
            mod_phases=modified_phases,
            W=W,
            ∆G°=∆G°)
    else
        out = multi_point_minimization(
            pressure_kbar,
            temperature_C,
            MAGEMin_db,
            X=X_bulk,
            Xoxides=X_oxides,
            sys_in=sys_in,
            name_solvus=name_solvus)
    end

    # filter out for successful minimizations
    out = filter(o -> o.status == 0, out)
    while length(out) < n
        @warn "Only $(length(out)) successful minimizations out of $n. Regenerate to reach target of $n minimizations."

        p_i = rand(rng, Uniform(pressure_range_kbar[1], pressure_range_kbar[2]), n - length(out))
        t_i = rand(rng, Uniform(temperature_range_C[1], temperature_range_C[2]), n - length(out))
        X_i = [X_bulk[rand(rng, 1:length(X_bulk))] for _ in 1:(n - length(out))]

        if !isnothing(modified_phases)
            @info "Generating data with altered thermodynamic properties for phases: $(modified_phases)."
            W_i = [W[rand(rng, 1:length(W))] for _ in 1:(n - length(out))]
            ∆G°_i = [∆G°[rand(rng, 1:length(∆G°))] for _ in 1:(n - length(out))]
            out_i = mpm_custom(
                p_i,
                t_i,
                MAGEMin_db,
                X_i,
                X_oxides,
                sys_in;
                name_solvus=name_solvus,
                mod_phases=modified_phases,
                W=W_i,
                ∆G°=∆G°_i)
        else
            out_i = multi_point_minimization(
                p_i,
                t_i,
                MAGEMin_db;
                X=X_i,
                Xoxides=X_oxides,
                sys_in=sys_in,
                name_solvus=name_solvus)
        end

        out_i = filter(o -> o.status == 0, out_i)
        out = vcat(out, out_i)
    end

    Finalize_MAGEMin(MAGEMin_db)

    return out
end


function extract_data(
        outs                  ::AbstractArray{<:MAGEMin_C.out_struct},
        db_info               ::DatabaseInfo;
        bulk_params           ::Vector{<:Symbol} = [:rho, :bulkMod, :shearMod]
    ) ::Tuple{DataFrame, DataFrame}

    n = length(outs)

    db          = db_info.db_MAGEMin
    oxides      = db_info.oxides
    n_oxides    = db_info.n_oxides
    pp_names    = db_info.pp_names
    ss_names    = db_info.ss_names
    sf_names    = db_info.ss_sf_names

    # calculate the start_idx for the sf vector for each solid solution
    start_idx_sf = cumsum(length.(db_info.ss_sf_names[1:end-1])) .+ 1
    start_idx_sf = vcat(1, start_idx_sf)

    phases = vcat(pp_names, ss_names)
    n_phases = db_info.n_pp + db_info.n_ss
    n_ss = db_info.n_ss
    n_sf = db_info.n_sf


    # check that oxides in all out structs are identical and match the oxides in db_info
    @assert all([outs[i].oxides == outs[1].oxides for i in eachindex(outs)]) "Not all out.oxides are identical."
    @assert Set(oxides) == Set(outs[1].oxides) "Oxides in db_info and X_bulk do not match. \n Oxides in db_info: $(oxides) \n Oxides in out: $(outs[1].oxides)"

    # pre-allocate X arrays
    pressure_kbar = zeros(n)
    temperature_C = zeros(n)
    bulks = zeros(length(oxides), n)

    # pre-allocate Y arrays
    ph_modes_mol   = zeros(n_phases, n)
    ss_comps_mol   = zeros(n_oxides * n_ss, n)
    ss_sf          = zeros(n_sf, n)

    phys_props     = zeros(length(bulk_params), n)

    @threads for i in ProgressBar(eachindex(outs))
        out_i = outs[i]

        pressure_kbar[i] = out_i.P_kbar
        temperature_C[i] = out_i.T_C
        bulks[:, i] = out_i.bulk

        # extract indices of predicted phases in the phase list (from db_info)
        ph_i = out_i.ph

        indices_in_phases = [findfirst(.==(p), phases) for p in ph_i]
        indices_in_ss = [findfirst(.==(s), ss_names) for s in ph_i if s in ss_names]

        # add ph_mode
        ph_modes_mol[indices_in_phases, i] .= out_i.ph_frac
        # add ss_comp
        indices_ss_in_ss_comps_mol = vcat([vcat((idx-1)*n_oxides+1:(idx-1)*n_oxides+n_oxides) for idx in indices_in_ss]...)
        ss_comps_mol[indices_ss_in_ss_comps_mol, i] .= reduce(vcat, [ss.Comp for ss in out_i.SS_vec])
        # add ss_sf
        indices_sf_in_sf = vcat([start_idx_sf[idx]:(start_idx_sf[idx + 1] - 1) for idx in indices_in_ss]...)
        ss_sf[indices_sf_in_sf, i] .= reduce(vcat, [ss.siteFractions for ss in out_i.SS_vec])

        # TODO - add phys_prop

    end

    # create DataFrames to write to CSV
    x_names = ["P_kbar", "T_C", oxides...]
    y_names = [(phases .* "_mol_frac")...,
               vcat([repeat([ss], n_oxides) .* ("_" .* oxides)  for ss in ss_names]...)...,
               vcat([repeat([ss], length(sf_names[idx])) .* ("_" .* sf_names[idx]) for (idx, ss) in enumerate(ss_names)]...)...,
               String.(bulk_params)...]

    x_data = vcat(pressure_kbar', temperature_C', bulks)
    y_data = vcat(ph_modes_mol,
                  ss_comps_mol,
                  ss_sf,
                  phys_props)

    x_data = DataFrame(x_data', Symbol.(x_names))
    y_data = DataFrame(y_data', Symbol.(y_names))
    return x_data, y_data
end


function write_to_csv(
        data                  ::Tuple{DataFrame, DataFrame},
        filename              ::AbstractString
    )
    x_data, y_data = data
    CSV.write(filename * "_x.csv", x_data)
    CSV.write(filename * "_y.csv", y_data)
end

#======================================================================
# LEGACY CODE: generate_dataset()
=======================================================================#
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

    @warn "The function `generate_dataset` is deprecated and will be removed in future versions."

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



# =====================================================================
# (1) Generate bulks for Metapelites
# =====================================================================
const MOLAR_MASS = Dict(
    "SiO2" => 60.083,
    "TiO2" => 79.865,
    "Al2O3" => 101.961,
    "Cr2O3" => 151.989,
    "Fe2O3" => 159.6874,
    "NiO" => 74.692,
    "FeO" => 71.8442,
    "MnO" => 70.937,
    "MgO" => 40.304,
    "CaO" => 56.0774,
    "Na2O" => 61.979,
    "K2O" => 94.195,
    "P2O5" => 141.9445,
    "F" => 18.998,
    "Cl" => 35.45,
    "H2O" => 18.015,
    "O" => 15.999
)


"""
Assign seperate FeO and Fe2O3 (in wt%) to analyses with only FeO_total measurement.
Sample the XF3+ form the μ±σ of bulk XFe3+ after Forshaw and Pattison (2021).
"""
function assign_missing_Fe2Fe3!(df_wt::DataFrame, X_Fe3::Float64, σ_XFe3::Float64; molar_mass_dict::Dict = MOLAR_MASS)::DataFrame
    n_noFe3 = count(ismissing, df_wt[!, "Fe2O3"])

    XFe3_dist = truncated(Normal(X_Fe3, σ_XFe3), 0., 1.)
    XFe3 = rand(XFe3_dist, n_noFe3)

    FeO_wt = df_wt[ismissing.(df_wt[!, "Fe2O3"]), "FeO"] .* (1 .- XFe3)
    Fe2O3_wt = df_wt[ismissing.(df_wt[!, "Fe2O3"]), "FeO"] .* XFe3 .* (molar_mass_dict["Fe2O3"] / (2 * molar_mass_dict["FeO"]))

    df_wt[ismissing.(df_wt[!, "Fe2O3"]), "FeO"] = FeO_wt
    df_wt[ismissing.(df_wt[!, "Fe2O3"]), "Fe2O3"] = Fe2O3_wt

    return df_wt
end

"""
Convert wt to mol.
"""
function wt_to_mol(df_wt::DataFrame; molar_mass_dict::Dict = MOLAR_MASS)::DataFrame
    molar_mass = [molar_mass_dict[oxide] for oxide in names(df_wt)]
    df_mol = df_wt ./ molar_mass'

    m = coalesce.(Matrix(df_mol), 0.0)
    return DataFrame(m ./ sum(m, dims=2), names(df_mol))
end

"""
Reduce the bulk CaO and get exclude P2O5 by projecting from Apatite.
"""
function project_from_Apatite(df_mol::DataFrame; min_CaO = eps(Float32))::DataFrame
    df_mol[!, "CaO"] .= clamp.(df_mol[!, "CaO"] .- 10/3 .* df_mol[!, "P2O5"], min_CaO, Inf)
    # drop P2O5 col
    df_mol = select(df_mol, Not("P2O5"))
    df_mol = DataFrame(Matrix(df_mol) ./ sum(Matrix(df_mol), dims=2), names(df_mol))
end

"""
Pre-process the FPWMP22 dataset by:
- Assigning FeO and Fe2O3 based on the bulk XFe3+ distribution after Forshaw and Pattison (2021).
- Converting wt% to mol% and normalizing.
- Projecting from Apatite to reduce CaO and exclude P2O5 (renormalizing after projection).
"""
function preprocess_fpwmp22(df::DataFrame, X_Fe3::Float64, σ_XFe3::Float64; min_CaO = eps(Float32), min_val = eps(Float32), molar_mass_dict::Dict = MOLAR_MASS)::DataFrame
    df = select(df, Not("LOI", "Total"))
    df_wt = assign_missing_Fe2Fe3!(df, X_Fe3, σ_XFe3)
    df_mol = wt_to_mol(df_wt; molar_mass_dict = molar_mass_dict)
    df_mol_proj = project_from_Apatite(df_mol; min_CaO = min_CaO)

    # Replace analyses with zero values with missing
    df_mol_proj_no_zeros = mapcols(c -> replace(c, 0.0 => missing), df_mol_proj)
    # Replace analyses with values below min_val with missing
    df_mol_proj_no_zeros_or_small = mapcols(c -> map(x -> ismissing(x) ? x : (x < min_val ? missing : x), c), df_mol_proj_no_zeros)
    df_mol_proj_no_zeros_or_small = dropmissing(df_mol_proj_no_zeros_or_small)
    return df_mol_proj_no_zeros_or_small
end

"""
Sample a narrow Dirichlet distribution around a bulk vector to generate unique bulk compositions for surrogate model training.
"""
function add_ϵ_noise(
    data_mol::Vector{Float64};
    n::Int = 1,
    λ_dirichlet::Number = 1000,
    rng::Union{Random.AbstractRNG, Nothing} = nothing
    )::Matrix{Float64}

    rng = rng === nothing ? Xoshiro() : rng
    return rand(rng, Dirichlet(data_mol .* λ_dirichlet), n)
end

"""
Generate a bulk array by sampling bulks from a DataFrame of bulks (must be in mol% and normalized).
Add ϵ noise to the bulks by sampling from a narrow Dirichlet distribution around each bulk vector
to generate unique bulk compositions for surrogate model training.
"""
function generate_bulks_from_df(df::DataFrame, n::Int; λ_dirichlet::Number = 1000, seed::Union{Number, Nothing} = nothing)::Tuple{Vector{Vector{Float64}}, Vector{String}}
    data_as_mat = Matrix(Matrix(df)')

    # random shuffle the columns of the data matrix to avoid any bias in the order of the bulks
    rng = seed === nothing ? Xoshiro() : Xoshiro(seed)
    data_as_mat = data_as_mat[:, shuffle(rng, 1:size(data_as_mat, 2))]
    x_oxides = names(df)

    X = Vector{Vector{Float64}}()
    for i in 1:n
        idx_in_fpwmp22 = mod1(i, size(data_as_mat, 2))
        bulk_noisy = add_ϵ_noise(vec(data_as_mat[:, idx_in_fpwmp22]), n=1, λ_dirichlet=λ_dirichlet, rng=rng)
        push!(X, bulk_noisy[:, 1])
    end

    return (X, x_oxides)
end
