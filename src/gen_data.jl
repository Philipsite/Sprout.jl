# Define mantle composition end-member after Kerswell et al. 2024
# following "Xoxides = ["SiO2"; "CaO";"Al2O3"; "FeO"; "MgO"; "Na2O"]"
DSUM_wt = [44.1, 0.22, 0.261, 7.96, 47.4, 0.042];
PSUM_wt = [46.2, 4.34, 4.88, 8.88, 35.2, 0.33];

"""
Bulk-system properties stored by `generate_dataset`

- `V_cm3_mol`      molar volume                                    [cm³/mol]
- `Cp_J_K_mol`     isobaric heat capacity, fixed composition       [J/K/mol]
- `alpha_1_K`      thermal expansivity, fixed composition          [1/K]
- `K_S_GPa`        adiabatic bulk modulus, fixed composition       [GPa]
- `G_shear_GPa`    shear modulus, fixed composition                [GPa]
- `Cp_eq_J_K_mol`  isobaric heat capacity, equilibrium (scp=1)     [J/K/mol]
- `alpha_eq_1_K`   thermal expansivity, equilibrium (scp=1)        [1/K]

Equilibrium values include reaction / latent-heat contributions.
MAGEMin finite differences within MAGEMin with dT = 2 K;
can spike to unphysical values where the stencil crosses a phase-field boundary.
"""
const BULK_PROPS = ["V_cm3_mol", "Cp_J_K_mol", "alpha_1_K", "K_S_GPa", "G_shear_GPa",
                    "Cp_eq_J_K_mol", "alpha_eq_1_K"]

"Oxide molar masses [g/mol] as used by MAGEMin (`convertBulk4MAGEMin`)."
const OXIDE_MOLAR_MASS = Dict("SiO2"  => 60.08,  "Al2O3" => 101.96, "CaO"   => 56.08,  "MgO" => 40.30,
                              "FeO"   => 71.85,  "Fe2O3" => 159.69, "K2O"   => 94.2,   "Na2O" => 61.98,
                              "TiO2"  => 79.88,  "O"     => 16.0,   "Cr2O3" => 151.99, "MnO" => 70.937,
                              "H2O"   => 18.015, "CO2"   => 44.01,  "S"     => 32.06)

"""
    molar_mass(X, oxides)

Molar mass [g/mol] of a bulk composition `X` (mol fractions, Σ = 1) with oxide names `oxides`.
Works on a vector or column-wise on a matrix (oxides × samples).
"""
molar_mass(X::AbstractVector, oxides) = sum(X .* [OXIDE_MOLAR_MASS[ox] for ox in oxides])
molar_mass(X::AbstractMatrix, oxides) = vec(sum(X .* [OXIDE_MOLAR_MASS[ox] for ox in oxides], dims=1))

"""
    bulk_props(o) -> Vector{Float64}

Molar bulk-system properties from MAGEMin.
Order and units follow `BULK_PROPS`

"""
function bulk_props(o::MAGEMin_C.out_struct)
    phases = vcat(o.SS_vec, o.PP_vec)       # same order as o.ph, o.ph_frac, o.ph_frac_vol
    vals = [o.V,
            1e3 * sum(o.ph_frac .* [ph.cp for ph in phases]),   # Cp_fix    = Σ nᵢ cp,ᵢ  (nᵢ: molar phase fractions, cp,ᵢ in kJ/K/mol)
            sum(o.ph_frac_vol .* [ph.alpha for ph in phases]),  # alpha_fix = Σ φᵢ αᵢ    (φᵢ: volume phase fractions)
            o.bulkMod,
            o.shearMod,
            only(o.s_cp) * o.M_sys / 1e3,   # J/K/kg -> J/K/mol
            only(o.alpha)]

    return all(isfinite, vals) ? vals : fill(NaN, length(BULK_PROPS))
end


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

    # GEM -> call with scp=1 for Cp & alpha values including reaction / latent-heat.
    out = multi_point_minimization(pressure_kbar, temperature_C, MAGEMin_db,
                                   X=X_bulk, Xoxides=Xoxides, sys_in=sys_in, scp=1)

    # check minimisation
    converged = [o.status == 0 for o in out]
    n_failed  = count(!, converged)
    n_failed > 0 && @warn "$n_failed / $n minimisations did not converge (status ≠ 0); excluded from the dataset."

    out           = out[converged]
    pressure_kbar = pressure_kbar[converged]
    temperature_C = temperature_C[converged]
    n_conv        = length(out)

    bulks = reduce(hcat, [o.bulk for o in out])

    oxides_in_out = Matrix{String}(undef, 6, n_conv)
    ph_mode       = zeros(22, n_conv)
    ss_comp       = zeros(90, n_conv)
    phys_prop     = zeros(length(BULK_PROPS), n_conv)

    @threads for i in ProgressBar(eachindex(out))
        oxides_in_out[:, i] .= out[i].oxides

        ph_i = out[i].ph
        indices_phaselist = [findfirst(.==(p), phase_list) for p in ph_i]
        is_ss = Bool.(out[i].ph_type)

        # add ph_mode
        ph_mode[indices_phaselist, i] .= out[i].ph_frac

        # add ph_comp (only ss phases considered)
        # update indices_phaselist to only consider the solid solutions (ss); subtract 7 pure phases
        indices_phaselist = indices_phaselist[is_ss] .- 7
        # adjust indices to ranges, e.g. idx 1 > 1:6, idx 5 > 25:30, etc.
        indices_phaselist = vcat([vcat((idx-1)*6+1:(idx-1)*6+6) for idx in indices_phaselist]...)
        ss_comp[indices_phaselist,i] .= reduce(vcat, [ss.Comp for ss in out[i].SS_vec])

        # bulks system physical properties
        phys_prop[:, i] .= bulk_props(out[i])
    end

    Finalize_MAGEMin(MAGEMin_db)

    # check that all out.oxides have the same order
    if !all(col -> col == oxides_in_out[:, 1], eachcol(oxides_in_out))
        @error "Not all out.oxides are identical."
    end
    oxides_in_out = oxides_in_out[:, 1]

    # write a CSV
    x_names = ["p_kbar", "t_c", oxides_in_out...]
    y_names = [(phase_list .* "_mol_frac")...,
               vcat([repeat([ss], 6) .* ("_" .* oxides_in_out) for ss in SS]...)...,
               BULK_PROPS...]

    x_data = vcat(pressure_kbar', temperature_C', bulks)
    y_data = vcat(ph_mode, ss_comp, phys_prop)

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
        x_noisy = rand(rng, dirichlet_x)
        push!(X, x_noisy)
    end

    return X
end
