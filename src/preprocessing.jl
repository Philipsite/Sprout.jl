# Canonical ordering of all extractable data groups.
# Three things must stay in sync whenever a new variable is added:
#   (1) This constant — defines the order
#   (2) `grouped_names` in `get_col_names` — maps each key to its column name vector
#   (3) `data_dict` in `extract_data` — maps each key to its data matrix
const EXTRACT_DATA_KEYS = [
    "P_Pa", "T_C", "bulk",
    "W", "∆G°",
    "G_sys_Jmol⁻¹", "H_sys_Jmol⁻¹", "S_sys_JK⁻¹mol⁻¹",
    "V_sys_m³mol⁻¹", "ρ_sys_kgm⁻³",
    "Cp_sys_JK⁻¹mol⁻¹", "Cv_sys_JK⁻¹mol⁻¹", "α_sys_K⁻¹", "K_sys_Pa",
    "shearMod_sys_Pa", "Vp_kms⁻¹", "Vs_kms⁻¹",
    "μ_oxides_Jmol⁻¹", "molar_fraction",
    "G_phases_Jmol⁻¹", "H_phases_Jmol⁻¹", "S_phases_JK⁻¹mol⁻¹",
    "V_phases_m³mol⁻¹", "ρ_phases_kgm⁻³",
    "Cp_phases_JK⁻¹mol⁻¹", "Cv_phases_JK⁻¹mol⁻¹", "α_phases_K⁻¹", "K_phases_Pa",
    "SS_compositions", "SS_emfrac", "SS_μem", "SS_site_fractions"
]

"""
    get_col_names(db_info, names; modified_phases, W_binary_names, ∆G°_names)

Return a `Vector{String}` of DataFrame column names for the given selection of grouped keys.
If `names = nothing`, all columns are returned.

# Available keys for `names`
**P–T**
- `"P_Pa"` — pressure [Pa]
- `"T_C"` — temperature [°C]

**Bulk composition**
- `"bulk"` — bulk oxide fractions [mol/mol], one column per oxide in `db_info.oxides`

**Modified thermodynamic parameters** (requires `modified_phases` + corresponding names args)
- `"W"` — interaction parameters W [H, S, V] for each binary in each modified phase
- `"∆G°"` — end-member Gibbs energy corrections for each modified phase

**Bulk system scalar properties**
- `"G_sys_Jmol⁻¹"`, `"H_sys_Jmol⁻¹"`, `"S_sys_JK⁻¹mol⁻¹"`, `"V_sys_m³mol⁻¹"`, `"ρ_sys_kgm⁻³"`
- `"Cp_sys_JK⁻¹mol⁻¹"`, `"Cv_sys_JK⁻¹mol⁻¹"`, `"α_sys_K⁻¹"`, `"K_sys_Pa"`
- `"shearMod_sys_Pa"`, `"Vp_kms⁻¹"`, `"Vs_kms⁻¹"`

**Bulk component chemical potentials**
- `"μ_oxides_Jmol⁻¹"` — one column per oxide in `db_info.oxides`

**MOlar fractions of stable phases**
- `"molar_fraction"` — molar phase fractions for all phases (pp + ss)

**Phase-wise thermodynamic properties** (one column per phase for each property)
- `"G_phases_Jmol⁻¹"`, `"H_phases_Jmol⁻¹"`, `"S_phases_JK⁻¹mol⁻¹"`, `"V_phases_m³mol⁻¹"`, `"ρ_phases_kgm⁻³"`
- `"Cp_phases_JK⁻¹mol⁻¹"`, `"Cv_phases_JK⁻¹mol⁻¹"`, `"α_phases_K⁻¹"`, `"K_phases_Pa"`

**Solid solution properties**
- `"SS_compositions"` — oxide compositions per SS phase [mol/mol]
- `"SS_emfrac"` — end-member fractions per SS phase
- `"SS_μem"` — end-member chemical potentials [J/mol] per SS phase
- `"SS_site_fractions"` — site fractions per SS phase
"""
function get_col_names(
    db_info         ::DatabaseInfo,
    names           ::Union{Vector{String}, Nothing};
    modified_phases ::Union{Vector{String}, Nothing}         = nothing,
    W_binary_names  ::Union{Vector{Vector{String}}, Nothing} = nothing,
    ∆G°_names       ::Union{Vector{Vector{String}}, Nothing} = nothing
    ) ::Vector{String}

    # Some arg checks for modified thermodynamic parameters
    if !isnothing(W_binary_names) || !isnothing(∆G°_names)
        !isnothing(modified_phases)|| throw(ArgumentError(
            "`modified_phases` must be provided when `W_binary_names`, `∆G°_names` is provided."))
    end

    oxides      = db_info.oxides
    pp_names    = db_info.pp_names
    ss_names    = db_info.ss_names
    ss_sf_names    = db_info.ss_sf_names
    ss_em_names = db_info.ss_em_names
    phases = vcat(pp_names, ss_names)

    # (1) Create all possible variable names
    # P–T
    P_Pa_names = ["P_Pa"]
    T_C_names  = ["T_C"]

    # bulk composition
    bulk_names = "bulk_" .* oxides

    # modified thermodynamic parameters (empty if not provided)
    W_col_names   = isnothing(W_binary_names) ? String[] :
                    vcat([["W_" * ph * "_" * nm * "_" * c
                           for nm in W_binary_names[j] for c in ["H", "S", "V"]]
                          for (j, ph) in enumerate(modified_phases)]...)
    ∆G°_col_names = isnothing(∆G°_names) ? String[] :
                    vcat([ph .* "_∆G°_" .* ∆G°_names[j]
                          for (j, ph) in enumerate(modified_phases)]...)

    # bulk system scalar properties
    sys_scalar_names = [
        "G_sys_Jmol⁻¹", "H_sys_Jmol⁻¹", "S_sys_JK⁻¹mol⁻¹",
        "V_sys_m³mol⁻¹", "ρ_sys_kgm⁻³",
        "Cp_sys_JK⁻¹mol⁻¹", "Cv_sys_JK⁻¹mol⁻¹", "α_sys_K⁻¹", "K_sys_Pa",
        "shearMod_sys_Pa", "Vp_kms⁻¹", "Vs_kms⁻¹"
    ]

    # bulk chemical potentials (one per oxide, db_info.oxides order)
    μ_ox_names = "μ_" .* oxides .* "_Jmol⁻¹"

    # phase modes
    molar_fraction_names = "molar_fraction_" .* phases

    # phase-wise thermodynamic properties (property-first: n_phases contiguous columns per property)
    G_ph_names  = "G_"  .* phases .* "_Jmol⁻¹"
    H_ph_names  = "H_"  .* phases .* "_Jmol⁻¹"
    S_ph_names  = "S_"  .* phases .* "_JK⁻¹mol⁻¹"
    V_ph_names  = "V_"  .* phases .* "_m³mol⁻¹"
    ρ_ph_names  = "ρ_"  .* phases .* "_kgm⁻³"
    Cp_ph_names = "Cp_" .* phases .* "_JK⁻¹mol⁻¹"
    Cv_ph_names = "Cv_" .* phases .* "_JK⁻¹mol⁻¹"
    α_ph_names  = "α_"  .* phases .* "_K⁻¹"
    K_ph_names  = "K_"  .* phases .* "_Pa"

    # SS compositions: block layout (idx-1)*n_oxides+1 : idx*n_oxides per SS
    ss_comp_names = vcat([(ph .* "_comp_") .* oxides for ph in ss_names]...)

    # SS end-member fractions and chemical potentials (variable n_em per SS)
    ss_emfrac_names = vcat([ph .* "_emfrac_" .* ss_em_names[i]              for (i, ph) in enumerate(ss_names)]...)
    ss_μem_names    = vcat([ph .* "_μem_"    .* ss_em_names[i] .* "_Jmol⁻¹" for (i, ph) in enumerate(ss_names)]...)

    # SS site fractions (variable n_sf per SS)
    ss_sf_names_col = vcat([ph .* "_sf_"     .* ss_sf_names[i]              for (i, ph) in enumerate(ss_names)]...)

    # (2) Map each key in EXTRACT_DATA_KEYS to its column name vector.
    # These keys match all "independent" subsets of variables that can be extracted.
    # E.g., if site fractions are extracted this must be done for all available solid solutions
    # when names=nothing (return all columns in a consistent order).
    grouped_names = [
        P_Pa_names, T_C_names, bulk_names, W_col_names, ∆G°_col_names,
        [sys_scalar_names[1]], [sys_scalar_names[2]], [sys_scalar_names[3]],
        [sys_scalar_names[4]], [sys_scalar_names[5]],
        [sys_scalar_names[6]], [sys_scalar_names[7]], [sys_scalar_names[8]], [sys_scalar_names[9]],
        [sys_scalar_names[10]], [sys_scalar_names[11]], [sys_scalar_names[12]],
        μ_ox_names, molar_fraction_names,
        G_ph_names, H_ph_names, S_ph_names,
        V_ph_names, ρ_ph_names,
        Cp_ph_names, Cv_ph_names, α_ph_names, K_ph_names,
        ss_comp_names, ss_emfrac_names, ss_μem_names, ss_sf_names_col
    ]

    name_dict = Dict(zip(EXTRACT_DATA_KEYS, grouped_names))

    if isnothing(names)
        return vcat([name_dict[key] for key in EXTRACT_DATA_KEYS]...)
    else
        return vcat([name_dict[key] for key in names]...)
    end
end

function load_dataset(data::DataFrame, x_cols, y_cols)

    return x::DataFrame, y::DataFrame
end


"""
Returns indices of data points that do not contain NaN values.
"""
function filter_NaN(data::Matrix)
    return [!any(isnan, data[:, j]) for j in 1:size(data, 2)]
end


"""
Used within preprocess-functions.
Set indices for the filtering.
"""
#//TODO - Adapt this function to work with the phases not considered from the database summary TOML file / config TOML file
# Further, this function should in the future return the indices to extract phase modes, solid solution compositions and site fractions
function indices_of_stable_phases()
    # set up indices of stable phases > to extract phase fractions
    n_phases = length(PP) + length(SS)
    idx_stable_phases = [i for i in 1:n_phases if i ∉ IDX_OF_PHASES_NEVER_STABLE]

    # setup indices of stable solid solution components > to extract solid solution compositions
    # offset by number of phases
    idx_stable_ss = 1:(length(SS)*6)
    idx_stable_ss = [i for i in idx_stable_ss if i ∉ [6 * (k-1) + j for k in Sprout.IDX_SS_NEVER_STABLE for j in 1:6]] .+ n_phases

    return idx_stable_phases, idx_stable_ss
end


"""
Takes DataFrame of Training/Validation/Test data, returns:
- x    :: Array{Float32, 3} (Vec, 1, N)        - Input features P [GPa], T [°C], bulk composition [molmol⁻¹]
- 𝑣    :: Array{Float32, 3} (Vec, 1, N)        - Phase fraction [molmol⁻¹]
- 𝐗_ss :: Array{Float32, 3} (Matrix, N)        - Solid solution phase compositions [molmol⁻¹]
— ρ    :: Array{Float32, 3} (Scalar, 1, N)     - System densities
- Κ    :: Array{Float32, 3} (Scalar, 1, N)     - Bulk moduli
- μ    :: Array{Float32, 3} (Scalar, 1, N)     - Shear moduli

Applies the following filters:
- filter observation containing NaN
- only extract phases that are predicted as part of the stable assemblage at least once in the dataset.

"""
#//TODO - Indices of what to extract should be passed, and no longer "appear" out of "thin air" within the function.
function preprocess_data(x_data::DataFrame, y_data::DataFrame)
    x = Matrix(Matrix{Float32}(x_data)')
    y = Matrix((Matrix{Float32}(y_data))')

    # filter data points with NaNs (failed minimisations? > failed volume computation!)
    cols_no_nan = filter_NaN(x) .& filter_NaN(y)

    # (1) INPUTS
    x = x[:, cols_no_nan]
    x = reshape(x, size(x, 1), 1, size(x, 2))

    # (2) OUTPUTS
    y = y[:, cols_no_nan]
    # filter the stable phases only
    idx_stable_phases, idx_stable_ss = indices_of_stable_phases()

    𝑣 = y[idx_stable_phases, :]
    𝑣 = reshape(𝑣, size(𝑣, 1), 1, size(𝑣, 2))
    vec_ss = y[idx_stable_ss, :]
    𝐗_ss = reshape(vec_ss, 6, Int(size(vec_ss, 1) / 6), :)
    ρ = y[end - 2, :]
    ρ = reshape(ρ, 1, 1, :)
    Κ = y[end - 1, :]
    Κ = reshape(Κ, 1, 1, :)
    μ = y[end, :]
    μ = reshape(μ, 1, 1, :)
    return x::Array{Float32, 3}, 𝑣::Array{Float32,3}, 𝐗_ss::Array{Float32,3}, ρ::Array{Float32,3}, Κ::Array{Float32,3}, μ::Array{Float32,3}
end


"""
Converts phase fraction matrix to one-hot encoded phase stability matrix.
"""
function one_hot_phase_stability(𝑣::Array{Float32})
    return 𝑣 .!= 0.0
end
