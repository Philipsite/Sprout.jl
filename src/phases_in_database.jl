
#=====================================================================
(1) Functions to extract phase information from MAGEMin databases
    and write to TOML files.
=====================================================================#

"""
Extracts relevant information from a thermodynamic database in MAGEMin, including:
- Database name and metadata
- List of oxides
- Pure phases and their compositions
- Solution phases, their endmembers, site fractions, and compositions
"""
function extract_db_info(dtb::String)::Dict
    db_info = retrieve_solution_phase_information(dtb)

    # some properties must be retrieved a bit hacky... by accessing the MAGEMin C struct directly...
    gv, z_b, DB, splx_data = init_MAGEMin(dtb)
    gv = use_predefined_bulk_rock(gv, 0, dtb)
    gv, z_b, DB, splx_data = pwm_init(10.0, 800.0, gv, z_b, DB, splx_data)

    x_oxides = unsafe_string.(unsafe_wrap(Vector{Ptr{Int8}}, gv.ox, gv.len_ox))

    # Get pp_flags array (this is used to filter buffers/activities from the pure phase reference database)
    pp_flags_ptr = unsafe_wrap(Vector{Ptr{Cint}}, gv.pp_flags, gv.len_pp)
    pp_realphase_flags = [unsafe_wrap(Vector{Cint}, pp_flags_ptr[i], gv.n_flags)[5] for i in 1:gv.len_pp]

    pp_names = db_info.data_pp[pp_realphase_flags .== 0]
    ss_names = [db_info.data_ss[i].ss_name for i in eachindex(db_info.data_ss)]
    ss_sf_names = [db_info.data_ss[i].ss_sf[2:end] for i in eachindex(db_info.data_ss)]


    # Access the pure phase reference database
    PP_ref_db = unsafe_wrap(Vector{LibMAGEMin.PP_ref}, DB.PP_ref_db, gv.len_pp)
    pp_comp = Dict{String, Vector{Float64}}()
    pp_idx = 0
    for i in 1:gv.len_pp
        if pp_realphase_flags[i] == 0
            pp_idx += 1
            # the Comp vector is always of length 15 (max number of oxides).
            # Only the oxides in the database are relevant.
            pp_comp[pp_names[pp_idx]] = collect(PP_ref_db[i].Comp[1:gv.len_ox])
        end
    end

    # Acess the solution phase reference database
    SS_ref_db = unsafe_wrap(Vector{LibMAGEMin.SS_ref}, DB.SS_ref_db, gv.len_ss)
    ss_data = Dict{String, Any}()
    for i in eachindex(ss_names)
        ss_name = ss_names[i]
        n_em = SS_ref_db[i].n_em

        em_names = unsafe_string.(unsafe_wrap(Vector{Ptr{Int8}}, SS_ref_db[i].EM_list, n_em))
        em_comp_vec_pointer = unsafe_wrap(Vector{Ptr{Cdouble}}, SS_ref_db[i].Comp, n_em)

        em_comp = Dict{String, Vector{Float64}}()
        for j in 1:n_em
            em_comp[em_names[j]] = collect(unsafe_wrap(Vector{Cdouble}, em_comp_vec_pointer[j], gv.len_ox))
        end

        ss_data[ss_name] = Dict(
            "endmembers" => em_names,  # ordered list
            "site_fractions" => ss_sf_names[i],
            "composition" => em_comp
        )
    end

    finalize_MAGEMin(gv, DB, z_b)

    # build a dict with the relevant information for each phase, to be written to TOML
    dtb_summary = Dict{String, Any}(
        "database" => db_info.db_info,
        "name" => dtb,
        "oxides" => x_oxides,
        "pure_phases" => Dict(
            "names" => pp_names,
            "composition" => pp_comp
        ),
        "solution_phases" => Dict(
            "names" => ss_names,
            "data" => ss_data
        )
    )

    return dtb_summary
end

"""
Extracts relevant information from a thermodynamic database in MAGEMin, including:
- Database name and metadata
- List of oxides
- Pure phases and their compositions
- Solution phases, their endmembers, site fractions, and compositions
"""
function extract_db_info_to_TOML(dtb::String; dir::String = joinpath(pwd(), "dtb_summaries"))
    dtb_summary = extract_db_info(dtb)
    filename = joinpath(dir, dtb * "_summary.toml")

    # write to TOML
    open(filename, "w") do io
        TOML.print(io, dtb_summary)
    end
end


#=====================================================================
(2) Functions to read phase information from TOML files generated
    for different MAGEMin databases and compute variables used for
    surrogate modelling.
=====================================================================#

"""
Information about thermodynamic database
- `oxides` : Vector of oxides in the database
- `n_oxides` : Number of oxides
- `pp_names` : Vector of pure phase names
- `ss_names` : Vector of solution phase names
- `ss_em_names` : Vecotr of vectors of endmember names for each solution phase
- `ss_sf_names` : Vector of vectors of site fraction names for each solution phase
- `n_pp` : Number of pure phases
- `n_ss` : Number of solution phases
- `n_sf` : Number of site fractions (overall)
- `pp_comp` : Matrix of pure phase compositions (n_oxides x n_pp)
- `var_mask_components_in_ss` : Matrix indicating which components are variable in each solution phase (n_oxides x n_ss)
- `fixed_components_in_ss` : Matrix of fixed component compositions in each solution phase (n_oxides x n_ss)
- `idx_variable_oxide_components_in_ss_flat` : Vector of indices of variable components in solution phases, flattened
"""
struct DatabaseInfo
    db_MAGEMin                  ::String
    oxides                      ::Vector{String}
    n_oxides                    ::Int
    pp_names                    ::Vector{String}
    ss_names                    ::Vector{String}
    ss_em_names                 ::Vector{Vector{String}}
    ss_sf_names                 ::Vector{Vector{String}}
    n_pp                        ::Int
    n_ss                        ::Int
    n_sf                        ::Int
    pp_comp                     ::Matrix{Float32}
    var_mask_components_in_ss   ::Matrix{Float32}
    fixed_components_in_ss      ::Matrix{Float32}
    idx_variable_oxide_components_in_ss_flat ::Vector{Int}
end

"""
Retrieves relevant information from a thermodynamic database summary stored in a TOML file.
"""
function load_db_info(toml_path::String)::DatabaseInfo
    data        = TOML.parsefile(toml_path)

    db          = data["name"]
    oxides      = data["oxides"]
    n_oxides    = length(oxides)

    pp_names    = data["pure_phases"]["names"]
    ss_names    = data["solution_phases"]["names"]
    ss_em_names = [data["solution_phases"]["data"][ss]["endmembers"] for ss in ss_names]
    ss_sf_names = [data["solution_phases"]["data"][ss]["site_fractions"] for ss in ss_names]

    n_pp        = length(pp_names)
    n_ss        = length(ss_names)
    n_sf        = sum(length.(ss_sf_names))

    pp_comp     = reduce(hcat, [Float32.(data["pure_phases"]["composition"][pp]) for pp in pp_names])
    pp_comp     = pp_comp ./ sum(pp_comp, dims=1)

    # Compute variable component mask and fixed component composition in solution phases
    var_mask_components_in_ss = Matrix{Float32}(undef, n_oxides, n_ss)
    fixed_components_in_ss    = Matrix{Float32}(undef, n_oxides, n_ss)
    for (i, ss) in enumerate(ss_names)
        ss_data = data["solution_phases"]["data"][ss]
        em_names = ss_data["endmembers"]
        var_mask, fixed_comp = compute_component_variability(ss_data, em_names, n_oxides)
        var_mask_components_in_ss[:, i] = var_mask
        fixed_components_in_ss[:, i]    = fixed_comp
    end

    idx_variable_oxide_components_in_ss_flat = findall(vec(var_mask_components_in_ss) .== 1.0)

    db_info = DatabaseInfo(
        db,
        oxides,
        n_oxides,
        pp_names,
        ss_names,
        ss_em_names,
        ss_sf_names,
        n_pp,
        n_ss,
        n_sf,
        pp_comp,
        var_mask_components_in_ss,
        fixed_components_in_ss,
        idx_variable_oxide_components_in_ss_flat
    )

    return db_info
end

"""
Helper function to extract relevant information from a thermodynamic database in MAGEMin;
used internally by `load_db_info` to compute `var_mask_components_in_ss` and `fixed_components_in_ss`.
"""
function compute_component_variability(ss_data, em_names, n_oxides)
    em_comp_mat = reduce(hcat, [ss_data["composition"][em] for em in em_names])
    em_comp_mat = em_comp_mat ./ sum(em_comp_mat; dims=1)
    em_comp_mat = replace!(em_comp_mat, NaN => 0.0)

    var_mask = [length(unique(em_comp_mat[j, :])) > 1 for j in 1:n_oxides]
    fixed_comp = Float32.([length(unique(em_comp_mat[j, :])) == 1 ?
                           unique(em_comp_mat[j, :])[1] : 0.0 for j in 1:n_oxides])
    return var_mask, fixed_comp
end


"""
Update `DatabaseInfo` to consider phases resulting from solid solutions with solvi, e.g., 'mu' and 'pat'.
Solvus names should be provided in the `DB_config.toml` and follow the convention used by MAGEMin.
"""
function update_solvus_phases_db_info(db_info::DatabaseInfo, db_config::Dict{String, Any})
    solvus_names = db_config["solvus_names"]

    # (1) update ss_names and n_ss
    ss_names_updated = reduce(vcat, [haskey(solvus_names, ss) ? solvus_names[ss] : [ss] for ss in db_info.ss_names])
    n_ss_updated = length(ss_names_updated)

    # (2) update ss_em_names and ss_sf_names
    ss_em_names_updated = reduce(vcat, [haskey(solvus_names, ss) ? repeat([db_info.ss_em_names[i]], length(solvus_names[ss])) : [db_info.ss_em_names[i]] for (i, ss) in enumerate(db_info.ss_names)])
    ss_sf_names_updated = reduce(vcat, [haskey(solvus_names, ss) ? repeat([db_info.ss_sf_names[i]], length(solvus_names[ss])) : [db_info.ss_sf_names[i]] for (i, ss) in enumerate(db_info.ss_names)])
    n_sf_updated = sum(length.(ss_sf_names_updated))

    # (3) update var_mask_components_in_ss and fixed_components_in_ss
    # Duplicate columns for phases with solvi
    var_mask_updated = Matrix{Float32}(undef, db_info.n_oxides, n_ss_updated)
    fixed_comp_updated = Matrix{Float32}(undef, db_info.n_oxides, n_ss_updated)

    col_idx = 1
    for (i, ss) in enumerate(db_info.ss_names)
        if haskey(solvus_names, ss)
            # Duplicate the column for each phase in the solvus
            n_solvus_phases = length(solvus_names[ss])
            for j in 1:n_solvus_phases
                var_mask_updated[:, col_idx] = db_info.var_mask_components_in_ss[:, i]
                fixed_comp_updated[:, col_idx] = db_info.fixed_components_in_ss[:, i]
                col_idx += 1
            end
        else
            # Keep the column as is
            var_mask_updated[:, col_idx] = db_info.var_mask_components_in_ss[:, i]
            fixed_comp_updated[:, col_idx] = db_info.fixed_components_in_ss[:, i]
            col_idx += 1
        end
    end

    # (4) update idx_variable_oxide_components_in_ss_flat
    idx_variable_oxide_components_in_ss_flat_updated = findall(vec(var_mask_updated) .== 1.0)

    return DatabaseInfo(
        db_info.db_MAGEMin,
        db_info.oxides,
        db_info.n_oxides,
        db_info.pp_names,
        ss_names_updated,     # updated field
        ss_em_names_updated,  # updated field
        ss_sf_names_updated,  # updated field
        db_info.n_pp,
        n_ss_updated,         # updated field
        n_sf_updated,         # updated field
        db_info.pp_comp,
        var_mask_updated,     # updated field
        fixed_comp_updated,   # updated field
        idx_variable_oxide_components_in_ss_flat_updated  # updated field
    )
end


"""
Filter the pure phase composition, the variable component mask and the fixed component composition in solution phases
to only exclude pure and solution phases from a database not considered in the surrogate model training.
"""
function filter_db_info(
    db_info::DatabaseInfo,
    pp_not_considered::Vector{String},
    ss_not_considered::Vector{String}
)
    idx_pp_considered = findall(x -> x ∉ pp_not_considered, db_info.pp_names)
    idx_ss_considered = findall(x -> x ∉ ss_not_considered, db_info.ss_names)
    pp_comp                   = db_info.pp_comp[:, idx_pp_considered]
    var_mask_components_in_ss = db_info.var_mask_components_in_ss[:, idx_ss_considered]
    fixed_components_in_ss    = db_info.fixed_components_in_ss[:, idx_ss_considered]

return pp_comp, var_mask_components_in_ss, fixed_components_in_ss
end
