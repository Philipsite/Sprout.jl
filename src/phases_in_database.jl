
"""
Extracts relevant information from a thermodynamic database in MAGEMin, including:
- Database name and metadata
- List of oxides
- Pure phases and their compositions
- Solution phases, their endmembers, site fractions, and compositions
"""
function extract_db_info(dtb::String)
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
function extract_db_info_to_TOML(dtb::String; dir::String = @__DIR__)
    dtb_summary = extract_db_info(dtb)
    filename = joinpath(dir, dtb * "_summary.toml")

    # write to TOML
    open(filename, "w") do io
        TOML.print(io, dtb_summary)
    end
end
