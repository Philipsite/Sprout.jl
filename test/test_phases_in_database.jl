
@testset "phases_in_database" begin
    # Expected data from phases_sb21.jl (hardcoded benchmark)
    EXPECTED_OXIDES = ["SiO2", "CaO", "Al2O3", "FeO", "MgO", "Na2O"]
    EXPECTED_PP = ["neph", "ky", "st", "coe", "qtz", "capv", "co"]
    EXPECTED_SS = ["plg", "sp", "ol", "wa", "ri", "opx", "cpx", "hpcpx", "ak", "gtmj", "pv", "ppv", "cf", "mw", "nal"]

    EXPECTED_PP_COMP = Matrix{Float32}([
        0.5  0.5 1.0 1.0 1.0 0.5 0.0;
        0.0  0.0 0.0 0.0 0.0 0.5 0.0;
        0.25 0.5 0.0 0.0 0.0 0.0 1.0;
        0.0  0.0 0.0 0.0 0.0 0.0 0.0;
        0.0  0.0 0.0 0.0 0.0 0.0 0.0;
        0.25 0.0 0.0 0.0 0.0 0.0 0.0
    ])

    EXPECTED_PP_COMP_ADJ = Matrix{Float32}([
        0.5  0.5 1.0 1.0 1.0 0.5;
        0.0  0.0 0.0 0.0 0.0 0.5;
        0.25 0.5 0.0 0.0 0.0 0.0;
        0.0  0.0 0.0 0.0 0.0 0.0;
        0.0  0.0 0.0 0.0 0.0 0.0;
        0.25 0.0 0.0 0.0 0.0 0.0
    ])

    EXPECTED_VAR_COMPONENT_MASK = Matrix{Float32}([
        1.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 0.0 1.0;
        1.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
        1.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
        0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
        0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
        1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 1.0 1.0 1.0
    ])

    EXPECTED_VAR_COMPONENT_MASK_ADJ = Matrix{Float32}([
        1.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0 0.0 1.0;
        1.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
        1.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0;
        0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
        0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
        1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 1.0 1.0 1.0
    ])

    EXPECTED_FC_SS = FC_SS = Matrix{Float32}([
        0.0 0.0 0.33333334 0.33333334 0.33333334 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    ])

    EXPECTED_FC_SS_ADJ = Matrix{Float32}([
        0.0 0.0 0.33333334 0.33333334 0.33333334 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    ])

    mktempdir() do tmp
        # Generate TOML file
        Sprout.extract_db_info_to_TOML("sb21"; dir=tmp)
        toml_path = joinpath(tmp, "sb21_summary.toml")
        db_info = Sprout.load_db_info(toml_path)

        # Validate oxides
        @test db_info.oxides == EXPECTED_OXIDES
        @test db_info.pp_names == EXPECTED_PP
        @test db_info.ss_names == EXPECTED_SS
        @test size(db_info.pp_comp) == (6, 7)
        @test db_info.pp_comp == EXPECTED_PP_COMP
        @test size(db_info.var_mask_components_in_ss) == (6, 15)
        @test db_info.var_mask_components_in_ss == EXPECTED_VAR_COMPONENT_MASK
        @test size(db_info.fixed_components_in_ss) == (6, 15)
        @test db_info.fixed_components_in_ss == EXPECTED_FC_SS

        # filter for phases that are never stable
        pp_not_considered = ["co",]
        ss_not_considered = ["ppv",]

        pp_comp, var_mask_components_in_ss, fixed_components_in_ss = Sprout.filter_db_info(db_info, pp_not_considered, ss_not_considered)
        @test pp_comp == EXPECTED_PP_COMP_ADJ
        @test var_mask_components_in_ss == EXPECTED_VAR_COMPONENT_MASK_ADJ
        @test fixed_components_in_ss == EXPECTED_FC_SS_ADJ
    end
end