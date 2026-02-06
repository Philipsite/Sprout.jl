
@testset "phases_in_database" begin
    # Expected data from phases_sb21.jl (hardcoded benchmark)
    EXPECTED_OXIDES = ["SiO2", "CaO", "Al2O3", "FeO", "MgO", "Na2O"]
    EXPECTED_PP = ["qtz", "coe", "st", "ky", "neph", "capv", "co"]
    EXPECTED_SS = ["plg", "sp", "ol", "wa", "ri", "opx", "cpx", "hpcpx", "ak", "gtmj", "pv", "ppv", "cf", "mw", "nal"]

    EXPECTED_PP_COMP = Matrix{Float32}([
        0.5  0.5 1.0 1.0 1.0 0.5 0.0;
        0.0  0.0 0.0 0.0 0.0 0.5 0.0;
        0.25 0.5 0.0 0.0 0.0 0.0 1.0;
        0.0  0.0 0.0 0.0 0.0 0.0 0.0;
        0.0  0.0 0.0 0.0 0.0 0.0 0.0;
        0.25 0.0 0.0 0.0 0.0 0.0 0.0
    ])

    EXPECTED_VAR_COMPONENT_MASK = Matrix{Float32}([
        1.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 0.0 1.0;
        1.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
        1.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
        0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
        0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
        1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 1.0 1.0 1.0
    ])

    EXPECTED_FC_SS = FC_SS = Matrix{Float32}([
        0.0 0.0 0.33333334 0.33333334 0.33333334 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    ])

    mktempdir() do tmp
        # Generate TOML file
        Sprout.extract_db_info_to_TOML("sb21"; dir=tmp)
        toml_path = joinpath(tmp, "sb21_summary.toml")
        db_info = Sprout.load_db_info(toml_path)

        # Validate oxides
        @test db_info.oxides == EXPECTED_OXIDES
        @test size(db_info.pp_comp) == (6, 7)
        @test db_info.pp_comp == EXPECTED_PP_COMP
        @test size(db_info.var_mask_components_in_ss) == (6, 15)
        @test db_info.var_mask_components_in_ss == EXPECTED_VAR_COMPONENT_MASK
        @test size(db_info.fixed_components_in_ss) == (6, 15)
        @test db_info.fixed_components_in_ss == EXPECTED_FC_SS

        @show db_info.pp_comp
        @show EXPECTED_PP_COMP
    end
end