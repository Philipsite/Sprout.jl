
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

    # add warm-up test for update_solvus_phases_db_info with mp database,
    # Don't fully understand why this is needed /helps
    # There seem to be some "leakage" from MAGEMin from one extraction of db_info into another!?
    mktempdir() do tmp
        # Generate mp database TOML file
        Sprout.extract_db_info_to_TOML("mp"; dir=tmp)
        toml_path = joinpath(tmp, "mp_summary.toml")
        db_info = Sprout.load_db_info(toml_path)
    end

    # Test update_solvus_phases_db_info with mp database
    mktempdir() do tmp
        # Generate mp database TOML file
        Sprout.extract_db_info_to_TOML("mp"; dir=tmp)
        toml_path = joinpath(tmp, "mp_summary.toml")
        db_info = Sprout.load_db_info(toml_path)

        # Create a test config matching mp_config.toml
        test_mp_config = Dict{String, Any}(
            "name" => "mp",
            "database" => "Metapelite (White et al., 2014)",
            "pp_not_considered" => String[],
            "ss_not_considered" => String[],
            "solvus_names" => Dict{String, Any}(
                "sp" => ["sp", "smt"],
                "fsp" => ["pl", "afs"],
                "mu" => ["mu", "pat"],
                "ilmm" => ["ilmm", "hemm"],
                "ilm" => ["ilm", "hem"]
            )
        )

        # Apply the update
        db_info_updated = Sprout.update_solvus_phases_db_info(db_info, test_mp_config)

        @test db_info_updated.n_ss == db_info.n_ss + 5
        @test db_info_updated.n_sf == db_info.n_sf + 33
        @test db_info_updated.ss_names == ["liq", "pl", "afs", "bi", "g", "ep", "ma", "mu", "pat", "opx", "sa", "cd", "st", "chl", "ctd", "sp", "smt", "mt", "ilm", "hem", "ilmm", "hemm"]
        @test db_info_updated.ss_em_names == [["q4L", "abL", "kspL", "anL", "slL", "fo2L", "fa2L", "h2oL"],
                                              ["ab", "an", "san"],
                                              ["ab", "an", "san"],
                                              ["phl", "annm", "obi", "east", "tbi", "fbi", "mmbi"],
                                              ["py", "alm", "spss", "gr", "kho"],
                                              ["cz", "ep", "fep"],
                                              ["mut", "celt", "fcelt", "pat", "ma", "fmu"],
                                              ["mut", "cel", "fcel", "pat", "ma", "fmu"],
                                              ["mut", "cel", "fcel", "pat", "ma", "fmu"],
                                              ["en", "fs", "fm", "mgts", "fopx", "mnopx", "odi"],
                                              ["spr4", "spr5", "fspm", "spro", "ospr"],
                                              ["crd", "fcrd", "hcrd", "mncd"],
                                              ["mstm", "fst", "mnstm", "msto", "mstt"],
                                              ["clin", "afchl", "ames", "daph", "ochl1", "ochl4", "f3clin", "mmchl"],
                                              ["mctd", "fctd", "mnct", "ctdo"],
                                              ["herc", "sp", "mt", "usp"],
                                              ["herc", "sp", "mt", "usp"],
                                              ["imt", "dmt", "usp"],
                                              ["oilm", "dilm", "dhem"],
                                              ["oilm", "dilm", "dhem"],
                                              ["oilm", "dilm", "dhem", "geik", "pnt"],
                                              ["oilm", "dilm", "dhem", "geik", "pnt"]
                                              ]
        @test db_info_updated.ss_sf_names == [["fac", "pq", "xab", "xksp", "pan", "psil", "pol", "xFe", "xMg", "ph2o"],
                                              ["xNaA", "xCaA", "xKA", "xAlTB", "xSiTB"],
                                              ["xNaA", "xCaA", "xKA", "xAlTB", "xSiTB"],
                                              ["xMgM3", "xMnM3", "xFeM3", "xFe3M3", "xTiM3", "xAlM3", "xMgM12", "xMnM12", "xFeM12", "xSiT", "xAlT", "xOHV", "xOV"],
                                              ["xMgX", "xFeX", "xMnX", "xCaX", "xAlY", "xFe3Y"],
                                              ["xFeM1", "xAlM1", "xFeM3", "xAlM3"],
                                              ["xKA", "xNaA", "xCaA", "xMgM2A", "xFeM2A", "xAlM2A", "xAlM2B", "xFe3M2B", "xSiT1", "xAlT1"],
                                              ["xKA", "xNaA", "xCaA", "xMgM2A", "xFeM2A", "xAlM2A", "xAlM2B", "xFe3M2B", "xSiT1", "xAlT1"],
                                              ["xKA", "xNaA", "xCaA", "xMgM2A", "xFeM2A", "xAlM2A", "xAlM2B", "xFe3M2B", "xSiT1", "xAlT1"],
                                              ["xMgM1", "xFeM1", "xMnM1", "xFe3M1", "xAlM1", "xMgM2", "xFeM2", "xMnM2", "xCaM2", "xSiT", "xAlT"],
                                              ["xMgM3", "xFeM3", "xFe3M3", "xAlM3", "xMgM456", "xFeM456", "xSiT", "xAlT"],
                                              ["xFeX", "xMgX", "xMnX", "xH2OH", "xvH"],
                                              ["xMgX", "xFeX", "xMnX", "xAlY", "xFe3Y", "xTiY", "xvY"],
                                              ["xMgM1", "xMnM1", "xFeM1", "xAlM1", "xMgM23", "xFeM23", "xMgM4", "xFeM4", "xFe3M4", "xAlM4", "xSiT2", "xAlT2"],
                                              ["xAlM1A", "xFe3M1A", "xFeM1B", "xMgM1B", "xMnM1B"],
                                              ["xAl", "xFe3", "xTi", "xMg", "xFe2"],
                                              ["xAl", "xFe3", "xTi", "xMg", "xFe2"],
                                              ["xTiM", "xFe3M", "xFeM", "xFe3T", "xFeT"],
                                              ["xFe2A", "xTiA", "xFe3A", "xFe2B", "xTiB", "xFe3B"],
                                              ["xFe2A", "xTiA", "xFe3A", "xFe2B", "xTiB", "xFe3B"],
                                              ["xFeA", "xTiA", "xMgA", "xMnA", "xFe3A", "xFeB", "xTiB"],
                                              ["xFeA", "xTiA", "xMgA", "xMnA", "xFe3A", "xFeB", "xTiB"]
                                              ]

        expected_var_mask = Matrix{Float32}([
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0;
            1.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            1.0 0.0 0.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 1.0 1.0;
            1.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
            1.0 1.0 1.0 1.0 0.0 0.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            1.0 1.0 1.0 0.0 0.0 0.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
            0.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0;
            0.0 0.0 0.0 1.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0;
            1.0 0.0 0.0 1.0 0.0 1.0 1.0 1.0 1.0 0.0 0.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        ])

        @test size(db_info_updated.var_mask_components_in_ss) == (11, 22)
        @test db_info_updated.var_mask_components_in_ss == expected_var_mask

        expected_indices = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 17, 18, 23, 24, 25, 28, 29, 34, 35, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 53, 54, 56, 57, 58, 60, 64, 66, 67, 68, 69, 70, 71, 72, 73, 75, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 90, 91, 92, 93, 94, 95, 97, 99, 100, 101, 102, 103, 104, 108, 109, 111, 112, 114, 115, 119, 122, 123, 125, 126, 131, 132, 133, 134, 136, 137, 140, 141, 142, 143, 144, 145, 147, 148, 152, 153, 154, 155, 156, 158, 159, 163, 164, 165, 167, 169, 170, 173, 174, 178, 180, 181, 184, 185, 192, 195, 196, 203, 206, 207, 214, 217, 218, 224, 225, 228, 229, 230, 235, 236, 239, 240, 241]
         @test db_info_updated.idx_variable_oxide_components_in_ss_flat == expected_indices
    end

end