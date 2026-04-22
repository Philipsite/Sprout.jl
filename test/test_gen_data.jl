
@testset "gen_data.jl" begin
    x, y = generate_dataset(10, "test_file", save_to_csv=false)

    sys_in = "mol"
    Xoxides = ["SiO2"; "CaO"; "Al2O3"; "FeO"; "MgO"; "Na2O"]
    MAGEMin_db = Initialize_MAGEMin("sb21", solver=2, verbose=false)
    # manually re-run the 2nd GEM
    p = [x[2,"p_kbar"]]
    t = [x[2,"t_c"]]
    bulk = Vector(x[2, ["SiO2", "CaO", "Al2O3", "FeO", "MgO", "Na2O"]])

    out = multi_point_minimization(p, t, MAGEMin_db, X=bulk, Xoxides=Xoxides, sys_in=sys_in)

    @test out[1].bulk_S ≈ Vector(x[2,3:8])

    vol_test = zeros(22)
    vol_test[[findfirst([PP..., SS...] .== p) for p in out[1].ph]] .= out[1].ph_frac
    @test vol_test ≈ Vector(y[2,1:22])

    # ρ_test = zeros(22)
    # # manually looked up corresponding densities in out struct
    # ρ_test[[findfirst([PP..., SS...] .== p) for p in out[1].ph]] .= [4040.5706490200764, 3992.3580267826483, 4504.590963296475, 4493.523996228808]
    # @test ρ_test == Vector(y[2,23:44])

    ss_comp_ri = out[1].SS_vec[1].Comp
    ss_comp_gtmj = out[1].SS_vec[2].Comp

    @test ss_comp_ri ≈ Vector(y[2,["ri_SiO2", "ri_CaO", "ri_Al2O3", "ri_FeO", "ri_MgO", "ri_Na2O"]])
    @test ss_comp_gtmj ≈ Vector(y[2,["gtmj_SiO2", "gtmj_CaO", "gtmj_Al2O3", "gtmj_FeO", "gtmj_MgO", "gtmj_Na2O"]])

    @test out[1].bulkMod ≈ y[2, ["bulk_modulus"]]...
    @test out[1].shearMod ≈ y[2, ["shear_modulus"]]...


    @testset "metapelites" begin

        # corresponds to rentires 1, 150 and 800 in the FPWMP 2022 database
        fpwmp22_testdata = DataFrame(
            SiO2 = [70.9, 59.141, 53.04],
            TiO2 = [0.52, 0.328, 0.67],
            Al2O3 = [14.6, 16.03, 25.44],
            Fe2O3 = Union{Missing, Float64}[missing, missing, 1.54],
            FeO = [3.43, 5.6864072, 4.09],
            MnO = [0.04, 0.034, 0.04],
            MgO = [0.76, 6.181, 2.37],
            CaO = [0.81, 0.174, 0.22],
            Na2O = [2.33, 0.71, 0.75],
            K2O = [5.84, 6.115, 6.45],
            P2O5 = Union{Missing, Float64}[missing, 0.095, 0.12],
            LOI = Union{Missing, Float64}[0.01, 1.919, 4.5],
            Total = [99.23, 94.403, 94.46]
        )

        #=
        # Calculated ground-truths using the spreadsheet from Dave P.
        =#
        gt_fpwmp22_1   = [78.28, 0.43, 9.50, 0.00, 3.17, 0.04, 1.25, 0.72, 2.49, 4.11]
        gt_fpwmp22_1 ./= sum(gt_fpwmp22_1)
        gt_fpwmp22_150 = [67.61, 0.28, 10.80, 0.00, 5.44, 0.03, 10.53, 0.06, 0.79, 4.46]
        gt_fpwmp22_150 ./= sum(gt_fpwmp22_150)
        gt_fpwmp22_800 = [65.47, 0.62, 18.51, 0.72, 4.22, 0.04, 4.36, 0.08, 0.90, 5.08]
        gt_fpwmp22_800 ./= sum(gt_fpwmp22_800)

        pred_fpwmp22_1 = preprocess_fpwmp22(fpwmp22_testdata[1:1, :], 0., eps(Float64), min_val=0.0)
        pred_fpwmp22_150 = preprocess_fpwmp22(fpwmp22_testdata[2:2, :], 0., eps(Float64), min_val=0.0)
        pred_fpwmp22_800 = preprocess_fpwmp22(fpwmp22_testdata[3:3, :], 0.23 - 0.08, 0.15, min_val=0.0)

        @test gt_fpwmp22_1 ≈ Vector(pred_fpwmp22_1[1, :]) atol = 1e-2
        @test gt_fpwmp22_150 ≈ Vector(pred_fpwmp22_150[1, :]) atol = 1e-2
        @test gt_fpwmp22_800 ≈ Vector(pred_fpwmp22_800[1, :]) atol = 1e-4
    end
    Finalize_MAGEMin(MAGEMin_db)

    @testset "minimisation with custom TD params" begin
        W = [2. 3. 3.; 1 1 1; 2 0 0]
        p = 10.
        t = 5.

        WG = Sprout.calculate_w_g(W, p, t)
        @test WG ≈ [17., 6, 2]

        MAGEMin_db = Initialize_MAGEMin("mp", solver=0, verbose=false)
        p = 12.
        t = 600.
        bulk = [78.28, 0.43, 9.50, 0.00, 3.17, 0.04, 1.25, 0.72, 2.49, 4.11]
        Xoxides = ["SiO2"; "TiO2"; "Al2O3"; "Fe2O3"; "FeO"; "MnO"; "MgO"; "CaO"; "Na2O"; "K2O"]
        sys_in = "mol"
        out = multi_point_minimization([p], [t], MAGEMin_db, X=[bulk], Xoxides=Xoxides, sys_in=sys_in)

        # (1) - test mpm_custom gives same result as multi_point_minimization when using the original TD params
        out_custom =  Sprout.mpm_custom([p], [t], MAGEMin_db, [bulk], Xoxides, sys_in)
        @test out_custom[1].ph == out[1].ph
        @test out_custom[1].ph_frac == out[1].ph_frac
        @test out_custom[1].bulk_S == out[1].bulk_S

        # (1.1) - test mpm_custom single point method
        out_custom_single = Sprout.mpm_custom(p, t, MAGEMin_db, bulk, Xoxides, sys_in)
        @test out_custom_single.ph == out_custom[1].ph
        @test out_custom_single.ph_frac == out_custom[1].ph_frac
        @test out_custom_single.bulk_S == out_custom[1].bulk_S

        # (2) - test mpm_custom gives same result as multi_point_minimization when using the original TD params
        # and passing them explicitly as arguments
        mod_phase = ["g", "bi"]

        W_g = [2.5  0  0 ;
               2.0  0  0 ;
               31.0 0  0 ;
               5.4  0  0 ;
               2.0  0  0 ;
               5.0  0  0 ;
               22.6 0  0 ;
               0.0  0  0 ;
               29.4 0  0 ;
               -15.3 0  0 ]

        W_bi = [12    0  0 ;
                 4    0  0 ;
                10    0  0 ;
                30    0  0 ;
                 8    0  0 ;
                 9    0  0 ;
                 8    0  0 ;
                15    0  0 ;
                32    0  0 ;
                13.6  0  0 ;
                 6.3  0  0 ;
                 7    0  0 ;
                24    0  0 ;
                 5.6  0  0 ;
                 8.1  0  0 ;
                40    0  0 ;
                 1    0  0 ;
                13    0  0 ;
                40    0  0 ;
                30    0  0 ;
                11.6  0  0]


        # ["py", "alm", "spss", "gr", "kho"]
        ∆G°_g = [0.0, 0.0, 0.0, 0.0, 0.0]

        # ["phl", "annm", "obi", "east", "tbi", "fbi", "mmbi"]
        ∆G°_bi = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        out_custom_mod =  Sprout.mpm_custom(p, t, MAGEMin_db, bulk, Xoxides, sys_in; mod_phases=mod_phase, W=[W_g, W_bi], ∆G°=[∆G°_g, ∆G°_bi])
        @test out_custom_mod.ph == out[1].ph
        @test out_custom_mod.ph_frac == out[1].ph_frac
        @test out_custom_mod.bulk_S == out[1].bulk_S

        Finalize_MAGEMin(MAGEMin_db)
    end

    @testset "outs_to_df" begin
        db_info = Sprout.load_db_info(joinpath("..", "dtb_summaries", "mp_summary.toml"))
        mp_config = TOML.parsefile(joinpath("..", "dtb_summaries", "mp_config.toml"))
        db_info = update_solvus_phases_db_info(db_info, mp_config)

        bulk    = [78.28, 0.43, 9.50, 0.00, 3.17, 0.04, 1.25, 0.72, 2.49, 4.11]
        Xoxides = ["SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO", "MgO", "CaO", "Na2O", "K2O"]
        sys_in  = "mol"
        p, t    = 12., 600.

        MAGEMin_db = Initialize_MAGEMin("mp", solver=0, verbose=false)
        outs = multi_point_minimization([p], [t], MAGEMin_db, X=[bulk], Xoxides=Xoxides, sys_in=sys_in)

        # this should trigger an error, as "fsp" is stable (`name_solvus=false`), but not in the db_info phase list
        @test_throws CompositeException outs_to_df(outs, db_info)

        outs = multi_point_minimization([p], [t], MAGEMin_db, X=[bulk], Xoxides=Xoxides, sys_in=sys_in, name_solvus=true)
        df   = outs_to_df(outs, db_info)
        out  = outs[1]

        # P-T
        @test df[1, "P_Pa"] ≈ out.P_kbar * 1e5
        @test df[1, "T_C"]  ≈ out.T_C
        # test that no mod TD data are present in the df
        @test !in("W_bi_phl-ann_H", names(df))
        @test !in("bi_∆G°_ann", names(df))

        @test collect(df[1, ["bulk_SiO2", "bulk_Al2O3", "bulk_CaO", "bulk_MgO", "bulk_FeO", "bulk_K2O", "bulk_Na2O", "bulk_TiO2", "bulk_O", "bulk_MnO", "bulk_H2O"]]) ≈ [0.7828782878287827, 0.095009500950095, 0.007200720072007199, 0.012501250125012499, 0.0317031703170317, 0.0411041104110411, 0.024902490249024894, 0.0043004300430043, 0.0, 0.00040004000400039994, 0.0]

        # bulk system scalars
        @test df[1, "G_sys_Jmol⁻¹"]    ≈ out.G_system   * 1000.0
        @test df[1, "S_sys_JK⁻¹mol⁻¹"] ≈ out.entropy[1] * 1000.0
        @test df[1, "ρ_sys_kgm⁻³"]     ≈ out.rho
        @test df[1, "Vp_kms⁻¹"]        ≈ out.Vp
        @test df[1, "Vs_kms⁻¹"]        ≈ out.Vs

        @test df[1, "μ_CaO_Jmol⁻¹"]       ≈ -785.1737376941828 * 1000

        # test some modes (manually extracted this from MAGEMin)
        @test df[1, "molar_fraction_afs"] ≈ 0.38539  atol=1e-5
        @test df[1, "molar_fraction_pl"] ≈ 0.15628   atol=1e-5

        @test df[1, "G_q_Jmol⁻¹"]     ≈ -939.6518062259456  * 1000
        @test df[1, "H_ky_Jmol⁻¹"]    ≈ -1245.8675301449211 * 1000
        @test df[1, "S_pl_JK⁻¹mol⁻¹"] ≈ 0.11904667784301644 * 1000

        # test some absent phases
        @test df[1, "S_hemm_JK⁻¹mol⁻¹"] ≈ 0.0
        @test df[1, "molar_fraction_liq"] ≈ 0.0


        # locate afs and g among stable SS phases to index SS_vec correctly
        ss_stable = [ph for ph in out.ph if ph in db_info.ss_names]
        afs_idx = findfirst(==("afs"), ss_stable)
        g_idx   = findfirst(==("g"),   ss_stable)

        # oxide composition
        for (k, ox) in enumerate(db_info.oxides)
            @test df[1, "afs_comp_$(ox)"] ≈ out.SS_vec[afs_idx].Comp[k]
            @test df[1, "g_comp_$(ox)"]   ≈ out.SS_vec[g_idx].Comp[k]
        end

        # end-member fractions
        for (k, em) in enumerate(["ab", "an", "san"])
            @test df[1, "afs_emfrac_$(em)"] ≈ out.SS_vec[afs_idx].emFrac[k]
        end
        for (k, em) in enumerate(["py", "alm", "spss", "gr", "kho"])
            @test df[1, "g_emfrac_$(em)"] ≈ out.SS_vec[g_idx].emFrac[k]
        end

        # site fractions
        for (k, sf) in enumerate(["xNaA", "xCaA", "xKA", "xAlTB", "xSiTB"])
            @test df[1, "afs_sf_$(sf)"] ≈ out.SS_vec[afs_idx].siteFractions[k]
        end
        for (k, sf) in enumerate(["xMgX", "xFeX", "xMnX", "xCaX", "xAlY", "xFe3Y"])
            @test df[1, "g_sf_$(sf)"] ≈ out.SS_vec[g_idx].siteFractions[k]
        end

        # test case with modified TD data
        mod_phases = ["bi", "g"]
        W          = [[12    0  0 ;
                        4    0  0 ;
                       10    0  0 ;
                       30    0  0 ;
                        8    0  0 ;
                        9    0  0 ;
                        8    0  0 ;
                       15    0  0 ;
                       32    0  0 ;
                       13.6  0  0 ;
                        6.3  0  0 ;
                        7    0  0 ;
                       24    0  0 ;
                        5.6  0  0 ;
                        8.1  0  0 ;
                       40    0  0 ;
                        1    0  0 ;
                       13    0  0 ;
                       40    0  0 ;
                       30    0  0 ;
                       11.6  0  0],
                      [2.5  0  0 ;
                       2.0  0  0 ;
                       31.0 0  0 ;
                       5.4  0  0 ;
                       2.0  0  0 ;
                       5.0  0  0 ;
                       22.6 0  0 ;
                       0.0  0  0 ;
                       29.4 0  0 ;
                       -15.3 0  0]]

        W_names = [["phl-annm",  "phl-obi",   "phl-east",  "phl-tbi",   "phl-fbi",   "phl-mmbi",
                    "annm-obi",  "annm-east", "annm-tbi",  "annm-fbi",  "annm-mmbi",
                    "obi-east",  "obi-tbi",   "obi-fbi",   "obi-mmbi",
                    "east-tbi",  "east-fbi",  "east-mmbi",
                    "tbi-fbi",   "tbi-mmbi",
                    "fbi-mmbi"],
                   ["py-alm",   "py-spss",   "py-gr",     "py-kho",
                    "alm-spss", "alm-gr",    "alm-kho",
                    "spss-gr",  "spss-kho",
                    "gr-kho"]]

        ∆G°        = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0]]

        ∆G°_names  = [["phl", "annm", "obi", "east", "tbi", "fbi", "mmbi"],
                      ["py", "alm", "spss", "gr", "kho"]]

        outs   = Sprout.mpm_custom([p], [t], MAGEMin_db, [bulk], Xoxides, sys_in;
                                   mod_phases=mod_phases, W=[W], ∆G°=[∆G°], name_solvus=true)
        df_mod = outs_to_df(outs, db_info;
                                        modified_phases=mod_phases,
                                        W=[W], W_binary_names=W_names,
                                        ∆G°=[∆G°], ∆G°_names=∆G°_names)

        @test df_mod[1, "W_bi_phl-annm_H"] ≈ W[1][1, 1]   # 12.0
        @test df_mod[1, "W_bi_phl-obi_H"]  ≈ W[1][2, 1]   #  4.0
        @test df_mod[1, "W_g_py-alm_H"]    ≈ W[2][1, 1]   #  2.5
        @test df_mod[1, "W_g_gr-kho_H"]    ≈ W[2][10, 1]  # -15.3

        @test df_mod[1, "bi_∆G°_phl"] ≈ ∆G°[1][1]   # 0.0
        @test df_mod[1, "g_∆G°_alm"]  ≈ ∆G°[2][2]   # 0.0

        Finalize_MAGEMin(MAGEMin_db)
    end

    @testset "extract_dataset" begin
        db_info = Sprout.load_db_info(joinpath("..", "dtb_summaries", "mp_summary.toml"))
        mp_config = TOML.parsefile(joinpath("..", "dtb_summaries", "mp_config.toml"))
        db_info = update_solvus_phases_db_info(db_info, mp_config)

        bulk    = [78.28, 0.43, 9.50, 0.00, 3.17, 0.04, 1.25, 0.72, 2.49, 4.11]
        Xoxides = ["SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO", "MgO", "CaO", "Na2O", "K2O"]
        sys_in  = "mol"
        p, t    = 12., 600.
        mod_phases = ["bi", "g"]
        W          = [[12    0  0 ;
                        4    0  0 ;
                       10    0  0 ;
                       30    0  0 ;
                        8    0  0 ;
                        9    0  0 ;
                        8    0  0 ;
                       15    0  0 ;
                       32    0  0 ;
                       13.6  0  0 ;
                        6.3  0  0 ;
                        7    0  0 ;
                       24    0  0 ;
                        5.6  0  0 ;
                        8.1  0  0 ;
                       40    0  0 ;
                        1    0  0 ;
                       13    0  0 ;
                       40    0  0 ;
                       30    0  0 ;
                       11.6  0  0],
                      [2.5  0  0 ;
                       2.0  0  0 ;
                       31.0 0  0 ;
                       5.4  0  0 ;
                       2.0  0  0 ;
                       5.0  0  0 ;
                       22.6 0  0 ;
                       0.0  0  0 ;
                       29.4 0  0 ;
                       -15.3 0  0]]

        W_names = [["phl-annm",  "phl-obi",   "phl-east",  "phl-tbi",   "phl-fbi",   "phl-mmbi",
                    "annm-obi",  "annm-east", "annm-tbi",  "annm-fbi",  "annm-mmbi",
                    "obi-east",  "obi-tbi",   "obi-fbi",   "obi-mmbi",
                    "east-tbi",  "east-fbi",  "east-mmbi",
                    "tbi-fbi",   "tbi-mmbi",
                    "fbi-mmbi"],
                   ["py-alm",   "py-spss",   "py-gr",     "py-kho",
                    "alm-spss", "alm-gr",    "alm-kho",
                    "spss-gr",  "spss-kho",
                    "gr-kho"]]

        ∆G°        = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0]]

        ∆G°_names  = [["phl", "annm", "obi", "east", "tbi", "fbi", "mmbi"],
                      ["py", "alm", "spss", "gr", "kho"]]

        MAGEMin_db = Initialize_MAGEMin("mp", solver=0, verbose=false)

        outs   = Sprout.mpm_custom([p], [t], MAGEMin_db, [bulk], Xoxides, sys_in;
                                   mod_phases=mod_phases, W=[W], ∆G°=[∆G°], name_solvus=true)

        df = outs_to_df(outs, db_info;
                        modified_phases=mod_phases,
                        W=[W], W_binary_names=W_names,
                        ∆G°=[∆G°], ∆G°_names=∆G°_names)
        x, y = extract_dataset(df, db_info,
                               ["P_Pa", "T_C", "bulk", "W", "∆G°"],
                               ["G_sys_Jmol⁻¹", "molar_fraction"],
                               modified_phases=mod_phases,
                               W_binary_names=W_names,
                               ∆G°_names=∆G°_names)

        @test x[1, 1] ≈ p * 1e5
        @test x[1, 2] ≈ t
        @test x[1, 3] ≈ 0.7828782878287828
        @test x[1, 13] ≈ 0.0 # H20 in bulk
        @test x[1, 14] ≈ W[1][1, 1]   # 12.0
        @test x[1, 15] ≈ W[1][1, 2]   #  0.0
        @test x[1, 77] ≈ W[2][1, 1]   # 2.5
        @test x[1, 107] ≈ ∆G°[1][1]   # 0.0
        @test x[1, 112] ≈ ∆G°[2][2]   # 0.0

        @test y[1, 1] ≈ outs[1].G_system * 1000.0
        @test y[1, "molar_fraction_liq"] ≈ 0.0
        @test y[1, "molar_fraction_afs"] ≈ 0.38539  atol=1e-5
    end

end
