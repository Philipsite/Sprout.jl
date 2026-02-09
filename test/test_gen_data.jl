
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

        pred_fpwmp22_1 = preprocess_fpwmp22(fpwmp22_testdata[1:1, :], 0., eps(Float64))
        pred_fpwmp22_150 = preprocess_fpwmp22(fpwmp22_testdata[2:2, :], 0., eps(Float64))
        pred_fpwmp22_800 = preprocess_fpwmp22(fpwmp22_testdata[3:3, :], 0.23 - 0.08, 0.15)

        @test gt_fpwmp22_1 ≈ Vector(pred_fpwmp22_1[1, :]) atol = 1e-2
        @test gt_fpwmp22_150 ≈ Vector(pred_fpwmp22_150[1, :]) atol = 1e-2
        @test gt_fpwmp22_800 ≈ Vector(pred_fpwmp22_800[1, :]) atol = 1e-4
    end
end
