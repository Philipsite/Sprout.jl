using Test
using sb21_surrogate

@testset "sb21_surrogate" begin

@testset "gen_data.jl" begin
    x, y = generate_dataset(10, "test_file", save_to_csv=false)

    sys_in = "wt"
    Xoxides = ["SiO2"; "CaO"; "Al2O3"; "FeO"; "MgO"; "Na2O"]
    MAGEMin_db = Initialize_MAGEMin("sb21", solver=2, verbose=false)
    # manually re-run the 2nd GEM
    p = [x[2,"p_kbar"]]
    t = [x[2,"t_c"]]
    bulk = Vector(x[2, ["SiO2", "CaO", "Al2O3", "FeO", "MgO", "Na2O"]])

    out = multi_point_minimization(p, t, MAGEMin_db, X=bulk, Xoxides=Xoxides, sys_in=sys_in)

    @test out[1].bulk_wt == Vector(x[2,3:8])

    vol_test = zeros(22)
    vol_test[[findfirst([PP..., SS...] .== p) for p in out[1].ph]] .= out[1].ph_frac_vol
    @test vol_test == Vector(y[2,1:22])

    ρ_test = zeros(22)
    # manually looked up corresponding densities in out struct
    ρ_test[[findfirst([PP..., SS...] .== p) for p in out[1].ph]] .= [4040.5706490200764, 3992.3580267826483, 4504.590963296475, 4493.523996228808]
    @test ρ_test == Vector(y[2,23:44])

    ss_comp_ri = out[1].SS_vec[1].Comp_wt
    ss_comp_gtmj = out[1].SS_vec[2].Comp_wt

    @test ss_comp_ri == Vector(y[2,["ri_SiO2", "ri_CaO", "ri_Al2O3", "ri_FeO", "ri_MgO", "ri_Na2O"]])
    @test ss_comp_gtmj == Vector(y[2,["gtmj_SiO2", "gtmj_CaO", "gtmj_Al2O3", "gtmj_FeO", "gtmj_MgO", "gtmj_Na2O"]])

    @test out[1].bulkMod == y[2, ["bulk_modulus"]]...
    @test out[1].shearMod == y[2, ["shear_modulus"]]...
end

@testset "model.jl" begin
    y_clas = [1. 0. 1. 0.;
              1. 0. 1. 0.;
              1. 0. 0. 0.;
              0. 0. 0. 0.;
              0. 0. 0. 0.;
              0. 0. 0. 0.;
              0. 0. 0. 0.;
              0. 0. 0. 1.;
              0. 0. 0. 1.;
              0. 0. 0. 0.;
              0. 0. 0. 1.;
              0. 0. 0. 0.;
              0. 0. 0. 0.;
              0. 0. 0. 0.;
              0. 0. 0. 0.;
              0. 0. 0. 0.;
              0. 0. 0. 0.;
              0. 0. 1. 0.;
              0. 0. 0. 0.;
              0. 0. 0. 0.;
              1. 0. 0. 0.;
              0. 1. 0. 0.]

    y_reg = ones(112, 4)
    y = connection(y_clas, y_reg)
    @test y[1:22, 1] == [1.; 1.; 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.; 0.]
    @test y[1:22, 3] == [1.; 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0.]
    @test y[23:end, 2] == vcat(repeat([0.], 6*14), ones(6))
    @test y[23:end, 3] == vcat(repeat([0.], 6*10), ones(6), repeat([0.], 6*4))

    y_reg = ones(79, 4)
    y = connection_reduced_ss_comp(y_clas, y_reg)
    @test y[1:22, 1] == [1.; 1.; 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.; 0.]
    @test y[1:22, 3] == [1.; 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0.]

    y_test_ss_comp1 = zeros(57)
    y_test_ss_comp1[36:39] .= 1.0
    @test y[23:end, 3] == y_test_ss_comp1
    # println(y[23:end, 3])
    # println(y_test_ss_comp1)

    y_test_ss_comp2 = zeros(57)
    y_test_ss_comp2[1:4]  .= 1.0
    y_test_ss_comp2[5:6] .= 1.0
    y_test_ss_comp2[9:10] .= 1.0
    @test y[23:end, 4] == y_test_ss_comp2

end

@testset "norm.jl" begin
    x = [0 2 1 0 0 0;
         0 0 0 0 0 0;
         6. 0 0 0 0 0]

    x_norm = Norm(x)
    @test x_norm.mean == [0.5; 0; 1;;]
    @test x_norm.std ≈ [0.83666002; 0.0; 2.449489742;;]

    x_n = x_norm(x)
    @test x_n ≈ [-0.5976143046671968 1.7928429140015905 0.5976143046671968 -0.5976143046671968 -0.5976143046671968 -0.5976143046671968; 0.0 0.0 0.0 0.0 0.0 0.0; 2.041241452319315 -0.4082482904638631 -0.4082482904638631 -0.4082482904638631 -0.4082482904638631 -0.4082482904638631]
end

@testset "phases_sb21" begin
    @test size(PP_COMP) == (length(PP) * 6, )
    @test size(SS_COMP) == (length(SS) * 6, )
end
end
