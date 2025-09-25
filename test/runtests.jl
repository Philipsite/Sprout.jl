using Test
using CSV, DataFrames
using sb21_surrogate

@testset "sb21_surrogate" begin

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
end

@testset "model.jl" begin
    # --- THIS BLOCK TESTS CONNECTION FUNCTIONS ----
    y_clas = [1. 0. 1. 0.;
              1. 0. 1. 0.;
              1. 0. 0. 0.;
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
              1. 0. 0. 0.;
              0. 1. 0. 0.]
    
    y_reg = ones(75, 4)
    y = connection_reduced_phys_params(y_clas, y_reg)

    @test y[1:19, 1] == [1.; 1.; 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.; 0.]
    @test y[1:19, 3] == [1.; 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.]
    
    y_test_ss_comp2 = zeros(53)
    y_test_ss_comp2[1:4]  .= 1.0
    y_test_ss_comp2[5:6] .= 1.0
    y_test_ss_comp2[9:10] .= 1.0
    @test y[20:end-3, 4] == y_test_ss_comp2

    # test if physical rock properties (last three entries) remain unaltered by the connection function
    @test y[end-2:end, 4] == [1., 1., 1.]

    y_reg = ones(72, 4)
    y = connection_reduced(y_clas, y_reg)

    @test y[1:19, 1] == [1.; 1.; 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.; 0.]
    @test y[1:19, 3] == [1.; 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.]
    
    y_test_ss_comp2 = zeros(53)
    y_test_ss_comp2[1:4]  .= 1.0
    y_test_ss_comp2[5:6] .= 1.0
    y_test_ss_comp2[9:10] .= 1.0
    @test y[20:end, 4] == y_test_ss_comp2


    # --- THIS BLOCK TESTS ARCHIVED CONNECTION FUNCTIONS ----
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
    y = sb21_surrogate.zz_connection(y_clas, y_reg)
    @test y[1:22, 1] == [1.; 1.; 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.; 0.]
    @test y[1:22, 3] == [1.; 1.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0.]
    @test y[23:end, 2] == vcat(repeat([0.], 6*14), ones(6))
    @test y[23:end, 3] == vcat(repeat([0.], 6*10), ones(6), repeat([0.], 6*4))

    y_reg = ones(79, 4)
    y = sb21_surrogate.zz_connection_reduced_ss_comp(y_clas, y_reg)
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

    # --------------------------------------------------------
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

    @test denorm(x_norm, x_n) == x
end

@testset "phases_sb21" begin
    @test size(PP_COMP) == (length(PP) * 6, )
    @test size(SS_COMP) == (length(SS) * 6, )
end

@testset "preprocessing" begin
    x_data = CSV.read("test_data/sb21_22Sep25_t_x.csv", DataFrame)
    y_data = CSV.read("test_data/sb21_22Sep25_t_y.csv", DataFrame)

    x, y = preprocess_for_classifier(x_data, y_data)

    @test x[:,2] ≈ [374.0, 2497.0, 0.4225897, 0.0597008, 0.0442537, 0.0640199, 0.4008087, 0.0086269]
    @test y[:, 9] == Bool.([0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0])

    x, y = preprocess_for_regressor(x_data, y_data)

    @test x[:,2] ≈ [374.0, 2497.0, 0.4225897, 0.0597008, 0.0442537, 0.0640199, 0.4008087, 0.0086269]

    @test y[:,10] ≈ [0.0,0.0,0.01807800334603538,0.0,0.0,0.0,0.0,0.0,
                    0.0,0.0,0.0,0.40019905077843004,0.0,0.0,0.5817229458755346,0.0,0.0,
                    0.0,0.0,

                    0.0,0.0,0.0,0.0,    # Plg
                    0.0,0.0,            # Spl
                    0.0,0.0,            # Ol
                    0.0,0.0,            # Wads
                    0.0,0.0,            # Ri
                    0.0,0.0,0.0,0.0,0.0,  # Opx
                    0.5344598048523903,0.1918682910538337,0.03449725638427598,0.015559247621014455,0.18914311139213358,0.03447228869635211, # Cpx
                    0.0,0.0,              # HP-Cpx
                    0.0,0.0,0.0,0.0,
                    0.448924168153612,0.07167527417649316,0.12598157057018322,0.10773672772397487,0.23773895708326762,0.007943302292469097,
                    0.0,0.0,0.0,0.0,
                    0.0,0.0,0.0,0.0,0.0,
                    0.0,0.0,0.0,0.0,
                    0.0,0.0,0.0,0.0,0.0,
                    
                    3777.8538289099642,186.4584485037721,101.56898524712557]
end
end
