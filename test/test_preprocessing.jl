
@testset begin
    # test Sprout.filter_NaN
    test_data = rand(Float32, 10, 5)
    test_data[2, 3] = NaN32
    @test Sprout.filter_NaN(test_data) == [true, true, false, true, true]

    # test Sprout.indices_of_stable_phases
    @test Sprout.indices_of_stable_phases()[1] == [i for i in 1:22 if i ∉ [7, 19]]
    @test Sprout.indices_of_stable_phases()[2] == [i for i in 1:(15*6) if i ∉ [6 * (k-1) + j for k in [12, ] for j in 1:6]] .+ 22

    # test Sprout.one_hot_phase_stability
    @test Sprout.one_hot_phase_stability(Float32[0.0 0.2; 0.5 0.0]) == Bool[false true; true false]
    @test Sprout.one_hot_phase_stability(Float32[0.0; 0.8; 0.2; 0.0; 0.0;;; 1.0; 0.0; 0.0; 0.0; 0.0;;; 0.0; 0.0; 0.0; 0.1; 0.9;;; 0.0; 0.0; 1.0; 0.0; 0.0]) == Bool[false; true; true; false; false;;; true; false; false; false; false;;; false; false; false; true; true;;; false; false; true; false; false]

    # test Sprout.preprocess_data
    x_data = CSV.read("test_data/sb21_22Sep25_t_x.csv", DataFrame)
    y_data = CSV.read("test_data/sb21_22Sep25_t_y.csv", DataFrame)

    x, 𝑣, 𝐗_ss, ρ, Κ, μ = Sprout.preprocess_data(x_data, y_data)

    # check types
    @test isa(x, Array{Float32,3})
    @test isa(𝑣, Array{Float32,3})
    @test isa(𝐗_ss, Array{Float32,3})
    @test isa(ρ, Array{Float32,3})
    @test isa(Κ, Array{Float32,3})
    @test isa(μ, Array{Float32,3})

    # check shapes > the 11th sample in the test dataset has NaNs and should be filtered out
    @test size(x) == (8, 1, 10)
    @test size(𝑣) == (20, 1, 10)
    @test size(𝐗_ss) == (6, 14, 10)
    @test size(ρ) == (1, 1, 10)
    @test size(Κ) == (1, 1, 10)
    @test size(μ) == (1, 1, 10)

    # check specific values
    @test x[:, 1, 2] == Float32[374.0, 2497.0, 0.4225897, 0.059700884, 0.0442537, 0.06401996, 0.40080875, 0.0086269975]
    @test 𝑣[:, 1, 10] == Float32[0.0,0.0,0.01807800334603538,0.0, 0.0, 0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.40019905077843004,0.0,0.0,0.5817229458755346,0.0,0.0, 0.0,0.0]
    @test 𝐗_ss[:, :, 5] == Matrix{Float32}([0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.3333333333333333 0.0 0.0 0.12171110340774004 0.5449555632589267 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.5191556682027444 0.1445434800199324 0.03219397911672438 0.031155151254258138 0.24944994956560312 0.023501771840737696;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.45394425638567004 0.06149585960335682 0.1136882713147488 0.061766069031069924 0.30191328230312475 0.007192261362029646;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            # 0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.0 0.0 0.010308448414124554 0.41051178898650653 0.5688713141852444 0.010308448414124554;
                                            0.0 0.0 0.0 0.0 0.0 0.0])'

    @test ρ[3] == Float32(3469.1982719555863)
    @test Κ[3] == Float32(134.7502950401204)
    @test μ[3] == Float32(81.48472549201534)

end
