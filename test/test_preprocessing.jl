
@testset "preprocessing.jl" begin
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
    x_data = CSV.read("test_data/sb21_25Nov26_t_x.csv", DataFrame)
    y_data = CSV.read("test_data/sb21_25Nov26_t_y.csv", DataFrame)

    x, 𝑣, 𝐗_ss, 𝐩_sys = Sprout.preprocess_data(x_data, y_data)

    # check types
    @test isa(x, Array{Float32,3})
    @test isa(𝑣, Array{Float32,3})
    @test isa(𝐗_ss, Array{Float32,3})
    @test isa(𝐩_sys, Array{Float32,3})

    # check shapes > the 11th sample in the test dataset has NaNs and should be filtered out
    @test size(x) == (8, 1, 10)
    @test size(𝑣) == (20, 1, 10)
    @test size(𝐗_ss) == (6, 14, 10)
    @test size(𝐩_sys) == (length(BULK_PROPS), 1, size(x, 3))
    @test 𝐩_sys[:, 1, 3] == Float32.(Vector(y_data[3, BULK_PROPS]))   # assuming no NaN rows before row 3


    # check specific values
    @test x[:, 1, 2] == Float32[303.3693961143624,1380.9666288297435,0.3632355745501365,0.003111355550740342,0.0019881351171604352,0.054943618287904554,0.5763123051702378,0.0004090113238203565]
    @test 𝑣[:, 1, 10] == Float32[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7204751817005397,0.0,0.0,0.1498446056531769,0.06116876409962413,0.0,0.0,0.06851144854665929,0.0,0.0,0.0,0.0]
    @test 𝐗_ss[:, :, 1] == Matrix{Float32}([
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.3333333333333333 0.0 0.0 0.06482680926429615 0.6018398574023706 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.49961948197056466 0.001490492286156892 0.0007610360588706407 0.04127683677898582 0.4568521529054219 0.0;
                                            0.5192864307381306 0.20915205158614983 0.019512631848677475 0.009262680056022269 0.22342437466270698 0.01936183110831289;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.4289956891751907 0.03963901694543573 0.14236700632037444 0.07234540265095979 0.3165334233511208 0.00011946155691860425;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0;
                                            0.0 0.0 0.0 0.0 0.0 0.0])'

end
