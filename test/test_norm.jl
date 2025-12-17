
@testset "norm.jl" begin
    # ---- TEST NORM ------
    x = Float32[0 2 1 0 0 0;
                0 0 0 0 0 0;
                6. 0 0 0 0 0]
    x = reshape(x, size(x, 1), 1, size(x,2))

    x_norm = Norm(x)
    @test x_norm.mean == [0.5; 0; 1;;;]
    @test x_norm.std ≈ [0.83666002; 0.0; 2.449489742;;;]

    x_n = x_norm(x)
    @test x_n ≈ Float32[-0.5976143046671968; 0.0; 2.041241452319315;;;
                         1.7928429140015905; 0.0; -0.4082482904638631;;;
                         0.5976143046671968; 0.0; -0.4082482904638631;;;
                         -0.5976143046671968; 0.0; -0.4082482904638631;;;
                         -0.5976143046671968; 0.0; -0.4082482904638631;;;
                         -0.5976143046671968; 0.0; -0.4082482904638631]

    @test denorm(x_norm, x_n) == x



    # ---- TEST MINMAX-SCALING ------
    x_scale = MinMaxScaler(x)

    @test x_scale.min == [0.; 0.; 0.;;;]
    @test x_scale.max ≈ [2.; 0.; 6.;;;]

    x_s = x_scale(x)

    @test x_s[1, :, :] ≈ Float32[0.0 1.0 0.5 0.0 0.0 0.0]
    @test descale(x_scale, x_s) == x
end
