using Test
using sb21_surrogate

@testset "sb21_surrogate" begin
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
