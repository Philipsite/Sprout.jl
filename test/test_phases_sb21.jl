
@testset "phases_sb21" begin
    @test size(PP_COMP) == (length(PP) * 6, )
    @test size(SS_COMP) == (length(SS) * 6, )
end
