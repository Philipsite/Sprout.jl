
@testset "model.jl" begin
    @test isconcretetype(typeof(FC_SS))
    @test isconst(Main, :FC_SS)

    @testset "ReshapeLayer" begin
        rsl = ReshapeLayer(6, 14)
        x = rand(Float32, 6*14, 1, 100)
        y = rsl(x)
        @test size(y) == (6, 14, 100)
    end

    @testset "InjectLayer" begin
        il = InjectLayer()
        x = rand(Float32, 6, 14, 100)
        ŷ = il(x)

        y_1 = x[:, :, 1]
        y_1[.!(Bool.(FC_SS_MASK))] .= FC_SS[.!(Bool.(FC_SS_MASK))]

        @test ŷ[:, :, 1] == y_1
    end

    @testset "Masking functions" begin
        clas_out = zeros(20, 1, 1)
        clas_out[[7, 8, 15], :] .= 1.0

        reg_𝐗 = ones(6, 14, 1) .* 5
        reg_𝑣 = ones(20, 1, 1) .* 5

        𝐗_mask = zeros(6, 14, 1)
        𝐗_mask[:, [1, 2, 9], :] .= 5
        𝑣_mask = zeros(20, 1, 1)
        𝑣_mask[[7, 8, 15], :, :] .= 5

        @test mask_𝐗(clas_out, reg_𝐗) == 𝐗_mask
        @test mask_𝑣(clas_out, reg_𝑣) == 𝑣_mask
    end
end


