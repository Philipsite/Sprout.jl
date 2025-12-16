
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

    @testset "create models" begin
        m1 = create_classifier_model(3, 64, 8, 20)

        @test m1 isa Chain
        @test length(m1) == 4
        @test m1[1] isa Dense && size(m1[1].weight) == (64, 8) && m1[1].σ === relu
        @test m1[2] isa Dense && size(m1[2].weight) == (64, 64) && m1[2].σ === relu
        @test m1[3] isa Dense && size(m1[3].weight) == (64, 64) && m1[3].σ === relu
        @test m1[4] isa Dense && size(m1[4].weight) == (20, 64)  && m1[4].σ === sigmoid

        m2 = create_model_pretrained_classifier(2//3, 3, 64,
                                                (clas_out, reg_out) -> (mask_𝑣(clas_out, reg_out[1]), mask_𝐗(clas_out, reg_out[2])),
                                                m1)

          # Top-level should be a Parallel of (masking_f, classifier, regressor)
          @test m2 isa Parallel
          @test length(m2.layers) == 2  # classifier + regressor branch

          # Classifier branch should equal m1 (structure-wise)
          @test m2.layers[1] isa Chain
          @test length(m2.layers[1]) == 4
          @test m2.layers[1][1] isa Dense && size(m2.layers[1][1].weight) == (64, 8) && m2.layers[1][1].σ === relu
          @test m2.layers[1][2] isa Dense && size(m2.layers[1][2].weight) == (64, 64) && m2.layers[1][2].σ === relu
          @test m2.layers[1][3] isa Dense && size(m2.layers[1][3].weight) == (64, 64) && m2.layers[1][3].σ === relu
          @test m2.layers[1][4] isa Dense && size(m2.layers[1][4].weight) == (20, 64) && m2.layers[1][4].σ === sigmoid

          # Regressor branch structure
          reg = m2.layers[2]
          @test reg isa Chain
          @test length(reg) == 3  # Dense -> Dense -> Parallel
          @test reg[1] isa Dense && size(reg[1].weight) == (64, 8) && reg[1].σ === relu
          @test reg[2] isa Dense && size(reg[2].weight) == (64, 64) && reg[2].σ === relu

          # Inner Parallel splitting into (v, X) heads
          @test reg[3] isa Parallel
          @test length(reg[3].layers) == 2
          head_v = reg[3].layers[1]
          head_X = reg[3].layers[2]

          # v head: Dense(64->64, relu) then Dense(64->20)
          @test head_v isa Chain
          @test length(head_v) == 2
          @test head_v[1] isa Dense && size(head_v[1].weight) == (64, 64) && head_v[1].σ === relu
          @test head_v[2] isa Dense && size(head_v[2].weight) == (20, 64)

          # X head: Dense(64->64, relu) -> Dense(64->(6*20)) -> ReshapeLayer(6,20) -> InjectLayer
          @test head_X isa Chain
          @test length(head_X) == 4
          @test head_X[1] isa Dense && size(head_X[1].weight) == (64, 64) && head_X[1].σ === relu
          @test head_X[2] isa Dense && size(head_X[2].weight) == (6*20, 64)
          @test head_X[3] isa ReshapeLayer && head_X[3].n == 6 && head_X[3].m == 20
          @test head_X[4] isa InjectLayer

    end
end


