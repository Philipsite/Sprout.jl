
@testset "misfit.jl" begin
    @testset "loss_asm" begin
        ŷ = Float32[0.9; 0.1; 0.8;;; 0.4; 0.6; 0.2]
        y = Bool[0.0; 0.0; 1.0;;; 0.0; 1.0; 0.0]

        loss = misfit.loss_asm(ŷ, y; ϵ=0.5)
        @test loss ≈ 0.25
    end

    @testset "binary_focal_loss" begin
        ŷ = Float32[0.9; 0.1; 0.8;;; 0.4; 0.6; 0.2]
        y = Bool[0.0; 0.0; 1.0;;; 0.0; 1.0; 0.0]

        loss = misfit.binary_focal_loss(ŷ, y; gamma=2)
        @test loss ≈ 0.34124
    end

    @testset "fraction_mismatched_asm" begin
        ŷ = Float32[0.9; 0.1; 0.8;;; 0.4; 0.6; 0.2;;; 0.6; 0.7; 0.8]
        y = Bool[0.0; 0.0; 1.0;;; 0.0; 1.0; 0.0;;; 1.0; 1.0; 1.0]

        frac = misfit.fraction_mismatched_asm(ŷ, y; ϵ=0.5)
        @test frac ≈ 1/3
    end

    @testset "fraction_mismatched_phases" begin
        ŷ = Float32[0.9; 0.1; 0.8;;; 0.4; 0.6; 0.2;;; 0.6; 0.7; 0.8]
        y = Bool[0.0; 0.0; 1.0;;; 0.0; 1.0; 0.0;;; 1.0; 1.0; 1.0]

        frac = misfit.fraction_mismatched_phases(ŷ, y; ϵ=0.5)
        @test frac ≈ 1/9
    end

    @testset "non-zero absolute/relative deviation" begin
        y = Float32[0.0 0.0 0.0 0.0 0.0;
                    0.0 0.0 0.0 0.0 0.0;
                    0.1 0.0 0.0 0.0 0.1;
                    0.0 0.1 0.1 0.1 0.1;
                    0.0 0.0 0.1 0.0 0.0]

        ŷ = Float32[0.0 0.0 0.0 0.0 0.0;
                    0.0 0.0 0.0 0.0 0.0;
                    0.2 0.0 0.1 0.0 0.1;
                    0.0 0.0 0.1 0.1 0.1;
                    0.0 0.0 0.1 0.0 0.0]

        @test misfit.mae_no_zeros(ŷ, y) ≈ 2/7 * 0.1
        @test misfit.mre_no_zeros(ŷ, y) ≈ 2/7 * 1

        @test misfit.mae_trivial_zeros(ŷ, y) ≈ 3/8 * 0.1
        @test misfit.mre_trivial_zeros(ŷ, y) ≈ sum([1, 1, 0.1/eps(Float32), 0, 0, 0, 0, 0]) / 8
        # test on batched data
        y_batched = repeat(reshape(y, (size(y)..., 1)), 1, 1, 3)
        ŷ_batched = repeat(reshape(ŷ, (size(ŷ)..., 1)), 1, 1, 3)

        # alter some values in batch dimension
        ŷ_batched[3, 1, 2] = 0.3f0
        ŷ_batched[4, 3, 3] = 0.2f0

        @test misfit.mae_no_zeros(ŷ_batched, y_batched) ≈ (6/21 * 0.1 + 1/21 * 0.2)
        @test misfit.mre_no_zeros(ŷ_batched, y_batched) ≈ (6/21 * 1 + 1/21 * 2)

        @test misfit.mae_trivial_zeros(ŷ_batched, y_batched) ≈ (9/24 * 0.1 + 1/24 * 0.2)
        @test misfit.mre_trivial_zeros(ŷ_batched, y_batched) ≈ 1/3 * (sum([1, 1, 0.1/eps(Float32), 0, 0, 0, 0, 0]) / 8 + sum([2, 1, 0.1/eps(Float32), 0, 0, 0, 0, 0]) / 8 + sum([1, 1, 0.1/eps(Float32), 1, 0, 0, 0, 0]) / 8)
    end
end
