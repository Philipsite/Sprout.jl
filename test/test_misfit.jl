
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

@testset "misfit.jl - Mass-balance misfits" begin
    ["qtz", "coe", "st", "ky", "neph", "capv", "plg", "sp", "ol", "wa", "ri", "opx", "cpx", "hpcpx", "ak", "gtmj", "pv", "cf", "mw", "nal"]
    𝑣_ŷ = [0.5; 0; 0; 0; 0; 0; 0.5; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0;;;
           0; 0; 0; 0; 0; 0; 0; 0; 0.8; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0.2; 0;;;
           0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0]

    𝐗_ŷ = repeat(reshape(FC_SS, size(FC_SS)..., 1), 1, 1, 3)
    𝐗_ŷ[:, 1, 1] = [0.7, 0.05, 0.15, 0.0, 0.0, 0.1]         # Plg with 0.2Ab + 0.8An
    𝐗_ŷ[:, 3, 2] = [1/3, 0.0, 0.0, 0.5*(2/3), 0.5*(2/3), 0.0]
    𝐗_ŷ[:, 13, 2] = [0.0, 0.0, 0.25, 0.5, 0.0, 0.25]        # Mw with 0.5 anao + 0.5 wustite

    𝐗_ŷ[:, :, 3] .= 39473       # garbage that should be zero-ed by 𝑣_ŷ

    bulk = [0.85; 0.025; 0.075; 0.0; 0.0; 0.05;;;
            0.26666666666666666; 0.0; 0.05; 0.3666666666666667;  0.26666666666666666; 0.05;;;
            0.0; 0.0; 0.0; 0.0; 0.0; 0.0]

    @testset "recalculate_bulk" begin
        bulk_r = misfit.recalculate_bulk((𝑣_ŷ, 𝐗_ŷ), pure_phase_comp = reshape(PP_COMP_adj, 6, :))
        @test bulk_r ≈ bulk atol=1e-5
    end

    @testset "mass_balance_abs_misfit" begin
        mae = misfit.mass_balance_abs_misfit((𝑣_ŷ, 𝐗_ŷ), bulk; pure_phase_comp = reshape(PP_COMP_adj, 6, :))
        @test mae ≈ 0.0 atol=1e-5

        𝑣_ŷ_mod = copy(𝑣_ŷ)
        𝑣_ŷ_mod[1, 1, 1] = 0.4

        mae = misfit.mass_balance_abs_misfit((𝑣_ŷ_mod, 𝐗_ŷ), bulk; agg=sum, pure_phase_comp = reshape(PP_COMP_adj, 6, :))
        @test mae ≈ 0.1 atol=1e-5
        mae = misfit.mass_balance_abs_misfit((𝑣_ŷ_mod, 𝐗_ŷ), bulk; pure_phase_comp = reshape(PP_COMP_adj, 6, :))
        @test mae ≈ 0.1/(3*6) atol=1e-5
    end

    @testset "mass_balance_rel_misfit" begin
        mre = misfit.mass_balance_rel_misfit((𝑣_ŷ, 𝐗_ŷ), bulk; pure_phase_comp = reshape(PP_COMP_adj, 6, :))
        @test mre ≈ 0.0 atol=1e-5

        𝑣_ŷ_mod = copy(𝑣_ŷ)
        𝑣_ŷ_mod[1, 1, 1] = 0.4

        mre = misfit.mass_balance_rel_misfit((𝑣_ŷ_mod, 𝐗_ŷ), bulk; agg=sum, pure_phase_comp = reshape(PP_COMP_adj, 6, :))
        @test mre ≈ 0.1/0.85 atol=1e-5
        mre = misfit.mass_balance_rel_misfit((𝑣_ŷ_mod, 𝐗_ŷ), bulk; pure_phase_comp = reshape(PP_COMP_adj, 6, :))
        @test mre ≈ (0.1/0.85)/(3*6) atol=1e-5
    end

    @testset "mass_residual" begin
        mr = misfit.mass_residual((𝑣_ŷ, 𝐗_ŷ); pure_phase_comp = reshape(PP_COMP_adj, 6, :))
        @test mr ≈ 1/3 atol=1e-5

        𝑣_ŷ_mod = copy(𝑣_ŷ)
        𝑣_ŷ_mod[1, 1, 1] = 0.4

        mr = misfit.mass_residual((𝑣_ŷ_mod, 𝐗_ŷ); agg=sum, pure_phase_comp = reshape(PP_COMP_adj, 6, :))
        @test mr ≈ 1.1 atol=1e-5
        mr = misfit.mass_residual((𝑣_ŷ_mod, 𝐗_ŷ); pure_phase_comp = reshape(PP_COMP_adj, 6, :))
        @test mr ≈ 1.1/3 atol=1e-5
    end
end
