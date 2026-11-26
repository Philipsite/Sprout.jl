
@testset "gen_data.jl" begin
    x, y = generate_dataset(10, "test_file", save_to_csv=false)

    sys_in = "mol"
    Xoxides = ["SiO2"; "CaO"; "Al2O3"; "FeO"; "MgO"; "Na2O"]
    MAGEMin_db = Initialize_MAGEMin("sb21", solver=0, verbose=false)
    # manually re-run the 2nd GEM
    p = [x[2,"p_kbar"]]
    t = [x[2,"t_c"]]
    bulk = Vector(x[2, ["SiO2", "CaO", "Al2O3", "FeO", "MgO", "Na2O"]])

    out = multi_point_minimization(p, t, MAGEMin_db, X=bulk, Xoxides=Xoxides, sys_in=sys_in)

    @test out[1].bulk ≈ Vector(x[2,3:8])

    vol_test = zeros(22)
    vol_test[[findfirst([PP..., SS...] .== p) for p in out[1].ph]] .= out[1].ph_frac
    @test vol_test ≈ Vector(y[2,1:22])

    ss_names = out[1].ph[Bool.(out[1].ph_type)]
    for (k, ss) in enumerate(ss_names)
        @test out[1].SS_vec[k].Comp ≈ Vector(y[2, ss .* "_" .* out[1].oxides])
    end

    # bulk-system properties: fixed composition vs an scp=0 run, equilibrium vs an scp=1 run
    o_fix = multi_point_minimization(p, t, MAGEMin_db, X=bulk, Xoxides=Xoxides, sys_in=sys_in, scp=0)[1]
    o_eq  = multi_point_minimization(p, t, MAGEMin_db, X=bulk, Xoxides=Xoxides, sys_in=sys_in, scp=1)[1]

    @test names(y)[end-length(BULK_PROPS)+1:end] == BULK_PROPS
    @test y[2, "V_cm3_mol"]     ≈ o_fix.V
    @test y[2, "Cp_J_K_mol"]    ≈ only(o_fix.s_cp) * o_fix.M_sys / 1e3
    @test y[2, "alpha_1_K"]     ≈ only(o_fix.alpha)
    @test y[2, "K_S_GPa"]       ≈ o_fix.bulkMod
    @test y[2, "G_shear_GPa"]   ≈ o_fix.shearMod
    @test y[2, "Cp_eq_J_K_mol"] ≈ only(o_eq.s_cp) * o_eq.M_sys / 1e3
    @test y[2, "alpha_eq_1_K"]  ≈ only(o_eq.alpha)

    @test molar_mass(bulk, o_fix.oxides)                                 ≈ o_fix.M_sys

    Finalize_MAGEMin(MAGEMin_db)
end
