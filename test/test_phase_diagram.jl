
@testset "model.jl" begin
    n_layers = 2;
    n_neurons = 200;
    input_dim = 8;
    output_dim = 20;

    m = create_classifier_model(n_layers, n_neurons, input_dim, output_dim);
    model_state = JLD2.load("test_data/saved_models/classifier/saved_model.jld2", "model_state");
    Flux.loadmodel!(m, model_state);
    xNorm = JLD2.load("test_data/saved_models/classifier/xNorm.jld2", "xNorm");

    P_bounds = (10., 400.)
    T_bounds = (500., 2500.)
    X = Float32[0.3871, 0.0294, 0.0222, 0.0617, 0.4985, 0.0011] # Pyrolite composition from Xu et al. (2008)
    resolution = 10


    asm_grid, var_vec_grid = generate_mineral_assemblage_diagram(P_bounds, T_bounds, X, resolution, m, xNorm)
    @test size(asm_grid) == (resolution, resolution)
    @test size(var_vec_grid) == (resolution, resolution)

    asm_grid_groundtruth_rows13 = ["capv-ak-pv-mw-nal" "capv-ak-pv-mw-nal" "capv-ak-pv-mw-nal" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw";
                                   "st-capv-ak-mw-nal" "capv-ak-pv-mw-nal" "capv-ak-pv-mw-nal" "capv-pv-mw-nal" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw";
                                   "st-capv-ak-mw-nal" "st-capv-ak-pv-mw-nal" "capv-ak-pv-mw-nal" "capv-pv-mw-nal" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw" "capv-pv-mw"]
    @test asm_grid[1:3, :] == asm_grid_groundtruth_rows13

    var_vec_grid_groundtruth_rows13 = [3 3 3 5 5 5 5 5 5 5; 3 3 3 4 5 5 5 5 5 5; 3 2 3 4 5 5 5 5 5 5]
    @test var_vec_grid[1:3, :] == var_vec_grid_groundtruth_rows13

    @test begin
        fig = plot_mineral_assemblage_diagram(asm_grid, var_vec_grid, P_bounds, T_bounds, :batlow)
        true
    end
end
