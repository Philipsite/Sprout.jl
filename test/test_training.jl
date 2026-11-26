
@testset "training.jl" begin
mktempdir() do tmp           # save all outputs to a temp dir for these tests
    @testset "test model freezing" begin
        n_layers = 4;
        n_neurons = 200
        fraction_backbone_layers = 1//2
        batch_size = 8

        # Load CLASSIFIER
        m_classifier = create_classifier_model(2, 200, 8, 20)
        model_state = JLD2.load("test_data/saved_models/classifier/saved_model.jld2", "model_state")
        Flux.loadmodel!(m_classifier, model_state)

        # load DATA
        x_train = CSV.read("test_data/sb21_25Nov26_t_x.csv", DataFrame)
        y_train = CSV.read("test_data/sb21_25Nov26_t_y.csv", DataFrame)
        x_val = CSV.read("test_data/sb21_25Nov26_t_x.csv", DataFrame)
        y_val = CSV.read("test_data/sb21_25Nov26_t_y.csv", DataFrame)

        x_train, 𝑣_train, 𝐗_ss_train, 𝐩_train = preprocess_data(x_train, y_train)
        x_val, 𝑣_val, 𝐗_ss_val, 𝐩_train = preprocess_data(x_val, y_val)

        # Normalise inputs
        xNorm = Norm(x_train)
        x_train = xNorm(x_train)
        x_val = xNorm(x_val)

        # Scale outputs
        𝐗Scale = MinMaxScaler(𝐗_ss_train)
        𝐗_ss_train = 𝐗Scale(𝐗_ss_train)
        𝐗_ss_val = 𝐗Scale(𝐗_ss_val)

        𝑣Scale = MinMaxScaler(𝑣_train)
        𝑣_train = 𝑣Scale(𝑣_train)
        𝑣_val = 𝑣Scale(𝑣_val)

        pp_mat = reshape(PP_COMP_adj, 6, :)
        masking_f = (clas_out, reg_out) -> (mask_𝑣(clas_out, reg_out[1]), mask_𝐗(clas_out, reg_out[2]))

        function loss((𝑣_ŷ, 𝐗_ŷ), (𝑣, 𝐗), x)
            return sum(abs2, 𝑣_ŷ .- 𝑣) + sum(abs2, 𝐗_ŷ .- 𝐗) + misfit.mass_balance_abs_misfit((descale(𝑣Scale, 𝑣_ŷ), descale(𝐗Scale, 𝐗_ŷ)), denorm(xNorm, x)[3:end,:,:], agg=sum, pure_phase_comp=pp_mat) + misfit.closure_condition((descale(𝑣Scale, 𝑣_ŷ), descale(𝐗Scale, 𝐗_ŷ)), (𝑣, 𝐗), agg=sum)
        end
        function mae_𝐗(ŷ, y)
            return misfit.mae_no_zeros(descale(𝐗Scale, ŷ[2]), descale(𝐗Scale, y[2]))
        end

        loader = Flux.DataLoader((x_train, (𝑣_train, 𝐗_ss_train)), batchsize=batch_size, shuffle=true)

        model = create_model_pretrained_classifier(fraction_backbone_layers, n_layers, n_neurons,
                                                   masking_f, m_classifier;
                                                   out_dim_𝑣 = 20, out_dim_𝐗 = (6, 14))
        opt_state = Flux.setup(Flux.Adam(0.001), model)
        Flux.freeze!(opt_state.layers[1])  # freeze the classifier part

        model_trained, opt_state, logs, log_dir_path = train_loop(model, loader, opt_state, (x_val, (𝑣_val, 𝐗_ss_val)), loss, 5; metrics = [mae_𝐗], save_to_subdir=tmp, show_progressbar=false)
        # test if the classifier layers have remained unchanged
        param_prior = Flux.destructure(m_classifier)
        param_post = Flux.destructure(model_trained.layers[1])
        @test param_prior == param_post
    end
end
end
