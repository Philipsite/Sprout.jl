module sb21_surrogate

using Reexport: @reexport
using Flux, JLD2, CUDA, cuDNN
using CSV, DataFrames, Statistics, Distributions, ProgressBars
@reexport using MAGEMin_C
using Base.Threads, Random, Dates, BenchmarkTools
using CairoMakie

include("phases_sb21.jl")
export PP, PP_COMP, SS, SS_COMP, IDX_OF_PHASES_NEVER_STABLE, IDX_of_variable_components_in_SS

include("custom_loss.jl")
export loss_asm, loss_vol
export fraction_mismatched_asm, fraction_mismatched_phases

include("gen_data.jl")
export generate_dataset, generate_bulk_array, generate_noisy_bulk_array

include("hyper_parameter_tuning.jl")
export create_model, create_composite_model, run_hyperparam_tuning, load_hyperparam_tuning_results, estimate_inference_time

include("model.jl")
export connection_reduced, connection_reduced_phys_params

include("norm.jl")
export Norm, MinMaxScaler, denorm, inv_scaling

include("phase_diagram.jl")
export generate_mineral_assemblage_diagram, plot_mineral_assemblage_diagram

include("preprocessing.jl")
export preprocess_for_classifier, preprocess_for_regressor, preprocess_for_regressor_modes_sscomp

include("training.jl")
export train_loop, post_training_plots

end
