module Sprout

using Reexport: @reexport
using ChainRulesCore
using Flux, ParameterSchedulers, CUDA, cuDNN
using Statistics
using JLD2, TOML, CSV, DataFrames
using Distributions, ProgressBars, ProgressMeter
@reexport using MAGEMin_C
using Base.Threads, Random, Dates, BenchmarkTools
using CairoMakie

include("phases_sb21.jl")
export PP, PP_COMP, PP_COMP_adj, SS, SS_COMP, SS_COMP_adj, IDX_OF_PHASES_NEVER_STABLE, IDX_of_variable_components_in_SS, IDX_phase_frac

include("phases_in_database.jl")
export DatabaseInfo, extract_db_info, extract_db_info_to_TOML, load_db_info, update_solvus_phases_db_info

include("preprocessing.jl")
export get_col_names
export preprocess_data, one_hot_phase_stability

include("norm.jl")
export Norm, denorm, MinMaxScaler, descale

include("model.jl")
export FC_SS, FC_SS_scaled, FC_SS_MASK, ReshapeLayer, InjectLayer, mask_𝐗, mask_𝑣
export create_classifier_model, create_model_pretrained_classifier, create_model_shared_backbone

include("misfit.jl")
export misfit

include("gen_data.jl")
export mpm_custom, run_gem, outs_to_df, extract_dataset, write_to_csv
export generate_dataset, generate_bulk_array, generate_noisy_bulk_array
export preprocess_fpwmp22, generate_bulks_from_df

include("training.jl")
export train_loop, post_training_plots

include("hyperparameter_tuning.jl")
export hpt_classifier, hpt_regressor_pretrained_classifier, hpt_regressor_common_backbone
export load_hyperparam_tuning_results, estimate_inference_time

include("phase_diagram.jl")
export generate_mineral_assemblage_diagram, plot_mineral_assemblage_diagram, plot_mineral_assemblage_diagram!

end
