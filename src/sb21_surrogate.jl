module sb21_surrogate

using Reexport: @reexport
using Flux, JLD2, CUDA, cuDNN
using CSV, DataFrames, Statistics, ProgressBars
@reexport using MAGEMin_C
using Base.Threads, Random, Dates
using CairoMakie

include("phases_sb21.jl")
export PP, PP_COMP, SS, SS_COMP, IDX_of_variable_components_in_SS

include("custom_loss.jl")
export loss_asm, loss_vol

include("gen_data.jl")
export generate_dataset, generate_bulk_array

include("model.jl")
export connection, connection_reduced_ss_comp

include("norm.jl")
export Norm, MinMaxScaler

include("training.jl")
export train_loop, post_training_plots

end
