module sb21_surrogate

using Flux, JLD2, CUDA, cuDNN
using Statistics, ProgressBars
using Dates

include("custom_loss.jl")
export loss_asm, loss_vol

include("norm.jl")
export Norm, MinMaxScaler

include("phases_sb21.jl")
export PP, PP_COMP, SS, SS_COMP, IDX_of_variable_components_in_SS

include("training.jl")
export train_loop

end
