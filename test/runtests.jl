using Test
using CSV, DataFrames, JLD2
using Flux
using Sprout

@testset "Sprout" begin

include("test_gen_data.jl")
include("test_phases_sb21.jl")
include("test_preprocessing.jl")
include("test_norm.jl")
include("test_model.jl")
include("test_misfit.jl")
include("test_phase_diagram.jl")

end
