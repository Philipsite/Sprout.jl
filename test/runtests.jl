using Test
using CSV, DataFrames, JLD2
using Flux
using sb21_surrogate

@testset "sb21_surrogate" begin

include("test_gen_data.jl")
include("test_phases_sb21.jl")
include("test_preprocessing.jl")

end
