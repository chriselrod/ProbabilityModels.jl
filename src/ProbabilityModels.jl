module ProbabilityModels

using   MacroTools, DiffRules,
        VectorizationBase, SIMDPirates, LoopVectorization, SLEEFPirates,
        PaddedMatrices, StructuredMatrices,
        DistributionParameters, ProbabilityDistributions,
        DynamicHMC, LogDensityProblems,
        Random, VectorizedRNG, RandomNumbers,
        LinearAlgebra, Statistics, Distributed

import MacroTools: postwalk, prewalk, @capture, @q
import PaddedMatrices: RESERVED_INCREMENT_SEED_RESERVED, RESERVED_DECREMENT_SEED_RESERVED,
    RESERVED_MULTIPLY_SEED_RESERVED, RESERVED_NMULTIPLY_SEED_RESERVED,
    AbstractFixedSizePaddedVector, AbstractMutableFixedSizePaddedVector,
    AbstractMutableFixedSizePaddedArray,
    StackPointer

export @model, NUTS_init_tune_mcmc_default, NUTS_init_tune_distributed, sample_cov, sample_mean

abstract type AbstractProbabilityModel{D} <: LogDensityProblems.AbstractLogDensityProblem end
LogDensityProblems.dimension(::AbstractProbabilityModel{D}) where {D} = D

include("adjoints.jl")
include("misc_functions.jl")
include("special_diff_rules.jl")
include("reverse_autodiff_passes.jl")
include("model_macro_passes.jl")
include("dynamic_hmc_interface.jl")

PaddedMatrices.@support_stack_pointer ITPExpectedValue
PaddedMatrices.@support_stack_pointer ∂ITPExpectedValue
function __init__()
    @eval const GLOBAL_ScalarVectorPCGs = threadrandinit()
    # Allocates 1 GiB per thread for the stack by default.
    # Can be controlled via the environmental variable PROBABILITY_MODELS_STACK_SIZE
    @eval const STACK_POINTER = PaddedMatrices.StackPointer( Libc.malloc(Threads.nthreads() * get(ENV, "PROBABILITY_MODELS_STACK_SIZE", 1 << 30 ) ))
    # @eval const GLOBAL_WORK_BUFFER = Vector{Vector{UInt8}}(Base.Threads.nthreads())
    # Threads.@threads for i ∈ eachindex(GLOBAL_WORK_BUFFER)
    #     GLOBAL_WORK_BUFFER[i] = Vector{UInt8}(0)
    # end
    for m ∈ (:ITPExpectedValue, :∂ITPExpectedValue)
        push!(PaddedMatrices.STACK_POINTER_SUPPORTED_METHODS, m)
    end
end
function realloc_stack(n::Integer)
    @warn """You must redefine all probability models; their stacks have been deallocated.
Re-evaluating densities without first recompiling them will likely crash Julia!"""
    global STACK_POINTER = Libc.realloc(STACK_POINTER, n)
end

end # module
