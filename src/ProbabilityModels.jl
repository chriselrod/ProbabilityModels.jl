module ProbabilityModels

using LinearAlgebra, VectorizationBase,
    SIMDPirates, SLEEFPirates,
    LoopVectorization, VectorizedRNG,
    PaddedMatrices, #StructuredMatrices,
    DistributionParameters,# ProbabilityDistributions,
    ReverseDiffExpressions, StackPointers, UnPack

using DistributionParameters: LengthParamDescription
using ReverseDiffExpressions: addvar!, getconstindex!

export @model

# using ReverseDiffExpressionsBase:
#     RESERVED_INCREMENT_SEED_RESERVED!,
#     initialize_target, uninitialized,
#     ∂mul, ∂getindex, alloc_adjoint,
#     ∂evaluate

# using PaddedMatrices:
#     AbstractFixedSizeVector,
#     AbstractMutableFixedSizeVector,
#     AbstractMutableFixedSizeArray

# import QuasiNewtonMethods:
#     AbstractProbabilityModel,
#     logdensity,
#     logdensity_and_gradient!,
#     dimension

# import DistributionParameters: parameter_names
# import MCMCChainSummaries: MCMCChainSummary
# using DistributionParameters: Bounds
# using InplaceDHMC: STACK_POINTER_REF, LOCAL_STACK_SIZE, NTHREADS,
#     mcmc_with_warmup, threaded_mcmc, NoProgressReport
# using SIMDPirates: lifetime_start!, lifetime_end!

# export @model, MCMCChainSummary,
#     logdensity, logdensity_and_gradient,
#     logdensity_and_gradient!, Bounds,
#     mcmc_with_warmup, threaded_mcmc,
#     RealFloat, RealVector, RealMatrix, RealArray,
#     Bounds, NoProgressReport
    
"""
For debugging, you can set the verbosity level to 0 (default), 1, or 2
ProbabilityModels.verbose_models() = 1
and then recompile your model.
verbose_models() == 0 will not print messages.
verbose_models() > 0 will print generated function output.
verbose_models() > 1 will print output while evaluating the logdensity (and gradients).
"""
verbose_models() = 0

include("read_model.jl")
include("preprocess_models.jl")
include("model_macro.jl")
# include("logdensity.jl")
# include("model_macro_passes.jl")
# include("mcmc_chains.jl")
# include("check_gradient.jl")
# May shave off 2/30 seconds or so...
# Would be better to actually work on compile times.
# @static if VERSION > v"1.3.0-rc1"
# end
# include("precompile.jl")
# _precompile_()


end # module
