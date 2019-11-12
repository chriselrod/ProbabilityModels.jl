module ProbabilityModels

using MacroTools, LinearAlgebra, VectorizationBase,
    SIMDPirates, SLEEFPirates,
    LoopVectorization, VectorizedRNG,
    PaddedMatrices, StructuredMatrices,
    DistributionParameters, ProbabilityDistributions,
    ReverseDiffExpressions, StackPointers

using MacroTools: postwalk, prewalk, @capture, @q

using ReverseDiffExpressionsBase:
    RESERVED_INCREMENT_SEED_RESERVED!,
    initialize_target, uninitialized,
    ∂mul, ∂getindex, alloc_adjoint

using PaddedMatrices:
    AbstractFixedSizeVector,
    AbstractMutableFixedSizeVector,
    AbstractMutableFixedSizeArray

import QuasiNewtonMethods:
    AbstractProbabilityModel,
    logdensity,
    logdensity_and_gradient!,
    dimension

import DistributionParameters: parameter_names
import MCMCChainSummaries: MCMCChainSummary
using DistributionParameters: Bounds
using InplaceDHMC: STACK_POINTER_REF, LOCAL_STACK_SIZE, NTHREADS,
    mcmc_with_warmup, threaded_mcmc

export @model, MCMCChainSummary,
    logdensity, logdensity_and_gradient,
    logdensity_and_gradient!, Bounds,
    mcmc_with_warmup, threaded_mcmc
    
"""
For debugging, you can set the verbosity level to 0 (default), 1, or 2
ProbabilityModels.verbose_models() = 1
and then recompile your model.
verbose_models() == 0 will not print messages.
verbose_models() > 0 will print generated function output.
verbose_models() > 1 will print output while evaluating the logdensity (and gradients).
"""
verbose_models() = 0

include("logdensity.jl")
include("model_macro_passes.jl")
include("mcmc_chains.jl")
include("check_gradient.jl")

end # module
