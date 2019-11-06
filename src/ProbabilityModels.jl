module ProbabilityModels

using MacroTools, Mmap,
    VectorizationBase, SIMDPirates, SLEEFPirates,
    LoopVectorization, VectorizedRNG,
    PaddedMatrices, StructuredMatrices,
    DistributionParameters, ProbabilityDistributions,
    ReverseDiffExpressions, StackPointers

using VectorizedRNG: AbstractPCG, PtrPCG
using MacroTools: postwalk, prewalk, @capture, @q

using ReverseDiffExpressionsBase:
    RESERVED_INCREMENT_SEED_RESERVED!,
    initialize_target, uninitialized,
    ∂mul, ∂getindex

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

export @model, MCMCChainSummary,
    logdensity, logdensity_and_gradient,
    logdensity_and_gradient!, stackpointer

# function logdensity_and_gradient! end

# const UNALIGNED_POINTER = Ref{Ptr{Cvoid}}()
const MMAP = Ref{Matrix{UInt8}}()
const STACK_POINTER_REF = Ref{StackPointer}()
const LOCAL_STACK_SIZE = Ref{Int}()
const GLOBAL_PCGs = Vector{PtrPCG{4}}(undef,0)
const NTHREADS = Ref{Int}()


# LogDensityProblems.capabilities(::Type{<:AbstractProbabilityModel}) = LogDensityProblems.LogDensityOrder{1}()
# `@inline` so that we can avoid the allocation for tuple creation
# additionally, the logdensity(_and_gradient!) method itself will not in general
# be inlined. There is only a single method (on PtrVectors) defined,
# so that the functions will only have to be compiled once per AbstractProbabilityModel.


verbose_models() = false


include("logdensity.jl")
include("model_macro_passes.jl")
include("mcmc_chains.jl")
include("rng.jl")
include("check_gradient.jl")


function __init__()
    NTHREADS[] = Threads.nthreads()
    # Note that 1 GiB == 2^30 == 1 << 30 bytesy
    # Allocates 0.5 GiB per thread for the stack by default.
    # Can be controlled via the environmental variable PROBABILITY_MODELS_STACK_SIZE
    LOCAL_STACK_SIZE[] = if "PROBABILITY_MODELS_STACK_SIZE" ∈ keys(ENV)
        parse(Int, ENV["PROBABILITY_MODELS_STACK_SIZE"])
    else
        1 << 30
    end + VectorizationBase.REGISTER_SIZE - 1 # so we have at least the indicated stack size after REGISTER_SIZE-alignment
    # UNALIGNED_POINTER[] = Libc.malloc( NTHREADS[] * LOCAL_STACK_SIZE[] )
    MMAP[] = Mmap.mmap(Matrix{UInt8}, LOCAL_STACK_SIZE[], NTHREADS[])
    STACK_POINTER_REF[] = PaddedMatrices.StackPointer(
        VectorizationBase.align(Base.unsafe_convert(Ptr{Cvoid}, pointer(MMAP[])))
    )
    # STACK_POINTER_REF[] = PaddedMatrices.StackPointer( VectorizationBase.align(UNALIGNED_POINTER[]) )
    STACK_POINTER_REF[] = threadrandinit!(STACK_POINTER_REF[], GLOBAL_PCGs)
    # @eval const STACK_POINTER = STACK_POINTER_REF[]
    # @eval const GLOBAL_WORK_BUFFER = Vector{Vector{UInt8}}(Base.Threads.nthreads())
    # Threads.@threads for i ∈ eachindex(GLOBAL_WORK_BUFFER)
    #     GLOBAL_WORK_BUFFER[i] = Vector{UInt8}(0)
    # end
#    for m ∈ (:ITPExpectedValue, :∂ITPExpectedValue)
#        push!(PaddedMatrices.STACK_POINTER_SUPPORTED_METHODS, m)
    #    end
    
end
function realloc_stack(new_local_stack_size::Integer)
    @warn """You must redefine all probability models. The stack pointers get dereferenced at compile time, and the stack has just been reallocated.
Re-evaluating densities without first recompiling them will likely crash Julia!"""
    LOCAL_STACK_SIZE[] = new_local_stack_size
    MMAP[] = Mmap.mmap(Matrix{UInt8}, LOCAL_STACK_SIZE[], NTHREADS[])
    STACK_POINTER_REF[] = PaddedMatrices.StackPointer(
        VectorizationBase.align(Base.unsafe_convert(Ptr{Cvoid}, pointer(MMAP[])))
    )
    STACK_POINTER_REF[] = threadrandinit!(STACK_POINTER_REF[], GLOBAL_PCGs)
end
stackpointer() = STACK_POINTER_REF[] + (Threads.threadid() - 1) * LOCAL_STACK_SIZE[]

end # module
