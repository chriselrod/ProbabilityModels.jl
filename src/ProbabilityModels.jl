module ProbabilityModels

using   MacroTools, DiffRules, Parameters,
        VectorizationBase, SIMDPirates, LoopVectorization, SLEEFPirates,
        PaddedMatrices, StructuredMatrices,
        DistributionParameters, ProbabilityDistributions,
        DynamicHMC, LogDensityProblems,
        Random, VectorizedRNG,
        LinearAlgebra, Statistics, Distributed,
        MCMCChains

using VectorizedRNG: AbstractPCG, PtrPCG
using LoopVectorization: @vvectorize
import MacroTools: postwalk, prewalk, @capture, @q
import PaddedMatrices: RESERVED_INCREMENT_SEED_RESERVED, RESERVED_DECREMENT_SEED_RESERVED,
    RESERVED_MULTIPLY_SEED_RESERVED, RESERVED_NMULTIPLY_SEED_RESERVED,
    AbstractFixedSizePaddedVector, AbstractMutableFixedSizePaddedVector,
    AbstractMutableFixedSizePaddedArray,
    StackPointer


export @model#, NUTS_init_tune_mcmc_default, NUTS_init_tune_distributed, sample_cov, sample_mean

function logdensity_and_gradient! end

abstract type AbstractProbabilityModel{D} end# <: LogDensityProblems.AbstractLogDensityProblem end
LogDensityProblems.dimension(::AbstractProbabilityModel{D}) where {D} = D
LogDensityProblems.capabilities(::Type{<:AbstractProbabilityModel}) = LogDensityProblems.LogDensityOrder{1}()
# `@inline` so that we can avoid the allocation for tuple creation
# additionally, the logdensity(_and_gradient!) method itself will not in general
# be inlined. There is only a single method (on PtrVectors) defined,
# so that the functions will only have to be compiled once per AbstractProbabilityModel.
@inline function LogDensityProblems.logdensity(l::AbstractProbabilityModel{D}, θ::AbstractVector{T}) where {D,T}
    @boundscheck length(θ) == D || PaddedMatrices.ThrowBoundsError()
    GC.@preserve θ begin
        θptr = PtrVector{D,T,D,D}(pointer(θ))
        lp = LogDensityProblems.logdensity(l, θptr)
    end
    lp
end
@inline function LogDensityProblems.logdensity(l::AbstractProbabilityModel{D}, θ::AbstractMutableFixedSizePaddedVector{D,T}) where {D,T}
    GC.@preserve θ begin
        θptr = PtrVector{D,T,D,D}(pointer(θ))
        lp = LogDensityProblems.logdensity(l, θptr)
    end
    lp
end
@inline function logdensity_and_gradient!(
    ∇::AbstractVector{T},
    l::AbstractProbabilityModel{D},
    θ::AbstractVector{T}
) where {D,T}
    @boundscheck max(length(∇),length(θ)) > D && PaddedMatrices.ThrowBoundsError()
    GC.@preserve ∇ θ begin
        ∇ptr = PtrVector{D,T,D,D}(pointer(∇));
        θptr = PtrVector{D,T,D,D}(pointer(θ));
        lp = logdensity_and_gradient!(∇ptr, l, θptr)
    end
    lp
end
@inline function logdensity_and_gradient!(
    ∇::AbstractMutableFixedSizePaddedVector{D,T},
    l::AbstractProbabilityModel{D},
    θ::AbstractMutableFixedSizePaddedVector{D,T}
) where {D,T}
    GC.@preserve ∇ θ begin
        ∇ptr = PtrVector{D,T,D,D}(pointer(∇));
        θptr = PtrVector{D,T,D,D}(pointer(θ));
        lp = logdensity_and_gradient!(∇ptr, l, θptr)
    end
    lp
end
@inline function LogDensityProblems.logdensity_and_gradient(
    sp::StackPointer,
    l::AbstractProbabilityModel{D},
    θ::AbstractMutableFixedSizePaddedVector{D,T}
) where {D,T}
    ∇ = PtrVector{D,T,D,D}(pointer(sp,T))
    sp += VectorizationBase.align(D*sizeof(T))
    sp, (logdensity_and_gradient!(∇, l, θ, sp), ∇)
end
@inline function LogDensityProblems.logdensity_and_gradient(
    sp::StackPointer,
    l::AbstractProbabilityModel{D},
    θ::AbstractVector{T}
) where {D,T}
    @boundscheck length(θ) > D && PaddedMatrices.ThrowBoundsError()
    ∇ = PtrVector{D,T,D,D}(pointer(sp,T))
    sp += VectorizationBase.align(D*sizeof(T))
    sp, (logdensity_and_gradient!(∇, l, θ, sp), ∇)
end
@inline function LogDensityProblems.logdensity_and_gradient(
    l::AbstractProbabilityModel{D}, θ
) where {D}
    ∇ = PaddedMatrices.mutable_similar(θ)
    logdensity_and_gradient!(∇, l, θ), ∇
end


verbose_models() = false


include("adjoints.jl")
include("misc_functions.jl")
include("special_diff_rules.jl")
include("reverse_autodiff_passes.jl")
include("model_macro_passes.jl")
include("mcmc_chains.jl")
include("rng.jl")
#include("dynamic_hmc_interface.jl")

const UNALIGNED_POINTER = Ref{Ptr{Cvoid}}()
const STACK_POINTER_REF = Ref{StackPointer}()
const LOCAL_STACK_SIZE = Ref{Int64}()
const GLOBAL_PCGs = Vector{PtrPCG{4}}(undef,0)


PaddedMatrices.@support_stack_pointer ITPExpectedValue
PaddedMatrices.@support_stack_pointer ∂ITPExpectedValue
PaddedMatrices.@support_stack_pointer HierarchicalCentering
PaddedMatrices.@support_stack_pointer ∂HierarchicalCentering
function __init__()
    # Note that 1 GiB == 2^30 == 1 << 30 bytesy
    # Allocates 0.5 GiB per thread for the stack by default.
    # Can be controlled via the environmental variable PROBABILITY_MODELS_STACK_SIZE
    LOCAL_STACK_SIZE[] = get(ENV, "PROBABILITY_MODELS_STACK_SIZE", 1 << 29 )
    UNALIGNED_POINTER[] = Libc.malloc( Threads.nthreads() * nprocs() * LOCAL_STACK_SIZE[] )
    STACK_POINTER_REF[] = PaddedMatrices.StackPointer( VectorizationBase.align(UNALIGNED_POINTER[]) )
    threadrandinit!(GLOBAL_PCGs)
    # @eval const STACK_POINTER = STACK_POINTER_REF[]
    # @eval const GLOBAL_WORK_BUFFER = Vector{Vector{UInt8}}(Base.Threads.nthreads())
    # Threads.@threads for i ∈ eachindex(GLOBAL_WORK_BUFFER)
    #     GLOBAL_WORK_BUFFER[i] = Vector{UInt8}(0)
    # end
#    for m ∈ (:ITPExpectedValue, :∂ITPExpectedValue)
#        push!(PaddedMatrices.STACK_POINTER_SUPPORTED_METHODS, m)
#    end
    for m ∈ (:ITPExpectedValue, :∂ITPExpectedValue, :HierarchicalCentering, :∂HierarchicalCentering)
        push!(PaddedMatrices.STACK_POINTER_SUPPORTED_METHODS, m)
    end
end
function realloc_stack(n::Integer)
    @warn """You must redefine all probability models. The stack pointers get dereferenced at compile time, and the stack has just been reallocated.
Re-evaluating densities without first recompiling them will likely crash Julia!"""
    LOCAL_STACK_SIZE[] = n
    UNALIGNED_POINTER[] = Libc.realloc(UNALIGNED_POINTER[], n)
    STACK_POINTER_REF[] = VectorizationBase.align(UNALIGNED_POINTER[])
end
    

end # module
