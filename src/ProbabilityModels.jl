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
const LOCAL_STACK_SIZE = Ref{Int}()
const GLOBAL_PCGs = Vector{PtrPCG{4}}(undef,0)
const NTHREADS = Ref{Int}()

PaddedMatrices.@support_stack_pointer ITPExpectedValue
PaddedMatrices.@support_stack_pointer ∂ITPExpectedValue
PaddedMatrices.@support_stack_pointer HierarchicalCentering
PaddedMatrices.@support_stack_pointer ∂HierarchicalCentering
function __init__()
    NTHREADS[] = Threads.nthreads()
    # Note that 1 GiB == 2^30 == 1 << 30 bytesy
    # Allocates 0.5 GiB per thread for the stack by default.
    # Can be controlled via the environmental variable PROBABILITY_MODELS_STACK_SIZE
    LOCAL_STACK_SIZE[] = if "PROBABILITY_MODELS_STACK_SIZE" ∈ keys(ENV)
        parse(Int, ENV["PROBABILITY_MODELS_STACK_SIZE"])
    else
        1 << 29
    end + VectorizationBase.REGISTER_SIZE - 1 # so we have at least the indicated stack size after REGISTER_SIZE-alignment
    UNALIGNED_POINTER[] = Libc.malloc( NTHREADS[] * LOCAL_STACK_SIZE[] )
    STACK_POINTER_REF[] = PaddedMatrices.StackPointer( VectorizationBase.align(UNALIGNED_POINTER[]) )
    STACK_POINTER_REF[] = threadrandinit!(STACK_POINTER_REF[], GLOBAL_PCGs)
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
function realloc_stack(new_local_stack_size::Integer)
    @warn """You must redefine all probability models. The stack pointers get dereferenced at compile time, and the stack has just been reallocated.
Re-evaluating densities without first recompiling them will likely crash Julia!"""
    LOCAL_STACK_SIZE[] = new_local_stack_size
    UNALIGNED_POINTER[] = Libc.realloc(UNALIGNED_POINTER[], new_local_stack_size + VectorizationBase.REGISTER_SIZE - 1)
    STACK_POINTER_REF[] = PaddedMatrices.StackPointer( VectorizationBase.align(UNALIGNED_POINTER[]) )
    STACK_POINTER_REF[] = threadrandinit!(STACK_POINTER_REF[], GLOBAL_PCGs)
end

rel_error(x, y) = (x - y) / y
function check_gradient(data, a = randn(LogDensityProblems.dimension(data)))
    lp, g = LogDensityProblems.logdensity_and_gradient(data, a)
    for i ∈ eachindex(a)
        aᵢ = a[i]
        step = cbrt(eps(aᵢ))
        a[i] = aᵢ + step
        lp_hi = LogDensityProblems.logdensity(data, a)
        a[i] = aᵢ - step
        lp_lo = LogDensityProblems.logdensity(data, a)
        a[i] = aᵢ
        fd = (lp_hi - lp_lo) / (2step)
        ad = g[i]
        relative_error = rel_error(ad, fd)
        @show (i, ad, fd, relative_error)
        if abs(relative_error) > 1e-5
            fd_f = (lp_hi - lp) / step
            fd_b = (lp - lp_lo) / step
            @show rel_error.(ad, (fd_f, fd_b))
        end
    end
end

end # module
