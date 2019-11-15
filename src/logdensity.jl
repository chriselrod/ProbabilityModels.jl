


@inline function logdensity(
    ℓ::AbstractProbabilityModel{D},
    θ::AbstractVector{T},
    sptr::StackPointer = STACK_POINTER_REF[]
) where {D,T}
    @boundscheck length(θ) == D || PaddedMatrices.ThrowBoundsError()
    GC.@preserve θ begin
        θptr = PtrVector{D,T,D}(pointer(θ))
        lp = logdensity(ℓ, θptr, sptr)
    end
    lp
end
@inline function logdensity(
    ℓ::AbstractProbabilityModel{D},
    θ::AbstractMutableFixedSizeVector{D,T},
    sptr::StackPointer = STACK_POINTER_REF[]
) where {D,T}
    GC.@preserve θ begin
        θptr = PtrVector{D,T,D}(pointer(θ))
        lp = logdensity(ℓ, θptr, sptr)
    end
    lp
end
@inline function logdensity_and_gradient!(
    ∇::AbstractVector{T},
    ℓ::AbstractProbabilityModel{D},
    θ::AbstractVector{T},
    sptr::StackPointer = STACK_POINTER_REF[]
) where {D,T}
    @boundscheck min(length(∇),length(θ)) < D && PaddedMatrices.ThrowBoundsError()
    GC.@preserve ∇ θ begin
        ∇ptr = PtrVector{D,T,D}(pointer(∇))
        θptr = PtrVector{D,T,D}(pointer(θ))
        lp = logdensity_and_gradient!(∇ptr, ℓ, θptr, sptr)
    end
    lp
end
@inline function logdensity_and_gradient!(
    ∇::AbstractMutableFixedSizeVector{D,T},
    ℓ::AbstractProbabilityModel{D},
    θ::AbstractMutableFixedSizeVector{D,T},
    sptr::StackPointer = STACK_POINTER_REF[]
) where {D,T}
    GC.@preserve ∇ θ begin
        ∇ptr = PtrVector{D,T,D}(pointer(∇));
        θptr = PtrVector{D,T,D}(pointer(θ));
        lp = logdensity_and_gradient!(∇ptr, ℓ, θptr, sptr)
    end
    lp
end
@inline function logdensity_and_gradient(
    sp::StackPointer,
    ℓ::AbstractProbabilityModel{D},
    θ::AbstractMutableFixedSizeVector{D,T}
) where {D,T}
    ∇ = PtrVector{D,T,D}(pointer(sp,T))
    sp += VectorizationBase.align(D*sizeof(T))
    sp, (logdensity_and_gradient!(∇, ℓ, θ, sp), ∇)
end
@inline function logdensity_and_gradient(
    sp::StackPointer,
    ℓ::AbstractProbabilityModel{D},
    θ::AbstractVector{T}
) where {D,T}
    @boundscheck length(θ) == D || PaddedMatrices.ThrowBoundsError()
    ∇ = PtrVector{D,T,D}(pointer(sp,T))
    sp += VectorizationBase.align(D*sizeof(T))
    sp, (logdensity_and_gradient!(∇, ℓ, θ, sp), ∇)
end
@inline function logdensity_and_gradient(
    l::AbstractProbabilityModel{D}, θ, sptr::StackPointer = STACK_POINTER_REF[]
) where {D}
    ∇ = PaddedMatrices.mutable_similar(θ)
    logdensity_and_gradient!(∇, l, θ, sptr), ∇
end

