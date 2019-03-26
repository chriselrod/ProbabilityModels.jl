
function ITPExpectedValue_quote(M::Int, N::Int, T::DataType, (track_β, track_κ), partial)
    # M x N output
    # M total times
    # N total β and κs
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W - 1
    P = (M + Wm1) & ~Wm1
    if !partial || (!track_β && !track_κ)
        return quote
            coefs = MutableFixedSizePaddedVector{$N,$T}(undef)
            final_t = τ[$M]
            @vectorize $T for n ∈ 1:$N
                coefs[n] = β[n] / (one($T) - exp(-κ[n] * final_t))
            end
            μ = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)
            @inbounds for n ∈ 0:$(N-1)
                coefₙ = coefs[n+1]
                κₙ = κ[n+1]
                @vectorize $T for m ∈ 1:$M
                    μ[m + $P*n] = coefₙ * (one($T) - exp(-κₙ * τ[m]))
                end
            end
            ConstantFixedSizePaddedMatrix(μ)
        end
    end
    if track_β && track_κ
        return quote
            ℯκd = MutableFixedSizePaddedVector{$N,$T}(undef)
            denoms = MutableFixedSizePaddedVector{$N,$T}(undef)
            # coefs = MutableFixedSizePaddedVector{$N,$T}(undef)
            final_t = τ[$M]
            @vectorize $T for n ∈ 1:$N
                ℯκdₙ = exp( - κ[n] * final_t)
                ℯκd[n] = ℯκdₙ
                denoms[n] = one($T) / (one($T) - ℯκdₙ)
            end
            μ = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)
            ∂β = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)
            ∂κ = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)
            @inbounds for n ∈ 0:$(N-1)
                denomₙ = denoms[n+1]
                βₙ = β[n+1]
                κₙ = κ[n+1]
                ℯκdₙ = ℯκd[n+1]
                @vectorize $T for m ∈ 1:$M
                    tₘ = τ[m]
                    ℯκt = exp(-κₙ * tₘ)
                    Omℯκt = one($T) - ℯκt
                    ∂βₘ = denomₙ * Omℯκt
                    ∂β[m + $P*n] = ∂βₘ
                    μ[m + $P*n] = βₙ * ∂βₘ
                    ∂κ[m + $P*n] = denomₙ * βₙ * (ℯκt * tₘ + denomₙ * final_t * Omℯκt * ℯκdₙ)
                end
            end
            ConstantFixedSizePaddedMatrix(μ), StructuredMatrices.BlockDiagonalColumnView(ConstantFixedSizePaddedMatrix(∂β)), StructuredMatrices.BlockDiagonalColumnView(ConstantFixedSizePaddedMatrix(∂κ))
        end
    elseif track_β
        return quote
            denoms = MutableFixedSizePaddedVector{$N,$T}(undef)
            # coefs = MutableFixedSizePaddedVector{$N,$T}(undef)
            final_t = τ[$M]
            @vectorize $T for n ∈ 1:$N
                denoms[n] = one($T) / (one($T) - exp( - κ[n] * final_t))
            end
            μ = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)
            ∂β = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)
            @inbounds for n ∈ 0:$(N-1)
                denomₙ = denoms[n+1]
                βₙ = β[n+1]
                κₙ = κ[n+1]
                @vectorize $T for m ∈ 1:$M
                    tₘ = τ[m]
                    ℯκt = exp(-κₙ * tₘ)
                    Omℯκt = one($T) - ℯκt
                    ∂βₘ = denomₙ * Omℯκt
                    ∂β[m + $P*n] = ∂βₘ
                end
            end
            ConstantFixedSizePaddedMatrix(μ), StructuredMatrices.BlockDiagonalColumnView(ConstantFixedSizePaddedMatrix(∂β))
        end
    else # track_κ
        return quote
            ℯκd = MutableFixedSizePaddedVector{$N,$T}(undef)
            denoms = MutableFixedSizePaddedVector{$N,$T}(undef)
            coefs = MutableFixedSizePaddedVector{$N,$T}(undef)
            # coefs = MutableFixedSizePaddedVector{$N,$T}(undef)
            final_t = τ[$M]
            @vectorize $T for n ∈ 1:$N
                ℯκdₙ = exp( - κ[n] * final_t)
                ℯκd[n] = ℯκdₙ
                denom = one($T) / (one($T) - ℯκdₙ)
                denoms[n] = denom
                coefs[n] = denom * β[n]
            end
            μ = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)
            ∂β = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)
            ∂κ = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)
            @inbounds for n ∈ 0:$(N-1)
                denomₙ = denoms[n+1]
                coefₙ = coefs[n+1]
                βₙ = β[n+1]
                κₙ = κ[n+1]
                ℯκdₙ = ℯκd[n+1]
                @vectorize $T for m ∈ 1:$M
                    tₘ = τ[m]
                    ℯκt = exp(-κₙ * tₘ)
                    Omℯκt = one($T) - ℯκt
                    μ[m + $P*n] = coefₙ * Omℯκt
                    ∂κ[m + $P*n] = coefₙ * (ℯκt * tₘ + denomₙ * final_t * Omℯκt * ℯκdₙ)
                end
            end
            ConstantFixedSizePaddedMatrix(μ), StructuredMatrices.BlockDiagonalColumnView(ConstantFixedSizePaddedMatrix(∂κ))
        end
    end
end

@generated function ITPExpectedValue(
            τ::Union{<:PaddedMatrices.AbstractFixedSizePaddedVector{R},<:StructuredMatrices.StaticUnitRange{R}},
            β::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
            κ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T}
        ) where {R,N,T}
    if isa(R, AbstractRange)
        M = length(R)
    else
        M = R
    end
    ITPExpectedValue_quote(M, N, T, (false, false), false)
end
@generated function ∂ITPExpectedValue∂β(
            τ::Union{<:PaddedMatrices.AbstractFixedSizePaddedVector{R},<:StructuredMatrices.StaticUnitRange{R}},
            β::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
            κ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T}
        ) where {R,N,T}
    if isa(R, AbstractRange)
        M = length(R)
    else
        M = R
    end
    ITPExpectedValue_quote(M, N, T, (true, false), true)
end
@generated function ∂ITPExpectedValue∂κ(
            τ::Union{<:PaddedMatrices.AbstractFixedSizePaddedVector{R},<:StructuredMatrices.StaticUnitRange{R}},
            β::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
            κ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T}
        ) where {R,N,T}
    if isa(R, AbstractRange)
        M = length(R)
    else
        M = R
    end
    ITPExpectedValue_quote(M, N, T, (false, true), true)
end
@generated function ∂ITPExpectedValue∂β∂κ(
            τ::Union{<:PaddedMatrices.AbstractFixedSizePaddedVector{R},<:StructuredMatrices.StaticUnitRange{R}},
            β::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
            κ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T}
        ) where {R,N,T}
    if isa(R, AbstractRange)
        M = length(R)
    else
        M = R
    end
    ITPExpectedValue_quote(M, N, T, (true, true), true)
end
