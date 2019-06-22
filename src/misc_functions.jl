
#=
struct StructuredMissingness{S,K,nT,P,PL,T,nTL}
    incr::MutableFixedSizePaddedVector{P,T,PL,PL}
    coef::MutableFixedSizePaddedVector{P,T,PL,PL}
    missingness::Vector{Bool}
    # missingness::BitVector
    times::ConstantFixedSizePaddedVector{nT,T,nTL,nTL}
end



function ITPModel(Y::MultivariateNormalVariate{T}, θ, κ, sm::StructuredMissingness{S,K,nT}) where {S,T,K,nT}
    incr = sm.incr
    coef = sm.coef
    ind_kt = 0
    ind = 0
    missingness = sm.missingness
    times = sm.times
    for k ∈ 1:K
        numerator = expm1(-κ[k] * times[nT])
        for t ∈ 1:nT-1
            ind_kt += 1
            missingness[ind_kt] || continue
            ind += 1
            coef[ind] = numerator / expm1(-κ[k] * times[t])
            incr[ind] = θ[k]
        end
        ind_kt += 1
        missingness[ind_kt] || continue
        ind += 1
        coef[ind] = one(T)
        incr[ind] = θ[k]
    end
    δ = Y.δ
    data = Y.data
    PaddedMatrices.muladd!(δ, coef, data, incr)
    δ
end
function ∂ITPModel(Y::MultivariateNormalVariate{T}, θ, κ, sm::StructuredMissingness{S,K,nT}, ::Val{(true,true)}) where {S,T,K,nT}
    incr = sm.incr
    coef = sm.coef
    ind_kt = 0
    ind = 0
    missingness = sm.missingness
    times = sm.times
    ∂κ = sm.∂κ
    for k ∈ 1:K
        d = times[nT]
        ekd = exp(-κ[k] * d)
        numerator = one(T) - ekd
        ekd *= d
        for t ∈ 1:nT-1
            ind_kt += 1
            missingness[ind_kt] || continue
            ind += 1
            tt = times[t]
            ekt = exp(-κ[k] * tt)
            denominator = 1 / (one(T) - ekt)
            coeft = numerator * denominator
            coef[ind] = coeft
            incr[ind] = θ[k]
            @fastmath ∂κ[ind] = ekd * denominator - coeft * denominator * tt * ekt
        end
        ind_kt += 1
        missingness[ind_kt] || continue
        ind += 1
        coef[ind] = one(T)
        ∂κ[ind] = zero(T)
        incr[ind] = θ[k]
    end
    δ = Y.δ
    data = Y.data
    PaddedMatrices.muladd!(δ, coef, data, incr)
    δ, Reducer{S}(), ReducerWrapper{S}(∂κ)
end
=#

function ITPExpectedValue_quote(M::Int, N::Int, T::DataType, track::NTuple{Nparamargs,Bool}, partial::Bool, sp = false) where {Nparamargs}
    if Nparamargs == 2
        (track_β, track_κ) = track
        track_θ = false
        add_θ = false
    else
        @assert Nparamargs == 3
        (track_β, track_κ, track_θ) = track
        add_θ = true
    end

    #TODO: the first time equals θ, and the last equals θ + β; can skip exponential calculations.

    # M x N output
    # M total times
    # N total β and κs
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W - 1
    P = (M + Wm1) & ~Wm1
    if !partial || (!track_β && !track_κ)
        return_expr = track_θ ? :(μ, ProbabilityModels.Reducer{:row}()) : :μ
        return_expr = sp ? :(sp, $return_expr) : return_expr
        return quote
            @inbounds for n ∈ 0:$(N-1)
                βₙ = β[n+1]
                κₙ = κ[n+1]
                $(add_θ ? :(θₙ = θ[n+1]) : nothing)
                @vectorize $T for m ∈ 1:$M
                    μ[m + $P*n] = $(add_θ ? :(βₙ * (one($T) - exp(-κₙ * τ[m])) + θₙ ) : :(βₙ * (one($T) - exp(-κₙ * τ[m])))  )
                end
            end
            # println("ITP μ")
            # println(μ)
            $return_expr
        end
    end
    if track_β && track_κ
        return_expr = :(μ, StructuredMatrices.BlockDiagonalColumnView(∂β), StructuredMatrices.BlockDiagonalColumnView(∂κ))
        track_θ && push!(return_expr.args, ProbabilityModels.Reducer{:row}())
        return_expr = sp ? :(sp, $return_expr) : return_expr
        return quote
            @inbounds for n ∈ 0:$(N-1)
                βₙ = β[n+1]
                κₙ = κ[n+1]
                $(add_θ ? :(θₙ = θ[n+1]) : nothing)
                @vectorize $T for m ∈ 1:$M
                    tₘ = τ[m]
                    ℯκt = exp(- κₙ * tₘ)
                    βₙℯκt = βₙ * ℯκt
                    Omℯκt = one($T) - ℯκt
                    ∂β[m + $P*n] = Omℯκt
                    μ[m + $P*n] = $(add_θ ? :(βₙ * Omℯκt + θₙ) : :(βₙ - βₙℯκt) )
                    ∂κ[m + $P*n] = βₙℯκt * tₘ
                end
            end
            $return_expr
        end
    elseif track_β
        return_expr = :(μ, StructuredMatrices.BlockDiagonalColumnView(∂β))
        track_θ && push!(return_expr.args, ProbabilityModels.Reducer{:row}())
        return_expr = sp ? :(sp, $return_expr) : return_expr

        return quote
            @inbounds for n ∈ 0:$(N-1)
                βₙ = β[n+1]
                κₙ = κ[n+1]
                $(add_θ ? :(θₙ = θ[n+1]) : nothing)
                @vectorize $T for m ∈ 1:$M
                    tₘ = τ[m]
                    ℯκt = exp(- κₙ * tₘ)
                    Omℯκt = one($T) - ℯκt
                    ∂β[m + $P*n] = Omℯκt
                    μ[m + $P*n] = $(add_θ ? :(βₙ * Omℯκt + θₙ) : :(βₙ * Omℯκt) )
                end
            end
            $return_expr
        end
    else # track_κ
        return_expr = :(μ, StructuredMatrices.BlockDiagonalColumnView(∂κ))
        track_θ && push!(return_expr.args, ProbabilityModels.Reducer{:row}())
        return_expr = sp ? :(sp, $return_expr) : return_expr

        return quote
            @inbounds for n ∈ 0:$(N-1)
                βₙ = β[n+1]
                κₙ = κ[n+1]
                $(add_θ ? :(θₙ = θ[n+1]) : nothing)
                @vectorize $T for m ∈ 1:$M
                    tₘ = τ[m]
                    ℯκt = exp(- κₙ * tₘ)
                    βₙℯκt = βₙ * ℯκt
                    μ[m + $P*n] = $(add_θ ? :(βₙ - βₙℯκt + θₙ) : :(βₙ - βℯκt) )
                    ∂κ[m + $P*n] = βₙℯκt * tₘ
                end
            end
            $return_expr
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
    quote
        μ = MutableFixedSizePaddedMatrix{$M,$N,$T,$M}(undef)
        $(ITPExpectedValue_quote(M, N, T, (false, false), false))
    end
end
@generated function ∂ITPExpectedValue(
            τ::Union{<:PaddedMatrices.AbstractFixedSizePaddedVector{R},<:StructuredMatrices.StaticUnitRange{R}},
            β::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
            κ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
            ::Val{track}
        ) where {R,N,T,track}
    if isa(R, AbstractRange)
        M = length(R)
    else
        M = R
    end
    (track_β, track_κ) = track
    q = quote
        μ = MutableFixedSizePaddedMatrix{$M,$N,$T,$M}(undef)
    end
    if track_β
        push!(q.args, :(∂β = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)))
    end
    if track_κ
        push!(q.args, :(∂κ = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)))
    end
    push!(q.args, ITPExpectedValue_quote(M, N, T, track, true))
    q
end
@generated function ITPExpectedValue(
            τ::Union{<:PaddedMatrices.AbstractFixedSizePaddedVector{R},<:StructuredMatrices.StaticUnitRange{R}},
            β::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
            κ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
            θ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T}
        # ) where {R,T,N}
        ) where {R,N,T}
    if isa(R, AbstractRange)
        M = length(R)
    else
        M = R
    end
    quote
        μ = MutableFixedSizePaddedMatrix{$M,$N,$T,$M}(undef)
        $(ITPExpectedValue_quote(M, N, T, (false, false, false), false))
    end
end
@generated function ∂ITPExpectedValue(
            τ::Union{<:PaddedMatrices.AbstractFixedSizePaddedVector{R},<:StructuredMatrices.StaticUnitRange{R}},
            β::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
            κ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
            θ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
            ::Val{track}
        ) where {R,T,N,track}
        # ) where {R,N,T,track}
    if isa(R, AbstractRange)
        M = length(R)
    else
        M = R
    end
    (track_β, track_κ, track_θ) = track
    q = quote
        μ = MutableFixedSizePaddedMatrix{$M,$N,$T,$M}(undef)
    end
    if track_β
        push!(q.args, :(∂β = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)))
    end
    if track_κ
        push!(q.args, :(∂κ = MutableFixedSizePaddedMatrix{$M,$N,$T}(undef)))
    end
    push!(q.args, ITPExpectedValue_quote(M, N, T, track, true))
    q
end
@generated function ITPExpectedValue(
    sp::StackPointer,
    τ::Union{<:PaddedMatrices.AbstractFixedSizePaddedVector{R},<:StructuredMatrices.StaticUnitRange{R}},
    β::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
    κ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T}
) where {R,N,T}
    if isa(R, AbstractRange)
        M = length(R)
    else
        M = R
    end
    quote
        (sp, μ) = PtrMatrix{$M,$N,$T,$M}(sp)
        $(ITPExpectedValue_quote(M, N, T, (false, false), false, true))
    end
end
@generated function ∂ITPExpectedValue(
    sp::StackPointer,
    τ::Union{<:PaddedMatrices.AbstractFixedSizePaddedVector{R},<:StructuredMatrices.StaticUnitRange{R}},
    β::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
    κ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
    ::Val{track}
) where {R,N,T,track}
    if isa(R, AbstractRange)
        M = length(R)
    else
        M = R
    end
    (track_β, track_κ) = track
    q = quote
        (sp, μ) = PtrMatrix{$M,$N,$T,$M}(sp)
    end
    if track_β
        push!(q.args, :((sp, ∂β) = PtrMatrix{$M,$N,$T}(sp)))
    end
    if track_κ
        push!(q.args, :((sp, ∂κ) = PtrMatrix{$M,$N,$T}(sp)))
    end
    push!(q.args, ITPExpectedValue_quote(M, N, T, track, true, true))
    q
end
@generated function ITPExpectedValue(
    sp::StackPointer,
    τ::Union{<:PaddedMatrices.AbstractFixedSizePaddedVector{R},<:StructuredMatrices.StaticUnitRange{R}},
    β::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
    κ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
    θ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T}
        # ) where {R,T,N}
) where {R,N,T}
    if isa(R, AbstractRange)
        M = length(R)
    else
        M = R
    end
    quote
        (sp, μ) = PtrMatrix{$M,$N,$T,$M}(sp)
        $(ITPExpectedValue_quote(M, N, T, (false, false, false), false, true))
    end
end
@generated function ∂ITPExpectedValue(
    sp::StackPointer,
    τ::Union{<:PaddedMatrices.AbstractFixedSizePaddedVector{R},<:StructuredMatrices.StaticUnitRange{R}},
    β::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
    κ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
    θ::PaddedMatrices.AbstractFixedSizePaddedVector{N,T},
    ::Val{track}
) where {R,T,N,track}
    # ) where {R,N,T,track}
    if isa(R, AbstractRange)
        M = length(R)
    else
        M = R
    end
    (track_β, track_κ, track_θ) = track
    q = quote
        (sp, μ) = PtrMatrix{$M,$N,$T,$M}(sp)
    end
    if track_β
        push!(q.args, :((sp,∂β) = PtrMatrix{$M,$N,$T}(undef)))
    end
    if track_κ
        push!(q.args, :((sp,∂κ) = PtrMatrix{$M,$N,$T}(undef)))
    end
    push!(q.args, ITPExpectedValue_quote(M, N, T, track, true, true))
    q
end

@support_stack_pointer ITPExpectedValue
@support_stack_pointer ∂ITPExpectedValue

struct Domains{S} end
Base.@pure Domains(S::NTuple{N,Int}) where {N} = Domains{S}()
Base.@pure Domains(S::Vararg{Int,N}) where {N} = Domains{S}()
Base.getindex(::Domains{S}, i) where {S} = S[i]
Base.length(::Domains{S}) where {S} = length(S)
Base.eltype(::Domains{S}) where {S} = eltype(S)
@inline Base.iterate(::Domains{S}) where {S} = @inbounds (S[1], 2)
@inline function Base.iterate(::Domains{S}, i) where {S}
    i > length(S) && return nothing
    S[i], i+1
end
Base.Array(::Domains{S}) where {S} = [S...]

function HierarchicalCentering_quote(M::Int, T::DataType, μisvec::Bool, σisvec::Bool, (track_y, track_μ, track_σ), partial)
    μsym = μisvec ? :(μ[m]) : :μ
    σsym = σisvec ? :(σ[m]) : :σ
    if !partial
        return quote
            xout = MutableFixedSizePaddedVector{$M,$T}(undef)
            @vectorize $T for m ∈ 1:$M
                xout[m] = $μsym + $σsym * y[m]
            end
            ConstantFixedSizePaddedVector(xout)
        end
    end
    # partial is true
    q = quote xout = MutableFixedSizePaddedVector{$M,$T}(undef) end
    loop_body = quote xout[m] = $μsym + $σsym * y[m] end
    return_expr = Expr(:tuple, :(ConstantFixedSizePaddedVector(xout)) )
    if track_y
        if σisvec
            push!(return_expr.args, :(Diagonal(σ)) )
        else
            push!(return_expr.args, :(LinearAlgebra.UniformScaling(σ)) )
        end
    end
    if track_μ
        if μisvec
            push!(return_expr.args, :(One()))
        else
            push!(return_expr.args, :(Reducer{true}()))
        end
    end
    if track_σ
        if σisvec
            push!(return_expr.args, :(Diagonal(y)) )
        else
            # push!(return_expr.args, :∂σ)
            # push!(q.args, :(∂σ = zero($T)) )
            # push!(loop_body.args, :(∂σ += y[m]) )
            push!(return_expr.args, :(y) )
        end
    end
    push!(q.args, quote
        @vectorize $T for m ∈ 1:$M
            $loop_body
        end
        $(ProbabilityDistributions.return_expression(return_expr))
    end)
    q
end
"""
This method takes a "Domains" argument, which is a tuple indicating number within each domain.

Depending on whether σ is a scalar or a vector:
∂y is either UniformScaling(σ), or a vector with the same type as y, with replciated σs.

if μ is a vector,
∂μ is Domains{S}()
which, when multiplied by a vector of length = sum(S), reduces the corresponding elements.
Otherwise, is a reducer that sums the vector it is multiplied with.

∂σ is a vector with the same type as y, and...
if σ is a scalar, will dot product on multiplication
if σ is a vector, will do length(σ) mini dot products.
"""
function HierarchicalCentering_quote(
                M::Int, P::Int, T::DataType, μisvec::Bool, σisvec::Bool, S::NTuple{N,Int}, (track_y, track_μ, track_σ)
            ) where {N}
    @assert sum(S) == M
    @assert μisvec | σisvec
    q = quote end
    outtup = Expr(:tuple,)

    ind = 0
    if track_y && σisvec
        ∂yexpr = Expr(:tuple,)
    end
    for (j,s) ∈ enumerate(S)
        for i ∈ 1:s
            ind += 1
            sym = gensym()
            if μisvec && σisvec
                push!(q.args, :($sym = μ[$j] + σ[$j] * y[$ind]))
                track_y && push!(∂yexpr.args, :(σ[$j]))
            elseif μisvec
                push!(q.args, :($sym = μ[$j] + σ * y[$ind]))

            else #if σisvec
                push!(q.args, :($sym = μ + σ[$j] * y[$ind]))
                track_y && push!(∂yexpr.args, :(σ[$j]))
            end
            push!(outtup.args, sym)
        end
    end
    for p ∈ M+1:P
        push!(outtup.args, zero(T))
    end
    push!(q.args, :(xout = ConstantFixedSizePaddedVector{$M,$T,$P}($outtup) ))
    return_expr = Expr(:tuple, :xout )
    if track_y
        if σisvec
            for p ∈ M+1:P
                push!(∂yexpr.args, zero(T))
            end
            push!(q.args, :( ∂y = ConstantFixedSizePaddedVector{$M,$T,$P}($∂yexpr) ))
            push!(return_expr.args, :(LinearAlgebra.Diagonal(∂y)) )
        else
            push!(return_expr.args,  :(LinearAlgebra.UniformScaling(σ)) )
        end
    end
    if track_μ
        push!(return_expr.args, μisvec ? :(Reducer{$S}()) : :(Reducer{true}()) )
    end
    if track_σ
        push!(return_expr.args, σisvec ? :(ReducerWrapper{$S}(y)) :  :y )
    end
    quote
        @fastmath @inbounds begin
            $q
        end
        $(ProbabilityDistributions.return_expression(return_expr))
    end
end

@generated function HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T},
            μ::Union{T, <: AbstractFixedSizePaddedVector{M,T}},
            σ::Union{T, <: AbstractFixedSizePaddedVector{M,T}}
        ) where {M,T}
        # ) where {T,M}

    HierarchicalCentering_quote(M, T, μ <: AbstractFixedSizePaddedVector, σ <: AbstractFixedSizePaddedVector, (false,false,false), false)
end

@generated function ∂HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T},
            μ::Union{T, <: AbstractFixedSizePaddedVector{M,T}},
            σ::Union{T, <: AbstractFixedSizePaddedVector{M,T}},
            ::Val{track}
        # ) where {T,M,track}
        ) where {M,T,track}

    HierarchicalCentering_quote(M, T, μ <: AbstractFixedSizePaddedVector, σ <: AbstractFixedSizePaddedVector, track, true)
end

@generated function HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T,P},
            μ::AbstractFixedSizePaddedVector{N,T},
            σ::AbstractFixedSizePaddedVector{N,T},
            ::Domains{S}
        ) where {M,N,T,P,S}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, true, true, S, (false,false,false))
end
@generated function HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T,P},
            μ::AbstractFixedSizePaddedVector{N,T},
            σ::T,
            ::Domains{S}
        ) where {M,N,T,P,S}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, true, false, S, (false,false,false))
end
@generated function HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T,P},
            μ::T,
            σ::AbstractFixedSizePaddedVector{N,T},
            ::Domains{S}
        ) where {M,N,T,P,S}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, false, true, S, (false,false,false))
end
@generated function ∂HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T,P},
            μ::AbstractFixedSizePaddedVector{N,T},
            σ::AbstractFixedSizePaddedVector{N,T},
            ::Domains{S}, ::Val{track}
        ) where {M,N,S,T,track,P}
        # ) where {M,N,T,S,track,P}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, true, true, S, track)
end
@generated function ∂HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T,P},
            μ::AbstractFixedSizePaddedVector{N,T},
            σ::T,
            ::Domains{S}, ::Val{track}
        ) where {M,N,T,S,track,P}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, true, false, S, track)
end
@generated function ∂HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T,P},
            μ::T,
            σ::AbstractFixedSizePaddedVector{N,T},
            ::Domains{S}, ::Val{track}
        ) where {M,N,T,S,track,P}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, false, true, S, track)
end

#@support_stack_pointer HierarchicalCentering
#@support_stack_pointer ∂HierarchicalCentering

struct ReshapeAdjoint{S} end
function ∂vec(a::AbstractMutableFixedSizePaddedVector{S}) where {S}
    vec(a), ReshapeAdjoint{S}()
end
function Base.:*(A::AbstractMutableFixedSizePaddedArray, ::ReshapeAdjoint{S}) where S
    reshape(A, Val{S}())
end
