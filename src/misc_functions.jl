
# a is emax, b is ed50, c is dose
function emax_dose_response_quote(M::Union{Int,Symbol}, T, isvec::NTuple{3,Bool}, track::NTuple{3,Bool}, sp::Bool)# partial::Bool, sp::Bool)
    #    f(a,b,c) =  a*c   / (b + c)
    # ∂f∂a(a,b,c) =    c   / (b + c)
    # ∂f∂b(a,b,c) = -a*c   / (b + c)^2
    # ∂f∂c(a,b,c) =  a * b / (b + c)^2
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    tracka, trackb, trackc = track
    aisvec, bisvec, cisvec = isvec
    need_loop = aisvec | bisvec | cisvec
    partial = tracka | trackb | trackc
    aexpr = aisvec ? :(a[i]) : :a
    bexpr = bisvec ? :(b[i]) : :b
    cexpr = cisvec ? :(c[i]) : :c
    head = quote end
    body = quote
        d = one($T) / ($bexpr + $cexpr)
        ∂f∂a = $cexpr * d
        f = $aexpr * ∂f∂a
    end
    return_f = need_loop ? :vf : :f
    return_expr = partial ? Expr(:tuple, return_f) : return_f
    if partial
        if trackb 
            if need_loop && !bisvec
                push!(body.args, :(∂f∂b += -d*f))
            else
                push!(body.args, :(∂f∂b = -d*f))
            end
        end
        if trackc
            if need_loop && !cisvec
                push!(body.args, :(∂f∂c += a*b*d*d))
            else
                push!(body.args, :(∂f∂c = a*b*d*d))
            end
        end
        for (sym, isvec, track) in ( (:∂f∂a,aisvec,tracka),(:∂f∂b,bisvec,trackb),(:∂f∂c,cisvec,trackc) )
            track || continue
            if need_loop && !isvec
                push!(head.args, :($sym = zero($T)))
            elseif isvec
                vsym = Symbol(:v, sym)
                push!(head.args, PaddedMatrices.pointer_vector_expr( vsym, M, T, sp, :sptr ) )
                push!(body.args, :($vsym[i] = $sym))
                push!(return_expr.args, vsym)
            end
            isvec || push!(return_expr.args, sym)
        end
    end
    body = macroexpand(Base, :(@fastmath $body))
    final_ret = sp ? :((sptr, $return_expr)) : return_expr
    if need_loop
        quote
            $head
            @inbounds @simd for i in 1:$M
                $body
            end
            $final_ret
        end
    else
        quote
            $head
            $body
            $final_ret
        end
    end
end

@generated function emax_dose_response(
    a::Union{T,<:AbstractVector{T}},
    b::Union{T,<:AbstractVector{T}},
    c::Union{T,<:AbstractVector{T}},
    ::Val{track} = Val{(false,false,false)}()
) where {T, track}
    (L,isvec) = if a <: AbstractFixedSizePaddedVector
        full_length(a), true
    elseif b <: AbstractFixedSizePaddedVector
        full_length(b), true
    elseif c <: AbstractFixedSizePaddedVector
        full_length(c), true
    elseif a != T
        :(length(a)), true
    elseif b != T
        :(length(b)), true
    elseif c != T
        :(length(c)), true
    else
        1, false
    end
    emax_dose_response_quote(L, T, isvec, track, false)
end
@generated function emax_dose_response(
    sptr::StackPointer,
    a::Union{T,<:AbstractVector{T}},
    b::Union{T,<:AbstractVector{T}},
    c::Union{T,<:AbstractVector{T}},
    ::Val{track} = Val{(false,false,false)}()
) where {T, track}
    (L,isvec) = if a <: AbstractFixedSizePaddedVector
        full_length(a), true
    elseif b <: AbstractFixedSizePaddedVector
        full_length(b), true
    elseif c <: AbstractFixedSizePaddedVector
        full_length(c), true
    elseif a != T
        :(length(a)), true
    elseif b != T
        :(length(b)), true
    elseif c != T
        :(length(c)), true
    else
        1, false
    end
    emax_dose_response_quote(L, T, isvec, track, true)
end


function ITPExpectedValue_quote(
    M::Int, N::Int, T, @nospecialize(track::NTuple{<:Any,Bool}), partial::Bool, sp::Bool = false, P::Int = (M + Wm1) & ~Wm1
)
    Nparamargs = length(track)
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
            vone = vbroadcast(Vec{$W,$T},one($T))
            @inbounds for n ∈ 0:$(N-1)
                βₙ = β[n+1]
                nκₙ = -κ[n+1]
                $(add_θ ? :(θₙ = θ[n+1]) : nothing)
                $(macroexpand(ProbabilityModels, quote
                              LoopVectorization.@vvectorize $T for m ∈ 1:$M
                              tₘ = τ[m]
                              ℯκt = exp( nκₙ * tₘ)
                              βₙℯκt = βₙ * ℯκt
#=                              @show typeof(vone)
                              @show typeof(ℯκt)
                              @show vone
                              @show ℯκt=#
                              Omℯκt = vone - ℯκt
                              ∂β[m + $P*n] = Omℯκt
                              μ[m + $P*n] = $(add_θ ? :(βₙ * Omℯκt + θₙ) : :(βₙ - βₙℯκt) )
                              ∂κ[m + $P*n] = βₙℯκt * tₘ
                              end
                end))
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
        $(ITPExpectedValue_quote(M, N, T, (false, false), false, false, M))
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
        push!(q.args, :(∂β = MutableFixedSizePaddedMatrix{$M,$N,$T,$M}(undef)))
    end
    if track_κ
        push!(q.args, :(∂κ = MutableFixedSizePaddedMatrix{$M,$N,$T,$N}(undef)))
    end
    push!(q.args, ITPExpectedValue_quote(M, N, T, track, true,false,M))
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
        $(ITPExpectedValue_quote(M, N, T, (false, false, false), false,false,M))
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
        push!(q.args, :(∂β = MutableFixedSizePaddedMatrix{$M,$N,$T,$M}(undef)))
    end
    if track_κ
        push!(q.args, :(∂κ = MutableFixedSizePaddedMatrix{$M,$N,$T,$M}(undef)))
    end
    push!(q.args, ITPExpectedValue_quote(M, N, T, track, true,false,M))
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
        $(ITPExpectedValue_quote(M, N, T, (false, false), false, true, M))
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
        push!(q.args, :((sp, ∂β) = PtrMatrix{$M,$N,$T,$M}(sp)))
    end
    if track_κ
        push!(q.args, :((sp, ∂κ) = PtrMatrix{$M,$N,$T,$M}(sp)))
    end
    push!(q.args, ITPExpectedValue_quote(M, N, T, track, true, true, M))
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
        $(ITPExpectedValue_quote(M, N, T, (false, false, false), false, true, M))
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
#    @show τ, β, κ, θ, track
    (track_β, track_κ, track_θ) = track
    q = quote
#        @show reinterpret(Int,pointer(sp)) - reinterpret(Int,pointer(ProbabilityModels.STACK_POINTER))
        (sp, μ) = PtrMatrix{$M,$N,$T,$M}(sp)
    end
    if track_β
        push!(q.args, :((sp,∂β) = PtrMatrix{$M,$N,$T,$M}(sp)))
    end
    if track_κ
        push!(q.args, :((sp,∂κ) = PtrMatrix{$M,$N,$T,$M}(sp)))
    end
#    push!(q.args, :(@show reinterpret(Int,pointer(sp)) - reinterpret(Int,pointer(ProbabilityModels.STACK_POINTER))))
    push!(q.args, ITPExpectedValue_quote(M, N, T, track, true, true, M))
    q
end


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

function HierarchicalCentering_quote(M::Int, @nospecialize(T), μisvec::Bool, σisvec::Bool, (track_y, track_μ, track_σ)::NTuple{3,Bool}, partial, sp::Bool)
    μsym = μisvec ? :(μ[m]) : :μ
    σsym = σisvec ? :(σ[m]) : :σ
    if !partial
        if sp
            return quote
                xout = PtrVector{$M,$T,$M,$M}(pointer(sp,$T))
                @vvectorize $T for m ∈ 1:$M
                    xout[m] = $μsym + $σsym * y[m]
                end
                sp + $(sizeof(T)*M), xout
            end
        else
            return quote
                xout = MutableFixedSizePaddedVector{$M,$T}(undef)
                @vvectorize $T for m ∈ 1:$M
                    xout[m] = $μsym + $σsym * y[m]
                end
                ConstantFixedSizePaddedVector(xout)
            end
        end
    end
    # partial is true
    if sp
        q = quote
            xout = PtrVector{$M,$T,$M,$M}(pointer(sp,$T))
            sp += $(sizeof(T)*M)
        end
        return_expr = Expr(:tuple, :xout )
    else
        q = quote xout = MutableFixedSizePaddedVector{$M,$T}(undef) end
        return_expr = Expr(:tuple, :(ConstantFixedSizePaddedVector(xout)) )
    end
    loop_body = quote xout[m] = $μsym + $σsym * y[m] end
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
          $(ProbabilityDistributions.return_expression(return_expr, sp))
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
    M::Int, P::Int, @nospecialize(T), μisvec::Bool, σisvec::Bool, @nospecialize(S::NTuple{<:Any,Int}), (track_y, track_μ, track_σ)::NTuple{3,Bool}, sp::Bool
)
    N = length(S)
    @assert sum(S) == M
    @assert μisvec | σisvec
    q = quote end
    if sp
        push!(q.args, :(xout = PtrVector{$M,$T,$M,$M}(pointer(sp,$T))))
        push!(q.args, :(sp += $(sizeof(T)*M)))
        if track_y && σisvec
            push!(q.args, :(∂y = PtrVector{$M,$T,$M,$M}(pointer(sp,$T))))
            push!(q.args, :(sp += $(sizeof(T)*M)))
        end
    else
        outtup = Expr(:tuple,)
        if track_y && σisvec
            ∂yexpr = Expr(:tuple,)
        end
    end

    ind = 0
    for (j,s) ∈ enumerate(S)
        for i ∈ 1:s
            ind += 1
            sym = gensym()
            if sp
                if μisvec && σisvec
                    push!(q.args, :($sym = μ[$j] + σ[$j] * y[$ind]))
                    track_y && push!(q.args, :(∂y[$ind] = σ[$j]))
                elseif μisvec
                    push!(q.args, :($sym = μ[$j] + σ * y[$ind]))
                else #if σisvec
                    push!(q.args, :($sym = μ + σ[$j] * y[$ind]))
                    track_y && push!(q.args, :(∂y[$ind] = σ[$j]))
                end
                push!(q.args, :(xout[$ind] = $sym))
            else
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
    end
    if !sp
        for p ∈ M+1:P
            push!(outtup.args, zero(T))
        end
        push!(q.args, :(xout = ConstantFixedSizePaddedVector{$M,$T,$P}($outtup) ))
    end
    return_expr = Expr(:tuple, :xout )
    if track_y
        if σisvec
            if !sp
                for p ∈ M+1:P
                    push!(∂yexpr.args, zero(T))
                end
                push!(q.args, :( ∂y = ConstantFixedSizePaddedVector{$M,$T,$P}($∂yexpr) ))
            end
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
        $(ProbabilityDistributions.return_expression(return_expr,sp))
    end
end

@generated function HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T},
            μ::Union{T, <: AbstractFixedSizePaddedVector{M,T}},
            σ::Union{T, <: AbstractFixedSizePaddedVector{M,T}}
        ) where {M,T}
        # ) where {T,M}

    HierarchicalCentering_quote(M, T, μ <: AbstractFixedSizePaddedVector, σ <: AbstractFixedSizePaddedVector, (false,false,false), false, false)
end

@generated function ∂HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T},
            μ::Union{T, <: AbstractFixedSizePaddedVector{M,T}},
            σ::Union{T, <: AbstractFixedSizePaddedVector{M,T}},
            ::Val{track}
        # ) where {T,M,track}
        ) where {M,T,track}

    HierarchicalCentering_quote(M, T, μ <: AbstractFixedSizePaddedVector, σ <: AbstractFixedSizePaddedVector, track, true, false)
end

@generated function HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T,P},
            μ::AbstractFixedSizePaddedVector{N,T},
            σ::AbstractFixedSizePaddedVector{N,T},
            ::Domains{S}
        ) where {M,N,T,P,S}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, true, true, S, (false,false,false),false)
end
@generated function HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T,P},
            μ::AbstractFixedSizePaddedVector{N,T},
            σ::T,
            ::Domains{S}
        ) where {M,N,T,P,S}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, true, false, S, (false,false,false),false)
end
@generated function HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T,P},
            μ::T,
            σ::AbstractFixedSizePaddedVector{N,T},
            ::Domains{S}
        ) where {M,N,T,P,S}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, false, true, S, (false,false,false),false)
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
    HierarchicalCentering_quote(M, P, T, true, true, S, track,false)
end
@generated function ∂HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T,P},
            μ::AbstractFixedSizePaddedVector{N,T},
            σ::T,
            ::Domains{S}, ::Val{track}
        ) where {M,N,T,S,track,P}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, true, false, S, track,false)
end
@generated function ∂HierarchicalCentering(
            #x::AbstractFixedSizePaddedVector{M,T},
            y::AbstractFixedSizePaddedVector{M,T,P},
            μ::T,
            σ::AbstractFixedSizePaddedVector{N,T},
            ::Domains{S}, ::Val{track}
        ) where {M,N,T,S,track,P}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, false, true, S, track,false)
end


@generated function HierarchicalCentering(
    sp::StackPointer,
    y::AbstractFixedSizePaddedVector{M,T},
    μ::Union{T, <: AbstractFixedSizePaddedVector{M,T}},
    σ::Union{T, <: AbstractFixedSizePaddedVector{M,T}}
) where {M,T}
    # ) where {T,M}
    HierarchicalCentering_quote(M, T, μ <: AbstractFixedSizePaddedVector, σ <: AbstractFixedSizePaddedVector, (false,false,false), false, true)
end

@generated function ∂HierarchicalCentering(
    sp::StackPointer,
    y::AbstractFixedSizePaddedVector{M,T},
    μ::Union{T, <: AbstractFixedSizePaddedVector{M,T}},
    σ::Union{T, <: AbstractFixedSizePaddedVector{M,T}},
    ::Val{track}
    # ) where {T,M,track}
) where {M,T,track}
    HierarchicalCentering_quote(M, T, μ <: AbstractFixedSizePaddedVector, σ <: AbstractFixedSizePaddedVector, track, true, true)
end

@generated function HierarchicalCentering(
    sp::StackPointer,
    y::AbstractFixedSizePaddedVector{M,T,P},
    μ::AbstractFixedSizePaddedVector{N,T},
    σ::AbstractFixedSizePaddedVector{N,T},
    ::Domains{S}
) where {M,N,T,P,S}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, true, true, S, (false,false,false),true)
end
@generated function HierarchicalCentering(
    sp::StackPointer,
    y::AbstractFixedSizePaddedVector{M,T,P},
    μ::AbstractFixedSizePaddedVector{N,T},
    σ::T,
    ::Domains{S}
) where {M,N,T,P,S}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, true, false, S, (false,false,false),true)
end
@generated function HierarchicalCentering(
    sp::StackPointer,
    y::AbstractFixedSizePaddedVector{M,T,P},
    μ::T,
    σ::AbstractFixedSizePaddedVector{N,T},
    ::Domains{S}
) where {M,N,T,P,S}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, false, true, S, (false,false,false),true)
end
@generated function ∂HierarchicalCentering(
    sp::StackPointer,
    y::AbstractFixedSizePaddedVector{M,T,P},
    μ::AbstractFixedSizePaddedVector{N,T},
    σ::AbstractFixedSizePaddedVector{N,T},
    ::Domains{S}, ::Val{track}
) where {M,N,S,T,track,P}
    # ) where {M,N,T,S,track,P}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, true, true, S, track, true)
end
@generated function ∂HierarchicalCentering(
    sp::StackPointer,
    y::AbstractFixedSizePaddedVector{M,T,P},
    μ::AbstractFixedSizePaddedVector{N,T},
    σ::T,
    ::Domains{S}, ::Val{track}
) where {M,N,T,S,track,P}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, true, false, S, track,true)
end
@generated function ∂HierarchicalCentering(
    sp::StackPointer,
    y::AbstractFixedSizePaddedVector{M,T,P},
    μ::T,
    σ::AbstractFixedSizePaddedVector{N,T},
    ::Domains{S}, ::Val{track}
) where {M,N,T,S,track,P}
    @assert length(S) == N
    HierarchicalCentering_quote(M, P, T, false, true, S, track,true)
end

#@support_stack_pointer HierarchicalCentering
#@support_stack_pointer ∂HierarchicalCentering

struct ReshapeAdjoint{S} end
function ∂vec(a::AbstractMutableFixedSizePaddedArray{S}) where {S}
    vec(a), ReshapeAdjoint{S}()
end
function Base.:*(A::AbstractMutableFixedSizePaddedArray, ::ReshapeAdjoint{S}) where S
    reshape(A, Val{S}())
end
