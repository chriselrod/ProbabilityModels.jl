struct One end
# This should be the only method I have to define.
@inline Base.:*(a, ::One) = a
# But I'll define this one too. Would it be better not to, so that we get errors
# if the seed is for some reason multiplied on the right?
@inline Base.:*(::One, a) = a

@inline RESERVED_INCREMENT_SEED_RESERVED(a::One, b, c) = RESERVED_INCREMENT_SEED_RESERVED(b, c)
@inline RESERVED_DECREMENT_SEED_RESERVED(a::One, b, c) = RESERVED_DECREMENT_SEED_RESERVED(b, c)

@inline RESERVED_MULTIPLY_SEED_RESERVED(a::One, b) = b
@inline RESERVED_NMULTIPLY_SEED_RESERVED(a::One, b) = RESERVED_NMULTIPLY_SEED_RESERVED(b)

@inline RESERVED_INCREMENT_SEED_RESERVED(a, b::One, c) = RESERVED_INCREMENT_SEED_RESERVED(a, c)
@inline RESERVED_DECREMENT_SEED_RESERVED(a, b::One, c) = RESERVED_DECREMENT_SEED_RESERVED(a, c)

@inline RESERVED_MULTIPLY_SEED_RESERVED(a, b::One) = a
@inline RESERVED_NMULTIPLY_SEED_RESERVED(a, b::One) = RESERVED_NMULTIPLY_SEED_RESERVED(a)

@inline RESERVED_INCREMENT_SEED_RESERVED(a::One, b::One, c) = c
@inline RESERVED_DECREMENT_SEED_RESERVED(a::One, b::One, c) = - c

@inline RESERVED_MULTIPLY_SEED_RESERVED(a::One, b::One) = One()
# @inline RESERVED_NMULTIPLY_SEED_RESERVED(a::One, b::One) = NegativeOne()

abstract type AbstractReducer{T} end
struct Reducer{T} <: AbstractReducer{T} end
struct ReducerWrapper{T,V} <: AbstractReducer{T}
    data::V
end
@inline ReducerWrapper{T}(data::V) where {T,V} = ReducerWrapper{T,V}(data)

Base.:*(a, ::Reducer{true}) = sum(a)
# Reducer{:row} reduces rows
function Base.:*(A::AbstractFixedSizePaddedMatrix{M,N,T,P}, ::Reducer{:row}) where {M,N,T,P}
    reduction = MutableFixedSizePaddedVector{N,T}(undef)
    @inbounds for n ∈ 0:(N-1)
        sₙ = zero(T)
        @simd ivdep for m ∈ 1:M
            sₙ += A[m + P*n]
        end
        reduction[n+1] = sₙ
    end
    ConstantFixedSizePaddedVector(reduction)
end
@generated function Base.:*(a::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedVector{M,T}}, ::Reducer{S}) where {M,T,S}
    @assert sum(S) == M
    N = length(S)
    q = quote end
    outtup = Expr(:tuple,)
    ind = 0
    for (j,s) ∈ enumerate(S)
        ind += 1
        accumulate_sym = gensym()
        push!(q.args, :($accumulate_sym = a[$ind]))
        for i ∈ 2:s
            ind += 1
            push!(q.args, :($accumulate_sym += a[$ind]))
        end
        push!(outtup.args, accumulate_sym)
    end
    P = PaddedMatrices.calc_padding(N, T)
    for p ∈ N+1:P
        push!(outtup.args, zero(T))
    end
    quote
        @fastmath @inbounds begin
            $q
            ConstantFixedSizePaddedVector{$N,$T,$P}($outtup)
        end
    end
end
@generated function Base.:*(
            a::LinearAlgebra.Adjoint{T,<:AbstractFixedSizePaddedVector{M,T}},
            b::ReducerWrapper{S, <: AbstractFixedSizePaddedVector{M,T}}
        ) where {M,T,S}
    @assert sum(S) == M
    N = length(S)
    q = quote end
    outtup = Expr(:tuple,)
    ind = 0
    for (j,s) ∈ enumerate(S)
        ind += 1
        accumulate_sym = gensym()
        push!(q.args, :($accumulate_sym = a[$ind] * b[$ind]))
        for i ∈ 2:s
            ind += 1
            push!(q.args, :($accumulate_sym += a[$ind] * b[$ind]))
        end
        push!(outtup.args, accumulate_sym)
    end
    P = PaddedMatrices.calc_padding(N, T)
    for p ∈ N+1:P
        push!(outtup.args, zero(T))
    end
    quote
        @fastmath @inbounds begin
            $q
            ConstantFixedSizePaddedVector{$N,$T,$P}($outtup)
        end
    end
end
