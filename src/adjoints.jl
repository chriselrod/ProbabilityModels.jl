using DistributionParameters: One


@inline function RESERVED_INCREMENT_SEED_RESERVED(a::One, b, c)
    RESERVED_INCREMENT_SEED_RESERVED(b, c)
end
@inline function RESERVED_DECREMENT_SEED_RESERVED(a::One, b, c)
    RESERVED_DECREMENT_SEED_RESERVED(b, c)
end
@inline function RESERVED_INCREMENT_SEED_RESERVED(sp::StackPointer, a::One, b, c)
    RESERVED_INCREMENT_SEED_RESERVED(sp, b, c)
end
@inline function RESERVED_DECREMENT_SEED_RESERVED(sp::StackPointer, a::One, b, c)
    RESERVED_DECREMENT_SEED_RESERVED(sp, b, c)
end

@inline RESERVED_MULTIPLY_SEED_RESERVED(a::One, b) = b
@inline RESERVED_NMULTIPLY_SEED_RESERVED(a::One, b) = -b #RESERVED_NMULTIPLY_SEED_RESERVED(b)

@inline function RESERVED_INCREMENT_SEED_RESERVED(a, b::One, c)
    out = RESERVED_INCREMENT_SEED_RESERVED(a, c)
    # @assert !(isa(out, Array) || isa(out, LinearAlgebra.Adjoint{<:Any,<:Array}))
    out
end
@inline function RESERVED_DECREMENT_SEED_RESERVED(a, b::One, c)
    out = RESERVED_DECREMENT_SEED_RESERVED(a, c)
    # @assert !(isa(out, Array) || isa(out, LinearAlgebra.Adjoint{<:Any,<:Array}))
    out
end
@inline function RESERVED_INCREMENT_SEED_RESERVED(sp::StackPointer, a, b::One, c)
    RESERVED_INCREMENT_SEED_RESERVED(sp, a, c)
end
@inline function RESERVED_DECREMENT_SEED_RESERVED(sp::StackPointer, a, b::One, c)
    RESERVED_DECREMENT_SEED_RESERVED(sp, a, c)
end
@inline RESERVED_INCREMENT_SEED_RESERVED(sp::StackPointer, a::One, b) = (sp, b)

@inline RESERVED_MULTIPLY_SEED_RESERVED(a::StackPointer, b::One) = (a, b)
@inline RESERVED_MULTIPLY_SEED_RESERVED(a, b::One) = a
@inline RESERVED_NMULTIPLY_SEED_RESERVED(a, b::One) = -a #RESERVED_NMULTIPLY_SEED_RESERVED(a)

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

Base.size(::AbstractReducer) = ()

# Base.:*(a, ::Reducer{true}) = sum(a)
# Reducer{:row} reduces across rows
@inline Base.:*(::Reducer{:row}, A::LinearAlgebra.Diagonal{<:Real,<:AbstractFixedSizeVector}) = A.diag'
@inline Base.:*(::Reducer{:row}, A::Number) = A
@inline Base.:*(::Reducer{:row}, A::LinearAlgebra.Adjoint{<:Any,<:AbstractFixedSizeVector}) = A

@generated function Base.:*(A::PaddedMatrices.AbstractFixedSizeMatrix{M,N,T,P}, ::Reducer{:row}) where {M,N,T,P}
    quote
        reduction = MutableFixedSizeVector{$N,$T}(undef)
        @inbounds for n ∈ 0:(N-1)
            sₙ = zero(T)
            @vvectorize $T for m ∈ 1:$M
                sₙ += A[m + P*n]
            end
            reduction[n+1] = sₙ
        end
        ConstantFixedSizeVector(reduction)'
    end
end
@generated function Base.:*(
    sp::StackPointer,
    A::PaddedMatrices.AbstractFixedSizeMatrix{M,N,T,P},
    ::Reducer{:row}
) where {M,N,T,P}
    quote
        reduction = PtrVector{$N,$T,$N}(pointer(sp,$T))
        @inbounds for n ∈ 0:(N-1)
            sₙ = zero($T)
            @vvectorize $T for m ∈ 1:$M
                sₙ += A[m + $P*n]
            end
            reduction[n+1] = sₙ
        end
        sp + $(N*sizeof(T)), reduction'
    end
end
@generated function PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(
    sp::StackPointer,
    A::PaddedMatrices.AbstractFixedSizeMatrix{M,N,T,PA},
    ::Reducer{:row},
    C′::LinearAlgebra.Adjoint{T,<:PaddedMatrices.AbstractMutableFixedSizeVector{N,T,PC}}
) where {M,N,T,PA,PC}
    quote
        #        reduction = PtrVector{N,T,P}(pointer(sp,T))
        C = C′'
        D = PtrVector{$N,$T,$PC}(pointer(sp,$T))
        @inbounds for n ∈ 0:(N-1)
            sₙ = zero(T)
            @vvectorize $T for m ∈ 1:$M
                sₙ += A[m + $PA*n]
            end
#            reduction[n+1] = sₙ
            D[n+1] = C[n+1] + sₙ
        end
        sp + $(sizeof(T)*PC), D'
    end
end
@generated function Base.:*(a::LinearAlgebra.Adjoint{T,<:AbstractVector{T}}, ::Reducer{S}) where {T,S}
    S == true && return quote
        $(Expr(:meta,:inline))
        sum(a.parent)
    end
    # @assert sum(S) == M
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
            ConstantFixedSizeVector{$N,$T,$P}($outtup)'
        end
    end
end
@generated function Base.:*(sp::StackPointer, a::LinearAlgebra.Adjoint{T,<:AbstractVector{T}}, ::Reducer{S}) where {T,S}
    S == true && return quote
        $(Expr(:meta,:inline))
        (sp, sum(a.parent))
    end
    # @assert sum(S) == M
    N = length(S)
    q = quote
        out = PtrVector{$N,$T,$N}(pointer(sp,$T))
    end
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
        push!(q.args, :(out[$j] = $accumulate_sym))
    end
    quote
        @fastmath @inbounds begin
            $q
        end
        sp + $(sizeof(T)*N), out'
    end
end
@generated function Base.:*(A::AbstractMatrix{T}, ::Reducer{S}) where {T,S}
    S == true && return :(sum(A))
    # y = Vector{T}(undef, 0)
    y = MutableFixedSizeVector{sum(S),T}(undef)
    quote
        # resize!(y, size(A,1))
        sum!($y, A)' * Reducer{$S}()
    end
end
@generated function Base.:*(sp::StackPointer, A::AbstractMatrix{T}, ::Reducer{S}) where {T,S}
    S == true && return :(sum(A))
    # y = Vector{T}(undef, 0)
    quote
        (sp, y) = PtrVector{$(sum(S)),$T}(sp)
        # resize!(y, size(A,1))
        *(sp, sum!(y, A)', Reducer{$S}())
    end
end
@generated function Base.:*(
            a::LinearAlgebra.Adjoint{T, <: AbstractVector{T}},
            b::ReducerWrapper{S, <: AbstractVector{T}}
        ) where {T,S}
    # @assert sum(S) == M
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
            ConstantFixedSizeVector{$N,$T,$P}($outtup)'
        end
    end
end
@generated function Base.:*(A::AbstractMatrix{T}, rw::ReducerWrapper{S}) where {T,S}
    # y = Vector{T}(undef, 0)
    y = MutableFixedSizeVector{sum(S),T}(undef)
    quote
        # resize!(y, size(A,1))
        sum!($y, A)' * rw
    end
end
@generated function Base.:*(
    sp::StackPointer,
    a::LinearAlgebra.Adjoint{T, <: AbstractVector{T}},
    b::ReducerWrapper{S, <: AbstractVector{T}}
) where {T,S}
    # @assert sum(S) == M
    N = length(S)
    q = quote
        (sp, out) = PtrVector{$N,$T}(sp)
    end
    ind = 0
    ind2 = 0
    for (j,s) ∈ enumerate(S)
        ind += 1
        accumulate_sym = gensym()
        push!(q.args, :($accumulate_sym = a[$ind] * b[$ind]))
        for i ∈ 2:s
            ind += 1
            push!(q.args, :($accumulate_sym += a[$ind] * b[$ind]))
        end
        ind2 += 1
        push!(q.args, :(out[$ind2] = accumulate_sym))
    end
#    P = PaddedMatrices.calc_padding(N, T)
#    for p ∈ N+1:P
#        push!(outtup.args, zero(T))
#    end
    quote
        @fastmath @inbounds begin
            $q
 #           ConstantFixedSizeVector{$N,$T,$P}($outtup)'
        end
        sp, out'
    end
end
@generated function Base.:*(sp::StackPointer, A::AbstractMatrix{T}, rw::ReducerWrapper{S}) where {T,S}
    # y = Vector{T}(undef, 0)
    quote
        (sp, y) = PtrVector{$(sum(S)),$T}(sp)

        # resize!(y, size(A,1))
        *(sp, sum!($y, A)', rw)
    end
end

#@inline ∂mul(x, y, ::Val{(true,true)}) = ((@show typeof(x), typeof(y)); return y, x)
@inline ∂mul(x, y, ::Val{(true,true)}) = y, x
@inline ∂mul(x, y, ::Val{(true,false)}) = y
@inline ∂mul(x, y, ::Val{(false,true)}) = x

@inline function ∂mul(D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T,P}},
                    L::StructuredMatrices.AbstractLowerTriangularMatrix{M,T,N}, ::Val{(true,true)}) where {M,N,P,T}

    #println("∂mul")
    StructuredMatrices.∂DiagLowerTri∂Diag(L), StructuredMatrices.∂DiagLowerTri∂LowerTri(D)
end
@inline function ∂mul(D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T,P}},
                    L::StructuredMatrices.AbstractLowerTriangularMatrix{M,T,N}, ::Val{(true,false)}) where {M,N,P,T}


    StructuredMatrices.∂DiagLowerTri∂Diag(L)
end
@inline function ∂mul(D::LinearAlgebra.Diagonal{T,<:AbstractFixedSizeVector{M,T,P}},
                    L::StructuredMatrices.AbstractLowerTriangularMatrix{M,T,N}, ::Val{(false,true)}) where {M,N,P,T}


    StructuredMatrices.∂DiagLowerTri∂LowerTri(D)
end


