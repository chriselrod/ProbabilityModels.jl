M
# to add:
# PDMatrix
# Cholesky Factor matrix
# Simplex
# abstract type AbstractParameter end
# abstract type AbstractScalarParameter end

struct RealFloat{T} <: Real
    data::T
end
struct PositiveFloat{T} <: Real
    data::T
end
struct LowerBoundedFloat{T,LB} <: Real
    data::T
end
struct UpperBoundedFloat{T,UB} <: Real
    data::T
end
struct BoundedFloat{T,LB,UB} <: Real
    data::T
end
struct ProbabilityFloat{T} <: Real
    data::T
end
const ScalarParameter{T} = Union{
    Float{T},
    PositiveFloat{T},
    LowerBoundedFloat{T},
    UpperBoundedFloat{T},
    BoundedFloat{T},
    ProbabilityFloat{T}
}


struct RealVector{S,T,P,L} <: AbstractConstantFixedSizePaddedVector{M,T,P,L}
    data::NTuple{L,T}
end
struct PositiveVector{M,T,P,L} <: AbstractConstantFixedSizePaddedVector{M,T,P,L}
    data::NTuple{L,T}
end
struct LowerBoundVector{M,LB,T,P,L} <: AbstractConstantFixedSizePaddedVector{M,T,P,L}
    data::NTuple{L,T}
end
struct UpperBoundVector{M,UB,T,P,L} <: AbstractConstantFixedSizePaddedVector{M,T,P,L}
    data::NTuple{L,T}
end
struct BoundedVector{M,LB,UB,T,P,L} <: AbstractConstantFixedSizePaddedVector{M,T,P,L}
    data::NTuple{L,T}
end
struct ProbabilityVector{M,T,P,L} <: AbstractConstantFixedSizePaddedVector{M,T,P,L}
    data::NTuple{L,T}
end
const VectorParameter{M,T,P,L} = Union{
    RealVector{M,T,P,L},
    PositiveVector{M,T,P,L},
    LowerBoundVector{M,LB,T,P,L} where {LB},
    UpperBoundVector{M,UB,T,P,L} where {UB},
    BoundedVector{M,LB,UB,T,P,L} where {LB, UB},
    ProbabilityVector{M,T,P,L}
}



struct RealMatrix{M,N,T,P,L} <: AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L}
    data::SizedSIMDMatrix{M,N,T,R,L}
end
struct PositiveMatrix{M,N,T,P,L} <: AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L}
    data::SizedSIMDMatrix{M,N,T,R,L}
end
struct LowerBoundMatrix{M,N,LB,T,P,L} <: AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L}
    data::SizedSIMDMatrix{M,N,T,R,L}
end
struct UpperBoundMatrix{M,N,UB,T,P,L} <: AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L}
    data::SizedSIMDMatrix{M,N,T,R,L}
end
struct BoundedMatrix{M,N,LB,UB,T,P,L} <: AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L}
    data::SizedSIMDMatrix{M,N,T,R,L}
end
struct ProbabilityMatrix{M,N,T,P,L} <: AbstractConstantFixedSizePaddedMatrix{M,N,T,P,L}
    data::SizedSIMDMatrix{M,N,T,R,L}
end
const MatrixParameter{M,N,T,P,L} = Union{
    RealMatrix{M,N,T,P,L},
    PositiveMatrix{M,N,T,P,L},
    LowerBoundMatrix{M,N,LB,T,P,L} where {LB},
    UpperBoundMatrix{M,N,UB,T,P,L} where {UB},
    BoundedMatrix{M,N,LB,UB,T,P,L} where {LB, UB},
    ProbabilityMatrix{M,N,T,P,L}
}

const Parameters{T} = Union{
    ScalarParameter{T},
    VectorParameter{L,T} where {L},
    MatrixParameter{M,N,T} where {M,N}
}
const PositiveParameter{T} = Union{
    PositiveFloat{T},
    PositiveVector{L,T} where {L},
    PositiveMatrix{M,N,T} where {M,N}
}
const LowerBoundedParameter{T,LB} = Union{
    LowerBoundedFloat{LB,T},
    LowerBoundVector{L,LB,T} where {L},
    LowerBoundMatrix{M,N,LB,T} where {M,N}
}
const UpperBoundedParameter{T,UB} = Union{
    UpperBoundedFloat{UB,T},
    UpperBoundVector{L,UB,T} where {L},
    UpperBoundMatrix{M,N,UB,T} where {M,N}
}
const BoundedParameter{T,LB,UB} = Union{
    BoundedFloat{LB,UB,T},
    BoundVector{L,LB,UB,T} where {L},
    BoundMatrix{M,N,LB,UB,T} where {M,N}
}
const ProbabilityParameter{T} = Union{
    ProbabilityFloat{T},
    ProbabilityVector{L,T} where {L},
    ProbabilityMatrix{M,N,T} where {M,N}
}


isparameter(::T) where {T <: Parameters} = true
isparameter(::Type{T}) where {T <: Parameters} = true
isparameter(::Any) = false

ispositive(::T) where {T <: PositiveParameter} = true
ispositive(::Type{T}) where {T <: PositiveParameter} = true
ispositive(::Any) = false

islowerbounded(::T) where {T <: LowerBoundedParameter} = true
islowerbounded(::Type{T}) where {T <: LowerBoundedParameter} = true
islowerbounded(::Any) = false

isupperbounded(::T) where {T <: UpperBoundedParameter} = true
isupperbounded(::Type{T}) where {T <: UpperBoundedParameter} = true
isupperbounded(::Any) = false

isbounded(::T) where {T <: BoundedParameter} = true
isbounded(::Type{T}) where {T <: BoundedParameter} = true
isbounded(::Any) = false

isprobability(::T) where {T <: ProbabilityParameter} = true
isprobability(::Type{T}) where {T <: ProbabilityParameter} = false
isprobability(::Any) = false

bounds(::LowerBoundedParameter{T,LB}) where {T,LB} = (T(LB),T(Inf))
bounds(::UpperBoundedParameter{T,UB}) where {T,UB} = (T(-Inf),T(UB))
bounds(::BoundedParameter{T,LB,UB}) where {T,LB,UB} = (T(LB), T(UB))
bounds(::Any) = (-Inf,Inf)

lower_bound(::Type{LowerBoundedParameter{T,LB}}) where {T,LB} = LB
upper_bound(::Type{UpperBoundedParameter{T,UB}}) where {T,UB} = UB
bounds(::Type{BoundedParameter{T,LB,UB}}) where {T,LB,UB} = (LB, UB)


type_length(::Type{<:ScalarParameter}) = 1
type_length(::Type{<:VectorParameter{L}}) where L = L
type_length(::Type{<:MatrixParameter{M,N,T,R,L}}) where {M,N,T,R,L} = L



function return_param_symbol(expr)


end
