

import SIMDPirates: Vec, vmul, vsub, vdiv

# const StaticSizedVector{N,T} = Union{Vec{N,T},SVector{N,T}}
# Mutable Sized Vector
const MSVec{N,T} = Union{PointerVector{N,T},MVector{N,T}}

function first_order_exp(assignments::Vector{Symbol},arg::Symbol)
    exp_arg = gensym(:exp_arg)
    quote
        $exp_arg = exp($(arg[1]))
        $(assignments[1]) = $exp_arg
        $(assignments[2]) = $exp_arg
    end
end
UNARY_FIRST_ORDER_DIFF_RULES[(:Base,:exp)] = UNARY_WRAPPER(first_order_exp)
function second_order_exp(assignments::Vector{Symbol},arg::Symbol)
    exp_arg = gensym(:exp_arg)
    quote
        $exp_arg = exp($arg)
        $(assignments[1]) = $exp_arg
        $(assignments[2]) = $exp_arg
        $(assignments[3]) = $exp_arg
    end
end
UNARY_SECOND_ORDER_DIFF_RULES[(:Base,:exp)] = UNARY_WRAPPER(second_order_exp)

const UNARY_FUNCTIONS = Tuple{Symbol,Tuple{Symbol,Symbol}}[]
const BINARY_FUNCTIONS = Tuple{Symbol,Tuple{Symbol,Symbol,Tuple{Bool,Bool}}}[]
const TERNARY_FUNCTIONS = Tuple{Symbol,Tuple{Symbol,Symbol,Tuple{Bool,Bool,Bool}}}[]
const QUADPLUS_FUNCTIONS = Tuple{Symbol,Tuple{Symbol,Symbol,Vector{Bool}}}[]

@inline function f_normal_lpdf(y::T, μ::T, σ::T) where {T <: Scalar}
    -log(σ) - T(0.5)*abs2((y-μ)/σ)
end
@inline function f_normal_prec_lpdf(y::T, μ::T, λ::T) where {T <: Scalar}
    T(0.5)*(log(λ) - λ*abs2((y-μ)))
end
@inline function f_normal_lpdf(y::V, μ::V, σ::V) where {N,T,V <: Vec{N,T}}
    δ = vdiv(vsub(y, μ), σ)
    vsub(vmul(T(-0.5),vmul(δ,δ)), SLEEFPirates.log_fast(σ))
end
@inline function f_normal_prec_lpdf(y::V, μ::V, λ::V) where {N,T,V <: Vec{N,T}}
    δ = vsub(y, μ)
    δ² = vmul(δ, δ)
    vmul(T(-0.5), vfmsub(δ², λ, SLEEFPirates.log_fast(λ)))
end
### Rely on SLEEFPirates vectorization macro for vector types.

@inline function fg_normal_y_mu_sigma_lpdf(y, μ, σ)

end
@inline function fg_normal_mu_sigma_lpdf(y, μ, σ)

end
@inline function fg_normal_y_sigma_lpdf(y, μ, σ)

end
@inline function fg_normal_y_mu_lpdf(y, μ, σ)

end
@inline function fg_normal_y_sigma_lpdf(y, μ, σ)

end
@inline function fg_normal_mu_lpdf(y, μ, σ)

end
@inline function fg_normal_sigma_lpdf(y, μ, σ)

end


@inline function fgh_normal_y_mu_sigma_lpdf(y, μ, σ)

end
@inline function fgh_normal_mu_sigma_lpdf(y, μ, σ)

end
@inline function fgh_normal_y_sigma_lpdf(y, μ, σ)

end
@inline function fgh_normal_y_mu_lpdf(y, μ, σ)

end
@inline function fgh_normal_y_sigma_lpdf(y, μ, σ)

end
@inline function fgh_normal_mu_lpdf(y, μ, σ)

end
@inline function fgh_normal_sigma_lpdf(y, μ, σ)

end


for (f,key) ∈ UNARY_FUNCTIONS
    let fg = Symbol(:fg_, f)
        UNARY_FIRST_ORDER_DIFF_RULES[key] = UNARY_WRAPPER(
            (assignments, args) -> Expr(:(=), Expr(:tuple, assignments...), Expr(:call, fg, args...))
        )
    end
    let fgh = Symbol(:fgh_, f)
        UNARY_SECOND_ORDER_DIFF_RULES[key] = UNARY_WRAPPER(
            (assignments, args) -> Expr(:(=), Expr(:tuple, assignments...), Expr(:call, fgh, args...))
        )
    end
end
for (f,key) ∈ BINARY_FUNCTIONS
    let fg = Symbol(:fg_, f)
        BINARY_FIRST_ORDER_DIFF_RULES[key] = BINARY_WRAPPER(
            (assignments, args) -> Expr(:(=), Expr(:tuple, assignments...), Expr(:call, fg, args...))
        )
    end
    let fgh = Symbol(:fgh_, f)
        BINARY_SECOND_ORDER_DIFF_RULES[key] = BINARY_WRAPPER(
            (assignments, args) -> Expr(:(=), Expr(:tuple, assignments...), Expr(:call, fgh, args...))
        )
    end
end
for (f,key) ∈ TERNARY_FUNCTIONS
    let fg = Symbol(:fg_, f)
        TERNARY_FIRST_ORDER_DIFF_RULES[key] = TERNARY_WRAPPER(
            (assignments, args) -> Expr(:(=), Expr(:tuple, assignments...), Expr(:call, fg, args...))
        )
    end
    let fgh = Symbol(:fgh_, f)
        TERNARY_SECOND_ORDER_DIFF_RULES[key] = TERNARY_WRAPPER(
            (assignments, args) -> Expr(:(=), Expr(:tuple, assignments...), Expr(:call, fgh, args...))
        )
    end
end
for (f,key) ∈ QUADPLUS_FUNCTIONS
    let fg = Symbol(:fg_, f)
        QUADPLUS_FIRST_ORDER_DIFF_RULES[key] =
            (assignments, args) -> Expr(:(=), Expr(:tuple, assignments...), Expr(:call, fg, args...))
    end
    let fgh = Symbol(:fgh_, f)
        QUADPLUS_SECOND_ORDER_DIFF_RULES[key] =
            (assignments, args) -> Expr(:(=), Expr(:tuple, assignments...), Expr(:call, fgh, args...))
    end
end
