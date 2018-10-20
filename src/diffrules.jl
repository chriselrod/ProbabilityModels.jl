
### The plan is to in general provide efficient implementations that
### provide the function values and the derivatives. However, failing
### that, the plan is to fall back to constructing the derivatves.

### Falls back to looking for DiffRules.jl
### However

using FunctionWrappers: FunctionWrapper

const UNARY_WRAPPER = FunctionWrapper{Expr,Tuple{NTuple{2,Symbol},Tuple{Symbol}}}
const BINARY_WRAPPER = FunctionWrapper{Expr,Tuple{NTuple{3,Symbol},NTuple{2,Symbol}}}
const TERNARY_WRAPPER = FunctionWrapper{Expr,Tuple{NTuple{4,Symbol},NTuple{3,Symbol}}}

const UNARY_FIRST_ORDER_DIFF_RULES = Dict{Tuple{Symbol,Symbol},UNARY_WRAPPER}()
const UNARY_SECOND_ORDER_DIFF_RULES = Dict{Tuple{Symbol,Symbol},UNARY_WRAPPER}()
const BINARY_FIRST_ORDER_DIFF_RULES = Dict{Tuple{Symbol,Symbol,Tuple{Bool,Bool}},BINARY_WRAPPER}()
const BINARY_SECOND_ORDER_DIFF_RULES = Dict{Tuple{Symbol,Symbol,Tuple{Bool,Bool}},BINARY_WRAPPER}()
const TERNARY_FIRST_ORDER_DIFF_RULES = Dict{Tuple{Symbol,Symbol,Tuple{Bool,Bool,Bool}},TERNARY_WRAPPER}()
const TERNARY_SECOND_ORDER_DIFF_RULES = Dict{Tuple{Symbol,Symbol,Tuple{Bool,Bool,Bool}},TERNARY_WRAPPER}()
const QUADPLUS_FIRST_ORDER_DIFF_RULES = Dict{Tuple{Symbol,Symbol,Vector{Bool}},Function}()
const QUADPLUS_SECOND_ORDER_DIFF_RULES = Dict{Tuple{Symbol,Symbol,Vector{Bool}},Function}()


function no_diffrule_for_function_error(M, f)
    throw("No diff rules for $f in module $M, with 1 argument.")
end
function first_order_diff_rule(M::Symbol, func::Symbol,
        deriv::NTuple{1,Bool}, args::NTuple{1,Symbol}, assignments::NTuple{N,Symbol}) where N
    if deriv[1] == false
        q = :($(assignments[1]) = $(M).$(func)($(args[1])))
    elseif (M,func) ∈ keys(UNARY_FIRST_ORDER_DIFF_RULES)
        q = UNARY_FIRST_ORDER_DIFF_RULES[(M,func)](assignments, args)
    elseif DiffRules.hasdiffrule(M, func, 1)
        q = quote
                $(assignments[1]) = $(M).$(func)($(args[1]))
                $(assignments[2]) = $(DiffRules.diffrule(M, func, args[1]))
            end
    else
        no_diffrule_for_function_error(M, func)
    end
    q
end

function first_order_diff_rule(M::Symbol, func::Symbol,
        deriv::NTuple{2,Bool}, args::NTuple{2,Symbol}, assignments::Expr)
    if deriv[1] == deriv[2] == false
        q = :($(assignments[1]) = $(M).$(func)($(args...)))
    elseif (M,func,deriv) ∈ keys(BINARY_FIRST_ORDER_DIFF_RULES)
        q = BINARY_FIRST_ORDER_DIFF_RULES[(M,func,deriv)](assignments, args...)
    elseif DiffRules.hasdiffrule(M, func, 2)
        q = quote
                $(assignments[1]) = $(M).$(func)($(args...))
            end
        diff_rules = DiffRules.diffrule(M, func, args...)
        for i ∈ eachindex(deriv)
            if deriv[i] == true
                push!(q.args, :($(assignments[i+1]) = diff_rules[i]))
            end
        end
    else
        no_diffrule_for_function_error(M, func)
    end
    q
end
function first_order_diff_rule(M::Symbol, func::Symbol,
        deriv::NTuple{3,Bool}, args::NTuple{3,Symbol}, assignments::Expr)
    if deriv[1] == deriv[2] == deriv[3] == false
        q = :($(assignments[1]) = $(M).$(func)($(args...)))
    elseif (M,func,deriv) ∈ keys(TERNARY_FIRST_ORDER_DIFF_RULES)
        q = TERNARY_FIRST_ORDER_DIFF_RULES[(M,func,deriv)](assignments, args...)
    elseif DiffRules.hasdiffrule(M, func, 3)
        q = quote
                $(assignments[1]) = $(M).$(func)($(args...))
            end
        diff_rules = DiffRules.diffrule(M, func, args...)
        for i ∈ eachindex(deriv)
            if deriv[i] == true
                push!(q.args, :($(assignments[i+1]) = diff_rules[i]))
            end
        end
    else
        no_diffrule_for_function_error(M, func)
    end
    q
end
function first_order_diff_rule(M::Symbol, func::Symbol,
        deriv::Vector{Bool}, args::Vector{Symbol})
    if !any(deriv)
        q = :($assignments = $(M).$(func)($(args...)))
    elseif (M,func) ∈ keys(QUADPLUS_FIRST_ORDER_DIFF_RULES)
        q = QUADPLUS_FIRST_ORDER_DIFF_RULES[(M,func,deriv)](assignments, args...)
    elseif DiffRules.hasdiffrule(M, func, length(args))
        q = quote
                $(assignments[1]) = $(M).$(func)($(args...))
            end
        diff_rules = DiffRules.diffrule(M, func, args...)
        for i ∈ eachindex(deriv)
            if deriv[i] == true
                push!(q.args, :($(assignments[i+1]) = diff_rules[i]))
            end
        end
    else
        no_diffrule_for_function_error(M, func)
    end
    q
end


function second_order_diff_rule(func::Symbol)

end
