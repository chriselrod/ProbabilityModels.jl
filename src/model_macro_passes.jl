
# includet("/home/chriselrod/Documents/progwork/julia/model_macro_passes.jl")

# using MacroTools
# using MacroTools: striplines, @capture, prewalk, postwalk, @q
# q = @q begin
#     β₀ ~ Normal(μ₀, σ₀)
#     β₁ ~ Normal(μ₁, σ₁)
#     y ~ Bernoulli_logit(β₀ + x * β₁)
# end

function determine_variables(expr)
    variables = Set{Symbol}()
    ignored_symbols = Set{Symbol}()
    push!(ignored_symbols, :target)
    prewalk(expr) do x
        if @capture(x, f_(ARGS__))
            push!(ignored_symbols, f)
        elseif isa(x, Symbol) && x ∉ ignored_symbols
            push!(variables, x)
        end
        return x
    end
    variables
end

function translate_sampling_statements(expr)
    prewalk(expr) do x
        if @capture(x, y_ ~ f_(θ__))
            if f ∈ ProbabilityDistributions.FMADD_DISTRIBUTIONS
                if @capture(x, y_ ~ f_(α_ + X_ * β_ , θ__))
                    return :(target = target + $(Symbol(f,:_fmadd_logeval_dropconst))($y, $X, $β, $α, $(θ...)))
                elseif @capture(x, y_ ~ f_(X_ * β_ + α_, θ__))
                    return :(target = target + $(Symbol(f,:_fmadd_logeval_dropconst))($y, $X, $β, $α, $(θ...)))
                elseif @capture(x, y_ ~ f_(α_ - X_ * β_, θ__))
                    return :(target = target + $(Symbol(f,:_fnmadd_logeval_dropconst))($y, $X, $β, $α, $(θ...)))
                elseif @capture(x, y_ ~ f_(X_ * β_ - α_, θ__))
                    return :(target = target + $(Symbol(f,:_fmsub_logeval_dropconst))($y, $X, $β, $α, $(θ...)))
                elseif @capture(x, y_ ~ f_(- X_ * β_ - α_, θ__))
                    return :(target = target + $(Symbol(f,:_fnmsub_logeval_dropconst))($y, $X, $β, $α, $(θ...)))
                elseif @capture(x, y_ ~ f_( - α_ - X_ * β_, θ__))
                    return :(target = target + $(Symbol(f,:_fnmsub_logeval_dropconst))($y, $X, $β, $α, $(θ...)))
                end
            end
            return :(target = target + $(Symbol(f,:_logeval_dropconst))($y, $(θ...)))
        elseif @capture(x, a_ += b_)
            return :($a = $a + $b)
        else
            return x
        end
    end
end

# q2 = translate_sampling_statements(q)
# postwalk(x -> (@show x), q2)
# prewalk(x -> (@show x), q2)



function flatten_subexpression!(flattened, args::AbstractVector)
    map(args) do arg
        flatten_subexpression!(flattened, arg)
    end
end
function flatten_subexpression!(flattened, arg::Expr)
    postwalk(arg) do x
        if @capture(x, f_(args__))
            a = gensym()
            push!(flattened, :($a = $f($(args...))))
            return a
        elseif @capture(x, A_[I__])
            a = gensym()
            push!(flattened, :($a = getindex($A, $(I...))))
            return a
        else
            return x
        end
    end
end
flatten_subexpression!(::Any, s::Symbol) = s

"""
Converts an expression into a sequence of single assignment codes (SACs)

"""
function flatten_expression(expr)
    q_flattened = @q begin end
    flattened = q_flattened.args
    for ex ∈ expr.args
        if @capture(ex, for i_ ∈ iter_ body_ end)
            push!(flattened, quote
                for $i ∈ $iter
                    $(flatten_expression(Expr(:block, body)))
                end
            end)
            continue
        elseif @capture(ex, a_ = f_(args__))
            args2 = flatten_subexpression!(flattened, args)
            push!(flattened, :($a = $f($(args2...))))
        elseif @capture(ex, a_ += f_(args__))
            args2 = flatten_subexpression!(flattened, args)
            b = gensym()
            push!(flattened, quote
                $b = $f($(args2...))
                $a = $a + $b
            end)
        elseif @capture(ex, a_ += b_)
            push!(flattened, :($a = $a + $b))
        elseif @capture(ex, a_ -= f_(args__))
            args2 = flatten_subexpression!(flattened, args)
            b = gensym()
            push!(flattened, quote
                $b = $f($(args2...))
                $a = $a - $b
            end)
        elseif @capture(ex, a_ -= b_)
            push!(flattened, :($a = $a - $b))
        elseif @capture(ex, a_ *= f_(args__))
            args2 = flatten_subexpression!(flattened, args)
            b = gensym()
            push!(flattened, quote
                $b = $f($(args2...))
                $a = $a * $b
            end)
        elseif @capture(ex, a_ *= b_)
            push!(flattened, :($a = $a * $b))
        elseif @capture(ex, a_ /= f_(args__))
            args2 = flatten_subexpression!(flattened, args)
            b = gensym()
            push!(flattened, quote
                $b = $f($(args2...))
                $a = $a / $b
            end)
        elseif @capture(ex, a_ /= b_)
            push!(flattened, :($a = $a / $b))
        end
    end
    q_flattened
end

"""
This pass is to be applied to a flattened expression, after expressions such as
a += b
a -= b
a *= b
a /= b
have been expanded. Later insertions of these symbols, for example in the constraining
transformations of the parameters, will be bypassed. This allows for incrementing of
the `ProbabilityModels.VectorizationBase.vectorizable` parameters:
Symbol("##θparameter##")
Symbol("##∂θparameter##")
"""
function rename_assignments(expr, vars = Dict{Symbol,Symbol}())
    postwalk(expr) do ex
        if @capture(ex, a_ = b_)
            if isa(b, Expr)
                c = postwalk(x -> get(vars, x, x), b)
            else
                c = b
            end
            if isa(a, Symbol)
                if haskey(vars, a)
                    lhs = gensym(a)
                    vars[a] = lhs
                    return :($lhs = $c)
                else
                    vars[a] = a
                    return :($a = $c)
                end
            else
                return :($a = $c)
            end
        else
            return ex
        end
    end, vars
    # for i ∈ eachindex(expr.args)
    #     ex = expr.args[i]
    #     isa(ex, Expr) || continue
    #     ex.head == :block && ProbabilityModels.rename_assignments(ex, vars)
    #     ex.head == :(=) || continue
    #     for j ∈ 2:length(ex.args)
    #         ex.args[j] = postwalk(ex.args[j]) do x
    #             get(vars, x, x)
    #         end
    #     end
    #     lhs = ex.args[1]
    #     vars[lhs] = ex.args[1] = gensym(lhs)
    # end
    # expr, vars
end

"""
Translates first update statements into assignments.

This is so that we can generate autodiff code via incrementing nodes, eg
node += adjoint * value
this saves us from having to initialize them all at 0.
While the compiler can easily eliminate clean up those initializations with scalars,
it may not be able to do so for arbitrary user types.


"""
function first_updates_to_assignemnts(expr, variables_input)
    variables = Set(variables_input)
    ignored_symbols = Set{Symbol}()
    for i ∈ eachindex(expr.args)
        ex = expr.args[i]
        isa(ex, Expr) || continue
        if ex.head == :block
            expr.args[i] = first_updates_to_assignemnts(ex, variables)
            continue
        end
        isa(ex.args[1], Symbol) || continue
        lhs = ex.args[1]
        # @show lhs
        new_assignment = lhs ∉ variables
        push!(variables, lhs)
        if ex.head == :(=)
            check = false
            for j ∈ 2:length(ex.args)
                let new_assignment = new_assignment
                    postwalk(ex.args[j]) do x
                        isa(x, Symbol) && push!(variables, x)
                        if new_assignment && x == lhs # Then we need to check that it doesn't appear on the lhs
                            check = true
                        end
                        x
                    end
                end
            end
            if check
                if @capture(ex, a_ = a_ + b__ ) || @capture(ex, a_ = b__ + a_ ) || @capture(ex, a_ = b__ )
                    expr.args[i] = :($a = $(b...))
                elseif @capture(ex, a_ = a_ - b__ )
                    expr.args[i] = :($a = -1 * $(b...))
                else
                    throw("""
                        This was the first assignment for $lhs in:
                            $(ex)
                        If this was meant to initialize $lhs, could not determine how to do so.
                        """)
                end
            end
        elseif ex.head == :(+=)# || ex.head == :(*=)
            if new_assignment
                ex.head = :(=)
            end
            for j ∈ 2:length(ex.args)
                postwalk(ex.args[j]) do x
                    isa(x, Symbol) && push!(variables, x)
                    x
                end
            end
        elseif ex.head == :(-=)# || ex.head == :(*=)
            if new_assignment
                ex.head = :(=)
                expr.args[i] = :($lhs = -1 * $(expr.args[2:end]...) )
            end
            for j ∈ 2:length(ex.args)
                postwalk(ex.args[j]) do x
                    isa(x, Symbol) && push!(variables, x)
                    x
                end
            end
        else # we walk and simply add symbols, assuming everything referenced must already be defined?
            postwalk(ex) do x
                isa(x, Symbol) && push!(variables, x)
                x
            end
        end
    end
    expr
end

# q3 = flatten_expression(q2)

# q3test = quote
#     a = foo(bar(biz(buz(r, zip(zop[2], grop), fab(bet)), x, s), r))
#     base(a, d, s)
# end |> flatten_expression
# q4test = quote
#     a = zero(eltype(zop))
#     for i ∈ eachindex(zop)
#         a = foo(a, bar(biz(buz(r, zip(zop[i], grop), fab(bet)), x, s), r))
#     end
#     base(a, d, s)
# end |> flatten_expression
#
# using DiffRules
# DiffRules.hasdiffrule(:Base, :exp, 1)
# DiffRules.diffrule(:Base, :exp, :x)

function constant_drop_pass!(first_pass, expr, tracked_vars)
    for x ∈ expr.args
        if @capture(x, for i_ ∈ iter_ body_ end)
            throw("Loops not yet supported!")
            # reverse_diff_loop_pass!(first_pass, second_pass, i, iter, body, expr, tracked_vars)
        elseif @capture(x, out_ = f_(A__))
            if f ∈ ProbabilityDistributions.DISTRIBUTION_DIFF_RULES
                track_tup = Expr(:tuple,)
                for a ∈ A
                    if a ∈ tracked_vars
                        push!(track_tup.args, true)
                        push!(tracked_vars, out)
                    else
                        push!(track_tup.args, false)
                    end
                end
                push!(first_pass.args, :($out = ProbabilityModels.ProbabilityDistributions.$f($(A...), Val{$track_tup}())))
                # push!(first_pass.args, :(@show $(A...)))
                # push!(first_pass.args, :(@show $out))
            else
                push!(first_pass.args, x)
            end
        else
            push!(first_pass.args, x)
        end
    end
    first_pass
end

# How to search modules?
function reverse_diff_pass(expr, gradient_targets)
    tracked_vars = Set{Symbol}(gradient_targets)
    q_first_pass = @q begin end
    first_pass = q_first_pass.args
    q_second_pass = @q begin end
    second_pass = q_second_pass.args
    ProbabilityModels.reverse_diff_pass!(first_pass, second_pass, expr, tracked_vars)
    q_first_pass, q_second_pass
end

# function reverse_diff_loop_pass!(first_pass, second_pass, i, iter, body, expr, tracked_vars)
#     # if we have a for loop, we apply the pass to the loop body, creating new loop expressions
#     q_first_pass_loop = @q begin end
#     first_pass_loop = q_first_pass_loop.args
#     q_second_pass_loop = @q begin end
#     second_pass_loop = q_second_pass_loop.args
#     ProbabilityModels.reverse_diff_pass!(first_pass_loop, second_pass_loop, body, tracked_vars)
#     # then we create two for loops.
#     push!(first_pass_loop, quote
#         for $i ∈ $iter
#         # @vectorize for $i ∈ $iter
#             $q_first_pass_loop
#         end
#     end)
#     # Do we need the reverse?
#     push!(second_pass_loop, quote
#         for $i ∈ $iter
#         # @vectorize for $i ∈ $iter
#         # for $i ∈ reverse($iter)
#             $q_second_pass_loop
#         end
#     end)
# end

function reverse_diff_pass!(first_pass, second_pass, expr, tracked_vars)
    for x ∈ expr.args
        if @capture(x, for i_ ∈ iter_ body_ end)
            throw("Loops not yet supported!")
            # reverse_diff_loop_pass!(first_pass, second_pass, i, iter, body, expr, tracked_vars)
        elseif @capture(x, out_ = f_(A__))
            diff!(first_pass, second_pass, tracked_vars, out, f, A)
        else
            push!(first_pass, x)
        end
    end
end

function apply_diff_rule!(first_pass, second_pass, tracked_vars, out, f, A, diffrules::NTuple{N}) where {N}
    track_out = false
    push!(first_pass.args, :($out = $f($(A...))))
    adjout = Symbol("###adjoint###", out)
    for i ∈ eachindex(A)
        A[i] ∈ tracked_vars || continue
        track_out = true
        ∂ = gensym(:∂)
        push!(first_pass.args, :($∂ = $(diffrules[i])))
        pushfirst!(second_pass.args, :( $(Symbol("###adjoint###", A[i])) += $∂ * $adjout ))
    end
    track_out && push!(tracked_vars, out)
    nothing
end
function apply_diff_rule!(first_pass, second_pass, tracked_vars, out, f, A, diffrule)
    length(A) == 1 || throw("length(A) == $(length(A)); must equal 1 when passed diffrules are: $(diffrule)")
    track_out = false
    push!(first_pass, :($out = $f($(A...))))
    adjout = Symbol("###adjoint###", out)
    for i ∈ eachindex(A)
        A[i] ∈ tracked_vars || continue
        track_out = true
        ∂ = gensym(:∂)
        push!(first_pass.args, :($∂ = $(diffrule)))
        pushfirst!(second_pass.args, :( $(Symbol("###adjoint###", A[i])) += $∂ * $adjout ))
    end
    track_out && push!(tracked_vars, out)
    nothing
end

function diff!(first_pass, second_pass, tracked_vars, out, f, A)
    arity = length(A)
    if f ∈ ProbabilityDistributions.DISTRIBUTION_DIFF_RULES
        ProbabilityDistributions.distribution_diff_rule!(:(ProbabilityModels.ProbabilityDistributions), first_pass, second_pass, tracked_vars, out, A, f)
    elseif haskey(SPECIAL_DIFF_RULES, f)
        SPECIAL_DIFF_RULES[f](first_pass, second_pass, tracked_vars, out, A)
    elseif DiffRules.hasdiffrule(:Base, f, arity)
        apply_diff_rule!(first_pass, second_pass, tracked_vars, out, f, A, DiffRules.diffrule(:Base, f, A...))
    elseif DiffRules.hasdiffrule(:SpecialFunctions, f, arity)
        apply_diff_rule!(first_pass, second_pass, tracked_vars, out, f, A, DiffRules.diffrule(:SpecialFunctions, f, A...))
    else # ForwardDiff?
        # Or, for now, Zygote for univariate.
        throw("Fall back differention rules not yet implemented, and no method yet to handle $f($(A...))")
    end
end

# vars2 = Set{Symbol}();
# push!(vars2, :a);
# push!(vars2, :b);
# sum_expr = :(Expr(:call, :+, $(QuoteNode.(vars2)...)))
# qgentest = quote
#     @generated function bar(; $(vars2...))
#     # function bar(; $(vars2...))
#         $sum_expr
#     end
# end
# bar(a = 3, b = 5) # 8
#
# sum_expr_2 = Expr(:call, :+, vars2...)
# qgentest2 = quote
#     @generated function bar2(; $(vars2...)) end
# end
# qgentest2_body = qgentest2.args[end].args[end].args[end].args;
# push!(qgentest2_body, :(expr = $(Expr(:quote, sum_expr_2))))
# push!(qgentest2_body, :expr)
#
#
# @generated function foo(a, b)
#     @show a <: Type
#     @show b <: Type
#     :(a, b)
# end
# foo(Array{Float64}, 4) # true, false

# So, the plan is to define a struct, defining the problem.
# This struct will have a field per variable.
# It will implement the LogDensityProblems interface, with methods:
#
# logdensity(resulttype, ℓ, x)
# for resulttype = Value, and ValueGradient
# x are the unconstrained parameters
# and ℓ the struct.
# Additionally, support kwarg version with constrained parameterization.

struct One end
# This should be the only method I have to define.
@inline Base.:*(a, ::One) = a
# But I'll define this one too. Would it be better not to, so that we get errors
# if the adjoint is for some reason multiplied on the right?
@inline Base.:*(::One, a) = a

types_to_vals(::Type{T}) where {T} = Val{T}()
types_to_vals(v) = v
extract_typeval(::Type{Val{T}}) where {T} = T

function generate_generated_funcs_expressions(model_name, expr)
    # Determine the set of variables that are either parameters or data.
    variable_set = determine_variables(expr)
    variables = [v for v ∈ variable_set] # ensure order is constant
    variable_type_names = [Symbol("##Type##", v) for v ∈ variables]

    struct_quote = quote
        struct $model_name{$(variable_type_names...)} <: LogDensityProblems.AbstractLogDensityProblem
            $([:( $(variables[i])::$(variable_type_names[i]) ) for i ∈ eachindex(variables)]...)
        end
    end
    # struct_kwarg_quote = quote
    #     function $model_name(; $([:($(variables[i])::$(variable_type_names[i])) for i ∈ eachindex(variables)]...))
    #         $model_name()
    #     end
    # end
    struct_kwarg_quote = quote
        function $model_name(; $(variables...))
            $model_name($([:(ProbabilityModels.types_to_vals($v)) for v ∈ variables]...))
        end
    end

    # Translate the sampling statements, and then flatten the expression to remove nesting.
    expr = translate_sampling_statements(expr) |>
                flatten_expression# |>
                # ProbabilityModels.rename_assignments

    # for Value definition:
    # flattened expression needs a tracking pass
    #
    # for ValueGradient:
    # diff!

    # both also need unconstraining pass (with and without addition of derivatives)


    # The plan in this definition is to make each keyword arg default to the appropriate field of ℓ
    # This allows one to optionally override in the case of a single evaluation.
    # Additionally, it will throw an error if one of the required fields was not defined.
    # q = quote
    #     @generated function logdensity(::Type{LogDensityProblems.Value},
    #             ℓ::$(model_name){$(variable_type_names...)};
    #             $([:($v = ℓ.$v) for v ∈ variables]...)) where {$(variable_type_names...)}
    #
    #     end
    # end
    V = gensym(:V)
    ℓ = gensym(:ℓ)
    θ = gensym(:θ)
    T = gensym(:T)

    q = quote
        @generated function LogDensityProblems.logdensity(::Type{$V}, $ℓ::$(model_name);
                $([:($(variables[i])::$(variable_type_names[i]) = $ℓ.$(variables[i])) for i ∈ eachindex(variables)]...)
                ) where {$V, $(variable_type_names...)}


            return_partials = $V == LogDensityProblems.ValueGradient
            model_parameters = Symbol[]
            first_pass = quote end
            second_pass = quote end


        end
    end
    q_body = q.args[end].args[end].args[end].args;
    push!(q_body, :(expr = $(Expr(:quote, deepcopy(expr)))))
    Tθ = gensym(:Tθ)
    θq = quote
        @generated function LogDensityProblems.logdensity(::Type{$V}, $ℓ::$(model_name){$(variable_type_names...)}, $θ::$Tθ) where {$V, $T, $Tθ<:AbstractArray{$T}, $(variable_type_names...)}


            return_partials = $V == LogDensityProblems.ValueGradient
            model_parameters = Symbol[]
            first_pass = quote end
            second_pass = quote end


        end
    end
    θq_body = θq.args[end].args[end].args[end].args;
    push!(θq_body, :(expr = $(Expr(:quote, deepcopy(expr)))))
    # Now, we work on assembling the function.


    for i ∈ eachindex(variables)
        push!(q_body, quote
            if $(variable_type_names[i]) <: Val
                push!(model_parameters, $(variables[i]))
            end
        end)
        load_data = Expr(:quote, :($(variables[i]) = $ℓ.$(variables[i])))
        push!(θq_body, quote
            if $(variable_type_names[i]) <: Val
                push!(model_parameters, $(QuoteNode(variables[i])))
                DistributionParameters.load_parameter(first_pass.args, second_pass.args, $(QuoteNode(variables[i])), ProbabilityModels.extract_typeval($(variable_type_names[i])), return_partials)
            else
                push!(first_pass.args, $load_data)
            end
        end)
    end
    push!(q_body, quote
        tracked_vars = Set(model_parameters)
        first_pass, name_dict = ProbabilityModels.rename_assignments(first_pass)
        expr, name_dict = ProbabilityModels.rename_assignments(expr, name_dict)
        θ_sym = $(QuoteNode(θ)) # This creates our symbol θ
        if return_partials
            second_pass, name_dict = ProbabilityModels.rename_assignments(second_pass, name_dict)
            ProbabilityModels.reverse_diff_pass!(first_pass, second_pass, expr, tracked_vars)
            # variable renaming rather than incrementing makes initiazing
            # target to an integer okay.
            expr_out = quote
                # target = 0
                $first_pass
                $(Symbol("###adjoint###", name_dict[:target])) = ProbabilityModels.One()
                $second_pass
                (
                    $(name_dict[:target]),
                    $(Expr(:tuple, [Symbol("###adjoint###", mp) for mp ∈ model_parameters]...))
                )
            end
        else
            ProbabilityModels.constant_drop_pass!(first_pass, expr, tracked_vars)
            expr_out = quote
                # target = 0
                $(Symbol("##θparameter##")) = ProbabilityModels.VectorizationBase.vectorizable($θ_sym)
                $first_pass
                $(name_dict[:target])
            end
        end
        quote
            @fastmath @inbounds begin
                $(ProbabilityModels.first_updates_to_assignemnts(expr_out))
            end
        end
    end)
    push!(θq_body, quote
        tracked_vars = Set(model_parameters)
        first_pass, name_dict = ProbabilityModels.rename_assignments(first_pass)
        expr, name_dict = ProbabilityModels.rename_assignments(expr, name_dict)
        θ_sym = $(QuoteNode(θ)) # This creates our symbol θ
        T_sym = $(QuoteNode(T))
        if return_partials
            second_pass, name_dict = ProbabilityModels.rename_assignments(second_pass, name_dict)
            TLθ = ProbabilityModels.PaddedMatrices.type_length($θ) # This refers to the type of the input
            ProbabilityModels.reverse_diff_pass!(first_pass, second_pass, expr, tracked_vars)
            expr_out = quote
                # target = zero($T_sym)
                $(Symbol("##θparameter##")) = ProbabilityModels.ProbabilityModels.VectorizationBase.vectorizable($θ_sym)
                $first_pass
                $(Symbol("##∂θparameter##m")) = ProbabilityModels.PaddedMatrices.MutableFixedSizePaddedVector{$TLθ,$T_sym}(undef)
                $(Symbol("##∂θparameter##")) = ProbabilityModels.ProbabilityModels.VectorizationBase.vectorizable($(Symbol("##∂θparameter##m")))
                $(Symbol("###adjoint###", name_dict[:target])) = ProbabilityModels.One()
                $second_pass

                $(Symbol("##∂θparameter##mconst")) = ProbabilityModels.PaddedMatrices.ConstantFixedSizePaddedVector($(Symbol("##∂θparameter##m")))
                LogDensityProblems.ValueGradient(
                    isfinite($(name_dict[:target])) ? (all(isfinite, $(Symbol("##∂θparameter##mconst"))) ? $(name_dict[:target]) : $T_sym(-Inf)) : $T_sym(-Inf),
                    $(Symbol("##∂θparameter##mconst"))
                )
            end
        else
            ProbabilityModels.constant_drop_pass!(first_pass, expr, tracked_vars)
            expr_out = quote
                # target = zero($T_sym)
                $(Symbol("##θparameter##")) = ProbabilityModels.VectorizationBase.vectorizable($θ_sym)
                $first_pass
                LogDensityProblems.Value( $(name_dict[:target]) )
            end
        end
        quote
            @fastmath @inbounds begin
                # @inbounds begin
                $(ProbabilityModels.first_updates_to_assignemnts(expr_out, model_parameters))
            end
        end
    end)


    # Now, we would like to apply
    # reverse_diff_pass(expr, gradient_targets)

    dim_q = quote
        @generated function LogDensityProblems.dimension(::$(model_name){$(variable_type_names...)}) where {$(variable_type_names...)}
            dim = 0
            $([quote
                if $v <: Val
                    dim += ProbabilityModels.PaddedMatrices.type_length(ProbabilityModels.extract_typeval($v))
                end
            end for v ∈ variable_type_names]...)
            ProbabilityModels.PaddedMatrices.Static{dim}()
        end
    end


    struct_quote, struct_kwarg_quote, q, θq, dim_q, variables
end

macro model(model_name, expr)
    struct_quote, struct_kwarg_quote, q, θq, dim_q, variables = generate_generated_funcs_expressions(model_name, expr)

    printstring = """
        Defined model: $model_name.
        Unknowns: $(variables[1])$([", " * string(variables[i]) for i ∈ 2:length(variables)]...).
    """
    esc(quote
        $struct_quote; $struct_kwarg_quote; $θq; $dim_q;
        println($printstring)
    end)
end
