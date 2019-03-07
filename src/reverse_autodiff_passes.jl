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
    postwalk(expr) do x
        if @capture(x, for i_ ∈ iter_ body_ end)
            throw("Loops not yet supported!")
            # reverse_diff_loop_pass!(first_pass, second_pass, i, iter, body, expr, tracked_vars)
        elseif @capture(x, out_ = f_(A__))
            diff!(first_pass, second_pass, tracked_vars, out, f, A)
        elseif @capture(x, out_ = A_) && isa(A, Symbol)
            push!(first_pass.args, x)
            pushfirst!(second_pass.args, :( $(Symbol("###seed###", A)) += $(Symbol("###seed###", out)) ))
        else
            push!(first_pass, x)
        end
        x
    end
end
# function reverse_diff_pass!(first_pass, second_pass, expr, tracked_vars)
#     for x ∈ expr.args
#         if @capture(x, for i_ ∈ iter_ body_ end)
#             throw("Loops not yet supported!")
#             # reverse_diff_loop_pass!(first_pass, second_pass, i, iter, body, expr, tracked_vars)
#         elseif @capture(x, out_ = f_(A__))
#             diff!(first_pass, second_pass, tracked_vars, out, f, A)
#         # elseif @capture(x, out_ = A_) && (isa(A,Symbol))
#         #     push!(first_pass, x)
#         #     pushfirst!(second_pass, :())
#         else
#             push!(first_pass, x)
#         end
#     end
# end


function apply_diff_rule!(first_pass, second_pass, tracked_vars, out, f, A, diffrules::NTuple{N}) where {N}
    track_out = false
    push!(first_pass.args, :($out = $f($(A...))))
    seedout = Symbol("###seed###", out)
    for i ∈ eachindex(A)
        a = A[i]
        a ∈ tracked_vars || continue
        track_out = true
        ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
        push!(first_pass.args, :($∂ = $(diffrules[i])))
        pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) += $∂ * $seedout ))
    end
    track_out && push!(tracked_vars, out)
    nothing
end
function apply_diff_rule!(first_pass, second_pass, tracked_vars, out, f, A, diffrule)
    length(A) == 1 || throw("length(A) == $(length(A)); must equal 1 when passed diffrules are: $(diffrule)")
    track_out = false
    push!(first_pass, :($out = $f($(A...))))
    seedout = Symbol("###seed###", out)
    for i ∈ eachindex(A)
        A[i] ∈ tracked_vars || continue
        track_out = true
        ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
        push!(first_pass.args, :($∂ = $(diffrule)))
        pushfirst!(second_pass.args, :( $(Symbol("###seed###", A[i])) += $∂ * $seedout ))
    end
    track_out && push!(tracked_vars, out)
    nothing
end


"""
This function applies reverse mode AD.

"A" lists the arguments of the function "f", while "tracked_vars" is a set
of all variables being tracked (those with respect to which we need derivatives).

out is the name of the output variable. Assuming at least one argument is tracked,
out will be added to the set of tracked variables.

"first_pass" and "second_pass" are expressions to which the AD with resect to "f" and "A"
will be added.
"first_pass" is an expression of the forward pass, while
"second_pass" is an expression for the reverse pass.
"""
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
        zygote_diff_rule!(first_pass, second_pass, tracked_vars, out, A, f)
        # throw("Fall back differention rules not yet implemented, and no method yet to handle $f($(A...))")
    end
end

@noinline outlinederror(x) = error(x)
function zygote_diff_rule!(first_pass, second_pass, tracked_vars, out, A, f)
    track = Symbol[]
    func_args = Symbol[]
    anon_args = Expr(:tuple)
    adjoints = Expr(:tuple)
    seedout = Symbol("###seed###", out)
    for i ∈ eachindex(A)
        a = A[i]
        if a ∉ tracked_vars
            push!(func_args, a)
            continue
        end
        push!(track, a)
        ga = gensym(a)
        push!(func_args, ga)
        push!(anon_args.args, ga)
        ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
        push!(adjoints.args, ∂)
        pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) += $∂ * $seedout ))
    end
    if length(track) > 0
        push!(tracked_vars, out)
    else
        push!(first_pass.args, :($out = $f($(A...))))
        return nothing
    end
    back = gensym(:back)
    if length(A) == length(anon_args)
        # we'll be easy on the compiler, and not create an anonymous function
        push!(first_pass.args, quote
            $out, $back = forward($f, $(A...)))
            $out isa Real || outlinederror("Function output is not scalar")
            $adjoints = $back(Int8(1))
        end)
    else
        push!(first_pass.args, quote
            $out, $back = forward($anon_args -> $f($(func_args...)), $(track...))
            $out isa Real || outlinederror("Function output is not scalar")
            $adjoints = $back(Int8(1))
        end)
    end
    nothing
end
