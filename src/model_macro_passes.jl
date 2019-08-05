

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
        elseif @capture(x, LHS_ = RHS_)
            push!(ignored_symbols, LHS)
            return x
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
                    return :(target = vadd(target, $(Symbol(f,:_fmadd))($y, $X, $β, $α, $(θ...))))
                elseif @capture(x, y_ ~ f_(X_ * β_ + α_, θ__))
                    return :(target = vadd(target, $(Symbol(f,:_fmadd))($y, $X, $β, $α, $(θ...))))
                elseif @capture(x, y_ ~ f_(α_ - X_ * β_, θ__))
                    return :(target = vadd(target, $(Symbol(f,:_fnmadd))($y, $X, $β, $α, $(θ...))))
                elseif @capture(x, y_ ~ f_(X_ * β_ - α_, θ__))
                    return :(target = vadd(target, $(Symbol(f,:_fmsub))($y, $X, $β, $α, $(θ...))))
                elseif @capture(x, y_ ~ f_(- X_ * β_ - α_, θ__))
                    return :(target = vadd(target, $(Symbol(f,:_fnmsub))($y, $X, $β, $α, $(θ...))))
                elseif @capture(x, y_ ~ f_( - α_ - X_ * β_, θ__))
                    return :(target = vadd(target, $(Symbol(f,:_fnmsub))($y, $X, $β, $α, $(θ...))))
                end
            end
            return :(target = vadd(target, $f($y, $(θ...))))
#            return :(target = DistributionParameters.add(target, $f($y, $(θ...))))
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


#=
function flatten_subexpression!(flattened, args::AbstractVector)
    map(args) do arg
        flatten_subexpression!(flattened, arg)
    end
end
function flatten_subexpression!(flattened, arg::Expr)
    postwalk(arg) do x
#        @show x
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

To be replaced with Meta.lower(ProbabilityModels, expr)
"""
function flatten_expression(expr)
#    println(expr)
#    println("\n\n\n")
    q_flattened = @q begin end
    flattened = q_flattened.args
    for ex ∈ expr.args
        if ex isa Symbol
            continue
        elseif @capture(ex, for i_ ∈ iter_ body_ end)
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
        elseif ex isa Expr && ex.head = :tuple
            
#        else
#            @show ex
        end
    end
#    println(q_flattened)
    q_flattened
end
=#
ssa_sym(i::Int) = Symbol("##SSAValue##$(i)##")
function ssa_to_sym(expr)
    postwalk(expr) do ex
        if ex isa Core.SSAValue
            return ssa_sym(ex.id)
        elseif ex isa GlobalRef
            return Expr(:., Symbol(ex.mod), QuoteNode(ex.name))
        else
            return ex
        end
    end
end
function flatten_expression(expr)
    lowered_array = Meta.lower(ProbabilityModels, expr).args
    @assert length(lowered_array) == 1
    lowered = first(lowered_array).code
    q = quote end
    for i ∈ 1:length(lowered) - 1 # skip return statement
        ex = lowered[i]
        if @capture(ex, a_ = b_)
            push!(q.args, :($a = $(ssa_to_sym(b))))
        else # assigns to ssa value
            push!(q.args, :($(ssa_sym(i)) = $(ssa_to_sym(ex))))
        end
    end
#    println(q)
    q
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

This pass skips assignments involving the following functions:
PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED
PaddedMatrices.RESERVED_DECREMENT_SEED_RESERVED

This pass returns the expression in a static single assignment (SSA) form.
"""
function rename_assignments(expr, vars = Dict{Symbol,Symbol}())
    postwalk(expr) do ex
        if @capture(ex, a_ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(args__)) || @capture(ex, a_ = ProbabilityModels.PaddedMatrices.RESERVED_DECREMENT_SEED_RESERVED(args__))
            return ex
        elseif @capture(ex, a_ = b_)
            if isa(b, Expr)
                c = postwalk(x -> get(vars, x, x), b)
            else
                c = b
            end
            if isa(a, Symbol) && !MacroTools.isgensym(a)
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
        elseif @capture(ex, if cond_; conditionaleval_ end)
            conditional = postwalk(x -> get(vars, x, x), cond)
            conditionaleval, tracked_vars = rename_assignments(conditionaleval, TrackedDict(vars))
            else_expr = quote end
            for (k, (vbase,vfinal)) ∈ tracked_vars.reassigned
                push!(else_expr.args, :($vfinal = $vbase))
                vars[k] = vfinal
            end
            return quote
                if $conditional
                    $conditionaleval
                else
                    $else_expr
                end
            end
        elseif @capture(ex, if cond_; conditionaleval_; else; alternateeval_ end)
            conditional = postwalk(x -> get(vars, x, x), cond)
            if_conditionaleval, if_tracked_vars = rename_assignments(conditionaleval, TrackedDict(vars))
            else_conditionaleval, else_tracked_vars = rename_assignments(alternateeval, TrackedDict(vars))
            for (k, (vbase,vfinal)) ∈ if_tracked_vars.reassigned
                if haskey(else_tracked_vars.reassigned, k)
                    push!(else_conditionaleval.args, :($vfinal = $(else_tracked_vars.reassigned[k])))
                else
                    push!(else_conditionaleval.args, :($vfinal = $vbase))
                end
                vars[k] = vfinal
            end
            for (k, (vbase,vfinal)) ∈ else_tracked_vars.reassigned
                haskey(if_tracked_vars.reassigned, k) && continue
                push!(if_conditionaleval.args, :($vfinal = $vbase))
                vars[k] = vfinal
            end
            for k ∈ union(keys(if_tracked_vars.newlyassigned),keys(else_tracked_vars.newlyassigned))
                canonical_name = if_tracked_vars.newlyassigned[k]
                vars[k] = canonical_name
                push!(else_expr.args, :($canonical_name = $(else_tracked_vars.newlyassigned[k])))
            end
            return quote
                if $conditional
                    $if_conditionaleval
                else
                    $else_conditionaleval
                end
            end
        else
            return ex
        end
    end, vars
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
    # Note that this function is recursive.
    # variables_input must NOT be a Set, otherwise that set will be mutated.
    # The idea is to call it with something other than a set.
    # That is then used to construct a set.
    # When called recursively, it will continue to build up said set.
    if isa(variables_input, Set)
        variables = variables_input
    else
        variables = Set(variables_input)
    end
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
        # @show new_assignment
        push!(variables, lhs)
        if ex.head == :(=)
            check = false
            for j ∈ 2:length(ex.args)
                if isa(ex.args[j], Expr) && ex.args[j].head == :block
                    ex.args[j] = first_updates_to_assignemnts(ex.args[j], variables)
                    continue
                end
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
            # @show lhs, new_assignment, check
            if check
                if @capture(ex, a_ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(b__, a_) )
                    expr.args[i] = :($a = ProbabilityModels.PaddedMatrices.RESERVED_MULTIPLY_SEED_RESERVED($(b...)))
                elseif @capture(ex, a_ = ProbabilityModels.PaddedMatrices.RESERVED_DECREMENT_SEED_RESERVED(b__, a_) )
                    expr.args[i] = :($a = ProbabilityModels.PaddedMatrices.RESERVED_NMULTIPLY_SEED_RESERVED($(b...)))
                elseif @capture(ex, a_ = a_ + b__ ) || @capture(ex, a_ = b__ + a_ ) || @capture(ex, a_ = Base.FastMath.add_fast(a_, b__)) || @capture(ex, a_ = Base.FastMath.add_fast(b__, a_))# || @capture(ex, a_ = b__ )
                    expr.args[i] = :($a = $(b...))
                elseif @capture(ex, a_ = a_ - b__ )
                    expr.args[i] = :($a = -1 * $(b...))
                elseif @capture(ex, a_ = SIMDPirates.vifelse(b_, LoopVectorization.SIMDPirates.vadd(a_, c_), a_))
                    expr.args[i] = :($a = SIMDPirates.vifelse($b, $c, 0.0))
#                elseif @capture(ex, target = ProbabilityModels.DistributionParameters.add(a_, b_) )
#                    expr.args[i] = :( a_ = a_ )
                else
                    println(expr)
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
#                printstring = "distribution $f (ret: $out): "
#                push!(first_pass.args, :(println($printstring, $out)))
#                push!(first_pass.args, :(@show $(A...)))
#                push!(first_pass.args, :(@show $out))
            # elseif f ∈ keys(SPECIAL_DIFF_RULES)

            else
                for a ∈ A
                    if a ∈ tracked_vars
                        push!(tracked_vars, out)
                        break
                    end
                end
                if f == :add
                    push!(first_pass.args, :( $out = ProbabilityModels.SIMDPirates.vadd($(A...))))
                else
                    push!(first_pass.args, x)
                end
            end
        elseif @capture(x, out_ = A__)
            for a ∈ A
                if a ∈ tracked_vars
                    push!(tracked_vars, out)
                    break
                end
            end
            push!(first_pass.args, x)            
        else
            push!(first_pass.args, x)
        end
    end
    first_pass
end


types_to_vals(::Type{T}) where {T} = Val{T}()
types_to_vals(v) = v
extract_typeval(::Type{Val{T}}) where {T} = T
extract_typeval(::Type{Val{Type{T}}}) where {T} = T

function load_and_constrain_quote(ℓ, model_name, variables, variable_type_names, θ, Tθ, T)
    Nparam = gensym(:number_parameters)
    logjac = gensym(:logjac)
    θq = quote
        @generated function ProbabilityModels.DistributionParameters.constrain($ℓ::$(model_name){$Nparam, $(variable_type_names...)}, $θ::AbstractVector{$T}, ::Val{$logjac} = Val{false}()) where {$Nparam, $T, $logjac, $(variable_type_names...)}


            return_partials = false
#            model_parameters = Symbol[]
            first_pass = quote end
#            push!(first_pass.args, Expr(:(=), :target, Expr(:call, :(ProbabilityModels.DistributionParameters.initialize_target), $(QuoteNode(T)))))
                                
            push!(first_pass.args, Expr(:(=), Symbol("##θparameter##"), Expr(:call, :(ProbabilityModels.VectorizationBase.vectorizable), $(QuoteNode(θ)))))
            second_pass = quote end

            transformed_params = Expr(:tuple,)
            return_expr = Expr(:if, $(QuoteNode(logjac)), Expr(:tuple, transformed_params, :target), transformed_params)

        end
    end
    v = Symbol("##storage_vector##")
    θq_vec = quote
        @generated function ProbabilityModels.DistributionParameters.constrain!($v::AbstractVector{$T}, $ℓ::$(model_name){$Nparam, $(variable_type_names...)}, $θ::AbstractVector{$T}) where {$Nparam, $T, $(variable_type_names...)}

            return_partials = false
#            model_parameters = Symbol[]
            first_pass = quote end
            push!(first_pass.args, Expr(:(=), Symbol("##stack_pointer##"), Expr(:call, :(PaddedMatrices.StackPointer), Expr(:call, :pointer, $(QuoteNode(Symbol("##storage_vector##")))))))
#            push!(first_pass.args, :(origpointer = pointer($(Symbol("##stack_pointer##")), Float64)))
 #           push!(first_pass.args, Expr(:(=), :target, Expr(:call, :(ProbabilityModels.DistributionParameters.initialize_target), $(QuoteNode(T)))))
                                
            push!(first_pass.args, Expr(:(=), Symbol("##θparameter##"), Expr(:call, :(ProbabilityModels.VectorizationBase.vectorizable), $(QuoteNode(θ)))))
            second_pass = quote end

        end
    end
                                                                
    param_names = gensym(:param_names)
    param_names_quote = quote            
        function ProbabilityModels.DistributionParameters.parameter_names($ℓ::$(model_name){$Nparam, $(variable_type_names...)}) where {$Nparam, $(variable_type_names...)}

            $param_names = String[]
            $([quote
               if $v <: Val
               # @show $v
                   append!($param_names, ProbabilityModels.DistributionParameters.parameter_names(ProbabilityModels.extract_typeval($v), $(QuoteNode(v)))::Vector{String})
               end
               end for v ∈ variable_type_names]...)
            $param_names
            
                                
        end
    end

    
    Nconstrainedparam = Symbol("##number_of_parameters##")
    constrained_length_quote = quote
        @generated function ProbabilityModels.DistributionParameters.constrained_length($ℓ::$(model_name){$Nparam, $(variable_type_names...)}) where {$Nparam, $(variable_type_names...)}

            $Nconstrainedparam = 0
            $([quote
               if $v <: Val
               # @show $v
               $Nconstrainedparam += ProbabilityModels.PaddedMatrices.type_length(ProbabilityModels.extract_typeval($v))
               end
               end for v ∈ variable_type_names]...)
            $Nconstrainedparam
        end
    end
    θq_body = θq.args[end].args[end].args[end].args;
    θq_vec_body = θq_vec.args[end].args[end].args[end].args;
   # param_name_body = param_names_quote.args[end].args[end].args[end].args;
    # push!(θq_body, :(expr = $(Expr(:quote, deepcopy(expr)))))
    # Now, we work on assembling the function.


    for i ∈ eachindex(variables)
        # push!(q_body, quote
        #     if $(variable_type_names[i]) <: Val
        #         push!(model_parameters, $(variables[i]))
        #     end
        # end)
        load_data = Expr(:quote, :($(variables[i]) = $ℓ.$(variables[i])))
        push!(θq_body, quote
            # @show $(variable_type_names[i])
            if $(variable_type_names[i]) <: Val
                # println($(variable_type_names[i]), "isa Val.")
#                push!(model_parameters, $(QuoteNode(variables[i])))
                ProbabilityModels.DistributionParameters.load_parameter!(first_pass.args, second_pass.args, $(QuoteNode(variables[i])), ProbabilityModels.extract_typeval($(variable_type_names[i])), false, ProbabilityModels, nothing, $logjac)
                push!(transformed_params.args, Expr(:(=), $(QuoteNode(variables[i])), $(QuoteNode(variables[i]))))
            end
        end)
        push!(θq_vec_body, quote
            # @show $(variable_type_names[i])
            if $(variable_type_names[i]) <: Val
                # println($(variable_type_names[i]), "isa Val.")
#                push!(model_parameters, $(QuoteNode(variables[i])))
                ProbabilityModels.DistributionParameters.load_parameter!(first_pass.args, second_pass.args, $(QuoteNode(variables[i])), ProbabilityModels.extract_typeval($(variable_type_names[i])), false, ProbabilityModels, Symbol("##stack_pointer##"), false)
                if ProbabilityModels.extract_typeval($(variable_type_names[i])) <: DistributionParameters.ScalarParameter
              push!(first_pass.args, Expr(:call, :(ProbabilityModels.VectorizationBase.store!),
                                          Expr(:call, :pointer, Symbol("##stack_pointer##"), $(QuoteNode(T))), $(QuoteNode(variables[i]))))
                    push!(first_pass.args, Expr(:(+=), Symbol("##stack_pointer##"), Expr(:call, :sizeof, $(QuoteNode(T)) )))
#                    push!(first_pass.args, :(VectorizationBase.store!(pointer(sp, $(QuoteNode(T))), $(QuoteNode(variables[i]))) ))
#                    push!(first_pass.args, Expr(:(+=), Symbol("##stack_pointer##"), Expr(:call, :sizeof, $(QuoteNode(T)) )))
              end
#              push!(first_pass.args, :(@show reinterpret(Int, pointer($(Symbol("##stack_pointer##")),Float64) - origpointer)>>3 ))
#                push!(transformed_params.args, Expr(:(=), $(QuoteNode(variables[i])), $(QuoteNode(variables[i]))))
              
            end
       end)
    end
    push!(θq_body, :(push!(first_pass.args, return_expr)))
    push!(θq_vec_body, Expr(:call, :push!, :(first_pass.args), QuoteNode(Symbol("##storage_vector##"))))

#    push!(θq_body, :(println(ProbabilityModels.MacroTools.striplines(first_pass))))
#    push!(θq_vec_body, :(println(ProbabilityModels.MacroTools.striplines(first_pass))))
    push!(θq_body, :(first_pass))
    push!(θq_vec_body, :(first_pass))


    θq, θq_vec, param_names_quote, constrained_length_quote
end

@inline function new_gradient(vgb::LogDensityProblems.ValueGradientBuffer, x::AbstractVector)
    vgb.buffer
end
@inline function new_gradient(vg, x::AbstractVector)
    PaddedMatrices.mutable_similar(x)
end

function generate_generated_funcs_expressions(model_name, expr)
    # Determine the set of variables that are either parameters or data.
    variable_set = determine_variables(expr)
    variables = [v for v ∈ variable_set] # ensure order is constant
    variable_type_names = [Symbol("##Type##", v) for v ∈ variables]

    Nparam = Symbol("##number_of_parameters##")
    stride = gensym(:LDA)
    L1 = gensym(:L)
    L2 = gensym(:L)
    T = gensym(:T)

    var_vartype_pairs = [:( $(variables[i])::$(variable_type_names[i]) ) for i ∈ eachindex(variables)]
    struct_quote = quote
        struct $model_name{$Nparam, $(variable_type_names...)} <: ProbabilityModels.AbstractProbabilityModel{$Nparam}
            # ∇RESERVED::PaddedMatrices.MutableFixedSizePaddedVector{$Nparam,$T,$stride,$L1}
            # ΣRESERVED::PaddedMatrices.MutableFixedSizePaddedMatrix{$Nparam,$Nparam,$T,$stride,$L2}
            $(var_vartype_pairs...)
        end
    end
    # struct_kwarg_quote = quote
    #     function $model_name(; $([:($(variables[i])::$(variable_type_names[i])) for i ∈ eachindex(variables)]...))
    #         $model_name()
    #     end
    # end
    struct_kwarg_quote = quote
        function $model_name(; $(var_vartype_pairs...)) where {$(variable_type_names...)}
            $Nparam = 0
            # @show $(variables...)
            # @show $(variable_type_names...)
            $([quote
                if isa(ProbabilityModels.types_to_vals($v), Val)
                    # @show $v
                    $Nparam += ProbabilityModels.PaddedMatrices.param_type_length($v)
                end
            end for v ∈ variables]...)
            $model_name{$Nparam}($([:(ProbabilityModels.types_to_vals($v)) for v ∈ variables]...))
        end
        function $model_name{$Nparam}( $(var_vartype_pairs...)) where {$Nparam, $(variable_type_names...)}
            $model_name{$Nparam,$(variable_type_names...)}($(variables...))
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
    Tθ = gensym(:Tθ)

    # q = quote
    #     @generated function LogDensityProblems.logdensity(::Type{$V}, $ℓ::$(model_name);
    #             $([:($(variables[i])::$(variable_type_names[i]) = $ℓ.$(variables[i])) for i ∈ eachindex(variables)]...)
    #             ) where {$V, $(variable_type_names...)}
    #
    #
    #         return_partials = $V == LogDensityProblems.ValueGradient
    #         model_parameters = Symbol[]
    #         first_pass = quote end
    #         second_pass = quote end
    #
    #
    #     end
    # end
    # q_body = q.args[end].args[end].args[end].args;
    # push!(q_body, :(expr = $(Expr(:quote, deepcopy(expr)))))

    constrain_quote, cvq, pn, clq  = load_and_constrain_quote(ℓ, model_name, variables, variable_type_names, θ, Tθ, T)

    # we have to split these, because of dispatch ambiguity errors
    θq_value = quote
        @generated function ProbabilityModels.LogDensityProblems.logdensity(::Type{ProbabilityModels.LogDensityProblems.Value}, $ℓ::$(model_name){$Nparam, $(variable_type_names...)}, $θ::AbstractVector{$T}) where {$Nparam, $T, $(variable_type_names...)}


            TLθ = $Nparam
            return_partials = false
            model_parameters = Symbol[]
            first_pass = quote end
            second_pass = quote end


        end
    end
    vgb = Symbol("##vgb##")
    θq_valuegradient = quote
        @generated function ProbabilityModels.LogDensityProblems.logdensity(
                        $vgb::Type{ProbabilityModels.LogDensityProblems.ValueGradient},
                        $ℓ::$(model_name){$Nparam, $(variable_type_names...)},
                        $θ::AbstractVector{$T}
                    ) where {$Nparam, $T, $(variable_type_names...)}


            TLθ = $Nparam
            return_partials = true
            model_parameters = Symbol[]
            first_pass = quote end
            second_pass = quote end


        end
    end
    θq_valuegradientbuffer = quote
        @generated function ProbabilityModels.LogDensityProblems.logdensity(
                        $vgb::ProbabilityModels.LogDensityProblems.ValueGradientBuffer,
                        $ℓ::$(model_name){$Nparam, $(variable_type_names...)},
                        $θ::AbstractVector{$T}
                    ) where {$Nparam, $T, $(variable_type_names...)}


            TLθ = $Nparam
            return_partials = true
            model_parameters = Symbol[]
            first_pass = quote end
            second_pass = quote end


        end
    end
    for (θq, return_partials) ∈ ((θq_value, false), (θq_valuegradient, true), (θq_valuegradientbuffer, true))

        θq_body = θq.args[end].args[end].args[end].args;
        push!(θq_body, :(expr = $(Expr(:quote, deepcopy(expr)))))
        # Now, we work on assembling the function.


        for i ∈ eachindex(variables)
            # push!(q_body, quote
            #     if $(variable_type_names[i]) <: Val
            #         push!(model_parameters, $(variables[i]))
            #     end
            # end)
            load_data = Expr(:quote, :($(variables[i]) = $ℓ.$(variables[i])))
            missingvar = Symbol("##missing##", variables[i])
            incompletevar = Symbol("##incomplete##", variables[i])
            load_incomplete_data = Expr(:quote, :($incompletevar = $ℓ.$(variables[i])))
            push!(θq_body, quote
                if $(variable_type_names[i]) <: Val
                    push!(model_parameters, $(QuoteNode(variables[i])))
                  ProbabilityModels.DistributionParameters.load_parameter!(
                      first_pass.args, second_pass.args,
                      $(QuoteNode(variables[i])), ProbabilityModels.extract_typeval($(variable_type_names[i])),
                      $return_partials, ProbabilityModels, Symbol("##stack_pointer##")
                  )
                    # if !$return_partials
                    #     push!(first_pass.args, Expr(:call, :println, $(string(variables[i]))))
                    #     push!(first_pass.args, Expr(:call, :println, $(QuoteNode(variables[i]))))
                  # end
                elseif $(variable_type_names[i]) <: ProbabilityModels.DistributionParameters.MissingDataArray
                    push!(model_parameters, $(QuoteNode(missingvar)))
                    push!(first_pass.args, $load_incomplete_data)
                  ProbabilityModels.DistributionParameters.load_parameter!(
                      first_pass.args, second_pass.args, $(QuoteNode(variables[i])), ($(variable_type_names[i])),
                      $return_partials, ProbabilityModels, Symbol("##stack_pointer##")
                  )
                else
                    push!(first_pass.args, $load_data)
                end
            end)
        end

        if return_partials
            # qprocessing = quote
            #     second_pass, name_dict = ProbabilityModels.rename_assignments(second_pass, name_dict)
            #     ProbabilityModels.reverse_diff_pass!(first_pass, second_pass, expr, tracked_vars)
            #     # variable renaming rather than incrementing makes initiazing
            #     # target to an integer okay.
            #     expr_out = quote
            #         # target = 0
            #         $first_pass
            #         $(Symbol("###seed###", name_dict[:target])) = ProbabilityModels.One()
            #         $second_pass
            #         (
            #             $(name_dict[:target]),
            #             $(Expr(:tuple, [Symbol("###seed###", mp) for mp ∈ model_parameters]...))
            #         )
            #     end
            # end

            processing = quote
                # println("About to rename assginemtns:\n")
                # display(second_pass)
#                second_pass, name_dict = ProbabilityModels.rename_assignments(second_pass, name_dict)
                # println("\nRenamed assginemtns:\n")
                # display(second_pass)
                # TLθ = ProbabilityModels.PaddedMatrices.type_length($θ) # This refers to the type of the input
                # TLθ = ProbabilityModels.PaddedMatrices.tonumber(ProbabilityModels.DynamicHMC.dimension($ℓ))
                ProbabilityModels.reverse_diff_pass!(first_pass, second_pass, expr, tracked_vars)
 #               first_pass = PaddedMatrices.stack_pointer_pass(first_pass, $(QuoteNode(Symbol("##stack_pointer##"))))
 #               second_pass = PaddedMatrices.stack_pointer_pass(second_pass, $(QuoteNode(Symbol("##stack_pointer##"))))
                expr_out = quote
                    target = ProbabilityModels.DistributionParameters.initialize_target($T_sym)
#                    target = zero(ProbabilityModels.DistributionParameters.Target{$T_sym})
                    $(Symbol("##θparameter##")) = ProbabilityModels.VectorizationBase.vectorizable($θ_sym)
                    $(Symbol("##stack_pointer##")) = ProbabilityModels.STACK_POINTER
                    $first_pass
                    # $(Symbol("##∂θparameter##m")) = $ℓ_sym.∇RESERVED
                    # $(Symbol("##∂θparameter##m")) = ProbabilityModels.PaddedMatrices.MutableFixedSizePaddedVector{$TLθ,$T_sym}(undef)
                    # $(2TLθ > ProbabilityModels.VectorizationBase.REGISTER_SIZE ? )
                    # $(Symbol("##∂θparameter##m")) = ProbabilityModels.PaddedMatrices.mutable_similar($θ_sym)
                    # $(Symbol("##∂θparameter##m")) = $(Symbol("##vgb##")).buffer
                    $(Symbol("##∂θparameter##m")) = ProbabilityModels.new_gradient($(Symbol("##vgb##")), $θ_sym)
                    $(Symbol("##∂θparameter##")) = ProbabilityModels.VectorizationBase.vectorizable($(Symbol("##∂θparameter##m")))
                    $(Symbol("###seed###", name_dict[:target])) = ProbabilityModels.One()
                    # $(Symbol("###seed###", name_dict[:target])) = ProbabilityModels.Reducer{:row}()
                    $second_pass

                    # LogDensityProblems.ValueGradient(
                    #     isfinite($(name_dict[:target])) ? (all(isfinite, $(Symbol("##∂θparameter##m"))) ? $(name_dict[:target]) : $T_sym(-Inf)) : $T_sym(-Inf),
                    #     $(Symbol("##∂θparameter##m"))
                    # )
                    #
                    # $(Symbol("##∂θparameter##mconst")) = ProbabilityModels.PaddedMatrices.ConstantFixedSizePaddedVector($(Symbol("##∂θparameter##m")))
                    # LogDensityProblems.ValueGradient(
                    #     isfinite($(name_dict[:target])) ? (all(isfinite, $(Symbol("##∂θparameter##mconst"))) ? $(name_dict[:target]) : $T_sym(-Inf)) : $T_sym(-Inf),
                    #     $(Symbol("##∂θparameter##mconst"))
                    # )
                end
                # VectorizationBase.REGISTER_SIZE is in bytes, so this is asking if 4 registers can hold the parameter vector
                
                if 2TLθ > ProbabilityModels.VectorizationBase.REGISTER_SIZE
                    push!(expr_out.args, quote
                          $(Symbol("##scalar_target##")) = ProbabilityModels.SIMDPirates.vsum($(name_dict[:target]))
#                          @show $(Symbol("##scalar_target##")) 
                          ProbabilityModels.LogDensityProblems.ValueGradient(
                              isfinite($(Symbol("##scalar_target##"))) ? (all(isfinite, $(Symbol("##∂θparameter##m"))) ? $(Symbol("##scalar_target##")) : $T_sym(-Inf)) : $T_sym(-Inf),  $(Symbol("##∂θparameter##m"))
                        )
                    end)
                else
                    push!(expr_out.args, quote
                          $(Symbol("##scalar_target##")) = ProbabilityModels.SIMDPirates.vsum($(name_dict[:target]))
                          $(Symbol("##∂θparameter##mconst")) = ProbabilityModels.PaddedMatrices.ConstantFixedSizePaddedVector($(Symbol("##∂θparameter##m")))
                          ProbabilityModels.LogDensityProblems.ValueGradient(
                              isfinite($(Symbol("##scalar_target##"))) ? (all(isfinite, $(Symbol("##∂θparameter##mconst"))) ? $(Symbol("##scalar_target##")) : $T_sym(-Inf)) : $T_sym(-Inf), $(Symbol("##∂θparameter##mconst"))
                        )
                    end)
                end
            end
        else
            # qprocessing = quote
            #     ProbabilityModels.constant_drop_pass!(first_pass, expr, tracked_vars)
            #     expr_out = quote
            #         # target = 0
            #         $(Symbol("##θparameter##")) = ProbabilityModels.VectorizationBase.vectorizable($θ_sym)
            #         $first_pass
            #         $(name_dict[:target])
            #     end
            # end
            processing = quote
                ProbabilityModels.constant_drop_pass!(first_pass, expr, tracked_vars)
#                first_pass = PaddedMatrices.stack_pointer_pass(first_pass, $(QuoteNode(Symbol("##stack_pointer##"))))
                expr_out = quote
                    target = ProbabilityModels.DistributionParameters.initialize_target($T_sym)
#                    target = zero(ProbabilityModels.DistributionParameters.Target{$T_sym})
                    $(Symbol("##θparameter##")) = ProbabilityModels.VectorizationBase.vectorizable($θ_sym)
                    $(Symbol("##stack_pointer##")) = ProbabilityModels.STACK_POINTER
                    $first_pass
                    $(Symbol("##scalar_target##")) = ProbabilityModels.SIMDPirates.vsum($(name_dict[:target]))
                    ProbabilityModels.LogDensityProblems.Value( isfinite($(Symbol("##scalar_target##"))) ? $(Symbol("##scalar_target##")) : $T_sym(-Inf) )
                end
            end
        end

        # push!(q_body, quote
        #     tracked_vars = Set(model_parameters)
        #     first_pass, name_dict = ProbabilityModels.rename_assignments(first_pass)
        #     expr, name_dict = ProbabilityModels.rename_assignments(expr, name_dict)
        #     θ_sym = $(QuoteNode(θ)) # This creates our symbol θ
        #     $qprocessing
        #     quote
        #         @fastmath @inbounds begin
        #             $(ProbabilityModels.first_updates_to_assignemnts(expr_out, model_parameters))
        #         end
        #     end
        # end)
        push!(θq_body, quote
            tracked_vars = Set(model_parameters)
#            first_pass, name_dict = ProbabilityModels.rename_assignments(first_pass)
#              expr, name_dict = ProbabilityModels.rename_assignments(expr, name_dict)
              name_dict = Dict(:target => :target)
            θ_sym = $(QuoteNode(θ)) # This creates our symbol θ
            T_sym = $(QuoteNode(T))
            ℓ_sym = $(QuoteNode(ℓ))
            $processing
            final_quote = quote
                # @fastmath @inbounds begin
                    @inbounds begin
                    $(PaddedMatrices.stack_pointer_pass(ProbabilityModels.first_updates_to_assignemnts(expr_out, model_parameters), $(QuoteNode(Symbol("##stack_pointer##")))))
                end
              end
#              println(MacroTools.striplines(final_quote))
##              println(final_quote)
##              println(ProbabilityModels.MacroTools.prettify(final_quote))
##              println(tracked_vars)
##            display(ProbabilityModels.MacroTools.prettify(final_quote))
            final_quote
        end)

        # push!(θq_body, display())
    end


    # Now, we would like to apply
    # reverse_diff_pass(expr, gradient_targets)

    # dim_q = quote
        # @generated function ProbabilityModels.LogDensityProblems.dimension(::Type{$(model_name){$(variable_type_names...)}}) where {$(variable_type_names...)}
        #     dim = 0
        #     $([quote
        #         if $v <: Val
        #             dim += ProbabilityModels.PaddedMatrices.param_type_length(ProbabilityModels.extract_typeval($v))
        #         end
        #     end for v ∈ variable_type_names]...)
        #     ProbabilityModels.PaddedMatrices.Static{dim}()
        #     # dim
        # end
        # @generated function ProbabilityModels.LogDensityProblems.dimension(::$(model_name){$(variable_type_names...)}) where {$(variable_type_names...)}
        #     # dim = 0
        #     # $([quote
        #     #     if $v <: Val
        #     #         dim += ProbabilityModels.PaddedMatrices.param_type_length(ProbabilityModels.extract_typeval($v))
        #     #     end
        #     # end for v ∈ variable_type_names]...)
        #     # ProbabilityModels.PaddedMatrices.Static{dim}()
        #     # # dim
        #     ProbabilityModels.LogDensityProblems.dimension($(model_name){$(variable_type_names...)})
        # end

    # end


    # struct_quote, struct_kwarg_quote, θq_value, θq_valuegradient, constrain_quote, dim_q, variables
    struct_quote, struct_kwarg_quote, θq_value, θq_valuegradient, θq_valuegradientbuffer, constrain_quote, cvq, pn, clq, variables
end

macro model(model_name, expr)
    # struct_quote, struct_kwarg_quote, θq_value, θq_valuegradient, constrain_q, dim_q, variables = generate_generated_funcs_expressions(model_name, expr)
    struct_quote, struct_kwarg_quote, θq_value, θq_valuegradient, θq_valuegradientbuffer, constrain_q, cvq, pn, clq, variables = generate_generated_funcs_expressions(model_name, expr)
#     display(ProbabilityModels.MacroTools.striplines(struct_kwarg_quote))
#     display(ProbabilityModels.MacroTools.striplines(θq_value))
#     display(ProbabilityModels.MacroTools.striplines(θq_valuegradient))
    printstring = """
        Defined model: $model_name.
        Unknowns: $(variables[1])$([", " * string(variables[i]) for i ∈ 2:length(variables)]...).
    """
    esc(quote
        # $struct_quote; $dim_q; $struct_kwarg_quote; $θq_value; $θq_valuegradient; $constrain_q;
        $struct_quote; $struct_kwarg_quote; $θq_value; $θq_valuegradient; $θq_valuegradientbuffer; $constrain_q; $cvq; $pn; $clq;
        ProbabilityModels.Distributed.myid() == 1 && println($printstring)
    end)
end
