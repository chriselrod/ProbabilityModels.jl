#@nospecialize

function determine_variables(expr)::Set{Symbol}
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

function translate_sampling_statements(expr)::Expr
    prewalk(expr) do x
        if @capture(x, y0_ ~ f0_(θ0__))
            # @show f, y, θ
            if f0 ∈ ProbabilityDistributions.FMADD_DISTRIBUTIONS
                if @capture(x, y_ ~ f_(α_ + X_ * β_ , θ__))
                    return :(target = vadd(target, $(Symbol(f,:_fmadd))($y, $X, $β, $α, $(θ...))))
                elseif @capture(x, y_ ~ f_(X_ * β_ + α_, θ__))
                    return :(target = vadd(target, $(Symbol(f,:_fmadd))($y, $X, $β, $α, $(θ...))))
                # elseif @capture(x, y_ ~ f_(α_ - X_ * β_, θ__))
                #     return :(target = vadd(target, $(Symbol(f,:_fnmadd))($y, $X, $β, $α, $(θ...))))
                # elseif @capture(x, y_ ~ f_(X_ * β_ - α_, θ__))
                #     return :(target = vadd(target, $(Symbol(f,:_fmsub))($y, $X, $β, $α, $(θ...))))
                # elseif @capture(x, y_ ~ f_(- X_ * β_ - α_, θ__))
                #     return :(target = vadd(target, $(Symbol(f,:_fnmsub))($y, $X, $β, $α, $(θ...))))
                # elseif @capture(x, y_ ~ f_( - α_ - X_ * β_, θ__))
                #     return :(target = vadd(target, $(Symbol(f,:_fnmsub))($y, $X, $β, $α, $(θ...))))
                end
            elseif @capture(x, y2_ ~ Normal(X_ * β_, θ2__))
                return :(target = vadd(target, Normal($y2, $X, $β, $(θ2...))))
            end
            return :(target = vadd(target, $f0($y0, $(θ0...))))
#            return :(target = DistributionParameters.add(target, $f($y, $(θ...))))
        elseif @capture(x, a_ += b_)
            return :($a = $a + $b)
        elseif @capture(x, a_:b_)
            if a isa Integer && b isa Integer
                return :(ProbabilityModels.PaddedMatrices.Static{$a:$b}())
            else
                return x
            end
        else
            return x
        end
    end
end
function interpolate_globals(expr)::Expr
    postwalk(expr) do x
        if x isa Expr && x.head == :$
            return eval(first(x.args))
        else
            return x
        end
    end
end

ssa_sym(i::Int) = Symbol("##SSAValue##$(i)##")
function ssa_to_sym(expr)::Expr
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
function flatten_expression(expr)::Expr
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

# """
# This pass is to be applied to a flattened expression, after expressions such as
# a += b
# a -= b
# a *= b
# a /= b
# have been expanded. Later insertions of these symbols, for example in the constraining
# transformations of the parameters, will be bypassed. This allows for incrementing of
# the `ProbabilityModels.VectorizationBase.vectorizable` parameters:
# Symbol("##θparameter##")
# Symbol("##∂θparameter##")

# This pass skips assignments involving the following functions:
# PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED
# PaddedMatrices.RESERVED_DECREMENT_SEED_RESERVED

# This pass returns the expression in a static single assignment (SSA) form.
# """
# function rename_assignments(expr, vars = Dict{Symbol,Symbol}())
#     postwalk(expr) do ex
#         if @capture(ex, a_ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(args__)) || @capture(ex, a_ = ProbabilityModels.PaddedMatrices.RESERVED_DECREMENT_SEED_RESERVED(args__))
#             return ex
#         elseif @capture(ex, a_ = b_)
#             if isa(b, Expr)
#                 c = postwalk(x -> get(vars, x, x), b)
#             else
#                 c = b
#             end
#             if isa(a, Symbol) && !MacroTools.isgensym(a)
#                 if haskey(vars, a)
#                     lhs = gensym(a)
#                     vars[a] = lhs
#                     return :($lhs = $c)
#                 else
#                     vars[a] = a
#                     return :($a = $c)
#                 end
#             else
#                 return :($a = $c)
#             end
#         elseif @capture(ex, if cond_; conditionaleval_ end)
#             conditional = postwalk(x -> get(vars, x, x), cond)
#             conditionaleval, tracked_vars = rename_assignments(conditionaleval, TrackedDict(vars))
#             else_expr = quote end
#             for (k, (vbase,vfinal)) ∈ tracked_vars.reassigned
#                 push!(else_expr.args, :($vfinal = $vbase))
#                 vars[k] = vfinal
#             end
#             return quote
#                 if $conditional
#                     $conditionaleval
#                 else
#                     $else_expr
#                 end
#             end
#         elseif @capture(ex, if cond_; conditionaleval_; else; alternateeval_ end)
#             conditional = postwalk(x -> get(vars, x, x), cond)
#             if_conditionaleval, if_tracked_vars = rename_assignments(conditionaleval, TrackedDict(vars))
#             else_conditionaleval, else_tracked_vars = rename_assignments(alternateeval, TrackedDict(vars))
#             for (k, (vbase,vfinal)) ∈ if_tracked_vars.reassigned
#                 if haskey(else_tracked_vars.reassigned, k)
#                     push!(else_conditionaleval.args, :($vfinal = $(else_tracked_vars.reassigned[k])))
#                 else
#                     push!(else_conditionaleval.args, :($vfinal = $vbase))
#                 end
#                 vars[k] = vfinal
#             end
#             for (k, (vbase,vfinal)) ∈ else_tracked_vars.reassigned
#                 haskey(if_tracked_vars.reassigned, k) && continue
#                 push!(if_conditionaleval.args, :($vfinal = $vbase))
#                 vars[k] = vfinal
#             end
#             for k ∈ union(keys(if_tracked_vars.newlyassigned),keys(else_tracked_vars.newlyassigned))
#                 canonical_name = if_tracked_vars.newlyassigned[k]
#                 vars[k] = canonical_name
#                 push!(else_expr.args, :($canonical_name = $(else_tracked_vars.newlyassigned[k])))
#             end
#             return quote
#                 if $conditional
#                     $if_conditionaleval
#                 else
#                     $else_conditionaleval
#                 end
#             end
#         else
#             return ex
#         end
#     end, vars
# end

"""
Translates first update statements into assignments.

This is so that we can generate autodiff code via incrementing nodes, eg
node += adjoint * value
this saves us from having to initialize them all at 0.
While the compiler can easily eliminate clean up those initializations with scalars,
it may not be able to do so for arbitrary user types.


"""
function first_updates_to_assignemnts(expr, variables_input)::Expr
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

"""
This pass is for when we aren't taking partial derivatives.
"""
function constant_drop_pass!(first_pass, expr, tracked_vars, verbose = false)
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
                if verbose
                    printstring = "distribution $f (ret: $out): "
                    push!(first_pass.args, :(println($printstring)))
                end
                push!(first_pass.args, :($out = ProbabilityModels.ProbabilityDistributions.$f($(A...), Val{$track_tup}())))
                verbose && push!(first_pass.args, :(println($out)))
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
    nothing
end

types_to_vals(::Type{T}) where {T} = Val{T}()
types_to_vals(v) = v
types_to_vals(A::AbstractArray{Union{Missing,T}}) where {T} = DistributionParameters.maybe_missing(A)
extract_typeval(::Type{Val{T}}) where {T} = T
extract_typeval(::Type{Val{Type{T}}}) where {T} = T

function load_and_constrain_quote(ℓ, model_name, variables, variable_type_names, θ, Tθ, T)
    Nparam = gensym(:number_parameters)
    logjac = gensym(:logjac)
    θq = quote
        @generated function ProbabilityModels.DistributionParameters.constrain($ℓ::$(model_name){$Nparam, $(variable_type_names...)}, $θ::AbstractVector{$T}, ::Val{$logjac} = Val{false}()) where {$Nparam, $T, $logjac, $(variable_type_names...)}
            return_partials = false
            first_pass = quote end
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
            first_pass = quote end
            push!(first_pass.args, Expr(:(=), Symbol("##stack_pointer##"), Expr(:call, :(PaddedMatrices.StackPointer), Expr(:call, :pointer, $(QuoteNode(Symbol("##storage_vector##")))))))
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
               elseif $v <: ProbabilityModels.DistributionParameters.MissingDataArray
               #append!($param_names, ProbabilityModels.DistributionParameters.parameter_names($v, $(QuoteNode(v)))::Vector{String})
               append!($param_names, ProbabilityModels.DistributionParameters.parameter_names($ℓ.$(variables[i]), $(QuoteNode(v)))::Vector{String})
               end
               end for (i,v) ∈ enumerate(variable_type_names)]...)
            $param_names
        end
    end  
    Nconstrainedparam = Symbol("##number_of_parameters##")
    constrained_length_quote = quote
        @generated function ProbabilityModels.DistributionParameters.constrained_length($ℓ::$(model_name){$Nparam, $(variable_type_names...)}) where {$Nparam, $(variable_type_names...)}
            $Nconstrainedparam = 0
            $([quote
               if $v <: Val
                   $Nconstrainedparam += ProbabilityModels.PaddedMatrices.type_length(ProbabilityModels.extract_typeval($v))
               elseif $v <: ProbabilityModels.DistributionParameters.MissingDataArray
                   $Nconstrainedparam += ProbabilityModels.PaddedMatrices.type_length($v)
               end
               end for v ∈ variable_type_names]...)
            $Nconstrainedparam
        end
    end
    θq_body = θq.args[end].args[end].args[end].args;
    θq_vec_body = θq_vec.args[end].args[end].args[end].args;
    for i ∈ eachindex(variables)
        load_data = Expr(:quote, :($(variables[i]) = $ℓ.$(variables[i])))
        load_incomplete_data = Expr(:quote, Expr(:(=), Symbol("##incomplete##", variables[i]), :($ℓ.$(variables[i]))))
        push!(θq_body, quote
            if $(variable_type_names[i]) <: Val
                ProbabilityModels.DistributionParameters.load_parameter!(first_pass.args, second_pass.args, $(QuoteNode(variables[i])), ProbabilityModels.extract_typeval($(variable_type_names[i])), false, ProbabilityModels, nothing, $logjac, true)
                push!(transformed_params.args, Expr(:(=), $(QuoteNode(variables[i])), $(QuoteNode(variables[i]))))
            elseif $(variable_type_names[i]) <: ProbabilityModels.DistributionParameters.MissingDataArray
                  push!(first_pass.args, $load_incomplete_data)
                  ProbabilityModels.DistributionParameters.load_missing_as_vector!(
                      first_pass.args, second_pass.args, $(QuoteNode(variables[i])), ($(variable_type_names[i])),
                      false, ProbabilityModels, nothing, $logjac, true
                  )
                  push!(transformed_params.args, Expr(:(=), $(QuoteNode(variables[i])), $(QuoteNode(variables[i]))))
              end
        end)
        push!(θq_vec_body, quote
            if $(variable_type_names[i]) <: Val
                ProbabilityModels.DistributionParameters.load_parameter!(first_pass.args, second_pass.args, $(QuoteNode(variables[i])), ProbabilityModels.extract_typeval($(variable_type_names[i])), false, ProbabilityModels, Symbol("##stack_pointer##"), false, true)
                if ProbabilityModels.extract_typeval($(variable_type_names[i])) <: DistributionParameters.RealFloat
                    push!(first_pass.args, Expr(:call, :(ProbabilityModels.VectorizationBase.store!),
                                          Expr(:call, :pointer, Symbol("##stack_pointer##"), $(QuoteNode(T))), $(QuoteNode(variables[i]))))
                    push!(first_pass.args, Expr(:(+=), Symbol("##stack_pointer##"), Expr(:call, :sizeof, $(QuoteNode(T)) )))
              end
              elseif $(variable_type_names[i]) <: ProbabilityModels.DistributionParameters.MissingDataArray
                  push!(first_pass.args, $load_incomplete_data)
                  ProbabilityModels.DistributionParameters.load_missing_as_vector!(
                      first_pass.args, second_pass.args, $(QuoteNode(variables[i])), ($(variable_type_names[i])),
                      false, ProbabilityModels, Symbol("##stack_pointer##"), false, true
                  )
            end
       end)
    end
    push!(θq_body, :(push!(first_pass.args, return_expr)))
    push!(θq_vec_body, Expr(:call, :push!, :(first_pass.args), QuoteNode(Symbol("##storage_vector##"))))
    push!(θq_body, :(first_pass))
    push!(θq_vec_body, :(first_pass))
    θq, θq_vec, param_names_quote, constrained_length_quote
end

function generate_generated_funcs_expressions(model_name, expr)
    # Determine the set of variables that are either parameters or data, after interpolating any globals inserted via `$`
    expr = interpolate_globals(expr)
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
            $(var_vartype_pairs...)
        end
    end

    struct_kwarg_quote = quote
        function $model_name(; $(var_vartype_pairs...)) where {$(variable_type_names...)}
            $Nparam = 0
            # @show $(variables...)
            # @show $(variable_type_names...)
            $([quote
                if isa(ProbabilityModels.types_to_vals($v), Val) || isa($v, ProbabilityModels.DistributionParameters.MissingDataArray)
                   $Nparam += ProbabilityModels.PaddedMatrices.param_type_length($v)
               elseif isa($v, AbstractArray{Union{Missing,T}} where T)
                    $v = ProbabilityModels.DistributionParameters.maybe_missing($v)
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
    # The plan in this definition is to make each keyword arg default to the appropriate field of ℓ
    # This allows one to optionally override in the case of a single evaluation.
    # Additionally, it will throw an error if one of the required fields was not defined.
    V = gensym(:V)
    ℓ = gensym(:ℓ)
    θ = gensym(:θ)
    Tθ = gensym(:Tθ)
    constrain_quote, cvq, pn, clq  = load_and_constrain_quote(ℓ, model_name, variables, variable_type_names, θ, Tθ, T)
    base_stack_pointer = ProbabilityModels.STACK_POINTER_REF[]# + 9VectorizationBase.REGISTER_SIZE
    stack_pointer_expr = NTHREADS[] == 1 ? base_stack_pointer : :($base_stack_pointer + (Threads.threadid()-1)*$(LOCAL_STACK_SIZE[]))
    # we have to split these, because of dispatch ambiguity errors
    θq_value = quote
        @generated function ProbabilityModels.logdensity(
                    $ℓ::$(model_name){$Nparam, $(variable_type_names...)},
                    $θ::PtrVector{$Nparam, $T, $Nparam, true},
                    $(Symbol("##stack_pointer##"))::ProbabilityModels.PaddedMatrices.StackPointer = $stack_pointer_expr
            ) where {$Nparam, $T, $(variable_type_names...)}

            TLθ = $Nparam
            return_partials = false
            model_parameters = Symbol[]
            first_pass = quote end
            second_pass = quote end
        end
    end
    θq_valuegradient = quote
        @generated function ProbabilityModels.logdensity_and_gradient!(
                        $(Symbol("##∂θparameter##m"))::PtrVector{$Nparam, $T, $Nparam, true},
                        $ℓ::$(model_name){$Nparam, $(variable_type_names...)},
                        $θ::PtrVector{$Nparam, $T, $Nparam, true},
                        $(Symbol("##stack_pointer##"))::ProbabilityModels.PaddedMatrices.StackPointer = $stack_pointer_expr
                    ) where {$Nparam, $T, $(variable_type_names...)}
            first_pass = quote end
            TLθ = $Nparam
            return_partials = true
            model_parameters = Symbol[]
            second_pass = quote end
        end
    end
    for (θq, return_partials) ∈ ((θq_value, false), (θq_valuegradient, true))
        θq_body = θq.args[end].args[end].args[end].args;
        push!(θq_body, :(expr = $(Expr(:quote, deepcopy(expr)))))
        # Now, we work on assembling the function.
        for i ∈ eachindex(variables)
            load_data = Expr(:quote, :($(variables[i]) = $ℓ.$(variables[i])))
              load_data_dynamic_ptr_array = Expr(:quote,
                                         :($(variables[i]) = ProbabilityModels.PaddedMatrices.DynamicPtrArray(
                                             pointer( $ℓ.$(variables[i])),
                                             size($ℓ.$(variables[i])),
                                             size($ℓ.$(variables[i]),1)) )
                                         )
            load_data_ptr_array = Expr(:quote, :($(variables[i]) = ProbabilityModels.PaddedMatrices.PtrArray( $ℓ.$(variables[i]) ) ) )
            missingvar = Symbol("##missing##", variables[i])
            load_incomplete_data = Expr(:quote, Expr(:(=), Symbol("##incomplete##", variables[i]), :($ℓ.$(variables[i]))))
            push!(θq_body, quote
                if $(variable_type_names[i]) <: Val
                    push!(model_parameters, $(QuoteNode(variables[i])))
                  ProbabilityModels.DistributionParameters.load_parameter!(
                      first_pass.args, second_pass.args,
                      $(QuoteNode(variables[i])), ProbabilityModels.extract_typeval($(variable_type_names[i])),
                      $return_partials, ProbabilityModels, Symbol("##stack_pointer##"), true, false
                  )
                elseif $(variable_type_names[i]) <: ProbabilityModels.DistributionParameters.MissingDataArray
                    push!(model_parameters, $(QuoteNode(variables[i])))
                    push!(first_pass.args, $load_incomplete_data)
                  ProbabilityModels.DistributionParameters.load_parameter!(
                      first_pass.args, second_pass.args, $(QuoteNode(variables[i])), ($(variable_type_names[i])),
                      $return_partials, ProbabilityModels, Symbol("##stack_pointer##"), true, false
                  )
                elseif $(variable_type_names[i]) <: Array
                  push!(first_pass.args, $load_data_dynamic_ptr_array)
                elseif $(variable_type_names[i]) <: ProbabilityModels.PaddedMatrices.AbstractMutableFixedSizeArray
                  push!(first_pass.args, $load_data_ptr_array)
                else
                    push!(first_pass.args, $load_data)
                end
            end)
        end
        if return_partials
            processing = quote
                ProbabilityModels.reverse_diff_pass!(first_pass, second_pass, expr, tracked_vars, $(verbose_models()))
                expr_out = quote
                    target = ProbabilityModels.DistributionParameters.initialize_target($T_sym)
                    $(Symbol("##θparameter##")) = ProbabilityModels.VectorizationBase.vectorizable($θ_sym)
                    $first_pass
                    $(Symbol("##∂θparameter##")) = ProbabilityModels.VectorizationBase.vectorizable($(Symbol("##∂θparameter##m")))
                    $(Symbol("###seed###", name_dict[:target])) = ProbabilityModels.One()
                    $second_pass
                end
                push!(expr_out.args, quote
                        $(Symbol("##scalar_target##")) = ProbabilityModels.SIMDPirates.vsum($(name_dict[:target]))
                        if !isfinite($(Symbol("##scalar_target##"))) || !all(isfinite, $(Symbol("##∂θparameter##m")))
                            $(Symbol("##scalar_target##")) = typemin($T_sym)
                        end
                        $(Symbol("##scalar_target##"))
                end)
            end
        else
            processing = quote
                ProbabilityModels.constant_drop_pass!(first_pass, expr, tracked_vars, $(verbose_models()))
                expr_out = quote
                    target = ProbabilityModels.DistributionParameters.initialize_target($T_sym)
                    $(Symbol("##θparameter##")) = ProbabilityModels.VectorizationBase.vectorizable($θ_sym)
                    $first_pass
                    $(Symbol("##scalar_target##")) = ProbabilityModels.SIMDPirates.vsum($(name_dict[:target]))
                    isfinite($(Symbol("##scalar_target##"))) ? $(Symbol("##scalar_target##")) : $T_sym(-Inf)
                end
            end
        end
        push!(θq_body, quote
            tracked_vars = Set(model_parameters)
              name_dict = Dict(:target => :target)
            θ_sym = $(QuoteNode(θ)) # This creates our symbol θ
            T_sym = $(QuoteNode(T))
            ℓ_sym = $(QuoteNode(ℓ))
            $processing
            final_quote = ProbabilityModels.StackPointers.stack_pointer_pass(ProbabilityModels.first_updates_to_assignemnts(expr_out, model_parameters), $(QuoteNode(Symbol("##stack_pointer##"))),nothing,$(verbose_models()),:(ProbabilityModels.StackPointers)) |> ProbabilityModels.PaddedMatrices.simplify_expr
              $(verbose_models()) && println(final_quote)
              quote
                  @inbounds begin
                      $final_quote
                  end
              end
        end)

    end

    # Now, we would like to apply
    PaddedMatrices.simplify_expr.((struct_quote, struct_kwarg_quote, θq_value, θq_valuegradient, constrain_quote, cvq, pn, clq, variables))
    #struct_quote, struct_kwarg_quote, θq_value, θq_valuegradient, constrain_quote, cvq, pn, clq, variables))
end

macro model(model_name, expr)
    # struct_quote, struct_kwarg_quote, θq_value, θq_valuegradient, constrain_q, dim_q, variables = generate_generated_funcs_expressions(model_name, expr)
    struct_quote, struct_kwarg_quote, θq_value, θq_valuegradient, constrain_q, cvq, pn, clq, variables = generate_generated_funcs_expressions(model_name, expr)
    printstring = """
        Defined model: $model_name.
        Unknowns: $(variables[1])$([", " * string(variables[i]) for i ∈ 2:length(variables)]...).
    """
    esc(quote
        # $struct_quote; $dim_q; $struct_kwarg_quote; $θq_value; $θq_valuegradient; $constrain_q;
        $struct_quote; $struct_kwarg_quote; $θq_value; $θq_valuegradient; $constrain_q; $cvq; $pn; $clq;
        ProbabilityModels.Distributed.myid() == 1 && println($printstring)
    end)
end

# @specialize
