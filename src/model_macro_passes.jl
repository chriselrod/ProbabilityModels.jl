#@nospecialize

function determine_variables(expr)::Set{Symbol}
    variables = Set{Symbol}()
    ignored_symbols = Set{Symbol}()
    push!(ignored_symbols, :target)
    prewalk(expr) do x
        if x isa Symbol
            x ∈ ignored_symbols || push!(variables, x)
        elseif x isa Expr
            if x.head === :(=)
                LHS = first(x.args)
                if LHS isa Symbol
                    push!(ignored_symbols, LHS)
                elseif LHS isa Expr && LHS.head === :ref
                    push!(ignored_symbols, first(LHS.args))
                end
            elseif x.head === :call
                push!(ignored_symbols, first(x.args))
            end
        end
        return x
    end
    variables
end

function no_fmadd_dist_call(y, dargs)
    dist = Expr(:call, first(dargs), y)
    for i ∈ 2:length(dargs)
        push!(dist.args, dargs[i])
    end
    Expr(:(=), :target, Expr(:call, Expr(:(.), :ProbabilityModels, QuoteNode(:vadd)), :target, dist))
end
function fmadd_dist_call(y, dargs)
    farg₁ = dargs[2]
    # farg₁ must be an expr of the form (X * β + α)
    (farg₁ isa Expr && farg₁.head === :call) || return no_fmadd_dist_call(y, dargs)
    args = farg₁.args
    Nargs = length(args)
    Nargs > 2 || return no_fmadd_dist_call(y, dargs)
    if first(args) !== :+
        if first(dargs) === :Normal && first(args) === :* && length(args) == 3
            dist = Expr(:call, :Normal, y, args[2], args[3])
            for i ∈ 3:length(dargs)
                push!(dist.args, dargs[i])
            end
            Expr(:(=), :target, Expr(:call, Expr(:(.), :ProbabilityModels, QuoteNode(:vadd)), :target, dist))
        else
            return no_fmadd_dist_call(y, dargs)
        end
    end
    j = 2
    while j ≤ Nargs
        argsⱼ = args[j]
        if argsⱼ isa Expr && (first(argsⱼ.args) === :*) && length(argsⱼ.args) == 3
            break
        end
        j += 1
    end
    j > Nargs && return no_fmadd_dist_call(y, dargs)
    # if we're here, j is the index of the mul
    argsⱼ = args[j].args
    dist = Expr(:call, first(dargs), y, argsⱼ[2], argsⱼ[3])
    if Nargs == 3
        push!(dist.args, args[j == 2 ? 3 : 2])
    else #Nargs > 3
        for i ∈ 2:Nargs
            i == j || push!(dist.args, args[i])
        end
    end
    for i ∈ 3:length(dargs)
        push!(dist.args, dargs[i])
    end
    Expr(:(=), :target, Expr(:call, Expr(:(.), :ProbabilityModels, QuoteNode(:vadd)), :target, dist))
end
function dist_call(x)
    distcall::Expr = x.args[3]
    distcall.head === :call || return x
    length(distcall.args) < 2 && return x
    y = x.args[2]
    dargs = distcall.args
    if first(dargs) ∈ ProbabilityDistributions.FMADD_DISTRIBUTIONS
        fmadd_dist_call(y, dargs)
    else
        no_fmadd_dist_call(y, dargs)
    end
end

function translate_sampling_statement(x)
    x isa Expr || return x
    if x.head === :(+=)
        return Expr(:(=), x.args[1], Expr(:+, x.args[1], x.args[2]))
    elseif x.head === :call && first(x.args) === :~
        dist_call(x)
    else
        return x
    end
end

translate_sampling_statements(expr::Expr)::Expr = prewalk(translate_sampling_statement, expr)

# This does not work, because eval only looks in the current module, and doesn't accept a module argument (ie, Main)
# function interpolate_globals(expr)::Expr
#     postwalk(expr) do x
#         if x isa Expr && x.head == :$
#             return eval(first(x.args))
#         else
#             return x
#         end
#     end
# end
maybeget(d::Dict{K,V}, ex) where {K,V} = ex isa K ? get(d, ex, ex) : ex
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
function subssasyms!(q, ex, substitutions, i)
    ex = postwalk(ex) do x
        y = ssa_to_sym(x)
        maybeget(substitutions, y)
    end
    push!(q.args, :($(ssa_sym(i)) = $ex))
    nothing
end
function flatten_expression(expr)::Expr
    lowered_array = Meta.lower(ProbabilityModels, expr).args
    substitutions = Dict{Symbol,Expr}()
    assignments = Dict{Symbol,Symbol}()
    @assert length(lowered_array) == 1
    lowered = first(lowered_array).code
    q = quote end
    for i ∈ 1:length(lowered) - 1 # skip return statement
        ex = lowered[i]
        if ex isa Expr
            if ex.head === :call
                if first(ex.args) === :(:)
                    push!(q.args, :($(ssa_sym(i)) = ProbabilityModels.PaddedMatrices.Static{($(ex.args[2]),$(ex.args[3]))}()))
                elseif ex.args[1] isa GlobalRef && ex.args[1].name == :getproperty
                    substitutions[ssa_sym(i)] = Expr(:., ex.args[2], ex.args[3])
                else
                    subssasyms!(q, ex, substitutions, i)
                end
            elseif ex.head === :(=)
                assignments[ssa_to_sym(ex.args[2])] = ex.args[1]
            else
                subssasyms!(q, ex, substitutions, i)
            end
        else
            subssasyms!(q, ex, substitutions, i)
        end
    end
    q = postwalk(q) do ex
        maybeget(assignments, ex)
    end
    # println(q)
    q
end

function drop_const_assignment!(first_pass, tracked_vars, ex, LHS, RHS, verbose)
    if RHS isa Expr && RHS.head === :call
        f = first(RHS.args)
        A = @view(RHS.args[2:end])
        if f ∈ ProbabilityDistributions.DISTRIBUTION_DIFF_RULES
            track_tup = Expr(:tuple,)
            for a ∈ A
                if a ∈ tracked_vars
                    push!(track_tup.args, true)
                    push!(tracked_vars, LHS)
                else
                    push!(track_tup.args, false)
                end
            end
            if verbose
                printstring = "distribution $f (ret: $LHS): "
                push!(first_pass, :(println($printstring)))
            end
            push!(first_pass, :($LHS = ProbabilityModels.ProbabilityDistributions.$f(Val{$track_tup}(), $(A...))))
            verbose && push!(first_pass, :(println($LHS)))
        else
            for a ∈ A
                if a ∈ tracked_vars
                    push!(tracked_vars, LHS)
                    break
                end
            end
            if f === :add
                push!(first_pass, :( $LHS = ProbabilityModels.vadd($(A...))))
            else
                push!(first_pass, ex)
            end
        end
    else
        A = RHS.args
        for a ∈ A
            if a ∈ tracked_vars
                push!(tracked_vars, out)
                break
            end
        end
        push!(first_pass, ex)
    end    
end

"""
This pass is for when we aren't taking partial derivatives.
"""
function constant_drop_pass!(first_pass::Vector{Any}, expr, tracked_vars, verbose = false)
    for x ∈ expr.args
        if !(x isa Expr)
            push!(first_pass, x)
            continue
        end
        ex::Expr = x
        if ex.head === :(=)
            drop_const_assignment!(first_pass, tracked_vars, ex, ex.args[1], ex.args[2], verbose)
        else
            push!(first_pass, ex)
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
            push!(first_pass.args, Expr(:(=), Symbol("##θparameter##"), Expr(:call, :(ProbabilityModels.vectorizable), $(QuoteNode(θ)))))
            second_pass = quote end
            transformed_params = Expr(:tuple,)
            return_expr = $logjac ? Expr(:tuple, transformed_params, :target) : transformed_params
        end
    end
    v = Symbol("##storage_vector##")
    θq_vec = quote
        @generated function ProbabilityModels.DistributionParameters.constrain!(
            $v::AbstractVector{$T}, $ℓ::$(model_name){$Nparam, $(variable_type_names...)}, $θ::AbstractVector{$T}
        ) where {$Nparam, $T, $(variable_type_names...)}
            return_partials = false
            first_pass = quote end
            push!(first_pass.args, Expr(:(=), Symbol("##stack_pointer##"), Expr(:call, :(ProbabilityModels.StackPointer), Expr(:call, :pointer, Symbol("##storage_vector##")))))
            push!(first_pass.args, Expr(:(=), Symbol("##θparameter##"), Expr(:call, :(ProbabilityModels.vectorizable), $(QuoteNode(θ)))))
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
              ProbabilityModels.DistributionParameters.load_parameter!(
                  first_pass.args, second_pass.args, $(QuoteNode(variables[i])),
                  ProbabilityModels.extract_typeval($(variable_type_names[i])),
                  false, ProbabilityModels, nothing, $logjac, true
              )
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
            #     if ProbabilityModels.extract_typeval($(variable_type_names[i])) <: ProbabilityModels.DistributionParameters.RealFloat
            #         push!(first_pass.args, Expr(:call, :(ProbabilityModels.store!),
            #                               Expr(:call, :pointer, Symbol("##stack_pointer##"), $(QuoteNode(T))), $(QuoteNode(variables[i]))))
            #         push!(first_pass.args, Expr(:(+=), Symbol("##stack_pointer##"), Expr(:call, :sizeof, $(QuoteNode(T)) )))
            #   end
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
    # push!(θq_body, :(QuoteNode(first_pass)))
    push!(θq_vec_body, :(first_pass))
    θq, θq_vec, param_names_quote, constrained_length_quote
end

function generate_generated_funcs_expressions(model_name, expr)
    # Determine the set of variables that are either parameters or data, after interpolating any globals inserted via `$`
    # expr = interpolate_globals(expr)]
    variable_set = determine_variables(expr)
    variables = sort!([v for v ∈ variable_set]) # ensure order is constant
    variable_type_names = [Symbol("##Type##", v) for v ∈ variables]

    Nparam = Symbol("##number_of_parameters##")
    stride = gensym(:LDA)
    L1 = gensym(:L)
    L2 = gensym(:L)
    T = Symbol("##element_type##")

    var_vartype_pairs = [:( $(variables[i])::$(variable_type_names[i]) ) for i ∈ eachindex(variables)]
    struct_quote = quote
        struct $model_name{$Nparam, $(variable_type_names...)} <: ProbabilityModels.AbstractProbabilityModel{$Nparam}
            $(var_vartype_pairs...)
        end
    end
    precomp = gensym(:precompile)
    # precomp_quote = if VERSION >= v"1.3.0-alpha.0" && ProbabilityModels.NTHREADS[] > 1
    #     tlag = gensym(:tlag); tl = gensym(:tl)
    #     quote
    #         $tlag = Threads.@spawn precompile(ProbabilityModels.logdensity_and_gradient!, (ProbabilityModels.PtrVector{$Nparam, Float64, $Nparam, false}, $model_name{$Nparam,$(variable_type_names...)}, ProbabilityModels.PtrVector{$Nparam, Float64, $Nparam, false}, ProbabilityModels.StackPointer))
    #         $tl = Threads.@spawn precompile(ProbabilityModels.logdensity, ($model_name{$Nparam,$(variable_type_names...)}, ProbabilityModels.PtrVector{$Nparam, Float64, $Nparam, false}, ProbabilityModels.StackPointer))
    #         Threads.@spawn precompile(ProbabilityModels.InplaceDHMC.mcmc_with_warmup!, (
    #             ProbabilityModels.VectorizedRNG.PtrPCG{4}, ProbabilityModels.StackPointer, ProbabilityModels.PaddedMatrices.DynamicPtrVector{Float64}, ProbabilityModels.PaddedMatrices.DynamicPtrVector{ProbabilityModels.InplaceDHMC.TreeStatisticsNUTS}, $model_name{$Nparam, $(variable_type_names...)}, Int))
    #         wait($tlag); wait($tl)
    #     end
    # else
    #     quote
    #         precompile(ProbabilityModels.logdensity_and_gradient!, (ProbabilityModels.PtrVector{$Nparam, Float64, $Nparam, false}, $model_name{$Nparam,$(variable_type_names...)}, ProbabilityModels.PtrVector{$Nparam, Float64, $Nparam, false}, ProbabilityModels.StackPointer))
    #         precompile(ProbabilityModels.logdensity, ($model_name{$Nparam,$(variable_type_names...)}, ProbabilityModels.PtrVector{$Nparam, Float64, $Nparam, false}, ProbabilityModels.StackPointer))
    #     end
    # end
    precomp_quote = quote
        precompile(ProbabilityModels.logdensity_and_gradient!, (ProbabilityModels.PtrVector{$Nparam, Float64, $Nparam, false}, $model_name{$Nparam,$(variable_type_names...)}, ProbabilityModels.PtrVector{$Nparam, Float64, $Nparam, false}, ProbabilityModels.StackPointer))
        precompile(ProbabilityModels.logdensity, ($model_name{$Nparam,$(variable_type_names...)}, ProbabilityModels.PtrVector{$Nparam, Float64, $Nparam, false}, ProbabilityModels.StackPointer))
    end
    struct_kwarg_quote = quote
        function $model_name{$Nparam}( $(var_vartype_pairs...)) where {$Nparam, $(variable_type_names...)}
            $model_name{$Nparam,$(variable_type_names...)}($(variables...))
        end
        function $model_name{$Nparam}($precomp::Bool, $(var_vartype_pairs...)) where {$Nparam, $(variable_type_names...)}
            if $precomp
                $precomp_quote
            end
            $model_name{$Nparam,$(variable_type_names...)}($(variables...))
        end
        function $model_name($precomp::Bool = true; $(var_vartype_pairs...)) where {$(variable_type_names...)}
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
            $model_name{$Nparam}($precomp, $([:(ProbabilityModels.types_to_vals($v)) for v ∈ variables]...))
        end
    end
    # Translate the sampling statements, and then flatten the expression to remove nesting.
    expr = translate_sampling_statements(expr) |>
                flatten_expression# |>
    # The plan in this definition is to make each keyword arg default to the appropriate field of ℓ
    # This allows one to optionally override in the case of a single evaluation.
    # Additionally, it will throw an error if one of the required fields was not defined.
    V = gensym(:V)
    ℓ = Symbol("####ℓ#data####")# gensym(:ℓ)
    θ = Symbol("##θ_parameter_vector##")#gensym(:θ)
    Tθ = gensym(:Tθ)
    constrain_quote, cvq, pn, clq  = load_and_constrain_quote(ℓ, model_name, variables, variable_type_names, θ, Tθ, T)
    base_stack_pointer = STACK_POINTER_REF[]# + 9VectorizationBase.REGISTER_SIZE
    stack_pointer_expr = NTHREADS[] == 1 ? base_stack_pointer : :($base_stack_pointer + (Threads.threadid()-1)*$(LOCAL_STACK_SIZE[]))
    # we have to split these, because of dispatch ambiguity errors
    θq_value = quote
        @generated function ProbabilityModels.logdensity(
                    $ℓ::$(model_name){$Nparam, $(variable_type_names...)},
                    $θ::ProbabilityModels.PtrVector{$Nparam, $T, $Nparam, false},
                    $(Symbol("##stack_pointer##"))::ProbabilityModels.StackPointer = $stack_pointer_expr
        ) where {$Nparam, $T, $(variable_type_names...)}

            return_partials = false
            TLθ = $Nparam
            model_parameters = Symbol[]
            first_pass = Any[]
            second_pass = Any[]
        end
    end
    θq_valuegradient = quote
        @generated function ProbabilityModels.logdensity_and_gradient!(
                        $(Symbol("##∂θ_parameter_vector##"))::ProbabilityModels.PtrVector{$Nparam, $T, $Nparam, false},
                        $ℓ::$(model_name){$Nparam, $(variable_type_names...)},
                        $θ::ProbabilityModels.PtrVector{$Nparam, $T, $Nparam, false},
                        $(Symbol("##initial_stack_pointer##"))::ProbabilityModels.StackPointer = $stack_pointer_expr
        ) where {$Nparam, $T, $(variable_type_names...)}

            return_partials = true
            TLθ = $Nparam
            model_parameters = Symbol[]
            first_pass = Any[]
            second_pass = Any[]
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
                      first_pass, second_pass,
                      $(QuoteNode(variables[i])), ProbabilityModels.extract_typeval($(variable_type_names[i])),
                      $return_partials, ProbabilityModels, Symbol("##stack_pointer##"), true, false
                  )
                elseif $(variable_type_names[i]) <: ProbabilityModels.DistributionParameters.MissingDataArray
                    push!(model_parameters, $(QuoteNode(variables[i])))
                    push!(first_pass, $load_incomplete_data)
                  ProbabilityModels.DistributionParameters.load_parameter!(
                      first_pass, second_pass, $(QuoteNode(variables[i])), ($(variable_type_names[i])),
                      $return_partials, ProbabilityModels, Symbol("##stack_pointer##"), true, false
                  )
                elseif $(variable_type_names[i]) <: Union{Array,SubArray{<:Any,<:Any,<:Array}}
                  push!(first_pass, $load_data_dynamic_ptr_array)
                elseif $(variable_type_names[i]) <: ProbabilityModels.PaddedMatrices.AbstractMutableFixedSizeArray
                  push!(first_pass, $load_data_ptr_array)
                else
                    push!(first_pass, $load_data)
                end
            end)
        end
        if return_partials
            processing = quote
                $(verbose_models() > 0 ? :(println("Before differentiating, the model is: \n", ProbabilityModels.PaddedMatrices.simplify_expr(expr), "\n\n")) : nothing)
                ProbabilityModels.ReverseDiffExpressions.reverse_diff_pass!(first_pass, second_pass, expr, tracked_vars, :ProbabilityModels, $(verbose_models() > 1))
                expr_out = quote
                    target = ProbabilityModels.initialize_target($(Symbol("##element_type##")))
                    # $(Symbol("##stack_pointer##")) = $(Symbol("##initial_stack_pointer##"))
                    $(Symbol("##stack_pointer##")) = ProbabilityModels.StackPointer(pointer($(Symbol("##initial_stack_pointer##")), $(Symbol("##element_type##"))))
                    $(Symbol("##θparameter##")) = ProbabilityModels.vectorizable($(Symbol("##θ_parameter_vector##")))
                    # $(Symbol("##θparameter##")) = ProbabilityModels.vectorizable(ProbabilityModels.SIMDPirates.noalias!(pointer($(Symbol("##θ_parameter_vector##")))))
                    # $(Symbol("##∂θparameter##")) = ProbabilityModels.vectorizable($(Symbol("##∂θ_parameter_vector##")))
                    $(Symbol("##∂θparameter##")) = ProbabilityModels.vectorizable(pointer($(Symbol("##∂θ_parameter_vector##"))))
                end
                append!(expr_out.args, first_pass)
                append!(expr_out.args, second_pass)
                push!(expr_out.args, Expr(:(=), Symbol("##scalar_target##"), :(ProbabilityModels.vsum($(name_dict[:target])))))
                push!(expr_out.args, :((isfinite($(Symbol("##scalar_target##"))) && all(isfinite, $(Symbol("##∂θ_parameter_vector##")))) || ($(Symbol("##scalar_target##")) = typemin($(Symbol("##element_type##"))))))
                push!(expr_out.args, Symbol("##scalar_target##"))
            end
        else
            processing = quote
                ProbabilityModels.constant_drop_pass!(first_pass, expr, tracked_vars, $(verbose_models() > 1))
                pushfirst!(first_pass, Expr(:(=), Symbol("##initial_stack_pointer##"), Expr(:call, :pointer, Symbol("##stack_pointer##"))))
                expr_out = quote
                    target = ProbabilityModels.initialize_target($(Symbol("##element_type##")))
                    $(Symbol("##θparameter##")) = ProbabilityModels.vectorizable($(Symbol("##θ_parameter_vector##")))
                end
                append!(expr_out.args, first_pass)
                push!(expr_out.args, Expr(:(=), Symbol("##scalar_target##"), :(ProbabilityModels.vsum($(name_dict[:target])))))
                push!(expr_out.args, Expr(:call, :(ProbabilityModels.lifetime_end!), Symbol("##initial_stack_pointer##"), Val{8192}()))
                push!(expr_out.args, :(isfinite($(Symbol("##scalar_target##"))) ? $(Symbol("##scalar_target##")) : typemin($(Symbol("##element_type##")))))
            end
        end
        push!(θq_body, quote
            tracked_vars = Set(model_parameters)
              name_dict = Dict(:target => :target)
              $processing
              final_quote = ProbabilityModels.StackPointers.stack_pointer_pass(
                  expr_out, $(QuoteNode(Symbol("##stack_pointer##"))),
                  nothing, $(verbose_models() > 1), :ProbabilityModels
              ) |> ProbabilityModels.PaddedMatrices.simplify_expr
              $(verbose_models() > 0) && println(final_quote)
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
#        ProbabilityModels.Distributed.myid() == 1 && println($printstring)
        println($printstring)
    end)
end
# macro show_model_expr(model_name, expr)
#     struct_quote, struct_kwarg_quote, θq_value, θq_valuegradient, constrain_q, cvq, pn, clq, variables = generate_generated_funcs_expressions(model_name, expr)
#     @show struct_quote
#     @show struct_kwarg_quote
#     @show θq_value
#     @show θq_valuegradient
#     @show constrain_q
#     @show cvq
#     @show, pn
#     @show clq
#     @show variables
#     nothing
# end

# @specialize
