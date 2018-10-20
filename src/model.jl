"""
Use push! and pushfirst! for writing forward and reverese passes, respectively.
"""
function _model(model_expr)
    model_lines = striplines(model_expr).args
    model_name = model_lines[1]
    g, d, label, parametric_types_to_parameters, parameters_to_parametric_types = _vertices(model_lines[2].args[3])
    f_body, ∇f_forward, ∇f_back = :(), :(), :()
    f_body = quote
        @generated function $model_name($(label...))

        end
    end
    fa, ∇fa = f_body.args, ∇f_body.args
    for i ∈ 3:length(model_lines)
        model_line = model_lines[i]
        if @capture(model_line, child_ ~ distribution_(parents__))
            child_ind = d[child]
            push!(fa, target += $distribution($child, $(parents...) ))

            for parent ∈ parents
                if isa(parent, Symbol)
                    add_edge!(g, child_ind, d[parent])
                else
                    #postwalk
                    postwalk(x -> x isa Symbol && x ∈ keys(d) ? (add_edge!(g, child_ind, d[x]); x) : x , parent)
                end
            end

        end
    end
end

macro model(model_expr)
    _model(model_expr)
end

### Use malloc or calloc, and then
### finalizer(f, x )
###
function define_struct(g, d, l, parametric_types_to_parameters, parameters_to_parametric_types, model_name)
    struct_q = quote
        struct $(Symbol(model_name, :Model)){_L,_T,$([Symbol(:_V,i) for i ∈ eachindex(l)]...)}
            unconstrained::SizedSIMDVector{_L,_T,_L,_L}
            ∇unconstrained::SizedSIMDVector{_L,_T,_L,_L}
        end
    end
    struct_qa = struct_q.args[2].args[3].args
    for i ∈ eachindex(l)
        push!(struct_qa, :($(l[i])::$(Symbol(:_V,i)) ) )
    end
    # parametric_types_to_parameters is a dictionary containing the type's parameters as keys,
    # while the values are vectors of all the parameters that share that key
    # so that we may use it to check if we can infer types without the user
    # having to specify val types.
    #
    # parameters_to_parametric_types is a dictionary containing
    parameter_v_data_check = :() # first pass, identify type parameters
    # parameter_second_pass = :() # second pass, insert type parameters into model parameters
    where_list = [Symbol(:_V, i) for i ∈ eachindex(l)]
    keywordargs = Expr(:parameters, )
    for (type_param, param_array) ∈ zip(l, parametric_types_to_parameters)
        param = Symbol(:_,type_param)
        push!(where_list, param)
        push!(keywordargs.args, Expr(:kw, :($type_param::Val{$param}), Val(nothing)))
    end
    for i ∈ eachindex(l)
        Vsym = Symbol(:_V, i)
        push!(keywordargs.args, Expr(:kw, :($(l[i])::($Vsym)), get_prop(g, i, :default)))
        # param_inds =
        param_vec = get!(parameters_to_parametric_types, param, Tuple{Int,Symbol})
        if length(param_vec) == 0
            push!(parameter_v_data_check.args,
            :(
            if isparameter($Vsym)
                push!(parameters, $(i, l[i]) )
                push!(types_of_parameters, $(get_prop(g, i, :default)))
            else # it is a constant, now try to read parameteric type information.
                push!(datapriors, $(i, l[i]) )
            end
            ))
        elseif length(param_vec) == 1
            push!(parameter_v_data_check.args,
            :(
            if isparameter($Vsym)
                push!(parameters, $(i, l[i]) )
                push!(types_of_parameters, $(get_prop(g, i, :default)))
            else # it is a constant, now try to read parameteric type information.
                push!(datapriors, $(i, l[i]) )
                parametric_type_dict[$(param_vec[1][2])] = $(Vsym).parameters[$(param_vec[1][1])]
            end
            ))
        elseif length(param_vec) == 2
            push!(parameter_v_data_check.args,
            :(
            if isparameter($Vsym)
                push!(parameters, $(i, l[i]) )
                push!(types_of_parameters, $(get_prop(g, i, :default)))
            else # it is a constant, now try to read parameteric type information.
                push!(datapriors, $(i, l[i]) )
                parametric_type_dict[$(param_vec[1][2])] = $(Vsym).parameters[$(param_vec[1][1])]
                parametric_type_dict[$(param_vec[2][2])] = $(Vsym).parameters[$(param_vec[2][1])]
            end
            ))
        else
            throw("Param vector of length $(length(param_vec)) not yet supported")
        end
        # push!(parameter_second_pass.args, :(
        # for (i,param) ∈ parameters
        #
        #     _L += type_length(param)# param is a symbol
        # end
        # ))
    end

    initialize_q = quote
        @generated function $(Symbol(model_name, :Model))(; $keywordargs ) where {$(where_list...)}
            _L = 0
            parametric_type_dict = Dict{Symbol,Int}()
            types_of_parameters = Expr[]
            parameters = Tuple{Int,Symbol}[]
            datapriors = Tuple{Int,Symbol}[] # what is this needed for?

            $parameter_v_data_check


            # insert parameter stuff here
            for param_type ∈ $(Expr(:tuple, keys(parametric_types_to_parameters)... ))
                if param_type ∉ keys(parametric_type_dict)

                end

            end
            for i ∈ eachindex(parameters)
                param, param_type = parameters[i], types_of_parameters[i]
                types_of_parameters[i] = postwalk(
                    x -> (isa(x, Symbol) && x ∈ keys(parametric_type_dict))
                        ?
                        parametric_type_dict[x] : x,
                    param_type
                )
            end

        end
    end
end


## What I need to be able to do is see in the program that a default type is given as
## :(SizedVector{N, Float64})
## and then take this to somehow write into the generated function that the parameter is
## :(SizedVector{$N, Float64})
## Easiest way: use a Dict{Symbol,Int}()
## when you find, eg, that :N = 56, then store in dict.
## Then postwalk expressions and substitute. Example:
## sv = :(SizedVector{N, Float64})
## d = Dict(:N => 56)
## julia> postwalk(x -> (isa(x, Symbol) && x ∈ keys(d)) ? d[x] : x, sv)
##  # :(SizedVector{56, Float64})
"""
Checks and identifies (parametric type) parameters.
When a variable is provided as data, extracts the parametric type information.

"""
function identify_params()

end


function construct_graph()


end

function construct_f_body!()


end
