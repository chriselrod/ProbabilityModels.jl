
function _vertices(all_vertex_expr)
    # Separate out the function from the macro, for better global scope performance
    # nodesexpr = striplines(nodesexpr)
    vertex_exprs = all_vertex_expr.args
    # g = SimpleDiGraph()
    g = MetaDiGraph(length(vertex_exprs))
    d = Dict{Symbol,Int}()
    label = Vector{Symbol}(undef, length(vertex_exprs))

    parametric_types_to_parameters = Dict{Symbol,Vector{Symbol}}()
    parameters_to_parametric_types = Dict{Symbol,Vector{Tuple{Int,Symbol}}}()
    for i ∈ eachindex(vertex_exprs)
        vertex = vertex_exprs[i]
        if @capture(vertex, param_::type_)
            update_metagraph!(g, i, d, param, type,
                        parametric_types_to_parameters, parameters_to_parametric_types)
        elseif @capture(vertex, param_::type_ = default_)
            # set_prop!(g, i, :default, default)
            update_metagraph!(g, i, d, param, type,
                        parametric_types_to_parameters, parameters_to_parametric_types)
        end
        d[param] = i
        label[i] = param
    end
    g, d, label, parametric_types_to_parameters, parameters_to_parametric_types
end

function update_metagraph!(g, i, d, param, type_expr, parametric_types_to_parameters, parameters_to_parametric_types, default = nothing)
    if @capture(type_expr, type_{T1_, T2_, bound1t_ = bound1_, bound2t_ = bound2_})
        if bound1t == :lower && bound2t == :upper
            matrix_data_or_param!(g, i, T1, T2, default, lower = bound1, upper = bound2)
        elseif bound1t == :upper && bound2t == :lower
            matrix_data_or_param!(g, i, T1, T2, default, lower = bound2, upper = bound1)
        else
            throw("Failed to match $type_expr")
        end
        if typeof(T1) == Symbol
            push!(get!(parametric_types_to_parameters, T1, Symbol[]), param)
            push!(get!(parameters_to_parametric_types, param, Tuple{Int,Symbol}), (1,T1))
        end
        if typeof(T2) == Symbol
            push!(get!(parametric_types_to_parameters, T2, Symbol[]), param)
            push!(get!(parameters_to_parametric_types, param, Tuple{Int,Symbol}), (2,T2))
        end
    elseif @capture(type_expr, type_{T1_, T2_, boundt_ = bound_})
        if boundt == :lower
            matrix_data_or_param!(g, i, T1, T2, default, lower = bound)
        elseif boundt == :upper
            matrix_data_or_param!(g, i, T1, T2, default, upper = bound)
        else
            throw("Failed to match $type_expr")
        end
        if typeof(T1) == Symbol
            push!(get!(parametric_types_to_parameters, T1, Symbol[]), param)
            push!(get!(parameters_to_parametric_types, param, Tuple{Int,Symbol}), (1,T1))
        end
        if typeof(T2) == Symbol
            push!(get!(parametric_types_to_parameters, T2, Symbol[]), param)
            push!(get!(parameters_to_parametric_types, param, Tuple{Int,Symbol}), (2,T2))
        end
    elseif @capture(type_expr, type_{T1_, T2_})
        matrix_data_or_param!(g, i, T1, T2, default)
        if typeof(T1) == Symbol
            push!(get!(parametric_types_to_parameters, T1, Symbol[]), param)
            push!(get!(parameters_to_parametric_types, param, Tuple{Int,Symbol}), (1,T1))
        end
        if typeof(T2) == Symbol
            push!(get!(parametric_types_to_parameters, T2, Symbol[]), param)
            push!(get!(parameters_to_parametric_types, param, Tuple{Int,Symbol}), (2,T2))
        end
    elseif @capture(type_expr, type_{T_, bound1t_ = bound1_, bound2t_ = bound2_})
        if bound1t == :lower && bound2t == :upper
            vector_data_or_param!(g, i, T, default, lower = bound1, upper = bound2)
        elseif bound1t == :upper && bound2t == :lower
            vector_data_or_param!(g, i, T, default, lower = bound2, upper = bound1)
        else
            throw("Failed to match $type_expr")
        end
        if typeof(T) == Symbol
            push!(get!(parametric_types_to_parameters, T, Symbol[]), param)
            # push!(get!(parameters_to_parametric_types, param, Tuple{Int,Symbol}), (1,T))
            parameters_to_parametric_types[param] = [(1,T)]
        end
    elseif @capture(type_expr, type_{T_, boundt_ = bound_})
        if boundt == :lower
            vector_data_or_param!(g, i, T, default, lower = bound)
        elseif bound1t == :upper
            vector_data_or_param!(g, i, T, default, upper = bound)
        else
            throw("Failed to match $type_expr")
        end
        if typeof(T) == Symbol
            push!(get!(parametric_types_to_parameters, T, Symbol[]), param)
            parameters_to_parametric_types[param] = [(1,T)]
        end
    elseif @capture(type_expr, type_{T_})
        vector_data_or_param!(g, i, T, default)
        if typeof(T) == Symbol
            push!(get!(parametric_types_to_parameters, T, Symbol[]), param)
            parameters_to_parametric_types[param] = [(1,T)]
        end
    elseif @capture(type_expr, type_{bound1t_ = bound1_, bound2t_ = bound2_})
        if bound1t == :lower && bound2t == :upper
            real_data_or_param!(g, i, default, lower = bound1, upper = bound2)
        elseif bound1t == :upper && bound2t == :lower
            real_data_or_param!(g, i, default, lower = bound2, upper = bound1)
        else
            throw("Failed to match $type_expr")
        end
    elseif @capture(type_expr, type_{boundt_ = bound_})
        if boundt == :lower
            real_data_or_param!(g, i, default, lower = bound)
        elseif bound1t == :upper
            real_data_or_param!(g, i, default, upper = bound)
        else
            throw("Failed to match $type_expr")
        end
    elseif @capture(type_expr, type_)
        real_data_or_param!(g, i, default)
    else
        throw("Failed to match $type_expr")
    end
end

macro vertices(all_vertex_expr)
    _vertices(all_vertex_expr)
end
