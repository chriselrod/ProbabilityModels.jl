

# function create_description()
# end

descript(::Any, i) = nothing
descript(::Type{T}, i) where {T <: AbstractParameter} = (T,LengthParamDescription(T, i))

function type_to_curlycall(::Type{T}) where {T}
    q = Expr(:curly, Symbol(T.name))
    for i ∈ eachindex(T.parameters)
        push!(q.args, T.parameters[i])
    end
    Expr(:call, q)
end

function deconstruct_namedtuple(::Type{NamedTuple{N,T}}) where {N,T}
    names = Symbol[N...]
    Tp = T.parameters
    lengthdescripts = LengthParamDescription[]
    descripts = Expr[]
    track = Vector{Bool}(undef, length(names))
    j = 1
    for i ∈ eachindex(names)
        dld = descript(Tp[i], j)
        track[i] = notnothing = !isnothing(dld)
        if notnothing
            push!(descripts, type_to_curlycall(first(dld)))
            push!(lengthdescripts, last(dld))
            j += 1
        end
    end
    names, track, lengthdescripts, descripts
end

preprocess(m::Model, descriptnt) = preprocess!(copy(m), descriptnt)
function preprocess!(m::Model, descriptnt)
    names, track, lengthdescripts, descripts = deconstruct_namedtuple(descriptnt)
    preprocess!(m, names, track, lengthdescripts, descripts)
end

function preprocess!(m::Model, names::Vector{Symbol}, track::Vector{Bool}, lengthdescripts::Vector{LengthParamDescription}, descripts)
    ReverseDiffExpressions.reset_var_tracked!(m)
    offsets = DistributionParameters.parameter_offsets(lengthdescripts)
    parametervar = addvar!(m, Symbol("#θ#")); parametervar.tracked = true
    datavar = addvar!(m, Symbol("#DATA#")); parametervar.tracked = false
    target = m.vars[0]
    j = 0
    for i ∈ eachindex(names)
        name = names[i]
        var = getvar!(m, name)
        if (var.tracked = track[i]) # initialized by loading
            descriptvar = addvar!(m, Symbol(""))
            offsetvar = addvar!(m, Symbol(""))
            constrainret = addvar!(m, gensym(:targetconstrained))
            descriptvar.ref = descripts[(j += 1)]
            offsetvar.ref = offsets[j]
            constrainfunc = Func(Instruction(:DistributionParameters,:constrain), false)
            uses!(constrainfunc, parametervar)
            uses!(constrainfunc, offsetvar)
            uses!(constrainfunc, descriptvar)
            returns!(constrainfunc, constrainret)
            addfunc!(m, constrainfunc)
            getconstindex!(m, target, constrainret, 1)
            getconstindex!(m, var, constrainret, 2)
        else
            getprop = Func(Instruction(:Base,:getproperty), false)
            namevar = addvar!(m, Symbol("")); namevar.ref = name
            uses!(getprop, datavar)
            uses!(getprop, namevar)
            returns!(getprop, var)
            addfunc!(m, getprop)
        end
    end
    parametervar.tracked = true
    ReverseDiffExpressions.propagate_var_tracked!(m)
    m
end



