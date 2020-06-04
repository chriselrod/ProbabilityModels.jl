

function create_description()

end

descript(::Any, i) = nothing
descript(::Type{T}, i) where {T} = (T,LengthParamDescription(T, i))

function deconstruct_namedtuple(::Type{NamedTuple{N,T}}) where {N,T}
    names = Symbol[N...]
    Tp = T.parameters
    lengthdescripts = LengthParamDescription[]
    descripts = Any[]
    track = Vector{Bool}(undef, length(names))
    for i ∈ eachindex(names)
        dld = descript(Tp[i], i)
        track[i] = isnoth = isnothing(dld)
        if !isnoth
            push!(descripts, first(dld))
            push!(lengthdescripts, last(dld))
        end
    end
    names, track, lengthdescripts, descripts
end

function preprocess!(m::Model, names::Vector{Symbol}, track::Vector{Bool}, lengthdescripts::Vector{LengthParamDescription}, descripts)
    reset_var_tracked!(m)
    offsets = parameter_offsets(lengthdescripts)
    parametervar = addvar!(m, Symbol("#θ#")); parametervar.tracked = true
    datavar = addvar!(m, Symbol("#DATA#")); parametervar.tracked = false
    j = 0
    for i ∈ eachindex(names)
        name = names[i]
        var = getvar!(m, s)
        if (var.tracked = track[i]) # initialized by loading
            descriptvar = addvar!(m, Symbol(""))
            offsetvar = addvar!(m, Symbol(""))
            descriptvar.ref = descripts[(j += 1)]
            offsetvar.ref = offsets[j]
            constrainfunc = Func(Instruction(:DistributionParameters,:constrain), false, false)
            uses!(constrainfunc, parametervar)
            uses!(constrainfunc, offsetvar)
            uses!(constrainfunc, descriptvar)
            returns!(constrainfunc, var)
        else
            getprop = Func(Instruction(:Base,:getproperty), false, false)
            uses!(getprop, datavar)
            namevar = addvar!(m, Symbol("")); namevar.ref = name
            uses!(getprop, datavar)
            uses!(getprop, namevar)
            returns!(getprop, var)
        end
    end
    propagate_var_tracked!(m)
end



