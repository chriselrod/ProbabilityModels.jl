

function MCMCChainSummary(
    chains::AbstractArray{T,3}, model::AbstractProbabilityModel{P}; threaded::Bool = Threads.nthreads() > 1
) where {T,P}
    P2, S, C = size(chains)
    @assert P == P2 "The model is of a $P-dimensional parameter space, but we have $P2 parameters in our chain."
    D = DistributionParameters.constrained_length(model)
    # May as well allocate it on our stack, because we're returning a summary of this chain rather than the chain
    # this way, its at least recoverable
    stride_ch1 = stride(chains, 2)
    # stride_ch2 = PaddedMatrices.calc_padding(D, T)
    # chainarray = PaddedMatrices.DynamicPtrArray{T,3}(pointer(STACK_POINTER_REF[],T), (D, samples, nchains), stride_ch2)
    stride_ch2 = D
    chainarray = Array{T}(undef, D, S, C)
    ptr_ch1 = pointer(chains)
    ptr_ch2 = pointer(chainarray)
    for _ ∈ 1:C*S
        s1 = PaddedMatrices.DynamicPtrVector{T}(ptr_ch1, P, stride_ch1)
        s2 = PaddedMatrices.DynamicPtrVector{T}(ptr_ch2, D, stride_ch2)
        DistributionParameters.constrain!(s2, model, s1)
        ptr_ch1 += sizeof(T) * stride_ch1
        ptr_ch2 += sizeof(T) * stride_ch2
    end
    MCMCChainSummary(
        chainarray,
        DistributionParameters.parameter_names(model),
        threaded = threaded
    )
end


@generated function store_named_tuple!(v::AbstractArray{T}, nt::NT) where {T, NT <: NamedTuple}
    P = first(NT.parameters)
    q = quote ptrv = pointer(v) end
    for p ∈ P
        θ = gensym(:θ)
        push!(q.args, :($θ = nt.$p; PaddedMatrices.unique_copyto!(ptrv, $θ); ptrv += $(sizeof(T)) * type_length($θ)))
    end
    push!(q.args, nothing)
    q
end

# The plan here is to take a function as an argument
# this function is to take keyword arguments with the same name as model parameters -- capturing the unused in _..., eg
# `f(; a, b, _...) = a * b
# and then return a named tuple.
# Calls MCMCChainSummary on the named tuple.
function MCMCChainSummary(
    f::F, chains::AbstractArray{T,3}, model::AbstractProbabilityModel{P}; threaded::Bool = Threads.nthreads() > 1
) where {T,P,F}

    P2, S, C = size(chains)
    @assert P == P2 "The model is of a $P-dimensional parameter space, but we have $P2 parameters in our chain."
    stride_ch1 = stride(chains, 2)
    ptr_ch1 = pointer(chains)
    s1 = PaddedMatrices.DynamicPtrVector{T}(ptr_ch1, P, stride_ch1)
    fnt = f(; constrain(data, s1)...)
    D = PaddedMatrices.type_length(fnt)
    chainarray = Array{T}(undef, D, S, C)

    param_names = DistributionParameters.parameter_names(fnt)
    
    stride_ch2 = D
    ptr_ch2 = pointer(chainarray)
    s2 = PaddedMatrices.DynamicPtrVector{T}(ptr_ch2, D, stride_ch2)
    store_named_tuple!(s2, fnt)
    for _ ∈ 2:C*S
        ptr_ch1 += sizeof(T) * stride_ch1
        ptr_ch2 += sizeof(T) * stride_ch2

        s1 = PaddedMatrices.DynamicPtrVector{T}(ptr_ch1, P, stride_ch1)
        s2 = PaddedMatrices.DynamicPtrVector{T}(ptr_ch2, D, stride_ch2)
        fnt = f(; constrain(data, s1)...)
        store_named_tuple!(s2, fnt)        
    end
    MCMCChainSummary(
        chainarray,
        param_names,
        threaded = threaded
    )
end

    


