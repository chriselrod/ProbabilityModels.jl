

### could make a direct method that doesn't allocate an extra wrap vector...
function MCMCChains.Chains(
    chains::Vector{DynamicHMC.NUTS_Transition{Vector{T},T}}, model::AbstractProbabilityModel{P}, args...
) where {T,P}
    Chains([chains], model, args...)
end
function MCMCChains.Chains(
    chains::Vector{Vector{DynamicHMC.NUTS_Transition{Vector{T},T}}}, model::AbstractProbabilityModel{P}, args...
) where {T,P}

    
    nchains = length(chains)
    samples = length(first(chains))
    D = DistributionParameters.constrained_length(model)

    chainarray = PaddedMatrices.DynamicPtrArray{T,3}(pointer(STACK_POINTER_REF[],T), (D, samples, nchains), D)
    ptr = pointer(chainarray)
    for c ∈ 1:nchains
        chain = chains[c]
        for s ∈ 1:samples
            v = PaddedMatrices.DynamicPtrVector{T}(ptr + sizeof(T) * ((c-1)*D*samples + (s-1)*D), (D,), D)
            DistributionParameters.constrain!(v, model, DynamicHMC.get_position(chain[s]))
        end
    end

    Chains(
        permutedims(chainarray, (2,1,3)),
        DistributionParameters.parameter_names(model),
        args...
    )
end

    
