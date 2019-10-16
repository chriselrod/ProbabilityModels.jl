

function MCMCChainSummary(
    chains::AbstractArray{T,3}, model::AbstractProbabilityModel{P}; threaded::Bool = Threads.nthreads > 1
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
    for c ∈ 0:C-1
        for s ∈ 0:S-1
            s1 = PaddedMatrices.DynamicPtrVector{T}(ptr_ch1, P, stride_ch1)
            s2 = PaddedMatrices.DynamicPtrVector{T}(ptr_ch2, D, stride_ch2)
            DistributionParameters.constrain!(s2, model, s1)
            ptr_ch1 += sizeof(T) * stride_ch1
            ptr_ch2 += sizeof(T) * stride_ch2
        end
    end
    MCMCChainSummary(
        chainarray,
        DistributionParameters.parameter_names(model),
        threaded = threaded
    )
end

