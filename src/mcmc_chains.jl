

function MCMCChains.ess(
    chn::MCMCChains.AbstractChains;
    showall::Bool=false,
    sections::Union{Symbol, Vector{Symbol}}=Symbol[:parameters],
    maxlag::Int = 250
)
    param = showall ? names(chn) : names(chn, sections)
    n_chain_orig = size(chn, 3)

    T = Float64
    # Split the chains.
    parameter_vec = Vector{Vector{Vector{T}}}(undef, length(param))
    midpoint = Int32(size(chn, 1) >>> 1)
    for i in 1:length(param)
	parameter_vec[i] = Vector{T}[]
	for j in 1:n_chain_orig
            c1 = vec(MCMCChains.cskip(chn[1:midpoint, param[i], j].value.data))
            c2 = vec(MCMCChains.cskip(chn[midpoint+1:end, param[i], j].value.data))
            push!(parameter_vec[i], c1, c2)
	end
    end

    # Misc allocations.
    m = 2n_chain_orig
    n = minimum(length, parameter_vec[1])
    maxlag = min(maxlag, n-1)
    lags = collect(0:maxlag)

    # Preallocate B, W, varhat, and Rhat vectors for each param.
    B = Vector{T}(undef, length(param))
    W = Vector{T}(undef, length(param))
    varhat = Vector{T}(undef, length(param))
    Rhat = Vector{T}(undef, length(param))

    # calculate B, W, varhat, and Rhat for each param.
    for i in 1:length(param)
	draw = parameter_vec[i]
	p = param[i]
        allchain = mean(vcat([d for d in draw]...))
        eachchain = [mean(draw[j]) for j in 1:m]
        s = [sum((draw[j] .- eachchain[j]).^2) / (n-1) for j in 1:m]
        B[i] = (n / (m - 1)) * sum((eachchain .- allchain).^2)
        W[i] = sum(s) / m
        varhat[i] = (n-1)/n * W[i] + (B[i] / n)
        Rhat[i] = sqrt(varhat[i] / W[i])
    end

    V = Vector{Vector{T}}(undef, length(param))
    ρ = Vector{Vector{T}}(undef, length(param))
    for p in eachindex(V)
        V[p] = Vector{T}(undef, length(lags))
	ρ[p] = Vector{T}(undef, length(lags))
    end

    # Calculate ρ
    c_autocor = Vector{Vector{T}}(undef, length(param))
    for i in 1:length(param)
        c_autocor[i] = [0.0]
    end

    for t in eachindex(lags)
        lag = lags[t]
        range1 = lag+1:n
        range2 = 1:(n-lag)
        for i in 1:length(param)
            draw = parameter_vec[i]
	    p = param[i]
            z = [draw[j][range1] .- draw[j][range2] for j in 1:m]
	    z = sum([zi .^ 2 for zi in z])
            V[i][t] = 1 / (m * (n-lag)) * sum(z)
            autocors = [MCMCChains._autocorrelation(draw[j], lag) for j in 1:m]
	    ρ[i][t] = 1 - V[i][t] / (2 * varhat[i])
        end
    end

	# Find first odd positive integer where ρ[p][T+1] + ρ[p][T+2] is negative
    P = Vector{Vector{T}}(undef, length(param))
    essv = Vector{T}(undef, length(param))
    for i in 1:length(param)
        big_P = 0.0
	ρ_val = Float64.(ρ[i])

        # Big P.
        P[i] = Float64[ρ_val[1]]
        k = tprime = 1
        for tprime in 1:(length(lags)>>1) - 1
            sumvals = ρ_val[2*tprime] + ρ_val[2*tprime+1]
            if sumvals < 0
                break
            else
                push!(P[i], sumvals)
                k = tprime
            end
        end

        # Create monotone.
        P_monotone = [min(P[i][t], P[i][1:t]...) for t in 1:length(P[i])]

        essv[i] = (n*m) / (-1 + 2*sum(P_monotone))
	end

    df = MCMCChains.DataFrame(parameters = Symbol.(param), ess = essv, r_hat = Rhat)
    return MCMCChains.ChainDataFrame("ESS", df)#, digits=MCMCChains.digits)
end



### could make a direct method that doesn't allocate an extra wrap vector...
function MCMCChains.Chains(
    chains::Vector{Vector{T}}, model::AbstractProbabilityModel{P}, args...
) where {T,P}
    Chains([chains], model, args...)
end
function MCMCChains.Chains(
    chains::Vector{Vector{Vector{T}}}, model::AbstractProbabilityModel{P}, args...
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
            DistributionParameters.constrain!(v, model, chain[s])
        end
    end

    Chains(
        permutedims(chainarray, (2,1,3)),
        DistributionParameters.parameter_names(model),
        args...
    )
end

### While not optimized, the outputs are at least not Any[]    
