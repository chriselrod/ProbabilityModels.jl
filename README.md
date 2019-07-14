

This is alpha-quality software. It is under active development. Optimistically, I hope to have it and its dependencies reasonably well documented and tested, and all the libraries registered, by the end of the year. There is a roadmap issue [here](https://github.com/chriselrod/ProbabilityModels.jl/issues/5).

The primary goal of this library is to make it as easy as possible to specify models that run as quickly as possible $-$ providing both log densities and the associated gradients. This allows the library to be a front end to Hamiltonian Monte Carlo backends, such as [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl). [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)/[Turing's NUTS](https://github.com/TuringLang/Turing.jl) is also probably worth looking into, as it [reportedly converges the most reliably](https://discourse.julialang.org/t/mcmc-landscape/25654/11?u=elrod), at least within DiffEqBayes.


*A brief introduction.*

First, you specify a model using a DSL:
```julia
using Distributed
addprocs(Sys.CPU_THREADS >> 1, enable_threaded_blas = false);

@everywhere begin
using PaddedMatrices, StructuredMatrices, DistributionParameters, LoopVectorization
using Random, SpecialFunctions, MCMCChains, DynamicHMC, LinearAlgebra, MCMCDiagnostics
using ProbabilityModels, LoopVectorization, SLEEFPirates, SIMDPirates, ProbabilityDistributions, PaddedMatrices
using ProbabilityModels: HierarchicalCentering, ∂HierarchicalCentering, ITPExpectedValue, ∂ITPExpectedValue
using DistributionParameters: CovarianceMatrix, MissingDataVector#, add
using LogDensityProblems: Value, ValueGradient, AbstractLogDensityProblem, dimension, logdensity
using PaddedMatrices: vexp
BLAS.set_num_threads(1)
end

BLAS.set_num_threads(1)
@everywhere begin

@model ITPModel begin
    # Non-hierarchical Priors
    ρ ~ Beta(3, 1)
    lκ ~ lsgg(8.5, 1.5, 3.0, 1.5271796258079011) # μ = 1, σ² = 10
    σ ~ Gamma(1.5, 0.25) # μ = 6, σ² = 2.4
    θ ~ Normal(10)
    L ~ LKJ(2.0)

    # Hierarchical Priors.
    # h subscript, for highest in the hierarhcy.
    μₕ₁ ~ Normal(10) # μ = 0
    μₕ₂ ~ Normal(10) # μ = 0
    σₕ ~ Normal(10) # μ = 0
    # Raw μs; non-cenetered parameterization
    μᵣ₁ ~ Normal() # μ = 0, σ = 1
    μᵣ₂ ~ Normal() # μ = 0, σ = 1
    # Center the μs
    μᵦ₁ = HierarchicalCentering(μᵣ₁, μₕ₁, σₕ)
    μᵦ₂ = HierarchicalCentering(μᵣ₂, μₕ₂, σₕ)
    σᵦ ~ Normal(10) # μ = 0
    # Raw βs; non-cenetered parameterization
    βᵣ₁ ~ Normal()
    βᵣ₂ ~ Normal()
    # Center the βs.
    β₁ = HierarchicalCentering(βᵣ₁, μᵦ₁, σᵦ, domains)
    β₂ = HierarchicalCentering(βᵣ₂, μᵦ₂, σᵦ, domains)

    # Likelihood
    κ = vexp(lκ)
    μ₁ = vec(ITPExpectedValue(time, β₁, κ, θ))
    μ₂ = vec(ITPExpectedValue(time, β₂, κ, θ))
    Σ = CovarianceMatrix(ρ, Diagonal(σ) * L, time)

    Y ~ Normal((μ₁, μ₂)[AvailableData], Σ[AvailableData])

end
end
# Evaluating the code prints:
#    Defined model: ITPModel.
#    Unknowns: domains, μₕ₂, μᵣ₁, time, σ, AvailableData, σᵦ, Y, θ, μᵣ₂, ρ, σₕ, lκ, μₕ₁, L, βᵣ₂, βᵣ₁.
```

The macro uses the following expression to define a struct and several functions.
The struct has one field for each unknown.

We can then define an instance of the struct, where we specify each of these unknowns either with an instance, or with a type.
Those specified with a type are unknown parameters (of that type); those specified with an instance treat that instance as known priors or data.

Before spewing boilerplate to generate random true values and fake data, a brief summary of the model:
We have longitudinal multivariate observations for some number of subjects. However, not all observations are measured at all times. That is, while subjects may be measured at multiple time (I use $T=36$ times below), only some measurements are taken at any given time, yielding missing data.
Therefore, we subset the full covariance matrix (produced from a vector of autocorrelations, and the Cholesky factor of a covariance matrix across measurements) to find the marginal.

The expected value is function of time (`ITPExpectedValue`). This function returns a matrix  (`time x measurement`), so we `vec` it and subset it.

We also expect some measurements to bare more in common than others, so we group them into "domains", and provide hierarchical priors. We use a non-cenetered parameterization, and the function `HierarchicalCentering` then centers our parameters for us. There are two methods we use above: one takes scalars, to transform a vector. The other accepts different domain means and standard deviations, and uses these to transform a vector, taking the indices from the `Domains` argument. That is, if the first element of `Domains` is 2, indicating that the first 2 measurements belong to the first domain, it will transform the first two elements of `βᵣ₁` with the first element of `μᵦ₁` and `σᵦ` (if either `μᵦ₁` or `σᵦ` are scalars, they will be broadcasted across each domain).


```julia
function rinvscaledgamma(::Val{N},a::T,b::T,c::T) where {N,T}
    rg = MutableFixedSizePaddedVector{N,T}(undef)
    log100 = log(100)
    @inbounds for n ∈ 1:N
        rg[n] = log100 / exp(log(PaddedMatrices.randgamma(a/c)) / c + b)
    end
    rg
end

const domains = ProbabilityModels.Domains(2,2,3)

const n_endpoints = sum(domains)

const times = MutableFixedSizePaddedVector{36,Float64,36,36}(undef); times .= 0:35;

missing = push!(vcat(([1,0,0,0,0] for i ∈ 1:7)...), 1);
missing_pattern = vcat(
    missing, fill(1, 4length(times)), missing, missing
);

const availabledata = MissingDataVector{Float64}(missing_pattern);

const κ₀ = (8.5, 1.5, 3.0)

AR1(ρ, t) = @. ρ ^ abs(t - t')

function generate_true_parameters(domains, time, κ₀)
    K = sum(domains)
    D = length(domains)
    T = length(time)
    μ = MutableFixedSizePaddedVector{K,Float64}(undef)

    offset = 0
    for i ∈ domains
        domain_mean = 10randn()
        for j ∈ 1+offset:i+offset
            μ[j] = domain_mean + 15randn()
        end
        offset += i
    end

    σ = PaddedMatrices.randgamma( 6.0, 1/6.0)

    ρ = PaddedMatrices.randbeta(4.0,4.0)

    κ = rinvscaledgamma(Val(K), κ₀...)
    lt = last(time)
    θ₁ = @. μ' - 0.05 * ( 1.0 - exp( - κ' * time) ) / (1.0 - exp( - κ' * lt) ) 
    θ₂ = @. μ' + 0.05 * ( 1.0 - exp( - κ' * time) ) / (1.0 - exp( - κ' * lt) ) 
    
    L_T, info = LAPACK.potrf!('L', AR1(ρ, time))
    @inbounds for tc ∈ 2:T, tr ∈ 1:tc-1
        L_T[tr,tc] = 0.0
    end
    X = PaddedMatrices.MutableFixedSizePaddedMatrix{K,K+3,Float64,K}(undef); randn!(X)
    U_K, info = LAPACK.potrf!('U', BLAS.syrk!('U', 'N', σ, X, 0.0, zero(MutableFixedSizePaddedMatrix{K,K,Float64,K})))
    (
        U_K = U_K, L_T = L_T, μ = μ, θ₁ = θ₁, θ₂ = θ₂, domains = domains, time = time
    )
end

@generated function randomize!(
    sp::PaddedMatrices.StackPointer,
    A::AbstractArray{T,P},
    B::PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{M,M,T},
    C::PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{N,N,T},
    D::PaddedMatrices.AbstractMutableFixedSizePaddedMatrix{M,N,T}
) where {M,N,T,P}
    quote
        @boundscheck begin
            d = size(A,1)
            for p ∈ 2:$(P-1)
                d *= size(A,p)
            end
            d == M*N || PaddedMatrices.ThrowBoundsError("Earlier dims size(A) == $(size(A)) does not match size(D) == ($M,$N)")
        end
        ptr = pointer(sp, $T)
        E = PtrMatrix{$M,$N,$T,$M}( ptr )
        F = PtrMatrix{$M,$N,$T,$M}( ptr + $(sizeof(T) * M * N) )
        ptr_A = pointer(A)
        GC.@preserve A begin
            for n ∈ 0:size(A,$P)-1
                Aₙ = PtrMatrix{$M,$N,$T,$M}( ptr_A + n*$(sizeof(T)*M*N) )
                randn!(ProbabilityModels.GLOBAL_ScalarVectorPCGs[1], E)
                mul!(F, B, E)
                Aₙ .= D
                PaddedMatrices.gemm!(Aₙ, F, C)
            end
        end
        sp
    end
end

sample_data( N, truth, missingness ) = sample_data( N, truth, missingness, truth.domains )
@generated function sample_data( N::Tuple{Int,Int}, truth, missingness, ::ProbabilityModels.Domains{S} ) where {S}
    K = sum(S)
    D = length(S)
    quote
        N₁, N₂ = N
        L_T = truth.L_T
        U_K = truth.U_K
        T = size(L_T,1)

        sp = ProbabilityModels.STACK_POINTER_REF[]
        (sp,Y₁) = PaddedMatrices.DynamicPtrArray{Float64,3}(sp, (T, $K, N₁), T)
        (sp,Y₂) = PaddedMatrices.DynamicPtrArray{Float64,3}(sp, (T, $K, N₂), T)
        randomize!(sp, Y₁, L_T, U_K, truth.θ₁)
        randomize!(sp, Y₂, L_T, U_K, truth.θ₂)
        
        c = length(missingness.indices)
        inds = missingness.indices
        ITPModel(
            domains = truth.domains,
            AvailableData = missingness,
            Y = (DynamicPaddedMatrix(reshape(Y₁, (T * $K, N₁))[inds, :], (c, N₁)),
                 DynamicPaddedMatrix(reshape(Y₂, (T * $K, N₂))[inds, :], (c, N₂))),
            time = truth.time,
            L = LKJCorrCholesky{$K},
            ρ = UnitVector{$K},
            lκ = RealVector{$K},
            θ = RealVector{$K},
            μₕ₁ = RealFloat,
            μₕ₂ = RealFloat,
            μᵣ₁ = RealVector{$D},
            μᵣ₂ = RealVector{$D},
            βᵣ₁ = RealVector{$K},
            βᵣ₂ = RealVector{$K},
            σᵦ = PositiveFloat,
            σₕ = PositiveFloat,
            σ = PositiveVector{$K}
        )
    end
end

truth = generate_true_parameters(domains, times, κ₀);

data = sample_data((100,100), truth, availabledata); # Sample size of 100 for both treatment and placebo groups.

```
The library [DistributionParameters.jl](https://github.com/chriselrod/DistributionParameters.jl) provides a variety of parameter types.
These types define constrianing transformations, to transform an unconstrained parameter vector and add the appropriate jacobians.

All parameters are typed by size. The library currently provides a DynamicHMC interface, defining `logdensity(::Value,::ITPModel)` and `logdensity(::ValueGradient,::ITPModel)` methods.

```julia
using LogDensityProblems: Value, ValueGradient, logdensity, dimension
using DynamicHMC

a = randn(dimension(data)); length(a) # 73

using BenchmarkTools

@benchmark logdensity(ValueGradient, $data, $a)
#BenchmarkTools.Trial: 
#  memory estimate:  720 bytes
#  allocs estimate:  3
#  --------------
#  minimum time:     567.967 μs (0.00% GC)
#  median time:      575.273 μs (0.00% GC)
#  mean time:        576.330 μs (0.00% GC)
#  maximum time:     787.441 μs (0.00% GC)
#  --------------
#  samples:          8645
#  evals/sample:     1
```
For comparison, a Stan implementation of this model takes close to 13ms to evaluate the gradient.

These are all you need for sampling using [DynamicHMC](https://github.com/tpapp/DynamicHMC.jl). The `@model` macro also defines a helper function for [constraining](https://mc-stan.org/docs/2_19/reference-manual/variable-transforms-chapter.html) unconstrained parameter vectors:

```julia
constrained_a = constrain(data, a)
#7×7 LKJCorrCholesky{7,Float64,28}:
#  1.0         0.0        0.0         0.0        0.0        0.0         0.0     
# -0.576887    0.816824   0.0         0.0        0.0        0.0         0.0     
#  0.0972928  -0.271262   0.957576    0.0        0.0        0.0         0.0     
#  0.0926267  -0.34767   -0.78463     0.504877   0.0        0.0         0.0     
#  0.0508829   0.281452  -0.0479868  -0.117385   0.949797   0.0         0.0     
# -0.309485    0.386767   0.0319051  -0.443837  -0.1412     0.732587    0.0     
# -0.357164    0.562609   0.0502195   0.691164  -0.077583  -0.00463478  0.263884
```
So you can use this to constrain the unconstrained parameter vectors `DynamicHMC` sampled and proceed with your convergence assessments and posterior analysis as normal.

Alternatively, it also supports [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl), constraining the parameters for you and providing posterior summaries as well as plotting methods:
```julia
julia> using MCMCChains

julia> bdt = DynamicHMC.bracketed_doubling_tuner(M=6, term=100); #default is M=5 doubling steps and term=50;

julia> @time chains, tuned_samplers = NUTS_init_tune_distributed(data, 10_000, report = DynamicHMC.ReportSilent(), tuners = bdt);
      From worker 3:	313.348820 seconds (21.70 M allocations: 3.125 GiB, 0.29% gc time)
      From worker 6:	333.319974 seconds (23.98 M allocations: 3.451 GiB, 0.28% gc time)
      From worker 15:	338.096758 seconds (24.61 M allocations: 3.542 GiB, 0.25% gc time)
      From worker 11:	342.206189 seconds (23.92 M allocations: 3.443 GiB, 0.25% gc time)
      From worker 4:	345.819174 seconds (24.98 M allocations: 3.594 GiB, 0.28% gc time)
      From worker 12:	345.986428 seconds (25.13 M allocations: 3.616 GiB, 0.25% gc time)
      From worker 14:	346.169429 seconds (24.84 M allocations: 3.574 GiB, 0.25% gc time)
      From worker 7:	346.550631 seconds (25.17 M allocations: 3.621 GiB, 0.25% gc time)
      From worker 8:	346.785025 seconds (24.43 M allocations: 3.517 GiB, 0.27% gc time)
      From worker 16:	346.974315 seconds (24.25 M allocations: 3.491 GiB, 0.25% gc time)
      From worker 13:	347.609331 seconds (24.97 M allocations: 3.593 GiB, 0.24% gc time)
      From worker 5:	347.755439 seconds (25.14 M allocations: 3.618 GiB, 0.25% gc time)
      From worker 19:	354.085933 seconds (25.19 M allocations: 3.625 GiB, 0.24% gc time)
      From worker 18:	355.429545 seconds (25.44 M allocations: 3.660 GiB, 0.25% gc time)
      From worker 9:	357.541106 seconds (25.63 M allocations: 3.687 GiB, 0.27% gc time)
      From worker 2:	359.304351 seconds (25.77 M allocations: 3.707 GiB, 0.19% gc time)
      From worker 17:	364.385363 seconds (26.38 M allocations: 3.794 GiB, 0.25% gc time)
      From worker 10:	382.036894 seconds (28.14 M allocations: 4.045 GiB, 0.24% gc time)
382.137428 seconds (6.94 M allocations: 299.073 MiB, 0.09% gc time)

julia> mcmcchains = Chains(chains, data)
Object of type Chains, with data of type 10000×80×18 Array{Float64,3}

Iterations        = 1:10000
Thinning interval = 1
Chains            = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
Samples per chain = 10000
parameters        = μₕ₂, μᵣ₁[1], μᵣ₁[2], μᵣ₁[3], σ[1], σ[2], σ[3], σ[4], σ[5], σ[6], σ[7], σᵦ, θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7], μᵣ₂[1], μᵣ₂[2], μᵣ₂[3], ρ[1], ρ[2], ρ[3], ρ[4], ρ[5], ρ[6], ρ[7], σₕ, lκ[1], lκ[2], lκ[3], lκ[4], lκ[5], lκ[6], lκ[7], μₕ₁, L[1,1], L[2,2], L[3,3], L[4,4], L[5,5], L[6,6], L[7,7], L[2,1], L[3,1], L[4,1], L[5,1], L[6,1], L[7,1], L[3,2], L[4,2], L[5,2], L[6,2], L[7,2], L[4,3], L[5,3], L[6,3], L[7,3], L[5,4], L[6,4], L[7,4], L[6,5], L[7,5], L[7,6], βᵣ₂[1], βᵣ₂[2], βᵣ₂[3], βᵣ₂[4], βᵣ₂[5], βᵣ₂[6], βᵣ₂[7], βᵣ₁[1], βᵣ₁[2], βᵣ₁[3], βᵣ₁[4], βᵣ₁[5], βᵣ₁[6], βᵣ₁[7]

2-element Array{ChainDataFrame,1}

Summary Statistics

│ Row │ parameters │ mean       │ std        │ naive_se    │ mcse        │ ess       │ r_hat   │
│     │ Symbol     │ Float64    │ Float64    │ Float64     │ Float64     │ Float64   │ Float64 │
├─────┼────────────┼────────────┼────────────┼─────────────┼─────────────┼───────────┼─────────┤
│ 1   │ L[1,1]     │ 1.0        │ 0.0        │ 0.0         │ 0.0         │ NaN       │ NaN     │
│ 2   │ L[2,1]     │ -0.478473  │ 0.0178024  │ 4.19606e-5  │ 7.63095e-5  │ 26709.1   │ 1.00058 │
│ 3   │ L[2,2]     │ 0.877868   │ 0.00970079 │ 2.2865e-5   │ 4.1953e-5   │ 25764.0   │ 1.00061 │
│ 4   │ L[3,1]     │ -0.402254  │ 0.0190476  │ 4.48957e-5  │ 9.41925e-5  │ 12925.5   │ 1.00474 │
│ 5   │ L[3,2]     │ 0.126741   │ 0.0159552  │ 3.76068e-5  │ 6.17012e-5  │ 43824.4   │ 1.0014  │
│ 6   │ L[3,3]     │ 0.906341   │ 0.0076011  │ 1.7916e-5   │ 3.84277e-5  │ 11782.5   │ 1.00476 │
│ 7   │ L[4,1]     │ 0.0492235  │ 0.0213896  │ 5.04158e-5  │ 9.75705e-5  │ 31217.3   │ 1.00086 │
│ 8   │ L[4,2]     │ 0.304339   │ 0.0150707  │ 3.55219e-5  │ 6.97791e-5  │ 27858.2   │ 1.00106 │
│ 9   │ L[4,3]     │ -0.182724  │ 0.0136038  │ 3.20644e-5  │ 5.79779e-5  │ 29839.0   │ 1.00069 │
│ 10  │ L[4,4]     │ 0.933098   │ 0.00503294 │ 1.18627e-5  │ 2.03199e-5  │ 45641.4   │ 1.00086 │
│ 11  │ L[5,1]     │ -0.203874  │ 0.0207951  │ 4.90146e-5  │ 8.28775e-5  │ 50094.0   │ 1.00047 │
│ 12  │ L[5,2]     │ -0.0236749 │ 0.0154886  │ 3.65069e-5  │ 5.46307e-5  │ 81443.7   │ 1.00013 │
│ 13  │ L[5,3]     │ 0.0557084  │ 0.0140963  │ 3.32253e-5  │ 5.14786e-5  │ 71861.0   │ 1.00015 │
│ 14  │ L[5,4]     │ -0.648184  │ 0.0100496  │ 2.36872e-5  │ 5.89598e-5  │ 9011.22   │ 1.00512 │
│ 15  │ L[5,5]     │ 0.730472   │ 0.00827856 │ 1.95128e-5  │ 4.57303e-5  │ 10107.6   │ 1.00401 │
│ 16  │ L[6,1]     │ -0.117446  │ 0.0227043  │ 5.35145e-5  │ 6.78709e-5  │ 1.08771e5 │ 1.00035 │
│ 17  │ L[6,2]     │ -0.198179  │ 0.0200177  │ 4.71821e-5  │ 7.77325e-5  │ 180000.0  │ 1.00057 │
│ 18  │ L[6,3]     │ 0.23825    │ 0.0196412  │ 4.62948e-5  │ 8.18355e-5  │ 54512.3   │ 1.0021  │
│ 19  │ L[6,4]     │ 0.490104   │ 0.0164317  │ 3.87299e-5  │ 5.84155e-5  │ 48300.3   │ 1.0005  │
│ 20  │ L[6,5]     │ 0.352864   │ 0.0172267  │ 4.06038e-5  │ 6.65666e-5  │ 180000.0  │ 1.00114 │
│ 21  │ L[6,6]     │ 0.7235     │ 0.0115637  │ 2.72559e-5  │ 4.14338e-5  │ 180000.0  │ 1.00088 │
│ 22  │ L[7,1]     │ -0.0998906 │ 0.0236285  │ 5.56929e-5  │ 8.03711e-5  │ 68482.3   │ 1.00009 │
│ 23  │ L[7,2]     │ 0.0290148  │ 0.0224258  │ 5.28582e-5  │ 7.83333e-5  │ 54812.1   │ 1.0004  │
│ 24  │ L[7,3]     │ -0.132429  │ 0.0218494  │ 5.14994e-5  │ 6.44875e-5  │ 180000.0  │ 1.00013 │
│ 25  │ L[7,4]     │ 0.505747   │ 0.0178294  │ 4.20244e-5  │ 7.44108e-5  │ 23256.8   │ 1.00129 │
│ 26  │ L[7,5]     │ -0.1484    │ 0.0209314  │ 4.93358e-5  │ 6.67215e-5  │ 180000.0  │ 1.00057 │
│ 27  │ L[7,6]     │ 0.0947709  │ 0.0206655  │ 4.87091e-5  │ 6.25594e-5  │ 180000.0  │ 1.00017 │
│ 28  │ L[7,7]     │ 0.825836   │ 0.0114891  │ 2.708e-5    │ 4.87939e-5  │ 21295.1   │ 1.00132 │
│ 29  │ lκ[1]      │ -0.260576  │ 0.215239   │ 0.000507322 │ 0.000759521 │ 180000.0  │ 1.00032 │
│ 30  │ lκ[2]      │ -0.269794  │ 0.211165   │ 0.000497721 │ 0.000682632 │ 180000.0  │ 1.00017 │
│ 31  │ lκ[3]      │ -0.257728  │ 0.21539    │ 0.00050768  │ 0.000696768 │ 1.01561e5 │ 1.00015 │
│ 32  │ lκ[4]      │ -0.25589   │ 0.216638   │ 0.000510622 │ 0.000708778 │ 180000.0  │ 1.00058 │
│ 33  │ lκ[5]      │ -0.255693  │ 0.217585   │ 0.000512852 │ 0.000736578 │ 91373.9   │ 1.00012 │
│ 34  │ lκ[6]      │ -0.258656  │ 0.216618   │ 0.000510573 │ 0.000818239 │ 43345.3   │ 1.00017 │
│ 35  │ lκ[7]      │ -0.261338  │ 0.217534   │ 0.000512732 │ 0.000758309 │ 54259.1   │ 1.00047 │
│ 36  │ βᵣ₁[1]     │ 0.0582778  │ 0.952366   │ 0.00224475  │ 0.00294041  │ 180000.0  │ 1.00034 │
│ 37  │ βᵣ₁[2]     │ 0.0625387  │ 0.920602   │ 0.00216988  │ 0.00291704  │ 180000.0  │ 1.00005 │
│ 38  │ βᵣ₁[3]     │ 0.202762   │ 0.974636   │ 0.00229724  │ 0.00454276  │ 180000.0  │ 1.00157 │
│ 39  │ βᵣ₁[4]     │ -0.0231755 │ 0.94925    │ 0.0022374   │ 0.00346116  │ 180000.0  │ 1.00084 │
│ 40  │ βᵣ₁[5]     │ -0.128748  │ 0.966131   │ 0.00227719  │ 0.00349783  │ 180000.0  │ 1.00063 │
│ 41  │ βᵣ₁[6]     │ 0.00713696 │ 0.949167   │ 0.00223721  │ 0.00309299  │ 180000.0  │ 1.00025 │
│ 42  │ βᵣ₁[7]     │ -0.166106  │ 0.939446   │ 0.0022143   │ 0.00317415  │ 180000.0  │ 1.00106 │
│ 43  │ βᵣ₂[1]     │ -0.0911731 │ 0.94888    │ 0.00223653  │ 0.00316671  │ 180000.0  │ 1.00015 │
│ 44  │ βᵣ₂[2]     │ 0.261473   │ 0.931469   │ 0.00219549  │ 0.00291179  │ 180000.0  │ 1.00011 │
│ 45  │ βᵣ₂[3]     │ -0.0985443 │ 0.966921   │ 0.00227906  │ 0.00426482  │ 180000.0  │ 1.00291 │
│ 46  │ βᵣ₂[4]     │ -0.29074   │ 0.96938    │ 0.00228485  │ 0.0034118   │ 180000.0  │ 1.00043 │
│ 47  │ βᵣ₂[5]     │ -0.0645795 │ 0.954458   │ 0.00224968  │ 0.00324615  │ 180000.0  │ 1.00017 │
│ 48  │ βᵣ₂[6]     │ 0.127262   │ 0.955353   │ 0.00225179  │ 0.00287524  │ 180000.0  │ 1.00028 │
│ 49  │ βᵣ₂[7]     │ 0.152003   │ 0.934001   │ 0.00220146  │ 0.00268437  │ 180000.0  │ 1.00008 │
│ 50  │ θ[1]       │ -16.8845   │ 0.06181    │ 0.000145688 │ 0.000254918 │ 26544.5   │ 1.00133 │
│ 51  │ θ[2]       │ -16.9102   │ 0.059423   │ 0.000140061 │ 0.00026824  │ 21319.3   │ 1.00074 │
│ 52  │ θ[3]       │ 18.5267    │ 0.0886497  │ 0.000208949 │ 0.000411193 │ 20299.6   │ 1.00225 │
│ 53  │ θ[4]       │ -13.2301   │ 0.0900982  │ 0.000212363 │ 0.000332172 │ 71353.8   │ 1.00024 │
│ 54  │ θ[5]       │ 7.58721    │ 0.0922159  │ 0.000217355 │ 0.000310865 │ 68292.9   │ 1.00042 │
│ 55  │ θ[6]       │ 6.54907    │ 0.0920535  │ 0.000216972 │ 0.000480391 │ 13453.7   │ 1.00175 │
│ 56  │ θ[7]       │ -7.72806   │ 0.0663461  │ 0.000156379 │ 0.000287743 │ 35950.2   │ 1.00045 │
│ 57  │ μᵣ₁[1]     │ 0.16687    │ 0.835823   │ 0.00197005  │ 0.00267075  │ 1.04374e5 │ 1.00011 │
│ 58  │ μᵣ₁[2]     │ 0.198932   │ 0.848831   │ 0.00200071  │ 0.00267088  │ 94427.9   │ 1.00022 │
│ 59  │ μᵣ₁[3]     │ -0.365877  │ 0.875541   │ 0.00206367  │ 0.00265895  │ 1.17193e5 │ 1.00012 │
│ 60  │ μᵣ₂[1]     │ 0.249491   │ 0.845326   │ 0.00199245  │ 0.00401125  │ 15550.5   │ 1.00329 │
│ 61  │ μᵣ₂[2]     │ -0.467879  │ 0.877655   │ 0.00206865  │ 0.00473845  │ 9934.4    │ 1.00488 │
│ 62  │ μᵣ₂[3]     │ 0.263443   │ 0.877093   │ 0.00206733  │ 0.0037542   │ 21822.5   │ 1.00207 │
│ 63  │ μₕ₁        │ -0.0388884 │ 0.068792   │ 0.000162144 │ 0.000428758 │ 18140.7   │ 1.00085 │
│ 64  │ μₕ₂        │ 0.0554376  │ 0.0836321  │ 0.000197123 │ 0.00117632  │ 1175.5    │ 1.03948 │
│ 65  │ ρ[1]       │ 0.706547   │ 0.0144883  │ 3.41493e-5  │ 5.58215e-5  │ 31369.7   │ 1.0015  │
│ 66  │ ρ[2]       │ 0.718447   │ 0.00954273 │ 2.24924e-5  │ 3.9038e-5   │ 28589.2   │ 1.00072 │
│ 67  │ ρ[3]       │ 0.732575   │ 0.00839972 │ 1.97983e-5  │ 3.10784e-5  │ 53882.4   │ 1.00046 │
│ 68  │ ρ[4]       │ 0.732384   │ 0.00798249 │ 1.88149e-5  │ 3.55089e-5  │ 25201.8   │ 1.00193 │
│ 69  │ ρ[5]       │ 0.717402   │ 0.00810842 │ 1.91117e-5  │ 2.68802e-5  │ 86440.5   │ 1.00013 │
│ 70  │ ρ[6]       │ 0.740136   │ 0.0174909  │ 4.12265e-5  │ 5.19209e-5  │ 180000.0  │ 1.00015 │
│ 71  │ ρ[7]       │ 0.678775   │ 0.0259173  │ 6.10877e-5  │ 0.000122586 │ 180000.0  │ 1.0031  │
│ 72  │ σ[1]       │ 1.59578    │ 0.0279171  │ 6.58011e-5  │ 8.51829e-5  │ 1.08873e5 │ 1.00024 │
│ 73  │ σ[2]       │ 1.43525    │ 0.019655   │ 4.63272e-5  │ 8.60034e-5  │ 24819.1   │ 1.0017  │
│ 74  │ σ[3]       │ 2.46364    │ 0.033579   │ 7.91464e-5  │ 0.000114648 │ 89069.6   │ 1.0001  │
│ 75  │ σ[4]       │ 2.58908    │ 0.0355521  │ 8.3797e-5   │ 0.000199726 │ 10454.6   │ 1.00475 │
│ 76  │ σ[5]       │ 2.75918    │ 0.0326132  │ 7.68702e-5  │ 0.000158809 │ 14596.8   │ 1.00313 │
│ 77  │ σ[6]       │ 2.51261    │ 0.042254   │ 9.95936e-5  │ 0.00017393  │ 26832.4   │ 1.00065 │
│ 78  │ σ[7]       │ 1.59019    │ 0.0280486  │ 6.61113e-5  │ 9.2046e-5   │ 69828.8   │ 1.00015 │
│ 79  │ σᵦ         │ 0.0389312  │ 0.0319806  │ 7.53791e-5  │ 0.000143832 │ 48354.4   │ 1.00034 │
│ 80  │ σₕ         │ 0.0727634  │ 0.0691276  │ 0.000162935 │ 0.000845613 │ 1886.93   │ 1.02068 │

Quantiles

│ Row │ parameters │ 2.5%       │ 25.0%      │ 50.0%      │ 75.0%      │ 97.5%      │
│     │ Symbol     │ Float64    │ Float64    │ Float64    │ Float64    │ Float64    │
├─────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
│ 1   │ L[1,1]     │ 1.0        │ 1.0        │ 1.0        │ 1.0        │ 1.0        │
│ 2   │ L[2,1]     │ -0.51285   │ -0.490586  │ -0.478617  │ -0.46661   │ -0.442924  │
│ 3   │ L[2,2]     │ 0.858478   │ 0.871393   │ 0.878024   │ 0.884463   │ 0.896559   │
│ 4   │ L[3,1]     │ -0.439069  │ -0.415211  │ -0.402397  │ -0.389456  │ -0.364638  │
│ 5   │ L[3,2]     │ 0.0954889  │ 0.116032   │ 0.126689   │ 0.137507   │ 0.158133   │
│ 6   │ L[3,3]     │ 0.89091    │ 0.901319   │ 0.906545   │ 0.911563   │ 0.920688   │
│ 7   │ L[4,1]     │ 0.00711479 │ 0.034955   │ 0.0490853  │ 0.0636012  │ 0.0911257  │
│ 8   │ L[4,2]     │ 0.274611   │ 0.294163   │ 0.304397   │ 0.314541   │ 0.333793   │
│ 9   │ L[4,3]     │ -0.2094    │ -0.19194   │ -0.182705  │ -0.173635  │ -0.15609   │
│ 10  │ L[4,4]     │ 0.922862   │ 0.929759   │ 0.933236   │ 0.936551   │ 0.94262    │
│ 11  │ L[5,1]     │ -0.244458  │ -0.218038  │ -0.203944  │ -0.189901  │ -0.162716  │
│ 12  │ L[5,2]     │ -0.0539329 │ -0.0341471 │ -0.0236596 │ -0.013254  │ 0.00670673 │
│ 13  │ L[5,3]     │ 0.0279306  │ 0.0462618  │ 0.0557891  │ 0.0652091  │ 0.0833003  │
│ 14  │ L[5,4]     │ -0.667973  │ -0.654886  │ -0.648213  │ -0.641465  │ -0.628428  │
│ 15  │ L[5,5]     │ 0.713676   │ 0.724964   │ 0.730543   │ 0.736106   │ 0.746457   │
│ 16  │ L[6,1]     │ -0.161654  │ -0.132781  │ -0.117636  │ -0.102148  │ -0.0726235 │
│ 17  │ L[6,2]     │ -0.236664  │ -0.211857  │ -0.198274  │ -0.18456   │ -0.15876   │
│ 18  │ L[6,3]     │ 0.199671   │ 0.224939   │ 0.23822    │ 0.251591   │ 0.276622   │
│ 19  │ L[6,4]     │ 0.457325   │ 0.479159   │ 0.490254   │ 0.50127    │ 0.521674   │
│ 20  │ L[6,5]     │ 0.319092   │ 0.341234   │ 0.352931   │ 0.364502   │ 0.386501   │
│ 21  │ L[6,6]     │ 0.700879   │ 0.715678   │ 0.72351    │ 0.731288   │ 0.746121   │
│ 22  │ L[7,1]     │ -0.145879  │ -0.115935  │ -0.0999413 │ -0.0839292 │ -0.0535237 │
│ 23  │ L[7,2]     │ -0.0147229 │ 0.0137868  │ 0.0289015  │ 0.0442174  │ 0.072867   │
│ 24  │ L[7,3]     │ -0.175151  │ -0.147109  │ -0.132437  │ -0.11783   │ -0.0892502 │
│ 25  │ L[7,4]     │ 0.470306   │ 0.493842   │ 0.505891   │ 0.517793   │ 0.54061    │
│ 26  │ L[7,5]     │ -0.189263  │ -0.162551  │ -0.148485  │ -0.13435   │ -0.107164  │
│ 27  │ L[7,6]     │ 0.054339   │ 0.0807244  │ 0.0947879  │ 0.108726   │ 0.135335   │
│ 28  │ L[7,7]     │ 0.802855   │ 0.8182     │ 0.825955   │ 0.833609   │ 0.848      │
│ 29  │ lκ[1]      │ -0.621804  │ -0.412519  │ -0.281087  │ -0.131208  │ 0.220109   │
│ 30  │ lκ[2]      │ -0.625794  │ -0.419636  │ -0.289551  │ -0.142702  │ 0.201834   │
│ 31  │ lκ[3]      │ -0.617064  │ -0.409569  │ -0.2787    │ -0.130185  │ 0.224449   │
│ 32  │ lκ[4]      │ -0.620107  │ -0.409242  │ -0.276722  │ -0.125668  │ 0.227167   │
│ 33  │ lκ[5]      │ -0.617892  │ -0.410158  │ -0.277921  │ -0.125055  │ 0.22954    │
│ 34  │ lκ[6]      │ -0.619824  │ -0.412358  │ -0.280068  │ -0.128671  │ 0.223733   │
│ 35  │ lκ[7]      │ -0.622541  │ -0.414653  │ -0.283007  │ -0.132787  │ 0.230045   │
│ 36  │ βᵣ₁[1]     │ -1.82538   │ -0.575792  │ 0.0621497  │ 0.699875   │ 1.91647    │
│ 37  │ βᵣ₁[2]     │ -1.77062   │ -0.541548  │ 0.0677598  │ 0.67477    │ 1.86221    │
│ 38  │ βᵣ₁[3]     │ -1.72842   │ -0.445737  │ 0.211593   │ 0.861842   │ 2.08981    │
│ 39  │ βᵣ₁[4]     │ -1.88485   │ -0.657307  │ -0.0241044 │ 0.612782   │ 1.83652    │
│ 40  │ βᵣ₁[5]     │ -2.00793   │ -0.783445  │ -0.130513  │ 0.521079   │ 1.77937    │
│ 41  │ βᵣ₁[6]     │ -1.85232   │ -0.631592  │ 0.00898927 │ 0.640062   │ 1.87949    │
│ 42  │ βᵣ₁[7]     │ -1.9986    │ -0.798667  │ -0.17071   │ 0.461791   │ 1.69269    │
│ 43  │ βᵣ₂[1]     │ -1.94157   │ -0.72838   │ -0.0971558 │ 0.544216   │ 1.77726    │
│ 44  │ βᵣ₂[2]     │ -1.61019   │ -0.350886  │ 0.273482   │ 0.888434   │ 2.06921    │
│ 45  │ βᵣ₂[3]     │ -2.00531   │ -0.750011  │ -0.0984606 │ 0.551833   │ 1.80298    │
│ 46  │ βᵣ₂[4]     │ -2.16789   │ -0.947279  │ -0.303615  │ 0.353141   │ 1.64666    │
│ 47  │ βᵣ₂[5]     │ -1.93604   │ -0.701198  │ -0.0668359 │ 0.575107   │ 1.79749    │
│ 48  │ βᵣ₂[6]     │ -1.7627    │ -0.508518  │ 0.125132   │ 0.769119   │ 1.99651    │
│ 49  │ βᵣ₂[7]     │ -1.71043   │ -0.467592  │ 0.159729   │ 0.776649   │ 1.97021    │
│ 50  │ θ[1]       │ -17.0052   │ -16.9261   │ -16.8845   │ -16.8432   │ -16.763    │
│ 51  │ θ[2]       │ -17.0292   │ -16.9496   │ -16.9093   │ -16.8701   │ -16.7958   │
│ 52  │ θ[3]       │ 18.3521    │ 18.4673    │ 18.5267    │ 18.5859    │ 18.7002    │
│ 53  │ θ[4]       │ -13.404    │ -13.2907   │ -13.2314   │ -13.1709   │ -13.05     │
│ 54  │ θ[5]       │ 7.4071     │ 7.52486    │ 7.58738    │ 7.64901    │ 7.7683     │
│ 55  │ θ[6]       │ 6.36903    │ 6.48785    │ 6.54838    │ 6.60957    │ 6.73371    │
│ 56  │ θ[7]       │ -7.85735   │ -7.77269   │ -7.72873   │ -7.68396   │ -7.59731   │
│ 57  │ μᵣ₁[1]     │ -1.52717   │ -0.373803  │ 0.172822   │ 0.713931   │ 1.81012    │
│ 58  │ μᵣ₁[2]     │ -1.47745   │ -0.36289   │ 0.194057   │ 0.756396   │ 1.885      │
│ 59  │ μᵣ₁[3]     │ -2.08482   │ -0.942541  │ -0.370237  │ 0.205423   │ 1.38607    │
│ 60  │ μᵣ₂[1]     │ -1.43851   │ -0.304723  │ 0.255622   │ 0.811419   │ 1.88912    │
│ 61  │ μᵣ₂[2]     │ -2.17723   │ -1.05023   │ -0.479178  │ 0.104108   │ 1.27506    │
│ 62  │ μᵣ₂[3]     │ -1.47551   │ -0.31836   │ 0.265287   │ 0.850042   │ 1.97709    │
│ 63  │ μₕ₁        │ -0.177654  │ -0.0794475 │ -0.0389967 │ 0.00198245 │ 0.0998822  │
│ 64  │ μₕ₂        │ -0.118697  │ 0.0188265  │ 0.0611488  │ 0.101748   │ 0.196528   │
│ 65  │ ρ[1]       │ 0.677434   │ 0.696922   │ 0.706854   │ 0.716505   │ 0.734138   │
│ 66  │ ρ[2]       │ 0.699604   │ 0.712024   │ 0.71849    │ 0.724841   │ 0.73726    │
│ 67  │ ρ[3]       │ 0.716067   │ 0.726943   │ 0.732565   │ 0.738196   │ 0.749028   │
│ 68  │ ρ[4]       │ 0.71672    │ 0.72695    │ 0.732388   │ 0.737825   │ 0.747907   │
│ 69  │ ρ[5]       │ 0.70148    │ 0.711929   │ 0.717423   │ 0.722884   │ 0.733262   │
│ 70  │ ρ[6]       │ 0.703164   │ 0.728957   │ 0.741013   │ 0.75223    │ 0.771925   │
│ 71  │ ρ[7]       │ 0.622075   │ 0.663104   │ 0.680776   │ 0.696746   │ 0.723807   │
│ 72  │ σ[1]       │ 1.54225    │ 1.5767     │ 1.59536    │ 1.61432    │ 1.65175    │
│ 73  │ σ[2]       │ 1.39766    │ 1.42187    │ 1.43495    │ 1.44837    │ 1.47444    │
│ 74  │ σ[3]       │ 2.39934    │ 2.44083    │ 2.46304    │ 2.48592    │ 2.53119    │
│ 75  │ σ[4]       │ 2.52152    │ 2.56465    │ 2.58821    │ 2.61261    │ 2.66064    │
│ 76  │ σ[5]       │ 2.69669    │ 2.73698    │ 2.75861    │ 2.78092    │ 2.8242     │
│ 77  │ σ[6]       │ 2.43152    │ 2.48367    │ 2.51194    │ 2.54083    │ 2.59748    │
│ 78  │ σ[7]       │ 1.5366     │ 1.57114    │ 1.58978    │ 1.60881    │ 1.64632    │
│ 79  │ σᵦ         │ 0.00147507 │ 0.0149294  │ 0.0316496  │ 0.0546134  │ 0.119273   │
│ 80  │ σₕ         │ 0.00281858 │ 0.0269499  │ 0.0544141  │ 0.0949492  │ 0.271869   │
```
If you don't pass the sampler a tuner object, it'll create one by default with `M=5` and `term=50`. The section on [Automatic Parameter Tuning](https://mc-stan.org/docs/2_19/reference-manual/hmc-algorithm-parameters.html) in the Stan reference manual explains what these mean. I added an extra slow adaptation step (with twice the length of the previous step) and doubled the length of the terminal adaptation window (a final window calculating step size), as this makes adaptation much more consistent. With default settings on this model, there would usually be 1 or more chains that ended up with smaller step sizes and higher average treedepths, resulting in much slower runtimes (and much higher average acceptance probability than the target of 0.8).


*Overview of how the library works.*

The `ITPModel` struct (or whatever you've named your model, with the first argument to the macro) is defined as a struct with a field for each unknown. Each field is parametrically typed.

`logdensity` and `constrain` are defined as generated functions, so that they can compile appropriate code given these parameteric types. The `@model` macro does a little preprocessing of the expression; most of the code generation occurs within these generated functions.

The macro's preprocessing consists of simply translating the sampling statements into log probability increments (`target`, terminology taken from the [Stan](https://mc-stan.org/users/documentation/) language), and lowering the expression.

We can manually perform these passes on the expression (note that variables with `#` in their names are NOT comments -- they're hygienic names, guaranteeing all the variables I add don't clash with any of the model's variables):
```julia
itp_q = quote
    # Non-hierarchical Priors
    ρ ~ Beta(3, 1)
    κ ~ Gamma(0.1, 0.1) # μ = 1, σ² = 10
    σ ~ Gamma(1.5, 0.25) # μ = 6, σ² = 2.4
    θ ~ Normal(10)
    L ~ LKJ(2.0)

    # Hierarchical Priors.
    # h subscript, for highest in the hierarhcy.
    μₕ₁ ~ Normal(10) # μ = 0
    μₕ₂ ~ Normal(10) # μ = 0
    σₕ ~ Normal(10) # μ = 0
    # Raw μs; non-cenetered parameterization
    μᵣ₁ ~ Normal() # μ = 0, σ = 1
    μᵣ₂ ~ Normal() # μ = 0, σ = 1
    # Center the μs
    μᵦ₁ = HierarchicalCentering(μᵣ₁, μₕ₁, σₕ)
    μᵦ₂ = HierarchicalCentering(μᵣ₂, μₕ₂, σₕ)
    σᵦ ~ Normal(10) # μ = 0
    # Raw βs; non-cenetered parameterization
    βᵣ₁ ~ Normal()
    βᵣ₂ ~ Normal()
    # Center the βs.
    β₁ = HierarchicalCentering(βᵣ₁, μᵦ₁, σᵦ, domains)
    β₂ = HierarchicalCentering(βᵣ₂, μᵦ₂, σᵦ, domains)

    # Likelihood
    μ₁ = vec(ITPExpectedValue(time, β₁, κ, θ))
    μ₂ = vec(ITPExpectedValue(time, β₂, κ, θ))
    Σ = CovarianceMatrix(ρ, Diagonal(σ) * L, time)

    (Y₁, Y₂) ~ Normal((μ₁, μ₂)[AvailableData], Σ[AvailableData])

end

itp_preprocessed = itp_q |> ProbabilityModels.translate_sampling_statements |> ProbabilityModels.flatten_expression;

using MacroTools: striplines

striplines(itp_preprocessed)
```
This yields the following expression:

```julia
quote
    ##SSAValue##1## = Beta(ρ, 3, 1)
    ##SSAValue##2## = vadd(target, ##SSAValue##1##)
    target = ##SSAValue##2##
    ##SSAValue##4## = Gamma(κ, 0.1, 0.1)
    ##SSAValue##5## = vadd(target, ##SSAValue##4##)
    target = ##SSAValue##5##
    ##SSAValue##7## = Gamma(σ, 1.5, 0.25)
    ##SSAValue##8## = vadd(target, ##SSAValue##7##)
    target = ##SSAValue##8##
    ##SSAValue##10## = Normal(θ, 10)
    ##SSAValue##11## = vadd(target, ##SSAValue##10##)
    target = ##SSAValue##11##
    ##SSAValue##13## = LKJ(L, 2.0)
    ##SSAValue##14## = vadd(target, ##SSAValue##13##)
    target = ##SSAValue##14##
    ##SSAValue##16## = Normal(μₕ₁, 10)
    ##SSAValue##17## = vadd(target, ##SSAValue##16##)
    target = ##SSAValue##17##
    ##SSAValue##19## = Normal(μₕ₂, 10)
    ##SSAValue##20## = vadd(target, ##SSAValue##19##)
    target = ##SSAValue##20##
    ##SSAValue##22## = Normal(σₕ, 10)
    ##SSAValue##23## = vadd(target, ##SSAValue##22##)
    target = ##SSAValue##23##
    ##SSAValue##25## = Normal(μᵣ₁)
    ##SSAValue##26## = vadd(target, ##SSAValue##25##)
    target = ##SSAValue##26##
    ##SSAValue##28## = Normal(μᵣ₂)
    ##SSAValue##29## = vadd(target, ##SSAValue##28##)
    target = ##SSAValue##29##
    ##SSAValue##31## = HierarchicalCentering(μᵣ₁, μₕ₁, σₕ)
    μᵦ₁ = ##SSAValue##31##
    ##SSAValue##33## = HierarchicalCentering(μᵣ₂, μₕ₂, σₕ)
    μᵦ₂ = ##SSAValue##33##
    ##SSAValue##35## = Normal(σᵦ, 10)
    ##SSAValue##36## = vadd(target, ##SSAValue##35##)
    target = ##SSAValue##36##
    ##SSAValue##38## = Normal(βᵣ₁)
    ##SSAValue##39## = vadd(target, ##SSAValue##38##)
    target = ##SSAValue##39##
    ##SSAValue##41## = Normal(βᵣ₂)
    ##SSAValue##42## = vadd(target, ##SSAValue##41##)
    target = ##SSAValue##42##
    ##SSAValue##44## = HierarchicalCentering(βᵣ₁, μᵦ₁, σᵦ, domains)
    β₁ = ##SSAValue##44##
    ##SSAValue##46## = HierarchicalCentering(βᵣ₂, μᵦ₂, σᵦ, domains)
    β₂ = ##SSAValue##46##
    ##SSAValue##48## = ITPExpectedValue(time, β₁, κ, θ)
    ##SSAValue##49## = vec(##SSAValue##48##)
    μ₁ = ##SSAValue##49##
    ##SSAValue##51## = ITPExpectedValue(time, β₂, κ, θ)
    ##SSAValue##52## = vec(##SSAValue##51##)
    μ₂ = ##SSAValue##52##
    ##SSAValue##54## = Diagonal(σ)
    ##SSAValue##55## = ##SSAValue##54## * L
    ##SSAValue##56## = CovarianceMatrix(ρ, ##SSAValue##55##, time)
    Σ = ##SSAValue##56##
    ##SSAValue##58## = Core.tuple(Y₁, Y₂)
    ##SSAValue##59## = Core.tuple(μ₁, μ₂)
    ##SSAValue##60## = Base.getindex(##SSAValue##59##, AvailableData)
    ##SSAValue##61## = Base.getindex(Σ, AvailableData)
    ##SSAValue##62## = Normal(##SSAValue##58##, ##SSAValue##60##, ##SSAValue##61##)
    ##SSAValue##63## = vadd(target, ##SSAValue##62##)
    target = ##SSAValue##63##
end
```
After translating the sampling statements into function calls, the code is transformed with `Meta.lower`. The resulting `Code.Info` is then transformed back into a Julia expression, as you see above. This has the advantage of flattening the expression, so that there is at most only a single function call per line.

The `logdensity` generated functions are defined with the above expressions assigned to a variable, so that they can apply additional transformations.


I'll focus on the `ValueGradient` method, as this one performs the more interesting transformations. Aside from loading and constraining all the parameters from an input vector, it performs a `reverse_diff!` pass on the expression.
```julia
forward_pass = quote end
reverse_pass = quote end
tracked_vars = Set([:ρ, :κ, :σ, :θ, :L, :μₕ₁, :μₕ₂, :σₕ, :μᵣ₁, :μᵣ₂, :σᵦ, :βᵣ₁, :βᵣ₂, ]);
ProbabilityModels.reverse_diff_pass!(forward_pass, reverse_pass, itp_preprocessed, tracked_vars);
striplines(forward_pass)
```
It walks the expression, replacing each function with a function that also returns the adjoint; the forward pass becomes:
```julia
quote
    (##SSAValue##1##, ###adjoint###_##∂##SSAValue##1####∂ρ##) = ProbabilityDistributions.∂Beta(ρ, 3, 1, Val{(true, false, false)}())
    ##SSAValue##2## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##1##)
    target = ##SSAValue##2##
    (##SSAValue##4##, ###adjoint###_##∂##SSAValue##4####∂κ##) = ProbabilityDistributions.∂Gamma(κ, 0.1, 0.1, Val{(true, false, false)}())
    ##SSAValue##5## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##4##)
    target = ##SSAValue##5##
    (##SSAValue##7##, ###adjoint###_##∂##SSAValue##7####∂σ##) = ProbabilityDistributions.∂Gamma(σ, 1.5, 0.25, Val{(true, false, false)}())
    ##SSAValue##8## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##7##)
    target = ##SSAValue##8##
    (##SSAValue##10##, ###adjoint###_##∂##SSAValue##10####∂θ##) = ProbabilityDistributions.∂Normal(θ, 10, Val{(true, false)}())
    ##SSAValue##11## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##10##)
    target = ##SSAValue##11##
    (##SSAValue##13##, ###adjoint###_##∂##SSAValue##13####∂L##) = ProbabilityDistributions.∂LKJ(L, 2.0, Val{(true, false)}())
    ##SSAValue##14## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##13##)
    target = ##SSAValue##14##
    (##SSAValue##16##, ###adjoint###_##∂##SSAValue##16####∂μₕ₁##) = ProbabilityDistributions.∂Normal(μₕ₁, 10, Val{(true, false)}())
    ##SSAValue##17## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##16##)
    target = ##SSAValue##17##
    (##SSAValue##19##, ###adjoint###_##∂##SSAValue##19####∂μₕ₂##) = ProbabilityDistributions.∂Normal(μₕ₂, 10, Val{(true, false)}())
    ##SSAValue##20## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##19##)
    target = ##SSAValue##20##
    (##SSAValue##22##, ###adjoint###_##∂##SSAValue##22####∂σₕ##) = ProbabilityDistributions.∂Normal(σₕ, 10, Val{(true, false)}())
    ##SSAValue##23## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##22##)
    target = ##SSAValue##23##
    (##SSAValue##25##, ###adjoint###_##∂##SSAValue##25####∂μᵣ₁##) = ProbabilityDistributions.∂Normal(μᵣ₁, Val{(true,)}())
    ##SSAValue##26## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##25##)
    target = ##SSAValue##26##
    (##SSAValue##28##, ###adjoint###_##∂##SSAValue##28####∂μᵣ₂##) = ProbabilityDistributions.∂Normal(μᵣ₂, Val{(true,)}())
    ##SSAValue##29## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##28##)
    target = ##SSAValue##29##
    (##SSAValue##31##, ###adjoint###_##∂##SSAValue##31####∂μᵣ₁##, ###adjoint###_##∂##SSAValue##31####∂μₕ₁##, ###adjoint###_##∂##SSAValue##31####∂σₕ##) = ∂HierarchicalCentering(μᵣ₁, μₕ₁, σₕ, Val{(true, true, true)}())
    μᵦ₁ = ##SSAValue##31##
    (##SSAValue##33##, ###adjoint###_##∂##SSAValue##33####∂μᵣ₂##, ###adjoint###_##∂##SSAValue##33####∂μₕ₂##, ###adjoint###_##∂##SSAValue##33####∂σₕ##) = ∂HierarchicalCentering(μᵣ₂, μₕ₂, σₕ, Val{(true, true, true)}())
    μᵦ₂ = ##SSAValue##33##
    (##SSAValue##35##, ###adjoint###_##∂##SSAValue##35####∂σᵦ##) = ProbabilityDistributions.∂Normal(σᵦ, 10, Val{(true, false)}())
    ##SSAValue##36## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##35##)
    target = ##SSAValue##36##
    (##SSAValue##38##, ###adjoint###_##∂##SSAValue##38####∂βᵣ₁##) = ProbabilityDistributions.∂Normal(βᵣ₁, Val{(true,)}())
    ##SSAValue##39## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##38##)
    target = ##SSAValue##39##
    (##SSAValue##41##, ###adjoint###_##∂##SSAValue##41####∂βᵣ₂##) = ProbabilityDistributions.∂Normal(βᵣ₂, Val{(true,)}())
    ##SSAValue##42## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##41##)
    target = ##SSAValue##42##
    (##SSAValue##44##, ###adjoint###_##∂##SSAValue##44####βᵣ₁##, ###adjoint###_##∂##SSAValue##44####∂μᵦ₁##, ###adjoint###_##∂##SSAValue##44####∂σᵦ##) = ∂HierarchicalCentering(βᵣ₁, μᵦ₁, σᵦ, domains, Val{(true, true, true)}())
    β₁ = ##SSAValue##44##
    (##SSAValue##46##, ###adjoint###_##∂##SSAValue##46####∂βᵣ₂##, ###adjoint###_##∂##SSAValue##46####∂μᵦ₂##, ###adjoint###_##∂##SSAValue##46####∂σᵦ##) = ∂HierarchicalCentering(βᵣ₂, μᵦ₂, σᵦ, domains, Val{(true, true, true)}())
    β₂ = ##SSAValue##46##
    (##SSAValue##48##, ###adjoint###_##∂##SSAValue##48####∂β₁##, ###adjoint###_##∂##SSAValue##48####∂κ##, ###adjoint###_##∂##SSAValue##48####∂θ##) = ProbabilityModels.∂ITPExpectedValue(time, β₁, κ, θ, Val{(true, true, true)}())
    (##SSAValue##49##, ###adjoint###_##∂##SSAValue##49####∂##SSAValue##48####) = ProbabilityModels.∂vec(##SSAValue##48##)
    μ₁ = ##SSAValue##49##
    (##SSAValue##51##, ###adjoint###_##∂##SSAValue##51####∂β₂##, ###adjoint###_##∂##SSAValue##51####∂κ##, ###adjoint###_##∂##SSAValue##51####∂θ##) = ProbabilityModels.∂ITPExpectedValue(time, β₂, κ, θ, Val{(true, true, true)}())
    (##SSAValue##52##, ###adjoint###_##∂##SSAValue##52####∂##SSAValue##51####) = ProbabilityModels.∂vec(##SSAValue##51##)
    μ₂ = ##SSAValue##52##
    ##SSAValue##54## = LinearAlgebra.Diagonal(σ)
    ##SSAValue##55## = ##SSAValue##54## * L
    (##SSAValue##56##, ###adjoint###_##∂##SSAValue##56####∂ρ##, ###adjoint###_##∂##SSAValue##56####∂##SSAValue##55####) = ProbabilityModels.DistributionParameters.∂CovarianceMatrix(ρ, ##SSAValue##55##, time, Val{(true, true)}())
    Σ = ##SSAValue##56##
    ##SSAValue##58## = Core.tuple(Y₁, Y₂)
    ##SSAValue##59## = Core.tuple(μ₁, μ₂)
    (##SSAValue##60##, ###adjoint###_##∂##SSAValue##60####∂##SSAValue##59####) = PaddedMatrices.∂getindex(##SSAValue##59##, AvailableData)
    (##SSAValue##61##, ###adjoint###_##∂##SSAValue##61####∂Σ##) = PaddedMatrices.∂getindex(Σ, AvailableData)
    (##SSAValue##62##, ###adjoint###_##∂##SSAValue##62####∂##SSAValue##60####, ###adjoint###_##∂##SSAValue##62####∂##SSAValue##61####) = ProbabilityDistributions.∂Normal(##SSAValue##58##, ##SSAValue##60##, ##SSAValue##61##, Val{(false, true, true)}())
    ##SSAValue##63## = ProbabilityModels.SIMDPirates.vadd(target, ##SSAValue##62##)
    target = ##SSAValue##63##
end
```
It also prepends the corresponding operations to a second expression for the reverse diff pass:
```julia
striplines(reverse_pass)
```
yielding:
```julia
quote
    ###seed#####SSAValue##63## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##63##)
    ###seed#####SSAValue##62## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##63##, ###seed#####SSAValue##62##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##63##, ###seed###target)
    ###seed#####SSAValue##61## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##62##, ###adjoint###_##∂##SSAValue##62####∂##SSAValue##61####, ###seed#####SSAValue##61##)
    ###seed#####SSAValue##60## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##62##, ###adjoint###_##∂##SSAValue##62####∂##SSAValue##60####, ###seed#####SSAValue##60##)
    ###seed###Σ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##61##, ###adjoint###_##∂##SSAValue##61####∂Σ##, ###seed###Σ)
    ###seed#####SSAValue##59## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##60##, ###adjoint###_##∂##SSAValue##60####∂##SSAValue##59####, ###seed#####SSAValue##59##)
    ###seed###μ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##59##[2], ###seed###μ₂)
    ###seed###μ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##59##[1], ###seed###μ₁)
    ###seed#####SSAValue##56## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###Σ, ###seed#####SSAValue##56##)
    ###seed#####SSAValue##55## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##56##, ###adjoint###_##∂##SSAValue##56####∂##SSAValue##55####, ###seed#####SSAValue##55##)
    ###seed###ρ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##56##, ###adjoint###_##∂##SSAValue##56####∂ρ##, ###seed###ρ)
    (###adjoint###_##∂##SSAValue##55####∂##SSAValue##54####, ###adjoint###_##∂##SSAValue##55####∂L##) = ProbabilityModels.∂mul(##SSAValue##54##, L, Val{(true, true)}())
    ###seed###L = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##55##, ###adjoint###_##∂##SSAValue##55####∂L##, ###seed###L)
    ###seed#####SSAValue##54## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##55##, ###adjoint###_##∂##SSAValue##55####∂##SSAValue##54####, ###seed#####SSAValue##54##)
    ###seed###σ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##54##, ###seed###σ)
    ###seed#####SSAValue##52## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###μ₂, ###seed#####SSAValue##52##)
    ###seed#####SSAValue##51## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##52##, ###adjoint###_##∂##SSAValue##52####∂##SSAValue##51####, ###seed#####SSAValue##51##)
    ###seed###θ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##51##, ###adjoint###_##∂##SSAValue##51####∂θ##, ###seed###θ)
    ###seed###κ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##51##, ###adjoint###_##∂##SSAValue##51####∂κ##, ###seed###κ)
    ###seed###β₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##51##, ###adjoint###_##∂##SSAValue##51####∂β₂##, ###seed###β₂)
    ###seed#####SSAValue##49## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###μ₁, ###seed#####SSAValue##49##)
    ###seed#####SSAValue##48## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##49##, ###adjoint###_##∂##SSAValue##49####∂##SSAValue##48####, ###seed#####SSAValue##48##)
    ###seed###θ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##48##, ###adjoint###_##∂##SSAValue##48####∂θ##, ###seed###θ)
    ###seed###κ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##48##, ###adjoint###_##∂##SSAValue##48####∂κ##, ###seed###κ)
    ###seed###β₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##48##, ###adjoint###_##∂##SSAValue##48####∂β₁##, ###seed###β₁)
    ###seed#####SSAValue##46## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###β₂, ###seed#####SSAValue##46##)
    ###seed###σᵦ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##46##, ###adjoint###_##∂##SSAValue##46####∂σᵦ##, ###seed###σᵦ)
    ###seed###μᵦ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##46##, ###adjoint###_##∂##SSAValue##46####∂μᵦ₂##, ###seed###μᵦ₂)
    ###seed###βᵣ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##46##, ###adjoint###_##∂##SSAValue##46####∂βᵣ₂##, ###seed###βᵣ₂)
    ###seed#####SSAValue##44## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###β₁, ###seed#####SSAValue##44##)
    ###seed###σᵦ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##44##, ###adjoint###_##∂##SSAValue##44####∂σᵦ##, ###seed###σᵦ)
    ###seed###μᵦ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##44##, ###adjoint###_##∂##SSAValue##44####∂μᵦ₁##, ###seed###μᵦ₁)
    ###seed###βᵣ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##44##, ###adjoint###_##∂##SSAValue##44####∂βᵣ₁##, ###seed###βᵣ₁)
    ###seed#####SSAValue##42## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##42##)
    ###seed#####SSAValue##41## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##42##, ###seed#####SSAValue##41##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##42##, ###seed###target)
    ###seed###βᵣ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##41##, ###adjoint###_##∂##SSAValue##41####∂βᵣ₂##, ###seed###βᵣ₂)
    ###seed#####SSAValue##39## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##39##)
    ###seed#####SSAValue##38## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##39##, ###seed#####SSAValue##38##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##39##, ###seed###target)
    ###seed###βᵣ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##38##, ###adjoint###_##∂##SSAValue##38####∂βᵣ₁##, ###seed###βᵣ₁)
    ###seed#####SSAValue##36## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##36##)
    ###seed#####SSAValue##35## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##36##, ###seed#####SSAValue##35##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##36##, ###seed###target)
    ###seed###σᵦ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##35##, ###adjoint###_##∂##SSAValue##35####∂σᵦ##, ###seed###σᵦ)
    ###seed#####SSAValue##33## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###μᵦ₂, ###seed#####SSAValue##33##)
    ###seed###σₕ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##33##, ###adjoint###_##∂##SSAValue##33####∂σₕ##, ###seed###σₕ)
    ###seed###μₕ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##33##, ###adjoint###_##∂##SSAValue##33####∂μₕ₂##, ###seed###μₕ)
    ###seed###μᵣ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##33##, ###adjoint###_##∂##SSAValue##33####∂μᵣ₂##, ###seed###μᵣ₂)
    ###seed#####SSAValue##31## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###μᵦ₁, ###seed#####SSAValue##31##)
    ###seed###σₕ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##31##, ###adjoint###_##∂##SSAValue##31####∂σₕ##, ###seed###σₕ)
    ###seed###μₕ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##31##, ###adjoint###_##∂##SSAValue##31####∂μₕ₁##, ###seed###μₕ₁)
    ###seed###μᵣ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##31##, ###adjoint###_##∂##SSAValue##31####∂μᵣ₁##, ###seed###μᵣ₁)
    ###seed#####SSAValue##29## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##29##)
    ###seed#####SSAValue##28## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##29##, ###seed#####SSAValue##28##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##29##, ###seed###target)
    ###seed###μᵣ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##28##, ###adjoint###_##∂##SSAValue##28####∂μᵣ₂##, ###seed###μᵣ₂)
    ###seed#####SSAValue##26## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##26##)
    ###seed#####SSAValue##25## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##26##, ###seed#####SSAValue##25##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##26##, ###seed###target)
    ###seed###μᵣ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##25##, ###adjoint###_##∂##SSAValue##25####∂μᵣ₁##, ###seed###μᵣ₁)
    ###seed#####SSAValue##23## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##23##)
    ###seed#####SSAValue##22## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##23##, ###seed#####SSAValue##22##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##23##, ###seed###target)
    ###seed###σₕ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##22##, ###adjoint###_##∂##SSAValue##22####∂σₕ##, ###seed###σₕ)
    ###seed#####SSAValue##20## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##20##)
    ###seed#####SSAValue##19## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##20##, ###seed#####SSAValue##19##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##20##, ###seed###target)
    ###seed###μₕ₂ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##19##, ###adjoint###_##∂##SSAValue##19####∂μₕ₂##, ###seed###μₕ₂)
    ###seed#####SSAValue##17## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##17##)
    ###seed#####SSAValue##16## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##17##, ###seed#####SSAValue##16##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##17##, ###seed###target)
    ###seed###μₕ₁ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##16##, ###adjoint###_##∂##SSAValue##16####∂μₕ₁##, ###seed###μₕ₁)
    ###seed#####SSAValue##14## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##14##)
    ###seed#####SSAValue##13## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##14##, ###seed#####SSAValue##13##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##14##, ###seed###target)
    ###seed###L = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##13##, ###adjoint###_##∂##SSAValue##13####∂L##, ###seed###L)
    ###seed#####SSAValue##11## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##11##)
    ###seed#####SSAValue##10## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##11##, ###seed#####SSAValue##10##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##11##, ###seed###target)
    ###seed###θ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##10##, ###adjoint###_##∂##SSAValue##10####∂θ##, ###seed###θ)
    ###seed#####SSAValue##8## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##8##)
    ###seed#####SSAValue##7## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##8##, ###seed#####SSAValue##7##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##8##, ###seed###target)
    ###seed###σ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##7##, ###adjoint###_##∂##SSAValue##7####∂σ##, ###seed###σ)
    ###seed#####SSAValue##5## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##5##)
    ###seed#####SSAValue##4## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##5##, ###seed#####SSAValue##4##)
    ###seed###target = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##5##, ###seed###target)
    ###seed###κ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##4##, ###adjoint###_##∂##SSAValue##4####∂κ##, ###seed###κ)
    ###seed#####SSAValue##2## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed###target, ###seed#####SSAValue##2##)
    ###seed#####SSAValue##1## = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##2##, ###seed#####SSAValue##1##)
    ###seed###ρ = ProbabilityModels.PaddedMatrices.RESERVED_INCREMENT_SEED_RESERVED(###seed#####SSAValue##1##, ###adjoint###_##∂##SSAValue##1####∂ρ##, ###seed###ρ)
end
```

The `RESERVED_INCREMENT_SEED_RESERVED` and `RESERVED_MULTIPLY_SEED_RESERVED` fall back to `mulladd` and `*` methods, respectively. They however allow for different overloads.

The keys to performance are that this source-to-source reverse mode differentation avoids unnecessary overhead, and does not interfere with vectorization, which typical dual-number based approaches, like Stan's var type or ForwardDiff's duals, do.
The reverse diff pass uses [DiffRules.jl](https://github.com/JuliaDiff/DiffRules.jl) as well as custom derivatives defined for probability distributions and a few other special functions.

By defining high level derivatives for commonly used functions we can optimize them much further than if we relied on autodiff for them. Additionally, we abuse multiple dispatch here: whenevever there is some structure we can exploit in the adjoint, we return a type with `INCREMENT/MULTIPLY` methods defined to exploit it. Examples include [BlockDiagonalColumnView](https://github.com/chriselrod/StructuredMatrices.jl/blob/master/src/block_diagonal.jl#L4)s or various [Reducer](https://github.com/chriselrod/ProbabilityModels.jl/blob/master/src/adjoints.jl#L55) types.

Another optimization is that it performs a [stack pointer pass](https://github.com/chriselrod/PaddedMatrices.jl/blob/master/src/stack_pointer.jl#L55). Supported functions are passed a pointer to a preallocated block of memory. This memory reuse can both give us considerable savings, and allow the memory to stay hot in the cache. [PaddedMatrices.jl](https://github.com/chriselrod/PaddedMatrices.jl), aside from providing [optimized matrix operations](https://bayeswatch.org/2019/06/06/small-matrix-multiplication-performance-shootout/) (contrary to the name, it's performance advantage over other libraries is actually greatest when none of them use padding), it also provides `PtrArray` (parameterized by size) and `DynamicPtrArray` (dynamically sized) types for taking advantage of this preallocated memory.





