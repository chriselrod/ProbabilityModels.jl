

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

	# Tuple (Y₁, Y₂)
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
#    domain_means = randn(n_domains); domain_means .*= 10
    μ = MutableFixedSizePaddedVector{K,Float64}(undef)

    offset = 0
    for i ∈ domains
#        global offset
        domain_mean = 5randn()
        for j ∈ 1+offset:i+offset
            μ[j] = domain_mean + 5randn()
        end
        offset += i
    end

    σ = PaddedMatrices.randgamma( 6.0, 1/6.0)

    ρ = PaddedMatrices.randbeta(4.0,4.0)

#    β = [0.1i - 0.15 for j ∈ 1:K, i ∈ 1:2]
    κ = rinvscaledgamma(Val(K), κ₀...)
    lt = last(time)
#    β₁ = MutableFixedSizePaddedVector{7,Float64,7,7}((-0.0625, -0.0575, -0.0525, -0.0475, -0.0425, -0.04, -0.0375))
    β = MutableFixedSizePaddedVector{7,Float64,7,7}(( 0.0625,  0.0575,  0.0525,  0.0475,  0.0425,  0.04,  0.0375))
    θ₁ = @. μ' - β' * ( 1.0 - exp( - κ' * time) ) / (1.0 - exp( - κ' * lt) ) 
    θ₂ = @. μ' + β' * ( 1.0 - exp( - κ' * time) ) / (1.0 - exp( - κ' * lt) ) 
    
    L_T, info = LAPACK.potrf!('L', AR1(ρ, time))
    @inbounds for tc ∈ 2:T, tr ∈ 1:tc-1
        L_T[tr,tc] = 0.0
    end
#    K = 7
    X = PaddedMatrices.MutableFixedSizePaddedMatrix{K,K+3,Float64,K}(undef); randn!(X)
    U_K, info = LAPACK.potrf!('U', BLAS.syrk!('U', 'N', σ, X, 0.0, zero(MutableFixedSizePaddedMatrix{K,K,Float64,K})))
#U_K
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

@benchmark logdensity(Value, $data, $a)
#BenchmarkTools.Trial: 
#  memory estimate:  0 bytes
#  allocs estimate:  0
#  --------------
#  minimum time:     182.464 μs (0.00% GC)
#  median time:      183.679 μs (0.00% GC)
#  mean time:        184.068 μs (0.00% GC)
#  maximum time:     294.935 μs (0.00% GC)
#  --------------
#  samples:          10000
#  evals/sample:     1

@benchmark logdensity(ValueGradient, $data, $a)
#BenchmarkTools.Trial: 
#  memory estimate:  720 bytes
#  allocs estimate:  3
#  --------------
#  minimum time:     560.593 μs (0.00% GC)
#  median time:      564.958 μs (0.00% GC)
#  mean time:        564.800 μs (0.00% GC)
#  maximum time:     738.522 μs (0.00% GC)
#  --------------
#  samples:          8821
#  evals/sample:     1

```
For comparison, a Stan implementation of this model takes about to 13ms to evaluate the gradient.

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

julia> bdt = DynamicHMC.bracketed_doubling_tuner(M=6, term=100); #1750 warmup iterations

julia> @time chains, tuned_samplers = NUTS_init_tune_distributed(
	data, 10_000, δ = 0.95, tuners = bdt, report = DynamicHMC.ReportSilent()
);
      From worker 12:	616.452941 seconds (44.52 M allocations: 6.365 GiB, 0.14% gc time)
      From worker 6:	632.276204 seconds (46.10 M allocations: 6.590 GiB, 0.14% gc time)
      From worker 19:	638.336930 seconds (48.69 M allocations: 6.958 GiB, 0.14% gc time)
      From worker 9:	647.171824 seconds (46.38 M allocations: 6.629 GiB, 0.13% gc time)
      From worker 2:	648.198840 seconds (47.82 M allocations: 6.835 GiB, 0.13% gc time)
      From worker 13:	648.341581 seconds (47.56 M allocations: 6.798 GiB, 0.14% gc time)
      From worker 7:	650.312980 seconds (46.99 M allocations: 6.717 GiB, 0.13% gc time)
      From worker 16:	654.895122 seconds (48.63 M allocations: 6.950 GiB, 0.14% gc time)
      From worker 3:	655.439721 seconds (48.84 M allocations: 6.981 GiB, 0.14% gc time)
      From worker 17:	660.534757 seconds (49.25 M allocations: 7.038 GiB, 0.14% gc time)
      From worker 11:	660.886714 seconds (48.14 M allocations: 6.881 GiB, 0.14% gc time)
      From worker 8:	662.441475 seconds (48.75 M allocations: 6.968 GiB, 0.13% gc time)
      From worker 18:	663.003943 seconds (48.33 M allocations: 6.907 GiB, 0.13% gc time)
      From worker 4:	665.319696 seconds (49.24 M allocations: 7.037 GiB, 0.13% gc time)
      From worker 10:	666.521578 seconds (48.93 M allocations: 6.992 GiB, 0.13% gc time)
      From worker 5:	673.141813 seconds (49.56 M allocations: 7.082 GiB, 0.13% gc time)
      From worker 14:	673.909099 seconds (48.89 M allocations: 6.987 GiB, 0.13% gc time)
      From worker 15:	747.844131 seconds (58.22 M allocations: 8.317 GiB, 0.12% gc time)
775.446771 seconds (14.33 M allocations: 726.365 MiB, 0.04% gc time)

julia> mcmcchains = Chains(chains, data)
Object of type Chains, with data of type 10000×80×18 Array{Float64,3}

Iterations        = 1:10000
Thinning interval = 1
Chains            = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
Samples per chain = 10000
parameters        = μₕ₂, μᵣ₁[1], μᵣ₁[2], μᵣ₁[3], σ[1], σ[2], σ[3], σ[4], σ[5], σ[6], σ[7], σᵦ, θ[1], θ[2], θ[3], θ[4], θ[5], θ[6], θ[7], μᵣ₂[1], μᵣ₂[2], μᵣ₂[3], ρ[1], ρ[2], ρ[3], ρ[4], ρ[5], ρ[6], ρ[7], σₕ, lκ[1], lκ[2], lκ[3], lκ[4], lκ[5], lκ[6], lκ[7], μₕ₁, L[1,1], L[2,2], L[3,3], L[4,4], L[5,5], L[6,6], L[7,7], L[2,1], L[3,1], L[4,1], L[5,1], L[6,1], L[7,1], L[3,2], L[4,2], L[5,2], L[6,2], L[7,2], L[4,3], L[5,3], L[6,3], L[7,3], L[5,4], L[6,4], L[7,4], L[6,5], L[7,5], L[7,6], βᵣ₂[1], βᵣ₂[2], βᵣ₂[3], βᵣ₂[4], βᵣ₂[5], βᵣ₂[6], βᵣ₂[7], βᵣ₁[1], βᵣ₁[2], βᵣ₁[3], βᵣ₁[4], βᵣ₁[5], βᵣ₁[6], βᵣ₁[7]

2-element Array{ChainDataFrame,1}

Summary Statistics

│ Row │ parameters │ mean       │ std        │ naive_se    │ mcse        │ ess       │ r_hat    │
│     │ Symbol     │ Float64    │ Float64    │ Float64     │ Float64     │ Float64   │ Float64  │
├─────┼────────────┼────────────┼────────────┼─────────────┼─────────────┼───────────┼──────────┤
│ 1   │ L[1,1]     │ 1.0        │ 0.0        │ 0.0         │ 0.0         │ NaN       │ NaN      │
│ 2   │ L[2,1]     │ -0.115938  │ 0.0195264  │ 4.60241e-5  │ 4.57083e-5  │ 180000.0  │ 0.999989 │
│ 3   │ L[2,2]     │ 0.993062   │ 0.00229587 │ 5.41141e-6  │ 5.39473e-6  │ 180000.0  │ 0.999983 │
│ 4   │ L[3,1]     │ -0.27123   │ 0.0187073  │ 4.40935e-5  │ 4.58644e-5  │ 1.71842e5 │ 1.00004  │
│ 5   │ L[3,2]     │ 0.2615     │ 0.0119658  │ 2.82036e-5  │ 2.805e-5    │ 180000.0  │ 1.00006  │
│ 6   │ L[3,3]     │ 0.926027   │ 0.00568813 │ 1.3407e-5   │ 1.29154e-5  │ 180000.0  │ 1.0      │
│ 7   │ L[4,1]     │ 0.565306   │ 0.0156496  │ 3.68865e-5  │ 3.79895e-5  │ 1.69801e5 │ 1.0      │
│ 8   │ L[4,2]     │ 0.401966   │ 0.0133034  │ 3.13563e-5  │ 3.45174e-5  │ 1.55165e5 │ 1.00002  │
│ 9   │ L[4,3]     │ -0.329034  │ 0.0133777  │ 3.15316e-5  │ 3.2936e-5   │ 1.66864e5 │ 0.999997 │
│ 10  │ L[4,4]     │ 0.640228   │ 0.00977393 │ 2.30374e-5  │ 2.2124e-5   │ 180000.0  │ 0.999983 │
│ 11  │ L[5,1]     │ 0.161789   │ 0.0199415  │ 4.70025e-5  │ 4.95926e-5  │ 165386.0  │ 0.999999 │
│ 12  │ L[5,2]     │ 0.0703995  │ 0.0119619  │ 2.81945e-5  │ 2.63173e-5  │ 180000.0  │ 1.00001  │
│ 13  │ L[5,3]     │ -0.525613  │ 0.011429   │ 2.69384e-5  │ 2.80888e-5  │ 1.64913e5 │ 1.00003  │
│ 14  │ L[5,4]     │ -0.257883  │ 0.0147761  │ 3.48277e-5  │ 3.73344e-5  │ 180000.0  │ 0.999975 │
│ 15  │ L[5,5]     │ 0.79066    │ 0.00791756 │ 1.86619e-5  │ 1.86103e-5  │ 180000.0  │ 0.999999 │
│ 16  │ L[6,1]     │ -0.178175  │ 0.0228955  │ 5.39652e-5  │ 4.99162e-5  │ 180000.0  │ 0.999969 │
│ 17  │ L[6,2]     │ 0.141128   │ 0.0208749  │ 4.92027e-5  │ 4.12013e-5  │ 180000.0  │ 0.999974 │
│ 18  │ L[6,3]     │ -0.0620575 │ 0.0215209  │ 5.07252e-5  │ 4.37231e-5  │ 180000.0  │ 0.999991 │
│ 19  │ L[6,4]     │ 0.0723175  │ 0.0221755  │ 5.22681e-5  │ 5.22457e-5  │ 180000.0  │ 0.999973 │
│ 20  │ L[6,5]     │ -0.534875  │ 0.0172105  │ 4.05656e-5  │ 3.74379e-5  │ 180000.0  │ 0.999972 │
│ 21  │ L[6,6]     │ 0.806731   │ 0.0117655  │ 2.77316e-5  │ 2.64284e-5  │ 180000.0  │ 0.999958 │
│ 22  │ L[7,1]     │ -0.377892  │ 0.0202325  │ 4.76885e-5  │ 4.25962e-5  │ 180000.0  │ 0.999963 │
│ 23  │ L[7,2]     │ 0.0825581  │ 0.0195844  │ 4.6161e-5   │ 4.2831e-5   │ 180000.0  │ 0.999999 │
│ 24  │ L[7,3]     │ -0.0937304 │ 0.0202352  │ 4.76949e-5  │ 4.44491e-5  │ 180000.0  │ 0.999985 │
│ 25  │ L[7,4]     │ 0.583477   │ 0.0144426  │ 3.40414e-5  │ 3.23891e-5  │ 180000.0  │ 1.00002  │
│ 26  │ L[7,5]     │ -0.176123  │ 0.0174216  │ 4.10631e-5  │ 3.63946e-5  │ 180000.0  │ 0.999951 │
│ 27  │ L[7,6]     │ -0.308631  │ 0.0156004  │ 3.67704e-5  │ 3.2472e-5   │ 180000.0  │ 0.999959 │
│ 28  │ L[7,7]     │ 0.61055    │ 0.012209   │ 2.8777e-5   │ 2.79579e-5  │ 180000.0  │ 0.999978 │
│ 29  │ lκ[1]      │ -0.262195  │ 0.216524   │ 0.000510352 │ 0.000485028 │ 180000.0  │ 0.999977 │
│ 30  │ lκ[2]      │ -0.246132  │ 0.217688   │ 0.000513095 │ 0.000500489 │ 180000.0  │ 1.00001  │
│ 31  │ lκ[3]      │ -0.259238  │ 0.217342   │ 0.00051228  │ 0.000510908 │ 180000.0  │ 1.00001  │
│ 32  │ lκ[4]      │ -0.254388  │ 0.215822   │ 0.000508698 │ 0.000464114 │ 180000.0  │ 1.00003  │
│ 33  │ lκ[5]      │ -0.258889  │ 0.217209   │ 0.000511967 │ 0.000546274 │ 180000.0  │ 1.00009  │
│ 34  │ lκ[6]      │ -0.25887   │ 0.215916   │ 0.000508919 │ 0.000522956 │ 180000.0  │ 0.99999  │
│ 35  │ lκ[7]      │ -0.260416  │ 0.216423   │ 0.000510114 │ 0.000525937 │ 180000.0  │ 0.999993 │
│ 36  │ βᵣ₁[1]     │ 0.00932526 │ 0.935623   │ 0.00220528  │ 0.00180003  │ 180000.0  │ 0.999947 │
│ 37  │ βᵣ₁[2]     │ -0.235581  │ 0.906371   │ 0.00213634  │ 0.00177767  │ 180000.0  │ 0.999981 │
│ 38  │ βᵣ₁[3]     │ 0.347066   │ 0.963142   │ 0.00227015  │ 0.0020875   │ 180000.0  │ 0.99999  │
│ 39  │ βᵣ₁[4]     │ -0.37658   │ 0.867431   │ 0.00204455  │ 0.00206514  │ 180000.0  │ 1.00002  │
│ 40  │ βᵣ₁[5]     │ 0.0477585  │ 0.930608   │ 0.00219346  │ 0.00185058  │ 180000.0  │ 0.999988 │
│ 41  │ βᵣ₁[6]     │ 0.244065   │ 0.94569    │ 0.00222901  │ 0.00204742  │ 180000.0  │ 0.999963 │
│ 42  │ βᵣ₁[7]     │ -0.031954  │ 0.952811   │ 0.0022458   │ 0.00174247  │ 180000.0  │ 0.999966 │
│ 43  │ βᵣ₂[1]     │ -0.241548  │ 0.938327   │ 0.00221166  │ 0.00191533  │ 180000.0  │ 0.999959 │
│ 44  │ βᵣ₂[2]     │ 0.426452   │ 0.926071   │ 0.00218277  │ 0.00211693  │ 180000.0  │ 1.00003  │
│ 45  │ βᵣ₂[3]     │ -0.235934  │ 0.951552   │ 0.00224283  │ 0.0017308   │ 180000.0  │ 0.99994  │
│ 46  │ βᵣ₂[4]     │ 0.280219   │ 0.860124   │ 0.00202733  │ 0.00183518  │ 180000.0  │ 0.999998 │
│ 47  │ βᵣ₂[5]     │ -0.279188  │ 0.935707   │ 0.00220548  │ 0.00196274  │ 180000.0  │ 0.999998 │
│ 48  │ βᵣ₂[6]     │ -0.110335  │ 0.938677   │ 0.00221248  │ 0.00193494  │ 180000.0  │ 0.99999  │
│ 49  │ βᵣ₂[7]     │ 0.157454   │ 0.958889   │ 0.00226012  │ 0.00195251  │ 180000.0  │ 0.99999  │
│ 50  │ θ[1]       │ 0.851964   │ 0.0765559  │ 0.000180444 │ 0.000202653 │ 1.44417e5 │ 1.00004  │
│ 51  │ θ[2]       │ -6.73726   │ 0.0665956  │ 0.000156967 │ 0.000195585 │ 1.19691e5 │ 1.00002  │
│ 52  │ θ[3]       │ -3.09027   │ 0.0774789  │ 0.000182619 │ 0.000185792 │ 180000.0  │ 0.999977 │
│ 53  │ θ[4]       │ -2.7242    │ 0.0522861  │ 0.00012324  │ 0.000146456 │ 1.26362e5 │ 1.00003  │
│ 54  │ θ[5]       │ 2.51497    │ 0.056447   │ 0.000133047 │ 0.000126323 │ 180000.0  │ 0.999942 │
│ 55  │ θ[6]       │ 3.48235    │ 0.0630903  │ 0.000148705 │ 0.000163857 │ 180000.0  │ 1.00002  │
│ 56  │ θ[7]       │ 4.44587    │ 0.0835761  │ 0.000196991 │ 0.000184001 │ 180000.0  │ 1.0      │
│ 57  │ μᵣ₁[1]     │ -0.164277  │ 0.901006   │ 0.00212369  │ 0.00186515  │ 180000.0  │ 1.0      │
│ 58  │ μᵣ₁[2]     │ -0.0544623 │ 0.874343   │ 0.00206085  │ 0.0018835   │ 180000.0  │ 0.999979 │
│ 59  │ μᵣ₁[3]     │ 0.224198   │ 0.922731   │ 0.0021749   │ 0.00203044  │ 180000.0  │ 0.999986 │
│ 60  │ μᵣ₂[1]     │ 0.15136    │ 0.897604   │ 0.00211567  │ 0.00186802  │ 180000.0  │ 0.999947 │
│ 61  │ μᵣ₂[2]     │ 0.0591124  │ 0.876562   │ 0.00206608  │ 0.00234211  │ 180000.0  │ 1.00004  │
│ 62  │ μᵣ₂[3]     │ -0.203772  │ 0.925122   │ 0.00218053  │ 0.00223444  │ 180000.0  │ 1.00001  │
│ 63  │ μₕ₁        │ -0.0969468 │ 0.0547423  │ 0.000129029 │ 0.000219364 │ 62322.7   │ 1.00024  │
│ 64  │ μₕ₂        │ 0.00677511 │ 0.0557182  │ 0.000131329 │ 0.00029395  │ 29480.7   │ 1.00037  │
│ 65  │ ρ[1]       │ 0.688789   │ 0.013261   │ 3.12565e-5  │ 2.93287e-5  │ 180000.0  │ 0.999986 │
│ 66  │ ρ[2]       │ 0.692458   │ 0.00849146 │ 2.00146e-5  │ 2.02137e-5  │ 180000.0  │ 1.0      │
│ 67  │ ρ[3]       │ 0.71418    │ 0.00847575 │ 1.99775e-5  │ 2.19001e-5  │ 1.59493e5 │ 1.00004  │
│ 68  │ ρ[4]       │ 0.698985   │ 0.0112199  │ 2.64456e-5  │ 2.44388e-5  │ 180000.0  │ 0.999994 │
│ 69  │ ρ[5]       │ 0.69353    │ 0.00873415 │ 2.05866e-5  │ 2.07133e-5  │ 180000.0  │ 1.0      │
│ 70  │ ρ[6]       │ 0.683968   │ 0.0246915  │ 5.81984e-5  │ 5.0361e-5   │ 180000.0  │ 1.00002  │
│ 71  │ ρ[7]       │ 0.693543   │ 0.0233371  │ 5.5006e-5   │ 4.86152e-5  │ 180000.0  │ 0.999986 │
│ 72  │ σ[1]       │ 2.17267    │ 0.036634   │ 8.63471e-5  │ 8.00631e-5  │ 180000.0  │ 0.999976 │
│ 73  │ σ[2]       │ 1.6547     │ 0.0226622  │ 5.34152e-5  │ 5.4681e-5   │ 1.76625e5 │ 0.999976 │
│ 74  │ σ[3]       │ 2.34151    │ 0.0309603  │ 7.29742e-5  │ 7.72551e-5  │ 1.61231e5 │ 1.00005  │
│ 75  │ σ[4]       │ 1.2508     │ 0.0143453  │ 3.38123e-5  │ 3.40573e-5  │ 180000.0  │ 0.999992 │
│ 76  │ σ[5]       │ 1.63435    │ 0.0188587  │ 4.44503e-5  │ 4.28134e-5  │ 180000.0  │ 0.999999 │
│ 77  │ σ[6]       │ 1.79708    │ 0.0315242  │ 7.43033e-5  │ 7.3137e-5   │ 180000.0  │ 0.999986 │
│ 78  │ σ[7]       │ 2.56289    │ 0.0455589  │ 0.000107383 │ 0.000110382 │ 180000.0  │ 0.999966 │
│ 79  │ σᵦ         │ 0.043449   │ 0.0342693  │ 8.07735e-5  │ 0.000161519 │ 41934.6   │ 1.00053  │
│ 80  │ σₕ         │ 0.0418692  │ 0.0442786  │ 0.000104366 │ 0.000233338 │ 35085.6   │ 1.00042  │

Quantiles

│ Row │ parameters │ 2.5%       │ 25.0%      │ 50.0%      │ 75.0%      │ 97.5%      │
│     │ Symbol     │ Float64    │ Float64    │ Float64    │ Float64    │ Float64    │
├─────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
│ 1   │ L[1,1]     │ 1.0        │ 1.0        │ 1.0        │ 1.0        │ 1.0        │
│ 2   │ L[2,1]     │ -0.154208  │ -0.129116  │ -0.115953  │ -0.102808  │ -0.0776037 │
│ 3   │ L[2,2]     │ 0.988038   │ 0.99163    │ 0.993255   │ 0.994701   │ 0.996984   │
│ 4   │ L[3,1]     │ -0.307722  │ -0.283877  │ -0.271351  │ -0.258619  │ -0.234366  │
│ 5   │ L[3,2]     │ 0.237978   │ 0.25344    │ 0.261513   │ 0.269537   │ 0.284972   │
│ 6   │ L[3,3]     │ 0.914478   │ 0.922279   │ 0.926159   │ 0.929933   │ 0.936818   │
│ 7   │ L[4,1]     │ 0.534065   │ 0.554871   │ 0.565529   │ 0.57598    │ 0.595323   │
│ 8   │ L[4,2]     │ 0.375852   │ 0.392996   │ 0.401991   │ 0.410945   │ 0.428062   │
│ 9   │ L[4,3]     │ -0.355335  │ -0.338067  │ -0.329003  │ -0.320013  │ -0.302875  │
│ 10  │ L[4,4]     │ 0.6212     │ 0.633609   │ 0.640203   │ 0.6468     │ 0.659448   │
│ 11  │ L[5,1]     │ 0.122717   │ 0.148313   │ 0.161813   │ 0.175337   │ 0.200651   │
│ 12  │ L[5,2]     │ 0.0468944  │ 0.0623143  │ 0.070403   │ 0.0785198  │ 0.0937806  │
│ 13  │ L[5,3]     │ -0.547865  │ -0.533389  │ -0.52565   │ -0.517909  │ -0.503063  │
│ 14  │ L[5,4]     │ -0.286814  │ -0.267825  │ -0.257943  │ -0.247931  │ -0.228817  │
│ 15  │ L[5,5]     │ 0.774957   │ 0.785337   │ 0.790729   │ 0.796031   │ 0.805957   │
│ 16  │ L[6,1]     │ -0.222726  │ -0.193611  │ -0.178303  │ -0.162837  │ -0.132805  │
│ 17  │ L[6,2]     │ 0.0999161  │ 0.127105   │ 0.141204   │ 0.155269   │ 0.181911   │
│ 18  │ L[6,3]     │ -0.104183  │ -0.0765312 │ -0.0621185 │ -0.0476222 │ -0.0196702 │
│ 19  │ L[6,4]     │ 0.0286965  │ 0.0574398  │ 0.0723844  │ 0.0873353  │ 0.115501   │
│ 20  │ L[6,5]     │ -0.567889  │ -0.546593  │ -0.535139  │ -0.523308  │ -0.500596  │
│ 21  │ L[6,6]     │ 0.783521   │ 0.798822   │ 0.806771   │ 0.814681   │ 0.829593   │
│ 22  │ L[7,1]     │ -0.417233  │ -0.39163   │ -0.378006  │ -0.364247  │ -0.337933  │
│ 23  │ L[7,2]     │ 0.0439544  │ 0.0694064  │ 0.0826257  │ 0.0957777  │ 0.120923   │
│ 24  │ L[7,3]     │ -0.133256  │ -0.107348  │ -0.0938212 │ -0.0801154 │ -0.0538128 │
│ 25  │ L[7,4]     │ 0.554734   │ 0.573801   │ 0.583603   │ 0.593313   │ 0.611255   │
│ 26  │ L[7,5]     │ -0.210166  │ -0.187897  │ -0.176169  │ -0.164439  │ -0.141788  │
│ 27  │ L[7,6]     │ -0.339281  │ -0.319127  │ -0.308599  │ -0.298127  │ -0.278015  │
│ 28  │ L[7,7]     │ 0.586727   │ 0.602298   │ 0.610484   │ 0.618751   │ 0.634627   │
│ 29  │ lκ[1]      │ -0.622115  │ -0.415351  │ -0.283732  │ -0.133858  │ 0.223513   │
│ 30  │ lκ[2]      │ -0.611764  │ -0.400383  │ -0.267346  │ -0.114982  │ 0.237611   │
│ 31  │ lκ[3]      │ -0.622246  │ -0.412667  │ -0.280909  │ -0.129597  │ 0.230915   │
│ 32  │ lκ[4]      │ -0.615976  │ -0.406495  │ -0.275647  │ -0.125478  │ 0.228314   │
│ 33  │ lκ[5]      │ -0.620588  │ -0.412372  │ -0.28091   │ -0.129564  │ 0.228889   │
│ 34  │ lκ[6]      │ -0.619014  │ -0.411537  │ -0.280215  │ -0.129126  │ 0.222025   │
│ 35  │ lκ[7]      │ -0.623338  │ -0.412699  │ -0.281523  │ -0.131848  │ 0.224927   │
│ 36  │ βᵣ₁[1]     │ -1.82205   │ -0.617815  │ 0.00449691 │ 0.632285   │ 1.85767    │
│ 37  │ βᵣ₁[2]     │ -2.00911   │ -0.845338  │ -0.235942  │ 0.366556   │ 1.55916    │
│ 38  │ βᵣ₁[3]     │ -1.57588   │ -0.294434  │ 0.358311   │ 0.999017   │ 2.20857    │
│ 39  │ βᵣ₁[4]     │ -2.0649    │ -0.948816  │ -0.38835   │ 0.180814   │ 1.37056    │
│ 40  │ βᵣ₁[5]     │ -1.765     │ -0.58047   │ 0.0418131  │ 0.670767   │ 1.89208    │
│ 41  │ βᵣ₁[6]     │ -1.63578   │ -0.385125  │ 0.250027   │ 0.882081   │ 2.08202    │
│ 42  │ βᵣ₁[7]     │ -1.89794   │ -0.672968  │ -0.0332775 │ 0.607868   │ 1.84476    │
│ 43  │ βᵣ₂[1]     │ -2.06076   │ -0.876876  │ -0.24775   │ 0.386181   │ 1.61898    │
│ 44  │ βᵣ₂[2]     │ -1.44459   │ -0.182041  │ 0.440007   │ 1.05499    │ 2.20516    │
│ 45  │ βᵣ₂[3]     │ -2.09837   │ -0.878089  │ -0.237065  │ 0.399421   │ 1.64407    │
│ 46  │ βᵣ₂[4]     │ -1.43895   │ -0.277774  │ 0.283857   │ 0.847888   │ 1.9734     │
│ 47  │ βᵣ₂[5]     │ -2.09861   │ -0.90898   │ -0.286356  │ 0.345589   │ 1.57415    │
│ 48  │ βᵣ₂[6]     │ -1.95829   │ -0.739759  │ -0.112586  │ 0.518946   │ 1.73264    │
│ 49  │ βᵣ₂[7]     │ -1.74253   │ -0.483755  │ 0.163365   │ 0.806098   │ 2.01553    │
│ 50  │ θ[1]       │ 0.703233   │ 0.800741   │ 0.851228   │ 0.902461   │ 1.00512    │
│ 51  │ θ[2]       │ -6.87166   │ -6.78042   │ -6.73638   │ -6.69326   │ -6.60808   │
│ 52  │ θ[3]       │ -3.2435    │ -3.14166   │ -3.08994   │ -3.03838   │ -2.93912   │
│ 53  │ θ[4]       │ -2.82706   │ -2.75891   │ -2.72419   │ -2.68952   │ -2.62109   │
│ 54  │ θ[5]       │ 2.40541    │ 2.47709    │ 2.51443    │ 2.55213    │ 2.62734    │
│ 55  │ θ[6]       │ 3.35582    │ 3.44126    │ 3.48312    │ 3.5246     │ 3.60406    │
│ 56  │ θ[7]       │ 4.27972    │ 4.39041    │ 4.44666    │ 4.50217    │ 4.60849    │
│ 57  │ μᵣ₁[1]     │ -1.92434   │ -0.762743  │ -0.170911  │ 0.428984   │ 1.63294    │
│ 58  │ μᵣ₁[2]     │ -1.76905   │ -0.63451   │ -0.0559209 │ 0.52154    │ 1.69252    │
│ 59  │ μᵣ₁[3]     │ -1.60982   │ -0.387917  │ 0.22946    │ 0.838912   │ 2.03433    │
│ 60  │ μᵣ₂[1]     │ -1.63105   │ -0.444192  │ 0.158891   │ 0.749251   │ 1.90904    │
│ 61  │ μᵣ₂[2]     │ -1.68641   │ -0.516244  │ 0.0618674  │ 0.637377   │ 1.78048    │
│ 62  │ μᵣ₂[3]     │ -2.01436   │ -0.823785  │ -0.207716  │ 0.409176   │ 1.64079    │
│ 63  │ μₕ₁        │ -0.211259  │ -0.128142  │ -0.095558  │ -0.0642836 │ 0.00794573 │
│ 64  │ μₕ₂        │ -0.097284  │ -0.0262976 │ 0.00480748 │ 0.0379734  │ 0.122952   │
│ 65  │ ρ[1]       │ 0.662532   │ 0.679866   │ 0.688926   │ 0.697858   │ 0.714269   │
│ 66  │ ρ[2]       │ 0.675752   │ 0.686767   │ 0.692463   │ 0.698168   │ 0.709125   │
│ 67  │ ρ[3]       │ 0.697536   │ 0.708426   │ 0.714198   │ 0.719905   │ 0.730766   │
│ 68  │ ρ[4]       │ 0.676964   │ 0.691409   │ 0.699008   │ 0.70656    │ 0.720941   │
│ 69  │ ρ[5]       │ 0.676518   │ 0.687618   │ 0.693519   │ 0.699422   │ 0.710746   │
│ 70  │ ρ[6]       │ 0.630339   │ 0.668808   │ 0.685782   │ 0.70116    │ 0.727205   │
│ 71  │ ρ[7]       │ 0.643245   │ 0.67911    │ 0.695215   │ 0.709736   │ 0.73456    │
│ 72  │ σ[1]       │ 2.10248    │ 2.14768    │ 2.17191    │ 2.19703    │ 2.24584    │
│ 73  │ σ[2]       │ 1.6115     │ 1.63919    │ 1.65424    │ 1.66971    │ 1.70033    │
│ 74  │ σ[3]       │ 2.28226    │ 2.32039    │ 2.34093    │ 2.36215    │ 2.40353    │
│ 75  │ σ[4]       │ 1.22322    │ 1.24107    │ 1.25064    │ 1.26035    │ 1.27951    │
│ 76  │ σ[5]       │ 1.59806    │ 1.62154    │ 1.63403    │ 1.64688    │ 1.67215    │
│ 77  │ σ[6]       │ 1.73656    │ 1.77565    │ 1.79662    │ 1.81809    │ 1.85995    │
│ 78  │ σ[7]       │ 2.47542    │ 2.53177    │ 2.56225    │ 2.59317    │ 2.65423    │
│ 79  │ σᵦ         │ 0.00175484 │ 0.0178145  │ 0.0362999  │ 0.0605639  │ 0.128847   │
│ 80  │ σₕ         │ 0.00124591 │ 0.0129366  │ 0.0288774  │ 0.055039   │ 0.162651   │
```
The difference in time between the slowest chain (748 s) and the overall time (775 s) roughly yields the compilation time. If we refit the model, the total time would roughly equal the length of time to sample from the slowest chain.

If you don't pass the sampler a tuner object, it'll create one by default with `M=5` and `term=50`. The section on [Automatic Parameter Tuning](https://mc-stan.org/docs/2_19/reference-manual/hmc-algorithm-parameters.html) in the Stan reference manual explains what these mean. I added an extra slow adaptation step (with twice the length of the previous step) and doubled the length of the terminal adaptation window (a final window calculating step size), as this makes adaptation more consistent.
The idea is to try and decrease the probability of one or two chains being much slower than the others. We can look at NUTS statistics of the chains:
```julia
julia> NUTS_statistics.(chains)
18-element Array{DynamicHMC.NUTS_Statistics{Float64,DataStructures.Accumulator{DynamicHMC.Termination,Int64},DataStructures.Accumulator{Int64,Int64}},1}:
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.94, min/25%/median/75%/max: 0.0 0.94 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 6% DoubledTurn => 93%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 13% 6 => 87%
                         
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.96 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 0% DoubledTurn => 100%
  depth: 4 => 0% 5 => 1% 6 => 99% 7 => 0%
                                 
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.96 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 0% DoubledTurn => 99%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 1% 6 => 99% 7 => 0%
                  
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.97 0.99 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 0% DoubledTurn => 99%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 0% 6 => 100% 7 => 0%
                 
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.98, min/25%/median/75%/max: 0.0 0.98 0.99 1.0 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 17% DoubledTurn => 83%
  depth: 4 => 0% 5 => 0% 6 => 93% 7 => 7% 8 => 0%
                          
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.97 0.99 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 0% DoubledTurn => 100%
  depth: 3 => 0% 4 => 0% 5 => 0% 6 => 100% 7 => 0%
                        
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.94, min/25%/median/75%/max: 0.0 0.95 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 4% DoubledTurn => 96%
  depth: 3 => 0% 4 => 0% 5 => 8% 6 => 92% 7 => 0%
                          
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.95, min/25%/median/75%/max: 0.0 0.96 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 1% DoubledTurn => 99%
  depth: 3 => 0% 4 => 0% 5 => 1% 6 => 99% 7 => 0%
                          
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.94, min/25%/median/75%/max: 0.0 0.95 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 3% DoubledTurn => 97%
  depth: 1 => 0% 2 => 0% 3 => 0% 4 => 0% 5 => 6% 6 => 94% 7 => 0%
          
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.94, min/25%/median/75%/max: 0.0 0.94 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 2% DoubledTurn => 98%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 4% 6 => 96% 8 => 0%
                  
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.91, min/25%/median/75%/max: 0.0 0.92 0.97 0.99 1.0
  termination: AdjacentDivergent => 1% AdjacentTurn => 12% DoubledTurn => 87%
  depth: 1 => 0% 2 => 0% 3 => 0% 4 => 0% 5 => 28% 6 => 71% 7 => 0%
        
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.96 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 0% DoubledTurn => 100%
  depth: 1 => 0% 2 => 0% 3 => 0% 4 => 0% 5 => 1% 6 => 99% 7 => 0% 8 => 0%
 
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.96 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 0% DoubledTurn => 100%
  depth: 3 => 0% 4 => 0% 5 => 1% 6 => 99% 7 => 0% 8 => 0%
                 
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.94, min/25%/median/75%/max: 0.0 0.94 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 4% DoubledTurn => 96%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 8% 6 => 92% 7 => 0%
                  
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.95, min/25%/median/75%/max: 0.0 0.95 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 1% DoubledTurn => 99%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 3% 6 => 97% 7 => 0%
                  
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.96, min/25%/median/75%/max: 0.0 0.96 0.98 0.99 1.0
  termination: AdjacentDivergent => 0% AdjacentTurn => 1% DoubledTurn => 99%
  depth: 2 => 0% 3 => 0% 4 => 0% 5 => 1% 6 => 99% 7 => 0% 8 => 0%
          
 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.91, min/25%/median/75%/max: 0.0 0.92 0.97 0.99 1.0
  termination: AdjacentDivergent => 1% AdjacentTurn => 16% DoubledTurn => 83%
  depth: 1 => 0% 2 => 1% 3 => 0% 4 => 1% 5 => 39% 6 => 60% 7 => 0% 8 => 0%

 Hamiltonian Monte Carlo sample of length 10000
  acceptance rate mean: 0.91, min/25%/median/75%/max: 0.0 0.92 0.97 0.99 1.0
  termination: AdjacentDivergent => 2% AdjacentTurn => 9% DoubledTurn => 89%
  depth: 1 => 0% 2 => 1% 3 => 0% 4 => 0% 5 => 20% 6 => 78% 7 => 0%
```
and see that the fifth chain had an acceptance rate of 0.98, vs the target acceptance rate of `δ = 0.95`. 7% of samples hit a treedepth of 7, meaning this was probably the slowest chain (worker 15). Maybe increasing term further would help.


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





