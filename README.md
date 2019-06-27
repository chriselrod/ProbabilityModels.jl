

This is alpha-quality software. It is under active development. Optimistically, I hope to have it and its dependencies reasonably well documented and tested, and all the libraries registered, by the end of the year. There is a roadmap issue [here](https://github.com/chriselrod/ProbabilityModels.jl/issues/5).

The primary goal of this library is to make it as easy as possible to specify models that run as quickly as possible $-$ providing both log densities and the associated gradients. This allows the library to be a front end to Hamiltonian Monte Carlo backends, such as [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl). [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)/[Turing's NUTS](https://github.com/TuringLang/Turing.jl) is also probably worth looking into, as it [reportedly converges the most reliably](https://discourse.julialang.org/t/mcmc-landscape/25654/11?u=elrod), at least within DiffEqBayes.


*A brief introduction.*

First, you specify a model using a DSL:
```julia

using PaddedMatrices
using StructuredMatrices, DistributionParameters
using ProbabilityModels, LoopVectorization, SLEEFPirates, SIMDPirates, ProbabilityDistributions, PaddedMatrices
using ProbabilityModels: HierarchicalCentering, ∂HierarchicalCentering, ITPExpectedValue, ∂ITPExpectedValue
using DistributionParameters: CovarianceMatrix#, add


@model ITPModel begin
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
# Evaluating the code prints:
#    Defined model: ITPModel.
#    Unknowns: Y₂, domains, μₕ₂, μᵣ₁, time, σ, AvailableData, σᵦ, θ, μᵣ₂, ρ, σₕ, μₕ₁, κ, L, Y₁, βᵣ₂, βᵣ₁.

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
using PaddedMatrices, LinearAlgebra, LoopVectorization, Random
using DistributionParameters
using DistributionParameters: MissingDataVector


function rinvscaledgamma(::Val{N},a::T,b::T,c::T) where {N,T}
    rg = MutableFixedSizePaddedVector{N,T}(undef)
    log100 = log(100)
    @inbounds for n ∈ 1:N
        rg[n] = log100 / exp(log(PaddedMatrices.randgamma(a/c)) / c + b)
    end
    rg
end

domains = ProbabilityModels.Domains(2,2,3)

n_endpoints = sum(domains)

time = MutableFixedSizePaddedVector{36,Float64,36,36}(undef); time .= 0:35;


missing = push!(vcat(([1,0,0,0,0] for i ∈ 1:7)...), 1);
length(missing)
missing_pattern = vcat(
    missing, fill(1, 4length(time)), missing, missing
);
length(missing_pattern)

availabledata = MissingDataVector{Float64}(missing_pattern);

κ₀ = (8.5, 1.5, 3.0)

AR1(ρ, t) = @. ρ ^ abs(t - t')

function generate_true_parameters(domains, time, κ₀)
    K = sum(domains)
    D = length(domains)
    T = length(time)

	θ = MutableFixedSizePaddedVector{K,Float64}(undef)

    offset = 0
    for i ∈ domains
        domain_mean = 10randn()
        for j ∈ 1+offset:i+offset
            θ[j] = domain_mean + 15randn()
        end
        offset += i
    end

    σ = PaddedMatrices.randgamma( 6.0, 1/6.0)

    ρ = PaddedMatrices.randbeta(4.0,4.0)

    κ = rinvscaledgamma(Val(K), κ₀...)
    lt = last(time)
    μ₁ = @. θ' - 0.05 * ( 1.0 - exp( - κ' * time) ) / (1.0 - exp( - κ' * lt) ) 
    μ₂ = @. θ' + 0.05 * ( 1.0 - exp( - κ' * time) ) / (1.0 - exp( - κ' * lt) ) 
    
    L_T, info = LAPACK.potrf!('L', AR1(ρ, time))
    @inbounds for tc ∈ 2:T, tr ∈ 1:tc-1
        L_T[tr,tc] = 0.0
    end

    X = PaddedMatrices.MutableFixedSizePaddedMatrix{K,K+3,Float64,K}(undef); randn!(X)
    U_K, info = LAPACK.potrf!('U', BLAS.syrk!('U', 'N', σ, X, 0.0, MutableFixedSizePaddedMatrix{K,K,Float64,K}(undef)))

    (
        U_K = U_K, L_T = L_T, μ = μ, μ₁ = μ₁, μ₂ = μ₂, domains = domains, time = time
    )
end


function sample_data( N::Tuple{Int,Int}, truth, missingness )

    K = sum(truth.domains)
    D = length(truth.domains)
    
    N₁, N₂ = N
    L_T = truth.L_T
    U_K = truth.U_K
    T = size(L_T,1)

    Y₁ = [ ConstantFixedSizePaddedArray(L_T * ((@Mutable randn(T,K) ) * U_K)) for n ∈ 1:N₁]
    Y₂ = [ ConstantFixedSizePaddedArray(L_T * ((@Mutable randn(T,K) ) * U_K)) for n ∈ 1:N₂]

    rT = LoopVectorization.stride_row(L_T)
    Y₁64 = reinterpret(Float64, Y₁)
    Y₂64 = reinterpret(Float64, Y₂)
    Y = (
        reshape(Y₁64, (rT,K,N₁)),
        reshape(Y₂64, (rT,K,N₂))
    )

    μ = (truth.μ₁, truth.μ₂) 
	@inbounds for i ∈ 1:2
        μᵢ = μ[i]
        Yᵢ = Y[i]
        for n ∈ 1:N[i]
            for k ∈ 1:K
                @simd for t ∈ 1:rT
                    Yᵢ[t,k,n] += μᵢ[t,k]
                end
            end
        end
    end
    
    c = length(missingness.indices)
    inds = vcat(missingness.indices, T+1:rT)
    ITPModel(
        domains = truth.domains,
        AvailableData = missingness,
        Y₁ = DynamicPaddedMatrix(reshape(Y₁64, (rT * K, N₁))[inds, :], (c, N₁)),
        Y₂ = DynamicPaddedMatrix(reshape(Y₂64, (rT * K, N₂))[inds, :], (c, N₂)),
        time = truth.time,
        L = LKJCorrCholesky{K},
        ρ = UnitVector{K},
        κ = PositiveVector{K},
        θ = RealVector{K},
        μₕ₁ = RealFloat,
        μₕ₂ = RealFloat,
        μᵣ₁ = RealVector{D},
        μᵣ₂ = RealVector{D},
        βᵣ₁ = RealVector{K},
        βᵣ₂ = RealVector{K},
        σᵦ = PositiveFloat,
        σₕ = PositiveFloat,
        σ = PositiveVector{K}
    )
end

truth = generate_true_parameters(domains, time, κ₀);

data = sample_data((100,100), truth, availabledata);

```
The library [DistributionParameters.jl](https://github.com/chriselrod/DistributionParameters.jl) provides a variety of parameter types.
These types define constrianing transformations, to transform an unconstrained parameter vector and add the appropriate jacobians.

All parameters are typed by size. The library currently provides a DynamicHMC interface, defining `logdensity(::Value,::ITPModel)` and `logdensity(::ValueGradient,::ITPModel)` methods.

```julia
using LogDensityProblems: Value, ValueGradient, logdensity, dimension
using DynamicHMC

a = randn(dimension(data)); length(a) # 73

logdensity(Value, data, a)
logdensity(ValueGradient, data, a)

using BenchmarkTools, LinearAlgebra
BLAS.set_num_threads(1)
@benchmark logdensity(ValueGradient, $data, $a)
#BenchmarkTools.Trial: 
#  memory estimate:  832 bytes
#  allocs estimate:  5
#  --------------
#  minimum time:     565.204 μs (0.00% GC)
#  median time:      574.494 μs (0.00% GC)
#  mean time:        573.796 μs (0.00% GC)
#  maximum time:     888.510 μs (0.00% GC)
#  --------------
#  samples:          8683
#  evals/sample:     1
```
For comparison, a Stan implementation of this model takes close to 10ms to evaluate the gradient.

These are all you need for sampling with `DynamicHMC.NUTS_init_tune_mcmc`. The `@model` macro also defines a helper function:

```julia
constrained = constrain(data, a);
constrained.L
#7×7 LKJCorrCholesky{7,Float64,28}:
#  1.0          0.0         0.0         0.0         0.0        0.0       0.0     
#  0.338466     0.940979    0.0         0.0         0.0        0.0       0.0     
# -0.104864    -0.404536    0.90849     0.0         0.0        0.0       0.0     
#  0.00499714   0.0150394  -0.0528119   0.998479    0.0        0.0       0.0     
#  0.158857     0.658467   -0.083386   -0.0334465   0.730147   0.0       0.0     
#  0.123625     0.650106   -0.361581    0.00885598  0.183654   0.6305    0.0     
# -0.533883     0.638004    0.0584124  -0.307069    0.0783033  0.240715  0.382285
```
So once you have constrained your MCMC samples, you can proceed with your convergence assessments and posterior analysis as normal.


*Overview of how the library works.*

The `ITPModel` struct (or whatever you've named your model, with the first argument to the macro) is defined as a struct with a field for each unknown. Each field is parametrically typed.

`logdensity` and `constrain` are defined as generated functions, so that they can compile appropriate code given these parameteric types. The `@model` macro does a little preprocessing of the expression; most of the code generation occurs within these generated functions.

The macro's preprocessing consists of simply translating the sampling statements into log probability increments (`target`, terminology taken from the [Stan](https://mc-stan.org/users/documentation/) language), and lowering the expression.

We can manually perform these passes on the expression:
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
The reverse diff pas uses [DiffRules.jl](https://github.com/JuliaDiff/DiffRules.jl) as well as custom derivatives defined for a the log density functions and others.

By liberally defining high level derivatives for often used functions, we have an easy route to high performance. Additionally, we abuse multiple dispatch here rather heavily: whenevever there is some structure we can exploit in the adjoint, we return a type with `INCREMENT/MULTIPLY` methods defined to exploit it. Examples include [BlockDiagonalColumnView](https://github.com/chriselrod/StructuredMatrices.jl/blob/master/src/block_diagonal.jl#L4)s or various [Reducer](https://github.com/chriselrod/ProbabilityModels.jl/blob/master/src/adjoints.jl#L55) types.

Another optimization is that it performs a [stack pointer pass](https://github.com/chriselrod/PaddedMatrices.jl/blob/master/src/stack_pointer.jl#L55). Supported functions are passed a pointer to preallocated memory as an argument. [PaddedMatrices.jl](https://github.com/chriselrod/PaddedMatrices.jl), aside from providing [optimized matrix operations](https://bayeswatch.org/2019/06/06/small-matrix-multiplication-performance-shootout/) (contrary to the name, it's performance advantage over other libraries is actually greatest when none of them use padding), it also provides `PtrArray` (parameterized by size) and `DynamicPtrArray` (dynamically sized) types for taking advantage of this preallocated memory.





