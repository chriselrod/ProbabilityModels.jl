
using MacroTools, PaddedMatrices, DiffRules, VectorizationBase, SLEEF
using MacroTools: striplines, @capture, prewalk, postwalk, @q

using LogDensityProblems, DynamicHMC, VectorizedRNG


includet("/home/chriselrod/.julia/dev/DistributionParameters/src/uniform_mapped_parameters.jl")
includet("/home/chriselrod/Documents/progwork/julia/autodiff_work/distribution_functions.jl")
includet("/home/chriselrod/Documents/progwork/julia/autodiff_work/special_diff_rules.jl")
includet("/home/chriselrod/Documents/progwork/julia/autodiff_work/model_macro_passes.jl")

# q = @q begin
#     β₀ ~ Normal(μ₀, σ₀)
#     β₁ ~ Normal(μ₁, σ₁)
#     y ~ Bernoulli_logit(β₀ + x * β₁)
# end
#
# q2 = translate_sampling_statements(q)
# q3 = flatten_expression(q2)

varnames = Set([:β₀,  :β₁])
first_updates_to_assignemnts(q3, varnames)

q = @q begin
    β₀ ~ Normal(μ₀, σ₀)
    β₁ ~ Normal(μ₁, σ₁)
    y ~ Bernoulli_logit(β₀ + x * β₁)
end

q2 = translate_sampling_statements(q)
expr = flatten_expression(q2)
return_partials = true
model_parameters = Symbol[]
first_pass = quote end
second_pass = quote end

push!(model_parameters, :β₀)
load_parameter(first_pass.args, second_pass.args, :β₀, RealFloat, return_partials)
push!(model_parameters, :β₁)
load_parameter(first_pass.args, second_pass.args, :β₁, RealVector{4,Float64}, return_partials)

tracked_vars = Set(model_parameters)
first_pass, name_dict = rename_assignments(first_pass)
expr, name_dict = rename_assignments(expr, name_dict)

second_pass, name_dict = rename_assignments(second_pass, name_dict)
TLθ = 5 #type_length($θ) # This refers to the type of the input
reverse_diff_pass!(first_pass, second_pass, expr, tracked_vars)
first_pass
second_pass

T_sym = gensym(:T); θ_sym = gensym(:θ);
expr_out = quote
    # target = zero($T_sym)
    $(Symbol("##θparameter##")) = VectorizationBase.vectorizable($θ_sym)
    $first_pass
    $(Symbol("##∂θparameter##m")) = PaddedMatrices.MutableFixedSizePaddedVector{$TLθ,$T_sym}(undef)
    $(Symbol("##∂θparameter##")) = VectorizationBase.vectorizable($(Symbol("##∂θparameter##m")))
    $(Symbol("###adjoint###", name_dict[:target])) = ProbabilityModels.One()
    $second_pass
    LogDensityProblems.ValueGradient(
        $(name_dict[:target]),
        PaddedMatrices.ConstantFixedSizePaddedVector($(Symbol("##∂θparameter##m")))
    )
end;
final_expr = first_updates_to_assignemnts(expr_out, model_parameters)



using MacroTools, PaddedMatrices, DiffRules, VectorizationBase, SLEEF, Random, SIMDPirates
using MacroTools: striplines, @capture, prewalk, postwalk, @q
using LogDensityProblems, DynamicHMC, VectorizedRNG, RandomNumbers

includet("/home/chriselrod/.julia/dev/DistributionParameters/src/uniform_mapped_parameters.jl")
includet("/home/chriselrod/Documents/progwork/julia/autodiff_work/distribution_functions.jl")
includet("/home/chriselrod/Documents/progwork/julia/autodiff_work/special_diff_rules.jl")
includet("/home/chriselrod/Documents/progwork/julia/autodiff_work/model_macro_passes.jl")
includet("/home/chriselrod/.julia/dev/ProbabilityModels/src/dynamic_hmc_interface.jl")

using ProbabilityModels, DistributionParameters, LogDensityProblems, DynamicHMC
using VectorizationBase, LoopVectorization
@model BernoulliLogitModel begin
    β₀ ~ Normal(μ₀, σ₀)
    β₁ ~ Normal(μ₁, σ₁)
    y ~ Bernoulli_logit(β₀ + x * β₁)
end




# using StaticArrays
N = 800; N_β = 4;
X = randn(N, N_β);
β₁ = [-1.595526740808615, -1.737875659746032, -0.26107993378119343, 0.6500851571519769];
β₀ = -0.05
Xβ₁ = X * β₁;
p = rand(N);
y = @. p < 1 / (1 + exp( - Xβ₁ - β₀));
sum(y)

ℓ = BernoulliLogitModel(
    σ₀ = 10.0, σ₁ = 5.0, μ₀ = 0.0, μ₁ = 0.0,
    β₀ = RealFloat, β₁ = RealVector{4},
    y = y, x = X
);


# eval(θquote)
dimension(ℓ)
a = fill(1.0, dimension(ℓ));
logdensity(LogDensityProblems.ValueGradient, ℓ, a)

using LogDensityProblems, DynamicHMC
@time mcmc_chain, tuned_sampler = NUTS_init_tune_mcmc_default(ℓ, 4000);
sample_mean(mcmc_chain)
sample_cov(mcmc_chain)



using BenchmarkTools
@benchmark logdensity(LogDensityProblems.ValueGradient, $ℓ, $a)
@benchmark NUTS_init_tune_mcmc_default($ℓ, 4000, report = DynamicHMC.ReportSilent())


struct_quote, struct_kwarg_quote, directquote, thetaquote, dimquote = generate_generated_funcs_expressions(:BernoulliLogitModel, q);
using LogDensityProblems, DynamicHMC
eval(struct_quote)
eval(struct_kwarg_quote)
eval(thetaquote)
eval(dimquote)
@time mcmc_chain, tuned_sampler = NUTS_init_tune_mcmc_default(rng, ℓ1, 4000);
sample_mean(mcmc_chain)
sample_cov(mcmc_chain)

@benchmark NUTS_init_tune_mcmc($rng, $ℓ1, 4000, report = DynamicHMC.ReportSilent())


mcmc_chainv = reshape(reinterpret(Float64, get_position.(mcmc_chain)), (8,length(mcmc_chain)))[1:5,:];


@code_warntype NUTS_init_tune_mcmc(rng, ℓ1, 4000, report = DynamicHMC.ReportSilent())
# Not type stable?

@code_warntype DynamicHMC.NUTS_init(rng, ℓ1, report = DynamicHMC.ReportSilent())
sampler_init = DynamicHMC.NUTS_init(rng, ℓ1, report = DynamicHMC.ReportSilent())
tuners = DynamicHMC.bracketed_doubling_tuner()
@code_warntype DynamicHMC.tune(sampler_init, tuners)
sampler_tuned = DynamicHMC.tune(sampler_init, DynamicHMC.bracketed_doubling_tuner());
@code_warntype DynamicHMC.mcmc(sampler_tuned, 4000)
mcmcres = DynamicHMC.mcmc(sampler_tuned, 4000)

function iterator_test(N)
    out = 0.0
    for i ∈ 1:4:N
        out += 2.3i
    end
    out
end

GK1 = GaussianKE(dimension(ℓ1));
H = DynamicHMC.Hamiltonian(ℓ1, GK1);
q = randn(rng, dimension(ℓ1));
p = rand(rng, GK1);
z = DynamicHMC.phasepoint_in(H, q, p)
DynamicHMC.find_initial_stepsize(DynamicHMC.InitialStepsizeSearch(), H, z)
@code_warntype DynamicHMC.NUTS_init(rng, ℓ1, report = false)

sampler_init = DynamicHMC.NUTS_init(rng, ℓ1);
tuners = DynamicHMC.bracketed_doubling_tuner()
@which DynamicHMC.tune(sampler_init,tuners.tuners[1])

sample, A = DynamicHMC.mcmc_adapting_ϵ(sampler_init, tuners.tuners[1].N)


function sample_sum(sample)
    N = length(sample)
    x̄ = DynamicHMC.get_position(sample[1])
    @inbounds for n ∈ 2:N
        x̄ += DynamicHMC.get_position(sample[n])
    end
    x̄
end
sample_v = get_position.(sample);
sample_mat = reshape(reinterpret(Float64, sample_v), (8,75))[1:5,:];



using ForwardDiff, Optim
inv_logit(x) = 1 / (1 + exp(-x))
function Bernoulli_logit_fmadd_simple(y, X, β, α)
    p = inv_logit.( X * β .+ α )
    T = eltype(p)
    target = zero(T)
    @inbounds @simd ivdep for i ∈ eachindex(p, y)
        OmP = one(T) - p[i]
        target += y[i] ? log(p[i]) : log(OmP)
    end
    target
end
opt = optimize(b -> -Bernoulli_logit_fmadd_simple(y, X, b, 1.0), ones(4), BFGS())


βones =  fill(1.0, PaddedMatrices.Static{4}())
∂Bernoulli_logit_fmadd_logeval_dropconst(y, X, βones, 1.0, Val{(false,false,true,true)}())

Bernoulli_logit_fmadd_simple(y, X, ones(4), 1.0)
ForwardDiff.derivative(a -> Bernoulli_logit_fmadd_simple(y, X, ones(4), a), 1.0)
ForwardDiff.gradient(b -> Bernoulli_logit_fmadd_simple(y, X, b, 1.0), ones(4))
Normal_logeval_dropconst(βones, 0.0, 5.0, Val{(true,false,false)}())




@model ITPModel begin

    # Priors
    ρ ~ Beta(α_ρ, β_ρ)
    β ~ Normal(μ_β, σ_β)
    κ ~ Gamma(α_κ, β_κ)
    L ~ LKJ(η₀)

    # Likelihood
    μ = ITPExpectedValue(t, β, κ)
    AR = AutoregressiveMatrixLowerCholeskyInverse(2ρ-1, δt)
    Y ~ Normal( μ, AR, L )

end

# Defined model: ITPModel.
# Unknowns: α_κ, δt, μ, μ_β, η₀, σ_β, AR, Y, β, ρ, α_ρ, κ, L, β_κ, t, β_ρ.

δt = rand(23);
t = cumsum(δt); pushfirst!(t, 0);
δtv = ConstantFixedSizePaddedVector{23,Float64}(δt);
tv = ConstantFixedSizePaddedVector{24,Float64}(t);

β = @CFixedSize randn(16);
κ = @CFixedSize rand(16);

ℓitp = BernoulliLogitModel(
    α_κ = , δt, μ, μ_β, η₀, σ_β, AR, Y, β, ρ, α_ρ, κ, L, β_κ, t, β_ρ
    σ₀ = 10.0, σ₁ = 5.0, μ₀ = 0.0, μ₁ = 0.0,
    β₀ = RealFloat, β₁ = RealVector{4},
    y = y, x = X
);

# y =
