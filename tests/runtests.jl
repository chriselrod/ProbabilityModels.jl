
using LightGraphs
dag = SimpleDiGraph(4)


@model begin LogisticRegression
    @vertices {
        y::Vector{N}
        x::Vector{N}
        μ₀::Real = 0
        σ₀::Real{lower=0} = 5
        μ₁::Real = 0
        σ₁::Real{lower=0} = 5
        β₀::Real
        β₁::Real
    }
    β₀ ~ Normal(μ₀, σ₀)
    β₁ ~ Normal(μ₁, σ₁)
    y ~ Bernoulli_logit(β₀ + x * β₁)
end

@model begin


end

g, d, label, type_params = @vertices {
    # comment 1
    y::Vector{N}
    x::Vector{N}
    μ₀::Real = 0
    σ₀::Real{lower=0} = 5
    μ₁::Real = 0
    # comment 2
    σ₁::Real{lower=0} = 5
    β₀::Real
    β₁::Real
}
