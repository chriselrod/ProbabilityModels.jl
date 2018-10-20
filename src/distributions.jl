
# use DiffRules?
# or define own? Leaning towards latter, to make it easier to share
# calculations between objective and gradient
# although `@cse` macro may work as well.

function Bernoulli_logit(y::AbstractVector, ι::AbstractVector)
    target = zero(eltype(ι))
    @inbounds @simd for i ∈ 1:full_length(ι)
        target += 
    end
    target
end

function Normal(y, μ, σ)

end

function Gamma(y, α, β)

end

function Beta(θ, α, β)

end


function fgradient(f, args...)
  y, J = Zygote.forward(f, args...)
  y isa Real || error("Function output is not scalar")
  return y, J(1)
end
