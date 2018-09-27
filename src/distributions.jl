
# use DiffRules?
# or define own? Leaning towards latter, to make it easier to share
# calculations between objective and gradient
# although `@cse` macro may work as well.

function Bernoulli_logit(y::AbstractVector, l::AbstractVector)

end

function Normal()

end

function Gamma()

end

function Beta()

end

function fgradient(f, args...)
  y, J = Zygote.forward(f, args...)
  y isa Real || error("Function output is not scalar")
  return y, J(1)
end
