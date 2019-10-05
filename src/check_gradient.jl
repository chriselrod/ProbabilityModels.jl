


rel_error(x, y) = (x - y) / y
function check_gradient(data, a = randn(length(data)))
    acopy = copy(a)
    lp, g = logdensity_and_gradient(data, a)
    all(i -> a[i] == acopy[i], eachindex(a)) || throw("Logdensity mutated inputs!?!?!?")
    for i ∈ eachindex(a)
        aᵢ = a[i]
        step = cbrt(eps(aᵢ))
        a[i] = aᵢ + step
        lp_hi = logdensity(data, a)
        a[i] = aᵢ - step
        lp_lo = logdensity(data, a)
        a[i] = aᵢ
        fd = (lp_hi - lp_lo) / (2step)
        ad = g[i]
        relative_error = rel_error(ad, fd)
        @show (i, ad, fd, relative_error)
        if abs(relative_error) > 1e-5
            fd_f = (lp_hi - lp) / step
            fd_b = (lp - lp_lo) / step
            @show rel_error.(ad, (fd_f, fd_b))
        end
    end
end


