
@inline DynamicHMC.rand_bool(pcg::AbstractPCG, prob::Float64) = Float64(rand(pcg, UInt64)) < typemax(UInt64) * prob
@inline DynamicHMC.rand_bool(pcg::AbstractPCG, prob::Float32) = Float32(rand(pcg, UInt32)) < typemax(UInt32) * prob
function DynamicHMC.random_position(pcg::AbstractPCG, N)
    # if we really want to optimize it, we would calculate r in one pass instead of two.
    r = Vector{Float64}(undef, N)
    rand!(pcg, r, -2.0, 2.0)
    r
end
function DynamicHMC.random_position(sp::StackPointer, pcg::ScalarVectorPCG, ::Val{N}) where {N}
    # if we really want to optimize it, we would calculate r in one pass instead of two.
    sp, r = PtrVector{N,Float64}(sp)
    rand!(pcg.vector, r, -2.0, 2.0)
    sp, r
end

function leapfrog(sp::StackPointer,
        H::Hamiltonian{DynamicHMC.GaussianKineticEnergy{<:Diagonal}},
        z::PhasePoint{DynamicHMC.EvaluatedLogDensity{T,S}}, ϵ
) where {P,L,S,T <: AbstractFixedSizePaddedVector{P,S,L,L}}
    @unpack ℓ, κ = H
    @unpack p, Q = z
    @argcheck isfinite(Q.ℓq) "Internal error: leapfrog called from non-finite log density"
    sptr = pointer(sp, S)
    B = L*sizeof(S)
    pₘ = PtrVector{P,S,L,L}(sptr)
    # Variables that escape:
    # p′, Q′ (q′, ∇ℓq)
    ϵₕ = S(0.5) * ϵ
    ∇ℓq = Q.∇ℓq
    @. pₘ = p + ϵₕ * ∇ℓq
#    ∇ke = ∇kinetic_energy(κ, pₘ)
    M⁻¹ = κ.M⁻¹.diag
    q = Q.q
    q′ = PtrVector{P,S,L,L}(sptr + B) # should this be a new Vector?
    @. q′ = q + ϵ * M⁻¹ * pₘ
    sp, Q′ = evaluate_ℓ(sp + 2B, H.ℓ, q′)
    ∇ℓq′ = Q′.∇ℓq
    p′ = pₘ # PtrVector{P,S,L,L}(sptr + 3bytes)
    @. p′ = pₘ + ϵₕ * ∇ℓq′
    sp + 3B, PhasePoint(Q′, p′)
end
function move(sp::StackPointer, trajectory::TrajectoryNUTS, z, fwd)
    @unpack H, ϵ = trajectory
    leapfrog(sp, H, z, fwd ? ϵ : -ϵ)
end
function DynamicHMC.adjacent_tree(sp::StackPointer, rng, trajectory, z, i, depth, is_forward)
    i′ = i + (is_forward ? 1 : -1)
    if depth == 0
        sp, z′ = move(sp, trajectory, z, is_forward)
        ζωτ, v = leaf(trajectory, z′, false)
        if ζωτ ≡ nothing
            sp, DynamicHMC.InvalidTree(i′), v
        else
            sp, (ζωτ..., z′, i′), v
        end
    else
        # “left” tree
        sp, t₋, v₋ = adjacent_tree(sp, rng, trajectory, z, i, depth - 1, is_forward)
        t₋ isa InvalidTree && return sp, t₋, v₋
        ζ₋, ω₋, τ₋, z₋, i₋ = t₋

        # “right” tree — visited information from left is kept even if invalid
        sp, t₊, v₊ = adjacent_tree(sp, rng, trajectory, z₋, i₋, depth - 1, is_forward)
        v = combine_visited_statistics(trajectory, v₋, v₊)
        t₊ isa InvalidTree && return sp, t₊, v
        ζ₊, ω₊, τ₊, z₊, i₊ = t₊

        # turning invalidates
        τ = combine_turn_statistics_in_direction(trajectory, τ₋, τ₊, is_forward)
        is_turning(trajectory, τ) && return sp, InvalidTree(i′, i₊), v

        # valid subtree, combine proposals
        ζ, ω = combine_proposals_and_logweights(rng, trajectory, ζ₋, ζ₊, ω₋, ω₊, is_forward, false)
        sp, (ζ, ω, τ, z₊, i₊), v
    end
end

function DynamicHMC.sample_trajectory(sp::StackPointer, rng, trajectory, z, max_depth::Integer, directions::Directions)
    @argcheck max_depth ≤ MAX_DIRECTIONS_DEPTH
    (ζ, ω, τ), v = leaf(trajectory, z, true)
    z₋ = z₊ = z
    depth = 0
    termination = REACHED_MAX_DEPTH
    i₋ = i₊ = 0
    while depth < max_depth
        is_forward, directions = next_direction(directions)
        sp, t′, v′ = adjacent_tree(sp, rng, trajectory, is_forward ? z₊ : z₋, is_forward ? i₊ : i₋,
                               depth, is_forward)
        v = combine_visited_statistics(trajectory, v, v′)

        # invalid adjacent tree: stop
        t′ isa InvalidTree && (termination = t′; break)

        # extract information from adjacent tree
        ζ′, ω′, τ′, z′, i′ = t′

        # update edges and combine proposals
        if is_forward
            z₊, i₊ = z′, i′
        else
            z₋, i₋ = z′, i′
        end

        # tree has doubled successfully
        ζ, ω = combine_proposals_and_logweights(rng, trajectory, ζ, ζ′, ω, ω′,
                                                is_forward, true)
        depth += 1

        # when the combined tree is turning, stop
        τ = combine_turn_statistics_in_direction(trajectory, τ, τ′, is_forward)
        is_turning(trajectory, τ) && (termination = InvalidTree(i₋, i₊); break)
    end
    ζ, v, termination, depth
end

function NUTS_sample_tree(sp::StackPointer, rng, options::TreeOptionsNUTS, H::Hamiltonian,
                          Q::EvaluatedLogDensity, ϵ;
                          p = rand_p(rng, H.κ), directions = rand(rng, Directions))
    @unpack max_depth, min_Δ, turn_statistic_configuration = options
    z = PhasePoint(Q, p)
    trajectory = TrajectoryNUTS(H, logdensity(H, z), ϵ, min_Δ, turn_statistic_configuration)
    ζ, v, termination, depth = sample_trajectory(rng, trajectory, z, max_depth, directions)
    statistics = TreeStatisticsNUTS(logdensity(H, ζ), depth, termination,
                                    acceptance_rate(v), v.steps, directions)
    ζ.Q, statistics
end

@generated function pointer_vector_type(::AbstractProbabilityModel{D}, ::Type{T}) where {D,T}
    W = VectorizationBase.pick_vector_width(D, T)
    L = VectorizationBase.align(D, W)
    PtrVector{D,T,L,L,true}
end

function warmup(sampling_logdensity, tuning::TuningNUTS{M}, warmup_state) where {M}
    @unpack rng, ℓ, algorithm, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    @unpack N, stepsize_adaptation, λ = tuning
    
    PV = pointer_vector_type(ℓ, T)
    L = PaddedMatrices.full_length(PV)
    chain = Matrix{typeof(Q.q)}(undef, L, N)
    chain_ptr = pointer(chain)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    ϵ_state = initial_adaptation_state(stepsize_adaptation, ϵ)
    ϵs = Vector{Float64}(undef, N)
    mcmc_reporter = make_mcmc_reporter(reporter, N; tuning = M ≡ Nothing ? "stepsize" :
                                       "stepsize and $(M) metric")
    for i in 1:N
        ϵ = current_ϵ(ϵ_state)
        ϵs[i] = ϵ
        Q, stats = sample_tree(rng, algorithm, H, Q, ϵ)
        copyto!( PV( chain_ptr ), Q.q ); chain_ptr += L*sizeof(T)
        tree_statistics[i] = stats
        ϵ_state = adapt_stepsize(stepsize_adaptation, ϵ_state, stats.acceptance_rate)
        report(mcmc_reporter, i; ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    end
    if M ≢ Nothing
        κ = GaussianKineticEnergy(regularize_M⁻¹(sample_M⁻¹(M, chain), λ))
        report(mcmc_reporter, "adaptation finished", adapted_kinetic_energy = κ)
    end
    ((chain = chain, tree_statistics = tree_statistics, ϵs = ϵs),
     WarmupState(Q, κ, final_ϵ(ϵ_state)))
end

function mcmc(sampling_logdensity::AbstractProbabilityModel{D}, N, warmup_state, sp = STACK_POINTER_REF[]) where {D}
    @unpack rng, ℓ, sampler_options, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    chain = Matrix{eltype(Q.q)}(undef, length(Q.q), N)
#    chain = Vector{typeof(Q.q)}(undef, N)
    tree_statistics = Vector{TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    mcmc_reporter = make_mcmc_reporter(reporter, N)
    for i in 1:N
        Q, tree_statistics[i] = NUTS_sample_tree(sp, rng, sampler_options, H, Q, ϵ)
        chain[:,i] .= Q.q
        report(mcmc_reporter, i)
    end
    (chain = chain, tree_statistics = tree_statistics)
end









