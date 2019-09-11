
function DynamicHMC.leapfrog(sp::StackPointer,
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
    sp, Q′ = DynamicHMC.evaluate_ℓ(sp + 2B, H.ℓ, q′)
    ∇ℓq′ = Q′.∇ℓq
    p′ = pₘ # PtrVector{P,S,L,L}(sptr + 3bytes)
    @. p′ = pₘ + ϵₕ * ∇ℓq′
    sp + 3B, DynamicHMC.PhasePoint(Q′, p′)
end
function DynamicHMC.move(sp::StackPointer, trajectory::DynamicHMC.TrajectoryNUTS, z, fwd)
    @unpack H, ϵ = trajectory
    leapfrog(sp, H, z, fwd ? ϵ : -ϵ)
end

"""
Returns a tuple of
dim, stride, element type
"""
describe_phase_point(z::PhasePoint{EvaluatedLogDensity{PtrVector{M,T,L,L}}}}) where {M,T,L} = M,L,T

function DynamicHMC.adjacent_tree(sp::StackPointer, rng, trajectory, z, i, depth, is_forward)
    i′ = i + (is_forward ? 1 : -1)
    if depth == 0
        sp, z′ = DynamicHMC.move(sp, trajectory, z, is_forward)
        ζωτ, v = DynamicHMC.leaf(trajectory, z′, false)
        if ζωτ ≡ nothing
            sp, DynamicHMC.InvalidTree(i′), v
        else
            sp, (ζωτ..., z′, i′), v
        end
    else
        # “left” tree
        sp, t₋, v₋ = DynamicHMC.adjacent_tree(sp, rng, trajectory, z, i, depth - 1, is_forward)
        t₋ isa InvalidTree && return sp, t₋, v₋
        ζ₋, ω₋, τ₋, z₋, i₋ = t₋

        # “right” tree — visited information from left is kept even if invalid
        sp, t₊, v₊ = DynamicHMC.adjacent_tree(sp, rng, trajectory, z₋, i₋, depth - 1, is_forward)
        v = DynamicHMC.combine_visited_statistics(trajectory, v₋, v₊)
        t₊ isa DynamicHMC.InvalidTree && return sp, t₊, v
        ζ₊, ω₊, τ₊, z₊, i₊ = t₊

        # turning invalidates
        τ = DynamicHMC.combine_turn_statistics_in_direction(trajectory, τ₋, τ₊, is_forward)
        is_turning(trajectory, τ) && return sp, DynamicHMC.InvalidTree(i′, i₊), v

        # valid subtree, combine proposals
        ζ, ω = DynamicHMC.combine_proposals_and_logweights(rng, trajectory, ζ₋, ζ₊, ω₋, ω₊, is_forward, false)
        sp, (ζ, ω, τ, z₊, i₊), v
    end
end

function DynamicHMC.sample_trajectory(sp::StackPointer, rng, trajectory, z, max_depth::Integer, directions::Directions)
    @argcheck max_depth ≤ DynamicHMC.MAX_DIRECTIONS_DEPTH
    (ζ, ω, τ), v = leaf(trajectory, z, true)
    z₋ = z₊ = z
    depth = 0
    termination = DynamicHMC.REACHED_MAX_DEPTH
    i₋ = i₊ = 0
    while depth < max_depth
        is_forward, directions = DynamicHMC.next_direction(directions)
        sp, t′, v′ = DynamicHMC.adjacent_tree(
            sp, rng, trajectory, is_forward ? z₊ : z₋, is_forward ? i₊ : i₋, depth, is_forward
        )
        v = DynamicHMC.combine_visited_statistics(trajectory, v, v′)

        # invalid adjacent tree: stop
        t′ isa DynamicHMC.InvalidTree && (termination = t′; break)

        # extract information from adjacent tree
        ζ′, ω′, τ′, z′, i′ = t′

        # update edges and combine proposals
        if is_forward
            z₊, i₊ = z′, i′
        else
            z₋, i₋ = z′, i′
        end

        # tree has doubled successfully
        ζ, ω = DynamicHMC.combine_proposals_and_logweights(
            rng, trajectory, ζ, ζ′, ω, ω′, is_forward, true
        )
        depth += 1

        # when the combined tree is turning, stop
        τ = DynamicHMC.combine_turn_statistics_in_direction(trajectory, τ, τ′, is_forward)
        DynamicHMC.is_turning(trajectory, τ) && (termination = DynamicHMC.InvalidTree(i₋, i₊); break)
    end
    ζ, v, termination, depth
end

function sample_tree(sp::StackPointer, rng, options::TreeOptionsNUTS, H::Hamiltonian,
                          Q::EvaluatedLogDensity, ϵ;
                          p0 = nothing, directions = rand(rng, Directions))
    @unpack max_depth, min_Δ, turn_statistic_configuration = options
    if p0 === nothing
        (sp, p) = rand_p(sp, rng, H.κ)
    else
        p = p0
    end
    z = DynamicHMC.PhasePoint(Q, p)
    trajectory = DynamicHMC.TrajectoryNUTS(
        H, logdensity(H, z), ϵ, min_Δ, turn_statistic_configuration
    )
    ζ, v, termination, depth = DynamicHMC.sample_trajectory(sp, rng, trajectory, z, max_depth, directions)
    tree_statistics = DynamicHMC.TreeStatisticsNUTS(
        logdensity(H, ζ), depth, termination,
        DynamicHMC.acceptance_rate(v), v.steps, directions
    )
    ζ.Q, tree_statistics
end

@generated function pointer_vector_type(::AbstractProbabilityModel{D}, ::Type{T}) where {D,T}
    W = VectorizationBase.pick_vector_width(D, T)
    L = VectorizationBase.align(D, W)
    PtrVector{D,T,L,L,true}
end

function regularized_cov_block_quote(W::Int, T, reps_per_block::Int, stride::Int, mask_last::Bool = false, mask::Unsigned = 0xff)
    # loads from ptr_sample
    # stores in ptr_s² and ptr_invs
    # needs vNinv, mulreg, and addreg to be defined
    reps_per_block -= 1
    WT = sizeof(T)*W
    V = Vec{W,T}
    quote
        $([Expr(:(=), Symbol(:μ_,i), :(SIMDPirates.vload($V, ptr_sample + $(WT*i), $([mask for _ ∈ 1:((i==reps_per_block) & mask_last)]...)))) for i ∈ 0:reps_per_block]...)
        $([Expr(:(=), Symbol(:Σδ_,i), :(SIMDPirates.vbroadcast($V,zero($T)))) for i ∈ 0:reps_per_block]...)
        $([Expr(:(=), Symbol(:Σδ²_,i), :(SIMDPirates.vbroadcast($V,zero($T)))) for i ∈ 0:reps_per_block]...)
        for n ∈ 1:N-1
            $([Expr(:(=), Symbol(:δ_,i), :(SIMDPirates.vsub(SIMDPirates.vload(ptrS + $(WT*i) + n*$stride*$size_T),$(Symbol(:μ_,i))))) for i ∈ 0:reps_per_block]...)
            $([Expr(:(=), Symbol(:Σδ_,i), :(SIMDPirates.vadd($(Symbol(:δ_,i)),$(Symbol(:Σδ_,i))))) for i ∈ 0:reps_per_block]...)
            $([Expr(:(=), Symbol(:Σδ²_,i), :(SIMDPirates.vmuladd($(Symbol(:δ_,i)),$(Symbol(:δ_,i)),$(Symbol(:Σδ²_,i))))) for i ∈ 0:reps_per_block]...)
        end
        $([Expr(:(=), Symbol(:ΣδΣδ_,i), :(SIMDPirates.vmul($(Symbol(:Σδ_,i)),$(Symbol(:Σδ_,i))))) i ∈ 0:reps_per_block]..)
        $([Expr(:(=), Symbol(:s²nm1_,i), :(SIMDPirates.vfnmadd($(Symbol(:ΣδΣδ_,i)),vNinv,$(Symbol(:Σδ²_,i))))) for i ∈ 0:reps_per_block]..)
        $([Expr(:(=), Symbol(:regs²_,i), :(SIMDPirates.vmuladd($(Symbol(:s²nm1_,i)), vmulreg, vaddreg))) for i ∈ 0:reps_per_block]...)
        $([Expr(:(=), Symbol(:reginvs_,i), :(SIMDPirates.rsqrt($(Symbol(:regs²_,i))))) for i ∈ 0:reps_per_block]...)
        $([:(vstore!(ptr_s² + $(WT*i), $(Symbol(:regs²_,i)), $([mask for _ ∈ 1:((i==reps_per_block) & mask_last)]...))) for i ∈ 0:reps_per_block]...)
        $([:(vstore!(ptr_invs + $(WT*i), $(Symbol(:reginvs_,i)), $([mask for _ ∈ 1:((i==reps_per_block) & mask_last)]...))) for i ∈ 0:reps_per_block]...)
    end
end

function DynamicHMC.kinetic_energy(κ::DynamicHMC.GaussianKineticEnergy{Diagonal{<:PtrVector{M,T}}, p::PtrVector{M,T}, q) where {M,T}
    M⁻¹ = κ.M⁻¹.diag
    ke = zero(T)
    @inbounds @simd for m ∈ 1:M
        @fastmath ke += p[m] * M⁻¹[m] * p[m]
    end
    ke
end

@generated function DynamicHMC.GaussianKineticEnergy(sp::StackPointer, sample::AbstractMatrix{T}, λ::T, ::Val{M}) where {M,T}
    W, Wshift = VectorizationBase.pick_vector_width_shift(M, T)
    Wm1 = W-1
    V = Vec{W,T}
    MdW = (M + Wm1) >> W
    Wrem = M & Wm1
    size_T = sizeof(T)
    WT = size_T*W
    need_to_mask = Wrem > 0
    reps_per_block = 4
    blocked_reps, blocked_rem = divrem(MdW, reps_per_block)
    if (MdW % (blocked_reps + 1)) == 0
        blocked_reps, blocked_rem = blocked_reps + 1, 0
        reps_per_block = MdW ÷ blocked_reps
    end
    if (need_to_mask && blocked_rem == 0) || blocked_reps == 1
        blocked_rem += reps_per_block
        blocked_reps -= 1
    end
    AL = VectorizationBase.align(M*size_T)
    q = quote
        ptr_s² = pointer(sp, $T)
        regs² = PtrVector{$M,$T}(ptr_s²)
        ptr_invs = ptr_s² + $AL
        invs = PtrVector{$M,$T}(ptr_invs)

        N = size(sample, 2)
        ptr_sample = pointer(sample)
        @fastmath begin
            Ninv = one($T) / N
            mulreg = N / ((N + λ)*(N - 1))
            addreg = $(T(1e-3)) * λ / (N + λ)
        end
        vNinv = SIMDPirates.vbroadcast($V, Ninv)
        vmulreg = SIMDPirates.vbroadcast($V, mulreg)
        vaddreg = SIMDPirates.vbroadcast($V, addreg)        
    end
    if blocked_reps > 
        loop_block = regularized_cov_block_quote(W, T, reps_per_block, M, false)
        block_rep_quote = quote
            for _ ∈ 1:$blocked_reps
                $loop_block
                ptr_sample += $WT*$reps_per_block
                ptr_s² += $WT*$reps_per_block
                ptr_invs += $WT*$reps_per_block
            end
        end
        push!(q.args, block_rep_quote)
    end
    if blocked_rem > 0
        push!(q.args, regularized_cov_block_quote(W, T, blocked_rem, M, need_to_mask, VectorizationBase.mask(T,Wrem)))        
    end
    push!(q.args, :(sp + $(2AL), DynamicHMC.GaussianKineticEnergy(Diagonal(regs²), Diagonal(invs))))
    q
end

function regularize_sample_M⁻¹(::Type{Diagonal}, chain, λ)
    sample_mat_stride, leftover_stride = divrem(sizeof(DynamicHMC.NUTS_Transition{Tv,Tf}), sizeof(Tf))
    @assert leftover_stride == 0
    W, Wshift = VectorizationBase.pick_vector_width_shift(P, Tf)
    Wm1 = W - 1
    # rem = P & Wm1
    # @show P, W
    # L = (P + Wm1) & ~Wm1
    if P > 4W
        return quote
            N = length(sample)
            x̄ = sample_sum(sample)
            Σ = MutableFixedSizePaddedVector{$P,$Tf,$L,$L}(undef)
            columnwise_variance!(Σ, sample, x̄, 1/N)

            # @show Σ
            # Σ = Statistics.var(reshape(reinterpret(Float64, get_position.(sample)), ($L,N)), dims = 2)
        #    @inbounds for l ∈ 1:$L
        #        x̄[l] = Σ[l]
        #    end
            # return Diagonal(x̄)
            # # @show x̄
            # m̃ = quickmedian!(x̄)
            # m̃ = median!(x̄)
            # regmul = 1 - reg / N
            # regadd = m̃ * reg / N
            regmul = $Tf(N / (N+λ))
            regadd = $Tf(1e-3 * (λ / (N+λ)))

            # x̄ already escaped the function in median!, so we'll store there and return it
            @fastmath @inbounds @simd ivdep for l ∈ 1:$L
                x̄[l] = Σ[l] * regmul + regadd
            end
            # @show x̄
            # @show Diagonal(d).diag
            Diagonal(x̄)
        end

    end
    quote
        # $(Expr(:meta,:inline))
        N = length(sample)
        x̄ = sample_sum(sample)
        nN⁻¹ = - 1 / N
        Σ = MutableFixedSizePaddedMatrix{$P,$P,$Tf,$L,$(L*P)}(undef)
        @inbounds for p ∈ 1:$P
            x̄ₚ = nN⁻¹ * x̄[p]
            s = $L * (p-1)
            @vectorize $Tf for l ∈ 1:$L
                Σ[l + s] = x̄[l] * x̄ₚ
            end
        end
        # @inbounds for i ∈ 1:$(L*P)
        #     out[i] = zero(T)
        # end
        # ConstantFixedSizePaddedMatrix(out)
        $(sample_crossprod_quote(:N,Tf,P,L,sample_mat_stride))
        m = MutableFixedSizePaddedVector{$P,$Tf,$L,$L}(undef)
        @inbounds for p ∈ 1:$P
            m[p] = Σ[p,p]
        end
        off_diag_reg = (1 - reg / N) / (N - 1)
        diag_reg = median!(m) * reg / (N - reg)
        # diag_reg = quickmedian!(m) * reg / (N - 1)
        # off_diag_reg = (1 - reg) / (N - 1)
        @vectorize $Tf for i ∈ 1:$(L*P)
            Σ[i] = Σ[i] * off_diag_reg
        end
        @inbounds for p ∈ 1:$P
            Σ[p,p] += diag_reg
        end
        # @inbounds for i ∈ 1:$(L*P)
        #     out[i] *= (1/(N-1))
        # end
        # out
        ConstantFixedSizePaddedMatrix(Σ)
    end    

end

function warmup(sampling_logdensity, tuning::TuningNUTS{M}, warmup_state) where {M}
    @unpack rng, ℓ, algorithm, reporter = sampling_logdensity
    @unpack Q, κ, ϵ = warmup_state
    @unpack N, stepsize_adaptation, λ = tuning
    
    PV = pointer_vector_type(ℓ, T)
    L = PaddedMatrices.full_length(PV)
    chain = Matrix{typeof(Q.q)}(undef, L, N)
    chain_ptr = pointer(chain)
    tree_statistics = Vector{DynamicHMC.TreeStatisticsNUTS}(undef, N)
    H = Hamiltonian(κ, ℓ)
    ϵ_state = DynamicHMC.initial_adaptation_state(stepsize_adaptation, ϵ)
    ϵs = Vector{Float64}(undef, N)
    mcmc_reporter = DynamicHMC.make_mcmc_reporter(reporter, N; tuning = M ≡ Nothing ? "stepsize" :
                                       "stepsize and $(M) metric")
    for i in 1:N
        ϵ = current_ϵ(ϵ_state)
        ϵs[i] = ϵ
        Q, stats = DynamicHMC.sample_tree(rng, algorithm, H, Q, ϵ)
        copyto!( PV( chain_ptr ), Q.q ); chain_ptr += L*sizeof(T)
        tree_statistics[i] = stats
        ϵ_state = DynamicHMC.adapt_stepsize(stepsize_adaptation, ϵ_state, stats.acceptance_rate)
        DynamicHMC.report(mcmc_reporter, i; ϵ = round(ϵ; sigdigits = REPORT_SIGDIGITS))
    end
    if M ≢ Nothing
        # κ = GaussianKineticEnergy(regularize_M⁻¹(sample_M⁻¹(M, chain), λ))
        κ = DynamicHMC.GaussianKineticEnergy(regularize_sample_M⁻¹(M, chain, λ))
        DynamicHMC.report(mcmc_reporter, "adaptation finished", adapted_kinetic_energy = κ)
    end
    ((chain = chain, tree_statistics = tree_statistics, ϵs = ϵs),
    DynamicHMC.WarmupState(Q, κ, DynamicHMC.final_ϵ(ϵ_state)))
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
        Q, tree_statistics[i] = sample_tree(sp, rng, sampler_options, H, Q, ϵ)
        chain[:,i] .= Q.q
        report(mcmc_reporter, i)
    end
    (chain = chain, tree_statistics = tree_statistics)
end









