
# using MacroTools, DiffRules
# using MacroTools: @capture, postwalk, prewalk, @q, striplines
# DiffRules.hasdiffrule(:Base, :exp, 1)
# DiffRules.diffrule(:Base, :exp, :x)
# DiffRules.diffrule(:Base, :^, :x, :y)


const SPECIAL_DIFF_RULES = Dict{Symbol,Function}()
function exp_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    a = A[1]
    push!(first_pass.args, :($out = ProbabilityModels.SLEEFPirates.exp($a)))
    a ∈ tracked_vars || return nothing
    push!(tracked_vars, out)
    # ∂ = gensym(:∂)
    ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
    push!(first_pass.args, :($∂ = $out))
    pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) += $(Symbol("###seed###", out)) * $∂ ))
    nothing
end
SPECIAL_DIFF_RULES[:exp] = exp_diff_rule!
function log_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    a = A[1]
    push!(first_pass.args, :($out = ProbabilityModels.SLEEFPirates.log($a)))
    a ∈ tracked_vars || return nothing
    push!(tracked_vars, out)
    # ∂ = gensym(:∂)
    ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
    push!(first_pass.args, :($∂ = inv($a)))
    pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) += $(Symbol("###seed###", out)) * $∂ ))
    nothing
end
SPECIAL_DIFF_RULES[:log] = log_diff_rule!
function plus_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    push!(first_pass.args, :($out = +($(A...)) ))
    track_out = false
    adjout = Symbol("###seed###", out)
    for i ∈ eachindex(A)
        a = A[i]
        a ∈ tracked_vars || continue
        track_out = true
        pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) += $adjout ))
    end
    track_out && push!(tracked_vars, out)
    nothing
end
SPECIAL_DIFF_RULES[:+] = plus_diff_rule!
function minus_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    @assert length(A) == 2
    a₁ = A[1]
    a₂ = A[2]
    push!(first_pass.args, :($out = $a₁ - $a₂ ))
    adjout = Symbol("###seed###", out)
    a₁ ∈ tracked_vars && pushfirst!(second_pass.args, :( $(Symbol("###seed###", a₁)) += $adjout ))
    a₂ ∈ tracked_vars && pushfirst!(second_pass.args, :( $(Symbol("###seed###", a₂)) -= $adjout ))
    track_out = (a₁ ∈ tracked_vars) || (a₂ ∈ tracked_vars)
    track_out && push!(tracked_vars, out)
    nothing
end
SPECIAL_DIFF_RULES[:-] = minus_diff_rule!
function inv_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    a = A[1]
    if a ∉ tracked_vars
        push!(first_pass.args, :($out = inv($a)))
        return nothing
    end
    push!(tracked_vars, out)
    # ∂ = gensym(:∂)
    ∂ = Symbol("###adjoint###_##∂", out, "##∂", a, "##")
    push!(first_pass.args, :(($out, $∂) = ProbabilityModels.StructuredMatrices.∂inv($a)))
    pushfirst!(second_pass.args, :( $(Symbol("###seed###", a)) += $(Symbol("###seed###", out)) * $∂ ))
    nothing
end
SPECIAL_DIFF_RULES[:inv] = inv_diff_rule!


function itp_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    a_t = A[1]
    a_β = A[2]
    a_κ = A[3]
    @assert a_t ∉ tracked_vars
    if a_β ∈ tracked_vars
        push!(tracked_vars, out)
        if a_κ ∈ tracked_vars # both β and κ are tracked
            ∂β = Symbol("###adjoint###_##∂", out, "##∂", a_β, "##")
            ∂κ = Symbol("###adjoint###_##∂", out, "##∂", a_κ, "##")
            push!(first_pass.args, :(($out, $∂β, $∂κ) = ProbabilityModels.∂ITPExpectedValue∂β∂κ($a_1, $a_β, $a_κ)))
            pushfirst!(second_pass.args, :( $(Symbol("###seed###", a_κ)) += $(Symbol("###seed###", out)) * $∂κ ))
            pushfirst!(second_pass.args, :( $(Symbol("###seed###", a_β)) += $(Symbol("###seed###", out)) * $∂β ))
        else # only β is tracked
            ∂β = Symbol("###adjoint###_##∂", out, "##∂", a_β, "##")
            push!(first_pass.args, :(($out, $∂β) = ProbabilityModels.∂ITPExpectedValue∂β($a_1, $a_β, $a_κ)))
            pushfirst!(second_pass.args, :( $(Symbol("###seed###", a_β)) += $(Symbol("###seed###", out)) * $∂β ))
        end
    elseif a_κ ∈ tracked_vars # only κ is tracked
        push!(tracked_vars, out)
        ∂κ = Symbol("###adjoint###_##∂", out, "##∂", a_κ, "##")
        push!(first_pass.args, :(($out, $∂κ) = ProbabilityModels.∂ITPExpectedValue∂κ($a_1, $a_β, $a_κ)))
        pushfirst!(second_pass.args, :( $(Symbol("###seed###", a_κ)) += $(Symbol("###seed###", out)) * $∂κ ))
    else # none are in tracked_vars
        push!(first_pass.args, :($out = ProbabilityModels.∂ITPExpectedValue($a_1, $a_β, $a_κ)))
    end
    nothing
end
SPECIAL_DIFF_RULES[:ITPExpectedValue] = itp_diff_rule!
