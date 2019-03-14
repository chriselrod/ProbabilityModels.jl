
# using MacroTools, DiffRules
# using MacroTools: @capture, postwalk, prewalk, @q, striplines
# DiffRules.hasdiffrule(:Base, :exp, 1)
# DiffRules.diffrule(:Base, :exp, :x)
# DiffRules.diffrule(:Base, :^, :x, :y)


const SPECIAL_DIFF_RULES = Dict{Symbol,Function}()
function exp_diff_rule!(first_pass, second_pass, tracked_vars, out, A)
    a = A[1]
    push!(first_pass.args, :($out = SLEEFPirates.exp($a)))
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
    push!(first_pass.args, :($out = SLEEFPirates.log($a)))
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
