function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(MacroTools, Symbol("#19#20")) && precompile(Tuple{getfield(MacroTools, Symbol("#19#20")),Float64})
    isdefined(MacroTools, Symbol("#21#22")) && precompile(Tuple{getfield(MacroTools, Symbol("#21#22")),Float64})
    isdefined(MacroTools, Symbol("#21#22")) && precompile(Tuple{getfield(MacroTools, Symbol("#21#22")),Int64})
    isdefined(MacroTools, Symbol("#21#22")) && precompile(Tuple{getfield(MacroTools, Symbol("#21#22")),LineNumberNode})
    isdefined(MacroTools, Symbol("#21#22")) && precompile(Tuple{getfield(MacroTools, Symbol("#21#22")),QuoteNode})
    precompile(Tuple{typeof(ProbabilityModels.generate_generated_funcs_expressions),Symbol,Expr})
end
