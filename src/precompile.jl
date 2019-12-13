function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    isdefined(MacroTools, Symbol("#21#22")) && precompile(Tuple{getfield(MacroTools, Symbol("#21#22")),LineNumberNode})
    precompile(Tuple{typeof(ProbabilityModels.drop_const_assignment!),Array{Any,1},Set{Symbol},Expr,Symbol,Expr,Bool})
    precompile(Tuple{typeof(ProbabilityModels.fmadd_dist_call),Symbol,Array{Any,1}})
    precompile(Tuple{typeof(ProbabilityModels.generate_generated_funcs_expressions),Symbol,Expr})
    precompile(Tuple{typeof(ProbabilityModels.no_fmadd_dist_call),Symbol,Array{Any,1}})
end
