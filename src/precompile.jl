function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(ProbabilityModels.generate_generated_funcs_expressions),Symbol,Expr})
end
