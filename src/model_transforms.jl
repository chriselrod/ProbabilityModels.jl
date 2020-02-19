
function add_loads_or_constraint_transforms!(m::Model, variablesin)

end

function logdensity_model(modelin::Model, variabledescription)
    m = deepcopy(modelin)
    @unpack vars = m
    nvars = length(vars)
    θ = addvar!(m, Symbol("##θ##")) # parameter vector
    for i ∈ 1:nvars
        var = vars[i]
        descript = variabledescription[i]
        if istracked(descript)# load from Symbol("##θ##")
            # Plan is to define func `constrain` taking in type of the variable, model, and Symbol("##θ##"), returning that variable.

            var.initialized = false
        else# load from Symbol("##DATA##")
            
        end
    end
    
end


