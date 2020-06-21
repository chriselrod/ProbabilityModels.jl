

function model(model_name, model_expr, mod = Base)
    model_object = read_model(model_expr, mod)
    model_object_name = Symbol(model_name, "#OBJECT#DEFINITION#")
    quote
        struct $model_name{T <: NamedTuple}
            data::T
        end
        const $model_object_name = $model_object
        @generated function ProbabilityModels.PaddedMatrices.logdensity(
            var"###DATA###"::$model_name{var"##PROB#MODEL#NAMED#TUPLE#TYPE##"}, var"#θ#"
        ) where {var"##PROB#MODEL#NAMED#TUPLE#TYPE##"}
            var"#__m__#" = deepcopy($model_object_name)
            ProbabilityModels.preprocess!(var"#__m__#", var"##PROB#MODEL#NAMED#TUPLE#TYPE##")
            q = ProbabilityModels.ReverseDiffExpressionsBase.lower(var"#__m__#")
            pushfirst!(q.args, Expr(:(=), var"#DATA#", Expr(:(.), var"###DATA###", QuoteNode(:data))))
            q
        end
        @generated function ProbabilityModels.PaddedMatrices.∂logdensity!(
            var"#∇#", var"###DATA###"::$model_name{var"##PROB#MODEL#NAMED#TUPLE#TYPE##"}, var"#θ#"
        ) where {var"##PROB#MODEL#NAMED#TUPLE#TYPE##"}
            var"#__m__#" = deepcopy($model_object_name)
            ProbabilityModels.preprocess!(var"#__m__#", var"##PROB#MODEL#NAMED#TUPLE#TYPE##")
            var"#__∂m__#" = ProbabilityModels.ReverseDiffExpressionsBase.differentiate(var"#__m__#")
            q = ProbabilityModels.ReverseDiffExpressionsBase.lower(var"#__∂m__#")
            pushfirst!(q.args, Expr(:(=), var"#DATA#", Expr(:(.), var"###DATA###", QuoteNode(:data))))
            q
        end
    end
end

macro model(model_name, model_expr)
    model(model_name, model_expr, __module__) |> esc
end













