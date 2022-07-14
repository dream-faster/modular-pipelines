# from configs import Const
# from model.pipeline import Pipeline
# from model.huggingface import HuggingfaceModel

# from model.sklearn import SKLearnModel
# from configs.config import (
#     global_preprocess_config,
#     huggingface_config,
#     sklearn_config,
# )
# from model.pipeline import Pipeline
# from model.ensemble import Ensemble
# from model.data import DataSource, StrConcat, VectorConcat
# from model.transformations.predicitions_to_text import PredictionsToText

# TODO: This doesn't work, just an example!
# def simple_pipeline() -> Pipeline:
#     input_data = DataSource("input")

#     pipeline1 = Pipeline(
#         "pipeline1",
#         input_data,
#         [SKLearnModel("model1", sklearn_config), PredictionsToText()],
#     )

#     pipeline2 = Pipeline(
#         "pipeline2",
#         StrConcat([input_data, pipeline1]),
#         [SKLearnModel("model2", sklearn_config), PredictionsToText()],
#     )

#     pipeline3 = Pipeline(
#         "pipeline3",
#         StrConcat([input_data, pipeline2]),
#         [SKLearnModel("model3", sklearn_config), PredictionsToText()],
#     )

#     end_to_end_pipeline = Pipeline(
#         "end-to-end",
#         StrConcat([input_data, pipeline1, pipeline2, pipeline3]),
#         [SKLearnModel("decoder", sklearn_config)],
#     )

#     return end_to_end_pipeline
