import swifter
from sklearn.preprocessing import FunctionTransformer


def pipelinize(function, active=True):
    def list_comprehend_a_function(list_or_series):
        return list_or_series.swifter.apply(function)

    return FunctionTransformer(list_comprehend_a_function, validate=False)
