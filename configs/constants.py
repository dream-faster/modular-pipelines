from dataclasses import dataclass


@dataclass
class Const:
    input_col = "input"
    label_col = "label"
    preds_col = "predictions"
    probs_col = "probabilities"
    output_path = "output/pipelines"
