from dataclasses import dataclass

output_path = "output/"


@dataclass
class Const:
    input_col = "input"
    label_col = "label"
    preds_col = "predictions"
    probs_col = "probabilities"

    output_pipelines_path = output_path + "pipelines"
    output_runs_path = output_path + "runs"

    final_eval_name = "final"


class LogConst:
    plugin_prefix = "    ┣━━┯ 🔌  Plugin"
