import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("recall")
launch_gradio_widget(module)