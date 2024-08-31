import gradio as gr
from inference.utils import *

def infer(un, deux, trois):
    return "quatre"

gr.Interface(
    fn = infer,
    inputs=[gr.Textbox(label="style description"), gr.Image(label="Ref Style File", type="filepath"), gr.Textbox(label="caption")],
    outputs=[gr.Textbox()]
).launch()