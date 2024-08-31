import sys
import os
from pathlib import Path

# Add the StableCascade directory to the Python path
stable_cascade_path = Path(__file__).parent / "third_party" / "StableCascade"
sys.path.append(str(stable_cascade_path))

import gradio as gr
from inference.utils import *

def infer(un, deux, trois):
    return "quatre"

gr.Interface(
    fn = infer,
    inputs=[gr.Textbox(label="style description"), gr.Image(label="Ref Style File", type="filepath"), gr.Textbox(label="caption")],
    outputs=[gr.Textbox()]
).launch()