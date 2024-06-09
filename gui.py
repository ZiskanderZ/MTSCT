import gradio as gr
from main import forward
import os

css = """
h1 {
    text-align: center;
    display:block;
}

.container {
        height: 10vh;
}
"""
theme = gr.themes.Default(primary_hue=gr.themes.colors.sky)
radio_dict = {'Train and Test': 'train', 'Test params': 'test_params', 'Test model': 'test_model'}

demo = gr.Blocks(css=css, theme=theme)

def greet(mode, train_path, test_path, file_input_params, file_input_model, file_config, output_folder_path):

    mode = radio_dict[mode]
    metric = forward(mode, train_path, test_path, file_config, output_folder_path, file_input_model, file_input_params)

    if mode == 'train':
        output = (os.path.join(output_folder_path, 'TSCT_model.pt'), os.path.join(output_folder_path, f'{metric}.xlsx'), str(metric))
    elif mode == 'test_params':
        output = (os.path.join(output_folder_path, 'TSCT_model.pt'), None, str(metric))
    else:
        output = (None, None, str(metric))

    return output

def filter(choice):

    if choice == radio_bttns[0]:
        return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                gr.update(visible=True), gr.update(visible=True, value=None), gr.update(visible=True, value=None), gr.update(visible=True, value='')]
    elif choice == radio_bttns[1]:
        return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), \
                gr.update(visible=True), gr.update(visible=True, value=None), gr.update(visible=False, value=None), gr.update(visible=True, value='')]
    elif choice == radio_bttns[2]:
        return [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), \
                gr.update(visible=True), gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=True, value='')]

with demo:
    gr.Markdown('<h1 style="font-size:36px;">Time Series Classification Transformer</h1>')

    radio_bttns = ['Train and Test', 'Test params', 'Test model']
    radio = gr.Radio(radio_bttns, label='Run modes')
  
    output_folder_path = gr.Text(label="Output Folder", visible=False)
    with gr.Row():
        file_train = gr.File(label="Data Train", visible=False, elem_classes=['container'])
        file_test = gr.File(label="Data Test", visible=False, elem_classes=['container'])
    with gr.Row():
        file_input_params = gr.File(label="Input Params", visible=False, elem_classes=['container'])
        file_input_model = gr.File(label="Input Model", visible=False, elem_classes=['container'])
    file_config = gr.File(label="Config", visible=False, elem_classes=['container'])

    greet_btn = gr.Button("Run", variant='primary', visible=False)
    with gr.Row():
        file_output_model = gr.File(label="Output Model", visible=False, elem_classes=['container'])
        file_output_params = gr.File(label="Output Params", visible=False, elem_classes=['container'])
    output_text = gr.Text(label="Metric", visible=False)

    radio.change(filter, inputs=radio, outputs=[output_folder_path, file_train, file_test, file_input_params, file_input_model, file_config, \
                                                greet_btn, file_output_model, file_output_params, output_text])

    greet_btn.click(fn=greet, inputs=[radio, file_train, file_test, file_input_params, file_input_model, file_config, output_folder_path], 
                              outputs=[file_output_model, file_output_params, output_text])


demo.launch()
