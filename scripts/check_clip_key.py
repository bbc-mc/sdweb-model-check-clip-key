import os
import torch
import safetensors.torch

import gradio as gr

from modules import sd_models, script_callbacks


def on_ui_tabs():
    with gr.Blocks() as main_block:
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=6):
                    dd_model_check_clip_key = gr.Dropdown(label="Model A", choices=sd_models.checkpoint_tiles())
                    txt_model_path = gr.Text(label="Model Path", value="",
                                            placeholder="Input full-filepath of model. Keep empty if not needed.")
                with gr.Column(scale=1):
                    btn_reload_model = gr.Button("Reload")
                    btn_check = gr.Button("Check", variant="primary")
            with gr.Row():
                html_output = gr.HTML(elem_id="html_check_clip_key")

        #
        # Events
        #
        def run_check(dd_model, txt_model_path):
            _ret_data = check(dd_model, txt_model_path)
            _ret_html = "" if "Model Name" not in _ret_data else f"<h1>Model Name : {_ret_data['Model Name']}</h1>"
            for cat in ["current_model", "after_cast", "result"]:
                _ret_html += f"<h1>{cat}</h1>"
                _ret_html += "<table>" + "<tr>"
                for k, v in _ret_data.items():
                    if cat in k:
                        _ret_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
                _ret_html += "</table>"
            return _ret_html
        btn_check.click(
            fn=run_check,
            inputs=[dd_model_check_clip_key, txt_model_path],
            outputs=[html_output]
        )

        def onclick_reload_model():
            sd_models.list_models()
            return gr.update(choices=sd_models.checkpoint_tiles())
        btn_reload_model.click(
            fn=onclick_reload_model,
            inputs=[],
            outputs=[dd_model_check_clip_key]
        )

    # return required as (gradio_component, title, elem_id)
    return (main_block, "CLIP Key Check", "clip_key_check"),

# on_UI
script_callbacks.on_ui_tabs(on_ui_tabs)


def check(dd_model, txt_model_path):
    _ret_html = {}

    sd_model = None
    if txt_model_path != "" and os.path.exists(txt_model_path):
        model = load_model(txt_model_path)
        _ret_html.update({"Model Name": os.path.basename(txt_model_path)})
    elif dd_model != "":
        model_info = sd_models.get_closet_checkpoint_match(dd_model)
        if model_info != "":
            model = load_model(model_info.filename)
            _ret_html.update({"Model Name": model_info.title})

    if model is None:
        return "ERROR: model can't load."

    if "state_dict" not in model:
        print("no state_dict. direct model.")
        sd_model = model
    else:
        sd_model = model.pop("state_dict", model)
        if "state_dict" in sd_model:
            sd_model.pop("state_dict", None)
    del model

    if KEY not in sd_model:
        print("This model dosen't have 'position_ids' key")
        _ret_html.update({"error_msg": "This model dosen't have 'position_ids' key"})
        return _ret_html

    tsr_current_key_data = sd_model[KEY]
    #print("# current data is:")
    _ret_html.update({"current_model_type": type(tsr_current_key_data)})
    _ret_html.update({"current_model_size": tsr_current_key_data.size()})
    _ret_html.update({"current_model_dtype": tsr_current_key_data.dtype})
    _ret_html.update({"current_model_value": tsr_current_key_data.tolist()})

    #print("# == if changed to torch.int64 ==")
    tsr_cast_to_int = tsr_current_key_data.to(torch.int64)
    _ret_html.update({"after_cast_type" : type(tsr_cast_to_int)})
    _ret_html.update({"after_cast_size" : tsr_cast_to_int.size()})
    _ret_html.update({"after_cast_dtype": tsr_cast_to_int.dtype})
    _ret_html.update({"after_cast_value": tsr_cast_to_int.tolist()})

    tsr_int_index = torch.tensor([[
        0,1,2,3,4,5,6,7,8,9,
        10,11,12,13,14,15,16,17,18,19,
        20,21,22,23,24,25,26,27,28,29,
        30,31,32,33,34,35,36,37,38,39,
        40,41,42,43,44,45,46,47,48,49,
        50,51,52,53,54,55,56,57,58,59,
        60,61,62,63,64,65,66,67,68,69,
        70,71,72,73,74,75,76
        ]],
        dtype=torch.int64
        )

    # compare
    tsr_eq = torch.eq(tsr_cast_to_int, tsr_int_index)
    _ret_html.update({"result_compare": tsr_eq.tolist()})

    _all_number = []
    _false_number = []
    _missing_number = []
    for k in range(77):
        if not tsr_eq[0, k].item():
            _false_number.append(k)
        _all_number.append(tsr_cast_to_int[0, k].item())
    for i in range(77):
        if i not in _all_number:
            _missing_number.append(i)
    if len(_false_number) > 0:
        _ret_html.update({"result_corrupt_token_indexes": f"{_false_number}"})
    if len(_missing_number) > 0:
        _ret_html.update({"result_missing_token_numbers": f"{_missing_number}"})

    print("done.")
    return _ret_html

chckpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

def transform_checkpoint_dict_key(k):
    for text, replacement in chckpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]
    return k

KEY = "cond_stage_model.transformer.text_model.embeddings.position_ids"

def load_model(filepath):

    print(f"loading ... {os.path.basename(filepath)}")
    _, extension = os.path.splitext(filepath)
    if extension.lower() == ".safetensors":
        pl_sd = safetensors.torch.load_file(filepath, device="cpu")
    else:
        pl_sd = torch.load(filepath, map_location="cpu")

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)
        if new_key is not None:
            sd[new_key] = v
    pl_sd.clear()
    pl_sd.update(sd)
    return pl_sd
