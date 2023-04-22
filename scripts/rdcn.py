import os
import copy
from random import choice
from glob import glob

import gradio as gr
import numpy as np
from PIL import Image

import modules
from modules import processing, shared, images, devices
from modules.processing import Processed
from modules.shared import opts, state

def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}

class Script(modules.scripts.Script):
    def __init__(self):
        self.original_scripts = None
        self.original_scripts_always = None
        
    def title(self):
        return "random controlnet"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        image_detectors = []
        
        with gr.Group():
            control_net_info = gr.HTML('<br><p style="margin-bottom:0.75em">T2I Control Net random image process</p>', visible=True)
            cn_models_num = shared.opts.data.get("control_net_max_models_num", 1)
            for n in range(cn_models_num):
                cn_image_detect_folder = gr.Textbox(label=f"{n} Control Model Image Random Folder(Using glob)", elem_id=f"{n}_cn_image_detector", value='',show_label=True, lines=1, placeholder="search glob image folder and file extension. ex ) - ./base/**/*.png", visible=True)
                image_detectors.append(cn_image_detect_folder)

        return image_detectors

    def run(self, p, *args):
        args_list = [*args]
        random_controlnet_list = args_list[:]
        
        initial_info = []
        initial_prompt = []
        initial_negative = []
        batch_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True
        
        if self.original_scripts is None: self.original_scripts = p.scripts.scripts.copy()
        else: 
            if len(p.scripts.scripts) != len(self.original_scripts): p.scripts.scripts = self.original_scripts.copy()
        if self.original_scripts_always is None: self.original_scripts_always = p.scripts.alwayson_scripts.copy()
        else: 
            if len(p.scripts.alwayson_scripts) != len(self.original_scripts_always): p.scripts.alwayson_scripts = self.original_scripts_always.copy()
        p.scripts.scripts = [x for x in p.scripts.scripts if os.path.basename(x.filename) not in [__file__]]
        
        controlnet = [x for x in p.scripts.scripts if os.path.basename(x.filename) in ['controlnet.py']]
        assert len(controlnet) > 0, 'Do not find controlnet, please install controlnet or disable random control net option'
        controlnet = controlnet[0]
        controlnet_args = p.script_args[controlnet.args_from:controlnet.args_to]
        controlnet_search_folders = random_controlnet_list.copy()
        controlnet_image_files = []
        for con_n, conet in enumerate(controlnet_args):
            files = []
            if conet.enabled:
                if '**' in controlnet_search_folders[con_n]:
                    files = glob(controlnet_search_folders[con_n], recursive=True)
                else:
                    files = glob(controlnet_search_folders[con_n])
            controlnet_image_files.append(files.copy())
                
        output_images = []
        state.job = 'T2I Generate'
        state.job_count = 0
        state.job_count += batch_count
        for n in range(batch_count):
            devices.torch_gc()
            cn_file_paths = []
            print(f"Processing initial image for output generation {n + 1} (Generate).")
            p.seed = -1
            for con_n, conet in enumerate(controlnet_args):
                cn_file_paths.append([])
                if len(controlnet_image_files[con_n]) > 0:
                    cn_file_paths[con_n].append(choice(controlnet_image_files[con_n]))
                    cn_image = Image.open(cn_file_paths[con_n][0])
                    cn_np = np.array(cn_image)
                    if cn_image.mode == 'RGB':
                        cn_np = np.concatenate([cn_np, 255*np.ones((cn_np.shape[0], cn_np.shape[1], 1), dtype=np.uint8)], axis=-1)
                    cn_np_image = copy.deepcopy(cn_np[:,:,:3])
                    cn_np_mask = copy.deepcopy(cn_np)
                    cn_np_mask[:,:,:3] = 0
                    conet.image = {'image':cn_np_image,'mask':cn_np_mask}
            processed = processing.process_images(p)
            for image_index, image in enumerate(processed.images):
                if image_index == processed.batch_size: break
                initial_info.append(processed.info)
                initial_info[n * processed.batch_size + image_index] += ', ' + ', '.join([f'ControlNet Random Image : {x}' for x in enumerate(cn_file_paths) if len(x) > 0])
                initial_prompt.append(processed.all_prompts[0])
                initial_negative.append(processed.all_negative_prompts[0])
                output_images.append(image)
            for image_index in range(processed.batch_size):
                images.save_image(output_images[n * processed.batch_size + image_index], p.outpath_samples, "", p.all_seeds[image_index], initial_prompt[n * processed.batch_size + image_index], opts.samples_format, info=initial_info[n * processed.batch_size + image_index], p=p)
        p.scripts.scripts = self.original_scripts.copy()
        p.scripts.alwayson_scripts = self.original_scripts_always.copy()
        return Processed(p, output_images, p.seed, initial_info[0], all_prompts=initial_prompt, all_negative_prompts=initial_negative, infotexts=initial_info)