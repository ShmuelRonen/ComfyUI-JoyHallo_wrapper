import os,sys
import folder_paths
import os.path as osp
import time
import shutil
import torchaudio
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from .inference import load_config, inference_process
import torchvision.io as io
import torch

# Define paths
now_dir = osp.dirname(osp.abspath(__file__))
CONFIG_FILE = osp.join(now_dir, "inference.yaml")
JOY_DIR = osp.join(folder_paths.models_dir, "JOY")
PRETRAINED_DIR = osp.join(JOY_DIR, "HALLO")
WAV2VEC_DIR = osp.join(PRETRAINED_DIR, "chinese-wav2vec2-base")
AUDIO_SEPARATOR_DIR = osp.join(PRETRAINED_DIR, "audio_separator")
BASE_MODEL_PATH = osp.join(PRETRAINED_DIR, "stable-diffusion-v1-5")
VAE_MODEL_PATH = osp.join(PRETRAINED_DIR, "sd-vae-ft-mse")
JOYHALLO_PATH = osp.join(PRETRAINED_DIR, "JoyHallo-v1")

def cleanup_temp_files(base_path, mlruns_path=None):
    """Clean up temporary files and folders"""
    # Clean up temporary files
    temp_files = [
        "0_refimg_audioTEMP_MPY_wvf_snd.mp3",
        "0_refimg_audio.mp4",
        "0_refimg_audio.mp4.temp.mp4"
    ]
    
    for temp_file in temp_files:
        temp_path = osp.join(base_path, temp_file)
        try:
            if osp.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Could not remove temp file {temp_path}: {str(e)}")
    
    # Clean up mlruns directory contents if specified
    if mlruns_path and osp.exists(mlruns_path):
        try:
            # Clean all contents except .trash directory
            for item in os.listdir(mlruns_path):
                item_path = os.path.join(mlruns_path, item)
                if item != '.trash' and item != 'meta.yaml':
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            
            # Clean .trash directory contents but keep the directory
            trash_dir = os.path.join(mlruns_path, '.trash')
            if os.path.exists(trash_dir):
                for item in os.listdir(trash_dir):
                    item_path = os.path.join(trash_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
        except Exception as e:
            print(f"Warning: Could not clean mlruns directory: {str(e)}")

class JoyHallo_wrapper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "audio":("AUDIO",),
                "image":("IMAGE",),
                "inference_steps":("INT",{
                    "default":40
                }),
                "cfg_scale":("FLOAT",{
                    "default":3.5
                }),
                "if_fp8":("BOOLEAN",{
                    "default":False,
                }),
                "seed":("INT",{
                    "default":42
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "gen_video"
    CATEGORY = "JoyHallo_wrapper"

    def gen_video(self, audio, image, inference_steps, cfg_scale, if_fp8, seed):
        config = load_config(CONFIG_FILE)
        config.inference_steps = inference_steps
        config.cfg_scale = cfg_scale
        config.seed = seed
        
        # Download base model and dependencies
        if not osp.exists(osp.join(BASE_MODEL_PATH, "unet", "diffusion_pytorch_model.safetensors")):
            print(f"Downloading base model to {PRETRAINED_DIR}")
            snapshot_download(repo_id="fudan-generative-ai/hallo",
                              local_dir=PRETRAINED_DIR,
                              ignore_patterns=["*et.pth"])

        # Download wav2vec model with config
        if not osp.exists(osp.join(WAV2VEC_DIR, "chinese-wav2vec2-base-fairseq-ckpt.pt")) or \
           not osp.exists(osp.join(WAV2VEC_DIR, "config.json")):
            print(f"Downloading wav2vec model to {WAV2VEC_DIR}")
            snapshot_download(
                repo_id="TencentGameMate/chinese-wav2vec2-base",
                local_dir=WAV2VEC_DIR,
                local_dir_use_symlinks=False
            )

        # Download JoyHallo model if needed
        if not osp.exists(osp.join(JOYHALLO_PATH, "net.pth")):
            print(f"Downloading JoyHallo model to {JOYHALLO_PATH}")
            snapshot_download(
                repo_id="jdh-algo/JoyHallo-v1",
                local_dir=JOYHALLO_PATH,
                local_dir_use_symlinks=False
            )

        # Set up all config paths
        config.wav2vec_config.model_path = WAV2VEC_DIR
        config.audio_separator.model_path = osp.join(AUDIO_SEPARATOR_DIR, "Kim_Vocal_2.onnx")
        config.base_model_path = BASE_MODEL_PATH
        config.vae_model_path = VAE_MODEL_PATH
        config.face_analysis_model_path = osp.join(PRETRAINED_DIR, "face_analysis")
        config.mm_path = osp.join(PRETRAINED_DIR, "motion_module/mm_sd_v15_v2.ckpt")
        config.output_dir = folder_paths.get_output_directory()
        config.audio_ckpt_dir = osp.join(JOYHALLO_PATH, "net.pth")
        
        img_np = image.numpy()[0] * 255
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        save_dir = osp.join(config.output_dir, config.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        ref_img_path = osp.join(save_dir, "refimg.jpg")
        img_pil.save(ref_img_path)

        audio_path = osp.join(save_dir, "audio.wav")
        torchaudio.save(audio_path, audio['waveform'].squeeze(0), audio['sample_rate'])
        config.ref_img_path = [ref_img_path]
        config.audio_path = [audio_path]
        config.data.train_meta_paths = [osp.join(now_dir, "inference.json")]
        config.if_fp8 = if_fp8
     
        # Define output paths before try block
        tmp_output_file = osp.join(save_dir, "0_refimg_audio.mp4")
        
        try:
            # Add default fps to config
            if not hasattr(config, 'video'):
                config.video = {}
            config.video['fps'] = 25.0  # Set default fps
            
            # Run inference process
            inference_process(config)
            
            if not osp.exists(tmp_output_file):
                raise RuntimeError(f"Inference failed to generate output file: {tmp_output_file}")

            # Read video with default fps
            vframes, _, _ = io.read_video(tmp_output_file, pts_unit='sec')
            
            # Convert frames to float and normalize
            processed_frames = vframes.float() / 255.0  # [T, H, W, C]
            
            # Format frames for compatibility
            if len(processed_frames.shape) == 3:  # Single frame
                processed_frames = processed_frames.unsqueeze(0)
                
            # Make sure it's in [T, H, W, C] format for video combine
            if processed_frames.shape[1] == 3:  # If in [T, C, H, W] format
                processed_frames = processed_frames.permute(0, 2, 3, 1)  # Convert to [T, H, W, C]
                
            # Ensure RGB output
            if processed_frames.shape[-1] == 4:  # If RGBA
                processed_frames = processed_frames[..., :3]  # Convert to RGB
            
            return (processed_frames, audio)

        except Exception as e:
            print(f"Error during video generation: {str(e)}")
            raise
            
        finally:
            # Clean up temp files
            cleanup_files = [tmp_output_file, ref_img_path, audio_path]
            for file in cleanup_files:
                try:
                    if osp.exists(file):
                        os.remove(file)
                except Exception as cleanup_error:
                    print(f"Warning: Error during cleanup: {str(cleanup_error)}")
            
            # Clean up additional temporary files and folders
            comfy_root = osp.dirname(osp.dirname(osp.dirname(now_dir)))
            cleanup_temp_files(save_dir)

NODE_CLASS_MAPPINGS = {
    "JoyHallo_wrapper": JoyHallo_wrapper
}