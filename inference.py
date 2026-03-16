import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_vife import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import WanModel

# ========== ViFeEdit: 2D Spatial Attention Initialization ==========
def initialize_spatial_attn_from_3d(model: WanModel):
    """
    Initializes the 2D spatial attention blocks with weights from the 
    pretrained 3D self-attention block and ensures the negative block's
    output is inverted. 
    """
    print("Initializing custom 2D spatial attention blocks...")
    with torch.no_grad():
        for i, block in enumerate(model.blocks):
            # Check if the custom modules exist before trying to initialize
            if not hasattr(block, 'spatial_attn_pos') or not hasattr(block, 'spatial_attn_neg'):
                print(f"  - Warning: DiT Block {i} does not have custom spatial attention modules. Skipping.")
                continue

            state_dict_3d = block.self_attn.state_dict()
            
            block.spatial_attn_pos.load_state_dict(state_dict_3d)
            block.spatial_attn_neg.load_state_dict(state_dict_3d)
            
            # Invert the weights of the output projection layer for the negative block
            block.spatial_attn_neg.o.weight.copy_(-block.spatial_attn_neg.o.weight)
            block.spatial_attn_neg.o.bias.copy_(-block.spatial_attn_neg.o.bias)
            
            print(f"  - DiT Block {i}: Initialized spatial_attn_pos and spatial_attn_neg.")
    print("Custom initialization complete.")
# ========== ViFeEdit: 2D Spatial Attention Initialization ==========


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
if hasattr(pipe, 'dit') and pipe.dit is not None:
    initialize_spatial_attn_from_3d(pipe.dit)

pipe.load_lora(pipe.dit, "/path/to/lora", alpha=1)

pipe.enable_vram_management()
vife_edit_video = VideoData("/path/to/source/video.mp4", height=480, width=832)

video = pipe(
    prompt="3D Chibi style, a woman in a pink jacket is walking in the street.", # for target prompt
    negative_prompt="Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down",
    vife_edit_video=vife_edit_video,
    input_video=vife_edit_video, denoising_strength=0.9, # optional: turn on for better consistency, but may reduce the edit strength
    seed=0, tiled=True,
    num_frames=81,
    num_inference_steps=50
)

save_video(video, "/path/to/output/video.mp4", fps=15, quality=5)