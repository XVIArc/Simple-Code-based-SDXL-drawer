import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)

model_path = r"C:\Users\cfzjl\Desktop\SD\ComfyUI\models\checkpoints\JANKUV5NSFWTrainedNoobai_v50.safetensors"
# 这里用的是我本地的模型。

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = None
last_error = None

# 1) 优先尝试按 SDXL 加载
try:
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,  # 禁用安全器时常需同时禁用 feature_extractor
    )
    is_sdxl = True
except Exception as e:
    last_error = e
    is_sdxl = False

# 2) 若 SDXL 失败，则回退为 SD1.5/2.x
if pipe is None:
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
    )

# 调度器换为 DPM-Solver（更快/更稳）
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 省显存与稳定性优化（按需）
if device == "cuda":
    try:
        pipe.enable_model_cpu_offload()  # 优先选这个（自动把未用模块放CPU）
    except Exception:
        pipe.enable_attention_slicing()  # 不支持时退而求其次
else:
    pipe.enable_attention_slicing()


try:
    pipe.vae.enable_tiling()
except Exception:
    pass

pipe = pipe.to(device)

print("使用本地模型加载完成！")

# 正反提示词
prompt = ""
negative_prompt = ""

if is_sdxl:
#一般来说这里是仔细的参数，不过作为初次尝试，暂时不多要求，不过这个CFG不要太高，这样已经差不多了，步数也是，24~32即可
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=24,
        guidance_scale=6.0,
    )
else:
    # 非 SDXL（如 SD1.5/2.x）
    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=24,
        guidance_scale=6.0,
        # height=512, width=512,
    )

result.images[0].save("output.png")

