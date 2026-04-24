#
# Modified from LLaVA/predict.py
# Please see ACKNOWLEDGEMENTS for details about LICENSE
#
import cv2
import torch
from PIL import Image
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

model_path = "checkpoints/llava-fastvithd_1.5b_stage3"
model, tokenizer, image_processor, input_ids = None, None, None, None

# Load model
def load_model():
    global model, tokenizer, image_processor, input_ids
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=model_path, model_base=None, model_name=model_name, device="cuda:0")

    # Construct prompt
    qs = "Describe the scene."
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(torch.device("cuda:0"))

def generate_caption(cv_image):
    # if model is not loaded
    if model is None or tokenizer is None or image_processor is None or input_ids is None:
        load_model()

    # Load and preprocess image
    image = Image.fromarray(cv_image)
    image_tensor = process_images([image], image_processor, model.config)[0]
    # Run inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            max_new_tokens=256,
            use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs

if __name__=="__main__":
    cv_image = cv2.imread('sample_images/car.jpg')
    print(generate_caption(cv_image=cv_image))
