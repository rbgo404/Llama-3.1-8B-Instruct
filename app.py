from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import os

class InferlessPythonModel:
    def initialize(self):
        
        HF_TOKEN = os.getenv("HF_TOKEN")
        VOLUME_NFS = os.getenv("VOLUME_NFS")
        
        if os.path.exists(VOLUME_NFS + "/generation_config.json") == False :
            snapshot_download(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            local_dir=VOLUME_NFS,
            token=HF_TOKEN)
        
        self.llm = LLM(model=VOLUME_NFS,dtype="float16")
        self.tokenizer = AutoTokenizer.from_pretrained(VOLUME_NFS)

    def infer(self, inputs):
        prompts = inputs["prompt"]
        temperature = inputs.get("temperature",0.7)
        top_p = inputs.get("top_p",0.1)
        repetition_penalty = inputs.get("repetition_penalty",1.18)
        top_k = inputs.get("top_k",40)
        max_tokens = inputs.get("max_tokens",256)

        sampling_params = SamplingParams(temperature=temperature,top_p=top_p,repetition_penalty=repetition_penalty,
                                         top_k=top_k,max_tokens=max_tokens)
        input_text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompts}], tokenize=False)
        result = self.llm.generate(input_text, sampling_params)
        result_output = [output.outputs[0].text for output in result]

        return {'result': result_output[0]}

    def finalize(self):
        self.llm = None