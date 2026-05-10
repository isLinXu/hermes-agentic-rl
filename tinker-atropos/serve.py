"""
Standalone tinker inference server. No training, no atropos — just chat.

Usage:
    python serve.py                                    # default model
    python serve.py --model Qwen/Qwen3-30B-A3B        # specific model
    python serve.py --weights tinker://...path...      # from saved weights
    python serve.py --port 8001                        # custom port
"""

import argparse
import random
import time

import tinker
from tinker.types import ModelInput, SamplingParams
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from tinker_atropos.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    LogprobsRequest,
    LogprobsResponse,
    TokenLogprob,
)

app = FastAPI(title="Tinker Inference Server")

# Global state
sampling_client = None
tokenizer = None
model_name = None


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_name, "ready": sampling_client is not None}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if sampling_client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        prompt_text = tokenizer.apply_chat_template(
            messages_dict, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        model_input = ModelInput.from_ints(prompt_tokens)

        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop if request.stop else [],
        )

        result = await sampling_client.sample_async(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=request.n,
        )

        choices = []
        for i, sequence in enumerate(result.sequences):
            output_text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            choices.append(
                {
                    "message": {"role": "assistant", "content": output_text},
                    "index": i,
                    "finish_reason": "stop",
                }
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{random.randint(0, 999999)}",
            choices=choices,
            created=int(time.time()),
            model=model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    if sampling_client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        all_choices = []

        for prompt in prompts:
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            model_input = ModelInput.from_ints(prompt_tokens)

            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop if request.stop else [],
            )

            result = await sampling_client.sample_async(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=request.n,
            )

            for sequence in result.sequences:
                output_text = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
                all_choices.append(
                    {
                        "text": output_text,
                        "index": len(all_choices),
                        "finish_reason": "stop",
                    }
                )

        return CompletionResponse(
            id=f"cmpl-{random.randint(0, 999999)}",
            choices=all_choices,
            created=int(time.time()),
            model=model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/logprobs", response_model=LogprobsResponse)
async def logprobs(request: LogprobsRequest):
    if sampling_client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if request.input_ids is not None:
            token_ids = request.input_ids
        elif request.text is not None:
            token_ids = tokenizer.encode(request.text, add_special_tokens=False)
        else:
            raise HTTPException(status_code=400, detail="input_ids or text required")

        if len(token_ids) == 0:
            raise HTTPException(status_code=400, detail="Empty input")

        model_input = ModelInput.from_ints(token_ids)
        prompt_lps = await sampling_client.compute_logprobs_async(model_input)

        token_logprobs = []
        for i, token_id in enumerate(token_ids):
            lp = prompt_lps[i] if i < len(prompt_lps) and prompt_lps[i] is not None else 0.0
            token_text = tokenizer.decode([token_id]) if request.return_text else None
            token_logprobs.append(TokenLogprob(token_id=token_id, logprob=lp, token=token_text))

        return LogprobsResponse(logprobs=token_logprobs, num_tokens=len(token_ids))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    global sampling_client, tokenizer, model_name

    parser = argparse.ArgumentParser(description="Tinker inference server")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B", help="Model name")
    parser.add_argument("--weights", default=None, help="Saved weights path (tinker://...)")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    args = parser.parse_args()

    model_name = args.model
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded")

    service_client = tinker.ServiceClient()

    if args.weights:
        print(f"Loading weights from: {args.weights}")
        sampling_client = service_client.create_sampling_client(model_path=args.weights)
    else:
        print("Using base model weights")
        sampling_client = service_client.create_sampling_client(base_model=model_name)

    print("Sampling client ready")

    import uvicorn

    print(f"\nServing on http://0.0.0.0:{args.port}")
    print("  /v1/chat/completions  - OpenAI chat")
    print("  /v1/completions       - OpenAI completions")
    print("  /logprobs             - per-token logprobs")
    print("  /health               - health check")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
