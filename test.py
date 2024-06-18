from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

generation_args = {
    "max_new_tokens": 600,
    "return_full_text": False,
    "temperature": 0.3,
    "do_sample": False,
}

json = pipe(
    [
        {
            "role": "system",
            "content": "You are a very helpful expert.",
        },
        {
            "role": "user",
            "content": "What is the capital of norway.",
        },
    ],
    **generation_args,
)

print(json[0]["generated_text"])
