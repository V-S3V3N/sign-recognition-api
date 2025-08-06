from transformers import pipeline

# Load LLM only once at startup
sign2sentence = pipeline("text2text-generation", model="facebook/bart-base")

def llm_generate_sentence(predicted_signs: list[str]) -> str:
    input_str = " ".join(predicted_signs)
    output = sign2sentence(input_str, max_length=20, clean_up_tokenization_spaces=True)
    return output[0]["generated_text"]
