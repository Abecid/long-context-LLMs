from transformers import BertModel, BertTokenizer

def main():
    # Load the pre-trained BERT model
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the input text
    input_text = "Hello, how are you?"
    tokens = tokenizer.encode_plus(input_text, add_special_tokens=True, return_tensors='pt')

    # Pass the tokenized input to the model and retrieve the outputs
    outputs = model(**tokens)

    # Access the attention scores
    attention_scores = outputs.attentions

    # Print the attention scores
    for layer, layer_attention in enumerate(attention_scores):
        print(f"Layer {layer + 1} attention scores:")
        for head, head_attention in enumerate(layer_attention[0]):
            print(f"Head {head + 1}: {head_attention}")
        
if __name__ == "__main__":
    main()