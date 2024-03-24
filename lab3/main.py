import streamlit as st
from transformers import pipeline, AutoTokenizer

# Load the pre-trained model and tokenizer
model = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model)
classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, top_k=3)

# Streamlit app
def main():
    st.title("Emotion Classifier App")
    st.write("### Sandikha Rahardi")
    st.write("### РИМ - 130908")

    text = st.text_input("Enter a sentence and let's classify its emotion!")
    clicked = st.button("Submit")

    if clicked:
        if text:
            model_outputs = classifier(text)

            # Prepare response JSON
            response = {
                "sentence": text,
                "predicted_emotion": model_outputs[0]
            }
            st.write("### Result :")
            st.json(response)

if __name__ == "__main__":
    main()
