# main.py

from text_generation import TextGenerationAI

if __name__ == "__main__":
    # Create the AI
    ai = TextGenerationAI(model_name="keras_rnn_model")
    try:
        with open("test_text.txt", "r", encoding="utf-8") as file:
            text_data = file.read()
            print(f"Loaded {len(text_data)} characters from test_text.txt")
    except FileNotFoundError:
        print("Error: test_text.txt file not found!")
        exit(1)

    print("Training the model (this might take a while)...")
    ai.train(text_data, epochs=5, batch_size=32)
    print("Training complete!")

    seed = "class MinimalRNNCell(Layer):"
    generated_code = ai.generate_text(
        seed_text=seed,
        length=200,
        temperature=0.3
    )

    print("\nGenerated Python Code:")
    print(generated_code)

    seed2 = "def get_initial_state"
    generated_code2 = ai.generate_text(
        seed_text=seed2,
        length=150,
        temperature=0.5
    )
    print("\nAnother Generated Sample:")
    print(generated_code2)