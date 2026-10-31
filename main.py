from ollama import chat

def ask_llm(question):
    response = chat(
        model="gemma3:1b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là AI Assistant chuyên giải thích "
                    "các kiến thức về CUDA và GPU. "
                    "Trả lời bằng tiếng Việt, dễ hiểu."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        options={
            "temperature": 0.2
        }
    )

    return response.message.content


if __name__ == "__main__":
    question = "Shared Memory trong CUDA là gì?"

    answer = ask_llm(question)

    print("\nQUESTION:")
    print(question)

    print("\nLLM RESPONSE:")
    print(answer)
