#Recursive LLM with human-in-the-loop
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def recursive_llm(prompt, iteration=0, max_iterations=10):
    """Recursively calls the LLM until the human-in-the-loop confirms stopping."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a recursive AI improving responses iteratively."},
            {"role": "user", "content": prompt}
        ]
    )
    
    output = response.choices[0].message.content
    print(f"Iteration {iteration}: {output}\n")

    # Human-in-the-loop confirmation
    if iteration >= max_iterations or input("Continue? (y/n): ").strip().lower() != 'y':
        print("Final response accepted.")
        return output
    else:
        return recursive_llm(output, iteration + 1, max_iterations)

# Example usage
initial_prompt = "Explain recursion simply."
recursive_llm(initial_prompt)
