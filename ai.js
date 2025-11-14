import { HfInference } from "@huggingface/inference";

// System prompt explaining the assistant's role
const SYSTEM_PROMPT = `
You are a helpful assistant that receives a list of ingredients
and suggests a recipe they could make with some or all of those ingredients.
Format the response in markdown.
`;

// Create Hugging Face client (using Vite env token)
const hf = new HfInference(import.meta.env.VITE_HF_ACCESS_TOKEN);

/**
 * Generates a recipe based on the ingredients provided.
 * Uses chatCompletion to match the provider's supported conversational task.
 * @param {string[]} ingredientsArr - Array of ingredient names
 * @returns {Promise<string>} - Recipe text in markdown format
 */
export async function getRecipeFromMistral(ingredientsArr) {
    const ingredientsString = ingredientsArr.join(", ");

    try {
        const response = await hf.chatCompletion({
            model: "meta-llama/Llama-3.1-8B-Instruct",
            messages: [
                { role: "system", content: SYSTEM_PROMPT },
                { role: "user", content: `I have these ingredients: ${ingredientsString}. Give me a recipe!` },
            ],
            max_tokens: 512,          // Limit response length
            temperature: 0.7,         // Creativity in responses
        });

        // chatCompletion response has choices array
        return response.choices[0].message.content;

    } catch (err) {
        console.error("HF API Error:", err);
        return "Sorry, something went wrong while fetching the recipe.";
    }
}
