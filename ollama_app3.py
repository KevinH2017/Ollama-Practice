# Describes each image in a folder and writes description to text file
import ollama, os

model = "llava:7b"

output_file = "./data/decoded_img3.txt"
open_file = open(output_file, "w")

prompt = """
Describe the image.
"""

input_dir = "./test_imgs/"
for i in os.listdir(input_dir):
    if (i.endswith(".PNG")):        # Add more .endswith() for each file type
        print("Reading:", i)
        input_image = input_dir+i

        # Send prompt to model and get the response, images only takes a list
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                images=[input_image]
            )    
            generated_text = response.get("response", "")
            print(generated_text)

            open_file.write("Describing" + i + ":\n" + generated_text.strip() + "\n\n")

        except Exception as e:
            print("ERROR:", str(e))

print(f"File saved to '{output_file}'")
open_file.close()