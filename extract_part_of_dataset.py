import json

def keep_every_Xth(input_file, output_file, X):
    # Load the JSON data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Make sure data is a list
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list at the top level")

    # Keep only every Xth entry (X, 2X, 3X, …)
    filtered_data = [entry for idx, entry in enumerate(data, start=1) if idx % X == 0]

    # Save filtered data to new JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(filtered_data)} entries (every {X}th) to {output_file}")


# Example usage
if __name__ == "__main__":
    keep_every_Xth("C:\\Users\\fabia\\Desktop\\PurpleLlama\\CybersecurityBenchmarks\\datasets\\prompt_injection\\prompt_injection.json", "C:\\Users\\fabia\\Desktop\\PurpleLlama\\CybersecurityBenchmarks\\datasets\\prompt_injection\\prompt_injection_every_2nd_extracted.json", 2)
