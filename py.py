import json

def keep_every_7th(input_file, output_file):
    # Load the JSON data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Make sure data is a list
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list at the top level")

    # Keep only every 7th entry (7th, 14th, 21st, …)
    filtered_data = [entry for idx, entry in enumerate(data, start=1) if idx % 7 == 0]

    # Save filtered data to new JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(filtered_data)} entries (every 7th) to {output_file}")


# Example usage
if __name__ == "__main__":
    keep_every_7th("C:\\Users\\fabia\\Desktop\\PurpleLlama\\CybersecurityBenchmarks\\datasets\\mitre\\mitre_prompts_multilingual_machine_translated.json", "C:\\Users\\fabia\\Desktop\\PurpleLlama\\CybersecurityBenchmarks\\datasets\\mitre\\mitre_prompts_multilingual_machine_translated_every_7nth_extracted.json")
