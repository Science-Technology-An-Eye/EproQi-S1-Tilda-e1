from datasets import load_dataset

# Use the 'simplified' configuration to avoid HTML parsing
dataset = load_dataset("natural_questions", "default", split="train", streaming=True)

print("--- Fetching STEM-related Google NQ Example ---")

for item in dataset:
    question = item['question']['text']

    # Check if this is a 'Science' style question
    if any(word in question.lower() for word in ['energy', 'element', 'who', 'how', 'what']):

        # 1. Reconstruct the document text from tokens
        document_tokens = item['document']['tokens']['token']

        # 2. Find the Long Answer (Paragraph)
        long_answer_meta = item['annotations']['long_answer'][0]
        if long_answer_meta['start_token'] != -1:
            start = long_answer_meta['start_token']
            end = long_answer_meta['end_token']
            paragraph = " ".join(document_tokens[start:end])

            # 3. Find the Short Answer (The Fact)
            short_meta = item['annotations']['short_answers'][0]
            if short_meta['start_token']:
                s_start = short_meta['start_token'][0]
                s_end = short_meta['end_token'][0]
                answer = " ".join(document_tokens[s_start:s_end])
            else:
                answer = "No short answer provided"

            print(f"\nQUESTION: {question}")
            print(f"CONTEXT PARAGRAPH: {paragraph}")
            print(f"FACTUAL ANSWER: {answer}")
            break