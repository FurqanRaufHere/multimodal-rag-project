import json
import requests
import csv
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def evaluate():
    # Load test cases
    with open("evaluation/test_cases.jsonl", "r", encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f]

    bleu_scores = []
    rouge_scores = []

    rouge = Rouge()
    smoother = SmoothingFunction().method4

    results = []

    for idx, case in enumerate(test_cases):
        query = case["query"]
        expected = case["expected_answer"]

        print(f"\n🔍 [Test {idx+1}] Query: {query}")
        try:
            response = requests.post("http://localhost:5000/api/query", json={"text": query})
            if response.status_code != 200:
                print(f"❌ Error from server: {response.status_code}")
                continue

            result_json = response.json()
            generated = result_json.get("answer", "").strip()

            print(f"✅ Generated: {generated}")
            print(f"🎯 Expected : {expected}")

            # BLEU
            bleu = sentence_bleu([expected.split()], generated.split(), smoothing_function=smoother)
            bleu_scores.append(bleu)

            # ROUGE
            rouge_score = rouge.get_scores(generated, expected, avg=True)['rouge-l']['f']
            rouge_scores.append(rouge_score)

            print(f"📏 BLEU: {bleu:.4f} | ROUGE-L: {rouge_score:.4f}")

            results.append({
                "Test #": idx + 1,
                "Query": query,
                "Expected Answer": expected,
                "Generated Answer": generated,
                "BLEU Score": round(bleu, 4),
                "ROUGE-L Score": round(rouge_score, 4)
            })

        except Exception as e:
            print(f"❌ Exception: {e}")

    # Save to CSV
    os.makedirs("evaluation", exist_ok=True)
    csv_path = "evaluation/eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results saved to: {csv_path}")

    # Averages
    if bleu_scores and rouge_scores:
        print("\n📊 Final Evaluation Summary:")
        print(f"🔵 Average BLEU Score   : {sum(bleu_scores)/len(bleu_scores):.4f}")
        print(f"🔴 Average ROUGE-L Score: {sum(rouge_scores)/len(rouge_scores):.4f}")
    else:
        print("⚠️ No scores collected — check errors.")

if __name__ == "__main__":
    evaluate()
