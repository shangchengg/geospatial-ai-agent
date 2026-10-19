
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag import answer_question

QUESTIONS_FILE = BASE_DIR / "evaluation" / "questions.json"


def evaluate():

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    for item in questions:

        print(f"\n{'=' * 60}")
        print(f"Testing: {item['id']}")
        print(f"Question: {item['question']}")

        try:
            answer, documents = answer_question(item["question"])
            answer = str(answer).strip()

            
            # Retrieval evaluation
            expected_pages = set(item.get("expected_pages", []))

            retrieved_pages = set()

            for doc in documents:
                page = doc.metadata.get("page")

                if page is not None:
                    try:
                        retrieved_pages.add(int(page))
                    except (TypeError, ValueError):
                        pass

            retrieval_hit = (
                bool(expected_pages.intersection(retrieved_pages))
                if expected_pages
                else None
            )

            print("Expected pages:", sorted(expected_pages))
            print("Retrieved pages:", sorted(retrieved_pages))
            print("Retrieval Hit@K:", retrieval_hit)

            keywords = item["expected_keywords"]

            # Only factual/numerical questions have keyword checks
            keyword_pass = (
                all(k.lower() in answer.lower() for k in keywords)
                if keywords else None
            )

            sources = [
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page")
                }
                for doc in documents
            ]

            print(f"\nAnswer:\n{answer}")
            print(f"\nKeyword check: {keyword_pass}")
            print(f"Retrieved sources: {sources}")

            results.append({
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "answer": answer,
                "keyword_pass": keyword_pass,
                "sources": sources,
                "error": None,
                "expected_pages": sorted(expected_pages),
                "retrieved_pages": sorted(retrieved_pages),
                "retrieval_hit": retrieval_hit,
                "needs_review": item.get("needs_review", False),
            })

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "id": item["id"],
                "error": str(e)
            })

    output_file = BASE_DIR / "evaluation" / "results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    
    # Evaluation summary
    valid_results = [
        r for r in results
        if r.get("error") is None
    ]

    keyword_results = [
        r for r in valid_results
        if r.get("keyword_pass") is not None
        and not r.get("needs_review", False)
    ]

    retrieval_results = [
        r for r in valid_results
        if r.get("retrieval_hit") is not None
    ]

    keyword_accuracy = (
        sum(r["keyword_pass"] for r in keyword_results)
        / len(keyword_results)
        if keyword_results else None
    )

    hit_at_k = (
        sum(r["retrieval_hit"] for r in retrieval_results)
        / len(retrieval_results)
        if retrieval_results else None
    )

    print("\n===== EVALUATION SUMMARY =====")
    print("Total questions:", len(results))
    print("Successful generations:", len(valid_results))

    if keyword_accuracy is not None:
        print(f"Keyword Pass Rate: {keyword_accuracy:.2%}")
    else:
        print("Keyword Pass Rate: N/A")

    if hit_at_k is not None:
        print(f"Retrieval Hit@K: {hit_at_k:.2%}")
    else:
        print("Retrieval Hit@K: N/A")

    print("==============================")

    print("\nEvaluation completed.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    evaluate()
