import json
import subprocess
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv

# 1. 先定义项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2. 再去加载 .env 文件
load_dotenv(PROJECT_ROOT / ".env")

# 3. 剩下的配置
QUERY_FILE = PROJECT_ROOT / "data" / "eval_queries_B.json"
MODE = "hybrid"
LIMIT = 5

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "batch_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single(item):
    q_id = item["query_id"]
    query = item.get("paraphrase_query") or item.get("original_query")

    print(f"[Start] Query {q_id}")

    prompt_path = PROJECT_ROOT / "prompts" / ("system_prompt.txt" if MODE == "hybrid" else "system_prompt_pure_dci.txt")
    run_dir = OUTPUT_DIR / f"run_{q_id}"

    cmd = [
        "uv", "run", "dci-agent-lite",
        "--provider", "deepseek",
        "--model", "deepseek-flash",
        "--cwd", str(PROJECT_ROOT / "corpus" / "bc_plus_docs"),
        "--system-prompt-file", str(prompt_path),
        "--output-dir", str(run_dir),
        "--extra-arg=--thinking high",
        query
    ]

    start_time = time.time()
    try:
        # 运行 DCI
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=300,cwd=str(PROJECT_ROOT))
        duration = time.time() - start_time

        final_file = run_dir / "final.txt"
        if final_file.exists():
            predicted = final_file.read_text(encoding="utf-8").strip()
            status = "success"
        else:
            err_msg = result.stderr if result.stderr else "Unknown Error"
            predicted = f"ERROR: {err_msg[-500:]}"
            status = "failed"

        print(f"[Done] Query {q_id} ({duration:.1f}s)")
        return {
            "query_id": q_id,
            "query": query,
            "gold_answer": item["gold_answer"],
            "predicted_answer": predicted,
            "duration_sec": round(duration, 2),
            "status": status
        }

    except subprocess.TimeoutExpired:
        print(f"[Timeout] Query {q_id}")
        return {"query_id": q_id, "status": "timeout"}
    except Exception as e:
        print(f"[Error] Query {q_id}: {e}")
        return {"query_id": q_id, "status": "error"}


def main():
    if not QUERY_FILE.exists():
        print(f"Error: {QUERY_FILE} does not exist.")
        return

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)[:LIMIT]

    results = []
    for item in data:
        res = run_single(item)
        results.append(res)
        time.sleep(2)

    out_file = OUTPUT_DIR / f"{MODE}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()