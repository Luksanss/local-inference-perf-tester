import time
import subprocess
import re
import uuid
import csv
import os
import json
from typing import Dict, List, Any
from openai import OpenAI

# --- GLOBAL CONFIGURATION ---
CONFIG_FILE = "benchmark_config.json"
RESULTS_BASE_DIR = "results"
LIVE_STREAM_FILE = "live_stream.txt"


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the benchmark configuration from a JSON file."""
    if not os.path.exists(config_path):
        print(f"❌ Error: {config_path} is missing! Please create it in the root directory.")
        exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
        
    if not config_data.get("MODELS_TO_TEST") or not config_data.get("BENCHMARKS"):
        print("❌ Error: MODELS_TO_TEST or BENCHMARKS array is empty in config.")
        exit(1)
        
    return config_data


def extract_python_code(text: str) -> str:
    """Extracts Python code from Markdown blocks, or returns raw text if none found."""
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def execute_generated_code(script_path: str, timeout: int = 10) -> tuple[int, str]:
    """Executes the saved python script and collects output."""
    try:
        run_res = subprocess.run(
            ["python3", script_path], 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return run_res.returncode, run_res.stdout.strip()
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT_EXPIRED"


def run_single_test(client: OpenAI, model: str, benchmark: Dict[str, Any], scripts_dir: str) -> tuple[str, float]:
    """Runs a prompt, saves the output to a file, and executes it."""
    bench_id = benchmark["id"]
    prompt = benchmark["prompt"]
    expected = benchmark["expected"]
    
    full_prompt = f"{prompt}\n\n[CACHE_MISS_ID: {uuid.uuid4()}]"
    
    collected_chunks = []
    token_count = 0
    
    start_time = time.time()
    first_token_time = None
    
    # 1. Initiate Streaming connection to oMLX
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert Python developer. Output only valid, runnable code."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.0,
        stream=True
    )
    
    # 2. Open the streaming pipeline and append characters live
    with open(LIVE_STREAM_FILE, "a", encoding="utf-8") as sf:
        sf.write(f"\n\n{'='*50}\n🤖 MODEL: {model}\n⚡ CONFIG: [{bench_id.upper()}]\n{'='*50}\n")
        sf.flush()
        
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content:
                    # TRIGGER THE STOPWATCH ON THE VERY FIRST REAL TOKEN
                    if first_token_time is None:
                        first_token_time = time.time()
                        
                    collected_chunks.append(content)
                    token_count += 1
                    sf.write(content)
                    sf.flush()
                    
    end_time = time.time()
    raw_content = "".join(collected_chunks)
    
    # 3. Calculate True Generation TPS (Ignoring TTFT)
    if first_token_time and token_count > 1:
        generation_time = end_time - first_token_time
        # We subtract 1 token because the first token marks the start of the time window
        tps = (token_count - 1) / generation_time
    else:
        tps = 0.0
    
    # Extract code and save it to the permanent scripts folder
    code_to_run = extract_python_code(raw_content)
    
    safe_model_name = model.replace("/", "_").replace(":", "_")
    script_filename = f"{safe_model_name}_{bench_id}.py"
    script_path = os.path.join(scripts_dir, script_filename)
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code_to_run)
    
    # Execute the saved script
    return_code, script_output = execute_generated_code(script_path)
    
    if return_code == 0 and script_output == expected:
        return "✅", tps
    elif return_code == -1:
        return "💥 (Timeout)", tps
    else:
        return "❌", tps


def generate_reports(matrix_results: Dict[str, Any], benchmarks: List[Dict[str, Any]], run_dir: str) -> None:
    """Generates both the raw CSV log and the beautified Markdown matrix table."""
    csv_path = os.path.join(run_dir, "benchmark_results.csv")
    md_path = os.path.join(run_dir, "matrix_report.md")
    
    bench_ids = [b["id"] for b in benchmarks]
    
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Model"] + [f"{b_id.upper()} Status" for b_id in bench_ids] + [f"{b_id.upper()} TPS" for b_id in bench_ids]
        writer.writerow(header)
        
        for model, res in matrix_results.items():
            row = [model] + [res[b_id] for b_id in bench_ids] + [f"{tps:.2f}" for tps in res["tps_list"]]
            writer.writerow(row)

    # Clean Markdown generation without HTML tags
    with open(md_path, mode="w", encoding="utf-8") as f:
        f.write("# 📊 Local Inference Performance - Detailed Matrix Report\n\n")
        f.write(f"> Run Directory / Unix Timestamp: `{os.path.basename(run_dir)}`\n\n")
        
        columns = [f"{b_id.upper()} (Status & Speed)" for b_id in bench_ids]
        f.write(f"| Model Name | {' | '.join(columns)} |\n")
        f.write(f"| :--- | {' | '.join([':---:'] * len(bench_ids))} |\n")
        
        for model, res in matrix_results.items():
            cells = []
            for idx, b_id in enumerate(bench_ids):
                status = res[b_id]
                tps = res["tps_list"][idx] if idx < len(res["tps_list"]) else 0.0
                cells.append(f"{status} ({tps:.1f} t/s)")
            
            f.write(f"| **{model}** | {' | '.join(cells)} |\n")


def run_benchmark():
    """Main entry point for the tester CLI command."""
    config = load_config(CONFIG_FILE)
    
    with open(LIVE_STREAM_FILE, "w", encoding="utf-8") as f:
        pass
        
    run_timestamp = str(int(time.time()))
    run_dir = os.path.join(RESULTS_BASE_DIR, run_timestamp)
    
    # Create the main directory AND the scripts subdirectory
    scripts_dir = os.path.join(run_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    
    client = OpenAI(
        base_url=config.get("API_BASE_URL", "http://127.0.0.1:8000/v1"), 
        api_key=config.get("API_KEY", "1234")
    )
    
    models = config["MODELS_TO_TEST"]
    benchmarks = config["BENCHMARKS"]
    
    matrix_results = {
        model: {b["id"]: "❌" for b in benchmarks} for model in models
    }
    for model in models:
        matrix_results[model]["tps_list"] = []

    print("🚀 Starting Automated Multi-Level Matrix Benchmark...")
    print(f"📁 Target Run Directory: {run_dir}/")
    print(f"📺 Live stream pipeline opened at: ./{LIVE_STREAM_FILE}\n")
    
    for model in models:
        print("=" * 60)
        print(f"🤖 Activating Target: {model}")
        print("=" * 60)
        
        for bench in benchmarks:
            bench_id = bench["id"]
            print(f"  ⚡ Running [{bench_id.upper()}] configuration...")
            
            # Pass the scripts_dir so the function knows where to save the files
            status, tps = run_single_test(client, model, bench, scripts_dir)
            
            matrix_results[model][bench_id] = status
            matrix_results[model]["tps_list"].append(tps)
            
            print(f"    Result: {status} | Execution Speed: {tps:.2f} tokens/s")
            
        print(f"🧹 Unloading {model} data... breathing for 3s...")
        time.sleep(3)

    print("\n💾 Compilation of results initiated...")
    generate_reports(matrix_results, benchmarks, run_dir)
    print(f"🎉 Benchmark execution successful! Check: {run_dir}/matrix_report.md")


if __name__ == "__main__":
    run_benchmark()