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
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def execute_generated_code(script_path: str, timeout: int = 10) -> tuple[int, str]:
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


def run_single_test(client: OpenAI, model: str, benchmark: Dict[str, Any], scripts_dir: str) -> Dict[str, Any]:
    bench_id = benchmark["id"]
    prompt = benchmark["prompt"]
    expected = benchmark["expected"]
    
    full_prompt = f"{prompt}\n\n[CACHE_MISS_ID: {uuid.uuid4()}]"
    
    collected_chunks = []
    completion_tokens = 0
    prompt_tokens = 0
    
    start_time = time.time()
    first_token_time = None
    
    # stream_options requests the usage stats in the final streaming chunk
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert Python developer. Output only valid, runnable code."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True}
    )
    
    with open(LIVE_STREAM_FILE, "a", encoding="utf-8") as sf:
        sf.write(f"\n\n{'='*50}\n🤖 MODEL: {model}\n⚡ CONFIG: [{bench_id.upper()}]\n{'='*50}\n")
        sf.flush()
        
        for chunk in response:
            # Check if this chunk contains the final usage statistics
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                
            if chunk.choices and len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content:
                    if first_token_time is None:
                        first_token_time = time.time()
                        
                    collected_chunks.append(content)
                    completion_tokens += 1
                    sf.write(content)
                    sf.flush()
                    
    end_time = time.time()
    raw_content = "".join(collected_chunks)
    
    # --- METRICS CALCULATION ---
    total_time = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else 0.0
    
    if first_token_time and completion_tokens > 1:
        generation_time = end_time - first_token_time
        tps = (completion_tokens - 1) / generation_time
    else:
        tps = 0.0
        
    # Fallback if the backend doesn't support stream_options usage reporting
    if prompt_tokens == 0:
        prompt_tokens = len(full_prompt) // 4  # Rough English token estimation
        
    total_context = prompt_tokens + completion_tokens
    
    # Save script
    code_to_run = extract_python_code(raw_content)
    safe_model_name = model.replace("/", "_").replace(":", "_")
    script_filename = f"{safe_model_name}_{bench_id}.py"
    script_path = os.path.join(scripts_dir, script_filename)
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code_to_run)
    
    # Execute & Validate
    return_code, script_output = execute_generated_code(script_path)
    
    if return_code == 0 and script_output == expected:
        status = "✅"
    elif return_code == -1:
        status = "💥" # Timeout
    else:
        status = "❌"
        
    return {
        "status": status,
        "tps": tps,
        "ttft": ttft,
        "total_time": total_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_context": total_context
    }


def generate_reports(matrix_results: Dict[str, Dict[str, Any]], benchmarks: List[Dict[str, Any]], run_dir: str) -> None:
    csv_path = os.path.join(run_dir, "benchmark_results.csv")
    md_path = os.path.join(run_dir, "matrix_report.md")
    
    bench_ids = [b["id"] for b in benchmarks]
    
    # 1. Detailed CSV Generation (Keeps raw integers for precise data analysis)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Model", "Warmup Status", "Warmup Time (s)"]
        for b_id in bench_ids:
            header.extend([
                f"{b_id.upper()} Status", 
                f"{b_id.upper()} TPS", 
                f"{b_id.upper()} TTFT (s)", 
                f"{b_id.upper()} Prompt Tokens",
                f"{b_id.upper()} Output Tokens",
                f"{b_id.upper()} Total Context"
            ])
        writer.writerow(header)
        
        for model, res in matrix_results.items():
            warmup = res.get("warmup", {"status": "N/A", "time": 0.0})
            row = [model, warmup["status"], f"{warmup['time']:.2f}"]
            for b_id in bench_ids:
                if b_id in res:
                    m = res[b_id]
                    row.extend([
                        m["status"], 
                        f"{m['tps']:.2f}", 
                        f"{m['ttft']:.2f}", 
                        m["prompt_tokens"], 
                        m["completion_tokens"], 
                        m["total_context"]
                    ])
                else:
                    row.extend(["N/A", "0.00", "0.00", "0", "0", "0"])
            writer.writerow(row)

    # 2. Dense, Clean Markdown Generation (Uses .1f kilotoken formatting)
    with open(md_path, mode="w", encoding="utf-8") as f:
        f.write("# 📊 Local Inference Performance - Detailed Matrix Report\n\n")
        f.write(f"> Run Directory / Unix Timestamp: `{os.path.basename(run_dir)}`\n\n")
        
        columns = ["WARMUP (Load Time)"] + [f"{b_id.upper()}<br>(Status, TPS, TTFT, Ctx)" for b_id in bench_ids]
        f.write(f"| Model Name | {' | '.join(columns)} |\n")
        f.write(f"| :--- | {' | '.join([':---'] * len(columns))} |\n")
        
        for model, res in matrix_results.items():
            warmup = res.get("warmup", {"status": "N/A", "time": 0.0})
            cells = [f"{warmup['status']} {warmup['time']:.2f}s"]
            
            for b_id in bench_ids:
                if b_id in res:
                    m = res[b_id]
                    
                    # Convert to kilotokens (k)
                    p_k = m['prompt_tokens'] / 1000
                    c_k = m['completion_tokens'] / 1000
                    t_k = m['total_context'] / 1000
                    
                    # Example format: Ctx: 0.1k+0.2k=0.3k tok
                    ctx_str = f"Ctx: {p_k:.1f}k+{c_k:.1f}k={t_k:.1f}k tok"
                    
                    cells.append(f"{m['status']} {m['tps']:.1f} t/s<br>`TTFT: {m['ttft']:.2f}s`<br>`{ctx_str}`")
                else:
                    cells.append("N/A")
            
            f.write(f"| **{model}** | {' | '.join(cells)} |\n")


def run_benchmark():
    config = load_config(CONFIG_FILE)
    
    with open(LIVE_STREAM_FILE, "w", encoding="utf-8") as f:
        pass
        
    run_timestamp = str(int(time.time()))
    run_dir = os.path.join(RESULTS_BASE_DIR, run_timestamp)
    
    scripts_dir = os.path.join(run_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    
    client = OpenAI(
        base_url=config.get("API_BASE_URL", "http://127.0.0.1:8000/v1"), 
        api_key=config.get("API_KEY", "1234")
    )
    
    models = config["MODELS_TO_TEST"]
    benchmarks = config["BENCHMARKS"]
    
    matrix_results = {model: {} for model in models}

    print("🚀 Starting Automated Multi-Level Matrix Benchmark...")
    print(f"📁 Target Run Directory: {run_dir}/")
    print(f"📺 Live stream pipeline opened at: ./{LIVE_STREAM_FILE}\n")
    
    for model in models:
        print("=" * 60)
        print(f"🤖 Activating Target: {model}")
        print("=" * 60)
        
        # --- WARM-UP PHASE ---
        print(f"  🔥 Warming up {model} (Loading weights into RAM)...")
        warmup_start = time.time()
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Wake up."}],
                max_tokens=1
            )
            warmup_time = time.time() - warmup_start
            warmup_status = "✅"
            print(f"    Result: {warmup_status} | Boot Time: {warmup_time:.2f}s")
        except Exception as e:
            warmup_time = 0.0
            warmup_status = "❌"
            print(f"    ⚠️ Warm-up failed: {e}")
            
        matrix_results[model]["warmup"] = {
            "status": warmup_status,
            "time": warmup_time
        }
        # ---------------------
        
        for bench in benchmarks:
            bench_id = bench["id"]
            print(f"  ⚡ Running [{bench_id.upper()}] configuration...")
            
            metrics = run_single_test(client, model, bench, scripts_dir)
            matrix_results[model][bench_id] = metrics
            
            # Print terminal status in standard format so it's precise, but visually distinct
            print(f"    Result: {metrics['status']} | Speed: {metrics['tps']:.2f} t/s | TTFT: {metrics['ttft']:.2f}s | Ctx: {metrics['total_context']/1000:.1f}k tok")
            
        print(f"🧹 Unloading {model} data... breathing for 3s...")
        time.sleep(3)

    print("\n💾 Compilation of results initiated...")
    generate_reports(matrix_results, benchmarks, run_dir)
    print(f"🎉 Benchmark execution successful! Check: {run_dir}/matrix_report.md")


if __name__ == "__main__":
    run_benchmark()