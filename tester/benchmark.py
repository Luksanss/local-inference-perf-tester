import time
import subprocess
import re
import uuid
import csv
import os
import json
from openai import OpenAI

# --- CONFIGURATION ---
CONFIG_FILE = "benchmark_config.json"

if not os.path.exists(CONFIG_FILE):
    print(f"❌ Error: {CONFIG_FILE} is missing! Please create it.")
    exit(1)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

API_BASE_URL = config.get("API_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = config.get("API_KEY", "1234")
MODELS_TO_TEST = config.get("MODELS_TO_TEST", [])

if not MODELS_TO_TEST:
    print("❌ Error: MODELS_TO_TEST array is empty in the config file.")
    exit(1)

TEST_PROMPT = """
Write a Python script that calculates the factorial of 10 (10!).
The script must print ONLY the final result exactly in this format:
Output: <number>
Do not print intermediate steps.
"""
EXPECTED_OUTPUT = "Output: 3628800"
RESULTS_BASE_DIR = "results"
# ---------------------

def extract_python_code(text):
    match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
    if match: return match.group(1)
    match = re.search(r'```\n(.*?)\n```', text, re.DOTALL)
    if match: return match.group(1)
    return text

def save_result(model, ttft, tps, status, csv_path, md_path):
    """Appends a single benchmark result to the run's CSV and MD files."""
    csv_exists = os.path.isfile(csv_path)
    md_exists = os.path.isfile(md_path)
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Save to raw CSV
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["Timestamp", "Model", "TTFT (s)", "TPS (tokens/s)", "Result Status"])
        writer.writerow([timestamp, model, f"{ttft:.2f}", f"{tps:.2f}", status])

    # 2. Save to Beautified Markdown
    with open(md_path, mode='a', encoding='utf-8') as f:
        if not md_exists:
            f.write("# 🚀 Apple Silicon LLM Benchmark Report\n\n")
            f.write(f"> Run Directory: `{os.path.basename(os.path.dirname(md_path))}`\n\n")
            f.write("## 📊 Results Summary\n\n")
            f.write("| Timestamp | Model | TTFT (s) | Speed (TPS) | Validation Result |\n")
            f.write("| :--- | :--- | :---: | :---: | :--- |\n")
        
        if "PASS" in status:
            md_status = f"✅ {status}"
        elif "FAIL" in status:
            md_status = f"❌ {status}"
        else:
            md_status = f"⚠️ {status}"
            
        f.write(f"| {timestamp} | `{model}` | **{ttft:.2f}** | **{tps:.2f}** | {md_status} |\n")

def run_benchmark():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # --- FOLDER SETUP ---
    run_timestamp = str(int(time.time()))
    run_dir = os.path.join(RESULTS_BASE_DIR, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    csv_file = os.path.join(run_dir, "benchmark_results.csv")
    md_file = os.path.join(run_dir, "benchmark_report.md")
    # --------------------

    print("🚀 Starting Cache-Busting Deterministic Benchmark...")
    print(f"📁 Session data will be saved to: {run_dir}/\n")
    
    for model_name in MODELS_TO_TEST:
        print("=" * 60)
        print(f"🧪 Testing Model: {model_name}")
        print("=" * 60)
        
        status_message = "Unknown"
        ttft = 0.0
        tps = 0.0
        
        try:
            start_time = time.time()
            first_token_time = None
            token_count = 0
            full_response = ""
            
            unique_id = str(uuid.uuid4())
            system_prompt = f"You are an expert Python developer. Output only valid, runnable code. [CACHE_MISS_ID: {unique_id}]"
            
            print(f"🧹 Cache Buster Injected: {unique_id[:8]}...")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": TEST_PROMPT}
                ],
                stream=True
            )
            
            print("Generating:\n", flush=True)
            
            for chunk in response:
                if first_token_time is None:
                    first_token_time = time.time()
                    
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True) 
                    full_response += content
                    token_count += 1
                    
            end_time = time.time()
            
            if first_token_time:
                ttft = first_token_time - start_time
                generation_time = end_time - first_token_time
                tps = token_count / generation_time if generation_time > 0 else 0
            
            print("\n\n📊 Metrics:")
            print(f"Time to First Token (TTFT): {ttft:.2f} s")
            print(f"Speed (TPS):                {tps:.2f} tokens/s")
            
            print("\n💾 Executing generated code...")
            code_to_run = extract_python_code(full_response)
            safe_model_name = model_name.replace("/", "_")
            
            # Save the python script inside the timestamped results folder
            filename = os.path.join(run_dir, f"generated_{safe_model_name}.py")
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(code_to_run)
                
            result = subprocess.run(
                ["python3", filename], 
                capture_output=True, 
                text=True,
                timeout=15 
            )
            
            if result.returncode == 0:
                stdout_lower = result.stdout.strip().lower()
                expected_lower = EXPECTED_OUTPUT.lower()
                
                if expected_lower in stdout_lower:
                    status_message = "PASS - Logic Flawless"
                    print(f"✅ RESULT: {status_message}")
                else:
                    status_message = f"FAIL - Wrong Output (Got: {result.stdout.strip()})"
                    print(f"❌ RESULT: {status_message}")
            else:
                status_message = "FAIL - Syntax/Runtime Crash"
                print(f"❌ RESULT: {status_message}")

        except subprocess.TimeoutExpired:
            status_message = "FAIL - Timeout (Infinite Loop)"
            print(f"\n⚠️ RESULT: {status_message}")
        except Exception as e:
            status_message = "ERROR - API/Script Failure"
            print(f"\n❌ Error with {model_name}: {e}\n")
            
        finally:
            save_result(model_name, ttft, tps, status_message, csv_file, md_file)
            print("-" * 40)
            time.sleep(3)

if __name__ == "__main__":
    run_benchmark()