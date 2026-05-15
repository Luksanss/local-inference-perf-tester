import time
import subprocess
import re
import uuid
from openai import OpenAI

# --- CONFIGURATION ---
API_BASE_URL = "http://127.0.0.1:8000/v1" 
API_KEY = "1234"

# Testing all three quantization methods to see how your M1 handles them
MODELS_TO_TEST = [
    "gemma-4-E4B-it-UD-MLX-4bit",
    "gemma-4-e4b-it-OptiQ-4bit",
    "gemma-4-E4B-it-MLX-4bit"
]

# A new deterministic prompt (Factorial of 10)
TEST_PROMPT = """
Write a Python script that calculates the factorial of 10 (10!).
The script must print ONLY the final result exactly in this format:
Output: <number>
Do not print intermediate steps.
"""

# 10! is 3628800
EXPECTED_OUTPUT = "Output: 3628800"
# ---------------------

def extract_python_code(text):
    match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
    if match: return match.group(1)
    match = re.search(r'```\n(.*?)\n```', text, re.DOTALL)
    if match: return match.group(1)
    return text

def run_benchmark():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("🚀 Starting Cache-Busting Deterministic Benchmark...\n")
    
    for model_name in MODELS_TO_TEST:
        print("=" * 60)
        print(f"🧪 Testing Model: {model_name}")
        print("=" * 60)
        
        try:
            start_time = time.time()
            first_token_time = None
            token_count = 0
            full_response = ""
            
            # --- THE CACHE BUSTER ---
            # Appending a random UUID guarantees the prompt hash is unique every run.
            # This forces the M1 to process the prompt from scratch, ignoring the SSD KV cache.
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
            
            ttft = first_token_time - start_time
            generation_time = end_time - first_token_time
            tps = token_count / generation_time if generation_time > 0 else 0
            
            print("\n\n📊 Metrics:")
            print(f"Time to First Token (TTFT): {ttft:.2f} s")
            print(f"Speed (TPS):                {tps:.2f} tokens/s")
            
            print("\n💾 Saving and executing generated code...")
            code_to_run = extract_python_code(full_response)
            safe_model_name = model_name.replace("/", "_")
            filename = f"generated_{safe_model_name}.py"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(code_to_run)
                
            print(f"▶️  Running {filename}...\n")
            
            result = subprocess.run(
                ["python3", filename], 
                capture_output=True, 
                text=True,
                timeout=15 
            )
            
            if result.returncode == 0:
                stdout_lower = result.stdout.strip().lower()
                expected_lower = EXPECTED_OUTPUT.lower()
                
                print("Terminal Output:")
                print(f"[{result.stdout.strip()}]")
                print("-" * 40)
                
                if expected_lower in stdout_lower:
                    print("✅ RESULT: PASS - Logic flawless!")
                else:
                    print("❌ RESULT: FAIL (Logic Error)")
                    print(f"Expected: '{EXPECTED_OUTPUT}'")
            else:
                print("❌ RESULT: FAIL (Syntax/Runtime Error) - The code crashed.")
                print("----------------------------------------")
                print(result.stderr.strip())
                print("----------------------------------------")

            # Let the unified memory breathe for a few seconds before loading the next weights
            time.sleep(3)

        except subprocess.TimeoutExpired:
            print("\n⚠️ RESULT: FAIL (Timeout) - Code got stuck in an infinite loop.")
        except Exception as e:
            print(f"\n❌ Error with {model_name}: {e}\n")

if __name__ == "__main__":
    run_benchmark()