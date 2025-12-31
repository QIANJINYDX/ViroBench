import torch
from pathlib import Path
import random
from typing import Optional
import sys
import time
import traceback
import json
from datetime import datetime
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
# from model.myevo2 import Evo2
from evo2 import Evo2

def random_dna_sequence(length: int, alphabet: str = "ACGT", seed: Optional[int] = None) -> str:
    """
    Generate a random nucleotide sequence of a specified length.

    Args:
        length: Desired sequence length (>= 0).
        alphabet: Characters to sample from (default "ACGT").
                  You can pass "ACGTN" if you want to allow N.
        seed: Optional random seed for reproducibility.

    Returns:
        A random sequence string of exactly `length`.
    """
    rng = random.Random(seed) if seed is not None else random
    return "".join(rng.choices(alphabet, k=length))


def get_gpu_memory_info():
    """
    Get current GPU memory usage information.
    
    Returns:
        dict: Memory info with allocated, reserved, and max allocated (in MB)
    """
    if not torch.cuda.is_available():
        return {
            'allocated_mb': 0,
            'reserved_mb': 0,
            'max_allocated_mb': 0,
            'available': False
        }
    
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2    # MB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'max_allocated_mb': max_allocated,
        'available': True
    }


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def test_max_generation_length(model, start_length: int = 1024, max_length: int = 100000, 
                                step_size: int = 1024, prompt_length: int = 512,
                                temperature: float = 1.0, top_k: int = 4,
                                seed: Optional[int] = 42):
    """
    Test the maximum generation length of the model by gradually increasing n_tokens.
    
    Args:
        model: The Evo2 model instance
        start_length: Starting generation length to test
        max_length: Maximum length to test (safety limit)
        step_size: Step size for increasing length
        prompt_length: Length of the input prompt sequence
        temperature: Generation temperature
        top_k: Top-k sampling parameter
        seed: Random seed for reproducibility
    
    Returns:
        dict: Results containing max_successful_length, test_history, etc.
    """
    print("=" * 80)
    print("Testing Maximum Generation Length for evo2_40b")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Start length: {start_length}")
    print(f"  Max test length: {max_length}")
    print(f"  Step size: {step_size}")
    print(f"  Prompt length: {prompt_length}")
    print(f"  Temperature: {temperature}, Top-k: {top_k}")
    print("=" * 80)
    
    max_successful_length = 0
    test_history = []
    current_length = start_length
    
    # Generate a fixed prompt for consistency
    prompt_seq = random_dna_sequence(length=prompt_length, seed=seed)
    print(f"\nUsing fixed prompt (length={prompt_length}): {prompt_seq[:50]}...")
    
    # Get initial GPU memory state
    initial_memory = get_gpu_memory_info()
    if initial_memory['available']:
        print(f"\nInitial GPU Memory:")
        print(f"  Allocated: {initial_memory['allocated_mb']:.2f} MB")
        print(f"  Reserved: {initial_memory['reserved_mb']:.2f} MB")
    
    while current_length <= max_length:
        print(f"\n{'='*80}")
        print(f"Testing generation length: {current_length} tokens")
        print(f"{'='*80}")
        
        # Reset peak memory stats and get baseline memory
        reset_gpu_memory_stats()
        baseline_memory = get_gpu_memory_info()
        
        start_time = time.time()
        try:
            output = model.generate(
                prompt_seqs=[prompt_seq],
                n_tokens=current_length,
                temperature=temperature,
                top_k=top_k
            )
            
            elapsed_time = time.time() - start_time
            
            # Get memory usage after generation
            memory_after = get_gpu_memory_info()
            memory_peak = get_gpu_memory_info()  # Get peak memory
            
            # Check if output is valid
            # Handle GenerationOutput object
            if hasattr(output, 'Output'):
                # GenerationOutput object with Output attribute
                generated_sequences = output.Output if isinstance(output.Output, list) else [output.Output]
            elif hasattr(output, 'sequences'):
                # GenerationOutput object with sequences attribute
                generated_sequences = output.sequences if isinstance(output.sequences, list) else [output.sequences]
            elif isinstance(output, list):
                # Direct list output
                generated_sequences = output
            else:
                # Try to convert to string or get the output directly
                generated_sequences = [str(output)]
            
            # Check if we have valid output
            if generated_sequences and len(generated_sequences) > 0:
                # Get the first sequence and calculate its length
                first_seq = generated_sequences[0]
                actual_length = len(first_seq) if isinstance(first_seq, str) else len(str(first_seq))
                print(f"✓ SUCCESS: Generated {actual_length} tokens in {elapsed_time:.2f}s")
                print(f"  Output preview: {str(first_seq)[:100]}...")
                
                # Print memory information
                if memory_after['available']:
                    memory_used = memory_peak['max_allocated_mb'] - baseline_memory['allocated_mb']
                    print(f"  GPU Memory:")
                    print(f"    Baseline: {baseline_memory['allocated_mb']:.2f} MB")
                    print(f"    Peak allocated: {memory_peak['max_allocated_mb']:.2f} MB")
                    print(f"    Memory used for generation: {memory_used:.2f} MB")
                    print(f"    Reserved: {memory_after['reserved_mb']:.2f} MB")
                
                max_successful_length = current_length
                memory_info = {}
                if memory_after['available']:
                    memory_info = {
                        'baseline_memory_mb': baseline_memory['allocated_mb'],
                        'peak_allocated_mb': memory_peak['max_allocated_mb'],
                        'memory_used_mb': memory_peak['max_allocated_mb'] - baseline_memory['allocated_mb'],
                        'reserved_mb': memory_after['reserved_mb']
                    }
                
                test_history.append({
                    'length': current_length,
                    'status': 'success',
                    'time': elapsed_time,
                    'actual_length': actual_length,
                    'memory': memory_info
                })
                
                # Move to next length
                current_length += step_size
            else:
                print(f"✗ FAILED: Empty output")
                test_history.append({
                    'length': current_length,
                    'status': 'failed',
                    'error': 'Empty output'
                })
                break
                
        except torch.cuda.OutOfMemoryError as e:
            elapsed_time = time.time() - start_time
            memory_after = get_gpu_memory_info()
            print(f"✗ FAILED: CUDA Out of Memory at length {current_length}")
            print(f"  Error: {str(e)}")
            print(f"  Time before failure: {elapsed_time:.2f}s")
            if memory_after['available']:
                print(f"  GPU Memory at failure:")
                print(f"    Allocated: {memory_after['allocated_mb']:.2f} MB")
                print(f"    Reserved: {memory_after['reserved_mb']:.2f} MB")
                print(f"    Peak allocated: {memory_after['max_allocated_mb']:.2f} MB")
            
            memory_info = {}
            if memory_after['available']:
                memory_info = {
                    'baseline_memory_mb': baseline_memory['allocated_mb'],
                    'allocated_mb': memory_after['allocated_mb'],
                    'reserved_mb': memory_after['reserved_mb'],
                    'peak_allocated_mb': memory_after['max_allocated_mb']
                }
            
            test_history.append({
                'length': current_length,
                'status': 'failed',
                'error': 'CUDA OOM',
                'time': elapsed_time,
                'memory': memory_info
            })
            break
            
        except RuntimeError as e:
            elapsed_time = time.time() - start_time
            error_msg = str(e)
            memory_after = get_gpu_memory_info()
            print(f"✗ FAILED: RuntimeError at length {current_length}")
            print(f"  Error: {error_msg}")
            print(f"  Time before failure: {elapsed_time:.2f}s")
            if memory_after['available']:
                print(f"  GPU Memory: {memory_after['allocated_mb']:.2f} MB allocated, {memory_after['reserved_mb']:.2f} MB reserved")
            
            memory_info = {}
            if memory_after['available']:
                memory_info = {
                    'baseline_memory_mb': baseline_memory['allocated_mb'],
                    'allocated_mb': memory_after['allocated_mb'],
                    'reserved_mb': memory_after['reserved_mb'],
                    'peak_allocated_mb': memory_after['max_allocated_mb']
                }
            
            test_history.append({
                'length': current_length,
                'status': 'failed',
                'error': error_msg,
                'time': elapsed_time,
                'memory': memory_info
            })
            # Check if it's a length-related error
            if 'length' in error_msg.lower() or 'max' in error_msg.lower():
                break
            # For other runtime errors, try next length
            current_length += step_size
            continue
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            memory_after = get_gpu_memory_info()
            print(f"✗ FAILED: Unexpected error at length {current_length}")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            print(f"  Time before failure: {elapsed_time:.2f}s")
            if memory_after['available']:
                print(f"  GPU Memory: {memory_after['allocated_mb']:.2f} MB allocated, {memory_after['reserved_mb']:.2f} MB reserved")
            traceback.print_exc()
            
            memory_info = {}
            if memory_after['available']:
                memory_info = {
                    'baseline_memory_mb': baseline_memory['allocated_mb'],
                    'allocated_mb': memory_after['allocated_mb'],
                    'reserved_mb': memory_after['reserved_mb'],
                    'peak_allocated_mb': memory_after['max_allocated_mb']
                }
            
            test_history.append({
                'length': current_length,
                'status': 'failed',
                'error': f"{type(e).__name__}: {str(e)}",
                'time': elapsed_time,
                'memory': memory_info
            })
            break
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Maximum successful generation length: {max_successful_length} tokens")
    print(f"\nTest History:")
    for entry in test_history:
        status_symbol = "✓" if entry['status'] == 'success' else "✗"
        time_str = f" ({entry.get('time', 0):.2f}s)" if 'time' in entry else ""
        print(f"  {status_symbol} Length {entry['length']}: {entry['status']}{time_str}")
        if 'memory' in entry and entry['memory']:
            mem = entry['memory']
            if 'memory_used_mb' in mem:
                print(f"      Memory used: {mem['memory_used_mb']:.2f} MB (Peak: {mem['peak_allocated_mb']:.2f} MB)")
            elif 'allocated_mb' in mem:
                print(f"      Memory: {mem['allocated_mb']:.2f} MB allocated, {mem['reserved_mb']:.2f} MB reserved")
        if 'error' in entry:
            print(f"      Error: {entry['error']}")
    print("=" * 80)
    
    return {
        'max_successful_length': max_successful_length,
        'test_history': test_history,
        'last_successful_test': test_history[-1] if test_history and test_history[-1]['status'] == 'success' else None,
        'config': {
            'start_length': start_length,
            'max_length': max_length,
            'step_size': step_size,
            'prompt_length': prompt_length,
            'temperature': temperature,
            'top_k': top_k,
            'seed': seed
        },
        'prompt_seq': prompt_seq,
        'initial_memory': initial_memory
    }


def save_results(results: dict, output_dir: Path):
    """
    Save test results to files in the output directory.
    
    Args:
        results: Test results dictionary
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    json_path = output_dir / f"max_length_test_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Detailed results saved to: {json_path}")
    
    # Save summary as text file
    summary_path = output_dir / f"max_length_test_{timestamp}.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Maximum Generation Length Test Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: evo2_40b\n\n")
        
        f.write("Configuration:\n")
        config = results['config']
        f.write(f"  Start length: {config['start_length']}\n")
        f.write(f"  Max test length: {config['max_length']}\n")
        f.write(f"  Step size: {config['step_size']}\n")
        f.write(f"  Prompt length: {config['prompt_length']}\n")
        f.write(f"  Temperature: {config['temperature']}\n")
        f.write(f"  Top-k: {config['top_k']}\n")
        f.write(f"  Seed: {config['seed']}\n\n")
        
        f.write(f"Prompt sequence: {results['prompt_seq']}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Maximum successful generation length: {results['max_successful_length']} tokens\n\n")
        
        f.write("Test History:\n")
        for entry in results['test_history']:
            status_symbol = "✓" if entry['status'] == 'success' else "✗"
            time_str = f" ({entry.get('time', 0):.2f}s)" if 'time' in entry else ""
            f.write(f"  {status_symbol} Length {entry['length']}: {entry['status']}{time_str}\n")
            if 'memory' in entry and entry['memory']:
                mem = entry['memory']
                if 'memory_used_mb' in mem:
                    f.write(f"      Memory used: {mem['memory_used_mb']:.2f} MB\n")
                    f.write(f"      Peak allocated: {mem['peak_allocated_mb']:.2f} MB\n")
                    f.write(f"      Reserved: {mem['reserved_mb']:.2f} MB\n")
                elif 'allocated_mb' in mem:
                    f.write(f"      Allocated: {mem['allocated_mb']:.2f} MB\n")
                    f.write(f"      Reserved: {mem['reserved_mb']:.2f} MB\n")
                    f.write(f"      Peak allocated: {mem.get('peak_allocated_mb', mem['allocated_mb']):.2f} MB\n")
            if 'error' in entry:
                f.write(f"      Error: {entry['error']}\n")
            if 'actual_length' in entry:
                f.write(f"      Actual generated length: {entry['actual_length']} tokens\n")
        
        # Add memory summary
        if results.get('initial_memory', {}).get('available', False):
            f.write("\n" + "=" * 80 + "\n")
            f.write("MEMORY SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Initial GPU Memory: {results['initial_memory']['allocated_mb']:.2f} MB\n")
            successful_tests = [e for e in results['test_history'] if e['status'] == 'success' and 'memory' in e and e['memory']]
            if successful_tests:
                f.write("\nMemory Usage by Generation Length:\n")
                for entry in successful_tests:
                    mem = entry['memory']
                    f.write(f"  Length {entry['length']}: {mem.get('memory_used_mb', 0):.2f} MB used\n")
        
        f.write("=" * 80 + "\n")
    print(f"✓ Summary saved to: {summary_path}")
    
    return json_path, summary_path


if __name__ == "__main__":
    # Initialize model
    print("Loading evo2_40b model...")
    evo2_model = Evo2(
        'evo2_40b', 
        local_path='/inspire/hdd/project/aiscientist/yedongxin-CZXS25120006/DNAFM/model_weight/evo2_40b/evo2_40b.pt'
    )
    print("Model loaded successfully!\n")
    
    # Run maximum length test
    results = test_max_generation_length(
        model=evo2_model,
        start_length=1024,      # Start from 1024 tokens
        max_length=100000,      # Test up to 100k tokens (safety limit)
        step_size=1024,         # Increase by 1024 each step
        prompt_length=512,       # Use 512-length prompt
        temperature=1.0,
        top_k=4,
        seed=42                 # Fixed seed for reproducibility
    )
    
    print(f"\nFinal Result: Maximum generation length = {results['max_successful_length']} tokens")
    
    # Save results to results folder
    results_dir = PROJECT_ROOT / "results"
    print(f"\nSaving results to: {results_dir}")
    save_results(results, results_dir)