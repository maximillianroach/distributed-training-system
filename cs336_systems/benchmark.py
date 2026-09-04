import timeit
import sys
from pathlib import Path
import torch
import statistics
import numpy as np

# gives us the parent of the current file
parent_dir = str(Path(__file__).resolve().parent.parent)

# appends the desired file directory to the parent directory name
basics_dir = str(Path(parent_dir) / "cs336-basics")

'''
adds the basics_dir to the array of file paths that will be searched when 
resolving import
'''
if basics_dir not in sys.path:
    sys.path.append(basics_dir)
from cs336_basics import model, data, nn_utils, optimizer


def benchmark(
        # model hyperparameters
        vocab_size: int = 10000, 
        context_length: int = 256,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 16,
        d_ff: int = 1344,
        rope_theta: int = 10000, 
        batch_size: int = 32,
        # benchmarking params
        bench_type: str = "fbo", # f for forward-only, fb for forward-backward, fbo for foward-backward with optimizer
        warmup_steps: int=5, 
        num_steps: int=10,
):
    benchmark_times = []

    # initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mod = model.BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)
    
    # generate batch of data
    dataset = np.random.randint(0, vocab_size, 100000)
    x, y = data.get_batch(dataset, batch_size, context_length, device)

    # FOWARD ONLY
    if bench_type == "f":
        # warmup
        for t in range(warmup_steps):
            with torch.inference_mode():
                mod(x)

        if device == "cuda":
            torch.cuda.synchronize()

        for t in range(num_steps):
            if device == "cuda":
                torch.cuda.synchronize()
            # start
            start = timeit.default_timer()
            
            with torch.inference_mode():
                mod(x)
            if device == "cuda":
                torch.cuda.synchronize()

            # end
            end = timeit.default_timer()
            benchmark_times.append(end - start)

    # FORWARD + BACKWARD
    elif bench_type == "fb":
        # warmup
        for t in range(warmup_steps):
            logits = mod(x)
            loss = nn_utils.cross_entropy(logits, y)
            loss.backward()
            mod.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()

        for t in range(num_steps):
            if device == "cuda":
                torch.cuda.synchronize()
            start = timeit.default_timer()
            logits = mod(x)
            loss = nn_utils.cross_entropy(logits, y)
            loss.backward()
            if device == "cuda":
                torch.cuda.synchronize()
            end = timeit.default_timer()
            benchmark_times.append(end - start)

    # FORWARD + BACKWARD + OPTIMIZER 
    elif bench_type == "fbo":
        opt = optimizer.AdamW(mod.parameters())
        # warmup
        for t in range(warmup_steps):
            opt.zero_grad()
            logits = mod(x)
            loss = nn_utils.cross_entropy(logits, y)
            loss.backward()
            opt.step()

        if device == "cuda":
            torch.cuda.synchronize()
            
        for t in range(num_steps):
            if device == "cuda":
                torch.cuda.synchronize()
            start = timeit.default_timer()
            opt.zero_grad()
            logits = mod(x)
            loss = nn_utils.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            end = timeit.default_timer()
            benchmark_times.append(end - start)
    else:
        raise ValueError(f"Unknown Benchmarking Type supplied: {bench_type}")

    return (statistics.mean(benchmark_times), statistics.stdev(benchmark_times))

def pytorch_benchmark():
    pass

def main():
    mean, std = benchmark()
    print(f"mean: {mean}")
    print(f"std: {std}")

if __name__ == "__main__":
    main()