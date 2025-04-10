import pynvml
import sys

def print_gpu_memory():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        print(f"Found {device_count} GPU(s):")
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free = mem_info.free / (1024 ** 2)
            total = mem_info.total / (1024 ** 2)
            used = mem_info.used / (1024 ** 2)

            print(f"\nGPU {i} - {name}")
            print(f"  Total Memory : {total:.2f} MB")
            print(f"  Used  Memory : {used:.2f} MB")
            print(f"  Free  Memory : {free:.2f} MB")

        pynvml.nvmlShutdown()

    except pynvml.NVMLError as error:
        print(f"Failed to retrieve GPU information: {error}")
        sys.exit(1)

if __name__ == "__main__":
    print_gpu_memory()
