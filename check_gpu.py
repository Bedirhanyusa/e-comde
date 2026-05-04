import torch
import sys

print(f"Python : {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA   : {torch.version.cuda}")
print()

if not torch.cuda.is_available():
    print("UYARI: CUDA kullanilabilir degil. CPU modunda calisacak.")
    sys.exit(0)

gpu = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

print(f"GPU    : {gpu}")
print(f"Compute: sm_{cap[0]}{cap[1]}")
print(f"VRAM   : {vram:.1f} GB")

if cap[0] < 8:
    print("UYARI: sm_80 altinda — BERTurk egitimi cok yavas olabilir.")
elif cap >= (12, 0):
    print("Blackwell mimarisi (sm_120) tespit edildi.")
else:
    print("Desteklenen mimari.")

# Kisa tensor testi
x = torch.randn(3, 3, device='cuda')
print(f"\nGPU tensor testi: {x.shape} — OK")
