"""
Launcher: distribuye el grid search de μ entre 2 GPUs y agrega resultados.
"""
import subprocess
import sys
import json
import time
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Valores de mu a probar (17 valores, escala logarítmica)
mu_values = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.0245, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]

# Dividir equitativamente entre 2 GPUs (intercalado para balancear carga)
gpu0_mus = mu_values[0::2]  # indices pares
gpu1_mus = mu_values[1::2]  # indices impares

gpu0_str = ','.join(f'{v:.4f}' for v in gpu0_mus)
gpu1_str = ','.join(f'{v:.4f}' for v in gpu1_mus)

print("="*70)
print("   DUAL-GPU GRID SEARCH DE μ PARA wTDV-QSM")
print("="*70)
print(f"\nGPU 0: {len(gpu0_mus)} valores: {[f'{v:.4f}' for v in gpu0_mus]}")
print(f"GPU 1: {len(gpu1_mus)} valores: {[f'{v:.4f}' for v in gpu1_mus]}")
print()

tic = time.time()

# Lanzar ambos procesos en paralelo
p0 = subprocess.Popen(
    [sys.executable, 'testing/mu_worker.py', '0', 'figures/mu_results_gpu0.json', gpu0_str],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)
p1 = subprocess.Popen(
    [sys.executable, 'testing/mu_worker.py', '1', 'figures/mu_results_gpu1.json', gpu1_str],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)

# Stream output intercalado
import threading
import queue

output_queue = queue.Queue()

def reader(proc, name):
    for line in proc.stdout:
        output_queue.put(f"{line.rstrip()}")
    proc.wait()

t0 = threading.Thread(target=reader, args=(p0, "GPU0"))
t1 = threading.Thread(target=reader, args=(p1, "GPU1"))
t0.start()
t1.start()

# Imprimir output en tiempo real
while t0.is_alive() or t1.is_alive() or not output_queue.empty():
    try:
        line = output_queue.get(timeout=0.5)
        print(line)
    except queue.Empty:
        pass

t0.join()
t1.join()

# Drenar cola
while not output_queue.empty():
    print(output_queue.get())

toc = time.time()
print(f"\n⏱️  Tiempo total dual-GPU: {toc-tic:.1f}s")

# Verificar exit codes
if p0.returncode != 0 or p1.returncode != 0:
    print(f"⚠️  GPU 0 exit code: {p0.returncode}, GPU 1 exit code: {p1.returncode}")
    sys.exit(1)

# ============================================================
# AGREGAR RESULTADOS
# ============================================================
with open('figures/mu_results_gpu0.json') as f:
    results0 = json.load(f)
with open('figures/mu_results_gpu1.json') as f:
    results1 = json.load(f)

all_results = {**results0, **results1}

# Ordenar por mu
sorted_mus = sorted(all_results.keys(), key=float)

print("\n" + "="*70)
print("   TABLA COMPARATIVA DE RESULTADOS")
print("="*70)
header = f"{'μ':>8} | {'Iter':>4} | {'DataFid':>10} | {'Sharp':>10} | {'BrStd':>8} | {'SNR':>6} | {'HF_E':>10} | {'ConeStr':>10} | {'DynRange':>8}"
print(header)
print("-"*len(header))

for mu_str in sorted_mus:
    m = all_results[mu_str]
    mu_val = float(mu_str)
    print(f"{mu_val:>8.4f} | {m['n_iter']:>4d} | {m['data_fidelity']:>10.6f} | {m['sharpness']:>10.6f} | {m['brain_std']:>8.6f} | {m['snr']:>6.2f} | {m['hf_energy']:>10.6f} | {m['cone_streak']:>10.6f} | {m['dynamic_range']:>8.4f}")

# ============================================================
# SCORING COMPUESTO
# ============================================================
metric_keys = ['data_fidelity', 'sharpness', 'brain_std', 'snr', 'cone_streak', 'dynamic_range', 'hf_energy']
metric_arrays = {}
mu_float_sorted = [float(mu_str) for mu_str in sorted_mus]

for key in metric_keys:
    metric_arrays[key] = np.array([all_results[mu_str][key] for mu_str in sorted_mus])

def normalize(arr):
    if arr.max() == arr.min():
        return np.ones_like(arr) * 0.5
    return (arr - arr.min()) / (arr.max() - arr.min())

scores = (
    0.25 * normalize(metric_arrays['sharpness']) +
    0.20 * (1 - normalize(metric_arrays['data_fidelity'])) +
    0.15 * normalize(metric_arrays['brain_std']) +
    0.15 * normalize(metric_arrays['snr']) +
    0.15 * (1 - normalize(metric_arrays['cone_streak'])) +
    0.10 * normalize(metric_arrays['dynamic_range'])
)

print("\n" + "="*70)
print("   RANKING POR SCORE COMPUESTO")
print("="*70)
print(f"  Pesos: Sharpness=0.25, DataFid=0.20, BrainStd=0.15, SNR=0.15, ConeStreak=0.15, DynRange=0.10")
print(f"{'Rank':>4} | {'μ':>8} | {'Score':>7} | {'Sharp':>10} | {'DataFid':>10} | {'SNR':>6} | {'Iter':>4}")
print("-"*70)

ranking = np.argsort(-scores)
for rank, idx in enumerate(ranking):
    mu_val = mu_float_sorted[idx]
    mu_str = sorted_mus[idx]
    m = all_results[mu_str]
    marker = " ◀ ACTUAL" if abs(mu_val - 0.0245) < 0.0001 else ""
    print(f"{rank+1:>4} | {mu_val:>8.4f} | {scores[idx]:>7.4f} | {m['sharpness']:>10.6f} | {m['data_fidelity']:>10.6f} | {m['snr']:>6.2f} | {m['n_iter']:>4d}{marker}")

best_idx = ranking[0]
best_mu = mu_float_sorted[best_idx]
actual_idx = mu_float_sorted.index(0.0245)

print(f"\n🏆 MEJOR μ = {best_mu:.4f} (Score: {scores[best_idx]:.4f})")
print(f"   vs μ actual = 0.0245 (Score: {scores[actual_idx]:.4f})")
improvement = (scores[best_idx] - scores[actual_idx]) / scores[actual_idx] * 100
print(f"   Mejora: {improvement:+.1f}%")

# ============================================================
# ANÁLISIS DETALLADO: Trade-offs clave
# ============================================================
print("\n" + "="*70)
print("   ANÁLISIS DE TRADE-OFFS")
print("="*70)

# Encontrar el "codo" de la L-curve
# El punto con mejor balance sharpness/data_fidelity
sharpness_norm = normalize(metric_arrays['sharpness'])
datafid_norm = 1 - normalize(metric_arrays['data_fidelity'])
l_curve_score = np.sqrt(sharpness_norm**2 + datafid_norm**2)
lcurve_best_idx = np.argmax(l_curve_score)
print(f"\nL-Curve óptimo (balance Sharpness vs DataFid): μ = {mu_float_sorted[lcurve_best_idx]:.4f}")

# Máxima nitidez (más agresivo, posible ruido)
max_sharp_idx = np.argmax(metric_arrays['sharpness'])
print(f"Máxima Sharpness: μ = {mu_float_sorted[max_sharp_idx]:.4f} (Sharp={metric_arrays['sharpness'][max_sharp_idx]:.6f})")

# Mínima data infidelity (más conservador)
min_datafid_idx = np.argmin(metric_arrays['data_fidelity'])
print(f"Mínima Data Infidelity: μ = {mu_float_sorted[min_datafid_idx]:.4f} (DataFid={metric_arrays['data_fidelity'][min_datafid_idx]:.6f})")

# Máximo SNR
max_snr_idx = np.argmax(metric_arrays['snr'])
print(f"Máximo SNR: μ = {mu_float_sorted[max_snr_idx]:.4f} (SNR={metric_arrays['snr'][max_snr_idx]:.2f})")

# ============================================================
# VISUALIZACIÓN
# ============================================================
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Análisis de μ para wTDV-QSM (Dual-GPU)', fontsize=16, fontweight='bold')

# Plot 1: Score vs mu
ax = axes[0, 0]
ax.semilogx(mu_float_sorted, scores, 'bo-', markersize=8, linewidth=2)
ax.axvline(x=0.0245, color='r', linestyle='--', linewidth=2, label='μ actual (0.0245)')
ax.axvline(x=best_mu, color='g', linestyle='--', linewidth=2, label=f'μ óptimo ({best_mu:.4f})')
ax.set_xlabel('μ')
ax.set_ylabel('Score Compuesto')
ax.set_title('Score vs μ')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Sharpness vs mu
ax = axes[0, 1]
ax.semilogx(mu_float_sorted, metric_arrays['sharpness'], 'go-', markersize=8, linewidth=2)
ax.axvline(x=0.0245, color='r', linestyle='--', alpha=0.5)
ax.axvline(x=best_mu, color='g', linestyle='--', alpha=0.5)
ax.set_xlabel('μ')
ax.set_ylabel('Sharpness')
ax.set_title('Sharpness (Gradiente Medio)')
ax.grid(True, alpha=0.3)

# Plot 3: Data Fidelity vs mu
ax = axes[0, 2]
ax.semilogx(mu_float_sorted, metric_arrays['data_fidelity'], 'ro-', markersize=8, linewidth=2)
ax.axvline(x=0.0245, color='r', linestyle='--', alpha=0.5)
ax.axvline(x=best_mu, color='g', linestyle='--', alpha=0.5)
ax.set_xlabel('μ')
ax.set_ylabel('Data Fidelity (RMSE)')
ax.set_title('Residuo de Data Fidelity')
ax.grid(True, alpha=0.3)

# Plot 4: SNR vs mu
ax = axes[1, 0]
ax.semilogx(mu_float_sorted, metric_arrays['snr'], 'mo-', markersize=8, linewidth=2)
ax.axvline(x=0.0245, color='r', linestyle='--', alpha=0.5)
ax.axvline(x=best_mu, color='g', linestyle='--', alpha=0.5)
ax.set_xlabel('μ')
ax.set_ylabel('SNR')
ax.set_title('Signal-to-Noise Ratio')
ax.grid(True, alpha=0.3)

# Plot 5: Cone Streaking vs mu
ax = axes[1, 1]
ax.semilogx(mu_float_sorted, metric_arrays['cone_streak'], 'co-', markersize=8, linewidth=2)
ax.axvline(x=0.0245, color='r', linestyle='--', alpha=0.5)
ax.axvline(x=best_mu, color='g', linestyle='--', alpha=0.5)
ax.set_xlabel('μ')
ax.set_ylabel('Cone Streak')
ax.set_title('Streaking en Cono Nulo')
ax.grid(True, alpha=0.3)

# Plot 6: Iterations vs mu
ax = axes[1, 2]
n_iters = [all_results[mu_str]['n_iter'] for mu_str in sorted_mus]
ax.semilogx(mu_float_sorted, n_iters, 'ko-', markersize=8, linewidth=2)
ax.axvline(x=0.0245, color='r', linestyle='--', alpha=0.5)
ax.axvline(x=best_mu, color='g', linestyle='--', alpha=0.5)
ax.set_xlabel('μ')
ax.set_ylabel('Iteraciones')
ax.set_title('Convergencia (# Iteraciones)')
ax.grid(True, alpha=0.3)

# Plot 7: Dynamic Range vs mu
ax = axes[2, 0]
ax.semilogx(mu_float_sorted, metric_arrays['dynamic_range'], 'o-', markersize=8, linewidth=2, color='orange')
ax.axvline(x=0.0245, color='r', linestyle='--', alpha=0.5)
ax.axvline(x=best_mu, color='g', linestyle='--', alpha=0.5)
ax.set_xlabel('μ')
ax.set_ylabel('Dynamic Range')
ax.set_title('Rango Dinámico del Cerebro')
ax.grid(True, alpha=0.3)

# Plot 8: Brain Std vs mu
ax = axes[2, 1]
ax.semilogx(mu_float_sorted, metric_arrays['brain_std'], 'bs-', markersize=8, linewidth=2)
ax.axvline(x=0.0245, color='r', linestyle='--', alpha=0.5)
ax.axvline(x=best_mu, color='g', linestyle='--', alpha=0.5)
ax.set_xlabel('μ')
ax.set_ylabel('Brain Std')
ax.set_title('Contraste (Std cerebro)')
ax.grid(True, alpha=0.3)

# Plot 9: L-curve (Data Fid vs Sharpness)
ax = axes[2, 2]
ax.plot(metric_arrays['data_fidelity'], metric_arrays['sharpness'], 'ko-', markersize=8, linewidth=2)
for i, mu_val in enumerate(mu_float_sorted):
    ax.annotate(f'{mu_val:.3f}', (metric_arrays['data_fidelity'][i], metric_arrays['sharpness'][i]),
                textcoords="offset points", xytext=(5,5), fontsize=7)
ax.plot(metric_arrays['data_fidelity'][actual_idx], metric_arrays['sharpness'][actual_idx], 
        'r*', markersize=15, label='Actual (0.0245)')
ax.plot(metric_arrays['data_fidelity'][best_idx], metric_arrays['sharpness'][best_idx], 
        'g*', markersize=15, label=f'Óptimo ({best_mu:.4f})')
ax.set_xlabel('Data Fidelity (↓ mejor)')
ax.set_ylabel('Sharpness (↑ mejor)')
ax.set_title('L-Curve: Data Fidelity vs Sharpness')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/mu_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Gráficos guardados en figures/mu_analysis.png")

# Convergence curves comparison
fig2, ax2 = plt.subplots(figsize=(12, 6))
for mu_str in sorted_mus:
    mu_val = float(mu_str)
    updates = all_results[mu_str]['updates']
    if mu_val in [0.001, 0.01, 0.0245, 0.05, 0.1, 0.5, best_mu]:
        style = '-' if mu_val != 0.0245 else '--'
        width = 3 if mu_val in [0.0245, best_mu] else 1
        ax2.semilogy(range(len(updates)), updates, style, linewidth=width, 
                     label=f'μ={mu_val:.4f} ({len(updates)} iter)')
ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='tolUpdate=1.0')
ax2.set_xlabel('Iteración')
ax2.set_ylabel('Update (%)')
ax2.set_title('Curvas de Convergencia para diferentes μ')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/mu_convergence.png', dpi=150, bbox_inches='tight')
print(f"📊 Convergencia guardada en figures/mu_convergence.png")

# Guardar resultados agregados
with open('figures/mu_results_all.json', 'w') as f:
    json.dump({
        'mu_values': mu_float_sorted,
        'scores': scores.tolist(),
        'best_mu': best_mu,
        'best_score': float(scores[best_idx]),
        'actual_mu': 0.0245,
        'actual_score': float(scores[actual_idx]),
        'results': all_results,
    }, f, indent=2)

print(f"\n✅ Análisis completo. Resultados en figures/mu_results_all.json")
