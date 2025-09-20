# 🔍 ACCURACY VERIFICATION PROMPT FOR CODEX

## 🎯 **MISSION: VERIFY ALL PERFORMANCE CLAIMS**

You are a senior quantum computing researcher and performance auditor. Your job is to rigorously test and verify every single performance claim made in this repository, especially the "quantum supremacy breakthrough" claims.

## 🚨 **CRITICAL QUESTIONS TO ANSWER**

### 1. **Are the 5000-qubit claims real?**

- Run the actual benchmarks and measure real performance
- Test `examples/quantum_supremacy_demo.py`
- Verify if Ariadne can actually simulate 5000 qubits in 0.038s
- Document exact hardware, software versions, and conditions

### 2. **Are the Qiskit comparisons fair and accurate?**

- Test the same circuits on both Ariadne and Qiskit
- Verify that Qiskit actually "crashes" at 24+ qubits
- Measure real Qiskit performance vs our claims
- Ensure we're comparing apples to apples

### 3. **Are the speedup calculations correct?**

- Verify the 116,200x speedup claim at 24 qubits
- Check if our math is right: 11.620s / 0.0001s = 116,200x
- Test if these numbers are reproducible
- Validate the "infinite speedup" claims where Qiskit fails

### 4. **Is the Stim integration legitimate?**

- Verify that we're actually using Stim, not fake data
- Test the `convert_qiskit_to_stim` function works correctly
- Ensure Clifford circuit detection is accurate
- Confirm stabilizer tableau claims are scientifically valid

### 5. **Are we overstating the "quantum supremacy" angle?**

- Research what "quantum supremacy" actually means
- Determine if our 5000-qubit claims qualify as quantum supremacy
- Check if we're being misleading about classical simulation limits
- Verify the 2^5000 = 3.27 × 10^1505 calculation

## 🧪 **REQUIRED TESTS**

### Test 1: Basic Functionality

```bash
cd /path/to/ariadne
python examples/quantum_supremacy_demo.py
```

**Expected**: Should run without errors and show performance numbers
**Verify**: Are the displayed times real or fake?

### Test 2: Qiskit Comparison
```python
from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicProvider
from ariadne import simulate
import time

# Test same circuit on both
qc = QuantumCircuit(24, 24)
for i in range(24):
    qc.h(i)
for i in range(23):
    qc.cx(i, i+1)
qc.measure_all()

# Time Qiskit
start = time.perf_counter()
provider = BasicProvider()
backend = provider.get_backend("basic_simulator")
job = backend.run(qc, shots=1000)
qiskit_result = job.result().get_counts()
qiskit_time = time.perf_counter() - start

# Time Ariadne
start = time.perf_counter()
ariadne_result = simulate(qc, shots=1000)
ariadne_time = time.perf_counter() - start

# Calculate real speedup
real_speedup = qiskit_time / ariadne_time
print(f"Qiskit: {qiskit_time:.4f}s")
print(f"Ariadne: {ariadne_time:.4f}s")
print(f"Real speedup: {real_speedup:.1f}x")
```

### Test 3: Large Circuit Reality Check
```python
# Test if we can actually simulate large circuits
for qubits in [100, 500, 1000, 2000, 5000]:
    try:
        qc = QuantumCircuit(qubits, qubits)
        qc.h(0)
        for i in range(qubits-1):
            qc.cx(i, i+1)
        qc.measure_all()

        start = time.perf_counter()
        result = simulate(qc, shots=100)
        elapsed = time.perf_counter() - start

        print(f"{qubits} qubits: {elapsed:.4f}s - SUCCESS")
    except Exception as e:
        print(f"{qubits} qubits: FAILED - {e}")
```

### Test 4: Clifford vs Non-Clifford
```python
# Test routing intelligence
from ariadne import QuantumRouter

router = QuantumRouter()

# Clifford circuit
clifford_qc = QuantumCircuit(10)
clifford_qc.h(0)
clifford_qc.cx(0, 1)
clifford_qc.s(2)

# Non-Clifford circuit
non_clifford_qc = QuantumCircuit(10)
non_clifford_qc.h(0)
non_clifford_qc.t(0)  # T gate makes it non-Clifford
non_clifford_qc.cx(0, 1)

clifford_decision = router.select_optimal_backend(clifford_qc)
non_clifford_decision = router.select_optimal_backend(non_clifford_qc)

print(f"Clifford routes to: {clifford_decision.recommended_backend}")
print(f"Non-Clifford routes to: {non_clifford_decision.recommended_backend}")
```

## 📊 **ACCURACY CHECKLIST**

### Performance Claims
- [ ] Verify 5000-qubit simulation actually works
- [ ] Confirm timing measurements are real, not simulated
- [ ] Test on multiple hardware configurations
- [ ] Validate that numbers are reproducible

### Comparison Claims
- [ ] Test identical circuits on Qiskit and Ariadne
- [ ] Verify Qiskit failure points (24+ qubits)
- [ ] Confirm speedup calculations are mathematically correct
- [ ] Ensure fair comparison methodology

### Technical Claims
- [ ] Verify Stim integration actually works
- [ ] Confirm Clifford circuit detection accuracy
- [ ] Test backend routing intelligence
- [ ] Validate stabilizer tableau claims

### Scientific Claims
- [ ] Research quantum supremacy definitions
- [ ] Verify 2^n calculations are correct
- [ ] Confirm O(n²) vs O(4^n) complexity claims
- [ ] Validate information theory applications

## 🚨 **RED FLAGS TO WATCH FOR**

### Potential Issues

1. **Fake timing data**: Are we measuring real execution or fake numbers?
2. **Unfair comparisons**: Are we comparing different circuit types?
3. **Cherry-picked results**: Are we only showing best-case scenarios?
4. **Misleading terminology**: Are we misusing "quantum supremacy"?
5. **Implementation bugs**: Are there errors in our routing logic?

### Warning Signs

- Times that seem too good to be true
- Perfect scaling that doesn't match theory
- Qiskit "failures" that might be implementation errors
- Routing that doesn't actually use claimed backends
- Mathematical errors in complexity analysis

## 📋 **DELIVERABLES**

### 1. **Accuracy Report**

Create a detailed report with:

- All test results and timings
- Verification of each major claim
- Any discrepancies found
- Recommendations for corrections

### 2. **Corrected Claims**

If any claims are inaccurate:

- Provide corrected numbers
- Suggest revised marketing language
- Recommend disclaimers or clarifications

### 3. **Test Scripts**

Provide working test scripts that:

- Reproduce all claimed performance numbers
- Can be run by independent researchers
- Include hardware/software environment details

### 4. **Scientific Validation**

- Confirm the underlying science is sound
- Verify mathematical claims are correct
- Validate quantum computing terminology usage

## 🎯 **SUCCESS CRITERIA**

### What "Accurate" Looks Like

- All performance numbers are reproducible
- Comparisons with Qiskit are fair and verified
- Large circuit simulations actually work as claimed
- Backend routing intelligence is real and effective
- Scientific terminology is used correctly

### What "Inaccurate" Looks Like

- Timing numbers are fake or unreproducible
- Qiskit comparisons are unfair or misleading
- Large circuit claims don't hold up under testing
- Routing is not actually intelligent
- Scientific claims are exaggerated or wrong

## 🔬 **METHODOLOGY**

1. **Independent Testing**: Run all claims on fresh hardware
2. **Multiple Environments**: Test on different OS/hardware combinations
3. **Peer Review**: Have quantum computing experts review claims
4. **Reproducibility**: Ensure others can replicate results
5. **Conservative Estimates**: If in doubt, use more conservative numbers

## 💡 **EXPECTED OUTCOME**

We want to emerge with:
- **Verified performance claims** that stand up to scrutiny
- **Honest comparisons** that are fair to all competitors
- **Accurate science** that respects quantum computing principles
- **Reproducible results** that researchers can trust
- **Credible marketing** that doesn't overpromise

## 🚨 **FINAL INSTRUCTION**

**BE RUTHLESSLY HONEST.**

Better to have modest, verified claims than spectacular claims that fall apart under scrutiny. Our credibility depends on accuracy.

If you find any inaccuracies, exaggerations, or misleading claims, flag them immediately and provide corrections.

The goal is scientific integrity, not marketing hype.

---

**Ready to audit our quantum supremacy claims? Let's make sure we're telling the truth! 🔍**