# Appendix D: W5700X Vulkan Validation

The W5700X has been validated with trueno's wgpu backend on **Metal** (macOS)
with documented performance numbers (trueno book, 2026-01-03). The intel box
runs **Linux**, so the backend will be **Vulkan** (not Metal). Vulkan support
for RDNA 1 on Linux via Mesa RADV is mature and well-tested.

**Action item**: Run trueno GPU tests on intel via Vulkan to confirm parity
with Metal benchmarks before relying on W5700X for compute tasks.
