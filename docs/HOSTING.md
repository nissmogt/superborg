# Hosting

Two questions: where to host the browser viewer with models larger than the 111 MB one, and
how to run the most capable open models where privacy is enforced by hardware and checked by
the client rather than promised in a policy. Sources were read on 2026-10-16 and every URL
below answered `curl -sI` that day (a GET where the host refuses HEAD; caniuse.com wants a
browser `User-Agent`). Prices and browser versions change; re-check before buying.

## 1. What runs where today

`wasm/build.sh` targets wasm32-wasi with `-msimd128 -DBORG_NO_THREADS` and a 4 GB
`--max-memory`: one thread, fixed-width SIMD128, no SharedArrayBuffer. `wasm/worker.js`
fetches `model.json` and the parts, keeps them in IndexedDB, copies them into linear memory
and runs the model there. The prompt, the tokens and the trace never leave the device; the
only network traffic is GET requests for static files, so a host learns an IP address and
which model was fetched.

* wasm32 linear memory is capped at 4 GB (https://v8.dev/blog/4gb-wasm-memory); the `.borg`
  file, the f16 KV cache and the activations share it, so a model file above about 3 GB does
  not fit, and on phones the device's RAM binds first.
* One thread. Measured in headless Chromium with SmolLM2-135M q4 (110.8 MB): 32 ms per
  decoded token, 6.6-9.4 tok/s with the full trace (README, "In the browser"). Single-core
  decoding is bound by weight bytes read per token, so expect about 4x slower for
  Qwen2.5-0.5B q4 (400 MB) and 11x for Qwen2.5-1.5B q4 (1,187 MB, README speed table), near
  0.35 s per token (extrapolated, not measured).
* Realistic today: up to about 1.5 B parameters at q4, slowly. 7B at q4 (about 4 GB) does
  not fit in wasm32 and would take seconds per token; see Memory64 and section 3.
  `wasm/make_artifact.py` splits models into 14,500,000-byte parts by default, sized for
  hosts that wrap pages; a CDN wants larger parts.

## 2. Static hosting of the viewer and model parts

Today's build needs no special response headers, so any HTTPS static host can serve the
viewer; hosts differ in where multi-GB parts can live (per-file caps, CORS, cost) and, for a
future threaded build only, in whether custom headers can be set.

### Recommended: a static Hugging Face Space plus a public model repo

* "Static Spaces are free for everyone" and are reachable at `https://<subdomain>.hf.space`
  (https://huggingface.co/docs/hub/spaces-overview); the Space holds `index.html`,
  `worker.js`, `borg.wasm` and `model.json`. Space repos are reported to stop at
  "Repository storage limit reached (Max: 1 GB)", a forum figure, not the docs
  (https://discuss.huggingface.co/t/1gb-storage-limit-in-spaces/170196), so the parts go in
  a public model repo: "no per-repo size limit for models and datasets", files under 200 GB
  recommended, 500 GB hard limit, CloudFront delivery, free best-effort public storage
  (https://huggingface.co/docs/hub/storage-limits).
* CORS, verified with `curl -sI` on 2026-10-16: a HEAD for a `/resolve/main/` file with an
  `Origin` header answers 302 with that origin echoed in `access-control-allow-origin`, and
  the redirect target on `us.aws.cdn.hf.co` answers `Range: bytes=0-1023` with 206 and
  `access-control-allow-origin: *`
  (https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors).
  So the viewer can live on any origin, and `model.json` can carry absolute part URLs since
  `worker.js` resolves paths with `new URL(p, base)`.
* Rate limit: all `/resolve/` URLs share one bucket of 3,000 requests per 5-minute window
  per anonymous IP, 5,000 for free accounts and 12,000 for PRO, a table the page dates
  "September '25" (https://huggingface.co/docs/hub/rate-limits). Use parts of 64-256 MB
  (`--part-size 134217728` is 128 MB): a 4 GB model is then 16-64 requests, and the CDN's
  `Range` support lets `worker.js` resume a partial part once it sends one.
* Client caching: IndexedDB, the Cache API and OPFS share one quota per origin, up to 60% of
  disk in Chromium, 10% of disk or 10 GiB in Firefox until `navigator.storage.persist()` is
  granted, about 60% in Safari (macOS 14, iOS 17 onward); with tracking prevention on,
  Safari deletes an origin's script-created data after seven days of browser use without
  user interaction, all-or-nothing per origin
  (https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria).
  So call `persist()`, check `navigator.storage.estimate()` before downloading, catch
  `QuotaExceededError`, and tell Safari users a model is downloaded again after a week
  without a visit. The Cache API or OPFS would avoid the copy `cachePut` makes today.
* Headers: the README front matter `custom_headers` accepts exactly the three lowercase
  `cross-origin-*-policy` headers, to "enable powerful features like SharedArrayBuffer"
  (https://huggingface.co/docs/hub/spaces-config-reference); needed only for a threaded
  build and only at the `.hf.space` URL, since huggingface.co's own responses send
  `cross-origin-opener-policy: same-origin` and no COEP. The user supplies a Hugging Face
  account (free) and uploads the parts with the Hub CLI.

### Second option: Cloudflare Workers static assets plus R2

Cloudflare's Pages docs now point new projects to Workers
(https://developers.cloudflare.com/workers/static-assets/migration-guides/migrate-from-pages/).
Workers static assets: 25 MiB per file, a `_headers` file of up to 100 rules of 2,000
characters, static asset requests free
(https://developers.cloudflare.com/workers/platform/limits/). Parts therefore go to R2 on a
custom domain with a CORS rule: $0.015 per GB-month, Class A writes $4.50 and Class B reads
$0.36 per million, no egress charge, 10 GB-month, 1 M Class A and 10 M Class B free each
month (https://developers.cloudflare.com/r2/pricing/); a 5 GB model is inside the free tier,
a 50 GB catalogue about $0.60 per month (derived). The user supplies a Cloudflare account
and a domain for the bucket.

### What GitHub Pages can and cannot do

Pages sites may be no larger than 1 GB with a soft bandwidth limit of 100 GB per month, and
the limits page says nothing about custom response headers
(https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits);
Git blocks files over 100 MiB
(https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
So Pages can host the viewer and models up to about 1 GB in parts under 100 MiB; larger
models come from the Hub or R2. Releases allow assets under 2 GiB with no bandwidth limit
(https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases), but
`release-assets.githubusercontent.com` answered a HEAD with an `Origin` header on
2026-10-16 without `Access-Control-Allow-Origin`, so another origin cannot read them.

### Threads, COOP, COEP and Memory64 (only if a new build is added)

SharedArrayBuffer works in Chrome 68+, Firefox 79+ and Safari 15.2+ only in a secure,
cross-origin-isolated context: `Cross-Origin-Opener-Policy: same-origin` plus a
`Cross-Origin-Embedder-Policy` (https://caniuse.com/sharedarraybuffer). `COEP:
credentialless` is Chrome/Edge 96+ and Firefox 119+ but no Safari through 27.2
(https://caniuse.com/mdn-http_headers_cross-origin-embedder-policy_credentialless), so send
`require-corp` and fetch parts with `mode: "cors"`, which the Hub CDN's
`access-control-allow-origin: *` satisfies. wasi-threads "is considered a legacy proposal"
(https://raw.githubusercontent.com/WebAssembly/wasi-threads/main/README.md), so the
maintained path is Emscripten `-pthread`, whose docs warn that "Pthreads code will not work
in deployed environment unless these headers are correctly set"
(https://emscripten.org/docs/porting/pthreads.html).

Memory64 is on by default in Chrome/Edge 133+ and Firefox 134+, in no shipping Safari,
73.55% of global usage (https://caniuse.com/wf-wasm-memory64); WebKit added it to Safari
Technology Preview 251 on August 26, 2026
(https://webkit.org/blog/18194/release-notes-for-safari-technology-preview-251/). V8 caps a
memory64 instance at 262,144 pages, 16 GB
(https://chromium.googlesource.com/v8/v8/+/main/src/wasm/wasm-limits.h). Keep wasm32 and
SIMD128 as the shipped build; an optional wasm64 build buys at most 16 GB in Chrome, still
on one thread. The route past that is the GPU.

## 3. Bigger models client-side: WebGPU

WebGPU keeps weights in per-tensor GPU buffers, so the 4 GB cap does not apply. WebLLM's
prebuilt list (library v0_2_86) needs 5,001 MB of GPU memory for Llama-3.1-8B-Instruct
q4f16_1 (4,598 MB with a 1k context) and 31,153 MB for Llama-3.1-70B-Instruct q3f16_1
(https://raw.githubusercontent.com/mlc-ai/web-llm/main/src/config.ts); its paper reports
"up to 80% native performance on the same device" (https://arxiv.org/abs/2412.15803).
transformers.js runs ONNX models with `device: "webgpu"` and puts global WebGPU support at
about 85% as of March 2026 (https://huggingface.co/docs/transformers.js/guides/webgpu).
Browser support (gpuweb status wiki): Chrome on Mac, Windows, ChromeOS and Android, Linux
partial; Firefox on Windows and macOS, Linux Nightly only, Android off; Safari on by default
in macOS 26, iOS 26, iPadOS 26 and visionOS 26
(https://raw.githubusercontent.com/wiki/gpuweb/gpuweb/Implementation-Status.md); 87.82%
global usage (https://caniuse.com/webgpu). Plan for borg: a WebGPU backend with WGSL kernels
for the q4/q8 matmuls, attention and norms, reading the same `.borg` parts into GPU buffers,
with the CPU wasm engine as the fallback; targets 7-8B at q4 on machines with 8 GB or more
of GPU or unified memory, 70B only on 32 GB+ unified-memory Macs. A large piece of work.

## 4. The most capable models with enforced privacy

Enforced means the prompt is decrypted only inside a CPU TEE (Intel TDX or AMD SEV-SNP) with
the GPU in NVIDIA confidential-computing mode; verifiable means the client checks the
attestation report against a published measurement before sending data. A retention policy
is neither.

### Attested providers an individual can use today

**Tinfoil.** Models run in confidential VMs (SEV-SNP or TDX) with Hopper or Blackwell GPUs
in CC mode. CI builds a reproducible CVM image and publishes its measurement to Sigstore;
the client SDK fetches the enclave's attestation report, compares measurements ("If these
don't match, the SDK throws an error and stops all connections") and checks the TLS key
against the enclave-generated key in the report; the CVM runs NVIDIA `nvattest` at boot
(https://docs.tinfoil.sh/verification/verification-in-tinfoil.md,
https://docs.tinfoil.sh/verification/attestation-architecture.md). An auditor rebuilds the
image with Nix and compares its hashes and dm-verity root hash with the Sigstore-signed
manifest; OVMF is built outside that recipe (https://docs.tinfoil.sh/verification/cvm-image.md).
Stated caveats: the `inference.tinfoil.sh` router reads request content inside its own
measured enclave, so it must be trusted too, and "V3 attestation and every feature that
depends on it are beta" (https://docs.tinfoil.sh/resources/feature-status.md). Catalogue in
October 2026 includes Kimi K3, GLM-5.3, DeepSeek V4.1 Flash and Llama 3.3 70B; chat $20 per
month, API pay-as-you-go priced per model in the dashboard
(https://docs.tinfoil.sh/models/chat.md, https://tinfoil.sh/pricing).

**NEAR AI Cloud.** TDX VMs with GPUs in CC mode, TLS terminated inside the enclave, and a
free attestation endpoint: `NONCE=$(openssl rand -hex 32)`, then
`GET https://<model>.completions.near.ai/v1/attestation/report?signing_algo=ecdsa&nonce=$NONCE`,
check the nonce, verify the TDX quote with `dcap-qvl` and the GPU evidence with NVIDIA NRAS,
and add `include_tls_fingerprint=true` to bind the TLS key
(https://near.ai/blog/nearai-cloud-on-saygm-verifiable-confidential-inference). The
MIT-licensed verifier (Python and TypeScript) does all of that, compares the Docker Compose
manifest with `mr_config` and checks Sigstore for the container images; its trust model
lists NVIDIA and Intel under "You Must Trust"
(https://raw.githubusercontent.com/nearai/nearai-cloud-verifier/main/README.md). Intel
Trust Authority verifies independently of NEAR and returns an Intel-signed JWT
(https://near.ai/blog/near-ai-cloud-integrates-intel-trust-authority). Caveats: the report
"does not prove that one particular request through a gateway reached that enclave", and
the OpenRouter route is not attested (https://near.ai/blog/near-ai-cloud-on-openrouter).
TEE-marked prices per 1 M tokens in/out: GLM 5.3 Flash $0.08/$0.50, Qwen 3.6 35B A3B
$0.17/$1.10, Kimi K3 $3.30/$16.50 (https://near.ai/models).

**Privatemode** (Edgeless Systems). The SDK or local proxy performs remote attestation
before sending; the source is public and built reproducibly, with guides to recompute the
reference values; BSI C5:2026 with a "very strong attestation" grade and ISO 27001; prompts
are not stored, operational metadata is kept up to 90 days. The "proxyless API" is not
client-verifiable ("Client-side remote attestation and reproducible verification of the
proxyless API deployment aren't currently supported")
(https://docs.privatemode.ai/security/trust-and-compliance/). Prices in EUR per 1 M tokens
in/out: gpt-oss-120b 0.20/0.65, GLM-5.3 1.55/5.74 (https://docs.privatemode.ai/pricing/).
It runs on Contrast, confidential containers for AMD SEV-SNP and Intel TDX under the
Business Source License (https://docs.edgeless.systems/contrast). For superborg: an
OpenAI-compatible remote backend that runs the provider's verifier first, shows the result
next to the model name, and refuses to send on failure.

### Self-hosting on a GPU TEE

The attestation chain has two layers: the CPU quote (TDX or SEV-SNP) and the NVIDIA GPU
report, verified by NRAS against NVIDIA's reference manifests into a JWT; Intel Trust
Authority calls this composite attestation the preferred method and warns that "a GPU on its
own is not a complete TEE"
(https://docs.trustauthority.intel.com/main/articles/articles/ita/concept-gpu-attestation.html,
https://docs.nvidia.com/attestation/quick-start-guide/latest/attestation-examples/hopper_single_gpu.html).
RA-TLS: the server generates its TLS key inside the CVM and binds it into the quote's report
data; the client verifies the quote, then the certificate, before sending anything.

* Phala GPU TEE (Intel TDX): H100 80 GB $3.08 per GPU-hour on demand ($2.38 reserved) and
  H200 141 GB $4.80 ($3.20), both with a 24-hour minimum; B300 288 GB $6.50 on a 1-month
  term with a 30-day minimum ($5.60 reserved) (https://phala.com/gpu-tee). dstack
  (Apache-2.0) deploys an unmodified Docker Compose file, such as vLLM with `runtime:
  nvidia`, into a TDX CVM; its KMS runs in its own TEE and "verifies TDX quotes before
  releasing keys"; it needs a bare-metal TDX host
  (https://raw.githubusercontent.com/Dstack-TEE/dstack/master/README.md).
* Google Cloud: `a3-highgpu-1g` (TDX, 1x H100 80 GB) and `g4-standard-48` (SEV, 1x RTX PRO
  6000 Blackwell 96 GB)
  (https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/supported-configurations);
  Confidential Space on H100 reached GA on April 29, 2026 and Intel Trust Authority
  attestation on September 15, 2026
  (https://docs.cloud.google.com/confidential-computing/confidential-space/docs/release-notes).
  The confidential uplift on `a3-highgpu-1g` is "N/A" on demand and +$0.4391592 per hour
  Spot; on `g4-standard-48` +$0.45 per hour plus an NVIDIA CC licence of $0.08 per GPU-hour
  (https://cloud.google.com/confidential-computing/confidential-vm/pricing). A third-party
  list puts non-confidential GCP H100 at $10.98 per GPU-hour and the cheapest commodity
  H100 at $2.99 (https://www.thundercompute.com/blog/nvidia-h100-pricing).
* Azure: `Standard_NCC40ads_H100_v5`, SEV-SNP on EPYC Genoa with 1x H100 NVL 94 GB, "TEE
  spans confidential VM on the CPU and attached GPU"
  (https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nccadsh100v5-series);
  the onboarding repo's step 3 is attestation and CC-mode verification
  (https://raw.githubusercontent.com/Azure/az-cgpu-onboarding/main/README.md). Microsoft
  lists no price; a third party lists $6.98-8.90 per hour by region
  (https://cloudprice.net/vm/Standard_NCC40ads_H100_v5).
* AWS has no GPU TEE: Nitro Enclaves are CPU-only
  (https://aws.amazon.com/ec2/nitro/nitro-enclaves/faqs/).

Overhead of CC mode: on one H100 under TDX, time to first token rose 21.8% (Mistral-7B) and
27.8% (Qwen3-30B-A3B) and throughput fell 17.7% and 21.1% at fixed request rates
(https://arxiv.org/abs/2607.19353); Chrapek et al. measured a 4-8% throughput penalty on
H100 CC (https://arxiv.org/abs/2509.18886); NVIDIA measured 1.0-7.5% on HGX B300
(https://developer.nvidia.com/blog/hardware-rooted-ai-security-that-wont-slow-you-down/).
Budget 10-30% on Hopper. Capacity: a vendor benchmark of vLLM with Llama 3.1 70B FP8 on
4x H100 reaches about 3,400 output tok/s at 256 concurrent requests
(https://www.generalcompute.com/blog/generalcompute-vs-vllm-throughput-latency-and-cost-benchmarks);
at $2.99 per GPU-hour that is about $12 per hour, roughly $1 per million output tokens at
full load and $288 per idle day (both derived). For one person the idle cost decides.

Steps: rent; deploy vLLM or SGLang through dstack's Compose flow or a Confidential Space
measured container; verify GPU CC mode and the CPU quote at boot (NRAS, Trust Authority or
Azure Attestation); terminate TLS inside the CVM with the key bound into the quote; build
the image reproducibly (Tinfoil's Nix recipe) and publish the measurements; make the
superborg client refuse to send when the measurement differs. OpenPCC is an open framework
of this shape (Apache-2.0 Go client, Oblivious HTTP through a relay, a compute node
"responsible for attesting to the TPM, GPU, and other hardware components"), but the compute
node's licence and supported TEE generations are not stated, so read the code first
(https://raw.githubusercontent.com/openpcc/openpcc/main/README.md,
https://raw.githubusercontent.com/confidentsecurity/confidentcompute/main/README.md).

### CPU TEE for 7-14B models

For mid-size models the borg engine itself can be the attested workload, with no GPU, CUDA
or vendor inference server in the trusted computing base. GCP's confidential surcharge on
top of the VM price: TDX on C3 $0.0033982 per vCPU-hour plus $0.0004555 per GiB-hour,
SEV-SNP on N2D $0.0027502 per vCPU-hour plus $0.0003686 per GiB-hour
(https://cloud.google.com/confidential-computing/confidential-vm/pricing); Chrapek et al.
measured under 10% throughput and under 20% latency overhead on CPU TEEs with AMX for
Llama-2 7B to 70B (https://arxiv.org/abs/2509.18886). borg's 4-thread AVX-512 VNNI figure
(23 tok/s for TinyLlama 1.1B q8, README) has not been measured on a 40-96 vCPU CVM; expect a
few tokens per second for 7B q4 single-stream until it is.

### What is not verifiable

OpenAI's Private Safety Processing stores encrypted content in the customer's own bucket
with a 30-day TTL and decrypts it for safety classification in "a hardware-attested
computing environment that disables human access"; the guide does not say the customer can
verify that attestation, and inference itself is not described as attested
(https://developers.openai.com/api/docs/guides/private-safety-processing). Anthropic's
Enterprise Frontier Safeguards pairs zero data retention with monitoring data kept in the
customer's cloud and mentions no TEE or attestation
(https://www.anthropic.com/news/enterprise-frontier-safeguards). Both are contracts. The
reference bar is Apple's Private Cloud Compute, where devices send only to nodes that attest
to software listed in a public log (https://security.apple.com/blog/private-cloud-compute/).

## 5. Residual trust assumptions

* Hardware roots of trust: Intel, AMD and NVIDIA sign the quotes; the NEAR verifier's README
  lists NVIDIA and Intel under "You Must Trust".
* Firmware: an assessment of GCP A3 (TDX + H100) found attestation and isolation working but
  TDVF firmware measured without source availability, in-VM key storage without revocation,
  and vendor-managed APIs that "introduce new trust boundaries"
  (https://census-labs.com/resources/challenging-the-boundaries-of-confidential-computing-for-ai);
  Tinfoil builds OVMF outside its reproducible recipe.
* Attestation services: NRAS, Intel Trust Authority and Azure Attestation are web services
  whose verdicts the client accepts; `dcap-qvl` removes the CPU one, not the GPU one.
* Gateways: Tinfoil's router sees plaintext inside its enclave; NEAR's report does not bind
  one request to one enclave; any route through OpenRouter is unattested.
* Reproducibility proves the artifact matches the source; only a transparency log ties the
  running measurement to that build, and someone has to read the source.
* Browser path: the model stays on the device, but the viewer's JavaScript comes from the
  host, so a modified page could exfiltrate; pin releases and keep the viewer under your own
  control. The CDN sees the IP address and which model was fetched.

## 6. Recommendation

1. Now, small: convert the models that fit (up to Qwen2.5-1.5B q4), split them into 128 MB
   parts, upload them to a public Hugging Face model repo, point `model.json` at absolute
   `/resolve/` URLs, deploy the viewer as a static Space, and add `persist()`, `estimate()`
   and the Safari seven-day note to `worker.js`.
2. Small, optional: mirror the viewer on Cloudflare Workers with the parts in R2 when an own
   domain is wanted; keep GitHub Pages for the viewer only.
3. Medium: an OpenAI-compatible remote backend in superborg for attested providers (Tinfoil
   SDK, NEAR verifier, Privatemode proxy), verification before the first request, result
   shown in the UI. This is the only route today to 70B and trillion-parameter MoE models
   with enforced privacy, at $0.08-3.30 per million input tokens (section 4).
4. Large: the WebGPU backend, for 7-8B models on the user's own GPU.
5. Large, optional: self-hosting. First a CPU TEE on GCP C3 running borg for 7-14B models
   as a fully self-authored attested stack; then, if 70B is required, a GPU TEE on Phala
   with dstack or GCP Confidential Space, budgeting $3-11 per GPU-hour plus 10-30%
   overhead, with reproducible images and published measurements.

What the user must supply: a Hugging Face account (step 1); a Cloudflare account and a
domain (step 2); accounts and payment at Tinfoil, NEAR AI or Privatemode (step 3); a Google
Cloud, Azure or Phala account with GPU quota and someone to run Linux (step 5); and a macOS
or iOS device to test Safari's storage eviction, COEP behaviour and Memory64, since this
build environment has only headless Chromium and Playwright WebKit.
