Escalation 1:


[PLAN] Escalation — Pass 4 blocked on OpenAI-as-compression; propose Pass 3.5 (OpenAI-only validation)

Why escalating

Pass 4 assumes “KV budget is request-time and produces measurable degradation” across OpenAI + vLLM. We’ve confirmed a hard constraint: real OpenAI API cannot apply SnapKV/KV compression (no backend control), and our safety gate correctly prevents sending unknown KV params to api.openai.com. Therefore, OpenAI cannot be used as a “compression engine” and multi-budget runs on OpenAI are not meaningful (and trigger TPM 429s / cost).

Reframe (must be canonical)

Carry forward (previous points):
	•	For real OpenAI API runs, we will not send extra_body KV params; Step 0.1 should be satisfied by:
	•	(a) distinct runs.kv_budget labels, and
	•	(b) explicit documented limitation that OpenAI won’t accept/echo/apply KV params.
	•	Step 0.1 strict “payload contains kv_budget” is enforced on GPU vLLM openai_compat only.
	•	OpenAI runs = reference/measurement validation only: binning, scoring, failure attribution, report plumbing, and stability checks.
	•	vLLM runs = compression validation: SnapKV actually applied; request-time KV knobs honored; “budget causes degradation” is meaningful here.

New constraints / goals:
	•	GPU environment is unstable → isolate tasks that require GPU from those that do not.
	•	Focus near-term on ensuring OpenAI functionality + diagnostics work end-to-end. Once harness is validated, swap/attach controllable serving backends later.

Request: generate “Pass 3.5” blueprint (execution-oriented)

Goal: Validate harness correctness & diagnostics on OpenAI-only without pretending compression is applied.

Pass 3.5 must:
	1.	Split lanes
	•	Lane A (OpenAI-only, no GPU): harness validation steps, single reference run per task.
	•	Lane B (GPU-only): all compression attribution + multi-budget comparisons (SnapKV).
	2.	OpenAI-only steps
	•	Use pinned configs but run one budget only (or formalize budgets as labels but do not run both).
	•	Checks:
	•	prepare produces correct binning and manifests,
	•	run produces responses with low failure rate (exclude prompt-too-long via max_model_len settings),
	•	score/analyze/report produce required artifacts,
	•	failure attribution computation is correct and stable.
	•	Include rate-limit friendly settings (pacing/TPM mitigation) as part of harness validation.
	3.	Deliverable
	•	A “Pass 3.5 Verdict” note: confirms diagnostic pipeline works, and clearly states compression is not tested on OpenAI.
	4.	Transition back to Pass 4
	•	Define the minimal prerequisites before returning to compression testing: stable GPU server, base_url non-OpenAI, payload echo shows kv_budget present, etc.

Question for PLAN: multi-provider abstraction

Would adopting a “single interface” multi-provider tool (e.g., LiteLLM / OpenRouter / similar) help?
	•	Evaluate pros/cons specifically for this project:
	•	pros: unified API, easier swapping providers/models, potentially fewer code paths
	•	cons: providers still won’t expose KV internals; compression testing still requires controllable server (vLLM, etc.)
	•	Decide whether this belongs in architecture now or is explicitly out-of-scope until after Pass 3.5.