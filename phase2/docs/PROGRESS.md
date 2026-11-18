# DDP Phase 2 — DermaMNIST skin-lesion classifier: improvement + final report

**Resume-from-here file.** Read this first in a new session. Everything below is the
state of the work; update it after every milestone.

## Context (fixed facts)
- Project: Dual Degree Project, 7-class DermaMNIST (64x64 variant, MedMNIST v2),
  Suharsh Kumar (BE22B037) & Sayantan Das (BS22B035), guide Dr. Jayaganthan R, mentor G. K. Harish Balaji.
- July 2026 baseline (summer report `DDP_Report_DermaMNIST.pdf`): EfficientNet-B0 @96px, label-smoothed CE,
  two-phase transfer learning, TTA. Test: **80.00% top-1 (TTA), 78.40% single pass, 97.36% top-3,
  54.87% balanced acc.** Checkpoint: `run/artifacts/single_b0/best_model.keras`; eval log `run/eval.log`.
- Known problems in the July report (found 2026-11-03): auto-sklearn benchmark should be 0.719 not 0.749;
  "leakage-free split" claim is wrong (Abhishek et al. 2025: duplicate lesions across splits);
  fine-tuning of the final model was cut at epoch 10/30 with val acc still rising (`run/single.log`);
  an unreported focal-loss run (`run/full.log`, 128px, gamma 2, cw^0.5) reached 78.10% top-1 / **70.84% balanced**.
- Hardware: Apple M1, 8 GB. TF 2.19 + tensorflow-metal in `run/.venv`. On 2026-11-18 the machine was
  memory-starved (2.2 GB swap used, disk reads ~8 MB/s): TF import alone took minutes. Train with
  batch 16 @96px, each phase in its own process (`run/run_v2.sh` does this).

## Phase-2 plan (what "improvement" means here)
1. Resume fine-tuning of the 80.0% checkpoint (`MODE=resume`, unfreeze top 60, cosine 5e-5 -> 1e-6, 20 ep).
2. Second member: EfficientNetV2-B0 @96px (head 12 ep, fine-tune 20 ep).
3. Validation-tuned ensemble (weights on simplex grid) + TTA.
4. Prior-corrected decision rule (logit adjustment, Menon et al. 2021): score = log p - tau*log prior,
   tau tuned on validation. Operating points A (tau=0, accuracy), B (screening), C (balanced).
   -> gives a controllable top-1 / balanced-accuracy trade-off without retraining.
5. Final report PDF (reportlab) in `DDP_Phase2/report_src/`, output `DDP_Final_Report_DermaMNIST.pdf`.

Working name for the method (use in report): **PRIME — Prior-corrected, Resolution-matched,
heterogeneous EfficientNet ensemble with TTA**. Only claim it beats 80.0% if the test numbers show it.

## Files
- `run/train_v2.py`  training/eval (modes head|finetune|resume|all|eval), saves val+test probs.
- `run/ensemble_v2.py`  ensemble weights + tau sweep + figures + `artifacts/v2_ensemble/ensemble_v2_results.json`.
- `run/run_v2.sh`  whole pipeline; skips finished stages; logs in `run/logs_v2/`.
- `DDP_Phase2/research/`  literature/benchmark notes (from web research).
- `DDP_Phase2/results/`  copies of every results.json; `DDP_Phase2/figures/` report figures.

## Status log (newest last)
- 2026-11-18 11:20  Scripts written and syntax-checked. TF import probe running (slow machine). Research subagent
  running (MedMNIST v2/+ benchmarks, Abhishek et al. leakage paper, references). Nothing trained yet.
- 2026-11-18 11:40  Research saved: `research/literature_notes.md` (MedMNIST v2/+ tables, Abhishek et al. leakage numbers, refs).
  Colab GPU notebook written: `DDP_Phase2_PRIME_colab.ipynb` (members b0_96, v2b0_96, b0_128n, v2b1_128n; ensemble+tau).
  BLOCKER: Mac has ~70 MB free RAM (Claude app ~2.8 GB, Firefox ~2 GB, Chrome); TF import never completes
  (647 MB libtensorflow_cc cannot stay paged in). Even sklearn import takes minutes. Local training impossible
  until apps are closed. `run/prior_correction_baseline.py` (numpy/sklearn only) queued in background.
- 2026-11-18 11:42  Pipeline launched in a Terminal tab ("DDP phase-2 training"): `cd run && ./run_v2.sh | tee logs_v2/pipeline.log`.
  Stages: single_b0 eval -> v2_b0_resume -> v2_v2b0 head -> v2_v2b0 finetune -> ensemble_v2. Re-run the same command to resume.
  Expect very slow progress while the Mac is swapping (TF import alone > 20 min). Check `run/logs_v2/*.log` for `EPOCH n ...` lines.
  Next: write report builder in `DDP_Phase2/report_src/build_report.py` (reportlab, stub socket.getfqdn), then build once results exist.
- 2026-11-18 11:50  Report builder written: `report_src/build_report.py` (reads all result JSONs, "pending" placeholders; DejaVu fonts for Greek).
  Killed the stalled `prior_correction_baseline.py` (swapped out, 0 progress in 25 min) to leave memory for the pipeline;
  RE-RUN IT after the pipeline finishes: `cd run && .venv/bin/python -I prior_correction_baseline.py`
  (writes DDP_Phase2/results/prior_correction_baseline.json + figures/prior_correction_baseline.png; Section 7.1 of report needs it).
  Pipeline stage 0 (single_b0 eval) still importing TF at 11:52 (started 11:42). Draft report builds fine (15 pp, placeholders).
- 2026-11-18 11:55  DONE (numpy-only, 3 s): `run/prior_correction_np.py` -> `results/prior_correction_baseline.json`.
  Archived 80.00% model: macro-AUC 0.9494; tau=0.5 -> 76.61% top-1 / 64.12% balanced / melanoma recall 42.6->56.1%;
  tau=1 over-corrects (59.25% top-1 / 64.37% balanced). Report Section 7.1 + Table 6a + Figure 4 (reportlab-native chart) filled.
  Draft `DDP_Final_Report_DermaMNIST.pdf` rebuilt (phase-2 member/ensemble tables still "pending").
- 2026-11-18 11:58  Session paused here. Pipeline process (train_v2.py, stage single_b0 eval) alive in Terminal tab but starved:
  47 MB RSS after 12 min, ~70 MB free RAM on the Mac. It will proceed by itself once memory is freed (close Chrome/Firefox,
  other apps). If it died, re-run: `cd run && ./run_v2.sh 2>&1 | tee logs_v2/pipeline.log` (resume-safe).
  Alternative: run `DDP_Phase2_PRIME_colab.ipynb` on Colab GPU, download WORK_DIR/*/{val,test}_tta_probs.npy + results.json
  into run/artifacts/<member>/ and run `python ensemble_v2.py artifacts/single_b0 artifacts/<members...>`.
  THEN: `run/.venv/bin/python -I DDP_Phase2/report_src/build_report.py` -> DDP_Final_Report_DermaMNIST.pdf (fills Tables 6-8, Figs 5-8).
  Also still to do when machine is free: `python -I prior_correction_baseline.py` for figures/prior_correction_baseline.png (optional;
  report already has a reportlab-native version of that chart).
- 2026-11-18 12:05  User decisions: train on Colab GPU; THIS report is the final one for this phase (64-px data only; input
  preprocessing allowed); higher-resolution data goes into the NEXT-TERM (6-month) plan, which must be added at the end of the report.
  Local pipeline + monitor stopped. Colab: built-in browser is signed in; notebook "Untitled3.ipynb" (Drive id 1Yiz7vyWw7lfm3munhx-aJWpjnBbzergL),
  T4 runtime, one cell = compressed loader of `DDP_Phase2/prime_colab.py` (5 members: b0_96, v2b0_96, b0_96_sog, b1_112, v2b1_128;
  then ensemble + tau + figures; zips to /content/DDP_Phase2_results.zip). Cell started ~12:05.
  AFTER RUN: download the zip (ask user), unzip into run/artifacts/ (member dirs + v2_ensemble/), rebuild report.
- 2026-11-18 12:10  First Colab run crashed (Keras compile positional-arg bug) -> fixed in prime_colab.py (metrics= keyword),
  one-liner regenerated (7755 chars) and re-run at 12:10. Report builder now has: dynamic member table (Table 6), Table 5b members,
  Section 13 six-month plan (G1-G7, month schedule, risks, stretch), refs 15-19. Draft = 20 pages.
- 2026-11-18 12:50  COLAB RUN COMPLETE (about 30 min on T4). Results zip downloaded (user approved) -> results/DDP_Phase2_results.zip,
  unpacked into run/artifacts/{b0_96,v2b0_96,b0_96_sog,b1_112,v2b1_128,v2_ensemble}. Summary: results/colab_run_summary.txt.
  HEADLINE: PRIME-B 82.64% top-1 / 97.91% top-3 / 66.20% balanced / AUC 0.9623 (Phase 1: 80.00 / 97.36 / 54.87 / 0.949).
  PRIME-C 79.95% top-1 / 71.63% balanced. Best member b1_112 81.70%. Weights (0.1,0,0.3,0.3,0.3), tau_B 0.15, tau_C 0.45.
  Comparative stats (run/comparative_stats.py -> results/comparative_stats.json): PRIME-B CI [81.0, 84.3]; McNemar vs Phase 1
  116 vs 63 discordant, p = 9e-5; paired delta CI [+1.3, +4.0] pts; 13.2% relative error reduction; #1 on 28/64-px leaderboard.
  Report: Section 7.4 comparative statistics + Figure 9 leaderboard added. Remaining: more evidence screenshots, final visual QA,
  push commented code + results to GitHub repo dermamnist-lesion-classification (user asked for code on GitHub).
- 2026-11-18 12:55  FINAL REPORT BUILT: ../DDP_Final_Report_DermaMNIST.pdf (36 pp): abstract, dataset+leakage, Phase 1, critical review,
  PRIME method, results (7.1 prior correction, 7.2 members, 7.3 ensemble/operating points, 7.4 comparative statistics + leaderboard),
  benchmarks, discussion, clinical implications, conclusion, project files (no code; GitHub link), Section 13 six-month plan,
  references, Appendix A (10 Colab screenshots), Appendix B (Phase-1 log extracts).
  GitHub repo clone ../dermamnist-lesion-classification staged: phase1/, phase2/ (code, results, figures, docs), report/, README rewritten.
  Next: commit + push to github.com/suharshkumar/dermamnist-lesion-classification (user asked for the code to be on GitHub).
- 2026-11-18 13:00  DONE. Prose rewritten in plain student voice (user request: no AI-sounding text, no code in report, screenshots as evidence).
  Final: ../DDP_Final_Report_DermaMNIST.pdf (36 pp). Repo pushed: github.com/suharshkumar/dermamnist-lesion-classification (phase1/, phase2/, report/, README).
  Colab notebook "Untitled3.ipynb" remains in the user's Drive with the full run output. Open items for next term are in report Section 13.
