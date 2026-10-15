<div align="center">

# 🧠 NeuroMap

**Map your mind in 15 minutes.**
An open-source cognitive test: 8 tasks from classic cognitive science, adaptive difficulty, age norms and a living 3D brain that lights up with your results.

**[▶ Live demo](https://maxmila0219-alt.github.io/neuromap/)** · One HTML file · No frameworks · No tracking · 10 languages

![NeuroMap](preview.png)

</div>

---

## ✨ What makes it different

- **A living 3D brain.** About 3,800 particles form an anatomically shaped brain: frontal, parietal, temporal and occipital lobes, plus the Sylvian and central sulci, the cerebellum and the brainstem. Drag it to spin it (it has inertia). After the test, each region glows according to your score.
- **Hand-written 3D renderer** for the mental-rotation task. It uses perspective, two-light shading, cube outlines, contact shadows and a holographic pedestal. A gentle ±6° sway gives motion parallax, so the depth is unambiguous. The renderer has no WebGL and no libraries.
- **Every puzzle is generated procedurally.** Nothing is stored, nothing can be googled, and every run is unique.
- **Real adaptive testing.** Logic and number tasks use IRT, with an EAP ability estimate and a guessing parameter.
- **Age-normed scores.** Results are compared with your own age group.
- **Your cognitive type.** You get an archetype based on your strongest domain: The Strategist, The Architect, The Pilot and so on.
- **Friend duels.** A challenge link gives your friend *the same puzzles* and then overlays both brain maps: "You win by 7 points!".
- **Share cards.** Download a 1080×1350 PNG with your index, your type and your glowing brain.
- **Resume anytime.** Progress is saved after every task.
- **100% private.** Everything runs in your browser.

## 🔬 The eight tasks

| Domain | Task | Based on | Metric |
|---|---|---|---|
| Logic | 3×3 matrices built from Carpenter's rule taxonomy: progression, distribution, XOR, addition | Raven's Progressive Matrices | IRT θ |
| Numbers | Number series, 8 difficulty levels | Number-series tests | IRT θ |
| Spatial | Same object or mirror image? Angles 40°–160° | Shepard & Metzler (1971) | Accuracy, RT, **rotation speed °/s** |
| Memory | Repeat sequences of lit blocks (1 block/s) | Corsi; Kessels et al. (2000) | Span |
| Attention | 1 → A → 2 → B … | Trail Making Test B | Time + errors |
| Speed | Symbol → digit, 60 s | WAIS Digit-Symbol Coding | Correct/min |
| Self-control | Name the ink color, not the word | Stroop (1935) | Inverse efficiency score |
| Reflexes | Click when the field turns green | Simple reaction time | Median RT |

## 📐 Methodology

1. **Practice trials.** Stroop and mental rotation start with unscored practice and feedback, as in standard psychometric administration.
2. **Raw → z-score** using approximate adult norms from the literature.
3. **Adaptive items.** Logic and Numbers use EAP estimation with an N(0,1) prior and a 3PL-style model (c = 1/6 and 1/5).
4. **Mental rotation.** RT is regressed on angle for correct "same" trials. The slope gives your mental rotation speed, which is the classic Shepard–Metzler finding.
5. **Age adjustment.** Cross-sectional decline per decade after age ~22, in SD units, after Salthouse (2009, 2010): speed .28, attention .24, reflexes/spatial/logic .20, memory .18, self-control .16, numbers .15.
6. **Input-latency correction.** Touchscreens add about 35 ms, so the reaction-time norm is shifted for touch input.
7. **Validity check.** Leaving the tab during a task is flagged on the results.
8. **Composite.** The index is computed as

   `z = Σ wᵢ·zᵢ / √(Σwᵢ² + ρ·Σᵢ≠ⱼ wᵢwⱼ)`

   The weights are approximate g-loadings and ρ = 0.30. The result is scaled to 100 ± 15, with a 95% range from SEM = 15·√(1 − .88).

> ⚠️ **NeuroMap is not a clinical IQ test.** The norms come from published literature, not from a standardization sample. Brain regions are illustrative: they show the areas most associated with each ability in neuroimaging research, not a scan of your brain.

## 🚀 Deploy

```bash
open index.html            # works offline, just open it
# GitHub Pages: Settings → Pages → Branch: main → / (root)
```

Before publishing, set the `REPO` constant at the top of the `<script>` to your repository URL. It powers the ⭐ button and the share card.

**Links**
- `?seed=AB12C`: the same puzzles for everyone who opens it
- `?seed=AB12C&vs=…`: a duel against a friend's score
- `#r=…`: a read-only shared result
- `?lang=ru`: force a language

## 🌍 Add a language

Copy the `en` blocks in `I18N` and `I18N_EXTRA`, translate them, add the language to `LANGS`, and open a PR. Missing keys fall back to English.

Supported: English, Русский, Español, Português, Français, Deutsch, 中文, 日本語, हिन्दी, العربية (RTL).

## 🛠 Tech

Vanilla JS, Canvas 2D and SVG. It includes a seeded PRNG (mulberry32), WebAudio sound with no audio files, and haptics on mobile. It respects `prefers-reduced-motion`.

## 📄 License

MIT
