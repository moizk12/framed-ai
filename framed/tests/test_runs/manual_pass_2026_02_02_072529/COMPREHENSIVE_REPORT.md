# FRAMED Intelligence Pipeline — Comprehensive Test Report

**Run ID:** manual_pass_2026_02_02_072529
**Generated:** 2026-02-02 07:31:10 UTC

---

## Executive Summary

- **Total images:** 5
- **Completed:** 5
- **Failed:** 0
- **Passed:** Yes
- **Duration:** N/A
- **Throughput:** 0 images/hour
- **Started:** 2026-02-02T07:31:10.472181+00:00
- **Ended:** 2026-02-02T07:31:10.472200+00:00

### Test Configuration

- Dataset: `C:\Users\moizk\Downloads\framed-clean\stress_test_master\dataset_v2`
- Shuffle: True, Seed: 42
- Expression layer: enabled
- Cache: disabled

## Metrics

### Intelligence Health
- Average confidence: 0.59
- Confidence std: 0.098
- Uncertainty acknowledged: 100.0%
- Multiple hypotheses: 60.0%

### Failure Metrics
- Hallucination rate: 0.0%
- Overconfidence rate: 0.0%
- Mentor drift: 0.0%

### Learning
- New patterns stored: 5
- Evolution entries: 5

---

## Per-Image Results

### 1. artistic_v2_artistic_001

- **Category:** artistic
- **Status:** OK
- **Confidence:** 0.78
- **Primary interpretation:** I see a degraded, weathered stone/concrete surface with small patches of organic growth (moss/lichen) integrated into the material.

**Visual evidence:**
- Green coverage: 0.060 (salience: minimal)
- Material condition: degraded (conf: 0.80)
- Integration: integration (moderate)

**Mentor critique (gpt-5-mini):**

> I see weathered stone — a mineral face pared by time, its edges softened and crumbling, a surface that remembers contact. Small, scattered green lives cling to it — not a carpet of reclaiming growth but quiet, intimate patches, embedded into the grain rather than lying loose upon it.
> 
> You’ve resolved something here you once struggled with: the difference between ruin and restraint. The texture speaks of high erosion, the vegetation of modest occupation; together they read as a patient dialogue between material and moisture, adhesion and abandonment. I used to read such images as simply “textured surface”; looking at this work now, I see a mineral substrate whose deterioration and small, integrated life-forms you’ve rendered with deliberate economy.
> 
> You keep circling the question of naming—stone, concrete, plaster—why risk an overconfident label when “mineral surface” holds the ambiguity that the image demands? Is the organism meant to be moss, lichen, algae, or simply the trace of time made green? And do you withhold context by choice: no coin for scale, no wide frame to tell us whether this is wall, pavement, or monument?
> 
> There is a contradiction here that intrigues me: edges that announce decay, while growth remains cautious and sparse. Is that contradiction intentional, or a residue of the way you framed and lit the scene? Will you test those edges with raking light, widen the frame to locate the object, or take a macro to insist on the life’s identity — or will you preserve the mystery?
> 
> Which secret will you choose to reveal next?

**Reflection:** quality=1.06, requires_regeneration=False

---

### 2. interiors_v2_interiors_001

- **Category:** interiors
- **Status:** OK
- **Confidence:** 0.52
- **Primary interpretation:** A severely weathered man-made surface with small but highly interwoven organic intrusion along edges/cracks (primary), though it could also be greenish staining/patina misread as growth or decayed wood with root/vine intrusion.

**Visual evidence:**
- Green coverage: 0.008 (salience: minimal)
- Material condition: degraded (conf: 0.80)
- Integration: reclamation (high)

**Mentor critique (gpt-5-mini):**

> [Critique generation returned empty response]

**Reflection:** quality=0.79, requires_regeneration=False

---

### 3. mixed_v2_mixed_004

- **Category:** mixed
- **Status:** OK
- **Confidence:** 0.60
- **Primary interpretation:** I see a degraded, weathered material surface with minor organic growth (moss/lichen) integrated into the surface.

**Visual evidence:**
- Green coverage: 0.003 (salience: minimal)
- Material condition: degraded (conf: 0.80)
- Integration: integration (moderate)

**Mentor critique (gpt-5-mini):**

> I see weathered stone — its edges worn back like a coastline after a long tide, a face softened rather than freshly cut. One plausible reading is that time has abraded the margins (the wear is emphatic), while tiny, reluctant patches of green have taken root and become partly one with the surface. This interpretation remains tentative: the green could be true moss or lichen, or it could be a stain, and without scale or alternate views the life here is more suggestion than confession.
> 
> You’ve resolved something here you used to struggle with: where once I admitted “N/A,” now I can speak of edge erosion that reads loud, of surface texture that whispers, of organic presence that is small in area but not merely superficial. The evidence leans toward a portrait of slow decay — strong edge degradation, very subtle roughness, and minimal but somewhat embedded growth — yet that very economy of signal breeds doubt.
> 
> A few quiet questions press at the work. What is the substrate beneath your attention — stone, concrete, wood? Is the green a biological guest or a pigment traced by weather? Do you intend to claim age, neglect, or a fragile coexistence of life and ruin? These are not technical quibbles; they determine the moral posture of the image.
> 
> A challenge stands: do not let a single metric substitute for looking. Low coverage of organic matter can be an argument for restraint or an accident of framing; attributing cause (climate, time, neglect) would be a step beyond what the evidence permits. I used to see this matter as unresolvable; looking at how you’ve tightened your observations, I now read a deliberate restraint — the scene chooses ambiguity.
> 
> If the photograph is a lesson in patience, what, then, do you want the viewer to believe about time: that it erodes everything equally, or that it leaves small, stubborn proofs of life in the cracks?

**Reflection:** quality=1.06, requires_regeneration=False

---

### 4. street_v2_street_001

- **Category:** street
- **Status:** OK
- **Confidence:** 0.52
- **Primary interpretation:** A heavily weathered surface with strong edge breakdown; either it is being physically reclaimed by organic matter (despite low detectable green coverage), or the 'reclamation' signal is a misread of paint/stains or purely structural decay.

**Visual evidence:**
- Green coverage: 0.000 (salience: minimal)
- Material condition: degraded (conf: 0.80)
- Integration: reclamation (high)

**Mentor critique (gpt-5-mini):**

> [Critique generation returned empty response]

**Reflection:** quality=0.79, requires_regeneration=False

---

### 5. street_v2_street_004

- **Category:** street
- **Status:** OK
- **Confidence:** 0.55
- **Primary interpretation:** A degraded man-made surface with strong edge breakdown; either subtle biological reclamation (lichen/moss/roots) or non-organic peeling/coating failure producing similar patterns.

**Visual evidence:**
- Green coverage: 0.000 (salience: minimal)
- Material condition: degraded (conf: 0.80)
- Integration: reclamation (high)

**Mentor critique (gpt-5-mini):**

> [Critique generation returned empty response]

**Reflection:** quality=0.79, requires_regeneration=False

---
