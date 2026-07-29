"""Single source of truth for the in-app theory manual.

THEORY_BLOCKS is a list of (kind, content) tuples rendered two ways:
  * on screen, by app.py, looping and calling st.markdown / st.latex / st.caption;
  * to PDF, by report.build_theory_pdf.

kinds: "md" (markdown prose / tables), "eq" (a LaTeX display equation), "cap"
(a small caption note). Keeping it here means the panel and the exported PDF can
never drift apart.
"""
from typing import List, Tuple

THEORY_TITLE = "Bolt Preload & Joint Analysis — Theory Manual"

THEORY_BLOCKS: List[Tuple[str, str]] = [
    ("md", r"""
        This panel is a concise **theory manual** for every calculation in the tool. Each section gives
        what is computed, the governing equation, the meaning of the symbols, and the modelling
        assumptions with their limits. Calculations run **internally in SI units** (mm, N, MPa, °C,
        N·mm); the interface converts to and from your selected display units only at input and output.

        **Primary references**
        - Budynas & Nisbett, *Shigley's Mechanical Engineering Design*, Ch. 8 — stress area, bolt and
          member stiffness, preload, fatigue and separation.
        - **VDI 2230**, *Systematic calculation of highly stressed bolted joints* — frustum member
          model and preload-scatter guidance.
        - Bickford, *An Introduction to the Design and Behavior of Bolted Joints* — nut factor and
          tightening-scatter ranges.
        - AISC *Specification* — elastic (vector) bolt-group shear and the slip-critical model.
        """),
    ("md", r"""
        ### 1. Nomenclature

        | Symbol | Meaning | Symbol | Meaning |
        |---|---|---|---|
        | $d,\ p$ | nominal diameter, pitch | $A_t,\ A_d$ | tensile-stress area, shank area |
        | $S_p,\ S_y$ | proof, yield strength | $S_{ut},\ S_e$ | ultimate, endurance strength |
        | $F_i,\ F_p$ | preload, proof load $S_pA_t$ | $P$ | external tensile load per bolt |
        | $k_b,\ k_m$ | bolt, member stiffness | $C$ | joint stiffness constant |
        | $E_b,\ E_i$ | bolt, member modulus | $\alpha$ | coefficient of thermal expansion |
        | $L,\ t_i$ | grip length, member thickness | $d_w,\ d_h$ | bearing, clearance-hole diameter |
        | $K$ | nut factor | $\Delta T$ | operating $-$ assembly temperature |
        | $n_f,\ n_p$ | fatigue, proof FoS | $n_{sep},\ n_{slip}$ | separation, slip FoS |
        """),
    ("md", r"""
        ### 2. Tensile Stress Area

        A threaded rod in tension carries load on neither the full nominal area nor the minor-diameter
        area, but on an effective area derived from the mean of the pitch and minor diameters:
        """),
    ("eq", r"A_t = \frac{\pi}{4}\left(d - 0.9382\,p\right)^2"),
    ("md", r"""
        The constant $0.9382$ is the average of the pitch- and minor-diameter offsets,
        $(0.6495 + 1.2269)/2$, per ISO 898-1 / Shigley. Every stress below is referred to $A_t$.

        **Thread series.** This formula describes the standard **60° profile** (ISO metric and Unified
        inch alike); only the pitch $p$ changes between series, so the tool offers a **thread series /
        pitch** selector — coarse and fine for metric, and **UNC / UNF / UNEF** for inch. Because $A_t$
        grows as $p$ shrinks, a *finer* pitch gives a **larger stress area and a higher proof load**
        $F_p = S_p A_t$ at the same nominal diameter, plus finer preload-per-turn control (§17); the
        trade-off is easier thread stripping in soft materials and greater cross-threading sensitivity.
        The choice propagates through every downstream result (proof load, stiffness, fatigue, torque,
        tightening stress and tap drill), so coarse and fine of the same diameter give different numbers.
        """),
    ("md", r"""
        ### 3. Preload and Proof Load

        The **proof load** is the largest tension the bolt sustains with no measurable permanent set:
        """),
    ("eq", r"F_p = S_p\,A_t"),
    ("md", r"""
        The recommended **preload** (assembly clamp force) is a fraction of proof — the long-standing
        rule that keeps the bolt below yield while maximising clamp:
        """),
    ("eq", r"F_i = 0.75\,F_p\ \text{(reused)}, \qquad F_i = 0.90\,F_p\ \text{(permanent)}"),
    ("md", r"""
        Alternatively, the **Target Yield Preload Tool** allows calculating preload as a direct percentage 
        of the bolt's yield strength rather than proof load:
        """),
    ("eq", r"F_i = \text{pct} \times S_y \times A_t"),
    ("md", r"""
        High preload is deliberately desirable: it flattens the fatigue load line (§11), resists
        self-loosening, and raises the separation margin (§12). If the bearing check (§6) limits the
        clamp, the tool reduces $F_i$ and flags it.
        
        **Custom Materials.** The tool includes built-in databases for common standard bolt grades and joint materials. 
        If your material is not listed, you can define **Custom Materials** via the dedicated tab. 
        Note that all custom material properties (Modulus $E$, Yield $S_y$, Ultimate $S_{ut}$, etc.) must be 
        input in **SI units** (MPa, GPa, °C), regardless of the current display unit system.
        """),
    ("md", r"""
        ### 4. Tightening Torque and the Nut Factor

        Where preload is set by torque, the **nut-factor** relation links applied torque to clamp force:
        """),
    ("eq", r"T = K\,F_i\,d"),
    ("md", r"""
        $K$ is a dimensionless torque coefficient lumping thread friction, under-head/nut friction and
        helix geometry — it is *not* a bare friction coefficient. Of the input torque, roughly half is
        lost under the head, about 40% in the threads, and only about 10% becomes useful tension.
        Typical $K$: about $0.20$ dry/as-received, $0.15$–$0.18$ plated, $0.12$ lubricated.

        **Tightening stress.** While being torqued the bolt carries axial tension *and* torsion from the
        thread torque $T_G = F_i(0.16\,p + 0.58\,d_2\,\mu_G)$, so the combined stress is checked against
        yield (assembly guideline $\le 90\%$):
        """),
    ("eq", r"\sigma_{red} = \sqrt{\sigma^2 + 3\tau^2}, \quad \sigma = \frac{F_i}{A_t}, \quad "
           r"\tau = \frac{T_G}{\pi d_s^3/16}"),
    ("md", r"""
        The thread friction $\mu_G$ is inferred from $K$. The torsion largely relaxes once the wrench is
        released, leaving mostly the axial stress.
        """),
    ("md", r"""
        ### 5. Preload Scatter and Tightening Method

        No method delivers an exact preload; the achieved clamp scatters about the target by $\pm s$.
        The tool shows the band and warns if the **upper** bound risks exceeding proof load:
        """),
    ("eq", r"F_{i,\max} = F_i\,(1 + s), \qquad F_{i,\min} = F_i\,(1 - s)"),
    ("md", r"""
        Representative $s$ (VDI 2230 / Bickford): torque wrench $\pm25\%$, turn-of-nut $\pm15\%$,
        hydraulic tensioner $\pm10\%$, hand/uncontrolled $\pm35\%$. Tighter control lets you safely aim
        for a higher mean preload because the upper tail stays below yield.
        """),
    ("md", r"""
        ### 6. Bearing Pressure and Crushing

        The clamp is reacted over the annular contact under the head/washer and nut. If the pressure
        exceeds the compressive yield $S_{yc}$ of the clamped material, the surface crushes and preload
        is lost by embedment:
        """),
    ("eq", r"\sigma_{br} = \frac{F_i}{A_b} \le S_{yc}, \qquad "
           r"A_b = \frac{\pi}{4}\left(d_w^2 - d_h^2\right)"),
    ("md", r"""
        $d_w$ is the effective bearing diameter (a hardened washer raises it to about $2d$) and
        $d_h \approx 1.1\,d$ the clearance hole. Only the **outer** faces bear, so the tool checks the
        first and last layers, not the interior. A crushing layer caps the preload at $S_{yc}A_b$ and is
        flagged; the usual fix is a hardened washer or a flanged head.
        """),
    ("md", r"""
        ### 7. Bolt Stiffness (two-section model)

        In the grip the bolt is a spring. The full-area shank ($A_d$) and the reduced-area threaded
        length ($A_t$) are different cross-sections, so they act as two springs in series:
        """),
    ("eq", r"\frac{1}{k_b} = \frac{l_d}{A_d E_b} + \frac{l_t}{A_t E_b} \;\Rightarrow\; "
           r"k_b = \frac{A_d A_t E_b}{A_d\,l_t + A_t\,l_d}"),
    ("md", r"""
        $l_d$ is the unthreaded shank length in the grip and $l_t$ the threaded length in the grip.
        Following Shigley, for $L \le 125$ mm the engaged thread is estimated as $l_t = 2d + 6$ mm, with
        $l_d = L - l_t$; if the whole grip is threaded this reduces to $k_b = A_t E_b / L$. A stiffer
        bolt relative to the members raises $C$ and so attracts more of the external load (§9).
        """),
    ("md", r"""
        ### 8. Member Stiffness (Rötscher / VDI 2230 frustum)

        The compressed material carries the clamp through a roughly conical zone. Each member $i$ is
        modelled as a hollow cone of 30° half-angle (Shigley Eq. 8-20):
        """),
    ("eq", r"k_i = \frac{0.5774\,\pi E_i\,d}"
           r"{\ln\dfrac{(1.155\,t_i + D - d)(D + d)}{(1.155\,t_i + D + d)(D - d)}}"),
    ("md", r"""
        Here $0.5774 = \tan 30^\circ$, $1.155 = 2\tan 30^\circ$, $d$ is the clearance-hole diameter and
        $D$ the cone-base (bearing) diameter. Members combine in series and pair with the bolt to give
        the **joint stiffness constant** $C$:
        """),
    ("eq", r"\frac{1}{k_m} = \sum_{i=1}^{N}\frac{1}{k_i}, \qquad C = \frac{k_b}{k_b + k_m}"),
    ("cap", "Simplification: each layer's cone base is taken at the head/washer diameter $d_w$ "
            "(the 30° frustum is not carried continuously through the stack), which is "
            "conservative for thick joints."),
    ("md", r"""
        ### 9. Joint Constant and External-Load Sharing

        When an external tensile load $P$ is applied to the preloaded joint, bolt and members share it
        by stiffness. The load is also influenced by the **VDI 2230 load-introduction factor** ($n$), 
        which dictates how much of the applied load acts directly under the bolt head versus deeper 
        in the joint. The bolt sees the fraction $n \cdot C$; the clamp force is relieved by $(1 - n \cdot C)P$:
        """),
    ("eq", r"F_b = F_i + n\,C\,P, \qquad F_m = F_i - (1 - n\,C)\,P"),
    ("md", r"""
        This is the key result of preloaded-joint theory. Because $C$ is usually small (often
        $0.1$–$0.3$) and $n \le 1$, most of an applied load *relieves the members* rather than 
        *loading the bolt* — which is exactly why a well-preloaded bolt is insensitive to external 
        load, provided the clamp never drops to zero (separation, §12).
        """),
    ("md", r"""
        ### 10. Thermal Effects

        A temperature change $\Delta T = T_{oper} - T_{assy}$ alters preload whenever the bolt and the
        members grow by different amounts — i.e. when their coefficients of thermal expansion $\alpha$
        differ. The differential free expansion is:
        """),
    ("eq", r"\Delta\delta = \sum_i t_i\,\alpha_i\,\Delta T \;-\; L\,\alpha_b\,\Delta T"),
    ("md", r"""
        The series bolt–member spring resists this mismatch, changing the preload by $\Delta F_{th}$,
        which is added to give the operating preload $F_{i,oper} = F_i + \Delta F_{th}$:
        """),
    ("eq", r"\Delta F_{th} = \Delta\delta\,\frac{k_b\,k_m}{k_b + k_m}"),
    ("md", r"""
        Members expanding more than the bolt (e.g. an aluminium joint, heated) **raise** preload; a bolt
        expanding more **loses** it. Fatigue, separation and proof all use $F_{i,oper}$, so thermal
        effects propagate through every downstream check automatically.

        **Embedment / relaxation.** Surface roughness flattens at the loaded interfaces, a length loss
        $f_z$ (typically a few µm per interface) that relaxes the joint through the same series spring
        and is deducted from the operating preload:
        """),
    ("eq", r"\Delta F_Z = f_z\,\frac{k_b\,k_m}{k_b + k_m}"),
    ("md", r"""
        ### 11. Fatigue (mean-stress criteria, preloaded load line)

        Under a load cycling between $P_{\min}$ and $P_{\max}$, only the bolt's share $C\,P$ varies; the
        preload is a constant mean. The stress oscillates about the preload stress $\sigma_i = F_i/A_t$:
        """),
    ("eq", r"\sigma_a = \frac{C\,(P_{\max}-P_{\min})}{2A_t}, \qquad "
           r"\sigma_m = \sigma_i + \frac{C\,(P_{\max}+P_{\min})}{2A_t}"),
    ("md", r"""
        The operating point starts at the preload point $(\sigma_i, 0)$ and climbs the **load line** as
        the external load grows. The factor of safety is the multiplier $n_f = S_a/\sigma_a$ by which
        the cyclic stress can grow before that load line meets the chosen failure locus — measured from
        the preload point, *not* the origin, so the steady preload is not double-counted. The locus is
        selectable in the Thermal & Fatigue tab:
        """),
    ("eq", r"\text{Goodman:}\quad \frac{\sigma_a}{S_e} + \frac{\sigma_m}{S_{ut}} = 1"),
    ("eq", r"\text{Gerber:}\quad \frac{\sigma_a}{S_e} + \left(\frac{\sigma_m}{S_{ut}}\right)^2 = 1"),
    ("eq", r"\text{ASME-elliptic:}\quad \left(\frac{\sigma_a}{S_e}\right)^2 + "
           r"\left(\frac{\sigma_m}{S_p}\right)^2 = 1"),
    ("eq", r"\text{Soderberg:}\quad \frac{\sigma_a}{S_e} + \frac{\sigma_m}{S_y} = 1"),
    ("md", r"""
        **Goodman** is linear and conservative; **Gerber** (parabola) best fits ductile-steel data;
        **ASME-elliptic** uses the proof strength $S_p$ on the mean axis (the usual bolt convention);
        **Soderberg** uses yield $S_y$ and is the most conservative (it also precludes yielding). For
        Goodman the intersection has the closed form (Shigley Eq. 8-38); the curved loci are solved for
        $S_a$ along the same load line:
        """),
    ("eq", r"n_f = \frac{S_e\,(S_{ut} - \sigma_i)}{S_{ut}\,\sigma_a + S_e\,(\sigma_m - \sigma_i)}"
           r"\quad\text{(Goodman)}"),
    ("md", r"""
        $S_e$ is the corrected axial endurance strength for **rolled** threads and already includes the
        thread-root stress concentration, so no extra $K_f$ is applied. With no alternating stress
        ($\sigma_a = 0$) there is no fatigue limit and $n_f \to \infty$.

        Three further options are available. **SWT** (Smith–Watson–Topper) and **Morrow** are extra
        mean-stress corrections; **VDI 2230** is a different, bolt-specific approach.
        """),
    ("eq", r"\text{SWT:}\quad \sigma_{ar} = \sqrt{\sigma_{max}\,\sigma_a} \le S_e"),
    ("eq", r"\text{Morrow:}\quad \frac{\sigma_a}{S_e} + \frac{\sigma_m}{\sigma_f} = 1,"
           r"\quad \sigma_f \approx S_{ut} + 345\ \text{MPa}"),
    ("md", r"""
        **SWT** corrects by the peak stress $\sigma_{max} = \sigma_m + \sigma_a$ and needs no extra
        constant, which suits the tensile mean stress of a preloaded bolt. **Morrow** replaces $S_{ut}$
        with the true fracture strength $\sigma_f$ (estimated as $S_{ut} + 345$ MPa for steels — only a
        rough figure for non-steels), making it less conservative than Goodman.

        **VDI 2230** does not use a mean-stress diagram: it compares the amplitude $\sigma_a$ to a
        permissible **endurance amplitude** that depends on bolt diameter and how the thread was formed:
        """),
    ("eq", r"\sigma_{ASV} = 0.85\left(\frac{150}{d} + 45\right)\quad\text{[MPa, rolled before HT]}"),
    ("eq", r"\sigma_{ASG} = \left(2 - \frac{F_{Sm}}{F_{0.2}}\right)\sigma_{ASV}"
           r"\quad\text{[rolled after HT]}"),
    ("md", r"""
        Here $d$ is the nominal diameter (mm), $F_{Sm}$ the mean bolt force and $F_{0.2} = S_y A_t$ the
        yield force. Rolling threads **after** heat treatment locks in beneficial compressive residual
        stress at the root, raising the limit by a factor of $1$–$2$. The VDI endurance limit is
        essentially **mean-stress independent** (the Haigh diagram of high-strength rolled bolts is
        nearly flat), and the factor of safety is $n_f = \sigma_A/\sigma_a$.
        """),
    ("md", r"""
        #### What this fatigue check is — and is not

        Every criterion above is an **infinite-life (endurance-limit) check**, *not* a life prediction.
        Each one reduces the cycle to an alternating stress, applies a mean-stress correction, and
        compares it to the single endurance strength $S_e$. The output is a **factor of safety on load**,
        $n_f$ — how far the alternating load can grow before the operating point reaches the failure
        locus. It is **not** a number of cycles to failure. This is deliberately different from the two
        finite-life methodologies an engineer might expect:

        | Method | Needs | Produces | Regime |
        |---|---|---|---|
        | **This tool** (constant-life) | $S_e$ (+$S_{ut},S_p,S_y$) | margin $n_f$ | high-cycle, ∞ life |
        | **S-N** (stress-life: Wöhler / Basquin) | full S-N curve ($\sigma_f',b$) | cycles $N$ | high-cycle, finite |
        | **$\varepsilon$-N** (Coffin–Manson) | $\sigma$–$\varepsilon$ + constants | reversals $2N$ | low-cycle |

        What this means in practice:
        - **A pass ($n_f \ge 1$) means infinite life *by that criterion*** — not "fails at $N$ cycles."
          A fail means the cycle sits above the fatigue limit (finite life), but the tool does **not**
          say *how* finite — that requires an S-N curve.
        - **$n_f$ is a margin on the alternating load**, read along the preloaded load line — it is not a
          safety factor on life and not a damage fraction.
        - **A single representative cycle** (the worst $P_{\min}$–$P_{\max}$ range) is checked. There is
          **no Miner's-rule cumulative-damage** summation over a variable-amplitude spectrum.
        - Valid for **constant-amplitude, high-cycle** loading of (predominantly ferrous) bolts.
          Variable-amplitude spectra, low-cycle/large-plasticity loading, or an explicit life estimate
          are out of scope and need an S-N or $\varepsilon$-N model.
        - $S_e$ here is the already-corrected, rolled-thread endurance limit (Shigley Table 8-17) — a
          *material-plus-detail* fatigue limit, not a polished-coupon $S_e'$ awaiting Marin factors.
        """),
    ("md", r"""
        ### 12. Joint Separation

        The joint opens when the external load drives the member force to zero. Setting $F_m = 0$ in §9:
        """),
    ("eq", r"P_{sep} = \frac{F_i}{1 - n\,C}, \qquad n_{sep} = \frac{P_{sep}}{P_{\max}}"),
    ("md", r"""
        Separation is both a serviceability and a fatigue failure: once open, the **full** external load
        (not just $C\,P$) is dumped onto the bolt and the cyclic range jumps. $F_i$ here is the operating
        preload after thermal effects.
        """),
    ("md", r"""
        ### 13. Static Proof (Yield) Margin

        The static check compares the maximum bolt force, $F_{b,\max} = F_{i,oper} + n\,C\,P_{\max}$, with
        the proof load:
        """),
    ("eq", r"n_p = \frac{S_p\,A_t}{F_{b,\max}} = \frac{F_p}{F_{b,\max}}"),
    ("md", r"""
        $n_p \ge 1$ keeps the bolt below proof (essentially yield) at the worst service condition.
        """),
    ("md", r"""
        ### 14. Thread Stripping and Engagement Length

        A bolt threaded into a weaker tapped material can shear ("strip") the internal threads before it
        fails in tension. Shear is taken on a cylinder at the major diameter, limited by the von Mises
        shear yield $0.577\,S_y$:
        """),
    ("eq", r"A_s = \pi\,d\,(0.75\,L_e), \qquad \tau = \frac{F_{b,\max}}{A_s} \le 0.577\,S_y"),
    ("md", r"""
        $L_e$ is the engagement length and $0.75$ the fraction of the cylinder in shear contact. Good
        practice engages enough thread that the bolt yields *before* the threads strip; the tool reports
        the minimum engagement that develops the bolt proof load:
        """),
    ("eq", r"L_{e,\min} = \frac{F_p}{\pi\,d\,(0.75)\,(0.577\,S_y)}"),
    ("md", r"""
        ### 15. Bolt-Group / Pattern Analysis (elastic method)

        An eccentrically loaded fastener pattern is analysed by the classical **elastic (rigid-plate)
        method**, superposing a direct share and a moment-induced share, all referred to the
        **centroid** of the group (equal bolt areas assumed).

        **Tension** — a concentric load $P$ is shared equally; an overturning moment $M$ adds tension in
        proportion to each bolt's distance $c_i$ from the centroidal bending axis:
        """),
    ("eq", r"F_{t,i} = \frac{P}{N} + \frac{M\,c_i}{\sum_j c_j^2}"),
    ("md", r"""
        The bolt with the largest $F_{t,i}$ is the **governing** bolt; its tension range drives the
        single-bolt fatigue, separation and proof checks above.

        **Shear** — an in-plane shear $V$ at eccentricity $e$ gives a direct share $V/N$ plus a
        torsional share scaling with radius $r_i$ and the group polar moment $J = \sum (x_i^2 + y_i^2)$,
        added as vectors; the largest resultant governs:
        """),
    ("eq", r"\vec F_{v,i} = \frac{\vec V}{N} + \frac{T\,\vec r_i}{J}, \qquad T = V\,e"),
    ("md", r"""
        **Slip-critical** — in a friction joint the shear is carried by friction from the clamp, not by
        the shank. With slip coefficient $\mu$ acting over $n_s$ faying surfaces:
        """),
    ("eq", r"n_{slip} = \frac{\mu\,n_s\,F_i}{V_{bolt}}"),
    ("cap", "The tension distribution assumes rigid members rotating about the centroid; it can "
            "be unconservative for prying-dominated or gasketed joints, where a neutral axis at "
            "the joint edge would load the extreme bolts more heavily."),
    ("md", r"""
        ### 16. Torque ↔ Preload Inversion (Fastener Tools)

        The nut-factor relation of §4 is linear in the clamp force, so it inverts directly: given a
        measured or applied wrench torque, the achieved preload is
        """),
    ("eq", r"F_i = \frac{T}{K\,d}"),
    ("md", r"""
        with $T$ the torque, $K$ the nut factor and $d$ the nominal diameter. This is what the
        **Torque → preload** tool reports, expressed as a force and as a fraction of the proof load.
        Because the same $K$ governs both directions, the **same relative scatter** of the chosen
        tightening method (§5) applies to the inferred preload, so the tool also shows the
        $F_i(1\pm s)$ band. The inversion inherits every caveat of $K$: the band is wide for dry or
        as-received hardware and the point value should not be trusted to better than the method's
        scatter.
        """),
    ("md", r"""
        ### 17. Exact Torque-Tension Relationship (Shigley)
        
        To avoid the bundled approximation of the nut factor $K$, the exact theoretical torque $T$ required 
        to raise a load (preload) is calculated using Shigley's power-screw mechanics, summing the thread 
        torque and the collar torque:
        """),
    ("eq", r"T = \frac{F_i \cdot d_p}{2} \left( \frac{l + \pi f_t d_p \sec\alpha}{\pi d_p - f_t l \sec\alpha} \right) + \frac{F_i \cdot f_c \cdot d_c}{2}"),
    ("md", r"""
        Where:
        - $F_i$ is the target preload force
        - $d_p = d - 0.649519 \cdot p$ is the pitch diameter (for standard $60^\circ$ threads)
        - $l = p$ is the thread lead (equal to pitch for single-start fasteners)
        - $\alpha = 30^\circ$ is the thread half-angle ($\sec 30^\circ \approx 1.1547$)
        - $f_t$ and $f_c$ are the independent thread and collar friction coefficients
        - $d_c$ is the effective collar bearing diameter (typically averaged from clearance and hex across-flats)
        
        This formulation allows direct comparison of the separated friction components against the simplified nut factor equation.
        """),
    ("md", r"""
        ### 18. Angle (Turn-of-Nut) Control (Fastener Tools)

        Angle control sets preload by rotating the nut a defined amount past a **snug** point, which
        avoids the friction scatter that dominates torque control. Past snug, the nut advances along
        the thread; one full turn ($360^\circ$) feeds one pitch $p$ of axial travel, which is taken up
        as bolt stretch plus member compression — the series-spring deflection of §7–§9:
        """),
    ("eq", r"\delta = (F_i - F_{snug})\left(\frac{1}{k_b} + \frac{1}{k_m}\right), \qquad "
           r"\theta = 360^\circ\,\frac{\delta}{p}"),
    ("md", r"""
        The tool takes the target preload $F_i$ and a user snug fraction $F_{snug}$ and returns
        $\theta$ in degrees and turns. This is the **elastic** angle only; in practice additional
        rotation is consumed by run-down, gasket/joint seating and embedment (§10), so a measured
        turn-of-nut specification is always larger than $\theta$. Its advantage is repeatability:
        because $k_b$, $k_m$ and $p$ are geometric, the angle is insensitive to the friction variability
        that plagues the torque method.
        """),
    ("md", r"""
        ### 19. Bolt Length and Thread Engagement in the Grip (Fastener Tools)

        The required bolt length is the grip (clamped thickness, §7) plus everything the bolt must span
        beyond it — washers, the nut height and a short thread protrusion:
        """),
    ("eq", r"L_{\min} = L_{grip} + n_w t_w + h_{nut} + \ell_{protr}"),
    ("md", r"""
        The tool returns $L_{\min}$ and rounds **up** to the next preferred standard length. It then
        checks whether the threaded portion intrudes into the grip. The nominal thread length $b$
        follows **ISO 888** for metric bolts and **ASME B18.2.1** for inch bolts; the unthreaded shank
        in a bolt of length $L$ is $\ell_{shank} = L - b$:
        """),
    ("eq", r"b_{\text{metric}} = \begin{cases} 2d + 6 & L \le 125 \\ 2d + 12 & 125 < L \le 200 \\ "
           r"2d + 25 & L > 200 \end{cases}\ \text{[mm]}"),
    ("eq", r"b_{\text{inch}} = \begin{cases} 2d + 6.35 & L \le 152.4 \\ 2d + 12.7 & L > 152.4 "
           r"\end{cases}\ \text{[mm]}\quad (2d + \tfrac14\,\text{in},\ 2d + \tfrac12\,\text{in})"),
    ("md", r"""
        If $\ell_{shank} < L_{grip}$ the **threads lie within the grip**: the reduced tensile-stress-area
        section then spans part of the clamped length, lowering the bolt stiffness $k_b$ (§7) and
        slightly raising the joint constant $C$. The tool flags this so a longer bolt or a shorter
        thread length can be chosen when a full shank in the grip is wanted.
        """),
    ("md", r"""
        ### 20. Bolt-Size / Grade Selection (Fastener Tools)

        The selector inverts the design problem: instead of analysing one bolt, it sweeps **every**
        size × grade in the database through the full single-bolt analysis (§1–§14) for the current
        joint and per-bolt loads, and keeps the combinations that satisfy the chosen factor-of-safety
        targets:
        """),
    ("eq", r"n_p \ge n_p^{\,*}, \quad n_{sep} \ge n_{sep}^{\,*}\ (\text{if } P>0), \quad "
           r"n_f \ge n_f^{\,*}\ (\text{if } P_{\max} \ne P_{\min})"),
    ("md", r"""
        The separation target is applied only when an external load is present, and the fatigue target
        only when the load actually cycles. Passing candidates are ranked by tensile-stress area $A_t$
        (a proxy for size, weight and cost), and the smallest is recommended. Because it reuses the
        same engine, the recommendation is fully consistent with the rest of the analysis — including
        crushing limits, thermal preload change and the selected fatigue criterion.
        """),
    ("md", r"""
        ### 21. Fastener Reference Dimensions (Fastener Tools)

        A convenience lookup of standard hardware dimensions for the selected size: the hex head/nut
        width across flats, the hex-key (Allen) size of a socket-head cap screw, a typical free-fit
        clearance hole, and the tap-drill diameter from the coarse-thread rule
        """),
    ("eq", r"d_{tap} \approx d - p"),
    ("md", r"""
        Wrench and hex-key sizes are standard tool sizes (ISO 272 / ISO 4762 for metric; inch tools for
        the imperial series); the clearance hole follows ISO 273 "medium" for metric and common
        clearance drills for imperial. These are nominal reference values for selection and layout —
        confirm against the governing fastener standard for critical work.
        """),
    ("md", r"""
        ### 22. Result Visualizations

        The visualizations are plots of the equations above, not new models:

        - **Joint cross-section & compression cone** — a scaled drawing of the stack with the
          Rötscher double frustum of §8 overlaid (30° half-angle, base at the bearing diameter $d_w$),
          so the load path and the compressed member zone are visible.
        - **Joint diagram (force vs. deflection)** — the classic preload triangle of §7–§9: the bolt
          line of slope $k_b$ and the member line of slope $k_m$ meeting at the preload $F_i$; their
          relative slopes are the geometric meaning of $C$.
        - **Bolt & member force vs. external load** — $F_b$ and $F_m$ from §9 plotted against the applied
          load $P$, with the **separation point** $P_{sep}=F_i/(1-C)$ marked. Past separation the member
          force stays at zero and the bolt carries the full load (the line steepens to slope 1).
        - **Clamp-load budget** — a waterfall stepping the installation preload through the embedment loss
          and thermal change (§10) and the external-load relief $(1-C)P_{\max}$ (§9) down to the residual
          member clamp $F_m$.
        - **External-load sharing** — the split of an applied load into the bolt share $C$ and the member
          relief $1-C$ (§9).
        - **Bolt-group shear vectors** — the per-bolt resultant shear $\vec F_{v,i}$ of §15 (direct
          $V/N$ plus torsional $T\vec r_i/J$) drawn as arrows on the pattern, with tension as the marker
          colour.
        """),
    ("md", r"""
        ### 22. FE Results Import (CSV)

        The **FE Import** tab evaluates per-bolt results from an external finite-element model. The CSV
        is read in **SI units** and the axial columns are the **total bolt force** over the duty cycle
        (FE already resolves preload and contact load-sharing), so the joint-stiffness constant $C$ is
        **not** re-applied — the stresses come straight from the imported forces, referred to $A_t$:
        """),
    ("eq", r"\sigma_{\max} = \frac{F_{ax,\max}}{A_t}, \quad "
           r"\sigma_a = \frac{F_{ax,\max} - F_{ax,\min}}{2A_t}, \quad "
           r"\sigma_m = \frac{F_{ax,\max} + F_{ax,\min}}{2A_t}"),
    ("md", r"""
        Each bolt gets four factors of safety; the **minimum governs** and is compared to the target:
        """),
    ("eq", r"n_{proof} = \frac{S_p A_t}{F_{ax,\max}}, \qquad "
           r"n_{shear} = \frac{0.577\,S_y}{\tau}, \quad \tau = \frac{V_{\max}}{A_t}"),
    ("eq", r"n_{comb} = \left[\left(\tfrac{\sigma_{\max}}{S_p}\right)^2 + "
           r"\left(\tfrac{\tau}{0.577\,S_y}\right)^2\right]^{-1/2}"),
    ("md", r"""
        $n_{fatigue}$ uses the selected mean-stress (or VDI) criterion of §11, with the load line from
        the preload point when `preload_N` is supplied. Shear is referred to $A_t$ (threads assumed in
        the shear plane — conservative), and $n_{comb}$ is an elliptic tension-shear interaction. Rows
        sharing a `bolt_id` are enveloped (max tension / min tension / max shear).

        The tab also presents **population graphics** for the whole fleet — a min-FOS histogram and
        cumulative distribution, a governing-check breakdown, a worst-bolts ranking, a normalised
        **Haigh** scatter ($\sigma_m/S_{ut}$ vs $\sigma_a/S_e$ with the Goodman reference) and a
        **tension-shear** scatter against the unit interaction ellipse, plus FOS-by-check box plots.
        This tab is independent of the rest of the app: those graphics and the per-bolt table are
        written **only** to its own CSV + PDF report and never appear in the other tabs' exports.
        """),
    ("md", r"""
        ### 23. Key Assumptions and Limitations

        - **Linear-elastic** bolt and members; no plasticity, creep or embedment relaxation is modelled
          (other than capping preload at the crushing limit).
        - **Concentric axial** single-bolt model; use the Bolt Group tab for eccentric/offset loads.
        - **Frustum member model** with the cone base at $d_w$ (§8); very thin, very thick or highly
          layered stacks are approximations.
        - **No prying** amplification in the single-bolt model; flexible flanges can load the bolt well
          above $C\,P$.
        - **Friction-based torque** carries real scatter; for critical joints prefer angle control or
          direct tensioning and verify by audit.
        - Database properties are **nominal, room-temperature** values — confirm $S_y$, $S_{ut}$ and
          $S_e$ against your certificates, especially at temperature.

        This tool supports design and study; it does not replace the governing code, standard, or the
        judgement of a responsible engineer.
        """),
]
