import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Info,
  LayoutDashboard,
  Ruler,
  FlaskConical,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Circle,
} from "lucide-react";
import { API_BASE } from "../config";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "LUMINA — Educational Breast Cancer Classifier" },
      {
        name: "description",
        content:
          "Educational ML demonstration classifying cell nucleus measurements as Benign or Malignant using the Wisconsin Breast Cancer Dataset.",
      },
      { property: "og:title", content: "LUMINA — Educational Breast Cancer Classifier" },
      { property: "og:description", content: "Educational ML demonstration. Not a medical diagnostic tool." },
    ],
  }),
  component: Lumina,
});

type FeatureKey =
  | "radius"
  | "texture"
  | "perimeter"
  | "area"
  | "smoothness"
  | "compactness"
  | "concavity"
  | "concave_points"
  | "symmetry"
  | "fractal_dim";

interface FeatureDef {
  key: FeatureKey;
  label: string;
  description: string;
  placeholder: string;
  min: number;
  max: number;
}

const FEATURES: FeatureDef[] = [
  { key: "radius",         label: "Mean Radius",         description: "Average distance from center to perimeter", placeholder: "6.0 – 28.0",    min: 6,     max: 28 },
  { key: "texture",        label: "Mean Texture",        description: "Variation in grey-scale pixel intensity",   placeholder: "9.0 – 40.0",    min: 9,     max: 40 },
  { key: "perimeter",      label: "Mean Perimeter",      description: "Total boundary length of the nucleus",      placeholder: "40.0 – 190.0",  min: 40,    max: 200 },
  { key: "area",           label: "Mean Area",           description: "Total area of the cell nucleus",            placeholder: "140 – 2500",    min: 140,   max: 2600 },
  { key: "smoothness",     label: "Mean Smoothness",     description: "Variation in radius lengths",               placeholder: "0.05 – 0.17",   min: 0.05,  max: 0.17 },
  { key: "compactness",    label: "Mean Compactness",    description: "Perimeter² / area − 1.0",                   placeholder: "0.02 – 0.35",   min: 0.02,  max: 0.35 },
  { key: "concavity",      label: "Mean Concavity",      description: "Severity of concave portions of contour",   placeholder: "0.00 – 0.45",   min: 0,     max: 0.45 },
  { key: "concave_points", label: "Mean Concave Points", description: "Number of concave portions of contour",     placeholder: "0.00 – 0.20",   min: 0,     max: 0.21 },
  { key: "symmetry",       label: "Mean Symmetry",       description: "Symmetry of the nucleus shape",             placeholder: "0.10 – 0.30",   min: 0.10,  max: 0.31 },
  { key: "fractal_dim",    label: "Mean Fractal Dim.",   description: "Coastline approximation of the boundary",   placeholder: "0.04 – 0.10",   min: 0.04,  max: 0.11 },
];

const BENIGN_SAMPLE: Record<FeatureKey, number> = {
  radius: 11.76, texture: 21.6, perimeter: 74.72, area: 427.9,
  smoothness: 0.08637, compactness: 0.04966, concavity: 0.01657,
  concave_points: 0.01115, symmetry: 0.1495, fractal_dim: 0.05888,
};
const MALIGNANT_SAMPLE: Record<FeatureKey, number> = {
  radius: 20.57, texture: 17.77, perimeter: 132.9, area: 1326.0,
  smoothness: 0.08474, compactness: 0.07864, concavity: 0.0869,
  concave_points: 0.07017, symmetry: 0.1812, fractal_dim: 0.05667,
};

type FormValues = Record<FeatureKey, string>;

const emptyForm = (): FormValues =>
  FEATURES.reduce((acc, f) => { acc[f.key] = ""; return acc; }, {} as FormValues);

interface ClassificationResult {
  classification: "BENIGN" | "MALIGNANT";
  confidence: number;
  probabilities: { benign: number; malignant: number };
}

function localClassify(values: Record<FeatureKey, number>): ClassificationResult {
  const z =
    -10 +
    0.18 * values.radius +
    0.025 * values.texture +
    0.0035 * values.area +
    18 * values.concavity +
    35 * values.concave_points +
    8 * values.compactness;
  const pMal = 1 / (1 + Math.exp(-z));
  const pBen = 1 - pMal;
  const isMal = pMal >= 0.5;
  return {
    classification: isMal ? "MALIGNANT" : "BENIGN",
    confidence: isMal ? pMal : pBen,
    probabilities: { benign: pBen, malignant: pMal },
  };
}

type SectionId = "overview" | "measurements" | "analyser" | "about";

const NAV: { id: SectionId; label: string; icon: typeof LayoutDashboard }[] = [
  { id: "overview",     label: "Overview",     icon: LayoutDashboard },
  { id: "measurements", label: "Measurements", icon: Ruler },
  { id: "analyser",     label: "Analyser",     icon: FlaskConical },
  { id: "about",        label: "About",        icon: BookOpen },
];

function Lumina() {
  const [values, setValues] = useState<FormValues>(emptyForm);
  const [invalid, setInvalid] = useState<Set<FeatureKey>>(new Set());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [barsAnimated, setBarsAnimated] = useState(false);
  const [active, setActive] = useState<SectionId>("overview");
  const [collapsed, setCollapsed] = useState(false);
  const scrollRef = useRef<HTMLElement | null>(null);
  const sectionRefs = useRef<Record<SectionId, HTMLElement | null>>({
    overview: null, measurements: null, analyser: null, about: null,
  });
  const resultRef = useRef<HTMLDivElement | null>(null);

  // Animate result bars
  useEffect(() => {
    if (result) {
      setBarsAnimated(false);
      const id = requestAnimationFrame(() => {
        requestAnimationFrame(() => setBarsAnimated(true));
      });
      resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      return () => cancelAnimationFrame(id);
    }
  }, [result]);

  // Track scroll position to highlight active nav
  useEffect(() => {
    const root = scrollRef.current;
    if (!root) return;
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActive(entry.target.getAttribute("data-section") as SectionId);
          }
        });
      },
      { root, rootMargin: "-40% 0px -55% 0px", threshold: 0 },
    );
    Object.values(sectionRefs.current).forEach((el) => el && obs.observe(el));
    return () => obs.disconnect();
  }, []);

  const allFilled = useMemo(
      () => FEATURES.every((f) => values[f.key].trim() !== ""),
      [values],
    );

  const handleChange = (key: FeatureKey, v: string) => {
    setValues((prev) => ({ ...prev, [key]: v }));
    if (invalid.has(key)) {
      const next = new Set(invalid);
      next.delete(key);
      setInvalid(next);
    }
  };

  const fillSample = (sample: Record<FeatureKey, number>) => {
    const filled = {} as FormValues;
    FEATURES.forEach((f) => { filled[f.key] = String(sample[f.key]); });
    setValues(filled);
    setInvalid(new Set());
    setError(null);
  };

  const scrollTo = (id: SectionId) => {
    const el = sectionRefs.current[id];
    if (el && scrollRef.current) {
      scrollRef.current.scrollTo({ top: el.offsetTop - 24, behavior: "smooth" });
    }
  };

  const handleAnalyse = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const empty = new Set<FeatureKey>();
    const outOfRange = new Set<FeatureKey>();
    const numeric = {} as Record<FeatureKey, number>;
    for (const f of FEATURES) {
      const raw = values[f.key].trim();
      if (raw === "") { empty.add(f.key); continue; }
      const n = Number(raw);
      if (!Number.isFinite(n)) { outOfRange.add(f.key); continue; }
      if (n < f.min || n > f.max) { outOfRange.add(f.key); }
      numeric[f.key] = n;
    }
    if (empty.size > 0) {
      setInvalid(empty);
      setError("Please fill in all 10 measurements.");
      return;
    }
    if (outOfRange.size > 0) {
      setInvalid(outOfRange);
      setError("Some values are outside the expected range.");
      return;
    }

    setLoading(true);
    try {
      let res: ClassificationResult;
      if (API_BASE) {
        const resp = await fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            inputs: numeric,
          }),
        });
        if (!resp.ok) {
          if (resp.status === 400) {
            const j = await resp.json().catch(() => ({}));
            throw new Error((j as { message?: string }).message ?? "Invalid input.");
          }
          throw new Error("Could not reach the server. Please try again later.");
        }
        const data = await resp.json();
        const probBen = Number(data.probabilities?.benign ?? data.benign ?? 0);
        const probMal = Number(data.probabilities?.malignant ?? data.malignant ?? 0);
        const cls = (String(
          data.classification ?? (probMal >= probBen ? "MALIGNANT" : "BENIGN"),
        ).toUpperCase()) as "BENIGN" | "MALIGNANT";
        res = {
          classification: cls,
          confidence: Number(data.confidence ?? Math.max(probBen, probMal)),
          probabilities: { benign: probBen, malignant: probMal },
        };
      } else {
        await new Promise((r) => setTimeout(r, 650));
        res = localClassify(numeric);
      }
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not reach the server. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setValues(emptyForm());
    setInvalid(new Set());
    setError(null);
    setResult(null);
    scrollTo("analyser");
  };

  const sidebarWidth = collapsed ? "72px" : "244px";

  return (
    <div
      className="flex min-h-screen w-full"
      style={{ backgroundColor: "var(--lumina-bg)", color: "var(--lumina-text)", fontFamily: "var(--font-inter)" }}
    >
      {/* ─────────────── SIDEBAR ─────────────── */}
      <aside
        className="hidden md:flex flex-col shrink-0 sticky top-0 h-screen border-r transition-[width] duration-300"
        style={{
          width: sidebarWidth,
          backgroundColor: "var(--lumina-panel)",
          borderColor: "var(--lumina-border)",
        }}
      >
        <div className="flex items-center gap-2.5 px-5 pt-6 pb-8">
          <span
            className="inline-flex items-center justify-center rounded-full shrink-0"
            style={{ width: 28, height: 28, backgroundColor: "var(--lumina-benign)" }}
          >
            <Circle className="w-3 h-3" style={{ color: "var(--lumina-panel)" }} fill="currentColor" />
          </span>
          {!collapsed && (
            <div className="flex flex-col leading-tight">
              <span
                className="text-[15px] font-semibold tracking-[0.14em]"
                style={{ fontFamily: "var(--font-serif)" }}
              >
                LUMINA
              </span>
              <span className="text-[10px] uppercase tracking-[0.2em]" style={{ color: "var(--lumina-muted)" }}>
                Cell Classifier
              </span>
            </div>
          )}
        </div>

        <nav className="flex-1 px-3">
          {!collapsed && (
            <p
              className="text-[10px] uppercase tracking-[0.22em] px-3 mb-3"
              style={{ color: "var(--lumina-muted)" }}
            >
              Sections
            </p>
          )}
          <ul className="flex flex-col gap-1">
            {NAV.map((n) => {
              const isActive = active === n.id;
              const Icon = n.icon;
              return (
                <li key={n.id}>
                  <button
                    type="button"
                    onClick={() => scrollTo(n.id)}
                    title={n.label}
                    className="w-full flex items-center gap-3 px-3 py-2.5 text-sm rounded-md transition-colors text-left"
                    style={{
                      backgroundColor: isActive ? "var(--lumina-surface)" : "transparent",
                      color: isActive ? "var(--lumina-text)" : "var(--lumina-muted)",
                      border: `1px solid ${isActive ? "var(--lumina-border)" : "transparent"}`,
                      fontWeight: isActive ? 500 : 400,
                    }}
                  >
                    <Icon className="w-4 h-4 shrink-0" style={{ color: isActive ? "var(--lumina-benign)" : "var(--lumina-muted)" }} />
                    {!collapsed && <span>{n.label}</span>}
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="px-3 py-4 border-t" style={{ borderColor: "var(--lumina-border)" }}>
          {!collapsed && (
            <p className="text-[10px] leading-relaxed px-2 mb-3" style={{ color: "var(--lumina-muted)" }}>
              Educational demonstration only.
              Not a medical diagnostic tool.
            </p>
          )}
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-md text-xs transition-colors"
            style={{
              border: "1px solid var(--lumina-border)",
              backgroundColor: "var(--lumina-surface)",
              color: "var(--lumina-muted)",
            }}
          >
            {collapsed ? <ChevronRight className="w-3.5 h-3.5" /> : (
              <>
                <ChevronLeft className="w-3.5 h-3.5" />
                <span>Collapse</span>
              </>
            )}
          </button>
        </div>
      </aside>

      {/* ─────────────── MAIN ─────────────── */}
      <main
        ref={scrollRef}
        className="flex-1 h-screen overflow-y-auto scroll-smooth"
      >
        {/* Mobile top bar */}
        <div
          className="md:hidden sticky top-0 z-10 flex items-center justify-between px-5 py-3 border-b backdrop-blur"
          style={{
            backgroundColor: "color-mix(in oklab, var(--lumina-bg) 92%, transparent)",
            borderColor: "var(--lumina-border)",
          }}
        >
          <div className="flex items-center gap-2">
            <span
              className="inline-block rounded-full"
              style={{ width: 10, height: 10, backgroundColor: "var(--lumina-benign)" }}
            />
            <span className="text-sm font-semibold tracking-[0.14em]" style={{ fontFamily: "var(--font-serif)" }}>
              LUMINA
            </span>
          </div>
          <div className="flex gap-1 overflow-x-auto">
            {NAV.map((n) => (
              <button
                key={n.id}
                onClick={() => scrollTo(n.id)}
                className="text-[11px] uppercase tracking-wider px-2.5 py-1 rounded"
                style={{
                  color: active === n.id ? "var(--lumina-text)" : "var(--lumina-muted)",
                  backgroundColor: active === n.id ? "var(--lumina-panel)" : "transparent",
                }}
              >
                {n.label}
              </button>
            ))}
          </div>
        </div>

        {/* ─────────────── OVERVIEW ─────────────── */}
        <section
          data-section="overview"
          ref={(el) => { sectionRefs.current.overview = el; }}
          className="px-6 sm:px-14 pt-16 pb-24"
        >
          <div className="max-w-3xl">
            <p
              className="text-[10px] uppercase tracking-[0.28em] lumina-fade-up"
              style={{ color: "var(--lumina-muted)", animationDelay: "0ms" }}
            >
              01 — Overview
            </p>
            <h1
              className="mt-6 leading-[1.05] tracking-tight lumina-fade-up"
              style={{
                fontFamily: "var(--font-serif)",
                fontSize: "clamp(2.6rem, 5.2vw, 4.2rem)",
                fontWeight: 500,
                animationDelay: "100ms",
              }}
            >
              Clarity in every
              <br />
              <span style={{ color: "var(--lumina-benign)" }}>sample.</span>
            </h1>
            <p
              className="mt-8 max-w-xl text-[15px] leading-[1.75] lumina-fade-up"
              style={{ color: "var(--lumina-muted)", animationDelay: "200ms" }}
            >
              LUMINA is an educational demonstration of a machine learning model trained on the
              Wisconsin Breast Cancer Dataset. It classifies ten cell nucleus measurements as
              <span style={{ color: "var(--lumina-benign)" }}> Benign</span> or
              <span style={{ color: "var(--lumina-malignant)" }}> Malignant</span> — for learning only.
            </p>

            <div
              className="mt-8 p-4 rounded-md flex gap-3 items-start lumina-fade-up"
              style={{
                backgroundColor: "var(--lumina-surface)",
                border: "1px solid var(--lumina-border)",
                animationDelay: "300ms",
              }}
            >
              <Info className="w-4 h-4 mt-0.5 shrink-0" style={{ color: "var(--lumina-muted)" }} />
              <p className="text-[13px] leading-relaxed" style={{ color: "var(--lumina-muted)" }}>
                For educational purposes only. This tool does not provide medical advice. Always
                consult a licensed medical professional.
              </p>
            </div>

            <div className="mt-10 flex flex-wrap gap-3 lumina-fade-up" style={{ animationDelay: "400ms" }}>
              <button
                type="button"
                onClick={() => scrollTo("analyser")}
                className="px-6 py-3 text-sm font-medium rounded-md transition-opacity hover:opacity-90"
                style={{ backgroundColor: "var(--lumina-benign)", color: "#FBF8F2" }}
              >
                Analyse a sample
              </button>
              <button
                type="button"
                onClick={() => scrollTo("measurements")}
                className="px-6 py-3 text-sm font-medium rounded-md transition-colors"
                style={{
                  border: "1px solid var(--lumina-border-med)",
                  color: "var(--lumina-text)",
                  backgroundColor: "transparent",
                }}
              >
                See the 10 measurements
              </button>
            </div>

            {/* Stat strip */}
            <div className="mt-16 grid grid-cols-2 sm:grid-cols-3 gap-6">
              {[
                { k: "10", v: "Nucleus measurements" },
                { k: "569", v: "Training samples" },
                { k: "2", v: "Output classes" },
              ].map((s, i) => (
                <div
                  key={s.v}
                  className="pl-4 lumina-fade-up"
                  style={{
                    borderLeft: "2px solid var(--lumina-accent)",
                    animationDelay: `${500 + i * 80}ms`,
                  }}
                >
                  <div
                    className="text-3xl font-medium"
                    style={{ fontFamily: "var(--font-serif)", color: "var(--lumina-text)" }}
                  >
                    {s.k}
                  </div>
                  <div className="text-[11px] uppercase tracking-wider mt-1" style={{ color: "var(--lumina-muted)" }}>
                    {s.v}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ─────────────── MEASUREMENTS ─────────────── */}
        <section
          data-section="measurements"
          ref={(el) => { sectionRefs.current.measurements = el; }}
          className="px-6 sm:px-14 py-20 border-t"
          style={{ borderColor: "var(--lumina-border)", backgroundColor: "var(--lumina-surface)" }}
        >
          <div className="max-w-5xl">
            <p className="text-[10px] uppercase tracking-[0.28em]" style={{ color: "var(--lumina-muted)" }}>
              02 — Measurements
            </p>
            <h2
              className="mt-4 text-3xl sm:text-4xl tracking-tight"
              style={{ fontFamily: "var(--font-serif)", fontWeight: 500 }}
            >
              The ten inputs the model reads.
            </h2>
            <p className="mt-3 max-w-xl text-sm" style={{ color: "var(--lumina-muted)" }}>
              Each measurement describes a geometric or textural property of the cell nucleus,
              averaged across the sample.
            </p>

            <div className="mt-10 grid grid-cols-1 sm:grid-cols-2 gap-x-10 gap-y-5">
              {FEATURES.map((f, i) => (
                <div
                  key={f.key}
                  className="flex gap-4 pt-4 border-t"
                  style={{ borderColor: "var(--lumina-border)" }}
                >
                  <div
                    className="text-xs font-medium pt-0.5 shrink-0 w-6"
                    style={{ color: "var(--lumina-accent)" }}
                  >
                    {String(i + 1).padStart(2, "0")}
                  </div>
                  <div>
                    <div className="font-medium text-sm" style={{ color: "var(--lumina-text)" }}>
                      {f.label}
                    </div>
                    <div className="text-xs mt-1 leading-relaxed" style={{ color: "var(--lumina-muted)" }}>
                      {f.description}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ─────────────── ANALYSER ─────────────── */}
        <section
          data-section="analyser"
          ref={(el) => { sectionRefs.current.analyser = el; }}
          className="px-6 sm:px-14 py-20 border-t"
          style={{ borderColor: "var(--lumina-border)" }}
        >
          <div className="max-w-3xl">
            <p className="text-[10px] uppercase tracking-[0.28em]" style={{ color: "var(--lumina-muted)" }}>
              03 — Analyser
            </p>
            <h2
              className="mt-4 text-3xl sm:text-4xl tracking-tight"
              style={{ fontFamily: "var(--font-serif)", fontWeight: 500 }}
            >
              Enter a sample.
            </h2>

            <div className="mt-3 flex flex-wrap gap-6">
              <button
                type="button"
                onClick={() => fillSample(BENIGN_SAMPLE)}
                className="text-xs underline underline-offset-4 hover:opacity-70"
                style={{ color: "var(--lumina-benign)" }}
              >
                Use benign sample
              </button>
              <button
                type="button"
                onClick={() => fillSample(MALIGNANT_SAMPLE)}
                className="text-xs underline underline-offset-4 hover:opacity-70"
                style={{ color: "var(--lumina-malignant)" }}
              >
                Use malignant sample
              </button>
            </div>

            <form onSubmit={handleAnalyse} noValidate className="mt-10">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-10 gap-y-7">
                {FEATURES.map((f) => {
                  const isInvalid = invalid.has(f.key);
                  return (
                    <div key={f.key}>
                      <label
                        htmlFor={f.key}
                        className="block text-[10px] uppercase tracking-[0.18em] mb-2"
                        style={{ color: "var(--lumina-muted)" }}
                      >
                        {f.label}
                      </label>
                      <input
                        id={f.key}
                        name={f.key}
                        type="number"
                        step="any"
                        inputMode="decimal"
                        placeholder={f.placeholder}
                        value={values[f.key]}
                        onChange={(e) => handleChange(f.key, e.target.value)}
                        className="lumina-input w-full bg-transparent py-2.5 text-sm outline-none appearance-none"
                        style={{
                          borderRadius: 0,
                          borderBottom: `1px solid ${isInvalid ? "var(--lumina-error)" : "var(--lumina-border-med)"}`,
                          color: "var(--lumina-text)",
                          transition: "border-color 250ms ease",
                        }}
                        onFocus={(e) => {
                          if (!isInvalid) e.currentTarget.style.borderBottomColor = "var(--lumina-benign)";
                        }}
                        onBlur={(e) => {
                          e.currentTarget.style.borderBottomColor = isInvalid
                            ? "var(--lumina-error)"
                            : "var(--lumina-border-med)";
                        }}
                      />
                    </div>
                  );
                })}
              </div>

              <div className="flex flex-wrap gap-3 mt-10">
                <button
                  type="submit"
                  disabled={!allFilled || loading}
                  className={`flex-1 min-w-[220px] py-3.5 font-medium text-sm rounded-md transition-opacity ${loading ? "lumina-pulse" : ""}`}
                  style={{
                    backgroundColor: !allFilled || loading ? "var(--lumina-border-med)" : "var(--lumina-benign)",
                    color: !allFilled || loading ? "var(--lumina-muted)" : "#FBF8F2",
                    cursor: !allFilled || loading ? "not-allowed" : "pointer",
                  }}
                >
                  {loading ? "Analysing..." : "Analyse sample"}
                </button>
                <button
                  type="button"
                  onClick={handleClear}
                  className="px-6 py-3.5 text-sm rounded-md transition-colors"
                  style={{
                    border: "1px solid var(--lumina-border-med)",
                    color: "var(--lumina-muted)",
                  }}
                >
                  Clear
                </button>
              </div>

              {error && (
                <p className="text-xs mt-3 lumina-fade-in" style={{ color: "var(--lumina-error)" }}>
                  {error}
                </p>
              )}
            </form>

            {/* Result */}
            {result && (
              <div
                ref={resultRef}
                className="mt-14 p-8 rounded-lg lumina-fade-up"
                style={{
                  backgroundColor: "var(--lumina-surface)",
                  border: "1px solid var(--lumina-border)",
                  animationDuration: "550ms",
                }}
              >
                <p className="text-[10px] uppercase tracking-[0.25em]" style={{ color: "var(--lumina-muted)" }}>
                  Classification result
                </p>
                <div className="flex items-baseline gap-4 mt-2 flex-wrap">
                  <div
                    className="font-medium tracking-tight"
                    style={{
                      fontFamily: "var(--font-serif)",
                      fontSize: "clamp(2rem, 4.5vw, 3rem)",
                      color:
                        result.classification === "BENIGN"
                          ? "var(--lumina-benign)"
                          : "var(--lumina-malignant)",
                    }}
                  >
                    {result.classification === "BENIGN" ? "Benign" : "Malignant"}
                  </div>
                  <div className="text-sm" style={{ color: "var(--lumina-muted)" }}>
                    {result.confidence.toFixed(1)}%
                  </div>
                </div>
                <p className="text-sm mt-3 max-w-md leading-relaxed" style={{ color: "var(--lumina-muted)" }}>
                  {result.classification === "BENIGN"
                    ? "The sample characteristics are consistent with non-cancerous tissue."
                    : "The sample characteristics show patterns associated with cancerous tissue."}
                </p>

                <div className="mt-8">
                  <p className="text-[10px] uppercase tracking-[0.2em] mb-4" style={{ color: "var(--lumina-muted)" }}>
                    Probability breakdown
                  </p>
                  {(["benign", "malignant"] as const).map((cls) => {
                    const color = cls === "benign" ? "var(--lumina-benign)" : "var(--lumina-malignant)";
                    const pct = result.probabilities[cls] ;
                    return (
                      <div key={cls} className="mb-5">
                        <div className="flex justify-between text-xs uppercase tracking-wider" style={{ color }}>
                          <span>{cls}</span>
                          <span>{pct.toFixed(1)}%</span>
                        </div>
                        <div
                          className="mt-2 overflow-hidden rounded-full"
                          style={{ height: 6, backgroundColor: "var(--lumina-panel)" }}
                        >
                          <div
                            style={{
                              height: "100%",
                              width: barsAnimated ? `${pct}%` : "0%",
                              backgroundColor: color,
                              transition: "width 900ms ease-out",
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div
                  className="mt-8 p-4 rounded-md flex gap-3 items-start"
                  style={{
                    backgroundColor: "var(--lumina-bg)",
                    border: "1px solid var(--lumina-border)",
                  }}
                >
                  <Info className="w-4 h-4 mt-0.5 shrink-0" style={{ color: "var(--lumina-muted)" }} />
                  <p className="text-xs leading-relaxed" style={{ color: "var(--lumina-muted)" }}>
                    Generated by a machine learning model for educational purposes only. This is not
                    a medical diagnosis. Consult a qualified medical professional for health advice.
                  </p>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* ─────────────── ABOUT ─────────────── */}
        <section
          data-section="about"
          ref={(el) => { sectionRefs.current.about = el; }}
          className="px-6 sm:px-14 py-20 border-t"
          style={{ borderColor: "var(--lumina-border)", backgroundColor: "var(--lumina-surface)" }}
        >
          <div className="max-w-3xl">
            <p className="text-[10px] uppercase tracking-[0.28em]" style={{ color: "var(--lumina-muted)" }}>
              04 — About
            </p>
            <h2
              className="mt-4 text-3xl sm:text-4xl tracking-tight"
              style={{ fontFamily: "var(--font-serif)", fontWeight: 500 }}
            >
              About the model.
            </h2>
            <div
              className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-6 text-sm leading-relaxed"
              style={{ color: "var(--lumina-muted)" }}
            >
              <div>
                <p className="text-xs uppercase tracking-wider mb-2" style={{ color: "var(--lumina-text)" }}>
                  Dataset
                </p>
                <p>
                  The Wisconsin Breast Cancer Dataset — 569 labeled samples of cell nucleus features,
                  digitized from fine needle aspirate (FNA) images.
                </p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wider mb-2" style={{ color: "var(--lumina-text)" }}>
                  Method
                </p>
                <p>
                  A supervised classifier maps the ten input measurements to a probability over the
                  benign and malignant classes.
                </p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wider mb-2" style={{ color: "var(--lumina-text)" }}>
                  Educational scope
                </p>
                <p>
                  LUMINA exists to help students and curious readers understand how tabular
                  classification works. It is not intended for clinical use.
                </p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wider mb-2" style={{ color: "var(--lumina-text)" }}>
                  Disclaimer
                </p>
                <p>
                  This tool does not provide medical advice, diagnosis, or treatment. Always consult
                  a licensed medical professional for health concerns.
                </p>
              </div>
            </div>

            <div
              className="mt-14 pt-6 flex flex-wrap items-center justify-between gap-3 text-[11px] uppercase tracking-[0.2em] border-t"
              style={{ borderColor: "var(--lumina-border)", color: "var(--lumina-muted)" }}
            >
              <span>LUMINA · Educational Classifier</span>
              <span>Not for clinical use</span>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}