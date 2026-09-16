/* =============================================================================
   HealthHub — front-end app script
   • theme (light/dark, persisted)          • app shell (sidebar + topbar)
   • REST client with graceful mock fallback • per-page controllers
   No build step. Loaded with `defer` from every page.
   ========================================================================== */
(() => {
  "use strict";

  // --------------------------------------------------------------- config ---
  const API_BASE = (window.HEALTHHUB_API || "http://localhost:8000/api").replace(/\/$/, "");
  const IN_PAGES = /\/pages\//.test(location.pathname);
  const ASSET = IN_PAGES ? "../assets" : "assets";
  const LINK  = IN_PAGES ? "" : "pages/";
  const HOME  = IN_PAGES ? "../index.html" : "index.html";
  const REQ_TIMEOUT = 12000;

  // ------------------------------------------------------------- tiny dom ---
  const $  = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const icon = (name, cls = "") => `<svg class="ic ${cls}" aria-hidden="true"><use href="${ASSET}/icons.svg#i-${name}"></use></svg>`;
  const fmt = (n, d = 0) => (n == null || isNaN(n)) ? "—" : Number(n).toLocaleString(undefined, { maximumFractionDigits: d });

  // ---------------------------------------------------------------- theme ---
  const Theme = {
    get() { return localStorage.getItem("hh-theme") || "system"; },
    apply(mode) {
      const root = document.documentElement;
      if (mode === "system") root.removeAttribute("data-theme");
      else root.setAttribute("data-theme", mode);
    },
    resolved() {
      const m = Theme.get();
      if (m !== "system") return m;
      return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    },
    toggle() {
      const next = Theme.resolved() === "dark" ? "light" : "dark";
      localStorage.setItem("hh-theme", next);
      Theme.apply(next);
      document.dispatchEvent(new CustomEvent("hh:theme", { detail: next }));
    },
    init() { Theme.apply(Theme.get()); }
  };
  Theme.init(); // pre-paint, before shell

  // ------------------------------------------------------- auth / profile ---
  // There is no password auth — the "session" is just which profile name is
  // active on this device. It lives in localStorage under "hh-profile".
  // Nothing is ever defaulted to a hardcoded person.
  const Auth = {
    KEY: "hh-profile",     // active profile name — the identity used for ?user=
    OBJ: "hh-user",        // {name,email,created_at} cached for quick UI display
    name() { try { return (localStorage.getItem(Auth.KEY) || "").trim() || null; } catch { return null; } },
    user() {
      try { return JSON.parse(localStorage.getItem(Auth.OBJ) || "null") || null; } catch { return null; }
    },
    set(n, obj) {
      const name = String(n || "").trim();
      if (!name) return;
      try {
        localStorage.setItem(Auth.KEY, name);
        localStorage.setItem(Auth.OBJ, JSON.stringify({
          name,
          email: (obj && obj.email) || (Auth.user() || {}).email || "",
          created_at: (obj && obj.created_at) || (Auth.user() || {}).created_at || new Date().toISOString(),
        }));
      } catch {}
    },
    _wipeSession() {
      try {
        // drop the active profile + any hh-* session keys, KEEP the theme choice
        Object.keys(localStorage)
          .filter(k => (k === "hh-profile" || (k.startsWith("hh-") && k !== "hh-theme")))
          .forEach(k => localStorage.removeItem(k));
        localStorage.removeItem("user");     // legacy keys, just in case
        localStorage.removeItem("profile");
      } catch {}
      try { sessionStorage.clear(); } catch {}
    },
    signOut() { Auth._wipeSession(); },
    // task 4: on load, if there is no active profile, purge residual keys
    sweepIfLoggedOut() { if (!Auth.name()) Auth._wipeSession(); },
  };
  Auth.sweepIfLoggedOut();

  // append ?user=<active profile> to a path when signed in
  const withUser = (path) => {
    const n = Auth.name();
    if (!n) return path;
    return path + (path.includes("?") ? "&" : "?") + "user=" + encodeURIComponent(n);
  };

  // ---------------------------------------------------------------- toast ---
  const Toast = {
    host: null,
    show(msg, kind = "") {
      if (!Toast.host) { Toast.host = el(`<div class="toast-host"></div>`); document.body.appendChild(Toast.host); }
      const t = el(`<div class="toast ${kind}">${icon(kind === "err" ? "alert" : kind === "ok" ? "check-circle" : "bell", "ic--sm")}<span>${esc(msg)}</span></div>`);
      Toast.host.appendChild(t);
      setTimeout(() => { t.style.opacity = "0"; t.style.transform = "translateX(24px)"; setTimeout(() => t.remove(), 250); }, 3600);
    }
  };

  // ----------------------------------------------------------- rest client ---
  async function request(method, path, body, isForm) {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), REQ_TIMEOUT);
    try {
      const opt = { method, signal: ctrl.signal, headers: {} };
      if (body !== undefined) {
        if (isForm) opt.body = body;
        else { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
      }
      const res = await fetch(`${API_BASE}${path}`, opt);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return { data: await res.json().catch(() => ({})), live: true };
    } catch (err) {
      const mock = resolveMock(method, path, body);
      if (mock !== undefined) return { data: mock, live: false };
      throw err;
    } finally { clearTimeout(to); }
  }
  const api = {
    get:  (p)      => request("GET", p),
    post: (p, b)   => request("POST", p, b),
    upload: (p, f) => request("POST", p, f, true),
  };

  // ------------------------------------------------------------- mock data ---
  // Offline demo content for FEATURE pages only. There is deliberately NO
  // default profile / user identity here — an offline+signed-out app shows the
  // "create a profile" state, exactly like the live backend.
  // offline-only: mirrors what a created profile would be. Persisted so an
  // offline reload keeps it; wiped by Auth.signOut() along with the other hh-* keys.
  let _demoProfile = (() => {
    try { return JSON.parse(localStorage.getItem("hh-demo-profile") || "null"); }
    catch { return null; }
  })();

  const MOCK = {
    "/dashboard": {
      calorie_target: 1860, protein_target: 128, tdee: 2210, bmi: 22.7,
      streak_days: 6, weight_change_kg: -1.8,
      activity: [
        { icon: "activity", text: "Fitness assessment updated", when: "2h ago" },
        { icon: "sparkles", text: "Skin analysis — Combination, 82%", when: "yesterday" },
        { icon: "chat",     text: "Asked the nutritionist about roti vs bread", when: "2 days ago" },
        { icon: "scale",    text: "Logged weight 61.0 kg", when: "3 days ago" },
      ],
    },
    "/fitness/assess": {
      bmr: 1410, tdee: 2210, calorie_target: 1860,
      macros: { protein_g: 128, carbs_g: 190, fat_g: 55 },
      meal_plan: {
        days: [
          { day: "Mon", meals: [ ["Breakfast", "2 egg omelette + 1 roti + tea"], ["Lunch", "Chicken karahi (150g) + 1 roti + salad"], ["Snack", "Greek yoghurt + banana"], ["Dinner", "Daal chana + 1 roti + cucumber"] ] },
          { day: "Tue", meals: [ ["Breakfast", "Vegetable poha + tea"], ["Lunch", "Beef seekh (2) + 1 roti + raita"], ["Snack", "Roasted chana (40g)"], ["Dinner", "Palak paneer + 1 roti"] ] },
          { day: "Wed", meals: [ ["Breakfast", "Anda paratha (1) + tea"], ["Lunch", "Chicken pulao (1 cup) + salad"], ["Snack", "Apple + peanut butter"], ["Dinner", "Fish curry + 1 roti"] ] },
        ],
        daily_cost_pkr: 233, within_budget: true,
      },
      workout: {
        split: "Full body · 3 days/week",
        days: [
          { day: "Day 1", items: ["Goblet squat 3×10", "Push-up 3×10", "1-arm row 3×10", "Plank 3×40s"] },
          { day: "Day 2", items: ["Romanian deadlift 3×10", "Incline press 3×10", "Lat pulldown 3×12", "Dead bug 3×12"] },
          { day: "Day 3", items: ["Split squat 3×10", "DB shoulder press 3×10", "Cable row 3×12", "Hanging knee raise 3×12"] },
        ],
      },
    },
    "/nutrition/history": {
      messages: [
        { role: "bot",  content: "Hi 👋 I'm your nutrition assistant. Ask me about calories, Pakistani foods or grocery prices." },
        { role: "user", content: "What's a cheap high-protein breakfast under PKR 200?" },
        { role: "bot",  content: "Two boiled eggs (~Rs 60) with one roti (~Rs 25) and a cup of milk tea (~Rs 30) gives you about 22 g protein for roughly Rs 115. Add a small banana for potassium and it's still under Rs 200." },
      ],
    },
    "/skin/analyze": {
      skin_type: "Combination", confidence: 82,
      reasoning: "Visible shine across the T-zone with matte, slightly tight cheeks — classic combination pattern. Lighting is even, so the forehead sheen reads as real sebum rather than glare.",
      blemish_count: 3, care_level: "light", conditions: ["Blemishes / breakouts"],
      concerns: [ { name: "Blemishes", severity: "mild" }, { name: "Uneven texture", severity: "mild" }, { name: "Redness", severity: "clear" } ],
      routine: [
        { step: "Gentle cleanser", product: "CeraVe Foaming Cleanser", price_pkr: 2450, why: "Cheapest gentle pick for combination skin", where_to_buy: "Daraz", alternatives: [ { name: "Simple Refreshing Face Wash", price_pkr: 1150 }, { name: "Neutrogena Fresh Foaming", price_pkr: 1900 } ] },
        { step: "Moisturizer", product: "Cetaphil Daily Oil-Free Moisturizer", price_pkr: 2100, why: "Lightweight, non-comedogenic", where_to_buy: "Pharmacy", alternatives: [ { name: "The Ordinary Natural Moisturizing Factors", price_pkr: 2600 } ] },
        { step: "Sunscreen", product: "ROTEX Sunblock SPF 60", price_pkr: 950, why: "Best value broad-spectrum", where_to_buy: "Daraz", alternatives: [ { name: "Beesline SPF 50", price_pkr: 3200 } ] },
        { step: "Optional niacinamide serum", product: "The Ordinary Niacinamide 10% + Zinc", price_pkr: 2500, why: "Evens tone, calms breakouts — 3×/week PM", where_to_buy: "Daraz", alternatives: [] },
      ],
      supplements: [
        { name: "Abbott Pakistan Surbex-Z", price_pkr: 480, dosage: "1 tablet", timing: "after breakfast", why: "Zinc + Vitamin C for skin repair" },
      ],
      supplement_note: "This combination is within safe daily limits. Don't add other zinc / vitamin-C / multivitamin products alongside it.",
      diet: [
        "2.5–3 L water/day",
        "More seasonal fruit, leafy greens, dahi, nuts",
        "Less deep-fried food, sugary chai and bakery items",
      ],
      brands_note: "Gentle products only — no retinoids or benzoyl peroxide. Supplements are Pakistani brands only.",
    },
    "/progress/entries": {
      entries: [
        { date: "2026-08-08", weight_kg: 62.8, waist_cm: 79, chest_cm: 93 },
        { date: "2026-08-15", weight_kg: 62.1, waist_cm: 78, chest_cm: 93 },
        { date: "2026-08-22", weight_kg: 61.6, waist_cm: 77.5, chest_cm: 92.5 },
        { date: "2026-08-29", weight_kg: 61.2, waist_cm: 77, chest_cm: 92.5 },
        { date: "2026-09-05", weight_kg: 61.0, waist_cm: 76.5, chest_cm: 92 },
      ],
    },
  };
  function resolveMock(method, path, body) {
    path = path.split("?")[0];   // ignore ?user=… when matching

    if (method === "GET" && path === "/profile") {
      return _demoProfile ? { ...(_demoProfile), exists: true } : { exists: false };
    }
    if (method === "POST" && path === "/profile") {
      _demoProfile = { ...(_demoProfile || {}), ...(body || {}) };
      try { localStorage.setItem("hh-demo-profile", JSON.stringify(_demoProfile)); } catch {}
      return { ok: true, profile: { ...(_demoProfile), exists: true } };
    }
    if (method === "POST" && path === "/nutrition/chat") {
      const q = (body && body.message || "").toLowerCase();
      let a = "Here's a quick take: aim for a slight calorie deficit, keep protein around 1.6 g per kg of body-weight, and lean on daal, eggs, chicken and yoghurt for cheap protein.";
      if (q.includes("roti") || q.includes("bread")) a = "One medium roti (~90 kcal, 3 g protein) beats two slices of white bread for fibre and satiety. Whole-wheat bread is comparable — pick whichever keeps you full longer for the same calories.";
      if (q.includes("hungry")) a = "Constant hunger usually means too few calories, not enough protein, or too little fibre. Add an egg or a cup of yoghurt to each meal and a large salad — volume with almost no calories.";
      return { role: "bot", content: a };
    }
    if (method === "POST" && path === "/nutrition/clear") return { ok: true, messages: [] };
    if (method === "POST" && path === "/fitness/assess") return MOCK["/fitness/assess"];
    if (method === "POST" && path === "/skin/analyze")  return MOCK["/skin/analyze"];
    if (method === "POST" && path === "/progress/entries") {
      const e = MOCK["/progress/entries"].entries;
      e.push({ date: new Date().toISOString().slice(0, 10), ...(body || {}) });
      return { ok: true, entries: e };
    }
    return MOCK[path];
  }

  // ------------------------------------------------------------- app shell ---
  const NAV = [
    { id: "dashboard", label: "Dashboard",  icon: "grid",     href: "dashboard.html" },
    { id: "fitness",   label: "Fitness",    icon: "dumbbell", href: "fitness.html" },
    { id: "nutrition", label: "Nutritionist", icon: "chat",   href: "nutrition.html" },
    { id: "skin",      label: "Skin Analysis", icon: "sparkles", href: "skin.html" },
    { id: "progress",  label: "Progress",   icon: "trending", href: "progress.html" },
    { id: "profile",   label: "Profile",    icon: "user",     href: "profile.html" },
  ];
  const TITLES = {
    dashboard: ["Dashboard", "Your health at a glance"],
    fitness:   ["Fitness Assessment", "Calorie target, meal plan & workout"],
    nutrition: ["AI Nutritionist", "Chat about diet, calories & prices"],
    skin:      ["Skin Analysis", "Upload a selfie for a skin-type read"],
    progress:  ["Progress Tracking", "Weight & measurements over time"],
    profile:   ["Profile", "Your details drive every recommendation"],
  };

  function mountShell(page) {
    const host = $("#shell");
    if (!host) return;
    const [title, sub] = TITLES[page] || ["HealthHub", ""];

    host.replaceWith(el(`
      <div class="sidebar" id="sidebar">
        <a class="brand" href="${HOME}">
          <img class="brand__logo" src="${ASSET}/logo.svg" alt="">
          <span class="brand__name">Health<span>Hub</span></span>
        </a>
        <nav class="nav">
          <div class="nav__label">Menu</div>
          ${NAV.map(n => `<a class="nav__link ${n.id === page ? "is-active" : ""}" href="${LINK}${n.href}">${icon(n.icon, "ic--sm")}<span>${n.label}</span></a>`).join("")}
        </nav>
        <div class="sidebar__foot">
          <span class="avatar sm" id="side-avatar">?</span>
          <div style="min-width:0;flex:1">
            <div style="font-weight:600;font-size:.9rem" id="side-name">Not signed in</div>
            <button type="button" id="sign-out-btn" class="faint"
                    style="font-size:.78rem;padding:0;border:0;background:none;cursor:pointer">Sign out</button>
          </div>
        </div>
      </div>
    `));

    // topbar injected at top of .app-body
    const body = $(".app-body");
    if (body) body.prepend(el(`
      <header class="topbar">
        <button class="icon-btn hamburger" id="hamburger" aria-label="Menu">${icon("menu")}</button>
        <div class="topbar__title">${title}<small>${sub}</small></div>
        <div class="spacer"></div>
        <button class="icon-btn" id="theme-btn" aria-label="Toggle theme">${icon(Theme.resolved() === "dark" ? "sun" : "moon")}</button>
        <span class="avatar" id="top-avatar">?</span>
      </header>
    `));

    // drawer
    const sidebar = $("#sidebar");
    const backdrop = el(`<div class="backdrop" id="backdrop"></div>`);
    document.body.appendChild(backdrop);
    const closeDrawer = () => { sidebar.classList.remove("is-open"); backdrop.classList.remove("show"); };
    $("#hamburger")?.addEventListener("click", () => { sidebar.classList.toggle("is-open"); backdrop.classList.toggle("show"); });
    backdrop.addEventListener("click", closeDrawer);
    $$(".nav__link").forEach(a => a.addEventListener("click", closeDrawer));

    // theme
    const paintThemeBtn = () => { const b = $("#theme-btn"); if (b) b.innerHTML = icon(Theme.resolved() === "dark" ? "sun" : "moon"); };
    $("#theme-btn")?.addEventListener("click", () => { Theme.toggle(); paintThemeBtn(); });
    document.addEventListener("hh:theme", paintThemeBtn);

    // sign out: clear the session and land on a clean create-profile form
    $("#sign-out-btn")?.addEventListener("click", () => {
      const wasIn = !!Auth.name();
      Auth.signOut();
      if (wasIn) Toast.show("Signed out.", "ok");
      location.href = LINK + "profile.html";
    });

    refreshIdentity();
  }

  function _setIdentity(name) {
    const initial = name ? (name.trim().charAt(0).toUpperCase() || "?") : "?";
    const nEl = $("#side-name"); if (nEl) nEl.textContent = name || "Not signed in";
    ["#side-avatar", "#top-avatar", "#pr-avatar"].forEach(s => { const e = $(s); if (e) e.textContent = initial; });
    const so = $("#sign-out-btn"); if (so) so.textContent = name ? "Sign out" : "Create profile";
  }

  // reflect the active profile in the shell. A stored name that the backend
  // doesn't (yet) know is kept — it just means "profile not finished", not
  // "signed out". Only the explicit Sign-out button clears the session.
  function refreshIdentity() {
    const n = Auth.name();
    _setIdentity(n || (Auth.user() || {}).name || null);
    if (!n) return;
    api.get(withUser("/profile")).then(({ data }) => {
      if (data && data.name) { Auth.set(data.name, { email: data.email }); _setIdentity(data.name); }
    }).catch(() => {});
  }

  // ------------------------------------------------------------ page: dash ---
  async function pageDashboard() {
    if (!Auth.name()) return renderNoProfile();
    const wrap = $("#dash-stats");
    const { data, live } = await api.get(withUser("/dashboard"));
    if (data && data.error) return renderNoProfile();
    banner(live);
    wrap.innerHTML = [
      ["Calorie target", fmt(data.calorie_target), "kcal / day", "flame"],
      ["Protein target", fmt(data.protein_target), "g / day", "target"],
      ["BMI", fmt(data.bmi, 1), "normal range", "activity"],
      ["Streak", fmt(data.streak_days), "days active", "check-circle"],
    ].map(([l, v, s, ic]) => `
      <div class="stat">
        <span class="stat__label">${l}</span>
        <span class="stat__value">${v}</span>
        <span class="muted" style="font-size:.82rem">${icon(ic, "ic--sm")} ${s}</span>
      </div>`).join("");

    $("#dash-activity").innerHTML = (data.activity || []).map(a => `
      <li class="row" style="padding:12px 0;border-bottom:1px solid var(--border)">
        <span class="avatar sm" style="background:var(--surface-2);color:var(--primary)">${icon(a.icon, "ic--sm")}</span>
        <span>${esc(a.text)}</span><span class="spacer"></span>
        <span class="faint" style="font-size:.8rem">${esc(a.when)}</span>
      </li>`).join("");

    const wc = data.weight_change_kg;
    $("#dash-weight").innerHTML = `
      <div class="row-between">
        <div><div class="stat__label">Weight change (30 days)</div>
        <div class="stat__value">${wc > 0 ? "+" : ""}${fmt(wc, 1)} kg</div></div>
        <span class="badge ${wc <= 0 ? "badge--ok" : "badge--warn"}">${wc <= 0 ? "On track" : "Review plan"}</span>
      </div>`;
  }

  // --------------------------------------------------------- page: fitness ---
  async function pageFitness() {
    const form = $("#fit-form");
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = $("#fit-submit"); btn.disabled = true; btn.innerHTML = `${icon("clock", "ic--sm")} Building…`;
      const payload = Object.fromEntries(new FormData(form).entries());
      ["age", "height_cm", "weight_kg", "budget_pkr"].forEach(k => payload[k] = Number(payload[k]));
      try {
        const { data, live } = await api.post("/fitness/assess", payload);
        renderFitness(data); banner(live, "#fit-banner");
        $("#fit-result").classList.remove("hidden");
        $("#fit-result").scrollIntoView({ behavior: "smooth", block: "start" });
      } catch { Toast.show("Could not reach the assessment service.", "err"); }
      finally { btn.disabled = false; btn.innerHTML = `${icon("activity", "ic--sm")} Generate plan`; }
    });
  }
  function renderFitness(d) {
    $("#fit-macros").innerHTML = [
      ["Calories", fmt(d.calorie_target), "kcal"], ["Protein", fmt(d.macros.protein_g), "g"],
      ["Carbs", fmt(d.macros.carbs_g), "g"], ["Fat", fmt(d.macros.fat_g), "g"],
    ].map(([l, v, u]) => `<div class="stat"><span class="stat__label">${l}</span><span class="stat__value">${v}</span><span class="muted">${u}</span></div>`).join("");

    const mp = d.meal_plan || { days: [] };
    $("#fit-meal-cost").innerHTML = `<span class="badge ${mp.within_budget ? "badge--ok" : "badge--warn"}">${icon("check", "ic--sm")} PKR ${fmt(mp.daily_cost_pkr)} / day · ${mp.within_budget ? "within budget" : "over budget"}</span>`;
    $("#fit-meal-tabs").innerHTML = mp.days.map((day, i) => `<button class="chip ${i === 0 ? "is-active" : ""}" data-i="${i}">${esc(day.day)}</button>`).join("");
    const paintDay = (i) => {
      const day = mp.days[i]; if (!day) return;
      $("#fit-meal-body").innerHTML = day.meals.map(([slot, txt]) =>
        `<div class="kv"><dt>${esc(slot)}</dt><dd style="font-weight:500;text-align:right;max-width:60%">${esc(txt)}</dd></div>`).join("");
    };
    $$("#fit-meal-tabs .chip").forEach(c => c.addEventListener("click", () => {
      $$("#fit-meal-tabs .chip").forEach(x => x.classList.remove("is-active"));
      c.classList.add("is-active"); paintDay(Number(c.dataset.i));
    }));
    paintDay(0);

    const wk = d.workout || { days: [] };
    $("#fit-workout-split").textContent = wk.split || "";
    $("#fit-workout").innerHTML = wk.days.map(day => `
      <div class="card" style="box-shadow:none">
        <div class="card__title" style="font-size:.95rem;margin-bottom:10px">${esc(day.day)}</div>
        <ul class="list-plain">${day.items.map(it => `<li>${icon("check", "ic--sm")}<span>${esc(it)}</span></li>`).join("")}</ul>
      </div>`).join("");
  }

  // ------------------------------------------------------- page: nutrition ---
  async function pageNutrition() {
    const log = $("#chat-log"), form = $("#chat-form"), input = $("#chat-input");
    const add = (role, content) => {
      const m = el(`<div class="msg msg--${role === "user" ? "user" : "bot"}">
        <div class="msg__meta">${role === "user" ? "You" : "HealthHub"}</div><div>${esc(content).replace(/\n/g, "<br>")}</div></div>`);
      log.appendChild(m); log.scrollTop = log.scrollHeight; return m;
    };
    const greet = () => add("bot", Auth.name()
      ? "Hi! Ask me about calories, Pakistani foods, swaps or grocery prices."
      : "Hi! Ask away — create a profile on the Profile page and I can tailor answers to your targets.");
    const { data, live } = await api.get(withUser("/nutrition/history"));
    banner(live, "#chat-banner");
    const msgs = data.messages || [];
    if (msgs.length) msgs.forEach(m => add(m.role, m.content)); else greet();

    $$("#chat-suggest .chip").forEach(c => c.addEventListener("click", () => { input.value = c.textContent; input.focus(); }));

    $("#chat-clear")?.addEventListener("click", async () => {
      if (!confirm("Clear this conversation? This can't be undone.")) return;
      try { await api.post(withUser("/nutrition/clear"), {}); } catch { /* offline: clear locally anyway */ }
      log.innerHTML = "";
      greet();
      Toast.show("Chat cleared.", "ok");
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const q = input.value.trim(); if (!q) return;
      add("user", q); input.value = "";
      const typing = el(`<div class="msg msg--bot"><span class="typing"><i></i><i></i><i></i></span></div>`);
      log.appendChild(typing); log.scrollTop = log.scrollHeight;
      try {
        const { data } = await api.post("/nutrition/chat", { message: q, user: Auth.name() });
        typing.remove();
        const bubble = add("bot", "");
        const words = (data.content || "").split(" ");
        let i = 0;
        const tick = () => {
          bubble.lastElementChild.innerHTML = esc(words.slice(0, ++i).join(" "));
          log.scrollTop = log.scrollHeight;
          if (i < words.length) setTimeout(tick, 24);
        };
        tick();
      } catch { typing.remove(); add("bot", "Sorry — I couldn't answer just now."); }
    });
  }

  // ------------------------------------------------------------ page: skin ---
  async function pageSkin() {
    const drop = $("#skin-drop"), file = $("#skin-file"), result = $("#skin-result");
    const pick = () => file.click();
    drop.addEventListener("click", pick);
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("is-drag"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("is-drag"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("is-drag"); if (e.dataTransfer.files[0]) handle(e.dataTransfer.files[0]); });
    file.addEventListener("change", () => file.files[0] && handle(file.files[0]));

    async function handle(f) {
      if (!f.type.startsWith("image/")) return Toast.show("Please choose an image file.", "err");
      const url = URL.createObjectURL(f);
      $("#skin-preview").src = url; $("#skin-preview-wrap").classList.remove("hidden");
      result.innerHTML = skeletonCard();
      const fd = new FormData(); fd.append("image", f);
      try {
        const { data, live } = await api.upload("/skin/analyze", fd);
        renderSkin(data); banner(live, "#skin-banner");
      } catch { result.innerHTML = `<div class="empty">${icon("alert")}<p>Analysis failed. Try another photo.</p></div>`; }
    }
    function renderSkin(d) {
      const sevClass = { clear: "badge--ok", mild: "badge--warn", moderate: "badge--warn", severe: "badge--danger" };
      const rs = (v) => (v == null || isNaN(v)) ? "—" : "Rs " + fmt(v);
      const routine = d.routine || [];
      const supps = d.supplements || [];
      const diet = d.diet || [];

      const routineCard = routine.length ? `
        <div class="card mt-3">
          <div class="card__head">
            <span class="card__title">Recommended routine</span>
            <span class="badge badge--primary">${icon("sparkles", "ic--sm")} ${esc((d.care_level || "basic").toUpperCase())} care</span>
          </div>
          <div class="table-wrap mt-2"><table class="table">
            <thead><tr><th>Step</th><th>Product</th><th>Price</th><th>Why</th></tr></thead>
            <tbody>${routine.map(r => `
              <tr>
                <td><b>${esc(r.step)}</b></td>
                <td>
                  ${esc(r.product)}
                  ${(r.alternatives && r.alternatives.length) ? `<div class="faint" style="font-size:.8rem;margin-top:2px">alt: ${r.alternatives.map(a => esc(a.name) + (a.price_pkr ? ` (${rs(a.price_pkr)})` : "")).join(" · ")}</div>` : ""}
                  ${r.where_to_buy ? `<div class="faint" style="font-size:.78rem">buy: ${esc(r.where_to_buy)}</div>` : ""}
                </td>
                <td style="white-space:nowrap">${rs(r.price_pkr)}</td>
                <td class="muted">${esc(r.why || "")}</td>
              </tr>`).join("")}</tbody>
          </table></div>
          ${d.brands_note ? `<p class="faint" style="font-size:.8rem;margin-top:10px">${esc(d.brands_note)}</p>` : ""}
        </div>` : "";

      const suppCard = supps.length ? `
        <div class="card">
          <div class="card__title mb-2">${icon("pill", "ic--sm")} Supplements</div>
          <div class="stack-sm">${supps.map(s => `
            <div class="row-between" style="align-items:flex-start;border-bottom:1px solid var(--border);padding-bottom:10px">
              <div>
                <div style="font-weight:600">${esc(s.name)}</div>
                <div class="faint" style="font-size:.82rem">${[s.dosage, s.timing].filter(Boolean).map(esc).join(" · ")}${s.why ? " — " + esc(s.why) : ""}</div>
              </div>
              <span class="badge">${rs(s.price_pkr)}</span>
            </div>`).join("")}</div>
          ${d.supplement_note ? `<p class="faint" style="font-size:.8rem;margin-top:10px">⚠️ ${esc(d.supplement_note)}</p>` : ""}
        </div>` : `<div class="card"><div class="card__title mb-2">${icon("pill", "ic--sm")} Supplements</div><p class="muted">None needed — skin looks clear enough that food covers it.</p></div>`;

      const dietCard = `
        <div class="card">
          <div class="card__title mb-2">${icon("leaf", "ic--sm")} Diet changes</div>
          <ul class="list-plain">${(diet.length ? diet : ["Balanced plate, seasonal fruit & veg, 2.5–3 L water/day"])
            .map(t => `<li>${icon("check", "ic--sm")}<span>${esc(t)}</span></li>`).join("")}</ul>
        </div>`;

      result.innerHTML = `
        <div class="grid cols-2">
          <div class="card">
            <div class="card__title">Skin type</div>
            <div class="row" style="gap:20px;margin-top:10px">
              <div class="ring" style="--val:${Number(d.confidence) || 0}">
                <div class="ring__label"><b>${fmt(d.confidence)}%</b><span>confidence</span></div>
              </div>
              <div>
                <div style="font-size:1.5rem;font-weight:800">${esc(d.skin_type)}</div>
                <div class="muted" style="font-size:.9rem">Gemini Vision read</div>
              </div>
            </div>
          </div>
          <div class="card">
            <div class="card__title">Why</div>
            <p class="muted" style="margin-top:8px">${esc(d.reasoning)}</p>
            <div class="row" style="margin-top:14px">
              ${(d.concerns || []).map(c => `<span class="badge ${sevClass[c.severity] || ""}">${esc(c.name)} · ${esc(c.severity)}</span>`).join("")}
            </div>
          </div>
        </div>
        ${routineCard}
        <div class="grid cols-2 mt-3">${suppCard}${dietCard}</div>`;
    }
  }

  // -------------------------------------------------------- page: progress ---
  let _trendChart = null;
  let _trendEntries = [];

  async function pageProgress() {
    if (!Auth.name()) return renderNoProfile();
    const { data, live } = await api.get(withUser("/progress/entries"));
    banner(live, "#prog-banner");
    paintProgress(data.entries || []);
    document.addEventListener("hh:theme", () => renderTrend(_trendEntries));

    $("#prog-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const p = Object.fromEntries(new FormData(e.target).entries());
      Object.keys(p).forEach(k => p[k] = Number(p[k]));
      p.user = Auth.name();
      try {
        const { data } = await api.post("/progress/entries", p);
        paintProgress(data.entries || []);
        e.target.reset(); Toast.show("Entry saved.", "ok");
      } catch { Toast.show("Could not save entry.", "err"); }
    });
  }
  function paintProgress(entries) {
    const first = entries[0], last = entries[entries.length - 1] || {};
    const dW = first && last ? (last.weight_kg - first.weight_kg) : 0;
    const dWa = first && last ? (last.waist_cm - first.waist_cm) : 0;
    $("#prog-stats").innerHTML = [
      ["Current weight", `${fmt(last.weight_kg, 1)} kg`, ""],
      ["Weight change", `${dW > 0 ? "+" : ""}${fmt(dW, 1)} kg`, dW <= 0 ? "up" : "down"],
      ["Waist", `${fmt(last.waist_cm, 1)} cm`, ""],
      ["Waist change", `${dWa > 0 ? "+" : ""}${fmt(dWa, 1)} cm`, dWa <= 0 ? "up" : "down"],
    ].map(([l, v, dir]) => `<div class="stat"><span class="stat__label">${l}</span><span class="stat__value">${v}</span>${dir ? `<span class="stat__delta ${dir}">${icon("trending", "ic--sm")} ${dir === "up" ? "improving" : "watch"}</span>` : ""}</div>`).join("");

    renderTrend(entries);

    $("#prog-table").innerHTML = `<table class="table"><thead><tr><th>Date</th><th>Weight</th><th>Waist</th><th>Chest</th></tr></thead><tbody>
      ${[...entries].reverse().map(e => `<tr><td>${esc(e.date)}</td><td>${fmt(e.weight_kg, 1)} kg</td><td>${fmt(e.waist_cm, 1)} cm</td><td>${fmt(e.chest_cm, 1)} cm</td></tr>`).join("")}
    </tbody></table>`;

    $("#prog-timeline").innerHTML = [...entries].reverse().map(e => `
      <div class="timeline__item">
        <div class="timeline__date">${esc(e.date)}</div>
        <div>Weight <b>${fmt(e.weight_kg, 1)} kg</b> · waist <b>${fmt(e.waist_cm, 1)} cm</b></div>
      </div>`).join("");
  }
  // Chart.js line chart (weight + waist + chest), theme-aware, with an inline
  // SVG sparkline fallback when Chart.js is unavailable or there's <2 points.
  function renderTrend(entries) {
    _trendEntries = entries || [];
    const host = $("#prog-spark");
    if (!host) return;

    if (!window.Chart || _trendEntries.length < 2) {
      if (_trendChart) { _trendChart.destroy(); _trendChart = null; }
      host.innerHTML = sparkline(_trendEntries.map(e => e.weight_kg));
      return;
    }

    const css = getComputedStyle(document.documentElement);
    const tok = (n, f) => (css.getPropertyValue(n).trim() || f);
    const primary = tok("--primary", "#1e3a8a");
    const accent  = tok("--accent", "#10b981");
    const amber    = "#f59e0b";
    const grid    = tok("--border", "#e2e8f0");
    const label   = tok("--text-muted", "#64748b");

    host.innerHTML = '<canvas id="prog-canvas"></canvas>';
    const ctx = $("#prog-canvas").getContext("2d");
    if (_trendChart) _trendChart.destroy();

    const ds = (name, key, color, hidden) => ({
      label: name, data: _trendEntries.map(e => e[key] ?? null),
      borderColor: color, backgroundColor: color + "22",
      borderWidth: 2.5, tension: 0.35, pointRadius: 3, pointHoverRadius: 5,
      pointBackgroundColor: color, spanGaps: true, hidden: !!hidden,
    });

    _trendChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: _trendEntries.map(e => e.date),
        datasets: [
          ds("Weight (kg)", "weight_kg", primary),
          ds("Waist (cm)", "waist_cm", accent),
          ds("Chest (cm)", "chest_cm", amber, true),
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { grid: { color: grid, drawBorder: false }, ticks: { color: label, maxRotation: 0, autoSkipPadding: 16 } },
          y: { grid: { color: grid, drawBorder: false }, ticks: { color: label } },
        },
        plugins: {
          legend: { labels: { color: label, usePointStyle: true, boxWidth: 8, padding: 16 } },
          tooltip: { padding: 10, cornerRadius: 8, usePointStyle: true },
        },
      },
    });
  }

  function sparkline(vals) {
    if (vals.length < 2) return "";
    const w = 640, h = 120, pad = 10;
    const min = Math.min(...vals), max = Math.max(...vals), span = (max - min) || 1;
    const pts = vals.map((v, i) => [pad + i * (w - 2 * pad) / (vals.length - 1), h - pad - (v - min) / span * (h - 2 * pad)]);
    const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
    const area = `${d} L ${pts[pts.length - 1][0].toFixed(1)} ${h} L ${pts[0][0].toFixed(1)} ${h} Z`;
    return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="width:100%;height:120px">
      <defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="var(--accent)" stop-opacity=".28"/><stop offset="1" stop-color="var(--accent)" stop-opacity="0"/>
      </linearGradient></defs>
      <path d="${area}" fill="url(#sg)"/>
      <path d="${d}" fill="none" stroke="var(--primary)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
      ${pts.map(p => `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3" fill="var(--primary)"/>`).join("")}
    </svg>`;
  }

  // --------------------------------------------------------- page: profile ---
  async function pageProfile() {
    const form = $("#profile-form");
    const heading = $("#profile-heading");
    const sub = $("#profile-sub");
    form.reset();                                   // always start from a blank form

    const active = Auth.name();
    const cached = Auth.user();                     // {name,email} carried from landing sign-up
    let mode = "create";

    const prefill = (obj) => {
      for (const [k, v] of Object.entries(obj || {})) {
        if (k === "exists") continue;
        const f = form.elements[k];
        if (f && v != null && v !== "") f.value = v;
      }
    };

    if (active) {
      const { data, live } = await api.get(withUser("/profile"));
      banner(live, "#profile-banner");
      if (data && data.exists !== false && data.name && data.height_cm) {
        mode = "edit";
        prefill(data);
      } else {
        // signed up (name + maybe email) but body stats not saved yet →
        // stay in CREATE mode, prefilled. Never wipe on a background miss.
        mode = "create";
        prefill(cached || { name: active });
        if (data && data.name) prefill({ email: data.email });
      }
    }

    if (heading) heading.textContent = mode === "edit" ? "Your profile" : "Create your profile";
    if (sub) sub.textContent = mode === "edit"
      ? "Update your details — every target recalculates from these."
      : "Fill this in to unlock your dashboard, plan and progress.";
    const lbl = $("#profile-submit-label");
    if (lbl) lbl.textContent = mode === "edit" ? "Save changes" : "Create profile";
    if (mode === "create") { const nm = form.elements["name"]; if (nm) nm.focus(); }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const p = Object.fromEntries(new FormData(form).entries());
      if (!p.name || !p.name.trim()) { Toast.show("Enter a name for your profile.", "err"); return; }
      ["age", "height_cm", "weight_kg", "budget_pkr"].forEach(k => { if (p[k]) p[k] = Number(p[k]); });
      try {
        const wasCreate = mode !== "edit";
        const { data } = await api.post("/profile", p);
        const savedName = (data && data.profile && data.profile.name) || p.name.trim();
        Auth.set(savedName, { name: savedName, email: p.email || "" });   // signed in
        _setIdentity(savedName);                    // update sidebar / topbar immediately
        mode = "edit";
        if (heading) heading.textContent = "Your profile";
        if (sub) sub.textContent = "Update your details — every target recalculates from these.";
        if (lbl) lbl.textContent = "Save changes";
        Toast.show(wasCreate ? "Profile created." : "Profile updated.", "ok");
      } catch { Toast.show("Could not save profile.", "err"); }
    });
  }

  // ------------------------------------------------------------- fragments ---
  function skeletonCard() { return `<div class="card"><div class="skeleton line"></div><div class="skeleton line sm"></div><div class="skeleton block mt-2"></div></div>`; }
  // shown on data pages when no profile is active
  function renderNoProfile() {
    const host = $(".app-main .page") || $(".app-main");
    if (!host) return;
    const started = !!Auth.name();
    host.innerHTML = `<div class="empty" style="padding:72px 20px">
      ${icon("user")}
      <h2 style="margin:14px 0 6px">${started ? "Finish your profile" : "No profile yet"}</h2>
      <p class="muted">${started
        ? "Add your body stats on the profile page to unlock your dashboard, plan and progress."
        : "Create a profile to unlock your dashboard, plan and progress."}</p>
      <a class="btn btn-primary mt-3" href="${LINK}profile.html">${started ? "Go to profile" : "Create profile"}</a>
    </div>`;
  }
  function banner(live, sel = "#page-banner") {
    const n = $(sel); if (!n) return;
    if (live) { n.classList.add("hidden"); return; }
    n.classList.remove("hidden");
    n.innerHTML = `<div class="badge badge--info">${icon("alert", "ic--sm")} Demo data — backend API not connected</div>`;
  }

  // --------------------------------------------------------- landing: auth ---
  // The landing page gates entry: no stored profile -> the "Create profile"
  // form; a stored (and still-valid) profile -> the "Welcome back" panel.
  // confirmed: undefined = backend not checked yet, true/false = whether the
  // backend has a complete profile for the stored name.
  function renderLandingAuth(confirmed) {
    const signup = $("#auth-signup"), welcome = $("#auth-welcome"), navAuth = $("#nav-auth");
    if (!signup || !welcome) return;
    const name = Auth.name();
    signup.hidden = !!name;
    welcome.hidden = !name;
    if (name) {
      const nm = (Auth.user() || {}).name || name;
      const uEl = $("#auth-user-name"); if (uEl) uEl.textContent = nm;
      const av = $("#auth-avatar"); if (av) av.textContent = (nm.trim().charAt(0).toUpperCase() || "?");
      const hint = $("#auth-hint");
      if (hint) {
        if (confirmed === false) { hint.hidden = false; hint.textContent = "Finish your body stats on the profile page to unlock the dashboard."; }
        else { hint.hidden = true; }
      }
    }
    if (navAuth) {
      navAuth.innerHTML = name
        ? `<a class="btn btn-primary" href="pages/dashboard.html">Open app</a>`
        : `<a class="btn btn-outline" href="#auth-panel">Create profile</a>`;
    }
  }

  function initLandingAuth() {
    renderLandingAuth();

    // if a name is stored, sync display name / email from the backend when it
    // still knows this profile (never auto-wipe — a stale name just prefills).
    if (Auth.name()) {
      api.get(withUser("/profile")).then(({ data }) => {
        const complete = !!(data && data.name && data.height_cm && data.weight_kg);
        if (data && data.name) Auth.set(data.name, { email: data.email });
        renderLandingAuth(complete);
      }).catch(() => renderLandingAuth());
    }

    $("#auth-form")?.addEventListener("submit", (e) => {
      e.preventDefault();
      const name = ($("#auth-name")?.value || "").trim();
      const email = ($("#auth-email")?.value || "").trim();
      if (!name) { $("#auth-name")?.focus(); Toast.show("Enter your name.", "err"); return; }
      if (!email || !/^\S+@\S+\.\S+$/.test(email)) { $("#auth-email")?.focus(); Toast.show("Enter a valid email.", "err"); return; }
      // Only name + email here — the profile record is created when the full
      // form (body stats) is saved, so nothing is stored with placeholder body
      // values. Carry name/email over and go finish the profile.
      Auth.set(name, { name, email, created_at: new Date().toISOString() });
      location.href = "pages/profile.html";
    });

    $("#auth-logout")?.addEventListener("click", () => {
      Auth.signOut();
      $("#auth-form")?.reset();
      renderLandingAuth();
      Toast.show("Signed out.", "ok");
      $("#auth-name")?.focus();
    });
  }

  // ----------------------------------------------------------------- boot ---
  const PAGES = { dashboard: pageDashboard, fitness: pageFitness, nutrition: pageNutrition, skin: pageSkin, progress: pageProgress, profile: pageProfile };
  document.addEventListener("DOMContentLoaded", () => {
    const page = document.body.dataset.page;
    // landing page: no shell — just the theme button + the auth gate
    if (!page || page === "landing") {
      $("#theme-btn")?.addEventListener("click", () => {
        Theme.toggle();
        $("#theme-btn").innerHTML = icon(Theme.resolved() === "dark" ? "sun" : "moon");
      });
      if ($("#theme-btn")) $("#theme-btn").innerHTML = icon(Theme.resolved() === "dark" ? "sun" : "moon");
      initLandingAuth();
      return;
    }
    mountShell(page);
    (PAGES[page] || (() => {}))().catch(err => { console.error(err); Toast.show("Something went wrong loading this page.", "err"); });
  });
})();
