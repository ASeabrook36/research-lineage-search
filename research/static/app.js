const $ = (selector) => document.querySelector(selector);
const text = (tag, content, className) => {
  const element = document.createElement(tag);
  if (content !== undefined) element.textContent = content;
  if (className) element.className = className;
  return element;
};
const countryNames = {JP: "Japan", US: "United States", NL: "Netherlands", DE: "Germany", GB: "United Kingdom", BE: "Belgium", KR: "South Korea", Unknown: "Unknown"};
const countryLabel = (value) => countryNames[value] || value;
const coverageLabel = (value) => ({full_text: "Full text available", abstract: "Abstract only", metadata: "Metadata only"}[value] || value);
let filterData;
let searchSequence = 0;
let detailSequence = 0;
let lineageLoaded = false;

async function api(path) {
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return data;
}

function link(label, url) {
  const anchor = text("a", label);
  try {
    const parsed = new URL(url);
    if (!["https:", "http:"].includes(parsed.protocol)) throw new Error("Unsupported link");
    anchor.href = parsed.href;
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
  } catch {
    anchor.removeAttribute("href");
    anchor.textContent += " (invalid source URL)";
  }
  return anchor;
}

function showError(target, message) {
  target.replaceChildren(text("div", message, "error"));
}

function paperButton(id, title, className = "text-button") {
  const button = text("button", title, className);
  button.type = "button";
  button.addEventListener("click", () => openPaper(id));
  return button;
}

function resultCard(result, rank) {
  const card = text("article", undefined, "card");
  const top = text("div", undefined, "card-top");
  if (rank) top.append(text("span", String(rank).padStart(2, "0")));
  top.append(text("span", String(result.year || "Unknown year")), text("span", result.countries.map(countryLabel).join(" · ")), text("span", coverageLabel(result.coverage), "badge coverage"));
  card.append(top, paperButton(result.id, result.title, "paper-title"));
  for (const passage of result.passages) {
    card.append(text("div", `${passage.section === "abstract" ? "Abstract" : `Full text · page ${passage.page}`} · passage ${passage.id.slice(0, 8)}`, "passage-label"));
    card.append(text("blockquote", passage.text));
  }
  const bottom = text("div", undefined, "card-bottom");
  bottom.append(paperButton(result.id, "Inspect paper & connections →"), text("span", `Similarity ${Number(result.score).toFixed(3)} · not confidence`, "muted"));
  card.append(bottom);
  return card;
}

function renderOptions(category, search = "") {
  const select = $(`#filter-${category}`);
  const selected = new Set(Array.from(select.selectedOptions).map(option => option.value));
  select.replaceChildren();
  for (const option of filterData[category]) {
    const label = category === "country" ? countryLabel(option.id) : option.name;
    if (search && !label.toLowerCase().includes(search.toLowerCase()) && !selected.has(option.id)) continue;
    const element = text("option", label + (option.count !== undefined ? ` (${option.count})` : ""));
    element.value = option.id;
    element.selected = selected.has(option.id);
    select.append(element);
  }
}

function createFilters(data) {
  filterData = data;
  for (const [category, label] of Object.entries({country: "Affiliation country", field: "Field or topic", institution: "Institution", author: "Author"})) {
    const fieldset = document.createElement("fieldset");
    fieldset.append(text("legend", label));
    const input = document.createElement("input");
    input.type = "search";
    input.className = "filter-search";
    input.placeholder = `Find ${label.toLowerCase()}...`;
    input.setAttribute("aria-label", `Find ${label.toLowerCase()} options`);
    input.addEventListener("input", () => renderOptions(category, input.value));
    const select = document.createElement("select");
    select.id = `filter-${category}`;
    select.name = category;
    select.multiple = true;
    select.size = 4;
    select.setAttribute("aria-label", label);
    fieldset.append(input, select, text("div", "⌘/Ctrl-click to select multiple; Apply filters to search.", "filter-hint"));
    $("#filters").append(fieldset);
    renderOptions(category);
  }
}

async function runSearch(event) {
  if (event) event.preventDefault();
  const sequence = ++searchSequence;
  const params = new URLSearchParams();
  for (const [key, value] of new FormData($("#search-form"))) {
    if (String(value).trim()) params.append(key, value);
  }
  $("#search-button").disabled = true;
  $("#notice").replaceChildren(text("p", "Searching local Cosmos DB vectors...", "muted small"));
  try {
    const data = await api(`/api/search?${params}`);
    if (sequence !== searchSequence) return;
    $("#result-heading").replaceChildren(text("h2", `${data.results.length} research matches`), text("span", `${data.elapsed_ms} ms · Cosmos vector search`));
    $("#notice").replaceChildren(text("p", "Ranked by semantic relevance, not historical importance. Passages are source text, not generated answers.", "small muted"));
    $("#results").replaceChildren(...data.results.map((result, index) => resultCard(result, index + 1)));
    if (!data.results.length) $("#results").append(text("div", "No passages match these filters. Try a broader topic or clear some filters. Metadata-only papers do not have searchable passages.", "empty"));
  } catch (error) {
    if (sequence === searchSequence) showError($("#notice"), error.message);
  } finally {
    if (sequence === searchSequence) $("#search-button").disabled = false;
  }
}

async function openPaper(id) {
  const sequence = ++detailSequence;
  if (!$("#paper-dialog").open) $("#paper-dialog").showModal();
  $("#paper-content").replaceChildren(text("p", "Loading paper and evidence...", "muted"));
  try {
    const [detail, connections] = await Promise.all([api(`/api/papers/${encodeURIComponent(id)}`), api(`/api/papers/${encodeURIComponent(id)}/connections`)]);
    if (sequence !== detailSequence) return;
    const paper = detail.paper;
    const container = $("#paper-content");
    container.replaceChildren(text("h2", paper.title), text("div", `${paper.year} · ${paper.countries.map(countryLabel).join(" · ")} · ${coverageLabel(paper.coverage)}`, "small muted"));
    container.append(text("p", paper.authors.map(a => a.name).join("; ") || "Author metadata unavailable", "authors"));
    const metadata = text("div", undefined, "metadata-grid");
    for (const [label, value] of [
      ["Institutions", paper.institutions.map(i => `${i.name} (${countryLabel(i.country)})`).join("; ") || "Unknown"],
      ["Fields / topics", paper.fields.join("; ")],
      ["Indexed coverage", `${paper.passage_count} of ${paper.available_passages} extracted passages. ${paper.coverage === "full_text" ? `Text was extracted from ${paper.extracted_page_count ?? "some"} of ${paper.pdf_page_count ?? "unknown"} PDF pages; the passage cap may omit later sections.` : "Full text was not indexed."}`],
      ["Access / license metadata", paper.licenses.join("; ") || "No license supplied by metadata provider; accessibility is not a redistribution license."],
    ]) {
      const item = text("div");
      item.append(text("strong", label), text("span", value));
      metadata.append(item);
    }
    container.append(metadata);
    const sources = text("p", undefined, "small");
    sources.append(link("Publisher / source", paper.source_url));
    for (const source of paper.sources) sources.append(document.createTextNode(" · "), link(source.provider, source.url));
    if (paper.full_text_url) sources.append(document.createTextNode(" · "), link("Full-text source", paper.full_text_url));
    container.append(sources);
    if (paper.abstract) container.append(text("h3", "Abstract"), text("p", paper.abstract, "small"));
    container.append(text("h3", `Indexed source passages (${detail.passages.length})`));
    const passages = text("div", undefined, "passages");
    for (const passage of detail.passages) {
      passages.append(text("div", `${passage.section === "abstract" ? "Abstract" : `Page ${passage.page}`} · ${passage.id}`, "passage-label"), text("blockquote", passage.text));
    }
    if (!detail.passages.length) passages.append(text("p", "This is a metadata-only record; no passages were fabricated.", "muted"));
    container.append(passages, text("h3", `Recorded citation connections (${connections.citations.length})`), text("p", connections.note, "small muted"));
    for (const edge of connections.citations) {
      const row = text("div", undefined, "connection");
      row.append(text("span", edge.from_title), text("span", " → ", "arrow"));
      row.append(edge.target_in_corpus ? paperButton(edge.to_id, edge.to_title) : link(`${edge.to_title} (outside collection)`, edge.to_id));
      row.append(document.createTextNode(" · "), link("Source record", edge.evidence_url));
      container.append(row);
    }
    if (!connections.citations.length) container.append(text("p", "No citation connections recorded in this prototype. Missing metadata does not mean there was no connection.", "small muted"));
    container.append(text("h3", "Semantically related research"), text("p", "These neighbors are based on a representative indexed passage, not proof of citation, collaboration, or influence.", "callout"));
    for (const related of connections.related_research) container.append(resultCard(related, ""));
  } catch (error) {
    if (sequence === detailSequence) showError($("#paper-content"), error.message);
  }
}

async function loadLineage() {
  if (lineageLoaded) return;
  try {
    const data = await api("/api/lineage");
    $("#history-note").textContent = data.source_note;
    $("#timeline").replaceChildren();
    for (const milestone of data.milestones) {
      const card = text("article", undefined, "milestone");
      card.append(text("div", `${milestone.year} · ${milestone.countries.map(countryLabel).join(" · ")}`, "eyebrow"), text("h3", milestone.title), text("p", milestone.description));
      const edges = data.historical_edges.filter(edge => edge.to_id === milestone.id);
      for (const edge of edges) {
        const details = document.createElement("details");
        details.append(text("summary", `${edge.from_title} → ${edge.to_title} · ${edge.relationship.replaceAll("_", " ")}`), text("p", edge.explanation), link("Inspect historical source (ASML retrospective)", edge.evidence_url));
        card.append(details);
      }
      if (!edges.length) card.append(link("Historical source (ASML retrospective)", "https://www.asml.com/en/company/stories/2022/making-euv-lab-to-fab"));
      const discover = text("button", "Find related research →", "text-button");
      discover.type = "button";
      discover.addEventListener("click", () => {
        switchTab(false);
        resetFilters();
        $("#query").value = `${milestone.title} EUV lithography`;
        runSearch();
      });
      card.append(text("p", "Related papers are retrieved semantically; this does not attach them as causal evidence.", "small muted"), discover);
      $("#timeline").append(card);
    }
    $("#citation-list").replaceChildren();
    for (const edge of data.citations) {
      const row = text("div", undefined, "citation-row");
      row.append(paperButton(edge.from_id, edge.from_title), text("span", "→", "arrow"), paperButton(edge.to_id, edge.to_title), document.createTextNode(" · "), link("Metadata evidence", edge.evidence_url));
      $("#citation-list").append(row);
    }
    $("#citation-list").append(text("p", `${data.citations.length} links connect papers in this collection; ${data.external_citation_count} additional references point outside it. This is not a complete genealogy.`, "small muted"));
    lineageLoaded = true;
  } catch (error) {
    showError($("#timeline"), error.message);
  }
}

function resetFilters() {
  for (const select of document.querySelectorAll("#filters select")) {
    for (const option of select.options) option.selected = false;
  }
  for (const input of document.querySelectorAll("aside input")) input.value = "";
  if (filterData) for (const category of Object.keys(filterData)) renderOptions(category);
}

function switchTab(lineage) {
  $("#discover").hidden = lineage;
  $("#lineage").hidden = !lineage;
  $("#search-tab").classList.toggle("active", !lineage);
  $("#lineage-tab").classList.toggle("active", lineage);
  $("#search-tab").setAttribute("aria-selected", String(!lineage));
  $("#lineage-tab").setAttribute("aria-selected", String(lineage));
  if (lineage) loadLineage();
}

$("#search-form").addEventListener("submit", runSearch);
$("#search-tab").addEventListener("click", () => switchTab(false));
$("#lineage-tab").addEventListener("click", () => switchTab(true));
$("#reset-filters").addEventListener("click", resetFilters);
$("#close-paper").addEventListener("click", () => { detailSequence++; $("#paper-dialog").close(); });
for (const button of document.querySelectorAll("[data-query]")) {
  button.addEventListener("click", () => { $("#query").value = button.dataset.query; runSearch(); });
}

async function initialize() {
  try {
    const [status, filters] = await Promise.all([api("/api/status"), api("/api/filters")]);
    $("#stats").replaceChildren();
    const stats = [
      [status.counts.papers, "research records"], [status.counts.passages, "searchable passages"],
      [status.ingestion?.full_text_papers || 0, "full-text sources"],
      [status.ingestion?.in_corpus_citations || 0, "in-collection citations"],
    ];
    for (const [count, label] of stats) {
      const span = text("span");
      span.append(text("b", Number(count).toLocaleString()), document.createTextNode(` ${label}`));
      $("#stats").append(span);
    }
    createFilters(filters);
    if (!status.ready) throw new Error("Local data or embedding model is missing. Run collection and ingestion using the README commands.");
  } catch (error) {
    showError($("#notice"), error.message);
  }
}
initialize();
