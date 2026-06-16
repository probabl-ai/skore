function skoreInitSummary(containerId) {
    const wrapper = document.getElementById(containerId + "-wrapper");
    const host = document.getElementById(containerId);
    if (!wrapper || !host) {
        return;
    }

    const template = document.getElementById(containerId + "-template");
    if (!template) {
        return;
    }

    const shadowRoot = host.attachShadow({ mode: "closed" });
    shadowRoot.appendChild(template.content.cloneNode(true));

    const container = shadowRoot.querySelector(".container");
    if (container) {
        applyThemeToReportContainer(container, containerId + "-wrapper");
    }

    const table = shadowRoot.querySelector(".summary-table");
    if (!table) {
        return;
    }

    const tbody = table.querySelector("tbody");
    const dataRows = Array.from(tbody.querySelectorAll("tr"));

    const ctx = {
        shadowRoot,
        container,
        table,
        tbody,
        dataRows,
        queryBox: shadowRoot.querySelector(".skore-summary-query"),
        copyButton: shadowRoot.querySelector(".skore-summary-copy"),
        searchInput: shadowRoot.querySelector(".skore-summary-search-input"),
        dateStart: shadowRoot.querySelector(".skore-summary-date-start"),
        dateEnd: shadowRoot.querySelector(".skore-summary-date-end"),
        filterValues: shadowRoot.querySelectorAll(".skore-summary-filter-value"),
        groupN: shadowRoot.querySelector(".skore-summary-group-n"),
        groupUnit: shadowRoot.querySelector(".skore-summary-group-unit"),
        groupOptions: shadowRoot.querySelectorAll(".skore-summary-group-option"),
        plotArea: shadowRoot.querySelector(".summary-plot"),
        plotEmpty: shadowRoot.querySelector(".summary-plot-empty"),
        colorSelect: shadowRoot.querySelector(".skore-summary-color-metric"),
        trendArea: shadowRoot.querySelector(".summary-trend"),
        trendEmpty: shadowRoot.querySelector(".summary-trend-empty"),
        trendUndatedEmpty: shadowRoot.querySelector(".summary-trend-undated-empty"),
        trendMetricSelect: shadowRoot.querySelector(".skore-summary-trend-metric"),
        viewButtons: shadowRoot.querySelectorAll(".summary-view-btn"),
        selectedIds: new Set(),
        sortState: null,
        groupState: null,
        brushes: {},
        trendXBrush: null,
        trendYBrush: null,
        currentView: "table",
        queryIsEmpty: true,
        numericColumns: [],
        keyToIndex: {},
    };

    const filterToggle = shadowRoot.querySelector(".skore-summary-filter-toggle");
    const filterPanel = shadowRoot.querySelector(".skore-summary-filter-panel");
    const groupToggle = shadowRoot.querySelector(".skore-summary-groupby-toggle");
    const groupPanel = shadowRoot.querySelector(".skore-summary-groupby-panel");
    const columnsToggle = shadowRoot.querySelector(".skore-summary-columns-toggle");
    const columnsPanel = shadowRoot.querySelector(".skore-summary-columns-panel");

    const popovers = [
        [filterToggle, filterPanel],
        [groupToggle, groupPanel],
        [columnsToggle, columnsPanel],
    ];

    function setPanel(toggle, panel, open) {
        if (!toggle || !panel) {
            return;
        }
        if (open) {
            panel.removeAttribute("hidden");
            toggle.setAttribute("aria-expanded", "true");
        } else {
            panel.setAttribute("hidden", "");
            toggle.setAttribute("aria-expanded", "false");
        }
    }

    ctx.closeMenus = function closeMenus() {
        popovers.forEach(([toggle, panel]) => setPanel(toggle, panel, false));
    };

    function wireToggle(toggle, panel) {
        if (!toggle || !panel) {
            return;
        }
        toggle.addEventListener("click", (event) => {
            event.stopPropagation();
            const willOpen = panel.hasAttribute("hidden");
            ctx.closeMenus();
            setPanel(toggle, panel, willOpen);
        });
        panel.addEventListener("click", (event) => event.stopPropagation());
    }

    popovers.forEach(([toggle, panel]) => wireToggle(toggle, panel));
    shadowRoot.addEventListener("click", ctx.closeMenus);

    setupTable(ctx);
    setupPlot(ctx);
    setupTrend(ctx);

    ctx.clearBrushes = function clearBrushes() {
        for (const key in ctx.brushes) {
            delete ctx.brushes[key];
        }
        ctx.trendXBrush = null;
        ctx.trendYBrush = null;
    };

    function emptyMessage() {
        if (ctx.currentView === "plot") {
            return "Brush an axis to select reports.";
        }
        if (ctx.currentView === "trend") {
            return "Drag the range cursors to select reports.";
        }
        return ctx.queryBox ? ctx.queryBox.dataset.empty : "";
    }

    function setQuery(text, isEmpty) {
        if (!ctx.queryBox) {
            return;
        }
        ctx.queryIsEmpty = isEmpty;
        ctx.queryBox.textContent = isEmpty ? emptyMessage() : text;
    }

    ctx.updateQuery = function updateQuery() {
        const ids = [];
        dataRows.forEach((row) => {
            const checkbox = row.querySelector(".skore-summary-row");
            if (checkbox && ctx.selectedIds.has(checkbox.dataset.id)) {
                ids.push(checkbox.dataset.id);
            }
        });
        if (ids.length === 0) {
            setQuery("", true);
            return;
        }
        const formatted = ids.map((id) => "'" + id + "'").join(", ");
        setQuery('"id in [' + formatted + ']"', false);
    };

    ctx.syncCheckboxes = function syncCheckboxes() {
        shadowRoot.querySelectorAll(".skore-summary-row").forEach((checkbox) => {
            checkbox.checked = ctx.selectedIds.has(checkbox.dataset.id);
        });
    };

    let lastClickedRow = null;

    shadowRoot.querySelectorAll(".skore-summary-row").forEach((checkbox) => {
        checkbox.addEventListener("mousedown", (event) => {
            if (event.shiftKey) {
                event.preventDefault();
            }
        });
        checkbox.addEventListener("click", (event) => {
            if (event.shiftKey && lastClickedRow && lastClickedRow !== checkbox) {
                const visibleCheckboxes = Array.from(
                    tbody.querySelectorAll(".skore-summary-row")
                ).filter((cb) => {
                    const tr = cb.closest("tr");
                    return tr && tr.style.display !== "none";
                });
                const current = visibleCheckboxes.indexOf(checkbox);
                const anchor = visibleCheckboxes.indexOf(lastClickedRow);
                if (current >= 0 && anchor >= 0) {
                    const lo = Math.min(current, anchor);
                    const hi = Math.max(current, anchor);
                    const target = checkbox.checked;
                    for (let i = lo; i <= hi; i++) {
                        visibleCheckboxes[i].checked = target;
                    }
                }
            }
            ctx.selectedIds.clear();
            shadowRoot.querySelectorAll(".skore-summary-row").forEach((cb) => {
                if (cb.checked) {
                    ctx.selectedIds.add(cb.dataset.id);
                }
            });
            ctx.clearBrushes();
            ctx.updateQuery();
            lastClickedRow = checkbox;
        });
    });

    if (ctx.copyButton && ctx.queryBox) {
        ctx.copyButton.addEventListener("click", () => {
            if (ctx.queryIsEmpty || !navigator.clipboard) {
                return;
            }
            navigator.clipboard.writeText(ctx.queryBox.textContent);
        });
    }

    const clearButton = shadowRoot.querySelector(".skore-summary-clear");
    if (clearButton) {
        clearButton.addEventListener("click", () => {
            ctx.selectedIds.clear();
            ctx.clearBrushes();
            ctx.syncCheckboxes();
            if (ctx.currentView === "plot") {
                ctx.renderPlot();
            } else if (ctx.currentView === "trend") {
                ctx.renderTrend();
            }
            ctx.updateQuery();
        });
    }

    function setView(view) {
        ctx.currentView = view;
        if (container) {
            container.dataset.view = view;
        }
        ctx.viewButtons.forEach((button) => {
            const active = button.dataset.view === view;
            button.classList.toggle("summary-view-btn--active", active);
            button.setAttribute("aria-pressed", active ? "true" : "false");
        });
        if (view === "plot") {
            ctx.renderPlot();
        } else if (view === "trend") {
            ctx.renderTrend();
        } else {
            ctx.syncCheckboxes();
        }
        ctx.updateQuery();
    }

    ctx.viewButtons.forEach((button) => {
        button.addEventListener("click", () => setView(button.dataset.view));
    });

    if (ctx.colorSelect) {
        ctx.colorSelect.addEventListener("change", () => {
            if (ctx.currentView === "plot") {
                ctx.renderPlot();
            }
        });
    }

    if (ctx.trendMetricSelect) {
        ctx.trendMetricSelect.addEventListener("change", () => {
            if (ctx.currentView !== "trend") {
                return;
            }
            ctx.trendYBrush = null;
            ctx.applyTrendSelection();
            ctx.syncCheckboxes();
            ctx.updateQuery();
            ctx.renderTrend();
        });
    }

    if (typeof ResizeObserver !== "undefined") {
        if (ctx.plotArea) {
            const observer = new ResizeObserver(() => {
                if (ctx.currentView === "plot") {
                    ctx.renderPlot();
                }
            });
            observer.observe(ctx.plotArea);
        }
        if (ctx.trendArea) {
            const observer = new ResizeObserver(() => {
                if (ctx.currentView === "trend") {
                    ctx.renderTrend();
                }
            });
            observer.observe(ctx.trendArea);
        }
    }

    function refreshPlotIfActive() {
        if (ctx.currentView === "plot") {
            if (Object.keys(ctx.brushes).length > 0) {
                ctx.applyBrushSelection();
                ctx.syncCheckboxes();
            }
            ctx.renderPlot();
            ctx.updateQuery();
        } else if (ctx.currentView === "trend") {
            if (ctx.trendXBrush || ctx.trendYBrush) {
                ctx.applyTrendSelection();
                ctx.syncCheckboxes();
            }
            ctx.renderTrend();
            ctx.updateQuery();
        }
    }

    ctx.filterValues.forEach((checkbox) => {
        checkbox.addEventListener("change", refreshPlotIfActive);
    });
    if (ctx.dateStart) {
        ctx.dateStart.addEventListener("input", refreshPlotIfActive);
    }
    if (ctx.dateEnd) {
        ctx.dateEnd.addEventListener("input", refreshPlotIfActive);
    }
    const plotDateClear = shadowRoot.querySelector(".skore-summary-date-clear");
    if (plotDateClear) {
        plotDateClear.addEventListener("click", refreshPlotIfActive);
    }
    if (ctx.searchInput) {
        ctx.searchInput.addEventListener("input", () => {
            ctx.refresh();
            refreshPlotIfActive();
        });
    }

    ctx.refresh();
}
