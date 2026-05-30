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

    const root = shadowRoot.querySelector(".container");
    if (root) {
        applyThemeToReportContainer(root, containerId + "-wrapper");
    }

    const table = shadowRoot.querySelector(".summary-table");
    const queryBox = shadowRoot.querySelector(".skore-summary-query");
    const copyButton = shadowRoot.querySelector(".skore-summary-copy");
    if (!table) {
        return;
    }

    const tbody = table.querySelector("tbody");
    const dataRows = Array.from(tbody.querySelectorAll("tr"));

    // Current sort/group state (null means "no sort"/"no grouping").
    let sortState = null;
    let groupState = null;

    // ------------------------------------------------------------------ query
    // A single selection (set of report ids) is shared by both views, so it
    // persists when switching between the table and the plot. The query box
    // always lists the selected ids, and ``clearBrushes`` / ``renderPlot`` are
    // defined later in the Plot section (hoisted function declarations).
    let currentView = "table";
    let queryIsEmpty = true;
    const selectedIds = new Set();

    function emptyMessage() {
        if (currentView === "plot") {
            return "Brush an axis to select reports.";
        }
        return queryBox ? queryBox.dataset.empty : "";
    }

    function setQuery(text, isEmpty) {
        if (!queryBox) {
            return;
        }
        queryIsEmpty = isEmpty;
        queryBox.textContent = isEmpty ? emptyMessage() : text;
    }

    function updateQuery() {
        const ids = [];
        dataRows.forEach((row) => {
            const checkbox = row.querySelector(".skore-summary-row");
            if (checkbox && selectedIds.has(checkbox.dataset.id)) {
                ids.push(checkbox.dataset.id);
            }
        });
        if (ids.length === 0) {
            setQuery("", true);
            return;
        }
        const formatted = ids.map((id) => "'" + id + "'").join(", ");
        // Wrap in double quotes so the copied text is a ready-to-paste Python
        // string literal: Summary.query(<paste>) -> Summary.query("id in [...]").
        setQuery('"id in [' + formatted + ']"', false);
    }

    function syncCheckboxes() {
        shadowRoot.querySelectorAll(".skore-summary-row").forEach((checkbox) => {
            checkbox.checked = selectedIds.has(checkbox.dataset.id);
        });
    }

    // Track the last checkbox that was clicked so a subsequent shift-click can
    // toggle the entire range between the two (only visible, rendered rows are
    // considered, matching the rows the user actually sees).
    let lastClickedRow = null;

    shadowRoot.querySelectorAll(".skore-summary-row").forEach((checkbox) => {
        checkbox.addEventListener("mousedown", (event) => {
            // Suppress the browser-default text selection when extending a
            // range with shift; the subsequent click still fires normally.
            if (event.shiftKey) {
                event.preventDefault();
            }
        });
        checkbox.addEventListener("click", (event) => {
            if (event.shiftKey && lastClickedRow && lastClickedRow !== checkbox) {
                // Only rows the user actually sees take part in the range, so a
                // shift-click never ticks reports hidden by an active filter,
                // search or date range.
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
                    // ``checkbox.checked`` already reflects the post-click state;
                    // align every row in between with that target value.
                    const target = checkbox.checked;
                    for (let i = lo; i <= hi; i++) {
                        visibleCheckboxes[i].checked = target;
                    }
                }
            }
            // Rebuild the shared selection from the current DOM state so both
            // single and range clicks end up consistent.
            selectedIds.clear();
            shadowRoot.querySelectorAll(".skore-summary-row").forEach((cb) => {
                if (cb.checked) {
                    selectedIds.add(cb.dataset.id);
                }
            });
            // Ticks define the selection directly, so any axis brush is stale.
            clearBrushes();
            updateQuery();
            lastClickedRow = checkbox;
        });
    });

    if (copyButton && queryBox) {
        copyButton.addEventListener("click", () => {
            if (queryIsEmpty || !navigator.clipboard) {
                return;
            }
            navigator.clipboard.writeText(queryBox.textContent);
        });
    }

    const clearButton = shadowRoot.querySelector(".skore-summary-clear");
    if (clearButton) {
        clearButton.addEventListener("click", () => {
            selectedIds.clear();
            clearBrushes();
            syncCheckboxes();
            if (currentView === "plot") {
                renderPlot();
            }
            updateQuery();
        });
    }

    // --------------------------------------------------------------- filtering
    const filterValues = shadowRoot.querySelectorAll(".skore-summary-filter-value");
    const searchInput = shadowRoot.querySelector(".skore-summary-search-input");
    const dateStart = shadowRoot.querySelector(".skore-summary-date-start");
    const dateEnd = shadowRoot.querySelector(".skore-summary-date-end");

    function currentSearchTerm() {
        return searchInput ? searchInput.value.trim().toLowerCase() : "";
    }

    function rowMatchesSearch(row, term) {
        if (!term) {
            return true;
        }
        for (const cell of row.querySelectorAll("td")) {
            const haystack = (
                cell.textContent +
                " " +
                (cell.getAttribute("title") || "") +
                " " +
                (cell.dataset.sort || "")
            ).toLowerCase();
            if (haystack.indexOf(term) !== -1) {
                return true;
            }
        }
        return false;
    }

    function collectCheckboxFilters() {
        const activeByField = {};
        filterValues.forEach((checkbox) => {
            const field = checkbox.dataset.filterField;
            if (!(field in activeByField)) {
                activeByField[field] = new Set();
            }
            if (checkbox.checked) {
                activeByField[field].add(checkbox.dataset.filterValue);
            }
        });
        return activeByField;
    }

    function rowDatasetValue(row, field) {
        return row.dataset[field === "report_type" ? "reportType" : field];
    }

    function passesCheckboxFilters(row, activeByField) {
        return Object.entries(activeByField).every(([field, values]) =>
            values.has(rowDatasetValue(row, field))
        );
    }

    function passesDateRange(row) {
        // Report dates are stored as UTC ISO strings, so the wall-clock entered
        // in the (timezone-less) datetime-local inputs is read as UTC too by
        // appending "Z"; this keeps the table column, grouping and filter aligned.
        const start =
            dateStart && dateStart.value ? new Date(dateStart.value + "Z").getTime() : null;
        const end =
            dateEnd && dateEnd.value ? new Date(dateEnd.value + "Z").getTime() : null;
        if (start === null && end === null) {
            return true;
        }
        const iso = row.dataset.date;
        if (!iso) {
            return false;
        }
        const time = new Date(iso).getTime();
        if (Number.isNaN(time)) {
            return false;
        }
        if (start !== null && time < start) {
            return false;
        }
        if (end !== null && time > end) {
            return false;
        }
        return true;
    }

    // ----------------------------------------------------------------- sorting
    function compareValues(a, b, kind) {
        const emptyA = a === "";
        const emptyB = b === "";
        if (emptyA || emptyB) {
            // Always push empty values to the bottom.
            return emptyA === emptyB ? 0 : emptyA ? 1 : -1;
        }
        if (kind === "number") {
            return parseFloat(a) - parseFloat(b);
        }
        if (kind === "date") {
            return new Date(a) - new Date(b);
        }
        return a < b ? -1 : a > b ? 1 : 0;
    }

    function updateSortIndicators() {
        shadowRoot.querySelectorAll(".summary-sortable").forEach((header) => {
            const columnIndex = parseInt(header.dataset.columnIndex, 10);
            if (sortState && sortState.columnIndex === columnIndex) {
                header.setAttribute(
                    "aria-sort",
                    sortState.ascending ? "ascending" : "descending"
                );
            } else {
                header.setAttribute("aria-sort", "none");
            }
        });
    }

    function sortByColumn(header) {
        const columnIndex = parseInt(header.dataset.columnIndex, 10);
        const kind = header.dataset.sortKind;
        const ascending = !(
            sortState &&
            sortState.columnIndex === columnIndex &&
            sortState.ascending
        );
        sortState = { columnIndex, kind, ascending };
        refresh();
    }

    shadowRoot.querySelectorAll(".summary-sortable").forEach((header) => {
        header.addEventListener("click", () => sortByColumn(header));
        header.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                sortByColumn(header);
            }
        });
    });

    // ---------------------------------------------------------------- grouping
    function pad(value) {
        return String(value).padStart(2, "0");
    }

    function formatBucket(date, unit) {
        const day =
            date.getUTCFullYear() +
            "-" +
            pad(date.getUTCMonth() + 1) +
            "-" +
            pad(date.getUTCDate());
        if (unit === "second") {
            return (
                day +
                " " +
                pad(date.getUTCHours()) +
                ":" +
                pad(date.getUTCMinutes()) +
                ":" +
                pad(date.getUTCSeconds())
            );
        }
        if (unit === "minute") {
            return day + " " + pad(date.getUTCHours()) + ":" + pad(date.getUTCMinutes());
        }
        if (unit === "hour") {
            return day + " " + pad(date.getUTCHours()) + ":00";
        }
        if (unit === "month") {
            return date.getUTCFullYear() + "-" + pad(date.getUTCMonth() + 1);
        }
        return day;
    }

    function customBucketLabel(date, count, unit) {
        const step = Math.max(1, count);
        if (unit === "month") {
            const months = date.getUTCFullYear() * 12 + date.getUTCMonth();
            const bucket = Math.floor(months / step) * step;
            const start = new Date(Date.UTC(Math.floor(bucket / 12), bucket % 12, 1));
            return formatBucket(start, "month");
        }
        const unitMs = {
            second: 1000,
            minute: 60000,
            hour: 3600000,
            day: 86400000,
            week: 604800000,
        }[unit];
        const buckets = Math.floor(date.getTime() / unitMs);
        const start = new Date(Math.floor(buckets / step) * step * unitMs);
        return formatBucket(start, unit);
    }

    function groupPrefix() {
        if (!groupState) {
            return "";
        }
        if (groupState.field === "learner") {
            return "Learner: ";
        }
        if (groupState.field === "report_type") {
            return "Estimator type: ";
        }
        return "Date: ";
    }

    function groupInfo(row) {
        if (groupState.field === "learner") {
            const value = row.dataset.learner || "(none)";
            return { key: "learner:" + value, label: value };
        }
        if (groupState.field === "report_type") {
            const value = row.dataset.reportType || "(none)";
            return { key: "report_type:" + value, label: value };
        }
        const iso = row.dataset.date;
        if (!iso) {
            return { key: "date:none", label: "(no date)" };
        }
        const date = new Date(iso);
        if (Number.isNaN(date.getTime())) {
            return { key: "date:" + iso, label: iso };
        }
        let label;
        if (groupState.unit === "custom") {
            label = customBucketLabel(date, groupState.count, groupState.customUnit);
        } else {
            label = formatBucket(date, groupState.unit);
        }
        return { key: "date:" + groupState.unit + ":" + label, label };
    }

    // ------------------------------------------------------------- main render
    function refresh() {
        const term = currentSearchTerm();
        const activeByField = collectCheckboxFilters();
        const visible = new Map();
        dataRows.forEach((row) => {
            const ok =
                passesCheckboxFilters(row, activeByField) &&
                passesDateRange(row) &&
                rowMatchesSearch(row, term);
            visible.set(row, ok);
        });

        let ordered = dataRows.slice();
        if (sortState) {
            const { columnIndex, kind, ascending } = sortState;
            ordered.sort((rowA, rowB) => {
                const a = cellAt(rowA, columnIndex).dataset.sort || "";
                const b = cellAt(rowB, columnIndex).dataset.sort || "";
                const result = compareValues(a, b, kind);
                if (a === "" || b === "") {
                    return result;
                }
                return ascending ? result : -result;
            });
        }

        tbody.replaceChildren();

        if (groupState) {
            // Group separators span only the columns currently shown so the bar
            // never overshoots when columns are hidden.
            const visibleColumnCount = shadowRoot.querySelectorAll(
                "thead th:not(.summary-col-hidden)"
            ).length;
            const groups = [];
            const byKey = new Map();
            ordered.forEach((row) => {
                const info = groupInfo(row);
                if (!byKey.has(info.key)) {
                    byKey.set(info.key, { label: info.label, rows: [] });
                    groups.push(byKey.get(info.key));
                }
                byKey.get(info.key).rows.push(row);
            });
            groups.forEach((group) => {
                const shown = group.rows.filter((row) => visible.get(row)).length;
                const header = document.createElement("tr");
                header.className = "summary-group-header";
                const cell = document.createElement("td");
                cell.colSpan = visibleColumnCount;
                cell.textContent = groupPrefix() + group.label + " (" + shown + ")";
                header.appendChild(cell);
                if (shown === 0) {
                    header.style.display = "none";
                }
                tbody.appendChild(header);
                group.rows.forEach((row) => {
                    row.style.display = visible.get(row) ? "" : "none";
                    tbody.appendChild(row);
                });
            });
        } else {
            ordered.forEach((row) => {
                row.style.display = visible.get(row) ? "" : "none";
                tbody.appendChild(row);
            });
        }

        updateSortIndicators();
    }

    filterValues.forEach((checkbox) => {
        checkbox.addEventListener("change", refresh);
    });
    if (searchInput) {
        searchInput.addEventListener("input", refresh);
    }
    if (dateStart) {
        dateStart.addEventListener("input", refresh);
    }
    if (dateEnd) {
        dateEnd.addEventListener("input", refresh);
    }
    const dateClear = shadowRoot.querySelector(".skore-summary-date-clear");
    if (dateClear) {
        dateClear.addEventListener("click", () => {
            if (dateStart) dateStart.value = "";
            if (dateEnd) dateEnd.value = "";
            refresh();
        });
    }

    // --------------------------------------------------------- group-by controls
    const groupN = shadowRoot.querySelector(".skore-summary-group-n");
    const groupUnit = shadowRoot.querySelector(".skore-summary-group-unit");
    const groupOptions = shadowRoot.querySelectorAll(".skore-summary-group-option");

    function markActiveGroup(active) {
        groupOptions.forEach((option) => {
            option.classList.toggle("skore-summary-group-option--active", option === active);
        });
    }

    groupOptions.forEach((option) => {
        option.addEventListener("click", (event) => {
            event.stopPropagation();
            const group = option.dataset.group;
            if (group === "none") {
                groupState = null;
                markActiveGroup(null);
            } else if (group === "date") {
                const unit = option.dataset.unit;
                if (unit === "custom") {
                    const count = Math.max(1, parseInt(groupN ? groupN.value : "1", 10) || 1);
                    groupState = {
                        field: "date",
                        unit: "custom",
                        count,
                        customUnit: groupUnit ? groupUnit.value : "hour",
                    };
                } else {
                    groupState = { field: "date", unit };
                }
                markActiveGroup(option);
            } else {
                groupState = { field: group };
                markActiveGroup(option);
            }
            refresh();
            closeMenus();
        });
    });

    // ----------------------------------------------------- popover open/close
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

    function closeMenus() {
        popovers.forEach(([toggle, panel]) => setPanel(toggle, panel, false));
    }

    function wireToggle(toggle, panel) {
        if (!toggle || !panel) {
            return;
        }
        toggle.addEventListener("click", (event) => {
            event.stopPropagation();
            const willOpen = panel.hasAttribute("hidden");
            closeMenus();
            setPanel(toggle, panel, willOpen);
        });
        panel.addEventListener("click", (event) => event.stopPropagation());
    }

    popovers.forEach(([toggle, panel]) => wireToggle(toggle, panel));
    shadowRoot.addEventListener("click", closeMenus);
    // The per-column value submenus open on hover/focus (handled in CSS).

    // ------------------------------------------------------ column ordering
    // Each cell carries a stable ``data-column-index`` matching its header, so
    // ``cellAt(row, naturalIndex)`` keeps working even after the DOM has been
    // reshuffled (it never relies on the cell's current child position).
    function cellAt(row, naturalIndex) {
        return row.querySelector(
            'td[data-column-index="' + naturalIndex + '"]'
        );
    }

    // Unchecked columns are hidden. When the user re-checks one, its header
    // and cells are appended to the end of their row so the newly-shown column
    // appears on the rightmost side of the visible columns; cells stay
    // addressable by their stable ``data-column-index`` regardless of order.
    const columnToggles = Array.from(
        shadowRoot.querySelectorAll(".skore-summary-column-toggle")
    );
    const headerRow = shadowRoot.querySelector("thead tr");

    function headerAt(naturalIndex) {
        return shadowRoot.querySelector(
            'thead th[data-column-index="' + naturalIndex + '"]'
        );
    }

    function setColumnVisible(naturalIndex, visible) {
        const header = headerAt(naturalIndex);
        if (visible && header && headerRow) {
            headerRow.appendChild(header);
        }
        if (header) {
            header.classList.toggle("summary-col-hidden", !visible);
        }
        dataRows.forEach((row) => {
            const cell = cellAt(row, naturalIndex);
            if (!cell) {
                return;
            }
            if (visible) {
                row.appendChild(cell);
            }
            cell.classList.toggle("summary-col-hidden", !visible);
        });
    }

    columnToggles.forEach((toggle) => {
        const index = toggle.dataset.columnIndex;
        // Apply the initial visibility: default-unchecked columns start hidden
        // in place (no reorder); they only move to the right once promoted.
        if (!toggle.checked) {
            const header = headerAt(index);
            if (header) {
                header.classList.add("summary-col-hidden");
            }
            dataRows.forEach((row) => {
                const cell = cellAt(row, index);
                if (cell) {
                    cell.classList.add("summary-col-hidden");
                }
            });
        }
        toggle.addEventListener("change", () => {
            setColumnVisible(index, toggle.checked);
        });
    });

    // ---------------------------------------------------------------- plot view
    const container = root;
    const plotArea = shadowRoot.querySelector(".summary-plot");
    const plotEmpty = shadowRoot.querySelector(".summary-plot-empty");
    const colorSelect = shadowRoot.querySelector(".skore-summary-color-metric");
    const viewButtons = shadowRoot.querySelectorAll(".summary-view-btn");
    const SVG_NS = "http://www.w3.org/2000/svg";

    // The plot axes are the numeric metric columns; cellAt() reads each row's
    // value via the column's stable data-column-index.
    const numericColumns = [];
    shadowRoot.querySelectorAll("thead th.summary-sortable").forEach((header) => {
        if (header.dataset.sortKind === "number") {
            const label = header.querySelector(".summary-th-label");
            numericColumns.push({
                index: parseInt(header.dataset.columnIndex, 10),
                key: header.dataset.columnKey,
                label: label ? label.textContent : header.textContent,
            });
        }
    });

    // Brushed ranges per axis (keyed by column name). Brushing is a tool to set
    // the shared id selection; the selection itself drives highlighting/query.
    const brushes = {};
    const keyToIndex = {};
    numericColumns.forEach((column) => {
        keyToIndex[column.key] = column.index;
    });

    function clearBrushes() {
        for (const key in brushes) {
            delete brushes[key];
        }
    }

    function rowMatchesBrushes(row) {
        for (const key in brushes) {
            const value = parseFloat(cellAt(row, keyToIndex[key]).dataset.sort);
            const brush = brushes[key];
            if (Number.isNaN(value) || value < brush.low || value > brush.high) {
                return false;
            }
        }
        return true;
    }

    // Recompute the shared selection from the current brushes, restricted to the
    // rows the Filter currently keeps visible.
    function applyBrushSelection() {
        selectedIds.clear();
        if (Object.keys(brushes).length === 0) {
            return;
        }
        const activeByField = collectCheckboxFilters();
        const term = currentSearchTerm();
        dataRows.forEach((row) => {
            const checkbox = row.querySelector(".skore-summary-row");
            if (!checkbox) {
                return;
            }
            if (
                !passesCheckboxFilters(row, activeByField) ||
                !passesDateRange(row) ||
                !rowMatchesSearch(row, term)
            ) {
                return;
            }
            if (rowMatchesBrushes(row)) {
                selectedIds.add(checkbox.dataset.id);
            }
        });
    }

    if (colorSelect) {
        numericColumns.forEach((column, i) => {
            const option = document.createElement("option");
            option.value = String(i);
            option.textContent = column.label;
            colorSelect.appendChild(option);
        });
    }

    function anchorFor(i, count) {
        if (i === 0) return "start";
        if (i === count - 1) return "end";
        return "middle";
    }

    function formatTick(value) {
        if (!isFinite(value)) {
            return "";
        }
        return Number.parseFloat(value.toPrecision(4)).toString();
    }

    function colorFor(t) {
        // Map t in [0, 1] across a blue -> light gray -> red gradient; NaN is gray.
        if (Number.isNaN(t)) {
            return "rgb(150, 150, 150)";
        }
        const stops = [
            [59, 76, 192],
            [221, 221, 221],
            [180, 4, 38],
        ];
        const clamped = Math.min(1, Math.max(0, t));
        const scaled = clamped * (stops.length - 1);
        const lower = Math.floor(scaled);
        const upper = Math.min(stops.length - 1, lower + 1);
        const frac = scaled - lower;
        const mix = (a, b) => Math.round(a + (b - a) * frac);
        return (
            "rgb(" +
            mix(stops[lower][0], stops[upper][0]) +
            ", " +
            mix(stops[lower][1], stops[upper][1]) +
            ", " +
            mix(stops[lower][2], stops[upper][2]) +
            ")"
        );
    }

    function appendText(group, className, x, y, anchor, text) {
        const node = document.createElementNS(SVG_NS, "text");
        node.setAttribute("class", className);
        node.setAttribute("x", x);
        node.setAttribute("y", y);
        node.setAttribute("text-anchor", anchor);
        node.textContent = text;
        group.appendChild(node);
    }

    function renderPlot() {
        if (!plotArea) {
            return;
        }
        if (numericColumns.length < 2) {
            if (plotEmpty) {
                plotEmpty.hidden = false;
            }
            plotArea.replaceChildren();
            return;
        }
        if (plotEmpty) {
            plotEmpty.hidden = true;
        }

        const width = plotArea.clientWidth || 600;
        const height = 360;
        const margin = { top: 30, right: 24, bottom: 28, left: 24 };
        const innerW = Math.max(1, width - margin.left - margin.right);
        const innerH = Math.max(1, height - margin.top - margin.bottom);

        const axes = numericColumns.map((column) => {
            let min = Infinity;
            let max = -Infinity;
            dataRows.forEach((row) => {
                const value = parseFloat(cellAt(row, column.index).dataset.sort);
                if (!Number.isNaN(value)) {
                    if (value < min) min = value;
                    if (value > max) max = value;
                }
            });
            if (min === Infinity) {
                min = 0;
                max = 1;
            } else if (min === max) {
                min -= 0.5;
                max += 0.5;
            }
            return { index: column.index, key: column.key, label: column.label, min, max };
        });

        const axisX = (i) => (innerW * i) / (axes.length - 1);
        const valueY = (axis, value) =>
            innerH - ((value - axis.min) / (axis.max - axis.min)) * innerH;
        const valueFromY = (axis, y) =>
            axis.min + ((innerH - y) / innerH) * (axis.max - axis.min);

        function plotYFromEvent(evt) {
            const point = svg.createSVGPoint();
            point.x = evt.clientX;
            point.y = evt.clientY;
            const ctm = svg.getScreenCTM();
            if (!ctm) {
                return 0;
            }
            const local = point.matrixTransform(ctm.inverse());
            return Math.min(innerH, Math.max(0, local.y - margin.top));
        }

        const hasSelection = selectedIds.size > 0;

        // The Filter menu (categorical values + date range) and the search term
        // narrow which lines are drawn, mirroring how they subset rows in the
        // table view so both stay consistent.
        const activeByField = collectCheckboxFilters();
        const term = currentSearchTerm();
        function rowVisible(row) {
            return (
                passesCheckboxFilters(row, activeByField) &&
                passesDateRange(row) &&
                rowMatchesSearch(row, term)
            );
        }

        const colorIndex = colorSelect ? parseInt(colorSelect.value, 10) || 0 : 0;
        const colorAxis = axes[Math.min(colorIndex, axes.length - 1)];

        const svg = document.createElementNS(SVG_NS, "svg");
        svg.setAttribute("viewBox", "0 0 " + width + " " + height);
        svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

        const plotG = document.createElementNS(SVG_NS, "g");
        plotG.setAttribute(
            "transform",
            "translate(" + margin.left + ", " + margin.top + ")"
        );
        svg.appendChild(plotG);

        dataRows.forEach((row) => {
            if (!rowVisible(row)) {
                return;
            }
            const checkbox = row.querySelector(".skore-summary-row");
            const colorValue = parseFloat(cellAt(row, colorAxis.index).dataset.sort);
            const t = (colorValue - colorAxis.min) / (colorAxis.max - colorAxis.min);

            // Break the polyline at axes where the report has no value.
            let d = "";
            let penDown = false;
            axes.forEach((axis, i) => {
                const value = parseFloat(cellAt(row, axis.index).dataset.sort);
                if (Number.isNaN(value)) {
                    penDown = false;
                    return;
                }
                d += (penDown ? " L" : " M") + axisX(i) + " " + valueY(axis, value);
                penDown = true;
            });
            if (!d) {
                return;
            }

            const path = document.createElementNS(SVG_NS, "path");
            const selectedRow = checkbox && selectedIds.has(checkbox.dataset.id);
            const dim = hasSelection && !selectedRow;
            path.setAttribute(
                "class",
                dim ? "summary-plot-line summary-plot-line--dim" : "summary-plot-line"
            );
            path.setAttribute("d", d.trim());
            path.setAttribute("stroke", colorFor(Number.isNaN(colorValue) ? NaN : t));
            const title = document.createElementNS(SVG_NS, "title");
            title.textContent = checkbox ? checkbox.dataset.id : "";
            path.appendChild(title);
            plotG.appendChild(path);
        });

        axes.forEach((axis, i) => {
            const x = axisX(i);
            const anchor = anchorFor(i, axes.length);
            const line = document.createElementNS(SVG_NS, "line");
            line.setAttribute("class", "summary-plot-axis-line");
            line.setAttribute("x1", x);
            line.setAttribute("y1", 0);
            line.setAttribute("x2", x);
            line.setAttribute("y2", innerH);
            plotG.appendChild(line);

            appendText(plotG, "summary-plot-axis-label", x, -12, anchor, axis.label);
            appendText(plotG, "summary-plot-tick", x, -2, anchor, formatTick(axis.max));
            appendText(
                plotG,
                "summary-plot-tick",
                x,
                innerH + 14,
                anchor,
                formatTick(axis.min)
            );

            // Persistent brush rectangle for an existing selection on this axis.
            const brush = brushes[axis.key];
            if (brush) {
                const top = valueY(axis, brush.high);
                const bottom = valueY(axis, brush.low);
                const rect = document.createElementNS(SVG_NS, "rect");
                rect.setAttribute("class", "summary-plot-brush");
                rect.setAttribute("x", x - 8);
                rect.setAttribute("width", 16);
                rect.setAttribute("y", top);
                rect.setAttribute("height", Math.max(0, bottom - top));
                plotG.appendChild(rect);
            }

            // Invisible track on top of the axis to capture range brushing.
            const track = document.createElementNS(SVG_NS, "rect");
            track.setAttribute("class", "summary-plot-brush-track");
            track.setAttribute("x", x - 8);
            track.setAttribute("width", 16);
            track.setAttribute("y", 0);
            track.setAttribute("height", innerH);
            plotG.appendChild(track);

            let dragStartY = null;
            let liveRect = null;
            track.addEventListener("pointerdown", (evt) => {
                evt.preventDefault();
                track.setPointerCapture(evt.pointerId);
                dragStartY = plotYFromEvent(evt);
                liveRect = document.createElementNS(SVG_NS, "rect");
                liveRect.setAttribute("class", "summary-plot-brush");
                liveRect.setAttribute("x", x - 8);
                liveRect.setAttribute("width", 16);
                liveRect.setAttribute("y", dragStartY);
                liveRect.setAttribute("height", 0);
                plotG.appendChild(liveRect);
            });
            track.addEventListener("pointermove", (evt) => {
                if (dragStartY === null) {
                    return;
                }
                const y = plotYFromEvent(evt);
                const top = Math.min(dragStartY, y);
                liveRect.setAttribute("y", top);
                liveRect.setAttribute("height", Math.abs(y - dragStartY));
            });
            track.addEventListener("pointerup", (evt) => {
                if (dragStartY === null) {
                    return;
                }
                const y = plotYFromEvent(evt);
                const top = Math.min(dragStartY, y);
                const bottom = Math.max(dragStartY, y);
                dragStartY = null;
                if (bottom - top < 3) {
                    // A click (no meaningful drag) clears the brush on this axis.
                    delete brushes[axis.key];
                } else {
                    brushes[axis.key] = {
                        low: valueFromY(axis, bottom),
                        high: valueFromY(axis, top),
                    };
                }
                applyBrushSelection();
                syncCheckboxes();
                renderPlot();
                updateQuery();
            });
        });

        plotArea.replaceChildren(svg);
    }

    function setView(view) {
        currentView = view;
        if (container) {
            container.dataset.view = view;
        }
        viewButtons.forEach((button) => {
            const active = button.dataset.view === view;
            button.classList.toggle("summary-view-btn--active", active);
            button.setAttribute("aria-pressed", active ? "true" : "false");
        });
        if (view === "plot") {
            renderPlot();
        } else {
            syncCheckboxes();
        }
        updateQuery();
    }

    viewButtons.forEach((button) => {
        button.addEventListener("click", () => setView(button.dataset.view));
    });

    if (colorSelect) {
        colorSelect.addEventListener("change", () => {
            if (container && container.dataset.view === "plot") {
                renderPlot();
            }
        });
    }

    if (plotArea && typeof ResizeObserver !== "undefined") {
        const observer = new ResizeObserver(() => {
            if (container && container.dataset.view === "plot") {
                renderPlot();
            }
        });
        observer.observe(plotArea);
    }

    // Keep the plot and the selection in sync with the shared Filter menu while
    // in Plot view: an active brush re-selects within the now-visible rows.
    function refreshPlotIfActive() {
        if (currentView !== "plot") {
            return;
        }
        if (Object.keys(brushes).length > 0) {
            applyBrushSelection();
            syncCheckboxes();
        }
        renderPlot();
        updateQuery();
    }
    filterValues.forEach((checkbox) => {
        checkbox.addEventListener("change", refreshPlotIfActive);
    });
    if (dateStart) {
        dateStart.addEventListener("input", refreshPlotIfActive);
    }
    if (dateEnd) {
        dateEnd.addEventListener("input", refreshPlotIfActive);
    }
    const plotDateClear = shadowRoot.querySelector(".skore-summary-date-clear");
    if (plotDateClear) {
        plotDateClear.addEventListener("click", refreshPlotIfActive);
    }

    refresh();
}
