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
    const columnCount = table.querySelectorAll("thead th").length;

    // Current sort/group state (null means "no sort"/"no grouping").
    let sortState = null;
    let groupState = null;

    // ------------------------------------------------------------------ query
    function updateQuery() {
        if (!queryBox) {
            return;
        }
        const ids = [];
        shadowRoot.querySelectorAll(".skore-summary-row:checked").forEach((checkbox) => {
            ids.push(checkbox.dataset.id);
        });
        if (ids.length === 0) {
            queryBox.textContent = queryBox.dataset.empty;
        } else {
            const formatted = ids.map((id) => "'" + id + "'").join(", ");
            // Wrap in double quotes so the copied text is a ready-to-paste Python
            // string literal: Summary.query(<paste>) -> Summary.query("id in [...]").
            queryBox.textContent = '"id in [' + formatted + ']"';
        }
    }

    shadowRoot.querySelectorAll(".skore-summary-row").forEach((checkbox) => {
        checkbox.addEventListener("change", updateQuery);
    });

    if (copyButton && queryBox) {
        copyButton.addEventListener("click", () => {
            const text = queryBox.textContent;
            if (!text || text === queryBox.dataset.empty || !navigator.clipboard) {
                return;
            }
            navigator.clipboard.writeText(text);
        });
    }

    // --------------------------------------------------------------- filtering
    const filterValues = shadowRoot.querySelectorAll(".skore-summary-filter-value");
    const searchInput = shadowRoot.querySelector(".skore-summary-search-input");
    const dateStart = shadowRoot.querySelector(".skore-summary-date-start");
    const dateEnd = shadowRoot.querySelector(".skore-summary-date-end");

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
        const start = dateStart && dateStart.value ? new Date(dateStart.value).getTime() : null;
        const end = dateEnd && dateEnd.value ? new Date(dateEnd.value).getTime() : null;
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
            date.getFullYear() + "-" + pad(date.getMonth() + 1) + "-" + pad(date.getDate());
        if (unit === "second") {
            return (
                day +
                " " +
                pad(date.getHours()) +
                ":" +
                pad(date.getMinutes()) +
                ":" +
                pad(date.getSeconds())
            );
        }
        if (unit === "minute") {
            return day + " " + pad(date.getHours()) + ":" + pad(date.getMinutes());
        }
        if (unit === "hour") {
            return day + " " + pad(date.getHours()) + ":00";
        }
        if (unit === "month") {
            return date.getFullYear() + "-" + pad(date.getMonth() + 1);
        }
        return day;
    }

    function customBucketLabel(date, count, unit) {
        const step = Math.max(1, count);
        if (unit === "month") {
            const months = date.getFullYear() * 12 + date.getMonth();
            const bucket = Math.floor(months / step) * step;
            const start = new Date(Math.floor(bucket / 12), bucket % 12, 1);
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
        // Date grouping.
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
        tbody.querySelectorAll(".summary-group-header").forEach((header) => header.remove());

        const term = searchInput ? searchInput.value.trim().toLowerCase() : "";
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
                const a = rowA.children[columnIndex].dataset.sort || "";
                const b = rowB.children[columnIndex].dataset.sort || "";
                const result = compareValues(a, b, kind);
                if (a === "" || b === "") {
                    return result;
                }
                return ascending ? result : -result;
            });
        }

        tbody.replaceChildren();

        if (groupState) {
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
                cell.colSpan = columnCount;
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
        setPanel(filterToggle, filterPanel, false);
        setPanel(groupToggle, groupPanel, false);
    }

    function wireToggle(toggle, panel, other, otherPanel) {
        if (!toggle || !panel) {
            return;
        }
        toggle.addEventListener("click", (event) => {
            event.stopPropagation();
            const willOpen = panel.hasAttribute("hidden");
            setPanel(other, otherPanel, false);
            setPanel(toggle, panel, willOpen);
        });
        panel.addEventListener("click", (event) => event.stopPropagation());
    }

    wireToggle(filterToggle, filterPanel, groupToggle, groupPanel);
    wireToggle(groupToggle, groupPanel, filterToggle, filterPanel);
    shadowRoot.addEventListener("click", closeMenus);
    // The per-column value submenus open on hover/focus (handled in CSS).

    refresh();
}
