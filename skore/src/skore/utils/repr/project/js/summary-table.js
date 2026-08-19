const SKORE_SUMMARY_ELLIPSIS_COLUMNS = new Set(["id", "dataset"]);

function setupTable(ctx) {
    const { shadowRoot, tbody, dataRows } = ctx;

    shadowRoot.querySelectorAll(".summary-sortable").forEach((th) => {
        const columnKey = th.dataset.columnKey;
        const columnIndex = parseInt(th.dataset.columnIndex, 10);
        const ellipsize = SKORE_SUMMARY_ELLIPSIS_COLUMNS.has(columnKey);
        const kind = th.dataset.sortKind;
        if (!ellipsize && kind !== "date" && kind !== "number") {
            return;
        }
        dataRows.forEach((row) => {
            const cell = row.querySelector(`td[data-column-index='${columnIndex}']`);
            if (!cell) {
                return;
            }
            // Numbers are sent raw in ``data-sort``; render the visible value here
            // (and an empty cell for NA) so formatting is not duplicated in Python.
            if (kind === "number") {
                cell.textContent = formatNumber(cell.dataset.sort);
                return;
            }
            if (!cell.textContent) {
                return;
            }
            if (ellipsize) {
                const full = cell.textContent;
                cell.title = full;
                cell.textContent = middleEllipsis(full);
            } else {
                cell.textContent = formatDate(cell.textContent);
            }
        });
    });

    shadowRoot.querySelectorAll(".skore-summary-filter-field-toggle").forEach((toggle) => {
        if (!SKORE_SUMMARY_ELLIPSIS_COLUMNS.has(toggle.dataset.field)) {
            return;
        }
        const field = toggle.closest(".skore-summary-filter-field");
        if (!field) {
            return;
        }
        field.querySelectorAll(".skore-summary-filter-option").forEach((label) => {
            const span = label.querySelector(".skore-summary-filter-option-label");
            if (!span) {
                return;
            }
            const full = span.textContent;
            label.title = full;
            span.textContent = middleEllipsis(full);
        });
    });

    function currentSearchTerm() {
        return ctx.searchInput ? ctx.searchInput.value.trim().toLowerCase() : "";
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
        ctx.filterValues.forEach((checkbox) => {
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
            ctx.dateStart && ctx.dateStart.value
                ? new Date(ctx.dateStart.value + "Z").getTime()
                : null;
        const end =
            ctx.dateEnd && ctx.dateEnd.value
                ? new Date(ctx.dateEnd.value + "Z").getTime()
                : null;
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

    function updateSortIndicators() {
        shadowRoot.querySelectorAll(".summary-sortable").forEach((header) => {
            const columnIndex = parseInt(header.dataset.columnIndex, 10);
            if (ctx.sortState && ctx.sortState.columnIndex === columnIndex) {
                header.setAttribute(
                    "aria-sort",
                    ctx.sortState.ascending ? "ascending" : "descending"
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
            ctx.sortState &&
            ctx.sortState.columnIndex === columnIndex &&
            ctx.sortState.ascending
        );
        ctx.sortState = { columnIndex, kind, ascending };
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

    function groupPrefix() {
        if (!ctx.groupState) {
            return "";
        }
        if (ctx.groupState.field === "learner") {
            return "Learner: ";
        }
        if (ctx.groupState.field === "report_type") {
            return "Estimator type: ";
        }
        return "Date: ";
    }

    function groupInfo(row) {
        if (ctx.groupState.field === "learner") {
            const value = row.dataset.learner || "(none)";
            return { key: "learner:" + value, label: value };
        }
        if (ctx.groupState.field === "report_type") {
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
        if (ctx.groupState.unit === "custom") {
            label = customBucketLabel(date, ctx.groupState.count, ctx.groupState.customUnit);
        } else {
            label = formatBucket(date, ctx.groupState.unit);
        }
        return { key: "date:" + ctx.groupState.unit + ":" + label, label };
    }

    function cellAt(row, naturalIndex) {
        return row.querySelector('td[data-column-index="' + naturalIndex + '"]');
    }

    function headerAt(naturalIndex) {
        return shadowRoot.querySelector('thead th[data-column-index="' + naturalIndex + '"]');
    }

    function setColumnVisible(naturalIndex, visible) {
        const header = headerAt(naturalIndex);
        if (header) {
            header.classList.toggle("summary-col-hidden", !visible);
        }
        dataRows.forEach((row) => {
            const cell = cellAt(row, naturalIndex);
            if (cell) {
                cell.classList.toggle("summary-col-hidden", !visible);
            }
        });
    }

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
        if (ctx.sortState) {
            const { columnIndex, kind, ascending } = ctx.sortState;
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

        if (ctx.groupState) {
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

    ctx.filterValues.forEach((checkbox) => {
        checkbox.addEventListener("change", refresh);
    });
    if (ctx.dateStart) {
        ctx.dateStart.addEventListener("input", refresh);
    }
    if (ctx.dateEnd) {
        ctx.dateEnd.addEventListener("input", refresh);
    }
    const dateClear = shadowRoot.querySelector(".skore-summary-date-clear");
    if (dateClear) {
        dateClear.addEventListener("click", () => {
            if (ctx.dateStart) ctx.dateStart.value = "";
            if (ctx.dateEnd) ctx.dateEnd.value = "";
            refresh();
        });
    }

    function markActiveGroup(active) {
        ctx.groupOptions.forEach((option) => {
            option.classList.toggle("skore-summary-group-option--active", option === active);
        });
    }

    ctx.groupOptions.forEach((option) => {
        option.addEventListener("click", (event) => {
            event.stopPropagation();
            const group = option.dataset.group;
            if (group === "none") {
                ctx.groupState = null;
                markActiveGroup(null);
            } else if (group === "date") {
                const unit = option.dataset.unit;
                if (unit === "custom") {
                    const count = Math.max(
                        1,
                        parseInt(ctx.groupN ? ctx.groupN.value : "1", 10) || 1
                    );
                    ctx.groupState = {
                        field: "date",
                        unit: "custom",
                        count,
                        customUnit: ctx.groupUnit ? ctx.groupUnit.value : "hour",
                    };
                } else {
                    ctx.groupState = { field: "date", unit };
                }
                markActiveGroup(option);
            } else {
                ctx.groupState = { field: group };
                markActiveGroup(option);
            }
            refresh();
            if (ctx.currentView === "trend" && ctx.renderTrend) {
                ctx.renderTrend();
            }
            if (ctx.closeMenus) {
                ctx.closeMenus();
            }
        });
    });

    const columnToggles = Array.from(
        shadowRoot.querySelectorAll(".skore-summary-column-toggle")
    );
    columnToggles.forEach((toggle) => {
        const index = toggle.dataset.columnIndex;
        setColumnVisible(index, toggle.checked);
        toggle.addEventListener("change", () => {
            setColumnVisible(index, toggle.checked);
        });
    });

    ctx.currentSearchTerm = currentSearchTerm;
    ctx.collectCheckboxFilters = collectCheckboxFilters;
    ctx.passesCheckboxFilters = passesCheckboxFilters;
    ctx.passesDateRange = passesDateRange;
    ctx.rowMatchesSearch = rowMatchesSearch;
    ctx.groupInfo = groupInfo;
    ctx.groupPrefix = groupPrefix;
    ctx.cellAt = cellAt;
    ctx.headerAt = headerAt;
    ctx.refresh = refresh;
}
