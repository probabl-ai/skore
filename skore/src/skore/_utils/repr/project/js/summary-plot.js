function setupPlot(ctx) {
    const { shadowRoot, dataRows } = ctx;
    const plotArea = ctx.plotArea;
    const plotEmpty = ctx.plotEmpty;
    const colorSelect = ctx.colorSelect;

    ctx.numericColumns = [];
    shadowRoot.querySelectorAll("thead th.summary-sortable").forEach((header) => {
        if (
            header.dataset.sortKind === "number" &&
            header.dataset.columnRole !== "std"
        ) {
            const label = header.querySelector(".summary-th-label");
            ctx.numericColumns.push({
                index: parseInt(header.dataset.columnIndex, 10),
                key: header.dataset.columnKey,
                label: label ? label.textContent : header.textContent,
            });
        }
    });

    ctx.keyToIndex = {};
    ctx.numericColumns.forEach((column) => {
        ctx.keyToIndex[column.key] = column.index;
    });

    function rowMatchesBrushes(row) {
        for (const key in ctx.brushes) {
            const value = parseFloat(ctx.cellAt(row, ctx.keyToIndex[key]).dataset.sort);
            const brush = ctx.brushes[key];
            if (Number.isNaN(value) || value < brush.low || value > brush.high) {
                return false;
            }
        }
        return true;
    }

    function applyBrushSelection() {
        ctx.selectedIds.clear();
        if (Object.keys(ctx.brushes).length === 0) {
            return;
        }
        const activeByField = ctx.collectCheckboxFilters();
        const term = ctx.currentSearchTerm();
        dataRows.forEach((row) => {
            const checkbox = row.querySelector(".skore-summary-row");
            if (!checkbox) {
                return;
            }
            if (
                !ctx.passesCheckboxFilters(row, activeByField) ||
                !ctx.passesDateRange(row) ||
                !ctx.rowMatchesSearch(row, term)
            ) {
                return;
            }
            if (rowMatchesBrushes(row)) {
                ctx.selectedIds.add(checkbox.dataset.id);
            }
        });
    }

    if (colorSelect) {
        ctx.numericColumns.forEach((column, i) => {
            const option = document.createElement("option");
            option.value = String(i);
            option.textContent = column.label;
            colorSelect.appendChild(option);
        });
    }

    function renderPlot() {
        if (!plotArea) {
            return;
        }
        if (ctx.numericColumns.length < 2) {
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

        const axes = ctx.numericColumns.map((column) => {
            let min = Infinity;
            let max = -Infinity;
            dataRows.forEach((row) => {
                const value = parseFloat(ctx.cellAt(row, column.index).dataset.sort);
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

        const hasSelection = ctx.selectedIds.size > 0;

        const activeByField = ctx.collectCheckboxFilters();
        const term = ctx.currentSearchTerm();
        function rowVisible(row) {
            return (
                ctx.passesCheckboxFilters(row, activeByField) &&
                ctx.passesDateRange(row) &&
                ctx.rowMatchesSearch(row, term)
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
            const colorValue = parseFloat(ctx.cellAt(row, colorAxis.index).dataset.sort);
            const t = (colorValue - colorAxis.min) / (colorAxis.max - colorAxis.min);

            let d = "";
            let penDown = false;
            axes.forEach((axis, i) => {
                const value = parseFloat(ctx.cellAt(row, axis.index).dataset.sort);
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
            const selectedRow = checkbox && ctx.selectedIds.has(checkbox.dataset.id);
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

            const brush = ctx.brushes[axis.key];
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
                    delete ctx.brushes[axis.key];
                } else {
                    ctx.brushes[axis.key] = {
                        low: valueFromY(axis, bottom),
                        high: valueFromY(axis, top),
                    };
                }
                applyBrushSelection();
                ctx.syncCheckboxes();
                renderPlot();
                ctx.updateQuery();
            });
        });

        plotArea.replaceChildren(svg);
    }

    ctx.applyBrushSelection = applyBrushSelection;
    ctx.renderPlot = renderPlot;
}
