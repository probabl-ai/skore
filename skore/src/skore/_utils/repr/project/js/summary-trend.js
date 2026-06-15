function trendPalette(container) {
    const blue = themeColor(container, "--color-blue", [48, 67, 240]);
    const orange = themeColor(container, "--color-orange", [249, 115, 22]);
    const toRgb = (c) => "rgb(" + c[0] + ", " + c[1] + ", " + c[2] + ")";
    return [
        toRgb(blue),
        toRgb(orange),
        "rgb(27, 158, 119)",
        "rgb(180, 4, 38)",
        "rgb(117, 112, 179)",
        "rgb(8, 145, 178)",
        "rgb(102, 102, 102)",
    ];
}

function setupTrend(ctx) {
    const { dataRows } = ctx;
    const trendArea = ctx.trendArea;
    const trendEmpty = ctx.trendEmpty;
    const trendUndatedEmpty = ctx.trendUndatedEmpty;
    const trendMetricSelect = ctx.trendMetricSelect;

    if (trendMetricSelect) {
        ctx.numericColumns.forEach((column, i) => {
            const option = document.createElement("option");
            option.value = String(i);
            option.textContent = column.label;
            trendMetricSelect.appendChild(option);
        });
    }

    function datedRowsSorted() {
        return dataRows
            .map((row) => ({ row, time: new Date(row.dataset.date || "").getTime() }))
            .filter((entry) => entry.row.dataset.date && !Number.isNaN(entry.time))
            .sort((a, b) => a.time - b.time)
            .map((entry) => entry.row);
    }

    function trendMetric() {
        if (ctx.numericColumns.length === 0) {
            return null;
        }
        const idx = trendMetricSelect ? parseInt(trendMetricSelect.value, 10) || 0 : 0;
        return ctx.numericColumns[Math.min(idx, ctx.numericColumns.length - 1)];
    }

    function applyTrendSelection() {
        ctx.selectedIds.clear();
        if (!ctx.trendXBrush && !ctx.trendYBrush) {
            return;
        }
        const metric = trendMetric();
        const activeByField = ctx.collectCheckboxFilters();
        const term = ctx.currentSearchTerm();
        const ordered = datedRowsSorted();
        ordered.forEach((row, i) => {
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
            if (ctx.trendXBrush && (i < ctx.trendXBrush.low || i > ctx.trendXBrush.high)) {
                return;
            }
            if (ctx.trendYBrush && metric) {
                const value = parseFloat(ctx.cellAt(row, metric.index).dataset.sort);
                if (
                    Number.isNaN(value) ||
                    value < ctx.trendYBrush.low ||
                    value > ctx.trendYBrush.high
                ) {
                    return;
                }
            }
            ctx.selectedIds.add(checkbox.dataset.id);
        });
    }

    function renderTrend() {
        if (!trendArea) {
            return;
        }
        const metric = trendMetric();
        if (!metric) {
            if (trendEmpty) {
                trendEmpty.hidden = false;
            }
            if (trendUndatedEmpty) {
                trendUndatedEmpty.hidden = true;
            }
            trendArea.replaceChildren();
            return;
        }

        const ordered = datedRowsSorted();
        if (ordered.length === 0) {
            ctx.trendXBrush = null;
            ctx.trendYBrush = null;
            if (trendEmpty) {
                trendEmpty.hidden = true;
            }
            if (trendUndatedEmpty) {
                trendUndatedEmpty.hidden = false;
            }
            trendArea.replaceChildren();
            return;
        }
        if (trendEmpty) {
            trendEmpty.hidden = true;
        }
        if (trendUndatedEmpty) {
            trendUndatedEmpty.hidden = true;
        }

        const indexOf = new Map();
        ordered.forEach((row, i) => indexOf.set(row, i));

        const width = trendArea.clientWidth || 600;
        const height = 360;
        const margin = { top: 24, right: 24, bottom: 56, left: 60 };
        const innerW = Math.max(1, width - margin.left - margin.right);
        const innerH = Math.max(1, height - margin.top - margin.bottom);

        let min = Infinity;
        let max = -Infinity;
        ordered.forEach((row) => {
            const cell = ctx.cellAt(row, metric.index);
            const value = parseFloat(cell.dataset.sort);
            if (Number.isNaN(value)) {
                return;
            }
            const stdRaw = cell.dataset.std;
            const std = stdRaw ? parseFloat(stdRaw) : NaN;
            const lo = Number.isFinite(std) ? value - std : value;
            const hi = Number.isFinite(std) ? value + std : value;
            if (lo < min) {
                min = lo;
            }
            if (hi > max) {
                max = hi;
            }
        });
        if (min === Infinity) {
            min = 0;
            max = 1;
        } else if (min === max) {
            min -= 0.5;
            max += 0.5;
        }

        const n = ordered.length;
        const xPad = n <= 1 ? 0 : Math.min(24, innerW * 0.06);
        const yPad = Math.min(16, innerH * 0.06);
        const xSpan = Math.max(1, innerW - 2 * xPad);
        const ySpan = Math.max(1, innerH - 2 * yPad);
        const xAt = (i) => (n <= 1 ? innerW / 2 : xPad + (xSpan * i) / (n - 1));
        const indexFromX = (x) => {
            if (n <= 1) {
                return 0;
            }
            const clamped = Math.min(xPad + xSpan, Math.max(xPad, x));
            return Math.round(((clamped - xPad) / xSpan) * (n - 1));
        };
        const valueY = (value) => innerH - yPad - ((value - min) / (max - min)) * ySpan;
        const valueFromY = (y) => min + ((innerH - yPad - y) / ySpan) * (max - min);

        const activeByField = ctx.collectCheckboxFilters();
        const term = ctx.currentSearchTerm();
        function rowVisible(row) {
            return (
                ctx.passesCheckboxFilters(row, activeByField) &&
                ctx.passesDateRange(row) &&
                ctx.rowMatchesSearch(row, term)
            );
        }

        const hasSelection = ctx.selectedIds.size > 0;

        const groups = [];
        const byKey = new Map();
        ordered.forEach((row) => {
            if (!rowVisible(row)) {
                return;
            }
            const cell = ctx.cellAt(row, metric.index);
            const value = parseFloat(cell.dataset.sort);
            if (Number.isNaN(value)) {
                return;
            }
            const stdRaw = cell.dataset.std;
            const std = stdRaw ? parseFloat(stdRaw) : NaN;
            const info = ctx.groupState
                ? ctx.groupInfo(row)
                : { key: "__all__", label: "" };
            let group = byKey.get(info.key);
            if (!group) {
                group = { label: info.label, points: [] };
                byKey.set(info.key, group);
                groups.push(group);
            }
            const checkbox = row.querySelector(".skore-summary-row");
            group.points.push({
                x: xAt(indexOf.get(row)),
                y: valueY(value),
                id: checkbox ? checkbox.dataset.id : "",
                key: row.dataset.key || "",
                value,
                std,
            });
        });

        const svg = document.createElementNS(SVG_NS, "svg");
        svg.setAttribute("viewBox", "0 0 " + width + " " + height);
        svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

        const plotG = document.createElementNS(SVG_NS, "g");
        plotG.setAttribute(
            "transform",
            "translate(" + margin.left + ", " + margin.top + ")"
        );
        svg.appendChild(plotG);

        function localPoint(evt) {
            const point = svg.createSVGPoint();
            point.x = evt.clientX;
            point.y = evt.clientY;
            const ctm = svg.getScreenCTM();
            if (!ctm) {
                return { x: 0, y: 0 };
            }
            const local = point.matrixTransform(ctm.inverse());
            return { x: local.x - margin.left, y: local.y - margin.top };
        }

        const tooltip = document.createElement("div");
        tooltip.className = "summary-trend-tooltip";
        tooltip.hidden = true;

        function setTooltipContent(id, key, value, std) {
            const idLine = document.createElement("div");
            idLine.textContent = "ID: " + middleEllipsis(id);
            const keyLine = document.createElement("div");
            keyLine.textContent = "Key: " + key;
            const valueLine = document.createElement("div");
            const text = formatTick(value);
            valueLine.textContent =
                metric.label +
                ": " +
                (Number.isFinite(std) && std > 0
                    ? text + " \u00b1 " + formatTick(std)
                    : text);
            tooltip.replaceChildren(idLine, keyLine, valueLine);
        }

        function positionTooltip(evt) {
            const rect = trendArea.getBoundingClientRect();
            tooltip.style.left = evt.clientX - rect.left + 12 + "px";
            tooltip.style.top = evt.clientY - rect.top + 12 + "px";
        }

        const yAxis = document.createElementNS(SVG_NS, "line");
        yAxis.setAttribute("class", "summary-plot-axis-line");
        yAxis.setAttribute("x1", 0);
        yAxis.setAttribute("y1", 0);
        yAxis.setAttribute("x2", 0);
        yAxis.setAttribute("y2", innerH);
        plotG.appendChild(yAxis);

        const xAxis = document.createElementNS(SVG_NS, "line");
        xAxis.setAttribute("class", "summary-plot-axis-line");
        xAxis.setAttribute("x1", 0);
        xAxis.setAttribute("y1", innerH);
        xAxis.setAttribute("x2", innerW);
        xAxis.setAttribute("y2", innerH);
        plotG.appendChild(xAxis);

        appendText(plotG, "summary-plot-axis-label", 0, -10, "start", metric.label);
        appendText(plotG, "summary-plot-tick", -8, valueY(max) + 4, "end", formatTick(max));
        appendText(plotG, "summary-plot-tick", -8, valueY(min), "end", formatTick(min));

        if (n > 0) {
            const tickCount = Math.min(n, 5);
            for (let t = 0; t < tickCount; t += 1) {
                const i = tickCount === 1 ? 0 : Math.round((t * (n - 1)) / (tickCount - 1));
                const row = ordered[i];
                const date = new Date(row.dataset.date);
                const label = Number.isNaN(date.getTime())
                    ? row.dataset.date
                    : formatBucket(date, "minute");
                const x = xAt(i);
                const node = document.createElementNS(SVG_NS, "text");
                node.setAttribute("class", "summary-plot-tick");
                node.setAttribute("x", x);
                node.setAttribute("y", innerH + 8);
                node.setAttribute("text-anchor", "end");
                node.setAttribute("transform", "rotate(-30 " + x + " " + (innerH + 8) + ")");
                node.textContent = label;
                plotG.appendChild(node);
            }
        }

        const palette = trendPalette(ctx.container);

        groups.forEach((group, gi) => {
            const color = ctx.groupState
                ? palette[gi % palette.length]
                : palette[0];
            if (group.points.length > 1) {
                let d = "";
                group.points.forEach((pt, i) => {
                    d += (i === 0 ? "M" : " L") + pt.x + " " + pt.y;
                });
                const lineSelected = group.points.some((pt) => ctx.selectedIds.has(pt.id));
                const path = document.createElementNS(SVG_NS, "path");
                path.setAttribute(
                    "class",
                    hasSelection && !lineSelected
                        ? "summary-trend-line summary-trend-line--dim"
                        : "summary-trend-line"
                );
                path.setAttribute("d", d);
                path.setAttribute("stroke", color);
                plotG.appendChild(path);
            }
            group.points.forEach((pt) => {
                const selected = ctx.selectedIds.has(pt.id);
                const dim = hasSelection && !selected;
                if (Number.isFinite(pt.std) && pt.std > 0) {
                    const yTop = valueY(pt.value + pt.std);
                    const yBot = valueY(pt.value - pt.std);
                    const capW = 4;
                    appendLine(plotG, pt.x, yTop, pt.x, yBot, color, "summary-trend-errorbar");
                    appendLine(
                        plotG,
                        pt.x - capW,
                        yTop,
                        pt.x + capW,
                        yTop,
                        color,
                        "summary-trend-errorbar"
                    );
                    appendLine(
                        plotG,
                        pt.x - capW,
                        yBot,
                        pt.x + capW,
                        yBot,
                        color,
                        "summary-trend-errorbar"
                    );
                }
                const circle = document.createElementNS(SVG_NS, "circle");
                circle.setAttribute(
                    "class",
                    dim ? "summary-trend-point summary-trend-point--dim" : "summary-trend-point"
                );
                circle.setAttribute("cx", pt.x);
                circle.setAttribute("cy", pt.y);
                circle.setAttribute("r", 4);
                circle.setAttribute("fill", color);
                circle.addEventListener("mouseenter", (evt) => {
                    circle.setAttribute("r", 6);
                    circle.classList.add("summary-trend-point--hover");
                    setTooltipContent(pt.id, pt.key, pt.value, pt.std);
                    tooltip.hidden = false;
                    positionTooltip(evt);
                });
                circle.addEventListener("mousemove", positionTooltip);
                circle.addEventListener("mouseleave", () => {
                    circle.setAttribute("r", 4);
                    circle.classList.remove("summary-trend-point--hover");
                    tooltip.hidden = true;
                });
                plotG.appendChild(circle);
            });
        });

        const eps = (max - min) * 1e-9 || 1e-9;
        let xLow = ctx.trendXBrush ? ctx.trendXBrush.low : 0;
        let xHigh = ctx.trendXBrush ? ctx.trendXBrush.high : n - 1;
        let yHighVal = ctx.trendYBrush ? ctx.trendYBrush.high : max;
        let yLowVal = ctx.trendYBrush ? ctx.trendYBrush.low : min;

        function commitX() {
            ctx.trendXBrush =
                xLow <= 0 && xHigh >= n - 1 ? null : { low: xLow, high: xHigh };
        }

        function commitY() {
            ctx.trendYBrush =
                yLowVal <= min + eps && yHighVal >= max - eps
                    ? null
                    : { low: yLowVal, high: yHighVal };
        }

        function makeCursor(orientation, read, write, commit) {
            const line = document.createElementNS(SVG_NS, "line");
            line.setAttribute("class", "summary-trend-cursor");

            const handle = document.createElementNS(SVG_NS, "g");
            handle.setAttribute(
                "class",
                "summary-trend-grip summary-trend-grip--" + orientation
            );
            const grip = document.createElementNS(SVG_NS, "rect");
            grip.setAttribute("class", "summary-trend-grip-box");
            grip.setAttribute("rx", 3);
            const arrow = document.createElementNS(SVG_NS, "text");
            arrow.setAttribute("class", "summary-trend-grip-arrow");
            arrow.setAttribute("text-anchor", "middle");
            arrow.setAttribute("dominant-baseline", "central");
            arrow.textContent = orientation === "x" ? "\u2194" : "\u2195";
            handle.appendChild(grip);
            handle.appendChild(arrow);

            function place() {
                if (orientation === "x") {
                    const x = xAt(read());
                    line.setAttribute("x1", x);
                    line.setAttribute("y1", 0);
                    line.setAttribute("x2", x);
                    line.setAttribute("y2", innerH);
                    grip.setAttribute("x", x - 11);
                    grip.setAttribute("y", -7);
                    grip.setAttribute("width", 22);
                    grip.setAttribute("height", 14);
                    arrow.setAttribute("x", x);
                    arrow.setAttribute("y", 0);
                } else {
                    const y = valueY(read());
                    line.setAttribute("x1", 0);
                    line.setAttribute("y1", y);
                    line.setAttribute("x2", innerW);
                    line.setAttribute("y2", y);
                    grip.setAttribute("x", -8);
                    grip.setAttribute("y", y - 11);
                    grip.setAttribute("width", 16);
                    grip.setAttribute("height", 22);
                    arrow.setAttribute("x", 0);
                    arrow.setAttribute("y", y);
                }
            }
            place();

            let dragging = false;
            handle.addEventListener("pointerdown", (evt) => {
                evt.preventDefault();
                handle.setPointerCapture(evt.pointerId);
                dragging = true;
            });
            handle.addEventListener("pointermove", (evt) => {
                if (!dragging) {
                    return;
                }
                const local = localPoint(evt);
                if (orientation === "x") {
                    write(indexFromX(local.x));
                } else {
                    const clampedY = Math.min(innerH, Math.max(0, local.y));
                    write(valueFromY(clampedY));
                }
                place();
            });
            function end() {
                if (!dragging) {
                    return;
                }
                dragging = false;
                commit();
                applyTrendSelection();
                ctx.syncCheckboxes();
                renderTrend();
                ctx.updateQuery();
            }
            handle.addEventListener("pointerup", end);
            handle.addEventListener("pointercancel", end);

            plotG.appendChild(line);
            plotG.appendChild(handle);
        }

        makeCursor(
            "x",
            () => xLow,
            (idx) => {
                xLow = Math.max(0, Math.min(idx, xHigh));
            },
            commitX
        );
        makeCursor(
            "x",
            () => xHigh,
            (idx) => {
                xHigh = Math.min(n - 1, Math.max(idx, xLow));
            },
            commitX
        );
        makeCursor(
            "y",
            () => yHighVal,
            (value) => {
                yHighVal = Math.min(max, Math.max(value, yLowVal));
            },
            commitY
        );
        makeCursor(
            "y",
            () => yLowVal,
            (value) => {
                yLowVal = Math.max(min, Math.min(value, yHighVal));
            },
            commitY
        );

        trendArea.replaceChildren(svg, tooltip);
    }

    ctx.applyTrendSelection = applyTrendSelection;
    ctx.renderTrend = renderTrend;
}
